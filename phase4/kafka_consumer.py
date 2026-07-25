import asyncio
import json
import logging

from aiokafka import AIOKafkaConsumer, AIOKafkaProducer

from phase1.document_ingestion import DocumentIngestionService

logger = logging.getLogger(__name__)

RECONNECT_DELAY_INITIAL = 3
RECONNECT_DELAY_MAX = 30
MAX_PROCESS_RETRY = 3
RETRY_BACKOFF = 2


class KafkaConsumerService:
    """
    비동기 Kafka Consumer.
    - start()는 즉시 반환하고 백그라운드에서 무한 재시도 연결
    - 연결 성공 후 메시지 소비, 연결 끊기면 자동 재연결
    - 메시지 처리 실패 시 최대 3회 재시도 후 Dead Letter Topic으로 전송
    """

    def __init__(
        self,
        bootstrap_servers: str,
        topic: str,
        group_id: str,
        ingestion_service: DocumentIngestionService,
        dead_letter_topic: str | None = None,
    ):
        self._bootstrap = bootstrap_servers
        self._topic = topic
        self._group_id = group_id
        self._ingestion = ingestion_service
        self._dlq_topic = dead_letter_topic
        self._consumer: AIOKafkaConsumer | None = None
        self._dlq_producer: AIOKafkaProducer | None = None
        self._main_task: asyncio.Task | None = None

    def start(self) -> None:
        self._main_task = asyncio.create_task(self._run_loop())

    async def stop(self) -> None:
        if self._main_task:
            self._main_task.cancel()
            try:
                await self._main_task
            except asyncio.CancelledError:
                pass
        await self._close_consumer()
        if self._dlq_producer is not None:
            try:
                await self._dlq_producer.stop()
            except Exception:
                pass
            self._dlq_producer = None

    async def _run_loop(self) -> None:
        attempt = 0
        delay = RECONNECT_DELAY_INITIAL
        while True:
            attempt += 1
            await self._close_consumer()
            try:
                self._consumer = AIOKafkaConsumer(
                    self._topic,
                    bootstrap_servers=self._bootstrap,
                    group_id=self._group_id,
                    value_deserializer=lambda v: json.loads(v.decode("utf-8")),
                    auto_offset_reset="earliest",
                    enable_auto_commit=True,
                )
                await self._consumer.start()
                logger.info("Kafka Consumer 연결 성공 (시도 %d회) — topic: %s",
                            attempt, self._topic)
                delay = RECONNECT_DELAY_INITIAL
                await self._consume_loop()
                # consume_loop 정상 종료 시 (CancelledError 아닌 경우) 재연결
                logger.warning("Kafka Consumer 루프 종료 — 재연결 시도")
            except asyncio.CancelledError:
                return
            except Exception as e:
                logger.warning("Kafka Consumer 연결 실패 (시도 %d회): %s — %ds 후 재시도",
                               attempt, e, delay)
            await asyncio.sleep(delay)
            delay = min(delay * 2, RECONNECT_DELAY_MAX)

    async def _consume_loop(self) -> None:
        async for msg in self._consumer:
            await self._process_with_retry(msg.value)

    async def _process_with_retry(self, event: dict) -> None:
        for attempt in range(1, MAX_PROCESS_RETRY + 1):
            try:
                await self._process(event)
                return
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.warning(
                    "Kafka 메시지 처리 실패 (시도 %d/%d): %s",
                    attempt, MAX_PROCESS_RETRY, e,
                )
                if attempt < MAX_PROCESS_RETRY:
                    await asyncio.sleep(RETRY_BACKOFF)

        logger.error(
            "Kafka 메시지 처리 최종 실패 — pipelineId: %s — DLQ로 전송",
            event.get("pipelineId", "unknown"),
        )
        await self._publish_to_dlq(event)

    async def _publish_to_dlq(self, event: dict) -> None:
        if not self._dlq_topic:
            return
        try:
            producer = await self._get_dlq_producer()
            key = event.get("pipelineId")
            await producer.send_and_wait(
                self._dlq_topic,
                key=key.encode("utf-8") if key else None,
                value=json.dumps(event, ensure_ascii=False).encode("utf-8"),
            )
        except Exception as e:
            logger.error("DLQ 전송 실패 — pipelineId: %s — %s", event.get("pipelineId", "unknown"), e)

    async def _get_dlq_producer(self) -> AIOKafkaProducer:
        if self._dlq_producer is None:
            self._dlq_producer = AIOKafkaProducer(bootstrap_servers=self._bootstrap, acks="all")
            await self._dlq_producer.start()
        return self._dlq_producer

    async def _process(self, event: dict) -> None:
        pipeline_id = event.get("pipelineId", "unknown")
        user_query = event.get("userQuery", "")
        logger.info("Kafka 메시지 수신 — pipelineId: %s", pipeline_id)

        base_meta = {
            "pipeline_id": pipeline_id,
            "project_name": event.get("projectName", ""),
            "platform": event.get("platform", ""),
            "tech_stack": event.get("techStack", []),
            "query": user_query,
        }

        feature_list = event.get("featureList") or []

        total = 0
        expected = 0
        failures: list[str] = []
        for field_key, doc_type in [
            ("prdDocument", "prd"),
            ("marketResearch", "market_research"),
            ("dbSchema", "erd"),
            ("apiSpec", "api"),
        ]:
            text = event.get(field_key, "")
            if text and text.strip():
                expected += 1
                meta = {**base_meta, "type": doc_type}
                try:
                    saved = await self._ingestion.ingest_fixed(text, meta)
                    logger.info("  %s 저장 완료 — %d chunks", doc_type, saved)
                    total += saved
                except Exception as e:
                    logger.warning("  %s 저장 실패: %s", doc_type, e)
                    failures.append(doc_type)

        if feature_list:
            expected += 1
            feature_text = "\n".join(feature_list)
            meta = {**base_meta, "type": "features"}
            try:
                saved = await self._ingestion.ingest_fixed(feature_text, meta)
                logger.info("  features 저장 완료 — %d chunks", saved)
                total += saved
            except Exception as e:
                logger.warning("  features 저장 실패: %s", e)
                failures.append("features")

        logger.info("처리 완료 — pipelineId: %s, 총 %d chunks 저장", pipeline_id, total)

        if failures:
            raise RuntimeError(
                f"pipelineId {pipeline_id}: {len(failures)}/{expected}개 문서 유형 저장 실패 ({', '.join(failures)})"
            )

    async def _close_consumer(self) -> None:
        if self._consumer is not None:
            try:
                await self._consumer.stop()
            except Exception:
                pass
            self._consumer = None
