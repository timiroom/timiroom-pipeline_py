import asyncio
import json
import logging

from aiokafka import AIOKafkaConsumer

from phase1.document_ingestion import DocumentIngestionService

logger = logging.getLogger(__name__)

RECONNECT_DELAY = 5


class KafkaConsumerService:
    """
    비동기 Kafka Consumer.
    - start()는 즉시 반환하고 백그라운드에서 무한 재시도 연결
    - 연결 성공 후 메시지 소비, 연결 끊기면 자동 재연결
    """

    def __init__(
        self,
        bootstrap_servers: str,
        topic: str,
        group_id: str,
        ingestion_service: DocumentIngestionService,
    ):
        self._bootstrap = bootstrap_servers
        self._topic = topic
        self._group_id = group_id
        self._ingestion = ingestion_service
        self._consumer: AIOKafkaConsumer | None = None
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

    async def _run_loop(self) -> None:
        attempt = 0
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
                await self._consume_loop()
                # consume_loop 정상 종료 시 (CancelledError 아닌 경우) 재연결
                logger.warning("Kafka Consumer 루프 종료 — 재연결 시도")
            except asyncio.CancelledError:
                return
            except Exception as e:
                logger.warning("Kafka Consumer 연결 실패 (시도 %d회): %s — %ds 후 재시도",
                               attempt, e, RECONNECT_DELAY)
            await asyncio.sleep(RECONNECT_DELAY)

    async def _consume_loop(self) -> None:
        async for msg in self._consumer:
            try:
                await self._process(msg.value)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.error("Kafka 메시지 처리 실패: %s", e, exc_info=True)

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

        total = 0
        for field_key, doc_type in [
            ("prdDocument", "prd"),
            ("marketResearch", "market_research"),
            ("dbSchema", "erd"),
            ("apiSpec", "api"),
        ]:
            text = event.get(field_key, "")
            if text and text.strip():
                meta = {**base_meta, "type": doc_type}
                try:
                    saved = await self._ingestion.ingest(text, meta)
                    logger.info("  %s 저장 완료 — %d chunks", doc_type, saved)
                    total += saved
                except Exception as e:
                    logger.warning("  %s 저장 실패: %s", doc_type, e)

        logger.info("처리 완료 — pipelineId: %s, 총 %d chunks 저장", pipeline_id, total)

    async def _close_consumer(self) -> None:
        if self._consumer is not None:
            try:
                await self._consumer.stop()
            except Exception:
                pass
            self._consumer = None
