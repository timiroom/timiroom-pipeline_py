import asyncio
import json
import logging

from aiokafka import AIOKafkaConsumer, AIOKafkaProducer
from aiokafka.structs import OffsetAndMetadata, TopicPartition

from phase1.document_ingestion import DocumentIngestionService
from phase4.event import PipelineResultEvent

logger = logging.getLogger(__name__)

RECONNECT_DELAY_INITIAL = 3
RECONNECT_DELAY_MAX = 30
MAX_PROCESS_RETRY = 3
RETRY_BACKOFF = 2


class KafkaConsumerService:
    def __init__(
        self,
        bootstrap_servers: str,
        topic: str,
        group_id: str,
        ingestion_service: DocumentIngestionService,
        dead_letter_topic: str | None = None,
        max_poll_interval_ms: int = 900_000,
    ):
        self._bootstrap = bootstrap_servers
        self._topic = topic
        self._group_id = group_id
        self._ingestion = ingestion_service
        self._dlq_topic = dead_letter_topic
        self._max_poll_interval_ms = max_poll_interval_ms
        self._consumer: AIOKafkaConsumer | None = None
        self._dlq_producer: AIOKafkaProducer | None = None
        self._main_task: asyncio.Task | None = None

    def start(self) -> None:
        if self._main_task is None or self._main_task.done():
            self._main_task = asyncio.create_task(self._run_loop())

    async def stop(self) -> None:
        if self._main_task:
            self._main_task.cancel()
            try:
                await self._main_task
            except asyncio.CancelledError:
                pass
            self._main_task = None
        await self._close_consumer()
        if self._dlq_producer is not None:
            try:
                await self._dlq_producer.stop()
            except Exception as exc:
                logger.warning("DLQ Producer 종료 실패: %s", exc)
            self._dlq_producer = None

    async def _run_loop(self) -> None:
        attempt = 0
        delay = RECONNECT_DELAY_INITIAL
        while True:
            attempt += 1
            await self._close_consumer()
            try:
                self._consumer = self._create_consumer()
                await self._consumer.start()
                logger.info("Kafka Consumer 연결 성공 (시도 %d회) — topic: %s", attempt, self._topic)
                delay = RECONNECT_DELAY_INITIAL
                await self._consume_loop()
                logger.warning("Kafka Consumer 루프 종료 — 재연결 시도")
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning("Kafka Consumer 오류 (시도 %d회): %s — %ds 후 재시도", attempt, exc, delay)
            await asyncio.sleep(delay)
            delay = min(delay * 2, RECONNECT_DELAY_MAX)

    def _create_consumer(self) -> AIOKafkaConsumer:
        return AIOKafkaConsumer(
            self._topic,
            bootstrap_servers=self._bootstrap,
            group_id=self._group_id,
            value_deserializer=lambda value: json.loads(value.decode("utf-8")),
            auto_offset_reset="earliest",
            enable_auto_commit=False,
            max_poll_interval_ms=self._max_poll_interval_ms,
        )

    async def _consume_loop(self) -> None:
        if self._consumer is None:
            raise RuntimeError("Kafka Consumer가 연결되지 않았습니다")
        async for message in self._consumer:
            await self._process_with_retry(message.value)
            partition = TopicPartition(message.topic, message.partition)
            await self._consumer.commit({partition: OffsetAndMetadata(message.offset + 1, "")})

    async def _process_with_retry(self, event: dict) -> None:
        last_error: Exception | None = None
        for attempt in range(1, MAX_PROCESS_RETRY + 1):
            try:
                await self._process(event)
                return
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                last_error = exc
                logger.warning("Kafka 메시지 처리 실패 (%d/%d): %s", attempt, MAX_PROCESS_RETRY, exc)
                if attempt < MAX_PROCESS_RETRY:
                    await asyncio.sleep(RETRY_BACKOFF)

        logger.error("Kafka 메시지 처리 최종 실패 — DLQ로 전송")
        await self._publish_to_dlq(event, str(last_error or "알 수 없는 처리 오류"))

    async def _publish_to_dlq(self, event: dict, error: str) -> None:
        if not self._dlq_topic:
            raise RuntimeError("DLQ topic이 설정되지 않아 실패 메시지를 커밋할 수 없습니다")
        producer = await self._get_dlq_producer()
        key = str(event.get("pipelineId", "")) if isinstance(event, dict) else ""
        envelope = {"originalEvent": event, "error": error, "failedAtStage": "ingestion"}
        await producer.send_and_wait(
            self._dlq_topic,
            key=key.encode("utf-8") if key else None,
            value=json.dumps(envelope, ensure_ascii=False).encode("utf-8"),
        )
        logger.info("DLQ 전송 완료 — pipelineId: %s", key or "unknown")

    async def _get_dlq_producer(self) -> AIOKafkaProducer:
        if self._dlq_producer is None:
            producer = AIOKafkaProducer(
                bootstrap_servers=self._bootstrap,
                acks="all",
                enable_idempotence=True,
            )
            await producer.start()
            self._dlq_producer = producer
        return self._dlq_producer

    async def _process(self, raw_event: dict) -> None:
        event = PipelineResultEvent.model_validate(raw_event)
        logger.info("Kafka 메시지 수신 — pipelineId: %s", event.pipeline_id)
        base_meta = {
            "pipeline_id": event.pipeline_id,
            "project_name": event.project_name,
            "platform": event.platform,
            "tech_stack": event.tech_stack,
            "query": event.user_query,
        }
        documents = [
            (event.prd_document, {**base_meta, "type": "prd"}),
            (event.db_schema, {**base_meta, "type": "erd"}),
            (event.api_spec, {**base_meta, "type": "api"}),
            ("\n".join(event.feature_list), {**base_meta, "type": "features"}),
        ]
        if event.feature_spec_document.strip() and event.feature_spec_document.strip() != "{}":
            documents.append((event.feature_spec_document, {**base_meta, "type": "feature_spec"}))
        if event.feature_registry:
            documents.append((json.dumps(event.feature_registry, ensure_ascii=False), {**base_meta, "type": "feature_registry"}))
        if event.market_research.strip():
            documents.append((event.market_research, {**base_meta, "type": "market_research"}))

        saved = await self._ingestion.ingest_event(documents)
        if saved <= 0:
            raise RuntimeError(f"pipelineId {event.pipeline_id}: 저장된 청크가 없습니다")
        logger.info("처리 완료 — pipelineId: %s, 총 %d chunks 저장", event.pipeline_id, saved)

    async def _close_consumer(self) -> None:
        consumer, self._consumer = self._consumer, None
        if consumer is not None:
            try:
                await consumer.stop()
            except Exception as exc:
                logger.warning("Kafka Consumer 종료 실패: %s", exc)

    @property
    def ready(self) -> bool:
        return self._consumer is not None
