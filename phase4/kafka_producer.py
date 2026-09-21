import asyncio
import json
import logging
from datetime import UTC, datetime

import psycopg2
from aiokafka import AIOKafkaProducer

from phase2.state import PipelineState
from phase4.event import PipelineResultEvent

logger = logging.getLogger(__name__)

RECONNECT_DELAY_INITIAL = 3
RECONNECT_DELAY_MAX = 30


class KafkaOutbox:
    """Kafka 전송 전 PostgreSQL에 결과 이벤트를 내구성 있게 보관한다."""

    def __init__(self, db_url: str):
        self._db_url = db_url

    async def put(self, pipeline_id: str, topic: str, event: dict) -> None:
        await asyncio.to_thread(self._put_sync, pipeline_id, topic, event)

    def _put_sync(self, pipeline_id: str, topic: str, event: dict) -> None:
        with psycopg2.connect(self._db_url) as conn, conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO pipeline_outbox (pipeline_id, topic, payload)
                VALUES (%s, %s, %s::jsonb)
                ON CONFLICT (pipeline_id) DO UPDATE
                SET topic = EXCLUDED.topic, payload = EXCLUDED.payload, last_error = NULL
                """,
                (pipeline_id, topic, json.dumps(event, ensure_ascii=False)),
            )

    async def pending(self, limit: int = 100) -> list[tuple[str, str, dict]]:
        return await asyncio.to_thread(self._pending_sync, limit)

    def _pending_sync(self, limit: int) -> list[tuple[str, str, dict]]:
        with psycopg2.connect(self._db_url) as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT pipeline_id, topic, payload
                FROM pipeline_outbox
                ORDER BY created_at
                LIMIT %s
                """,
                (limit,),
            )
            return [(row[0], row[1], row[2]) for row in cur.fetchall()]

    async def delete(self, pipeline_id: str) -> None:
        await asyncio.to_thread(self._delete_sync, pipeline_id)

    def _delete_sync(self, pipeline_id: str) -> None:
        with psycopg2.connect(self._db_url) as conn, conn.cursor() as cur:
            cur.execute("DELETE FROM pipeline_outbox WHERE pipeline_id = %s", (pipeline_id,))

    async def mark_error(self, pipeline_id: str, error: str) -> None:
        await asyncio.to_thread(self._mark_error_sync, pipeline_id, error)

    def _mark_error_sync(self, pipeline_id: str, error: str) -> None:
        with psycopg2.connect(self._db_url) as conn, conn.cursor() as cur:
            cur.execute(
                """
                UPDATE pipeline_outbox
                SET attempts = attempts + 1, last_error = %s
                WHERE pipeline_id = %s
                """,
                (error[:2000], pipeline_id),
            )


class KafkaProducerService:
    def __init__(
        self,
        bootstrap_servers: str,
        topic: str,
        db_url: str,
        max_publish_retry: int = 3,
        outbox: KafkaOutbox | None = None,
    ):
        if max_publish_retry <= 0:
            raise ValueError("max_publish_retry는 1 이상이어야 합니다")
        self._topic = topic
        self._bootstrap = bootstrap_servers
        self._max_publish_retry = max_publish_retry
        self._outbox = outbox or KafkaOutbox(db_url)
        self._producer: AIOKafkaProducer | None = None
        self._ready = False
        self._main_task: asyncio.Task | None = None
        self._connect_lock = asyncio.Lock()
        self._publish_lock = asyncio.Lock()

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
        await self._close_producer()

    async def publish(self, state: PipelineState, pipeline_id: str) -> str:
        event = PipelineResultEvent(
            pipelineId=pipeline_id,
            projectName=state.project_name,
            platform=(
                getattr(state.platform, "value", state.platform)
                if state.platform
                else ""
            ),
            techStack=state.tech_stack,
            userQuery=state.user_query,
            featureList=state.feature_list,
            prdDocument=state.prd_document,
            marketResearch=state.market_research,
            dbSchema=state.db_schema,
            apiSpec=state.api_spec,
            featureSpecDocument=state.feature_spec_document or "{}",
            featureRegistry=state.feature_registry or [],
            retryCount=state.retry_count,
            createdAt=datetime.now(UTC).isoformat(),
        ).model_dump(by_alias=True)

        async with self._publish_lock:
            await self._outbox.put(pipeline_id, self._topic, event)
            if not await self._send_with_retry(pipeline_id, self._topic, event):
                raise RuntimeError(f"Kafka 발행 실패 — outbox에 보관됨: {pipeline_id}")
        return pipeline_id

    async def _send_with_retry(self, pipeline_id: str, topic: str, event: dict) -> bool:
        last_error = ""
        for attempt in range(1, self._max_publish_retry + 1):
            try:
                if not self._ready or self._producer is None:
                    await self._connect_now()
                if self._producer is None:
                    raise RuntimeError("Kafka Producer 연결 없음")
                await self._producer.send_and_wait(topic, key=pipeline_id, value=event)
                await self._outbox.delete(pipeline_id)
                logger.info("Kafka 발행 완료 — pipelineId: %s", pipeline_id)
                return True
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                last_error = str(exc)
                self._ready = False
                await self._close_producer()
                logger.warning(
                    "Kafka 발행 실패 (%d/%d) — %s", attempt, self._max_publish_retry, exc
                )
                if attempt < self._max_publish_retry:
                    await asyncio.sleep(min(2 ** (attempt - 1), 5))
        await self._outbox.mark_error(pipeline_id, last_error)
        return False

    async def _run_loop(self) -> None:
        delay = RECONNECT_DELAY_INITIAL
        while True:
            try:
                if not self._ready or self._producer is None:
                    await self._connect_now()
                await self._drain_outbox()
                delay = RECONNECT_DELAY_INITIAL
                await asyncio.sleep(5)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning("Kafka/outbox 백그라운드 처리 실패: %s — %ds 후 재시도", exc, delay)
                await asyncio.sleep(delay)
                delay = min(delay * 2, RECONNECT_DELAY_MAX)

    async def _drain_outbox(self) -> None:
        async with self._publish_lock:
            for pipeline_id, topic, event in await self._outbox.pending():
                if not await self._send_with_retry(pipeline_id, topic, event):
                    break

    async def _connect_now(self) -> None:
        async with self._connect_lock:
            if self._ready and self._producer is not None:
                return
            await self._close_producer()
            producer = AIOKafkaProducer(
                bootstrap_servers=self._bootstrap,
                value_serializer=lambda value: json.dumps(value, ensure_ascii=False).encode("utf-8"),
                key_serializer=lambda key: key.encode("utf-8") if key else None,
                acks="all",
                enable_idempotence=True,
            )
            try:
                await producer.start()
            except Exception:
                # start() can allocate sockets before failing; always release them
                # so reconnect loops do not leak producers.
                await producer.stop()
                raise
            self._producer = producer
            self._ready = True
            logger.info("Kafka Producer 연결 성공")

    async def _close_producer(self) -> None:
        producer, self._producer = self._producer, None
        self._ready = False
        if producer is not None:
            try:
                await producer.stop()
            except Exception as exc:
                logger.warning("Kafka Producer 종료 실패: %s", exc)

    @property
    def ready(self) -> bool:
        return self._ready
