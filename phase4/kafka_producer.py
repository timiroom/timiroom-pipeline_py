import asyncio
import json
import logging
import uuid
from datetime import datetime

from aiokafka import AIOKafkaProducer

from phase2.state import PipelineState

logger = logging.getLogger(__name__)

RECONNECT_DELAY_INITIAL = 3
RECONNECT_DELAY_MAX = 30


class KafkaProducerService:
    """
    비동기 Kafka Producer.
    - start()는 즉시 반환하고 백그라운드에서 연결 유지
    - 연결 끊기면 자동 재연결 (무한 재시도)
    - Kafka 미준비 시 publish()는 경고 후 스킵 (앱 기동 차단 없음)
    """

    def __init__(self, bootstrap_servers: str, topic: str):
        self._topic = topic
        self._bootstrap = bootstrap_servers
        self._producer: AIOKafkaProducer | None = None
        self._ready = False
        self._connect_task: asyncio.Task | None = None
        self._connect_lock = asyncio.Lock()

    def start(self) -> None:
        self._connect_task = asyncio.create_task(self._connect_loop())

    async def stop(self) -> None:
        if self._connect_task:
            self._connect_task.cancel()
            try:
                await self._connect_task
            except asyncio.CancelledError:
                pass
        await self._close_producer()

    async def publish(self, state: PipelineState) -> str | None:
        if not self._ready or self._producer is None:
            # 백그라운드 재연결을 기다리지 않고 발행 시점에 직접 연결을 시도한다
            # (rag-pipeline의 KafkaTemplate처럼 항상 발행을 시도 — 짧은 단절로 인한 무의미한 드롭 방지)
            logger.warning("Kafka Producer 미준비 — 즉시 연결 시도")
            if not await self._connect_now():
                self._ensure_connecting()
                return None

        pipeline_id = str(uuid.uuid4())
        event = {
            "pipelineId": pipeline_id,
            "projectName": state.project_name,
            "platform": state.platform.value if state.platform else "",
            "techStack": state.tech_stack,
            "userQuery": state.user_query,
            "featureList": state.feature_list,
            "prdDocument": state.prd_document,
            "marketResearch": state.market_research,
            "dbSchema": state.db_schema,
            "apiSpec": state.api_spec,
            "retryCount": state.retry_count,
            "createdAt": datetime.utcnow().isoformat(),
        }
        try:
            await self._producer.send_and_wait(self._topic, key=pipeline_id, value=event)
            logger.info("Kafka 발행 완료 — pipelineId: %s", pipeline_id)
            return pipeline_id
        except Exception as e:
            logger.error("Kafka 발행 실패: %s", e)
            self._ready = False
            self._ensure_connecting()
            return None

    def _ensure_connecting(self) -> None:
        """연결 태스크가 없거나 종료된 경우 재시작."""
        if self._connect_task is None or self._connect_task.done():
            self._connect_task = asyncio.create_task(self._connect_loop())

    async def _connect_now(self) -> bool:
        """발행 시점에 한 번만 즉시 연결 시도 (백그라운드 재연결 루프와 별개). 성공 시 True.

        동시 publish() 호출들이 각자 재연결을 시도하면 서로의 producer를 stop/start
        하며 경합할 수 있으므로 락으로 직렬화한다."""
        async with self._connect_lock:
            if self._ready and self._producer is not None:
                return True
            await self._close_producer()
            try:
                self._producer = AIOKafkaProducer(
                    bootstrap_servers=self._bootstrap,
                    value_serializer=lambda v: json.dumps(v, ensure_ascii=False).encode("utf-8"),
                    key_serializer=lambda k: k.encode("utf-8") if k else None,
                    acks="all",
                )
                await self._producer.start()
                self._ready = True
                logger.info("Kafka Producer 즉시 연결 성공")
                return True
            except Exception as e:
                logger.warning("Kafka Producer 즉시 연결 실패: %s", e)
                await self._close_producer()
                return False

    async def _connect_loop(self) -> None:
        attempt = 0
        delay = RECONNECT_DELAY_INITIAL
        while True:
            attempt += 1
            try:
                async with self._connect_lock:
                    if self._ready and self._producer is not None:
                        return
                    await self._close_producer()
                    self._producer = AIOKafkaProducer(
                        bootstrap_servers=self._bootstrap,
                        value_serializer=lambda v: json.dumps(v, ensure_ascii=False).encode("utf-8"),
                        key_serializer=lambda k: k.encode("utf-8") if k else None,
                        acks="all",
                    )
                    await self._producer.start()
                    self._ready = True
                logger.info("Kafka Producer 연결 성공 (시도 %d회)", attempt)
                # 연결 유지 — 태스크가 살아있는 동안 대기
                # publish()에서 오류 발생 시 _ready=False + _ensure_connecting()으로 재진입
                return
            except asyncio.CancelledError:
                return
            except Exception as e:
                logger.warning("Kafka Producer 연결 실패 (시도 %d회): %s — %ds 후 재시도",
                               attempt, e, delay)
                await asyncio.sleep(delay)
                delay = min(delay * 2, RECONNECT_DELAY_MAX)

    async def _close_producer(self) -> None:
        if self._producer is not None:
            try:
                await self._producer.stop()
            except Exception:
                pass
            self._producer = None
        self._ready = False
