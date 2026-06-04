import asyncio
import json
import logging
import uuid
from datetime import datetime

from aiokafka import AIOKafkaProducer

from phase2.state import PipelineState

logger = logging.getLogger(__name__)

RECONNECT_DELAY = 5


class KafkaProducerService:
    """
    비동기 Kafka Producer.
    - start()는 즉시 반환하고 백그라운드에서 무한 재시도 연결
    - Kafka 미준비 시 publish()는 경고 후 스킵 (앱 기동 차단 없음)
    """

    def __init__(self, bootstrap_servers: str, topic: str):
        self._topic = topic
        self._bootstrap = bootstrap_servers
        self._producer: AIOKafkaProducer | None = None
        self._ready = False
        self._connect_task: asyncio.Task | None = None

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
            logger.warning("Kafka Producer 미준비 — 결과 저장 건너뜀")
            return None

        pipeline_id = str(uuid.uuid4())
        event = {
            "pipelineId": pipeline_id,
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
            return None

    async def _connect_loop(self) -> None:
        attempt = 0
        while True:
            attempt += 1
            await self._close_producer()
            try:
                self._producer = AIOKafkaProducer(
                    bootstrap_servers=self._bootstrap,
                    value_serializer=lambda v: json.dumps(v, ensure_ascii=False).encode("utf-8"),
                    key_serializer=lambda k: k.encode("utf-8") if k else None,
                )
                await self._producer.start()
                self._ready = True
                logger.info("Kafka Producer 연결 성공 (시도 %d회)", attempt)
                return
            except asyncio.CancelledError:
                return
            except Exception as e:
                logger.warning("Kafka Producer 연결 실패 (시도 %d회): %s — %ds 후 재시도",
                               attempt, e, RECONNECT_DELAY)
                await asyncio.sleep(RECONNECT_DELAY)

    async def _close_producer(self) -> None:
        if self._producer is not None:
            try:
                await self._producer.stop()
            except Exception:
                pass
            self._producer = None
        self._ready = False
