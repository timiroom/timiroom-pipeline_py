import asyncio
import json
import logging
import time

logger = logging.getLogger(__name__)


class PipelineProgressService:
    """
    SSE 이벤트를 asyncio.Queue로 관리.
    subscribe() 호출 전에 send()가 먼저 호출되면 버퍼에 저장 후 재전송.
    """

    def __init__(self):
        self._queues: dict[str, asyncio.Queue] = {}
        self._buffers: dict[str, list[dict]] = {}

    def _get_or_create_queue(self, pipeline_id: str) -> asyncio.Queue:
        if pipeline_id not in self._queues:
            self._queues[pipeline_id] = asyncio.Queue()
        return self._queues[pipeline_id]

    def send(self, pipeline_id: str | None, step: str, message: str, percent: int) -> None:
        if not pipeline_id:
            return
        data = {
            "step": step,
            "message": message,
            "percent": percent,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        self._buffer(pipeline_id, "progress", data)
        self._enqueue(pipeline_id, "progress", data)

    def complete(self, pipeline_id: str | None, result: dict) -> None:
        if not pipeline_id:
            return
        data = {"result": result}
        self._enqueue(pipeline_id, "complete", data)
        self._enqueue(pipeline_id, "__done__", {})
        self._buffers.pop(pipeline_id, None)

    def error(self, pipeline_id: str | None, message: str) -> None:
        if not pipeline_id:
            return
        data = {"message": message}
        self._buffer(pipeline_id, "error", data)
        self._enqueue(pipeline_id, "error", data)
        self._enqueue(pipeline_id, "__done__", {})
        self._buffers.pop(pipeline_id, None)

    async def subscribe(self, pipeline_id: str):
        """FastAPI StreamingResponse 용 async generator."""
        q = self._get_or_create_queue(pipeline_id)

        # 버퍼에 쌓인 이벤트 먼저 전송
        for evt in self._buffers.get(pipeline_id, []):
            yield self._format(evt["type"], evt["data"])

        while True:
            evt = await asyncio.wait_for(q.get(), timeout=1800)
            if evt["type"] == "__done__":
                self._queues.pop(pipeline_id, None)
                break
            yield self._format(evt["type"], evt["data"])

    def _format(self, event_type: str, data: dict) -> str:
        return f"event: {event_type}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"

    def _buffer(self, pipeline_id: str, event_type: str, data: dict) -> None:
        self._buffers.setdefault(pipeline_id, []).append(
            {"type": event_type, "data": data}
        )

    def _enqueue(self, pipeline_id: str, event_type: str, data: dict) -> None:
        q = self._get_or_create_queue(pipeline_id)
        try:
            q.put_nowait({"type": event_type, "data": data})
        except asyncio.QueueFull:
            logger.warning("SSE 큐 가득 참: %s", pipeline_id)
