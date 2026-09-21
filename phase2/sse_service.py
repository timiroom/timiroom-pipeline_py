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

    def __init__(self, terminal_ttl_seconds: int = 3600, max_terminal_events: int = 100):
        if terminal_ttl_seconds <= 0:
            raise ValueError("terminal_ttl_seconds는 0보다 커야 합니다")
        if max_terminal_events <= 0:
            raise ValueError("max_terminal_events는 0보다 커야 합니다")
        self._queues: dict[str, asyncio.Queue] = {}
        self._buffers: dict[str, list[dict]] = {}
        # 완료/실패 이벤트는 늦게 연결된 클라이언트도 원인을 확인할 수 있도록
        # 짧은 시간 동안 별도로 보관한다. 진행 이벤트 버퍼와 분리해 대형 결과가
        # 재생될 때 terminal event가 유실되지 않도록 한다.
        self._terminal_events: dict[str, dict] = {}
        self._terminal_ttl_seconds = terminal_ttl_seconds
        self._max_terminal_events = max_terminal_events

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
        self._remember_terminal(pipeline_id, "complete", data)
        self._enqueue(pipeline_id, "complete", data)
        self._enqueue(pipeline_id, "__done__", {})
        self._buffers.pop(pipeline_id, None)

    def error(self, pipeline_id: str | None, message: str, details: dict | None = None) -> None:
        if not pipeline_id:
            return
        data = {"message": message}
        if details:
            data.update(details)
        self._remember_terminal(pipeline_id, "error", data)
        self._enqueue(pipeline_id, "error", data)
        self._enqueue(pipeline_id, "__done__", {})
        self._buffers.pop(pipeline_id, None)

    async def subscribe(self, pipeline_id: str):
        """FastAPI StreamingResponse 용 async generator."""
        self._cleanup_terminal_events()
        q = self._get_or_create_queue(pipeline_id)

        # 버퍼에 쌓인 이벤트 먼저 전송
        for evt in self._buffers.get(pipeline_id, []):
            yield self._format(evt["type"], evt["data"])

        # 이미 끝난 파이프라인에 늦게 연결된 구독자는 terminal event를 즉시
        # 재생하고 대기하지 않는다. 이 경로가 Phase3 blocker 확인을 보장한다.
        terminal = self._terminal_events.get(pipeline_id)
        if terminal is not None:
            yield self._format(terminal["type"], terminal["data"])
            self._queues.pop(pipeline_id, None)
            return

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

    def _remember_terminal(self, pipeline_id: str, event_type: str, data: dict) -> None:
        self._cleanup_terminal_events()
        self._terminal_events[pipeline_id] = {
            "type": event_type,
            "data": data,
            "expires_at": time.monotonic() + self._terminal_ttl_seconds,
        }
        while len(self._terminal_events) > self._max_terminal_events:
            oldest = min(
                self._terminal_events,
                key=lambda key: self._terminal_events[key]["expires_at"],
            )
            self._terminal_events.pop(oldest, None)

    def _cleanup_terminal_events(self) -> None:
        now = time.monotonic()
        expired = [
            pipeline_id
            for pipeline_id, event in self._terminal_events.items()
            if event["expires_at"] <= now
        ]
        for pipeline_id in expired:
            self._terminal_events.pop(pipeline_id, None)

    def _enqueue(self, pipeline_id: str, event_type: str, data: dict) -> None:
        q = self._get_or_create_queue(pipeline_id)
        try:
            q.put_nowait({"type": event_type, "data": data})
        except asyncio.QueueFull:
            logger.warning("SSE 큐 가득 참: %s", pipeline_id)
