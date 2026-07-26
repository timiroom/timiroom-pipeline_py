import logging
import uuid
from contextvars import ContextVar

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

# 현재 요청의 requestId를 비동기 컨텍스트에 저장 (MDC 대체)
request_id_var: ContextVar[str] = ContextVar("request_id", default="NO_REQ")

# 실행 중인 파이프라인의 pipelineId를 비동기 컨텍스트에 저장 (Java OrchestrationController의 MDC.put("pipelineId", ...)에 대응)
pipeline_id_var: ContextVar[str] = ContextVar("pipeline_id", default="NO_PIPE")


class RequestIdMiddleware(BaseHTTPMiddleware):
    """모든 요청에 고유 requestId 부여 — Java MdcLoggingFilter에 대응."""

    async def dispatch(self, request: Request, call_next):
        req_id = uuid.uuid4().hex[:8]
        token = request_id_var.set(req_id)
        try:
            response = await call_next(request)
            response.headers["X-Request-Id"] = req_id
            return response
        finally:
            request_id_var.reset(token)


class RequestIdFilter(logging.Filter):
    """로그 레코드에 request_id, pipeline_id 필드를 자동 주입."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_var.get()
        record.pipeline_id = pipeline_id_var.get()
        return True
