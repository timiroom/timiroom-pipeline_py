"""백엔드 전용 경로 보호와 요청 단위 사용량 수집.

- 내부 토큰: settings.internal_api_token이 설정돼 있으면 AI를 쓰는 경로는
  X-Timiroom-Internal-Token이 일치해야 한다. 브라우저가 이 서버를 직접 불러 AI를 공짜로 쓰는
  경로를 막는다. 비어 있으면(로컬·이행 기간) 검사하지 않는다.
- 사용량: 백엔드가 X-Timiroom-Usage-Tracking: 1을 보내면 요청마다 수집기를 만든다.
  스트리밍이 아닌 응답에는 X-Timiroom-Usage 헤더(base64 JSON)로 돌려준다. 스트리밍 응답과
  생성 파이프라인은 각 라우터가 본문 안에 사용량을 넣는다.
- 사용자 키: X-Timiroom-Llm-* 헤더가 모두 있으면 그 요청의 글 생성 호출을 사용자 키로 보낸다.
"""
from __future__ import annotations

import base64
import hmac
import json
from urllib.parse import urlparse

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from common.usage import LlmOverride, UsageCollector, current_collector, current_override
from config.settings import settings

PROTECTED_PREFIXES = (
    "/api/agent",
    "/api/v1/chat",
    "/api/v1/document",
    "/api/v1/orchestration",
    "/api/v1/recommendation",
    "/api/v1/rag",
)
USAGE_HEADER = "X-Timiroom-Usage"


def _is_protected(path: str) -> bool:
    return any(path == prefix or path.startswith(prefix + "/") for prefix in PROTECTED_PREFIXES)


def _override_from(request: Request) -> LlmOverride | None:
    headers = request.headers
    values = {
        "provider": headers.get("x-timiroom-llm-provider", "").strip(),
        "base_url": headers.get("x-timiroom-llm-base-url", "").strip(),
        "api_key": headers.get("x-timiroom-llm-api-key", "").strip(),
        "model": headers.get("x-timiroom-llm-model", "").strip(),
    }
    if not any(values.values()):
        return None
    if not all(values.values()):
        raise ValueError("사용자 키 헤더가 일부만 왔습니다")
    url = urlparse(values["base_url"])
    local = url.hostname in ("localhost", "127.0.0.1")
    if url.scheme != "https" and not (url.scheme == "http" and local and settings.byok_allow_local_http):
        raise ValueError("사용자 키의 주소는 https여야 합니다")
    return LlmOverride(**values)


def encode_usage(summary: list[dict]) -> str:
    return base64.urlsafe_b64encode(json.dumps(summary, separators=(",", ":")).encode()).decode()


class UsageMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next) -> Response:
        if _is_protected(request.url.path) and settings.internal_api_token:
            given = request.headers.get("x-timiroom-internal-token", "")
            if not hmac.compare_digest(given.encode(), settings.internal_api_token.encode()):
                return JSONResponse({"detail": "내부 요청만 허용됩니다"}, status_code=401)

        tracking = request.headers.get("x-timiroom-usage-tracking") == "1"
        try:
            override = _override_from(request)
        except ValueError as e:
            return JSONResponse({"detail": str(e)}, status_code=400)

        collector = UsageCollector() if tracking else None
        collector_token = current_collector.set(collector)
        override_token = current_override.set(override)
        try:
            response = await call_next(request)
        finally:
            current_collector.reset(collector_token)
            current_override.reset(override_token)

        if collector is not None and "text/event-stream" not in response.headers.get("content-type", ""):
            response.headers[USAGE_HEADER] = encode_usage(collector.summary())
        return response
