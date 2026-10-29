import asyncio
import logging
import random
from collections.abc import Awaitable, Callable
from typing import TypeVar

from openai import (
    APIConnectionError,
    APITimeoutError,
    InternalServerError,
    RateLimitError,
)

from common.ai_errors import is_quota_exhausted

T = TypeVar("T")

logger = logging.getLogger(__name__)

# 게이트웨이가 잠깐 바쁘거나(429) 순간 오류(5xx, 연결)일 때는 생성 전체를 실패시키지 않고 기다렸다 다시 부른다.
# 다른 서비스(정합성 검사, AI 채팅)와 같은 게이트웨이를 나눠 쓰므로 동시 요청 한도에 자주 닿는다.
_RETRYABLE = (RateLimitError, APIConnectionError, APITimeoutError, InternalServerError)


class LlmRuntime:
    """Phase 2 전체가 공유하는 LLM 호출 동시성·시간 제한과 일시 오류 재시도."""

    def __init__(
        self,
        max_concurrency: int,
        request_timeout_seconds: float,
        busy_retries: int = 5,
        busy_backoff_seconds: float = 2.0,
    ):
        if max_concurrency <= 0:
            raise ValueError("max_concurrency는 1 이상이어야 합니다")
        if request_timeout_seconds <= 0:
            raise ValueError("request_timeout_seconds는 0보다 커야 합니다")
        if busy_retries < 0:
            raise ValueError("busy_retries는 0 이상이어야 합니다")
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._request_timeout_seconds = request_timeout_seconds
        self._busy_retries = busy_retries
        self._busy_backoff_seconds = busy_backoff_seconds

    async def call(self, factory: Callable[[], Awaitable[T]], *, timeout_seconds: float | None = None) -> T:
        timeout = self._request_timeout_seconds if timeout_seconds is None else timeout_seconds
        if timeout <= 0:
            raise ValueError("timeout_seconds must be > 0")
        attempt = 0
        while True:
            try:
                async with self._semaphore:
                    async with asyncio.timeout(timeout):
                        return await factory()
            except _RETRYABLE as exc:
                # 크레딧 소진은 재시도해도 같은 429만 돌아오고 단계 시간만 늘린다.
                if attempt >= self._busy_retries or is_quota_exhausted(exc):
                    raise
                attempt += 1
                # 기다리는 동안에는 자리(semaphore)를 내준다. 2, 4, 8, 16, 32초에 흔들림을 더한다.
                delay = min(30.0, self._busy_backoff_seconds * (2 ** (attempt - 1))) * (0.8 + random.random() * 0.4)
                logger.warning("LLM 일시 오류(%s) — %.1f초 뒤 다시 시도 (%d/%d)",
                               type(exc).__name__, delay, attempt, self._busy_retries)
                await asyncio.sleep(delay)
