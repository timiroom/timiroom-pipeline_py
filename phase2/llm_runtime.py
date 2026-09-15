import asyncio
from collections.abc import Awaitable, Callable
from typing import TypeVar

T = TypeVar("T")


class LlmRuntime:
    """Phase 2 전체가 공유하는 LLM 호출 동시성·시간 제한."""

    def __init__(self, max_concurrency: int, request_timeout_seconds: float):
        if max_concurrency <= 0:
            raise ValueError("max_concurrency는 1 이상이어야 합니다")
        if request_timeout_seconds <= 0:
            raise ValueError("request_timeout_seconds는 0보다 커야 합니다")
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._request_timeout_seconds = request_timeout_seconds

    async def call(self, factory: Callable[[], Awaitable[T]]) -> T:
        async with self._semaphore:
            async with asyncio.timeout(self._request_timeout_seconds):
                return await factory()
