import asyncio
import os
from contextlib import asynccontextmanager


_limit = max(1, int(os.getenv("PHASE2_LLM_CONCURRENCY", "3")))
_semaphore: asyncio.Semaphore | None = None


@asynccontextmanager
async def llm_slot():
    """Share one bounded EXAONE request pool across all parallel agents."""
    global _semaphore
    if _semaphore is None:
        _semaphore = asyncio.Semaphore(_limit)
    async with _semaphore:
        yield
