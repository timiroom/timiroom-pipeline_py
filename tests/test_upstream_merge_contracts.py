import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import httpx
from openai import RateLimitError

from config.settings import settings
from phase1.models import FormData
from phase2.llm_runtime import LlmRuntime
from phase2.state import PipelineState
from routers import orchestration


def test_busy_retry_preserves_a_feature_specific_timeout():
    calls = 0

    async def factory():
        nonlocal calls
        calls += 1
        if calls == 1:
            response = httpx.Response(429, request=httpx.Request("POST", "https://gateway.test/v1/chat/completions"))
            raise RateLimitError("busy", response=response, body=None)
        await asyncio.sleep(0.01)
        return "ok"

    runtime = LlmRuntime(1, 0.001, busy_retries=1, busy_backoff_seconds=0)
    assert asyncio.run(runtime.call(factory, timeout_seconds=2)) == "ok"
    assert calls == 2


def test_kafka_disabled_never_claims_stored_completion(monkeypatch):
    monkeypatch.setattr(settings, "kafka_enabled", False)
    state = PipelineState(project_name="test", validated=True)
    progress = MagicMock()
    kafka = MagicMock(publish=AsyncMock(), confirm_storage=AsyncMock())
    asyncio.run(orchestration._run_pipeline_inner(
        "test-run", FormData.model_construct(project_name="test"), [],
        SimpleNamespace(build_from_form=AsyncMock(return_value=state)),
        SimpleNamespace(run=AsyncMock(return_value=state)),
        SimpleNamespace(validate=lambda value: value), MagicMock(), kafka, progress, MagicMock(),
    ))
    kafka.publish.assert_not_awaited()
    kafka.confirm_storage.assert_not_awaited()
    progress.complete.assert_not_called()
    details = progress.error.call_args.kwargs["details"]
    assert details["storage"] == {"status": "skipped", "kafkaPublished": False, "reason": "KAFKA_DISABLED"}
    assert details["failedStage"] == "PHASE4"
