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


def test_stored_pipeline_emits_every_result_artifact(monkeypatch):
    """Phase 4 저장까지 끝난 실행은 프론트가 쓰는 산출물을 빠짐없이 내보낸다."""
    import json

    monkeypatch.setattr(settings, "kafka_enabled", True)
    registry = [
        {"featureId": "items.read", "name": "목록 조회", "priority": "P0"},
        {"featureId": "items.share", "name": "공유", "priority": "P1"},
    ]
    state = PipelineState(
        project_name="타이미룸",
        feature_list=["목록 조회", "공유"],
        feature_registry=registry,
        feature_specs=registry,
        feature_spec_document=json.dumps({"features": registry}, ensure_ascii=False),
        project_plan={"goal": "테스트"},
        prd_document=json.dumps({"coreFeatures": [{"name": "목록 조회"}]}, ensure_ascii=False),
        db_schema=json.dumps({"tables": [{"name": "items"}]}, ensure_ascii=False),
        api_spec=json.dumps({"endpoints": [{"method": "GET", "path": "/api/v1/items"}]}, ensure_ascii=False),
        openapi_spec=json.dumps({"openapi": "3.1.0"}, ensure_ascii=False),
        market_research="조사 결과",
        validated=True, qa_approved=True, status_message="완료",
        retry_count=1, rag_chunks_retrieved=7,
    )
    progress = MagicMock()
    kafka = MagicMock(
        publish=AsyncMock(return_value="run-1"),
        confirm_storage=AsyncMock(return_value={"status": "stored", "kafkaPublished": True}),
    )

    asyncio.run(orchestration._run_pipeline_inner(
        "run-1", FormData.model_construct(project_name="타이미룸"), [],
        SimpleNamespace(build_from_form=AsyncMock(return_value=state)),
        SimpleNamespace(run=AsyncMock(return_value=state)),
        SimpleNamespace(validate=lambda value: value),
        MagicMock(), kafka, progress, MagicMock(),
    ))

    progress.error.assert_not_called()
    result = progress.complete.call_args[0][1]
    assert result["completionStatus"] == "STORED"
    assert result["storage"]["status"] == "stored"
    assert result["featureNames"] == ["목록 조회", "공유"]
    assert result["featureList"] == registry
    assert result["allFeatures"] == ["목록 조회", "공유"]
    assert result["featureSummary"] == {"coreCount": 1, "supportingCount": 1, "totalCount": 2}
    assert result["prdDocument"]["coreFeatures"] == [{"name": "목록 조회"}]
    assert result["dbSchema"]["tables"] == [{"name": "items"}]
    assert result["apiSpec"]["endpoints"][0]["path"] == "/api/v1/items"
    assert result["openApiSpec"] == {"openapi": "3.1.0"}
    assert result["featureSpecDocument"] == {"features": registry}
    assert result["marketResearch"] == "조사 결과"
    assert result["retryCount"] == 1
    assert result["rag"] == {"chunksRetrieved": 7, "usedFormDataOnly": False}
