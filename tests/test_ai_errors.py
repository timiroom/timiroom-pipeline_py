import httpx
import openai
import pytest

from common.ai_errors import describe
from common.usage import LlmOverride, current_override

_REQUEST = httpx.Request("POST", "https://api.example.com/v1/chat/completions")


def _status(cls, status: int, body: dict):
    return cls("provider says: Incorrect API key provided: sk-abc***xyz", response=httpx.Response(status, request=_REQUEST), body=body)


@pytest.fixture
def byok():
    token = current_override.set(LlmOverride(provider="openai", base_url="https://api.example.com/v1", api_key="sk-test", model="gpt-x"))
    yield
    current_override.reset(token)


def test_user_key_problems_tell_the_user_to_fix_their_key(byok):
    invalid = describe(_status(openai.AuthenticationError, 401, {"error": {"code": "invalid_api_key"}}))
    assert invalid.code == "AI_KEY_INVALID" and invalid.status == 400 and "내 AI 키" in invalid.message
    quota = describe(_status(openai.RateLimitError, 429, {"error": {"code": "insufficient_quota"}}))
    assert quota.code == "AI_KEY_QUOTA_EXCEEDED" and quota.status == 402
    assert describe(_status(openai.NotFoundError, 404, {})).code == "AI_MODEL_NOT_FOUND"
    assert describe(_status(openai.PermissionDeniedError, 403, {})).code == "AI_KEY_FORBIDDEN"


def test_server_key_problems_never_blame_the_user():
    assert describe(_status(openai.AuthenticationError, 401, {})).code == "AI_SERVICE_UNAVAILABLE"
    assert describe(_status(openai.RateLimitError, 429, {"error": {"code": "insufficient_quota"}})).code == "AI_SERVICE_UNAVAILABLE"


def test_temporary_and_input_problems():
    assert describe(_status(openai.RateLimitError, 429, {"error": {"code": "rate_limit_exceeded"}})).code == "AI_BUSY"
    too_long = describe(_status(openai.BadRequestError, 400, {"error": {"code": "context_length_exceeded"}}))
    assert too_long.code == "AI_INPUT_TOO_LONG"
    assert describe(openai.APITimeoutError(request=_REQUEST)).code == "AI_TIMEOUT"
    assert describe(openai.APIConnectionError(request=_REQUEST)).code == "AI_SERVICE_UNAVAILABLE"


def test_messages_never_leak_provider_text_and_wrapped_errors_are_found(byok):
    try:
        try:
            raise _status(openai.AuthenticationError, 401, {})
        except openai.AuthenticationError as inner:
            raise RuntimeError("retry wrapper") from inner
    except RuntimeError as wrapped:
        ai = describe(wrapped)
    assert ai.code == "AI_KEY_INVALID"
    assert "sk-" not in ai.message and "Incorrect" not in ai.message
    assert describe(ValueError("not an AI error")) is None


def test_handler_returns_korean_message_and_code_instead_of_500():
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from common.exception_handler import register_exception_handlers

    app = FastAPI()
    register_exception_handlers(app)

    @app.get("/boom")
    async def boom():
        raise _status(openai.RateLimitError, 429, {"error": {"code": "rate_limit_exceeded"}})

    response = TestClient(app, raise_server_exceptions=False).get("/boom")
    assert response.status_code == 429
    body = response.json()
    assert body["code"] == "AI_BUSY" and "다시 시도" in body["message"] and "sk-" not in response.text


def test_generation_failure_sends_korean_message_and_code_to_progress(byok):
    import asyncio
    from unittest.mock import MagicMock

    from phase1.models import FormData
    from routers import orchestration

    progress = MagicMock()
    form_to_query = MagicMock()
    form_to_query.synthesize.side_effect = _status(openai.AuthenticationError, 401, {})
    asyncio.run(orchestration._run_pipeline_inner(
        "pid-1", FormData.model_construct(project_name="p", platform="web", tech_stack=[], problem_definition="x", target_users="y"),
        [], MagicMock(), MagicMock(), MagicMock(), MagicMock(), MagicMock(), progress, form_to_query, skip_phase1=True,
    ))
    _, message = progress.error.call_args.args
    details = progress.error.call_args.kwargs["details"]
    assert details["errorCode"] == "AI_KEY_INVALID" and "내 AI 키" in message
    assert details["errorDetail"] is None and "sk-" not in message
