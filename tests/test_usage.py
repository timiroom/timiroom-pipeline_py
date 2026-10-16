"""AI 사용량 수집·사용자 키 전환·내부 토큰 보호 테스트. 실제 AI는 부르지 않는다."""
import asyncio
import base64
import json

import httpx
import pytest
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.testclient import TestClient
from openai import AsyncOpenAI

from common import usage
from common.usage import LlmOverride, UsageCollector, current_collector, current_override
from common.usage_middleware import UsageMiddleware
from config.settings import settings


def _completion(model="server-model", prompt=11, completion=7):
    return {
        "id": "c1", "object": "chat.completion", "created": 0, "model": model,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "안녕"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": prompt, "completion_tokens": completion, "total_tokens": prompt + completion},
    }


def _client(handler, base_url="https://server.test/v1"):
    return AsyncOpenAI(api_key="k", base_url=base_url, max_retries=0,
                       http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))


@pytest.fixture(autouse=True)
def _installed():
    usage.install()
    yield
    usage.uninstall()


def test_chat_usage_is_recorded_per_request():
    async def run():
        client = _client(lambda request: httpx.Response(200, json=_completion()))
        collector = UsageCollector()
        token = current_collector.set(collector)
        try:
            await client.chat.completions.create(model="server-model", messages=[{"role": "user", "content": "hi"}])
            await client.chat.completions.create(model="server-model", messages=[{"role": "user", "content": "hi"}])
        finally:
            current_collector.reset(token)
        assert collector.summary() == [{
            "kind": "chat", "provider": "server", "model": "server-model", "byok": False,
            "inputTokens": 22, "outputTokens": 14, "calls": 2,
        }]

    asyncio.run(run())


def test_nothing_is_recorded_without_collector():
    async def run():
        client = _client(lambda request: httpx.Response(200, json=_completion()))
        response = await client.chat.completions.create(model="m", messages=[{"role": "user", "content": "hi"}])
        assert response.choices[0].message.content == "안녕"

    asyncio.run(run())


def test_stream_requests_usage_and_records_last_chunk():
    async def run():
        seen = {}

        def handler(request):
            seen["body"] = json.loads(request.content)
            chunks = [
                {"id": "s", "object": "chat.completion.chunk", "created": 0, "model": "m",
                 "choices": [{"index": 0, "delta": {"content": "가"}, "finish_reason": None}]},
                {"id": "s", "object": "chat.completion.chunk", "created": 0, "model": "m", "choices": [],
                 "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8}},
            ]
            body = "".join(f"data: {json.dumps(c)}\n\n" for c in chunks) + "data: [DONE]\n\n"
            return httpx.Response(200, text=body, headers={"content-type": "text/event-stream"})

        client = _client(handler)
        collector = UsageCollector()
        token = current_collector.set(collector)
        try:
            stream = await client.chat.completions.create(model="m", messages=[{"role": "user", "content": "hi"}], stream=True)
            texts = [c.choices[0].delta.content async for c in stream if c.choices]
        finally:
            current_collector.reset(token)
        assert texts == ["가"]
        assert seen["body"]["stream_options"] == {"include_usage": True}
        assert collector.summary()[0]["inputTokens"] == 5
        assert collector.summary()[0]["outputTokens"] == 3

    asyncio.run(run())


def test_byok_redirects_chat_to_user_provider_and_model(monkeypatch):
    async def run():
        server_calls, user_calls = [], []
        server = _client(lambda r: server_calls.append(r) or httpx.Response(200, json=_completion()))

        def user_handler(request):
            body = json.loads(request.content)
            user_calls.append(body)
            return httpx.Response(200, json=_completion(model=body["model"], prompt=3, completion=2))

        user_client = _client(user_handler, base_url="https://user.test/v1")
        monkeypatch.setattr(usage, "_client_for", lambda override: user_client)

        collector = UsageCollector()
        c_token = current_collector.set(collector)
        o_token = current_override.set(LlmOverride(provider="openai", base_url="https://user.test/v1",
                                                   api_key="sk-user", model="gpt-user"))
        try:
            await server.chat.completions.create(model="server-model", messages=[{"role": "user", "content": "hi"}],
                                                 extra_body={"server_only": True})
        finally:
            current_override.reset(o_token)
            current_collector.reset(c_token)

        assert server_calls == []
        assert user_calls[0]["model"] == "gpt-user"
        assert "server_only" not in user_calls[0]
        assert collector.summary() == [{
            "kind": "chat", "provider": "openai", "model": "gpt-user", "byok": True,
            "inputTokens": 3, "outputTokens": 2, "calls": 1,
        }]

    asyncio.run(run())


def test_override_repr_hides_key():
    text = repr(LlmOverride(provider="openai", base_url="https://x", api_key="sk-secret", model="m"))
    assert "sk-secret" not in text


def _app():
    app = FastAPI()
    app.add_middleware(UsageMiddleware)

    @app.post("/api/agent/chat")
    async def chat():
        collector = current_collector.get()
        if collector is not None:
            collector.add(usage.UsageRecord(kind="chat", provider="server", model="m", input_tokens=4, output_tokens=2))
        override = current_override.get()
        return {"override": override.model if override else None}

    @app.post("/api/agent/chat/stream")
    async def stream():
        async def gen():
            yield "data: x\n\n"
        return StreamingResponse(gen(), media_type="text/event-stream")

    @app.get("/actuator/health")
    async def health():
        return {"status": "UP"}

    return app


def test_internal_token_required_when_configured(monkeypatch):
    monkeypatch.setattr(settings, "internal_api_token", "secret-token")
    client = TestClient(_app())
    assert client.post("/api/agent/chat").status_code == 401
    assert client.post("/api/agent/chat", headers={"X-Timiroom-Internal-Token": "wrong"}).status_code == 401
    assert client.post("/api/agent/chat", headers={"X-Timiroom-Internal-Token": "secret-token"}).status_code == 200
    assert client.get("/actuator/health").status_code == 200


def test_usage_header_returned_only_when_tracking(monkeypatch):
    monkeypatch.setattr(settings, "internal_api_token", "")
    client = TestClient(_app())
    assert "x-timiroom-usage" not in client.post("/api/agent/chat").headers
    response = client.post("/api/agent/chat", headers={"X-Timiroom-Usage-Tracking": "1"})
    decoded = json.loads(base64.urlsafe_b64decode(response.headers["x-timiroom-usage"]))
    assert decoded[0]["inputTokens"] == 4
    streamed = client.post("/api/agent/chat/stream", headers={"X-Timiroom-Usage-Tracking": "1"})
    assert "x-timiroom-usage" not in streamed.headers


def test_byok_headers_validated(monkeypatch):
    monkeypatch.setattr(settings, "internal_api_token", "")
    monkeypatch.setattr(settings, "byok_allow_local_http", False)
    client = TestClient(_app())
    full = {"X-Timiroom-Llm-Provider": "openai", "X-Timiroom-Llm-Base-Url": "https://api.openai.com/v1",
            "X-Timiroom-Llm-Api-Key": "sk-x", "X-Timiroom-Llm-Model": "gpt-x"}
    assert client.post("/api/agent/chat", headers=full).json() == {"override": "gpt-x"}
    partial = dict(full)
    partial.pop("X-Timiroom-Llm-Model")
    assert client.post("/api/agent/chat", headers=partial).status_code == 400
    plain = dict(full, **{"X-Timiroom-Llm-Base-Url": "http://evil.test/v1"})
    assert client.post("/api/agent/chat", headers=plain).status_code == 400
