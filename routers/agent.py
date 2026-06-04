import json
import logging
from typing import Annotated

import httpx
from fastapi import APIRouter, Header, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/agent", tags=["agent"])

ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"

def _timeouts() -> tuple[int, int]:
    from main import settings
    return settings.agent_stream_timeout, settings.agent_sync_timeout


class AgentRequest(BaseModel):
    messages: list[dict]
    system_prompt: str | None = None


class AgentResponse(BaseModel):
    content: str
    metadata: dict = {}


def _api_keys():
    from main import settings
    return settings.openai_api_key, getattr(settings, "anthropic_api_key", "")


# ── SSE 스트리밍 ──────────────────────────────────────────────────

@router.post("/chat/stream")
async def chat_stream(
    x_llm_provider: Annotated[str, Header(alias="X-LLM-Provider")],
    x_llm_model: Annotated[str, Header(alias="X-LLM-Model")],
    request: AgentRequest,
):
    openai_key, anthropic_key = _api_keys()
    is_anthropic = x_llm_provider.lower() == "anthropic"
    api_key = anthropic_key if is_anthropic else openai_key

    async def generate():
        try:
            if is_anthropic:
                async for chunk in _stream_anthropic(api_key, x_llm_model, request):
                    yield chunk
            else:
                async for chunk in _stream_openai(api_key, x_llm_model, request):
                    yield chunk
        except Exception as e:
            logger.error("스트리밍 오류: %s", e)
            yield f"data: {{\"error\": \"{e}\"}}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


# ── 단일 응답 ─────────────────────────────────────────────────────

@router.post("/chat")
async def chat(
    x_llm_provider: Annotated[str, Header(alias="X-LLM-Provider")],
    x_llm_model: Annotated[str, Header(alias="X-LLM-Model")],
    request: AgentRequest,
) -> AgentResponse:
    openai_key, anthropic_key = _api_keys()
    is_anthropic = x_llm_provider.lower() == "anthropic"
    api_key = anthropic_key if is_anthropic else openai_key

    try:
        if is_anthropic:
            content = await _call_anthropic(api_key, x_llm_model, request)
        else:
            content = await _call_openai(api_key, x_llm_model, request)
        return AgentResponse(content=content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── API 연결 유효성 검증 ──────────────────────────────────────────

@router.post("/test")
async def test(
    x_llm_provider: Annotated[str, Header(alias="X-LLM-Provider")],
    x_llm_model: Annotated[str, Header(alias="X-LLM-Model")],
) -> dict:
    openai_key, anthropic_key = _api_keys()
    is_anthropic = x_llm_provider.lower() == "anthropic"
    api_key = anthropic_key if is_anthropic else openai_key

    ping = AgentRequest(messages=[{"role": "user", "content": "Hello"}], system_prompt=None)
    try:
        if is_anthropic:
            await _call_anthropic(api_key, x_llm_model, ping)
        else:
            await _call_openai(api_key, x_llm_model, ping)
        return {"ok": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Anthropic ─────────────────────────────────────────────────────

async def _call_anthropic(api_key: str, model: str, req: AgentRequest) -> str:
    _, sync_timeout = _timeouts()
    body = _build_anthropic_body(model, req, stream=False)
    async with httpx.AsyncClient(timeout=sync_timeout) as client:
        resp = await client.post(
            ANTHROPIC_API_URL,
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json=body,
        )
        resp.raise_for_status()
    data = resp.json()
    return data["content"][0]["text"]


async def _stream_anthropic(api_key: str, model: str, req: AgentRequest):
    stream_timeout, _ = _timeouts()
    body = _build_anthropic_body(model, req, stream=True)
    async with httpx.AsyncClient(timeout=stream_timeout) as client:
        async with client.stream(
            "POST",
            ANTHROPIC_API_URL,
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json=body,
        ) as resp:
            async for line in resp.aiter_lines():
                if not line.startswith("data:"):
                    continue
                data_str = line[5:].strip()
                if not data_str:
                    continue
                try:
                    node = json.loads(data_str)
                    if node.get("type") == "content_block_delta":
                        text = node.get("delta", {}).get("text", "")
                        if text:
                            yield f"data: {json.dumps({'delta': text})}\n\n"
                    elif node.get("type") == "message_stop":
                        break
                except Exception:
                    pass
    yield "data: [DONE]\n\n"


def _build_anthropic_body(model: str, req: AgentRequest, stream: bool) -> dict:
    messages = [m for m in req.messages if m.get("role") != "system"]
    body: dict = {"model": model, "max_tokens": 4096, "stream": stream, "messages": messages}
    if req.system_prompt:
        body["system"] = req.system_prompt
    return body


# ── OpenAI ────────────────────────────────────────────────────────

async def _call_openai(api_key: str, model: str, req: AgentRequest) -> str:
    _, sync_timeout = _timeouts()
    body = _build_openai_body(model, req, stream=False)
    async with httpx.AsyncClient(timeout=sync_timeout) as client:
        resp = await client.post(
            OPENAI_API_URL,
            headers={"Authorization": f"Bearer {api_key}", "content-type": "application/json"},
            json=body,
        )
        resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


async def _stream_openai(api_key: str, model: str, req: AgentRequest):
    stream_timeout, _ = _timeouts()
    body = _build_openai_body(model, req, stream=True)
    async with httpx.AsyncClient(timeout=stream_timeout) as client:
        async with client.stream(
            "POST",
            OPENAI_API_URL,
            headers={"Authorization": f"Bearer {api_key}", "content-type": "application/json"},
            json=body,
        ) as resp:
            async for line in resp.aiter_lines():
                if not line.startswith("data:"):
                    continue
                data_str = line[5:].strip()
                if data_str == "[DONE]":
                    break
                if not data_str:
                    continue
                try:
                    node = json.loads(data_str)
                    text = node["choices"][0].get("delta", {}).get("content", "")
                    if text:
                        yield f"data: {json.dumps({'delta': text})}\n\n"
                except Exception:
                    pass
    yield "data: [DONE]\n\n"


def _build_openai_body(model: str, req: AgentRequest, stream: bool) -> dict:
    messages = []
    if req.system_prompt:
        messages.append({"role": "system", "content": req.system_prompt})
    messages.extend(req.messages)
    return {"model": model, "stream": stream, "messages": messages}
