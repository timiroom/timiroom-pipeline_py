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

def _timeouts() -> tuple[int, int]:
    from main import settings
    return settings.agent_stream_timeout, settings.agent_sync_timeout


class AgentRequest(BaseModel):
    messages: list[dict]
    system_prompt: str | None = None


class AgentResponse(BaseModel):
    content: str
    metadata: dict = {}


def _anthropic_key() -> str:
    from main import settings
    return getattr(settings, "anthropic_api_key", "")


# ── SSE 스트리밍 ──────────────────────────────────────────────────

@router.post("/chat/stream")
async def chat_stream(
    x_llm_provider: Annotated[str, Header(alias="X-LLM-Provider")],
    x_llm_model: Annotated[str, Header(alias="X-LLM-Model")],
    request: AgentRequest,
):
    if x_llm_provider.lower() != "anthropic":
        raise HTTPException(status_code=400, detail="지원하지 않는 LLM 프로바이더입니다. anthropic만 사용 가능합니다.")
    api_key = _anthropic_key()

    async def generate():
        try:
            async for chunk in _stream_anthropic(api_key, x_llm_model, request):
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
    if x_llm_provider.lower() != "anthropic":
        raise HTTPException(status_code=400, detail="지원하지 않는 LLM 프로바이더입니다. anthropic만 사용 가능합니다.")
    api_key = _anthropic_key()
    try:
        content = await _call_anthropic(api_key, x_llm_model, request)
        return AgentResponse(content=content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── API 연결 유효성 검증 ──────────────────────────────────────────

@router.post("/test")
async def test(
    x_llm_provider: Annotated[str, Header(alias="X-LLM-Provider")],
    x_llm_model: Annotated[str, Header(alias="X-LLM-Model")],
) -> dict:
    if x_llm_provider.lower() != "anthropic":
        raise HTTPException(status_code=400, detail="지원하지 않는 LLM 프로바이더입니다. anthropic만 사용 가능합니다.")
    api_key = _anthropic_key()
    ping = AgentRequest(messages=[{"role": "user", "content": "Hello"}], system_prompt=None)
    try:
        await _call_anthropic(api_key, x_llm_model, ping)
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

