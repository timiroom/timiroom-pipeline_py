"""문서 패널의 AI 어시스턴트 채팅 — EXAONE 스트리밍.

파이프라인 전 구간(추천·에이전트·수집 채팅)이 EXAONE을 쓰므로 이 채팅도 같은 모델을 쓴다.
예전에는 Anthropic API를 직접 호출했고 프론트가 프로바이더/모델을 골랐지만,
모델 선택 UI를 없애면서 그 경로도 함께 제거했다.

문서를 실제로 고치는 것은 이 라우터가 아니라 routers/document.py다.
여기는 자유 대화(질문·조언)만 담당한다.
"""
import asyncio
import json
import logging

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from openai import InternalServerError, APITimeoutError, APIConnectionError
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/agent", tags=["agent"])

# EXAONE 모델 카드 권장 샘플링 파라미터
# https://huggingface.co/LGAI-EXAONE/K-EXAONE-236B-A23B
_TEMPERATURE = 1.0
_TOP_P = 0.95
_PRESENCE_PENALTY = 0.0

_MAX_TOKENS = 4096


def _exaone_endpoint_id() -> str:
    from main import settings
    return settings.exaone_endpoint_id


class AgentRequest(BaseModel):
    messages: list[dict]
    system_prompt: str | None = None


class AgentResponse(BaseModel):
    content: str
    metadata: dict = {}


def _build_messages(req: AgentRequest) -> list[dict]:
    """system_prompt를 맨 앞 system 메시지로 세우고 나머지 대화를 잇는다.
    클라이언트가 messages 안에 넣어 보낸 system 역할은 무시한다 — 프롬프트 주입 경로가 된다."""
    messages: list[dict] = []
    if req.system_prompt:
        messages.append({"role": "system", "content": req.system_prompt})
    for m in req.messages:
        role = m.get("role")
        content = m.get("content")
        if role in ("user", "assistant") and isinstance(content, str) and content.strip():
            messages.append({"role": role, "content": content})
    return messages


# ── SSE 스트리밍 ──────────────────────────────────────────────────

@router.post("/chat/stream")
async def chat_stream(request: AgentRequest):
    from main import exaone_client

    async def generate():
        try:
            stream = await exaone_client.chat.completions.create(
                model=_exaone_endpoint_id(),
                max_tokens=_MAX_TOKENS,
                temperature=_TEMPERATURE,
                top_p=_TOP_P,
                presence_penalty=_PRESENCE_PENALTY,
                messages=_build_messages(request),
                stream=True,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
            async for chunk in stream:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                text = getattr(delta, "content", None)
                if text:
                    yield f"data: {json.dumps({'delta': text})}\n\n"
        except Exception as e:
            logger.error("EXAONE 스트리밍 오류: %s", e, exc_info=True)
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        # 프록시(nginx 등)가 SSE를 버퍼링해 청크가 한꺼번에 도착하는 것을 막는다
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ── 단일 응답 ─────────────────────────────────────────────────────

@router.post("/chat")
async def chat(request: AgentRequest) -> AgentResponse:
    try:
        content = await _call_exaone(_build_messages(request))
        return AgentResponse(content=content)
    except Exception as e:
        logger.error("EXAONE 호출 실패: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ── 연결 확인 ─────────────────────────────────────────────────────

@router.post("/test")
async def test() -> dict:
    try:
        await _call_exaone([{"role": "user", "content": "안녕하세요"}], max_tokens=32)
        return {"ok": True}
    except Exception as e:
        logger.error("EXAONE 연결 테스트 실패: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


async def _call_exaone(messages: list[dict], max_tokens: int = _MAX_TOKENS) -> str:
    """EXAONE 호출 (재시도 3회) — routers/chat.py의 동일 헬퍼와 같은 파라미터."""
    from main import exaone_client

    for attempt in range(3):
        try:
            resp = await exaone_client.chat.completions.create(
                model=_exaone_endpoint_id(),
                max_tokens=max_tokens,
                temperature=_TEMPERATURE,
                top_p=_TOP_P,
                presence_penalty=_PRESENCE_PENALTY,
                messages=messages,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
            return resp.choices[0].message.content or ""
        except (InternalServerError, APITimeoutError, APIConnectionError) as e:
            logger.warning("EXAONE 일시 오류 (attempt %d): %s — 재시도", attempt + 1, e)
            if attempt < 2:
                await asyncio.sleep(3 * (attempt + 1))
            else:
                raise
    return ""
