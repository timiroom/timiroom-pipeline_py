"""AI 사용량 수집과 사용자 키(BYOK) 적용.

백엔드가 사용자별 사용량을 셀 수 있도록, 이 서버가 호출하는 모든 OpenAI 호환 요청의
토큰 수를 요청 단위로 모은다. 호출 지점이 20곳이 넘어 각각 고치지 않고, 라이브러리의
create 함수를 한 번 감싸서 처리한다(install()).

- 요청 단위 범위: 미들웨어가 요청마다 UsageCollector를 만들어 contextvar에 둔다.
  asyncio.create_task는 contextvar를 복사하므로, 백그라운드로 도는 생성 파이프라인도
  같은 수집기에 기록된다.
- 사용자 키: 백엔드가 사용자 키를 헤더로 넘기면 그 요청 안의 "글 생성" 호출(chat completions)은
  사용자 서비스·모델로 보낸다. 검색용 임베딩은 서버 키를 그대로 쓴다.
"""
from __future__ import annotations

import logging
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any

from openai import AsyncOpenAI
from openai.resources.chat.completions import AsyncCompletions
from openai.resources.embeddings import AsyncEmbeddings

logger = logging.getLogger(__name__)


@dataclass
class UsageRecord:
    kind: str            # "chat" | "embedding"
    provider: str        # "server" 또는 사용자 키의 제공자 이름
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    byok: bool = False


@dataclass
class UsageCollector:
    records: list[UsageRecord] = field(default_factory=list)

    def add(self, record: UsageRecord) -> None:
        self.records.append(record)

    def summary(self) -> list[dict[str, Any]]:
        """같은 (종류, 제공자, 모델, 사용자 키 여부)끼리 합친 목록. 백엔드로 보내는 형식이다."""
        merged: dict[tuple, dict[str, Any]] = {}
        for r in self.records:
            key = (r.kind, r.provider, r.model, r.byok)
            row = merged.setdefault(key, {
                "kind": r.kind, "provider": r.provider, "model": r.model, "byok": r.byok,
                "inputTokens": 0, "outputTokens": 0, "calls": 0,
            })
            row["inputTokens"] += r.input_tokens
            row["outputTokens"] += r.output_tokens
            row["calls"] += 1
        return list(merged.values())


@dataclass(frozen=True)
class LlmOverride:
    """사용자 키 설정. api_key는 로그에 남기지 않는다."""
    provider: str
    base_url: str
    api_key: str
    model: str

    def __repr__(self) -> str:  # 키가 로그·예외 메시지로 새지 않게 한다
        return f"LlmOverride(provider={self.provider!r}, base_url={self.base_url!r}, model={self.model!r})"


current_collector: ContextVar[UsageCollector | None] = ContextVar("usage_collector", default=None)
current_override: ContextVar[LlmOverride | None] = ContextVar("llm_override", default=None)
_redirecting: ContextVar[bool] = ContextVar("llm_redirecting", default=False)

_installed = False
_original_chat_create = AsyncCompletions.create
_original_embedding_create = AsyncEmbeddings.create
_override_clients: dict[tuple[str, str], AsyncOpenAI] = {}


def _record(kind: str, model: str, usage: Any, *, provider: str, byok: bool) -> None:
    collector = current_collector.get()
    if collector is None or usage is None:
        return
    input_tokens = getattr(usage, "prompt_tokens", None) or getattr(usage, "input_tokens", None) or 0
    output_tokens = getattr(usage, "completion_tokens", None) or getattr(usage, "output_tokens", None) or 0
    collector.add(UsageRecord(kind=kind, provider=provider, model=model or "",
                              input_tokens=int(input_tokens), output_tokens=int(output_tokens), byok=byok))


class _UsageTrackingStream:
    """스트리밍 응답을 그대로 흘려보내면서 마지막 청크의 usage를 기록한다."""

    def __init__(self, stream: Any, model: str, provider: str, byok: bool):
        self._stream = stream
        self._model = model
        self._provider = provider
        self._byok = byok

    def __aiter__(self):
        return self._iterate()

    async def _iterate(self):
        usage = None
        async for chunk in self._stream:
            if getattr(chunk, "usage", None) is not None:
                usage = chunk.usage
            yield chunk
        _record("chat", self._model, usage, provider=self._provider, byok=self._byok)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        close = getattr(self._stream, "close", None)
        if close is not None:
            await close()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._stream, name)


def _client_for(override: LlmOverride) -> AsyncOpenAI:
    key = (override.base_url, override.api_key)
    client = _override_clients.get(key)
    if client is None:
        client = AsyncOpenAI(api_key=override.api_key, base_url=override.base_url, max_retries=1, timeout=120.0)
        if len(_override_clients) > 256:  # 오래 도는 프로세스에서 무한히 쌓이지 않게 한다
            _override_clients.clear()
        _override_clients[key] = client
    return client


async def _patched_chat_create(self: AsyncCompletions, *args: Any, **kwargs: Any) -> Any:
    override = current_override.get()
    if override is not None and not _redirecting.get():
        # 사용자 키 요청: 모델을 사용자가 고른 것으로 바꾸고, 서버 제공자 전용 옵션은 뺀다.
        kwargs["model"] = override.model
        kwargs.pop("extra_body", None)
        token = _redirecting.set(True)
        try:
            return await _call_chat(_client_for(override).chat.completions, args, kwargs,
                                    provider=override.provider, byok=True)
        finally:
            _redirecting.reset(token)
    return await _call_chat(self, args, kwargs, provider="server", byok=False)


async def _call_chat(target: AsyncCompletions, args: tuple, kwargs: dict, *, provider: str, byok: bool) -> Any:
    model = kwargs.get("model", "")
    if kwargs.get("stream") and current_collector.get() is not None:
        from config.settings import settings
        if settings.usage_stream_include_usage:
            kwargs.setdefault("stream_options", {"include_usage": True})
        stream = await _original_chat_create(target, *args, **kwargs)
        return _UsageTrackingStream(stream, model, provider, byok)
    response = await _original_chat_create(target, *args, **kwargs)
    if not kwargs.get("stream"):
        _record("chat", getattr(response, "model", None) or model, getattr(response, "usage", None),
                provider=provider, byok=byok)
    return response


async def _patched_embedding_create(self: AsyncEmbeddings, *args: Any, **kwargs: Any) -> Any:
    response = await _original_embedding_create(self, *args, **kwargs)
    _record("embedding", kwargs.get("model", ""), getattr(response, "usage", None), provider="server", byok=False)
    return response


def install() -> None:
    """앱 시작 시 한 번 호출한다. 여러 번 불러도 안전하다."""
    global _installed
    if _installed:
        return
    AsyncCompletions.create = _patched_chat_create  # type: ignore[method-assign]
    AsyncEmbeddings.create = _patched_embedding_create  # type: ignore[method-assign]
    _installed = True
    logger.info("AI 사용량 수집 설치 완료")


def uninstall() -> None:
    """테스트용: 원래 함수로 되돌린다."""
    global _installed
    AsyncCompletions.create = _original_chat_create  # type: ignore[method-assign]
    AsyncEmbeddings.create = _original_embedding_create  # type: ignore[method-assign]
    _installed = False
