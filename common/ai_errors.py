"""AI 제공사 오류를 사용자에게 보여 줄 한국어 안내로 바꾼다.

제공사 원문(영어, 키 일부가 섞일 수 있음)은 화면에 내보내지 않고 로그에만 남긴다.
사용자 키(BYOK)로 부른 호출이면 키를 고치라는 안내를, 서버 키면 티미룸 쪽 문제라는 안내를 준다.
"""
from __future__ import annotations

from dataclasses import dataclass

import openai


@dataclass(frozen=True)
class AiError:
    status: int
    code: str
    message: str


def _byok() -> bool:
    from common.usage import current_override
    return current_override.get() is not None


def _root(exc: BaseException) -> BaseException:
    """재시도 래퍼나 다른 예외에 싸여 있어도 제공사 오류를 찾는다."""
    seen = 0
    current: BaseException | None = exc
    while current is not None and seen < 10:
        if isinstance(current, (openai.APIError, TimeoutError)):
            return current
        current = current.__cause__ or current.__context__
        seen += 1
    return exc


def _text(exc: BaseException) -> str:
    body = getattr(exc, "body", None)
    parts = [str(getattr(exc, "code", "") or ""), str(body or ""), str(exc)]
    return " ".join(parts).lower()


def describe(exc: BaseException) -> AiError | None:
    """AI 호출 오류면 안내를 돌려준다. AI 오류가 아니면 None."""
    root = _root(exc)
    byok = _byok()
    text = _text(root)
    if isinstance(root, openai.AuthenticationError):
        if byok:
            return AiError(400, "AI_KEY_INVALID", "연결한 AI 키가 올바르지 않거나 만료되었습니다. 설정 > 내 AI 키에서 키를 확인해 주세요.")
        return AiError(503, "AI_SERVICE_UNAVAILABLE", "티미룸 AI 연결에 문제가 생겼습니다. 잠시 후 다시 시도해 주세요.")
    if isinstance(root, openai.PermissionDeniedError):
        if byok:
            return AiError(400, "AI_KEY_FORBIDDEN", "연결한 AI 키로 이 모델을 쓸 권한이 없습니다. 설정 > 내 AI 키에서 모델이나 키 권한을 확인해 주세요.")
        return AiError(503, "AI_SERVICE_UNAVAILABLE", "티미룸 AI 연결에 문제가 생겼습니다. 잠시 후 다시 시도해 주세요.")
    if isinstance(root, openai.NotFoundError):
        if byok:
            return AiError(400, "AI_MODEL_NOT_FOUND", "연결한 AI 키에서 고른 모델을 찾지 못했습니다. 설정 > 내 AI 키에서 모델 이름을 확인해 주세요.")
        return AiError(503, "AI_SERVICE_UNAVAILABLE", "티미룸 AI 모델을 찾지 못했습니다. 잠시 후 다시 시도해 주세요.")
    if isinstance(root, openai.RateLimitError):
        if "insufficient_quota" in text or "quota" in text or "billing" in text or "credit" in text:
            if byok:
                return AiError(402, "AI_KEY_QUOTA_EXCEEDED", "연결한 AI 키의 크레딧이나 사용 한도를 모두 썼습니다. AI 제공사 결제 화면에서 확인하거나, 설정 > 내 AI 키에서 키 사용을 끄고 티미룸 AI로 계속하세요.")
            return AiError(503, "AI_SERVICE_UNAVAILABLE", "티미룸 AI 사용량이 잠시 한도에 닿았습니다. 잠시 후 다시 시도해 주세요.")
        return AiError(429, "AI_BUSY", "AI 요청이 잠시 몰렸습니다. 30초쯤 뒤에 다시 시도해 주세요.")
    if isinstance(root, openai.BadRequestError):
        if "context_length" in text or "maximum context" in text or "too long" in text or "max_tokens" in text:
            return AiError(400, "AI_INPUT_TOO_LONG", "보낸 내용이 AI가 한 번에 읽을 수 있는 길이를 넘었습니다. 내용을 줄여서 다시 보내 주세요.")
        return AiError(400, "AI_REQUEST_REJECTED", "AI가 이 요청을 처리하지 못했습니다. 내용을 조금 바꿔서 다시 시도해 주세요.")
    if isinstance(root, (openai.APITimeoutError, TimeoutError)):
        return AiError(504, "AI_TIMEOUT", "AI 응답이 오래 걸려 중단했습니다. 잠시 후 다시 시도해 주세요.")
    if isinstance(root, (openai.APIConnectionError, openai.InternalServerError)):
        return AiError(503, "AI_SERVICE_UNAVAILABLE", "AI 서버에 잠시 연결할 수 없습니다. 잠시 후 다시 시도해 주세요.")
    if isinstance(root, openai.APIError):
        return AiError(502, "AI_PROVIDER_ERROR", "AI 응답을 받지 못했습니다. 잠시 후 다시 시도해 주세요.")
    return None
