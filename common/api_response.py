from __future__ import annotations

import time
from enum import Enum
from typing import Any

from pydantic import BaseModel


class ErrorCode(str, Enum):
    # 공통
    INVALID_INPUT = "COMMON_001"
    INTERNAL_ERROR = "COMMON_002"
    # RAG
    RAG_CONTEXT_FAILED = "RAG_001"
    DOCUMENT_INGEST_FAILED = "RAG_002"
    EMBEDDING_FAILED = "RAG_003"
    # 파이프라인
    PIPELINE_TIMEOUT = "PIPELINE_001"
    PIPELINE_VALIDATION_FAILED = "PIPELINE_002"
    PIPELINE_HUMAN_REVIEW = "PIPELINE_003"
    PIPELINE_QA_FAILED = "PIPELINE_004"
    # Kafka
    KAFKA_PUBLISH_FAILED = "KAFKA_001"


_MESSAGES: dict[ErrorCode, str] = {
    ErrorCode.INVALID_INPUT: "입력값이 올바르지 않습니다",
    ErrorCode.INTERNAL_ERROR: "서버 내부 오류가 발생했습니다",
    ErrorCode.RAG_CONTEXT_FAILED: "RAG 컨텍스트 생성에 실패했습니다",
    ErrorCode.DOCUMENT_INGEST_FAILED: "문서 저장에 실패했습니다",
    ErrorCode.EMBEDDING_FAILED: "임베딩 생성에 실패했습니다",
    ErrorCode.PIPELINE_TIMEOUT: "GPT 응답 시간이 초과되었습니다",
    ErrorCode.PIPELINE_VALIDATION_FAILED: "결과물 검증에 실패했습니다",
    ErrorCode.PIPELINE_HUMAN_REVIEW: "자동 검증 실패 — 관리자 검토가 필요합니다",
    ErrorCode.PIPELINE_QA_FAILED: "QA 에이전트 검수에 실패했습니다",
    ErrorCode.KAFKA_PUBLISH_FAILED: "Kafka 메시지 발행에 실패했습니다",
}


def error_message(code: ErrorCode) -> str:
    return _MESSAGES.get(code, "알 수 없는 오류")


# ── 응답 모델 ──────────────────────────────────────────────────────

def ok(data: Any, message: str | None = None) -> dict:
    """성공 응답: {"success": true, "code": "SUCCESS", "data": ...}"""
    result: dict = {"success": True, "code": "SUCCESS"}
    if message is not None:
        result["message"] = message
    result["data"] = data
    return result


def error(code: ErrorCode, message: str | None = None) -> dict:
    """에러 응답: {"code": "...", "message": "...", "timestamp": "..."}"""
    return {
        "code": code.value,
        "message": message or error_message(code),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
