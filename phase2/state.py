from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any


@dataclass
class PipelineState:
    # Phase 1 폼 데이터
    session_id: str = ""
    project_name: str = ""
    platform: Any = None
    tech_stack: list[str] = field(default_factory=list)
    problem_definition: Any = None
    target_users: list[Any] = field(default_factory=list)

    must_features: list[str] = field(default_factory=list)
    should_features: list[str] = field(default_factory=list)
    could_features: list[str] = field(default_factory=list)
    excluded_features: list[str] = field(default_factory=list)

    # Phase 1 결과
    user_query: str = ""
    context_prompt: str = ""

    # PM 에이전트 결과
    feature_list: list[str] = field(default_factory=list)
    dba_instruction: str = ""
    api_instruction: str = ""

    # 병렬 에이전트 결과
    db_schema: str = ""
    api_spec: str = ""

    # Phase 3 검증
    retry_count: int = 0
    prd_document: str = ""
    last_validation_error: str = ""
    validated: bool = False

    # Search 에이전트
    market_research: str = ""

    # PRD rollback
    prd_feedback_from_dba: str = ""
    prd_feedback_from_api: str = ""
    rollback_count: int = 0

    # 상태 메시지
    status_message: str = ""

    def copy(self, **kwargs) -> PipelineState:
        return replace(self, **kwargs)
