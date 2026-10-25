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
    supporting_feature_names: list[str] = field(default_factory=list)
    should_features: list[str] = field(default_factory=list)
    could_features: list[str] = field(default_factory=list)
    excluded_features: list[str] = field(default_factory=list)

    # Phase 1 결과
    user_query: str = ""
    context_prompt: str = ""
    rag_chunks_retrieved: int = 0
    rag_context_fallback: bool = False
    pdf_files_total: int = 0
    pdf_files_processed: int = 0
    pdf_files_failed: int = 0

    # PM 에이전트 결과
    feature_list: list[str] = field(default_factory=list)
    feature_specs: list[dict[str, Any]] = field(default_factory=list)
    feature_registry: list[dict[str, Any]] = field(default_factory=list)
    feature_registry_hash: str = ""
    project_plan: dict[str, Any] = field(default_factory=dict)
    feature_spec_document: str = ""
    dba_instruction: str = ""
    api_instruction: str = ""

    # 병렬 에이전트 결과
    db_schema: str = ""
    api_spec: str = ""
    openapi_spec: str = ""

    # Phase 3 검증
    retry_count: int = 0
    prd_document: str = ""
    last_validation_error: str = ""
    validated: bool = False

    # QA 에이전트 결과 (선택적 재시도용 카테고리별 결함 + 품질 점수)
    qa_quality_score: float = 0.0
    qa_approved: bool | None = None
    qa_validation_revision: str = ""
    qa_db_issues: list[str] = field(default_factory=list)
    qa_api_issues: list[str] = field(default_factory=list)
    qa_prd_issues: list[str] = field(default_factory=list)
    qa_db_blockers: list[str] = field(default_factory=list)
    qa_api_blockers: list[str] = field(default_factory=list)
    qa_prd_blockers: list[str] = field(default_factory=list)
    qa_db_warnings: list[str] = field(default_factory=list)
    qa_api_warnings: list[str] = field(default_factory=list)
    qa_prd_warnings: list[str] = field(default_factory=list)
    qa_issue_details: list[dict[str, Any]] = field(default_factory=list)
    qa_repair_issues: list[dict[str, Any]] = field(default_factory=list)
    qa_blocker_details: list[dict[str, Any]] = field(default_factory=list)
    # Upstream/LLM 생성 실패를 품질 결함과 분리해 Phase3까지 보존한다.
    generation_blockers: list[str] = field(default_factory=list)
    generation_diagnostics: list[dict[str, Any]] = field(default_factory=list)
    artifact_quality: dict[str, Any] = field(default_factory=dict)
    failed_stage: str = ""
    last_completed_stage: str = ""
    skipped_stages: list[str] = field(default_factory=list)

    # Search 에이전트
    market_research: str = ""
    market_research_status: list[dict[str, Any]] = field(default_factory=list)

    # PRD rollback
    prd_feedback_from_dba: str = ""
    prd_feedback_from_api: str = ""
    rollback_count: int = 0

    # 상태 메시지
    status_message: str = ""

    # Phase 3 구조화 검증·재시도 상태
    validation_error_codes: list[str] = field(default_factory=list)
    validation_repair_targets: list[str] = field(default_factory=list)
    validation_failure_fingerprints: list[str] = field(default_factory=list)
    # Phase3 targeted repair budget. Keys are pm/prd/db/api and each domain is
    # allowed one automatic repair per pipeline.
    targeted_repair_attempts: dict[str, int] = field(default_factory=dict)
    # Repair agents report whether a requested delta was actually applied.
    # Rejected contract-external patches must not consume the domain budget.
    targeted_repair_results: dict[str, str] = field(default_factory=dict)
    # Phase 3 blocker bundle: repair 전후에 어떤 결함이 남고 해결됐는지 추적한다.
    validation_blockers: list[str] = field(default_factory=list)
    validation_resolved_blockers: list[str] = field(default_factory=list)
    validation_unresolved_blockers: list[str] = field(default_factory=list)

    def copy(self, **kwargs) -> PipelineState:
        return replace(self, **kwargs)
