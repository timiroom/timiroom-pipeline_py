import logging

from phase2.state import PipelineState
from phase3.schema_validator import SchemaValidator

logger = logging.getLogger(__name__)


class ValidationService:

    def __init__(self, validator: SchemaValidator):
        self._validator = validator

    @staticmethod
    def _canonical_blockers(items: list[str]) -> list[str]:
        """QA/구조 검증 메시지의 표현 차이를 제거하고 중복을 합친다."""
        normalized: list[str] = []
        seen: set[str] = set()
        for item in items:
            value = " ".join(str(item or "").split())
            if value.startswith("QA:"):
                value = value[3:].strip()
            if value and value not in seen:
                seen.add(value)
                normalized.append(value)
        return normalized

    def validate(self, state: PipelineState) -> PipelineState:
        logger.info("=== Phase 3 검증 시작 === (retry: %d)", state.retry_count)

        result = self._validator.validate(
            feature_list=state.feature_list,
            db_schema=state.db_schema,
            api_spec=state.api_spec,
            prd_document=state.prd_document,
            feature_registry=state.feature_registry,
        )

        normalized = state.copy(
            db_schema=result.normalized_db_schema or state.db_schema,
            api_spec=result.normalized_api_spec or state.api_spec,
            prd_document=result.normalized_prd_document or state.prd_document,
            validation_error_codes=result.error_codes,
            validation_repair_targets=result.repair_targets,
        )

        qa_errors = (
            list(state.qa_db_blockers or state.qa_db_issues)
            + list(state.qa_api_blockers or state.qa_api_issues)
            + list(state.qa_prd_blockers or state.qa_prd_issues)
        )
        current_blockers = self._canonical_blockers(qa_errors + list(result.errors))
        previous_blockers = self._canonical_blockers(
            state.validation_unresolved_blockers or state.validation_blockers
        )
        resolved_now = [item for item in previous_blockers if item not in current_blockers]
        resolved_blockers = self._canonical_blockers(
            list(state.validation_resolved_blockers) + resolved_now
        )
        normalized = normalized.copy(
            validation_blockers=current_blockers,
            validation_unresolved_blockers=current_blockers,
            validation_resolved_blockers=resolved_blockers,
        )

        if state.qa_approved is False:
            qa_text = "\n".join(f"QA: {item}" for item in current_blockers)
            qa_targets = set(result.repair_targets)
            if state.qa_db_blockers or state.qa_db_issues:
                qa_targets.add("db")
            if state.qa_api_blockers or state.qa_api_issues:
                qa_targets.add("api")
            if state.qa_prd_blockers or state.qa_prd_issues:
                qa_targets.add("prd")
            logger.warning("Phase 2 QA hard gate 실패 — Phase 3 통과를 허용하지 않습니다")
            return normalized.copy(
                validated=False,
                last_validation_error=(qa_text or "Phase 2 QA 미승인"),
                validation_repair_targets=sorted(qa_targets),
                status_message="Phase 2 QA 실패 — 자동 발행 차단",
            )

        if result.success:
            logger.info("=== Phase 3 검증 통과 ===")
            return normalized.copy(
                validated=True,
                last_validation_error="",
                validation_error_codes=[],
                validation_repair_targets=[],
                validation_blockers=[],
                validation_unresolved_blockers=[],
                status_message="Phase 3 완료 — 검증 통과",
            )
        else:
            error_text = "\n".join(current_blockers)
            logger.warning("=== Phase 3 검증 실패 === 오류: %s", error_text)
            return normalized.copy(
                validated=False,
                last_validation_error=error_text,
                status_message="Phase 3 검증 실패 — 재시도 필요",
            )
