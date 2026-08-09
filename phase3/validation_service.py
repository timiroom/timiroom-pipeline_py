import logging

from phase2.state import PipelineState
from phase3.schema_validator import SchemaValidator
from phase2.agent_contract import IssueSeverity, classify_issue

logger = logging.getLogger(__name__)


class ValidationService:

    def __init__(self, validator: SchemaValidator):
        self._validator = validator

    def validate(self, state: PipelineState) -> PipelineState:
        logger.info("=== Phase 3 검증 시작 === (retry: %d)", state.retry_count)

        result = self._validator.validate(
            feature_list=state.feature_list,
            db_schema=state.db_schema,
            api_spec=state.api_spec,
            feature_specs=state.feature_specs,
        )

        details = list(state.qa_issue_details or [])
        if not details:
            legacy = list(state.qa_db_issues or []) + list(state.qa_api_issues or []) + list(state.qa_prd_issues or [])
            details = [{"reason": message, "severity": classify_issue(message).value} for message in legacy]
        blocking_issues = [
            item for item in details
            if str(item.get("severity")) in {
                IssueSeverity.BLOCKER.value,
                IssueSeverity.ERROR.value,
            }
        ]
        if blocking_issues:
            result.errors.extend(
                f"QA {item.get('severity')}: {item.get('reason')}"
                for item in blocking_issues
            )
            result.success = False
        non_blocking = len(details) - len(blocking_issues)
        if non_blocking:
            logger.warning("Phase 3 비차단 QA 이슈 %d건은 결과에 경고로 유지", non_blocking)

        if result.success:
            logger.info("=== Phase 3 검증 통과 ===")
            return state.copy(validated=True, status_message="Phase 3 완료 — 검증 통과")
        else:
            logger.warning("=== Phase 3 검증 실패 === 오류: %s", result.errors_to_string())
            return state.copy(
                validated=False,
                last_validation_error=result.errors_to_string(),
                status_message="Phase 3 검증 실패 — 재시도 필요",
            )
