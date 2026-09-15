import logging

from phase2.state import PipelineState
from phase3.schema_validator import SchemaValidator

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
            prd_document=state.prd_document,
        )

        normalized = state.copy(
            db_schema=result.normalized_db_schema or state.db_schema,
            api_spec=result.normalized_api_spec or state.api_spec,
            prd_document=result.normalized_prd_document or state.prd_document,
            validation_error_codes=result.error_codes,
            validation_repair_targets=result.repair_targets,
        )

        if state.qa_approved is False:
            logger.info("Phase 2 QA 미승인은 Phase 3 구조 검증의 참고 신호로만 사용합니다")

        if result.success:
            logger.info("=== Phase 3 검증 통과 ===")
            return normalized.copy(
                validated=True,
                last_validation_error="",
                validation_error_codes=[],
                validation_repair_targets=[],
                status_message="Phase 3 완료 — 검증 통과",
            )
        else:
            error_text = "\n".join(result.errors)
            logger.warning("=== Phase 3 검증 실패 === 오류: %s", error_text)
            return normalized.copy(
                validated=False,
                last_validation_error=error_text,
                status_message="Phase 3 검증 실패 — 재시도 필요",
            )
