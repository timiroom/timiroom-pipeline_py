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
        )

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
