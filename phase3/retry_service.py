import logging

from phase2.state import PipelineState

logger = logging.getLogger(__name__)


class RetryService:

    def __init__(self, max_retry: int = 3):
        self._max_retry = max_retry

    async def retry(self, state: PipelineState) -> PipelineState:
        from phase2.orchestration_graph import OrchestrationGraph
        from phase3.validation_service import ValidationService
        raise RuntimeError(
            "RetryService.retry()는 orchestration_graph와 validation_service 인스턴스가 필요합니다. "
            "OrchestrationController.run_pipeline()에서 직접 호출하세요."
        )

    async def retry_with(
        self,
        state: PipelineState,
        orchestration_graph,
        validation_service,
    ) -> PipelineState:
        current_retry = state.retry_count + 1

        if current_retry > self._max_retry:
            logger.warning("최대 재시도 횟수 초과 (%d/%d)", current_retry, self._max_retry)
            return state.copy(status_message="Human-in-the-Loop 필요 — 관리자 검토 요청")

        logger.info("재시도 시작 (%d/%d)", current_retry, self._max_retry)

        retry_prompt = (
            (state.context_prompt or "")
            + "\n\n[이전 시도 실패 원인]\n"
            + (state.last_validation_error or "")
            + "\n\n위 오류를 반드시 수정해서 다시 생성해주세요."
        )

        retry_state = await orchestration_graph.execute(state.user_query, retry_prompt)
        retry_state = retry_state.copy(retry_count=current_retry)

        validated = validation_service.validate(retry_state)

        if validated.validated:
            logger.info("재시도 성공 (%d/%d)", current_retry, self._max_retry)
            return validated

        return await self.retry_with(validated, orchestration_graph, validation_service)
