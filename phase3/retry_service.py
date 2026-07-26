import logging

from phase2.state import PipelineState

logger = logging.getLogger(__name__)


class RetryService:

    def __init__(self, max_retry: int = 3):
        self._max_retry = max_retry

    async def retry_with(
        self,
        state: PipelineState,
        orchestration_graph,
        validation_service,
        pipeline_id: str | None = None,
    ) -> PipelineState:
        current_retry = state.retry_count + 1

        if current_retry > self._max_retry:
            logger.warning("최대 재시도 횟수 초과 (%d/%d)", current_retry, self._max_retry)
            return state.copy(status_message="Human-in-the-Loop 필요 — 관리자 검토 요청")

        logger.info("Phase 3 재시도 시작 (%d/%d)", current_retry, self._max_retry)

        retry_prompt = (
            (state.context_prompt or "")
            + "\n\n[이전 시도 실패 원인]\n"
            + (state.last_validation_error or "")
            + "\n\n위 오류를 반드시 수정해서 다시 생성해주세요."
        )

        # feature_list/project_name/platform 등 기존 state를 그대로 보존하고
        # context_prompt만 재시도 피드백으로 교체 — 문자열만으로 새 state를 만들면
        # Phase1/2에서 이미 구축된 정보가 전부 소실된다.
        retry_state = state.copy(context_prompt=retry_prompt, retry_count=current_retry)
        result = await orchestration_graph.run(retry_state, pipeline_id)

        validated = validation_service.validate(result)

        if validated.validated:
            logger.info("Phase 3 재시도 성공 (%d/%d)", current_retry, self._max_retry)
            return validated

        return await self.retry_with(validated, orchestration_graph, validation_service, pipeline_id)
