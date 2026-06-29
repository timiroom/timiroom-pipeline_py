import logging

from debug_dump import PipelineDump
from phase2.agents.api_agent import ApiAgent
from phase2.agents.dba_agent import DbaAgent
from phase2.agents.pm_agent import PmAgent
from phase2.agents.prd_agent import PrdAgent
from phase2.agents.qa_agent import QaAgent
from phase2.agents.search_agent import SearchAgent
from phase2.sse_service import PipelineProgressService
from phase2.state import PipelineState

logger = logging.getLogger(__name__)

MAX_QA_RETRY = 5


class OrchestrationGraph:

    def __init__(
        self,
        search_agent: SearchAgent,
        pm_agent: PmAgent,
        prd_agent: PrdAgent,
        dba_agent: DbaAgent,
        api_agent: ApiAgent,
        qa_agent: QaAgent,
        progress_service: PipelineProgressService,
    ):
        self._search = search_agent
        self._pm = pm_agent
        self._prd = prd_agent
        self._dba = dba_agent
        self._api = api_agent
        self._qa = qa_agent
        self._progress = progress_service

    async def run(self, initial_state: PipelineState, pipeline_id: str | None = None) -> PipelineState:
        logger.info("=== Phase 2 오케스트레이션 시작 ===")
        dump = PipelineDump(pipeline_id or "no-id")
        final_state = initial_state

        try:
            self._progress.send(pipeline_id, "SEARCH", "시장 조사 중...", 30)
            after_search = await self._search.execute(initial_state, dump)
            dump.log_state("SEARCH", after_search)

            self._progress.send(pipeline_id, "PM", "기능 분석 및 설계 지시 생성 중...", 40)
            after_pm = await self._pm.execute(after_search, dump)
            dump.log_state("PM", after_pm)

            self._progress.send(pipeline_id, "PRD", "PRD 문서 작성 중...", 50)
            after_prd_dba_api = await self._run_prd_with_rollback(after_pm, pipeline_id, dump)
            dump.log_state("PRD_DBA_API", after_prd_dba_api)

            self._progress.send(pipeline_id, "QA", "QA 검수 중...", 75)
            final_state = await self._run_qa_retry(after_prd_dba_api, 0, pipeline_id, dump)
            dump.log_state("QA", final_state)

            logger.info("=== Phase 2 완료 ===")
        except Exception as e:
            logger.error("오케스트레이션 예외: %s", e)
            raise
        finally:
            dump.close(final_state)

        return final_state

    async def execute(self, user_query: str, context_prompt: str) -> PipelineState:
        import uuid
        initial = PipelineState(user_query=user_query, context_prompt=context_prompt)
        return await self.run(initial, f"retry-{uuid.uuid4().hex[:8]}")

    async def _run_prd_with_rollback(
        self, pm_state: PipelineState, pipeline_id: str | None, dump=None
    ) -> PipelineState:
        current = pm_state

        for attempt in range(3):
            logger.info("PRD 에이전트 실행 중... (시도 %d)", attempt + 1)
            after_prd = await self._prd.execute(current, dump)

            self._progress.send(pipeline_id, "DBA_API", "DB 스키마 설계 중...", 58)
            dba_result = await self._dba.execute(after_prd, dump)
            if dump:
                dump.log_state(f"DBA (attempt {attempt + 1})", dba_result)

            # API 에이전트는 DBA 결과(DB 스키마)를 받아 일관된 경로를 설계
            self._progress.send(pipeline_id, "DBA_API", "API 스펙 설계 중...", 65)
            api_input = after_prd.copy(
                context_prompt=(after_prd.context_prompt or "")
                    + "\n\n=== DBA 설계 DB 스키마 (아래 테이블명을 API 경로에 반영하세요) ===\n"
                    + (dba_result.db_schema or ""),
            )
            api_result = await self._api.execute(api_input, dump)
            if dump:
                dump.log_state(f"API (attempt {attempt + 1})", api_result)

            has_dba_feedback = bool((dba_result.prd_feedback_from_dba or "").strip())
            has_api_feedback = bool((api_result.prd_feedback_from_api or "").strip())

            if not has_dba_feedback and not has_api_feedback:
                logger.info("PRD ↔ DBA/API 검증 통과 (시도 %d)", attempt + 1)
                return after_prd.copy(
                    db_schema=dba_result.db_schema,
                    api_spec=api_result.api_spec,
                )

            if attempt < 2:
                logger.warning("PRD rollback #%d", attempt + 1)
                self._progress.send(
                    pipeline_id, "PRD_ROLLBACK",
                    f"PRD 재작성 중... ({attempt + 1}/2회)", 52,
                )
                current = after_prd.copy(
                    prd_feedback_from_dba=dba_result.prd_feedback_from_dba,
                    prd_feedback_from_api=api_result.prd_feedback_from_api,
                    rollback_count=attempt + 1,
                )
            else:
                logger.warning("PRD rollback 최대 횟수 초과 — 현재 결과로 진행")
                return after_prd.copy(
                    db_schema=dba_result.db_schema,
                    api_spec=api_result.api_spec,
                )

        return current

    async def _run_qa_retry(
        self, state: PipelineState, attempt: int, pipeline_id: str | None, dump=None
    ) -> PipelineState:
        if attempt > 0:
            self._progress.send(
                pipeline_id, "QA_RETRY",
                f"QA 재검수 중... ({attempt}/{MAX_QA_RETRY}회)", 76,
            )

        logger.info("QA 에이전트 실행 중... (시도 %d)", attempt + 1)
        qa_result = await self._qa.execute(state, dump)
        if dump:
            dump.log_state(f"QA (attempt {attempt + 1})", qa_result)

        qa_failed = bool((qa_result.last_validation_error or "").strip())

        if not qa_failed:
            logger.info("QA 검수 통과 (시도 %d)", attempt + 1)
            return qa_result

        if attempt >= MAX_QA_RETRY:
            logger.warning("QA 최대 재시도 초과 (%d) — Phase 3로 위임", MAX_QA_RETRY)
            return qa_result.copy(
                last_validation_error="",
                status_message="QA 최대 재시도 초과 — Phase 3 형식 검증으로 위임",
            )

        logger.warning("QA 검수 실패 — 재시도 (%d/%d)", attempt + 1, MAX_QA_RETRY)

        retry_context = (
            state.context_prompt
            + "\n\n=== 이전 설계의 치명적 결함 (반드시 수정) ===\n"
            + qa_result.last_validation_error
            + "\n\n위 결함을 모두 수정하여 완전한 설계를 다시 생성하세요."
        )

        retry_base = state.copy(
            context_prompt=retry_context,
            last_validation_error="",
            retry_count=attempt + 1,
        )

        dba_result = await self._dba.execute(retry_base, dump)
        api_retry_input = retry_base.copy(
            context_prompt=retry_context
                + "\n\n=== DBA 설계 DB 스키마 ===\n"
                + (dba_result.db_schema or ""),
        )
        api_result = await self._api.execute(api_retry_input, dump)

        merged = retry_base.copy(
            db_schema=dba_result.db_schema,
            api_spec=api_result.api_spec,
            prd_feedback_from_dba=dba_result.prd_feedback_from_dba,
            prd_feedback_from_api=api_result.prd_feedback_from_api,
            retry_count=attempt + 1,
        )

        return await self._run_qa_retry(merged, attempt + 1, pipeline_id, dump)
