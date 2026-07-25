import asyncio
import logging

from phase2.agents.api_agent import ApiAgent
from phase2.agents.dba_agent import DbaAgent
from phase2.agents.pm_agent import PmAgent
from phase2.agents.prd_agent import PrdAgent
from phase2.agents.qa_agent import QaAgent
from phase2.agents.search_agent import SearchAgent
from phase2.sse_service import PipelineProgressService
from phase2.state import PipelineState

logger = logging.getLogger(__name__)


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
        dump = None  # 디버그 덤프 비활성화 — 필요 시 debug_dump.PipelineDump 구현체를 연결
        final_state = initial_state

        try:
            self._progress.send(pipeline_id, "SEARCH", "시장 조사 중...", 30)
            after_search = await self._search.execute(initial_state, dump)
            if dump:
                dump.log_state("SEARCH", after_search)

            self._progress.send(pipeline_id, "PM", "기능 분석 및 설계 지시 생성 중...", 40)
            after_pm = await self._pm.execute(after_search, dump)
            if dump:
                dump.log_state("PM", after_pm)

            self._progress.send(pipeline_id, "PRD", "PRD 문서 작성 중...", 50)
            after_prd_dba_api = await self._run_prd_with_rollback(after_pm, pipeline_id, dump)
            if dump:
                dump.log_state("PRD_DBA_API", after_prd_dba_api)

            self._progress.send(pipeline_id, "QA", "QA 검수·수정 중...", 75)
            final_state = await self._qa.execute(after_prd_dba_api, dump)
            if dump:
                dump.log_state("QA", final_state)

            logger.info("=== Phase 2 완료 ===")
        except Exception as e:
            logger.error("오케스트레이션 예외: %s", e)
            raise
        finally:
            if dump:
                dump.close(final_state)

        return final_state

    async def _run_prd_with_rollback(
        self, pm_state: PipelineState, pipeline_id: str | None, dump=None
    ) -> PipelineState:
        current = pm_state

        for attempt in range(3):
            logger.info("PRD 에이전트 실행 중... (시도 %d)", attempt + 1)
            after_prd = await self._prd.execute(current, dump)

            # DBA·API는 rag-pipeline과 동일하게 독립적으로 병렬 실행 (교차 주입 없음)
            self._progress.send(pipeline_id, "DBA_API", "DB 스키마 · API 설계 중...", 60)
            dba_result, api_result = await asyncio.gather(
                self._dba.execute(after_prd, dump),
                self._api.execute(after_prd, dump),
            )
            if dump:
                dump.log_state(f"DBA (attempt {attempt + 1})", dba_result)
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
                # DBA_API(60%) 이후이므로 62→64로 항상 순방향 증가
                rollback_pct = 62 + attempt * 2
                self._progress.send(
                    pipeline_id, "PRD_ROLLBACK",
                    f"PRD 재작성 중... ({attempt + 1}/2회)", rollback_pct,
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
