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
        timeout_seconds: float = 900.0,
        repair_timeout_seconds: float = 300.0,
    ):
        self._search = search_agent
        self._pm = pm_agent
        self._prd = prd_agent
        self._dba = dba_agent
        self._api = api_agent
        self._qa = qa_agent
        self._progress = progress_service
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds는 0보다 커야 합니다")
        if repair_timeout_seconds <= 0:
            raise ValueError("repair_timeout_seconds는 0보다 커야 합니다")
        self._timeout_seconds = timeout_seconds
        self._repair_timeout_seconds = repair_timeout_seconds

    async def run(self, initial_state: PipelineState, pipeline_id: str | None = None) -> PipelineState:
        async with asyncio.timeout(self._timeout_seconds):
            return await self._run_inner(initial_state, pipeline_id)

    async def _run_inner(self, initial_state: PipelineState, pipeline_id: str | None = None) -> PipelineState:
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

    async def repair(
        self,
        state: PipelineState,
        validation_error: str,
        pipeline_id: str | None = None,
        repair_targets: list[str] | None = None,
    ) -> PipelineState:
        """Phase 3 오류가 난 산출물만 다시 생성하고 QA를 재실행한다."""
        # Phase3 보정은 Phase2 전체 생성보다 짧게 제한한다. 기존에는 재시도마다
        # 900~1800초가 다시 적용되어 QA 장애가 수십 분 동안 누적됐다.
        async with asyncio.timeout(self._repair_timeout_seconds):
            return await self._repair_inner(state, validation_error, pipeline_id, repair_targets)

    async def _repair_inner(
        self,
        state: PipelineState,
        validation_error: str,
        pipeline_id: str | None,
        repair_targets: list[str] | None,
    ) -> PipelineState:
        targets = set(repair_targets or [])
        if not targets:
            if "DB 스키마" in validation_error:
                targets.add("db")
            if "API 스펙" in validation_error:
                targets.add("api")
            if "PRD 문서" in validation_error or "Phase 2 QA" in validation_error:
                targets.add("prd")
            if "featureList" in validation_error:
                targets.add("pm")

        if "pm" in targets or not targets:
            logger.warning("선택적 재생성 대상을 판단할 수 없어 전체 Phase 2를 재실행")
            return await self._run_inner(state, pipeline_id)

        if "prd" in targets:
            self._progress.send(pipeline_id, "PRD_REPAIR", "PRD부터 종속 산출물을 재생성 중...", 68)
            repaired = await self._run_prd_with_rollback(state, pipeline_id)
            self._progress.send(pipeline_id, "QA_REPAIR", "수정 산출물 재검수 중...", 78)
            return await self._qa.execute(repaired)

        repair_state = state
        tasks = []
        labels = []
        if "db" in targets:
            tasks.append(self._dba.execute(state))
            labels.append("DB")
        if "api" in targets:
            tasks.append(self._api.execute(state))
            labels.append("API")

        self._progress.send(
            pipeline_id,
            "PHASE2_REPAIR",
            f"검증 실패 영역만 재생성 중: {', '.join(labels)}",
            72,
        )
        results = await asyncio.gather(*tasks)
        for label, result in zip(labels, results, strict=True):
            if label == "DB":
                repair_state = repair_state.copy(
                    db_schema=result.db_schema,
                    prd_feedback_from_dba=result.prd_feedback_from_dba,
                )
            else:
                repair_state = repair_state.copy(
                    api_spec=result.api_spec,
                    prd_feedback_from_api=result.prd_feedback_from_api,
                )

        self._progress.send(pipeline_id, "QA_REPAIR", "수정 산출물 재검수 중...", 78)
        return await self._qa.execute(repair_state)

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
