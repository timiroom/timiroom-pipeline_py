import asyncio
import copy
import hashlib
import json
import logging

from common.ai_errors import describe
from phase2.agents.api_agent import (
    ApiAgent,
    _api_feature_mappings,
    finalize_api_contracts,
)
from phase2.agents.dba_agent import DbaAgent
from phase2.agents.pm_agent import PmAgent
from phase2.agents.prd_agent import PrdAgent, sync_core_features_with_registry
from phase2.agents.qa_agent import QaAgent
from phase2.agents.search_agent import SearchAgent
from phase2.artifact_shapes import shape_issues
from phase2.execution_error import PipelineStageError, mark_stage, stage_snapshot
from phase2.json_utils import try_parse_json
from phase2.openapi_spec import build_openapi_document
from phase2.repair_budget import available_targets, finish_attempt, merge_attempts
from phase2.sse_service import PipelineProgressService
from phase2.state import PipelineState

logger = logging.getLogger(__name__)


class OrchestrationGraph:
    manages_repair_budget = True

    def __init__(
        self,
        search_agent: SearchAgent,
        pm_agent: PmAgent,
        prd_agent: PrdAgent,
        dba_agent: DbaAgent,
        api_agent: ApiAgent,
        qa_agent: QaAgent,
        progress_service: PipelineProgressService,
        feature_spec_agent=None,
        timeout_seconds: float = 900.0,
        repair_timeout_seconds: float = 300.0,
        resync_timeout_seconds: float = 120.0,
        dba_resync_timeout_seconds: float | None = None,
        api_resync_timeout_seconds: float | None = None,
        max_targeted_repair_per_domain: int = 1,
    ):
        self._search = search_agent
        self._pm = pm_agent
        self._prd = prd_agent
        self._dba = dba_agent
        self._api = api_agent
        self._qa = qa_agent
        self._feature_spec = feature_spec_agent
        self._progress = progress_service
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds는 0보다 커야 합니다")
        if repair_timeout_seconds <= 0:
            raise ValueError("repair_timeout_seconds는 0보다 커야 합니다")
        if resync_timeout_seconds <= 0:
            raise ValueError("resync_timeout_seconds는 0보다 커야 합니다")
        if max_targeted_repair_per_domain <= 0:
            raise ValueError("max_targeted_repair_per_domain은 1 이상이어야 합니다")
        # 복구를 실제로 수행하는 쪽이 예산을 몰라 상수 1을 쓰면, RetryService가
        # 더 큰 예산으로 재시도를 보내도 _bounded_repair가 budget_exhausted로
        # 아무 일도 하지 않고 끝난다. 설정된 예산이 조용히 무시된다.
        self._max_targeted_repair_per_domain = max_targeted_repair_per_domain
        self._timeout_seconds = timeout_seconds
        self._repair_timeout_seconds = repair_timeout_seconds
        self._dba_resync_timeout_seconds = dba_resync_timeout_seconds or resync_timeout_seconds
        self._api_resync_timeout_seconds = api_resync_timeout_seconds or resync_timeout_seconds

    async def run(self, initial_state: PipelineState, pipeline_id: str | None = None) -> PipelineState:
        async with asyncio.timeout(self._timeout_seconds):
            return await self._run_inner(initial_state, pipeline_id)

    @staticmethod
    async def _stage_execute(stage: str, agent, state: PipelineState, dump=None) -> PipelineState:
        """Run one stage, keeping the artifacts built before it attached to any failure.

        Without this the route only knows the state from before Phase 2, so a
        provider error in FEATURE_SPEC or QA threw away a finished PRD and
        project plan that the caller could still inspect or resume from.
        """
        try:
            return await agent.execute(state, dump)
        except asyncio.CancelledError as exc:
            # asyncio.timeout re-raises this as TimeoutError; the snapshot has to
            # ride the cause chain because we must not swallow the cancellation.
            mark_stage(exc, stage, state)
            raise
        except Exception as exc:
            raise PipelineStageError(stage, state) from exc

    async def _run_inner(self, initial_state: PipelineState, pipeline_id: str | None = None) -> PipelineState:
        logger.info("=== Phase 2 오케스트레이션 시작 ===")
        dump = None  # 디버그 덤프 비활성화 — 필요 시 debug_dump.PipelineDump 구현체를 연결
        final_state = initial_state

        try:
            self._progress.send(pipeline_id, "SEARCH", "시장 조사 중...", 30)
            after_search = await self._stage_execute("SEARCH", self._search, initial_state, dump)
            after_search = after_search.copy(last_completed_stage="SEARCH")
            if dump:
                dump.log_state("SEARCH", after_search)

            self._progress.send(pipeline_id, "PM", "프로젝트 목표·범위·우선순위 정리 중...", 40)
            after_pm = await self._stage_execute("PM", self._pm, after_search, dump)
            if after_pm.generation_blockers:
                return after_pm
            after_pm = after_pm.copy(last_completed_stage="PM")
            if dump:
                dump.log_state("PM", after_pm)

            self._progress.send(pipeline_id, "PRD", "프로젝트 계획 기반 PRD 문서 작성 중...", 50)
            after_prd = await self._stage_execute("PRD", self._prd, after_pm, dump)
            if after_prd.generation_blockers:
                return after_prd
            after_prd = after_prd.copy(last_completed_stage="PRD")

            after_prd_dba_api = await self._run_prd_with_rollback(
                after_prd, pipeline_id, dump, prd_ready=True,
            )
            if after_prd_dba_api.generation_blockers:
                return after_prd_dba_api
            after_prd_dba_api = self._align_api_to_final_db(after_prd_dba_api)
            if dump:
                dump.log_state("PRD_DBA_API", after_prd_dba_api)

            self._progress.send(pipeline_id, "QA", "QA 검수·수정 중...", 75)
            final_state = await self._stage_execute("QA", self._qa, after_prd_dba_api, dump)
            if final_state.qa_repair_issues:
                final_state = await self._repair_once(final_state, pipeline_id, dump)
            if dump:
                dump.log_state("QA", final_state)

            logger.info("=== Phase 2 완료 ===")
            final_state = final_state.copy(last_completed_stage="QA")
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
        targets = {target for target in (repair_targets or []) if target in {"pm", "prd", "db", "api"}}
        if not targets:
            if "DB 스키마" in validation_error:
                targets.add("db")
            if "API 스펙" in validation_error:
                targets.add("api")
            if "PRD 문서" in validation_error or "Phase 2 QA" in validation_error:
                targets.add("prd")
            if "featureList" in validation_error:
                targets.add("pm")

        if "pm" in targets:
            self._progress.send(pipeline_id, "PM_REPAIR", "기능 목록을 재정리 중...", 64)
            after_pm = await self._bounded_repair("pm", self._pm, state, validation_error)
            self._progress.send(pipeline_id, "PRD_REPAIR", "변경된 기능 목록 기준 산출물을 재생성 중...", 68)
            after_prd = await self._bounded_repair("prd", self._prd, after_pm, validation_error)
            repaired, _ = await self._resync_prd(after_pm, after_prd, validation_error, pipeline_id,
                                                domains=targets & {"db", "api"})
            self._progress.send(pipeline_id, "QA_REPAIR", "수정 산출물 재검수 중...", 78)
            return await self._qa.execute(self._align_api_to_final_db(repaired))

        if not targets:
            logger.warning("선택적 재생성 대상을 판단할 수 없어 QA 정규화 후 Phase 3로 반환")
            return await self._qa.execute(state)

        if "prd" in targets:
            self._progress.send(pipeline_id, "PRD_REPAIR", "지적된 PRD 섹션만 수정 중...", 68)
            repair = getattr(self._prd, "repair", None)
            if callable(repair):
                before = state
                state = await self._bounded_repair("prd", self._prd, state, validation_error)
                state, _ = await self._resync_prd(
                    before, state, validation_error, pipeline_id,
                    domains=targets & {"db", "api"},
                )
                targets.clear()
            else:
                state = await self._repair_agent(self._prd, state, validation_error)
                targets.clear()

        repair_state = state
        tasks = []
        labels = []
        if "db" in targets:
            tasks.append(self._bounded_repair("db", self._dba, state, validation_error))
            labels.append("DB")
        if "api" in targets:
            tasks.append(self._bounded_repair("api", self._api, state, validation_error))
            labels.append("API")

        # PRD만 복구된 경우 targets가 비어 재생성할 산출물이 없다. 빈 라벨로
        # PHASE2_REPAIR를 보내면 화면이 DB·API 설계 단계로 되돌아간 것처럼 보인다.
        if labels:
            self._progress.send(
                pipeline_id,
                "PHASE2_REPAIR",
                f"검증 실패 영역만 재생성 중: {', '.join(labels)}",
                72,
            )
        results = await asyncio.gather(*tasks)
        for label, result in zip(labels, results, strict=True):
            domain = "db" if label == "DB" else "api"
            repair_state = repair_state.copy(targeted_repair_attempts=merge_attempts(repair_state.targeted_repair_attempts, result.targeted_repair_attempts))
            if label == "DB":
                repair_state = repair_state.copy(
                    db_schema=result.db_schema,
                    prd_feedback_from_dba=result.prd_feedback_from_dba,
                    targeted_repair_results={
                        **repair_state.targeted_repair_results,
                        domain: result.targeted_repair_results.get(domain, "noop"),
                    },
                    generation_blockers=list(dict.fromkeys(
                        [*repair_state.generation_blockers, *result.generation_blockers]
                    )),
                    qa_db_blockers=list(dict.fromkeys(
                        result.qa_db_blockers
                    )),
                )
            else:
                repair_state = repair_state.copy(
                    api_spec=result.api_spec,
                    prd_feedback_from_api=result.prd_feedback_from_api,
                    targeted_repair_results={
                        **repair_state.targeted_repair_results,
                        domain: result.targeted_repair_results.get(domain, "noop"),
                    },
                    generation_blockers=list(dict.fromkeys(
                        [*repair_state.generation_blockers, *result.generation_blockers]
                    )),
                    qa_api_blockers=list(dict.fromkeys(
                        result.qa_api_blockers
                    )),
                )

        self._progress.send(pipeline_id, "QA_REPAIR", "수정 산출물 재검수 중...", 78)
        return await self._qa.execute(self._align_api_to_final_db(repair_state))

    @staticmethod
    def _align_api_to_final_db(state: PipelineState) -> PipelineState:
        """Reconcile API contracts against the final DB schema.

        Runs after the parallel DBA/API drafts and again after every repair that
        can move either artifact. A db-only repair otherwise leaves
        featureMappings and request/response schemas bound to the previous
        tables, which QA and Phase 3 then report as a contract blocker.
        """
        api = try_parse_json(state.api_spec)
        db = try_parse_json(state.db_schema)
        if not isinstance(api, dict) or not isinstance(db, dict):
            return state
        if shape_issues("api", api) or shape_issues("db", db):
            return state
        if not isinstance(api.get("endpoints"), list) or not db.get("tables"):
            return state
        api["endpoints"] = finalize_api_contracts(
            api["endpoints"], state.db_schema, state.prd_document,
            state.feature_list, state.feature_registry, state.context_prompt,
        )
        features = [
            str(item["name"]) for item in state.feature_specs or state.feature_registry
            if isinstance(item, dict) and item.get("name")
        ] or state.feature_list
        api["featureMappings"] = _api_feature_mappings(
            api["endpoints"], state.db_schema, features, state.feature_registry,
        )
        api_spec = json.dumps(api, ensure_ascii=False)
        return state.copy(
            api_spec=api_spec,
            openapi_spec=json.dumps(
                build_openapi_document(api, state.project_name), ensure_ascii=False,
            ),
        )

    async def _repair_once(self, state: PipelineState, pipeline_id: str | None, dump=None) -> PipelineState:
        """Repair only QA findings, with API observing the repaired PRD and DB."""
        if state.generation_blockers:
            return state
        grouped = {"DBA": [], "PRD": [], "API": []}
        for issue in state.qa_repair_issues:
            if isinstance(issue, dict) and str(issue.get("agent", "")).upper() in grouped:
                grouped[str(issue["agent"]).upper()].append(issue)
        async with asyncio.timeout(self._repair_timeout_seconds):
            current = state
            handled = set()
            for label, agent in (("PRD", self._prd), ("DBA", self._dba), ("API", self._api)):
                issues = grouped[label]
                if label == "API" and grouped["DBA"] and not issues:
                    issues = [{"agent": "API", "target": "erd-contract", "reason": "DB 계약 변경 반영"}]
                if not issues:
                    continue
                domain = "db" if label == "DBA" else label.lower()
                if domain in handled:
                    continue
                if current.targeted_repair_attempts.get(domain, 0) >= 1:
                    continue
                self._progress.send(pipeline_id, f"{label}_PATCH", f"{label} 검증 실패 항목 수정 중...", 80)
                before = current
                feedback = json.dumps(issues, ensure_ascii=False)
                current = await self._bounded_repair(domain, agent, current, feedback)
                if label == "PRD":
                    current, handled = await self._resync_prd(before, current, feedback, pipeline_id, dump)
            self._progress.send(pipeline_id, "QA_RECHECK", "수정된 문서 관계 재검증 중...", 88)
            return await self._qa.execute(self._align_api_to_final_db(current), dump)

    @staticmethod
    async def _repair_agent(agent, state: PipelineState, feedback: str) -> PipelineState:
        """Never replace targeted repair with full generation."""
        repair = getattr(agent, "repair", None)
        if callable(repair):
            return await repair(state, feedback)
        blocker = "TARGETED_REPAIR_BLOCKER: agent does not support patch; full generation blocked"
        return state.copy(generation_blockers=[*state.generation_blockers, blocker], qa_approved=False)

    async def _bounded_repair(self, domain, agent, state, feedback):
        if not available_targets(state, {domain}, self._max_targeted_repair_per_domain):
            return state.copy(targeted_repair_results={**state.targeted_repair_results, domain: "budget_exhausted"})
        if not callable(getattr(agent, "repair", None)):
            result = await self._repair_agent(agent, state, feedback)
            return result.copy(targeted_repair_results={**result.targeted_repair_results, domain: "unsupported"})
        try:
            timeout = self._dba_resync_timeout_seconds if domain == "db" else self._api_resync_timeout_seconds if domain == "api" else self._repair_timeout_seconds
            dispatch = state.copy(targeted_repair_results={key: value for key, value in state.targeted_repair_results.items() if key != domain})
            async with asyncio.timeout(timeout):
                result = await self._repair_agent(agent, dispatch, feedback)
        except Exception as exc:
            logger.exception("Targeted repair failed: %s", domain)
            ai = describe(exc)
            reason = ai.message if ai else type(exc).__name__
            blocker = f"TARGETED_REPAIR_BLOCKER: {domain} ({reason})"
            result = state.copy(targeted_repair_results={**state.targeted_repair_results, domain: "failed"},
                                generation_blockers=[*state.generation_blockers, blocker], qa_approved=False)
            if domain in {"api", "db", "prd"}:
                result = result.copy(**{f"qa_{domain}_blockers": [*getattr(state, f"qa_{domain}_blockers"), blocker]})
        return finish_attempt(state, result, domain)

    @staticmethod
    def _merge_agent_quality(base: PipelineState, dba: PipelineState, api: PipelineState) -> PipelineState:
        generation_blockers = list(dict.fromkeys(
            [*base.generation_blockers, *dba.generation_blockers, *api.generation_blockers]
        ))
        merged = base.copy(
            generation_blockers=generation_blockers,
            qa_db_blockers=list(dict.fromkeys([*base.qa_db_blockers, *dba.qa_db_blockers])),
            qa_api_blockers=list(dict.fromkeys([*base.qa_api_blockers, *api.qa_api_blockers])),
            qa_approved=False if generation_blockers or dba.qa_db_blockers or api.qa_api_blockers else base.qa_approved,
        )
        if not generation_blockers:
            return merged
        # A designer that could not generate stopped this stage. Say so, and keep
        # its provider diagnostic, so Phase3 reports one cause instead of reading
        # the empty artifact as an independent defect.
        diagnostics = list(dict.fromkeys(
            json.dumps(item, ensure_ascii=False, sort_keys=True)
            for item in [*base.generation_diagnostics, *dba.generation_diagnostics, *api.generation_diagnostics]
        ))
        return stage_snapshot("DBA_API", merged).copy(
            generation_diagnostics=[json.loads(item) for item in diagnostics],
            last_completed_stage=base.last_completed_stage,
        )

    async def _run_db_and_api(self, after_prd: PipelineState, dump=None):
        """Run both designers and keep whichever one finished if the other fails.

        A plain gather cancels the sibling and raises, so a provider error on one
        side discarded a complete schema or API spec the other side had already
        produced — and the route then reported both as missing.
        """
        dba_task = asyncio.ensure_future(self._dba.execute(self._clone_frozen_registry(after_prd), dump))
        api_task = asyncio.ensure_future(self._api.execute(self._clone_frozen_registry(after_prd), dump))
        try:
            results = await asyncio.gather(dba_task, api_task, return_exceptions=True)
        except asyncio.CancelledError as exc:
            for task in (dba_task, api_task):
                task.cancel()
            mark_stage(exc, "DBA_API", self._finished_siblings(after_prd, dba_task, api_task))
            raise
        failures = [item for item in results if isinstance(item, BaseException)]
        if failures:
            raise PipelineStageError(
                "DBA_API", self._finished_siblings(after_prd, dba_task, api_task)
            ) from failures[0]
        return results

    @staticmethod
    def _finished_siblings(after_prd: PipelineState, dba_task, api_task) -> PipelineState:
        partial = after_prd
        for task, field in ((dba_task, "db_schema"), (api_task, "api_spec")):
            if task.done() and not task.cancelled() and task.exception() is None:
                partial = partial.copy(**{field: getattr(task.result(), field)})
        return partial

    async def _run_prd_with_rollback(
        self, pm_state: PipelineState, pipeline_id: str | None, dump=None, *, prd_ready: bool = False
    ) -> PipelineState:
        current = pm_state

        # PRD repair는 최대 한 번만 수행한다. 동일 blocker는 Phase3 blocker로 남긴다.
        for attempt in range(2):
            logger.info("PRD/Feature Spec 계약 처리 (PRD ready=%s, 시도 %d)", prd_ready, attempt + 1)
            after_prd = current if prd_ready and attempt == 0 else await self._stage_execute("PRD", self._prd, current, dump)
            if self._feature_spec is not None:
                self._progress.send(pipeline_id, "FEATURE_SPEC", "기능명세서 및 지원 기능 확정 중...", 55)
                after_prd = sync_core_features_with_registry(
                    await self._stage_execute("FEATURE_SPEC", self._feature_spec, after_prd, dump)
                )
            if after_prd.generation_blockers:
                return after_prd
            after_prd = after_prd.copy(last_completed_stage="FEATURE_SPEC")
            after_prd = self._freeze_feature_registry(after_prd)

            # DBA·API는 rag-pipeline과 동일하게 독립적으로 병렬 실행 (교차 주입 없음)
            self._progress.send(pipeline_id, "DBA_API", "DB 스키마 · API 설계 중...", 60)
            dba_result, api_result = await self._run_db_and_api(after_prd, dump)
            if dump:
                dump.log_state(f"DBA (attempt {attempt + 1})", dba_result)
                dump.log_state(f"API (attempt {attempt + 1})", api_result)

            has_dba_feedback = bool((dba_result.prd_feedback_from_dba or "").strip())
            has_api_feedback = bool((api_result.prd_feedback_from_api or "").strip())

            if not has_dba_feedback and not has_api_feedback:
                agent_blockers = [*dba_result.qa_db_blockers, *api_result.qa_api_blockers]
                if agent_blockers:
                    logger.warning(
                        "PRD 계약 생성 완료, DBA/API 자체검증 blocker %d건을 QA repair로 이관 (시도 %d)",
                        len(agent_blockers), attempt + 1,
                    )
                else:
                    logger.info("PRD ↔ DBA/API 검증 통과 (시도 %d)", attempt + 1)
                return self._merge_agent_quality(after_prd, dba_result, api_result).copy(
                    db_schema=dba_result.db_schema,
                    api_spec=api_result.api_spec,
                )

            if attempt < 1:
                feedback = "\n".join(
                    item for item in (
                        dba_result.prd_feedback_from_dba,
                        api_result.prd_feedback_from_api,
                    ) if item
                )
                logger.warning("PRD targeted repair #%d", attempt + 1)
                self._progress.send(
                    pipeline_id, "PRD_ROLLBACK",
                    f"지적된 PRD 섹션만 수정 중... ({attempt + 1}/2회)", 62 + attempt * 2,
                )
                repair = getattr(self._prd, "repair", None)
                if callable(repair):
                    repaired_prd = await self._bounded_repair("prd", self._prd,
                        after_prd.copy(
                            prd_feedback_from_dba=dba_result.prd_feedback_from_dba,
                            prd_feedback_from_api=api_result.prd_feedback_from_api,
                            rollback_count=attempt + 1,
                        ),
                        feedback,
                    )
                    # PRD repair는 PRD 문서만 바꾼다. repair agent가 새 state를
                    # 만들면서 downstream 산출물을 비워도 기존 계약을 보존한다.
                    # PRD repair returns a PRD-centric state and may omit the
                    # sibling artifacts produced immediately before the repair.
                    # Carry those artifacts forward from the actual DBA/API
                    # results so targeted resync can patch them instead of
                    # seeing an empty state and being forced into a blocker.
                    repaired_prd = repaired_prd.copy(
                        db_schema=(
                            repaired_prd.db_schema
                            or dba_result.db_schema
                            or after_prd.db_schema
                        ),
                        api_spec=(
                            repaired_prd.api_spec
                            or api_result.api_spec
                            or after_prd.api_spec
                        ),
                    )
                    baseline = self._merge_agent_quality(after_prd, dba_result, api_result).copy(
                        db_schema=dba_result.db_schema, api_spec=api_result.api_spec,
                    )
                    domains = set()
                    if has_dba_feedback:
                        domains.add("db")
                    if has_api_feedback:
                        domains.add("api")
                    resynced, _ = await self._resync_prd(
                        baseline, repaired_prd, feedback, pipeline_id, dump, domains=domains,
                    )
                    return resynced.copy(rollback_count=attempt + 1)
                blocker = "TARGETED_REPAIR_BLOCKER: PRD patch unsupported; full generation blocked"
                return self._merge_agent_quality(after_prd, dba_result, api_result).copy(
                    db_schema=dba_result.db_schema,
                    api_spec=api_result.api_spec,
                    generation_blockers=[*after_prd.generation_blockers, blocker],
                    prd_feedback_from_dba=dba_result.prd_feedback_from_dba,
                    prd_feedback_from_api=api_result.prd_feedback_from_api,
                    rollback_count=attempt + 1,
                )
            else:
                logger.warning("PRD rollback 최대 횟수 초과 — 현재 결과로 진행")
                return self._merge_agent_quality(after_prd, dba_result, api_result).copy(
                    db_schema=dba_result.db_schema,
                    api_spec=api_result.api_spec,
                    prd_feedback_from_dba=dba_result.prd_feedback_from_dba,
                    prd_feedback_from_api=api_result.prd_feedback_from_api,
                )

        return current

    async def _resync_prd(self, before, repaired, feedback, pipeline_id, dump=None, *, domains=None):
        """All PRD repair paths preserve siblings and use the same delta resync."""
        current = before.copy(
            prd_document=repaired.prd_document, rollback_count=repaired.rollback_count,
            generation_blockers=list(dict.fromkeys([*before.generation_blockers, *repaired.generation_blockers])),
            targeted_repair_results={**before.targeted_repair_results, **repaired.targeted_repair_results},
            targeted_repair_attempts={**before.targeted_repair_attempts, **repaired.targeted_repair_attempts},
        )
        changed = self._changed_prd_feature_ids(before.prd_document, current.prd_document, before.feature_registry)
        def core_ids(document):
            parsed = try_parse_json(document) or {}
            by_name = {item.get("name"): item.get("featureId") for item in before.feature_registry}
            return {str(item.get("featureId") or item.get("id") or by_name.get(item.get("name")) or item.get("name") or "")
                    for item in parsed.get("coreFeatures") or [] if isinstance(item, dict)}
        removed = core_ids(before.prd_document) - core_ids(current.prd_document)
        while True:
            children = {str(item.get("featureId") or item.get("id") or "") for item in before.feature_registry
                        if item.get("parentFeatureId") in removed}
            if children <= removed:
                break
            removed.update(children)
        if removed:
            removed_names = {item.get("name") for item in before.feature_registry if item.get("featureId") in removed}
            current = current.copy(
                feature_registry=[item for item in current.feature_registry if item.get("featureId") not in removed],
                feature_specs=[item for item in current.feature_specs if item.get("featureId") not in removed],
                feature_list=[name for name in current.feature_list if name not in removed_names],
            )
        if changed and self._feature_spec is None and before.feature_registry:
            blocker = "TARGETED_REPAIR_BLOCKER: Feature Spec resync unavailable"
            return current.copy(generation_blockers=[*current.generation_blockers, blocker], qa_approved=False), set()
        if changed and self._feature_spec is not None:
            try:
                async with asyncio.timeout(self._repair_timeout_seconds):
                    candidate = await self._feature_spec.execute(current, dump)
            except Exception as exc:
                logger.exception("Feature Spec resync failed")
                blocker = f"TARGETED_REPAIR_BLOCKER: Feature Spec resync failed ({type(exc).__name__})"
                return current.copy(generation_blockers=[*current.generation_blockers, blocker], qa_approved=False), set()
            if candidate.generation_blockers:
                return candidate.copy(db_schema=before.db_schema, api_spec=before.api_spec), set()
            candidate = candidate.copy(feature_registry=[item for item in candidate.feature_registry if item.get("featureId") not in removed])
            current = self._retain_unchanged_registry(before, candidate, changed)
            current = sync_core_features_with_registry(current)
        current = self._freeze_feature_registry(current)
        affected = set(changed)
        while True:
            children = {str(item.get("featureId") or item.get("id") or "")
                        for item in [*before.feature_registry, *current.feature_registry]
                        if item.get("parentFeatureId") in affected}
            if children <= affected:
                break
            affected.update(children)
        targets = set(domains or []) | self._changed_registry_domains(
            before.feature_registry, current.feature_registry, affected,
        )
        scoped_feedback = self._targeted_resync_feedback(feedback, affected, targets)
        artifact_keys = {"api": set(), "db": set()}
        for feature in [*before.feature_registry, *current.feature_registry]:
            if str(feature.get("featureId") or feature.get("id") or "") not in affected:
                continue
            artifact_keys["api"].update(f"{item.get('method', 'GET')} {item.get('path', '')}"
                                        for item in feature.get("apiContract") or [] if isinstance(item, dict))
            artifact_keys["db"].update(str(name.get("name") or name.get("table")) if isinstance(name, dict) else str(name)
                                       for name in (feature.get("dbContract") or {}).get("tables") or [])
        scoped_feedback += "\n" + json.dumps({
            "artifactKeys": {key: sorted(values) for key, values in artifact_keys.items()},
            "removedFeatureIds": sorted(removed),
        }, ensure_ascii=False)

        async def patch(domain):
            try:
                result = await self._bounded_repair(domain, self._dba if domain == "db" else self._api, current, scoped_feedback)
                return domain, result
            except Exception as exc:
                logger.exception("Targeted %s resync failed", domain)
                ai = describe(exc)
                reason = ai.message if ai else type(exc).__name__
                blocker = f"TARGETED_REPAIR_BLOCKER: {domain} resync failed ({reason})"
                return domain, current.copy(**{
                    "qa_db_blockers" if domain == "db" else "qa_api_blockers": [blocker],
                    "qa_approved": False,
                })

        for domain, result in await asyncio.gather(*(patch(domain) for domain in sorted(targets))):
            fields = ("db_schema", "qa_db_blockers", "prd_feedback_from_dba") if domain == "db" else (
                "api_spec", "openapi_spec", "qa_api_blockers", "prd_feedback_from_api",
            )
            current = current.copy(**{key: getattr(result, key) for key in fields},
                generation_blockers=list(dict.fromkeys([*current.generation_blockers, *result.generation_blockers])),
                targeted_repair_results={**current.targeted_repair_results, domain: result.targeted_repair_results.get(domain, "noop")},
                targeted_repair_attempts=merge_attempts(current.targeted_repair_attempts, result.targeted_repair_attempts),
                qa_approved=False if result.qa_db_blockers or result.qa_api_blockers else current.qa_approved,
            )
        return current, targets

    @staticmethod
    def _changed_prd_feature_ids(before: str, after: str, registry: list[dict] | None = None) -> set[str]:
        """Compare PRD features using Registry IDs, resilient to omitted LLM fields."""
        registry_by_name = {
            str(item.get("name") or "").strip().casefold(): str(item.get("featureId") or item.get("id") or "").strip()
            for item in registry or [] if isinstance(item, dict)
        }

        def indexed(document: str) -> dict[str, str]:
            try:
                parsed = json.loads(document or "{}")
            except (TypeError, json.JSONDecodeError):
                return {}
            items = parsed.get("coreFeatures", []) if isinstance(parsed, dict) else []
            result = {}
            for item in items if isinstance(items, list) else []:
                if not isinstance(item, dict):
                    continue
                name = str(item.get("name") or "").strip()
                key = str(item.get("featureId") or item.get("id") or registry_by_name.get(name.casefold()) or name).strip()
                if key:
                    normalized = {**item, "featureId": key, "name": name}
                    normalized.pop("id", None)
                    def canonical(value):
                        if isinstance(value, str):
                            return " ".join(value.split())
                        if isinstance(value, dict):
                            return {field: canonical(content) for field, content in value.items()}
                        if isinstance(value, list):
                            return [canonical(content) for content in value]
                        return value
                    normalized = canonical(normalized)
                    result[key] = json.dumps(normalized, ensure_ascii=False, sort_keys=True)
            return result

        old, new = indexed(before), indexed(after)
        return {key for key in set(old) | set(new) if old.get(key) != new.get(key)}

    @staticmethod
    def _changed_registry_domains(before: list[dict], after: list[dict], changed_ids: set[str]) -> set[str]:
        def by_id(items):
            return {str(item.get("featureId") or item.get("id") or ""): item for item in items or [] if isinstance(item, dict)}
        old, new = by_id(before), by_id(after)
        domains = set()
        for feature_id in changed_ids:
            before_item, after_item = old.get(feature_id, {}), new.get(feature_id, {})
            if any(before_item.get(key) != after_item.get(key) for key in (
                "apiContract", "sessionPolicy", "readDependencies", "statePolicies", "intervalPolicies", "roles", "ownership",
            )):
                domains.add("api")
            if any(before_item.get(key) != after_item.get(key) for key in (
                "dbContract", "statePolicies", "intervalPolicies", "ownership", "stateTransitions", "transactionRules",
            )):
                domains.add("db")
        return domains

    @staticmethod
    def _targeted_resync_feedback(feedback: str, changed_ids: set[str], domains: set[str]) -> str:
        payload = {
            "mode": "targeted_resync",
            "changedFeatureIds": sorted(changed_ids),
            "domains": sorted(domains),
            "instruction": "수정 대상 artifact만 patch하고 나머지 산출물은 보존하세요.",
        }
        return str(feedback or "")[:8000] + "\n\n[TARGETED_RESYNC]\n" + json.dumps(payload, ensure_ascii=False)

    @staticmethod
    def _freeze_feature_registry(state: PipelineState) -> PipelineState:
        """Detach and fingerprint the Registry before parallel consumers run."""
        registry = copy.deepcopy([
            item for item in state.feature_registry if isinstance(item, dict)
        ])
        canonical = json.dumps(
            registry, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        )
        return state.copy(
            feature_registry=registry,
            feature_specs=copy.deepcopy(state.feature_specs),
            feature_registry_hash=hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        )

    @staticmethod
    def _clone_frozen_registry(state: PipelineState) -> PipelineState:
        return state.copy(
            feature_registry=copy.deepcopy(state.feature_registry),
            feature_specs=copy.deepcopy(state.feature_specs),
        )

    @classmethod
    def _retain_unchanged_registry(
        cls,
        before: PipelineState,
        candidate: PipelineState,
        changed_ids: set[str],
    ) -> PipelineState:
        """Accept Feature Spec resync output only for PRD feature IDs that changed."""
        old_by_id = {
            str(item.get("featureId") or item.get("id") or ""): copy.deepcopy(item)
            for item in before.feature_registry if isinstance(item, dict)
        }
        new_by_id = {
            str(item.get("featureId") or item.get("id") or ""): copy.deepcopy(item)
            for item in candidate.feature_registry if isinstance(item, dict)
        }
        affected = set(changed_ids)
        # Supporting descendants share the lifecycle of their changed parent.
        while True:
            children = {
                key for key, item in {**old_by_id, **new_by_id}.items()
                if item.get("parentFeatureId") in affected
            }
            if children <= affected:
                break
            affected.update(children)
        ordered_ids = [key for key in old_by_id if key not in affected or key in new_by_id]
        ordered_ids.extend(key for key in new_by_id if key not in old_by_id and key in affected)
        merged = [
            new_by_id.get(feature_id, old_by_id.get(feature_id))
            if feature_id in affected else old_by_id[feature_id]
            for feature_id in ordered_ids
        ]
        merged = [item for item in merged if isinstance(item, dict)]
        return candidate.copy(
            feature_registry=merged,
            feature_specs=copy.deepcopy(merged),
            feature_spec_document=json.dumps({"features": merged}, ensure_ascii=False),
        )
