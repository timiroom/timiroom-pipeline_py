import asyncio
import json
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
        feature_spec_agent=None,
        timeout_seconds: float = 900.0,
        repair_timeout_seconds: float = 300.0,
        resync_timeout_seconds: float = 120.0,
        dba_resync_timeout_seconds: float | None = None,
        api_resync_timeout_seconds: float | None = None,
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
        self._timeout_seconds = timeout_seconds
        self._repair_timeout_seconds = repair_timeout_seconds
        self._dba_resync_timeout_seconds = dba_resync_timeout_seconds or resync_timeout_seconds
        self._api_resync_timeout_seconds = api_resync_timeout_seconds or resync_timeout_seconds

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
            after_pm = await self._repair_agent(self._pm, state, validation_error)
            self._progress.send(pipeline_id, "PRD_REPAIR", "변경된 기능 목록 기준 산출물을 재생성 중...", 68)
            repaired = await self._run_prd_with_rollback(after_pm, pipeline_id)
            self._progress.send(pipeline_id, "QA_REPAIR", "수정 산출물 재검수 중...", 78)
            return await self._qa.execute(repaired)

        if not targets:
            logger.warning("선택적 재생성 대상을 판단할 수 없어 QA 정규화 후 Phase 3로 반환")
            return await self._qa.execute(state)

        if "prd" in targets:
            self._progress.send(pipeline_id, "PRD_REPAIR", "지적된 PRD 섹션만 수정 중...", 68)
            repair = getattr(self._prd, "repair", None)
            repaired = await repair(state, validation_error) if callable(repair) else await self._run_prd_with_rollback(state, pipeline_id)
            self._progress.send(pipeline_id, "QA_REPAIR", "수정 산출물 재검수 중...", 78)
            return await self._qa.execute(repaired)

        repair_state = state
        tasks = []
        labels = []
        if "db" in targets:
            tasks.append(self._repair_agent(self._dba, state, validation_error))
            labels.append("DB")
        if "api" in targets:
            tasks.append(self._repair_agent(self._api, state, validation_error))
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
                    generation_blockers=list(dict.fromkeys(
                        [*repair_state.generation_blockers, *result.generation_blockers]
                    )),
                    qa_db_blockers=list(dict.fromkeys(
                        [*repair_state.qa_db_blockers, *result.qa_db_blockers]
                    )),
                )
            else:
                repair_state = repair_state.copy(
                    api_spec=result.api_spec,
                    prd_feedback_from_api=result.prd_feedback_from_api,
                    generation_blockers=list(dict.fromkeys(
                        [*repair_state.generation_blockers, *result.generation_blockers]
                    )),
                    qa_api_blockers=list(dict.fromkeys(
                        [*repair_state.qa_api_blockers, *result.qa_api_blockers]
                    )),
                )

        self._progress.send(pipeline_id, "QA_REPAIR", "수정 산출물 재검수 중...", 78)
        return await self._qa.execute(repair_state)

    @staticmethod
    async def _repair_agent(agent, state: PipelineState, feedback: str) -> PipelineState:
        """Agent가 지원하면 delta patch를 사용하고, 구형/mock Agent는 기존 execute를 쓴다."""
        repair = getattr(agent, "repair", None)
        if callable(repair):
            return await repair(state, feedback)
        return await agent.execute(state)

    @staticmethod
    def _merge_agent_quality(base: PipelineState, dba: PipelineState, api: PipelineState) -> PipelineState:
        generation_blockers = list(dict.fromkeys(
            [*base.generation_blockers, *dba.generation_blockers, *api.generation_blockers]
        ))
        return base.copy(
            generation_blockers=generation_blockers,
            qa_db_blockers=list(dict.fromkeys([*base.qa_db_blockers, *dba.qa_db_blockers])),
            qa_api_blockers=list(dict.fromkeys([*base.qa_api_blockers, *api.qa_api_blockers])),
            qa_approved=False if generation_blockers else base.qa_approved,
        )

    async def _run_prd_with_rollback(
        self, pm_state: PipelineState, pipeline_id: str | None, dump=None
    ) -> PipelineState:
        current = pm_state

        # PRD repair는 최대 한 번만 수행한다. 동일 blocker는 Phase3 blocker로 남긴다.
        for attempt in range(2):
            logger.info("PRD 에이전트 실행 중... (시도 %d)", attempt + 1)
            after_prd = await self._prd.execute(current, dump)
            if self._feature_spec is not None:
                self._progress.send(pipeline_id, "FEATURE_SPEC", "기능명세서 및 지원 기능 확정 중...", 55)
                after_prd = await self._feature_spec.execute(after_prd, dump)

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
                return self._merge_agent_quality(after_prd, dba_result, api_result).copy(
                    db_schema=dba_result.db_schema,
                    api_spec=api_result.api_spec,
                )

            if attempt < 2:
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
                    repaired_prd = await repair(
                        after_prd.copy(
                            prd_feedback_from_dba=dba_result.prd_feedback_from_dba,
                            prd_feedback_from_api=api_result.prd_feedback_from_api,
                            rollback_count=attempt + 1,
                        ),
                        feedback,
                        dump,
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
                    changed_ids = self._changed_prd_feature_ids(
                        after_prd.prd_document, repaired_prd.prd_document,
                        after_prd.feature_registry or repaired_prd.feature_registry,
                    )
                    repeated_repair = attempt >= 1 or repaired_prd.rollback_count >= 2
                    repair_domains = set()
                    if has_dba_feedback:
                        repair_domains.add("db")
                    if has_api_feedback:
                        repair_domains.add("api")

                    # PRD coreFeatures가 실제로 바뀐 경우에만 Feature Spec을
                    # 다시 계산한다. KPI/문구만 바뀐 repair는 downstream을 건드리지 않는다.
                    resynced = repaired_prd
                    if changed_ids and self._feature_spec is not None and not repeated_repair:
                        self._progress.send(
                            pipeline_id, "FEATURE_SPEC_RESYNC",
                            "수정된 PRD 기준 기능 계약 재동기화 중...", 70,
                        )
                        resynced = await self._feature_spec.execute(resynced, dump)

                    if changed_ids and not repeated_repair:
                        repair_domains.update(self._changed_registry_domains(
                            after_prd.feature_registry, resynced.feature_registry, changed_ids,
                        ))

                    # 반복 repair는 full execute를 금지하고, 현재 blocker가 지적한
                    # domain만 delta patch한다. 변경 domain이 없으면 기존 결과를 유지한다.
                    if repair_domains:
                        targeted_feedback = self._targeted_resync_feedback(
                            feedback, changed_ids, repair_domains,
                        )
                        self._progress.send(
                            pipeline_id, "PHASE2_TARGETED_RESYNC",
                            f"변경된 계약·blocker 대상만 재동기화 중: {', '.join(sorted(repair_domains))}", 72,
                        )
                        async def targeted(domain: str):
                            timeout = (
                                self._dba_resync_timeout_seconds
                                if domain == "db"
                                else self._api_resync_timeout_seconds
                            )
                            try:
                                async with asyncio.timeout(timeout):
                                    result = await (
                                        self._repair_agent(self._dba, resynced, targeted_feedback)
                                        if domain == "db"
                                        else self._repair_agent(self._api, resynced, targeted_feedback)
                                    )
                                return domain, result
                            except Exception as exc:
                                label = "DBA" if domain == "db" else "API"
                                blocker = (
                                    f"TARGETED_REPAIR_BLOCKER: {label} targeted resync 실패 "
                                    f"({type(exc).__name__}: {str(exc) or type(exc).__name__})"
                                )
                                logger.error(blocker)
                                if domain == "db":
                                    return domain, resynced.copy(
                                        prd_feedback_from_dba=blocker,
                                        qa_db_blockers=[blocker],
                                        qa_approved=False,
                                    )
                                return domain, resynced.copy(
                                    prd_feedback_from_api=blocker,
                                    qa_api_blockers=[blocker],
                                    qa_approved=False,
                                )

                        results = await asyncio.gather(*(targeted(domain) for domain in sorted(repair_domains)))
                        for domain, result in results:
                            if domain == "db":
                                resynced = resynced.copy(
                                    db_schema=result.db_schema,
                                    prd_feedback_from_dba=result.prd_feedback_from_dba,
                                )
                            else:
                                resynced = resynced.copy(
                                    api_spec=result.api_spec,
                                    prd_feedback_from_api=result.prd_feedback_from_api,
                                )
                    return resynced.copy(
                        db_schema=resynced.db_schema or after_prd.db_schema,
                        api_spec=resynced.api_spec or after_prd.api_spec,
                        rollback_count=attempt + 1,
                    )
                # 구형 Agent/mock 호환용 fallback
                current = after_prd.copy(
                    prd_feedback_from_dba=dba_result.prd_feedback_from_dba,
                    prd_feedback_from_api=api_result.prd_feedback_from_api,
                    rollback_count=attempt + 1,
                )
            else:
                logger.warning("PRD rollback 최대 횟수 초과 — 현재 결과로 진행")
                return self._merge_agent_quality(after_prd, dba_result, api_result).copy(
                    db_schema=dba_result.db_schema,
                    api_spec=api_result.api_spec,
                )

        return current

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
                    normalized = {
                        "featureId": key,
                        "name": name,
                        "priority": item.get("priority"),
                        "actions": item.get("actions") or [],
                        "apiContract": item.get("apiContract") or [],
                        "dbContract": item.get("dbContract") or {},
                    }
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
            if before_item.get("apiContract") != after_item.get("apiContract"):
                domains.add("api")
            if before_item.get("dbContract") != after_item.get("dbContract"):
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
