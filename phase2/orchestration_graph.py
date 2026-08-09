import asyncio
import logging
import os
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from phase2.agents.api_agent import ApiAgent
from phase2.agents.dba_agent import DbaAgent
from phase2.agents.pm_agent import PmAgent
from phase2.agents.prd_agent import PrdAgent
from phase2.agents.qa_agent import QaAgent
from phase2.agents.search_agent import SearchAgent
from phase2.sse_service import PipelineProgressService
from phase2.state import PipelineState

logger = logging.getLogger(__name__)


class _Phase2GraphState(TypedDict, total=False):
    """Top-level Phase 2 state.

    Search evidence feeds the PRD first. PM then turns the completed PRD into a
    shared architecture contract. DBA and API draft in parallel from that same
    immutable contract, and a deterministic alignment step reconciles API fields
    with the finalized ERD before QA sees the fan-in result.
    """

    initial_state: PipelineState
    search_state: PipelineState
    pm_state: PipelineState
    prd_state: PipelineState
    dba_state: PipelineState
    api_state: PipelineState
    merged_state: PipelineState
    aligned_state: PipelineState
    final_state: PipelineState
    pipeline_id: str | None
    dump: Any


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
        self._graph = self._build_graph()

    def _build_graph(self):
        graph = StateGraph(_Phase2GraphState)
        graph.add_node("search", self._search_node)
        graph.add_node("pm", self._pm_node)
        graph.add_node("prd", self._prd_node)
        graph.add_node("dba", self._dba_node)
        graph.add_node("api", self._api_node)
        graph.add_node("merge", self._merge_node)
        graph.add_node("align", self._align_node)
        graph.add_node("qa", self._qa_node)
        graph.add_edge(START, "search")
        graph.add_edge("search", "prd")
        graph.add_edge("prd", "pm")
        graph.add_edge("pm", "dba")
        graph.add_edge("pm", "api")
        graph.add_edge(["dba", "api"], "merge")
        graph.add_edge("merge", "align")
        graph.add_edge("align", "qa")
        graph.add_edge("qa", END)
        return graph.compile()

    async def _search_node(self, graph_state: _Phase2GraphState) -> dict:
        pipeline_id, dump = graph_state.get("pipeline_id"), graph_state.get("dump")
        self._progress.send(pipeline_id, "SEARCH", "시장 조사 중...", 30)
        result = await self._search.execute(graph_state["initial_state"], dump)
        if dump:
            dump.log_state("SEARCH", result)
        return {"search_state": result}

    async def _pm_node(self, graph_state: _Phase2GraphState) -> dict:
        pipeline_id, dump = graph_state.get("pipeline_id"), graph_state.get("dump")
        self._progress.send(pipeline_id, "PM", "기능 범위와 공통 계약 확정 중...", 40)
        result = await self._pm.execute(graph_state["prd_state"], dump)
        if dump:
            dump.log_state("PM", result)
        return {"pm_state": result}

    @staticmethod
    def _independent_branch_state(pm_state: PipelineState) -> PipelineState:
        # Both draft agents consume the same PRD + PM contract, but never consume
        # one another's draft. Field-level agreement is handled after fan-in.
        return pm_state.copy(
            db_schema="", api_spec="",
            prd_feedback_from_dba="", prd_feedback_from_api="",
        )

    async def _prd_node(self, graph_state: _Phase2GraphState) -> dict:
        pipeline_id, dump = graph_state.get("pipeline_id"), graph_state.get("dump")
        self._progress.send(pipeline_id, "PRD", "PRD 독립 초안 작성 중...", 50)
        result = await self._prd.execute(graph_state["search_state"], dump)
        return {"prd_state": result}

    async def _dba_node(self, graph_state: _Phase2GraphState) -> dict:
        pipeline_id, dump = graph_state.get("pipeline_id"), graph_state.get("dump")
        self._progress.send(pipeline_id, "DBA", "DB 스키마 독립 초안 작성 중...", 52)
        result = await self._dba.execute(self._independent_branch_state(graph_state["pm_state"]), dump)
        return {"dba_state": result}

    async def _api_node(self, graph_state: _Phase2GraphState) -> dict:
        pipeline_id, dump = graph_state.get("pipeline_id"), graph_state.get("dump")
        self._progress.send(pipeline_id, "API", "API 계약 독립 초안 작성 중...", 54)
        result = await self._api.execute(self._independent_branch_state(graph_state["pm_state"]), dump)
        return {"api_state": result}

    async def _merge_node(self, graph_state: _Phase2GraphState) -> dict:
        pm_state = graph_state["pm_state"]
        merged = pm_state.copy(
            db_schema=graph_state["dba_state"].db_schema,
            api_spec=graph_state["api_state"].api_spec,
            prd_feedback_from_dba="",
            prd_feedback_from_api="",
        )
        if graph_state.get("dump"):
            graph_state["dump"].log_state("PARALLEL_DRAFTS", merged)
        return {"merged_state": merged}

    async def _align_node(self, graph_state: _Phase2GraphState) -> dict:
        """Deterministically reconcile parallel drafts before semantic QA.

        API generation intentionally remains parallel with DBA for latency. Once
        both drafts exist, ApiAgent.repair aligns resource paths and request/
        response fields to the finalized ERD without another LLM generation.
        """
        pipeline_id, dump = graph_state.get("pipeline_id"), graph_state.get("dump")
        self._progress.send(pipeline_id, "CONTRACT_ALIGN", "ERD와 API 계약 정렬 중...", 68)
        aligned = await self._api.repair(
            graph_state["merged_state"],
            [{
                "agent": "API",
                "target": "erd-contract",
                "reason": "병렬 생성된 API를 최종 ERD 필드와 정렬",
                "action": "align",
            }],
            dump,
        )
        if dump:
            dump.log_state("CONTRACT_ALIGN", aligned)
        return {"aligned_state": aligned}

    async def _qa_node(self, graph_state: _Phase2GraphState) -> dict:
        pipeline_id, dump = graph_state.get("pipeline_id"), graph_state.get("dump")
        self._progress.send(pipeline_id, "QA", "산출물 간 연관성 검증 중...", 72)
        result = await self._qa.execute(graph_state["aligned_state"], dump)
        if result.qa_repair_issues:
            result = await self._repair_once(result, pipeline_id, dump)
        return {"final_state": result}

    async def _repair_once(self, state: PipelineState, pipeline_id: str | None, dump=None) -> PipelineState:
        """Run one bounded, item-scoped repair round and revalidate relationships."""
        grouped = {"DBA": [], "API": [], "PRD": []}
        for issue in state.qa_repair_issues:
            agent = str(issue.get("agent") or "").upper()
            if agent in grouped:
                grouped[agent].append(issue)
        current = state
        if grouped["DBA"]:
            self._progress.send(pipeline_id, "DBA_PATCH", "실패한 DB 항목만 교정 중...", 80)
            current = await self._dba.repair(current, grouped["DBA"], dump)
        if grouped["PRD"] or grouped["API"] or grouped["DBA"]:
            self._progress.send(pipeline_id, "ITEM_PATCH", "실패 항목 순차 교정 중...", 84)
        if grouped["PRD"]:
            current = await self._prd.repair(current, grouped["PRD"], dump)
        if grouped["API"] or grouped["DBA"]:
            api_issues = grouped["API"] or [{
                "agent": "API", "target": "erd-contract", "reason": "DB 계약 변경 반영",
            }]
            # API repair must observe PRD changes made immediately above.
            current = await self._api.repair(current, api_issues, dump)
        self._progress.send(pipeline_id, "QA_RECHECK", "영향받은 문서 관계 재검증 중...", 88)
        return await self._qa.execute(current, dump)

    async def run(self, initial_state: PipelineState, pipeline_id: str | None = None) -> PipelineState:
        logger.info("=== Phase 2 오케스트레이션 시작 ===")
        # PIPELINE_DEBUG_DUMP 환경변수가 설정된 경로일 때만 원문 LLM 응답을 덤프한다
        # (프로덕션 기본값은 비활성화 — 재생성/파싱 실패 원인 진단용 임시 스위치)
        dump_dir = os.environ.get("PIPELINE_DEBUG_DUMP")
        dump = None
        if dump_dir:
            from phase2.debug_dump import PipelineDump
            dump = PipelineDump(pipeline_id or "unknown", dump_dir)
        final_state = initial_state

        try:
            result = await self._graph.ainvoke({
                "initial_state": initial_state,
                "pipeline_id": pipeline_id,
                "dump": dump,
            })
            final_state = result["final_state"]
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

    async def retry_failed_domains(self, state: PipelineState, pipeline_id: str | None = None) -> PipelineState:
        """Phase 3 retries only the findings already routed by QA."""
        findings = list(state.qa_repair_issues)
        existing_reasons = {str(item.get("reason") or "") for item in findings}
        error = state.last_validation_error or ""
        for line in (value.strip() for value in error.splitlines() if value.strip()):
            if line in existing_reasons or line.startswith("QA "):
                continue
            if any(token in line for token in ("FK", "DB", "테이블", "스키마", "컬럼")):
                agent = "DBA"
            elif any(token in line for token in ("API", "엔드포인트", "requestBody", "errorCodes", "트랜잭션")):
                agent = "API"
            else:
                agent = "PRD"
            findings.append({"agent": agent, "target": "document", "reason": line, "action": "patch"})
        if not findings:
            findings = [{"agent": "API", "target": "document", "reason": "검증 실패", "action": "patch"}]
        state = state.copy(qa_repair_issues=findings)
        logger.info("Phase 3 선택 재시도 — QA가 지목한 항목 %d건만 교정", len(findings))
        return await self._repair_once(state, pipeline_id)
