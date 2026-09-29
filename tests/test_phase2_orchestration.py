import asyncio
import json
import time

from phase2.agents.qa_agent import QaAgent
from phase2.orchestration_graph import OrchestrationGraph
from phase2.state import PipelineState


class _Progress:
    def send(self, *_args, **_kwargs):
        return None


class _Agent:
    def __init__(self, name, events, delay=0.0, **updates):
        self.name = name
        self.events = events
        self.delay = delay
        self.updates = updates

    async def execute(self, state, _dump=None):
        self.events.append((self.name, "start", time.perf_counter()))
        await asyncio.sleep(self.delay)
        self.events.append((self.name, "end", time.perf_counter()))
        return state.copy(**self.updates)

    async def repair(self, state, _issues, _dump=None):
        return state


def test_top_level_phase2_builds_prd_then_pm_and_fans_out_dba_api():
    events = []
    graph = OrchestrationGraph(
        _Agent("search", events, market_research="research"),
        _Agent("pm", events, feature_list=["feature"]),
        _Agent("prd", events, delay=0.05, prd_document='{"coreFeatures":[]}'),
        _Agent("dba", events, delay=0.05, db_schema='{"tables":[]}'),
        _Agent("api", events, delay=0.05, api_spec='{"endpoints":[]}'),
        _Agent("qa", events),
        _Progress(),
    )
    started = time.perf_counter()
    result = asyncio.run(graph.run(PipelineState()))
    elapsed = time.perf_counter() - started

    starts = {name: stamp for name, kind, stamp in events if kind == "start"}
    ends = {name: stamp for name, kind, stamp in events if kind == "end"}
    assert starts["prd"] >= ends["search"]
    assert starts["pm"] >= ends["prd"]
    assert min(starts[name] for name in ("dba", "api")) >= ends["pm"]
    assert max(starts[name] for name in ("dba", "api")) < min(ends[name] for name in ("dba", "api"))
    assert starts["qa"] >= max(ends[name] for name in ("dba", "api"))
    assert elapsed < 0.2
    assert result.market_research == "research"
    assert result.prd_document == '{"coreFeatures":[]}'


def test_qa_reports_findings_without_rewriting_documents():
    state = PipelineState(
        feature_list=["예약 생성"],
        db_schema=json.dumps({"tables": []}, ensure_ascii=False),
        api_spec=json.dumps({"endpoints": []}, ensure_ascii=False),
        prd_document=json.dumps({"coreFeatures": []}, ensure_ascii=False),
    )
    result = asyncio.run(QaAgent(object()).execute(state))

    assert result.db_schema == state.db_schema
    assert result.api_spec == state.api_spec
    assert result.prd_document == state.prd_document
    assert result.qa_repair_issues
    assert {issue["agent"] for issue in result.qa_repair_issues} <= {"DBA", "API", "PRD"}


def test_repair_runs_prd_before_api_and_api_observes_new_prd():
    events = []

    class _PrdRepair(_Agent):
        async def repair(self, state, _issues, _dump=None):
            events.append("prd")
            return state.copy(prd_document='{"version":"repaired"}')

    class _ApiRepair(_Agent):
        async def repair(self, state, _issues, _dump=None):
            events.append(("api", state.prd_document))
            return state.copy(api_spec='{"observed":"repaired-prd"}')

    graph = OrchestrationGraph(
        _Agent("search", events), _Agent("pm", events),
        _PrdRepair("prd", events), _Agent("dba", events),
        _ApiRepair("api", events), _Agent("qa", events), _Progress(),
    )
    state = PipelineState(
        prd_document='{"version":"old"}',
        qa_repair_issues=[
            {"agent": "PRD", "reason": "repair PRD"},
            {"agent": "API", "reason": "repair API"},
        ],
    )

    result = asyncio.run(graph._repair_once(state, None))

    assert events[:2] == ["prd", ("api", '{"version":"repaired"}')]
    assert result.api_spec == '{"observed":"repaired-prd"}'
