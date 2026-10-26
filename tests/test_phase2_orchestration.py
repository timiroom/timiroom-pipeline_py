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


def test_top_level_phase2_builds_pm_then_prd_and_fans_out_dba_api():
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
    result = asyncio.run(graph.run(PipelineState()))

    starts = {name: stamp for name, kind, stamp in events if kind == "start"}
    ends = {name: stamp for name, kind, stamp in events if kind == "end"}
    assert starts["pm"] >= ends["search"]
    assert starts["prd"] >= ends["pm"]
    assert min(starts[name] for name in ("dba", "api")) >= ends["prd"]
    assert max(starts[name] for name in ("dba", "api")) < min(ends[name] for name in ("dba", "api"))
    assert starts["qa"] >= max(ends[name] for name in ("dba", "api"))
    assert result.market_research == "research"
    assert result.prd_document == '{"coreFeatures":[]}'


def test_registry_is_frozen_and_parallel_clones_do_not_share_mutations():
    state = PipelineState(
        feature_registry=[{
            "featureId": "items.read", "name": "목록 조회",
            "apiContract": [{"method": "GET", "path": "/api/v1/items"}],
        }],
        feature_specs=[{"featureId": "items.read", "name": "목록 조회"}],
    )

    frozen = OrchestrationGraph._freeze_feature_registry(state)
    left = OrchestrationGraph._clone_frozen_registry(frozen)
    right = OrchestrationGraph._clone_frozen_registry(frozen)
    left.feature_registry[0]["name"] = "변경됨"

    assert frozen.feature_registry[0]["name"] == "목록 조회"
    assert right.feature_registry[0]["name"] == "목록 조회"
    assert len(frozen.feature_registry_hash) == 64


def test_registry_resync_preserves_unchanged_feature_ids():
    before = PipelineState(feature_registry=[
        {"featureId": "a", "name": "A", "apiContract": [{"path": "/api/v1/a"}]},
        {"featureId": "b", "name": "B", "apiContract": [{"path": "/api/v1/b"}]},
    ])
    candidate = PipelineState(feature_registry=[
        {"featureId": "a", "name": "A2", "apiContract": [{"path": "/api/v1/a2"}]},
        {"featureId": "b", "name": "B2", "apiContract": [{"path": "/api/v1/b2"}]},
    ])

    result = OrchestrationGraph._retain_unchanged_registry(before, candidate, {"a"})

    assert [item["name"] for item in result.feature_registry] == ["A2", "B"]


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


def test_db_only_repair_realigns_api_contracts_to_new_db_schema():
    """db만 복구돼도 API 계약은 최종 DB 스키마를 따라가야 한다.

    정렬이 빠지면 api_spec의 featureMappings가 이전 테이블에 묶인 채 남아
    QA·Phase 3가 계약 blocker로 보고하고 결과가 HITL로 빠진다.
    """
    registry = [{
        "featureId": "items.read", "name": "목록 조회",
        "apiContract": [{"method": "GET", "path": "/api/v1/items"}],
        "dbContract": {"tables": [{"name": "item_records"}]},
    }]

    class _DbaRename(_Agent):
        async def repair(self, state, _issues, _dump=None):
            return state.copy(db_schema=json.dumps({
                "tables": [{
                    "name": "item_records", "featureIds": ["items.read"],
                    "columns": [{"name": "id", "type": "BIGINT", "constraints": "PRIMARY KEY"}],
                }],
                "featureMappings": [
                    {"featureId": "items.read", "featureName": "목록 조회", "table": "item_records"},
                ],
            }, ensure_ascii=False))

    events = []
    graph = OrchestrationGraph(
        _Agent("search", events), _Agent("pm", events), _Agent("prd", events),
        _DbaRename("dba", events), _Agent("api", events), _Agent("qa", events), _Progress(),
    )
    state = PipelineState(
        project_name="probe", feature_list=["목록 조회"],
        feature_registry=registry, feature_specs=registry,
        db_schema=json.dumps({
            "tables": [{
                "name": "items", "featureIds": ["items.read"],
                "columns": [{"name": "id", "type": "BIGINT", "constraints": "PRIMARY KEY"}],
            }],
            "featureMappings": [{"featureId": "items.read", "featureName": "목록 조회", "table": "items"}],
        }, ensure_ascii=False),
        api_spec=json.dumps({
            "endpoints": [{
                "method": "GET", "path": "/api/v1/items", "featureIds": ["items.read"],
                "featureName": "목록 조회", "authRequired": True,
                "responseSchema": {"type": "object"}, "successResponse": "200 — OK",
                "errorCodes": "400 — 잘못된 요청",
            }],
            "featureMappings": [{
                "featureId": "items.read", "featureName": "목록 조회", "table": "items",
                "operations": [{"method": "GET", "path": "/api/v1/items"}],
            }],
        }, ensure_ascii=False),
        prd_document=json.dumps(
            {"coreFeatures": [{"featureId": "items.read", "name": "목록 조회"}]}, ensure_ascii=False,
        ),
    )

    result = asyncio.run(graph.repair(state, "DB 스키마 오류", repair_targets=["db"]))

    assert json.loads(result.db_schema)["tables"][0]["name"] == "item_records"
    assert json.loads(result.api_spec)["featureMappings"][0]["table"] == "item_records"


def test_configured_repair_budget_reaches_the_agent_not_a_hardcoded_one():
    """설정된 도메인별 복구 예산이 실제 복구 수행 지점까지 전달돼야 한다.

    실측(세탁소 라이브): Phase2 QA repair가 api 예산 1을 쓰자, 예산 2로 보낸
    Phase3 재시도가 _bounded_repair의 상수 1에 걸려 0.37초 만에
    budget_exhausted로 끝났다. 설정값이 조용히 무시된 것이다.
    """
    calls = []

    class _Api(_Agent):
        async def repair(self, state, _feedback, _dump=None):
            calls.append(state.api_spec)
            return state.copy(api_spec='{"endpoints":[],"repaired":true}')

    def graph(limit):
        events = []
        return OrchestrationGraph(
            _Agent("search", events), _Agent("pm", events), _Agent("prd", events),
            _Agent("dba", events), _Api("api", events), _Agent("qa", events), _Progress(),
            max_targeted_repair_per_domain=limit,
        )

    # Phase2 QA repair가 이미 api 예산 1을 소모한 상태.
    used = PipelineState(api_spec='{"endpoints":[]}', targeted_repair_attempts={"api": 1})

    calls.clear()
    exhausted = asyncio.run(graph(1)._bounded_repair("api", graph(1)._api, used, "fix"))
    assert calls == [], "예산 1이면 재시도하지 않아야 한다"
    assert exhausted.targeted_repair_results["api"] == "budget_exhausted"

    calls.clear()
    allowed = asyncio.run(graph(2)._bounded_repair("api", graph(2)._api, used, "fix"))
    assert len(calls) == 1, "예산 2면 두 번째 복구가 실제로 수행돼야 한다"
    assert allowed.targeted_repair_results["api"] != "budget_exhausted"
    assert allowed.targeted_repair_attempts["api"] == 2


def test_orchestration_graph_rejects_a_non_positive_repair_budget():
    import pytest

    events = []
    with pytest.raises(ValueError, match="max_targeted_repair_per_domain"):
        OrchestrationGraph(
            _Agent("search", events), _Agent("pm", events), _Agent("prd", events),
            _Agent("dba", events), _Agent("api", events), _Agent("qa", events), _Progress(),
            max_targeted_repair_per_domain=0,
        )
