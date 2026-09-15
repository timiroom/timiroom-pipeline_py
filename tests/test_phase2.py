import asyncio
from types import SimpleNamespace

import pytest

from phase2.agents.api_agent import ApiAgent, _endpoint_soft_cap, _normalize_endpoints, _prune_plan
from phase2.agents.prd_agent import reconcile_core_features
from phase2.agents.qa_agent import QaAgent
from phase2.agents.search_agent import SearchAgent, _polish_market_research_text
from phase2.llm_runtime import LlmRuntime
from phase2.orchestration_graph import OrchestrationGraph
from phase2.state import PipelineState


class _GraphResult:
    def __init__(self, value):
        self.value = value

    async def ainvoke(self, _state):
        return self.value


class _Progress:
    def __init__(self):
        self.events = []

    def send(self, _pipeline_id, step, _message, _percent):
        self.events.append(step)


class _Agent:
    def __init__(self, transform=None):
        self.calls = 0
        self.transform = transform or (lambda state: state)

    async def execute(self, state, _dump=None):
        self.calls += 1
        return self.transform(state)


def test_api_agent_propagates_prd_feedback():
    agent = ApiAgent(object())
    agent._graph = _GraphResult({"api_spec": '{"endpoints":[]}', "prd_issues": "권한 정책 누락"})

    result = asyncio.run(agent.execute(PipelineState(feature_list=["로그인"])))

    assert result.prd_feedback_from_api == "권한 정책 누락"


def test_api_plan_soft_cap_scales_with_feature_count():
    cap = _endpoint_soft_cap(15)

    assert 15 <= cap <= 36
    assert cap < 40


def test_prune_plan_keeps_auth_and_core_actions_under_cap():
    features = ["견적서 생성", "계약 상태 관리"]
    plan = [
        {"method": "GET", "path": "/api/v1/projects/history", "description": "프로젝트 히스토리 조회"},
        {"method": "POST", "path": "/api/v1/auth/login", "description": "사용자 로그인"},
        {"method": "POST", "path": "/api/v1/estimates", "description": "견적서 생성: 견적 생성"},
        {
            "method": "PATCH",
            "path": "/api/v1/contracts/{contract-id}/status",
            "description": "계약 상태 관리: 상태 변경",
        },
        {"method": "GET", "path": "/api/v1/projects/{project-id}/dashboard", "description": "대시보드 조회"},
    ]

    pruned = _prune_plan(plan, features, 3)

    keys = {f"{ep['method']} {ep['path']}" for ep in pruned}
    assert len(pruned) == 3
    assert "POST /api/v1/auth/login" in keys
    assert "POST /api/v1/estimates" in keys
    assert "PATCH /api/v1/contracts/{contract-id}/status" in keys


def test_normalize_endpoints_drops_semantic_auth_duplicates():
    endpoints = [
        {"method": "POST", "path": "/api/v1/auth/signup", "description": "회원가입"},
        {"method": "POST", "path": "/api/v1/auth/registrations", "description": "사용자 회원가입"},
        {"method": "POST", "path": "/api/v1/auth/signin", "description": "로그인"},
        {"method": "POST", "path": "/api/v1/auth/sessions", "description": "사용자 로그인"},
        {"method": "DELETE", "path": "/api/v1/auth/sessions/current", "description": "로그아웃"},
    ]

    normalized = _normalize_endpoints(endpoints)

    keys = {f"{ep['method']} {ep['path']}" for ep in normalized}
    assert len(normalized) == 3
    assert "POST /api/v1/auth/signup" in keys
    assert "POST /api/v1/auth/signin" in keys
    assert "DELETE /api/v1/auth/sessions/current" in keys


def test_reconcile_core_features_caps_excessive_p0_priorities():
    core_features = [
        {"name": f"핵심 기능 {i}", "priority": "P0", "description": "설명", "requirements": ["요구사항"]}
        for i in range(10)
    ]

    result = reconcile_core_features(core_features, {"included": ["핵심 기능 0", "핵심 기능 1", "핵심 기능 2"]})

    assert sum(1 for item in result if item["priority"] == "P0") <= 4
    assert any(item["priority"] == "P1" for item in result)


def test_market_research_polish_removes_system_limitation_phrasing():
    text = "현재 웹 검색을 수행할 수 없어 최신 법규를 확인할 수 없습니다. 따라서 답변할 수 없습니다."

    polished = _polish_market_research_text(text)

    assert "웹 검색을 수행할 수 없어" not in polished
    assert "확인할 수 없습니다" not in polished
    assert "답변할 수 없습니다" not in polished
    assert "공개적으로 확인 가능한 자료 기준" in polished


def test_qa_max_round_ends_without_false_approval():
    agent = QaAgent(object())
    check = agent._make_check_node("db")

    result = asyncio.run(check({"db_round": 1, "db_issues": ["결함"], "ctx": {}}))

    assert result["db_approved"] is False
    assert result["db_forced"] is True


def test_qa_quality_score_distinguishes_unapproved_result():
    agent = QaAgent(object())

    assert agent._compute_quality_score(False, 1) < agent._compute_quality_score(True, 1)


def test_qa_execute_uses_lightweight_phase3_gate():
    async def fail_if_called(_state):
        raise AssertionError("strict QA graph should not run in lightweight mode")

    agent = QaAgent(object())
    agent._graph = SimpleNamespace(ainvoke=fail_if_called)
    state = PipelineState(
        feature_list=["사용자 로그인"],
        db_schema='{"tables":[{"name":"users","description":"사용자 로그인","columns":[{"name":"id","type":"BIGINT","constraints":"PRIMARY_KEY"}]}],"relationships":[]}',
        api_spec='{"authentication":"Bearer JWT","endpoints":[{"method":"POST","path":"/api/v1/login","description":"사용자 로그인","successResponse":"ok","errorCodes":"400"}]}',
        prd_document='{"coreFeatures":[{"name":"사용자 로그인"}],"mvpScope":{"included":["사용자 로그인"]}}',
    )

    result = asyncio.run(agent.execute(state))

    assert result.qa_approved is True
    assert result.status_message == "QA 경량 게이트 완료 — Phase 3 구조 검증으로 이관"


def test_llm_runtime_bounds_concurrency():
    async def scenario():
        runtime = LlmRuntime(max_concurrency=2, request_timeout_seconds=1)
        active = 0
        max_active = 0

        async def request():
            nonlocal active, max_active
            active += 1
            max_active = max(max_active, active)
            await asyncio.sleep(0.01)
            active -= 1

        await asyncio.gather(*(runtime.call(request) for _ in range(8)))
        return max_active

    assert asyncio.run(scenario()) == 2


def test_llm_runtime_enforces_request_timeout():
    async def scenario():
        runtime = LlmRuntime(max_concurrency=1, request_timeout_seconds=0.01)

        async def request():
            await asyncio.sleep(1)

        await runtime.call(request)

    with pytest.raises(TimeoutError):
        asyncio.run(scenario())


def test_orchestration_enforces_phase_timeout():
    class SlowAgent(_Agent):
        async def execute(self, state, _dump=None):
            await asyncio.sleep(1)
            return state

    graph = OrchestrationGraph(
        SlowAgent(), _Agent(), _Agent(), _Agent(), _Agent(), _Agent(), _Progress(),
        timeout_seconds=0.01,
    )

    with pytest.raises(TimeoutError):
        asyncio.run(graph.run(PipelineState(), "p1"))


def test_selective_retry_only_reruns_failed_domain_and_qa():
    search = _Agent()
    pm = _Agent()
    prd = _Agent()
    dba = _Agent(lambda state: state.copy(db_schema='{"tables":[]}'))
    api = _Agent()
    qa = _Agent()
    graph = OrchestrationGraph(search, pm, prd, dba, api, qa, _Progress(), timeout_seconds=1)

    result = asyncio.run(graph.repair(PipelineState(), "DB 스키마: 관계 누락", "p1"))

    assert result.db_schema == '{"tables":[]}'
    assert (search.calls, pm.calls, prd.calls, api.calls) == (0, 0, 0, 0)
    assert (dba.calls, qa.calls) == (1, 1)


def test_prd_repair_regenerates_prd_and_dependent_artifacts():
    search = _Agent()
    pm = _Agent()
    prd = _Agent()
    dba = _Agent(lambda state: state.copy(db_schema="{}", prd_feedback_from_dba=""))
    api = _Agent(lambda state: state.copy(api_spec="{}", prd_feedback_from_api=""))
    qa = _Agent()
    graph = OrchestrationGraph(search, pm, prd, dba, api, qa, _Progress())

    asyncio.run(graph.repair(PipelineState(), "PRD 문서 오류", "p1", repair_targets=["prd"]))

    assert (search.calls, pm.calls) == (0, 0)
    assert (prd.calls, dba.calls, api.calls, qa.calls) == (1, 1, 1, 1)


def test_pm_repair_skips_search_and_regenerates_dependent_artifacts():
    search = _Agent()
    pm = _Agent(lambda state: state.copy(feature_list=["로그인"]))
    prd = _Agent(lambda state: state.copy(prd_document="{}"))
    dba = _Agent(lambda state: state.copy(db_schema="{}", prd_feedback_from_dba=""))
    api = _Agent(lambda state: state.copy(api_spec="{}", prd_feedback_from_api=""))
    qa = _Agent()
    graph = OrchestrationGraph(search, pm, prd, dba, api, qa, _Progress())

    result = asyncio.run(graph.repair(PipelineState(), "featureList: 기능 목록 오류", "p1", repair_targets=["pm"]))

    assert result.feature_list == ["로그인"]
    assert search.calls == 0
    assert (pm.calls, prd.calls, dba.calls, api.calls, qa.calls) == (1, 1, 1, 1, 1)


def test_search_agent_uses_responses_web_search():
    calls = []

    class Responses:
        async def create(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(
                output_text="조사 결과",
                model_dump=lambda: {
                    "output": [{"action": {"sources": [{"title": "통계청", "url": "https://example.com"}]}}]
                },
            )

    client = SimpleNamespace(responses=Responses())
    agent = SearchAgent(client, "gpt-5.4-mini", web_search_enabled=True)

    result = asyncio.run(agent._query("한국 예약 시장"))

    assert result.startswith("조사 결과")
    assert "통계청: https://example.com" in result
    assert calls[0]["tools"] == [{"type": "web_search"}]
    assert "web_search_call.action.sources" in calls[0]["include"]


def test_api_feedback_triggers_prd_rollback():
    prd = _Agent()

    class Api:
        def __init__(self):
            self.calls = 0

        async def execute(self, state, _dump=None):
            self.calls += 1
            feedback = "PRD 권한 정의 누락" if self.calls == 1 else ""
            return state.copy(api_spec="{}", prd_feedback_from_api=feedback)

    api = Api()
    dba = _Agent(lambda state: state.copy(db_schema="{}", prd_feedback_from_dba=""))
    graph = OrchestrationGraph(_Agent(), _Agent(), prd, dba, api, _Agent(), _Progress())

    result = asyncio.run(graph._run_prd_with_rollback(PipelineState(), "p1"))

    assert prd.calls == 2
    assert api.calls == 2
    assert result.rollback_count == 1
