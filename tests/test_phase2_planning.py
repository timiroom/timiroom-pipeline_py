import asyncio
import json

from phase2.agents.api_agent import ApiAgent
from phase2.agents.dba_agent import DbaAgent


class _PlanningDba(DbaAgent):
    def __init__(self):
        self.prompts = []

    async def _call(self, user_prompt, **_kwargs):
        self.prompts.append(user_prompt)
        return (
            "TABLE: reservable_resources ||| 예약 생성 대상 자원과 가용 상태\n"
            "TABLE: reservations ||| 예약 생성 및 예약 취소 상태와 이력\n"
            "TABLE: reservation_policies ||| 예약 생성과 예약 취소 정책\n"
            "SELF_CHECK: PASS"
        )


class _PlanningApi(ApiAgent):
    def __init__(self):
        self.prompts = []

    async def _call(self, user_prompt, *_args, **_kwargs):
        self.prompts.append(user_prompt)
        return json.dumps({
            "plan": [
                {"method": "GET", "path": "/api/v1/reservations", "description": "예약 생성 대상 조회", "authRequired": False},
                {"method": "POST", "path": "/api/v1/reservations", "description": "예약 생성", "authRequired": False},
                {"method": "GET", "path": "/api/v1/reservations/{id}", "description": "예약 취소 대상 조회", "authRequired": False},
                {"method": "DELETE", "path": "/api/v1/reservations/{id}", "description": "예약 취소", "authRequired": False},
            ]
        }, ensure_ascii=False)


def test_dba_manager_uses_pm_instruction_and_domain_entity_plan():
    agent = _PlanningDba()
    result = asyncio.run(agent._manager_plan_node({"ctx": {
        "feature_list": ["예약 생성", "예약 취소"],
        "instruction": "예약과 자원 사이의 중복 방지 제약을 설계한다",
        "prd_document": "{}",
        "context": "PRD evidence",
        "dump": None,
    }}))

    assert "중복 방지 제약" in agent.prompts[0]
    assert {item["name"] for item in result["plan"]} == {
        "reservable_resources", "reservations", "reservation_policies",
    }


def test_api_manager_uses_pm_instruction_in_actual_plan_prompt():
    agent = _PlanningApi()
    result = asyncio.run(agent._manager_plan_node({"ctx": {
        "feature_list": ["예약 생성", "예약 취소"],
        "instruction": "예약 충돌은 409로 응답한다",
        "prd_document": "{}",
        "context": "PRD and DB schema evidence",
        "dump": None,
    }}))

    assert "예약 충돌은 409" in agent.prompts[0]
    assert len(result["plan"]) == 4
    assert {item["method"] for item in result["plan"]} == {"GET", "POST", "DELETE"}


def test_dba_coverage_repair_generates_missing_feature_table():
    agent = _PlanningDba()

    async def name_worker(feature, _index, _dump=None):
        assert feature == "예약 알림"
        return "reservation_notifications"

    async def table_worker(state):
        skeleton = state["skeleton"]
        return {"tables": [{
            "name": skeleton["name"],
            "description": skeleton["purpose"],
            "columns": [],
            "indexes": [],
        }]}

    agent._table_name_worker = name_worker
    agent._worker_node = table_worker
    result = asyncio.run(agent._generate_missing_tables(
        [], ["예약 알림"], {"context": "PRD", "dump": None},
    ))

    assert result[0]["name"] == "reservation_notifications"
    assert result[0]["description"] == "예약 알림"


def test_dba_manager_backfills_distinct_event_entity():
    agent = _PlanningDba()
    result = asyncio.run(agent._manager_plan_node({"ctx": {
        "feature_list": ["할 일 등록", "담당자 배정", "완료 상태 기록"],
        "instruction": "상태 변경 이력을 보존한다",
        "prd_document": "{}",
        "context": "사용자가 로그인한다",
        "dump": None,
    }}))

    event_items = [
        item for item in result["plan"]
        if item.get("purpose") == "완료 상태 기록"
    ]
    assert event_items
    assert any(token in event_items[0]["name"] for token in ("record", "history", "log", "event"))


def test_dba_manager_merges_missing_feature_purpose_when_table_name_collides():
    class _CollisionDba(DbaAgent):
        def __init__(self):
            pass

        async def _call(self, *_args, **_kwargs):
            return "TABLE: work_items ||| 문서 보관\nSELF_CHECK: PASS"

        async def _table_name_worker(self, _feature, _index, _dump=None):
            return "work_items"

    result = asyncio.run(_CollisionDba()._manager_plan_node({"ctx": {
        "feature_list": ["문서 등록", "문서 보관", "문서 공유"],
        "instruction": "",
        "prd_document": "{}",
        "context": "",
        "dump": None,
    }}))
    work_items = next(item for item in result["plan"] if item["name"] == "work_items")

    assert "문서 등록" in work_items["purpose"]
    assert "문서 보관" in work_items["purpose"]
