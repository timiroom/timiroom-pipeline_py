import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from phase2.agents.api_agent import (
    ApiAgent,
    _endpoint_soft_cap,
    _normalize_endpoints,
    _prune_plan,
    _sanitize_plan_paths,
    _semantic_endpoint_key,
    api_self_check,
)
from phase2.agents.dba_agent import (
    DbaAgent,
    _fk_target,
    _normalize_special_references,
    annotate_table_feature_ids,
    dba_self_check,
    ensure_primary_keys,
    finalize_table_indexes,
    normalize_table_contract_names,
    reconcile_fk_types,
)
from phase2.agents.feature_spec_agent import (
    FeatureSpecAgent,
    _ensure_required_contracts,
    _score_supporting,
)
from phase2.agents.prd_agent import reconcile_core_features
from phase2.agents.qa_agent import QaAgent
from phase2.agents.search_agent import SearchAgent, _polish_market_research_text
from phase2.feature_registry import normalize_feature_registry
from phase2.llm_runtime import LlmRuntime
from phase2.orchestration_graph import OrchestrationGraph
from phase2.state import PipelineState


class _GraphResult:
    def __init__(self, value):
        self.value = value

    async def ainvoke(self, _state):
        return self.value


def test_api_self_check_removes_unknown_and_duplicate_endpoints():
    registry = [{
        "featureId": "market.list",
        "name": "물품 목록",
        "apiContract": [{"method": "GET", "path": "/api/v1/items"}],
    }]
    endpoints, issues = api_self_check([
        {
            "method": "GET", "path": "/api/v1/items", "featureId": "market.list",
            "description": "물품 목록 조회", "requestBody": "없음",
            "successResponse": "items: array", "errorCodes": "404 — 목록 없음",
        },
        {
            "method": "GET", "path": "/api/v1/items", "featureId": "market.list",
            "description": "물품 목록 조회", "requestBody": "없음",
            "successResponse": "items: array", "errorCodes": "404 — 목록 없음",
        },
        {"method": "POST", "path": "/api/v1/reports", "featureId": "unknown.feature"},
    ], registry)
    assert endpoints
    assert any(issue.startswith("API_FEATURE_ID_REQUIRED:") for issue in issues)
    assert len([item for item in endpoints if item.get("featureId") == "market.list"]) == 1


def test_api_self_check_rejects_auxiliary_only_and_unsafe_get_contracts():
    registry = [{
        "featureId": "notifications.delivery",
        "name": "배송 상태 알림",
        "actions": ["notify"],
        "apiContract": [{"method": "GET", "path": "/api/v1/notifications/{id}/retry"}],
    }]

    _endpoints, issues = api_self_check([], registry)

    assert "API_CORE_ACTION_MISSING: notifications.delivery POST" in issues
    assert "API_AUXILIARY_ONLY_CONTRACT: notifications.delivery" in issues
    assert any(issue.startswith("API_UNSAFE_GET_ACTION:") for issue in issues)


def test_feature_spec_budget_preserves_twelve_features_and_caps_p0():
    features = [
        {
            "featureId": f"feature.{index}",
            "name": f"기능 {index}",
            "source": "prd_core" if index <= 6 else "supporting",
            "priority": "P0" if index <= 6 else "P1",
            "priorityScore": 13 if index <= 6 else 10,
        }
        for index in range(1, 15)
    ]

    result = FeatureSpecAgent._apply_mvp_budget(features)

    assert len(result) == 12
    assert 3 <= sum(item["priority"] == "P0" for item in result) <= 5


def test_feature_spec_merges_generic_validation_and_duplicate_auth():
    features = [
        {
            "featureId": "auth.login", "name": "로그인", "source": "supporting",
            "actions": ["login"],
            "apiContract": [{"method": "POST", "path": "/api/v1/auth/login"}],
            "dbContract": {"tables": ["users"], "foreignKeys": []},
        },
        {
            "featureId": "supporting.account", "name": "회원가입 및 로그인", "source": "supporting",
            "actions": ["signup", "login"],
            "apiContract": [{"method": "POST", "path": "/api/v1/auth/signup"}],
            "dbContract": {"tables": ["users"], "foreignKeys": []},
        },
        {
            "featureId": "supporting.validation", "name": "입력 검증 및 오류 처리",
            "source": "supporting", "parentFeatureId": "auth.login",
            "requirements": ["입력 오류를 반환한다"],
        },
    ]

    result = FeatureSpecAgent._merge_semantic_duplicates(features)

    assert [item["featureId"] for item in result] == ["auth.login", "supporting.account"]
    assert {item["path"] for item in result[0]["apiContract"]} == {
        "/api/v1/auth/login",
    }
    assert "입력 오류를 반환한다" in result[0]["requirements"]


def test_api_self_check_replaces_wrong_path_with_contract_endpoint():
    registry = [{
        "featureId": "market.create",
        "name": "물품 등록",
        "apiContract": [{"method": "POST", "path": "/api/v1/items"}],
    }]
    endpoints, issues = api_self_check([{
        "method": "POST",
        "path": "/api/v1/item-register",
        "featureId": "market.create",
        "description": "물품 등록",
        "requestBody": "계약 기반 요청 본문",
        "successResponse": "success: boolean",
        "errorCodes": "400 — 요청 오류",
    }], registry)

    assert not any(issue.startswith("API_CONTRACT_PATH_MISMATCH:") for issue in issues)
    assert any(issue.startswith("API_CONTRACT_DETAIL_PLACEHOLDER:") for issue in issues)
    assert any(endpoint.get("path") == "/api/v1/items" for endpoint in endpoints)


def test_api_self_check_drops_contract_external_endpoint_without_leaking_its_placeholder():
    registry = [{
        "featureId": "auth.login",
        "name": "로그인",
        "apiContract": [{"method": "POST", "path": "/api/v1/auth/login"}],
    }]
    endpoints, issues = api_self_check([
        {
            "method": "POST", "path": "/api/v1/auth/login", "featureId": "auth.login",
            "description": "로그인: 자격 증명 검증", "requestBody": "email, password",
            "successResponse": "accessToken", "errorCodes": "401 — 인증 실패",
        },
        {
            "method": "POST", "path": "/api/v1/auth/refresh", "featureId": "auth.login",
            "description": "Registry 계약 endpoint", "requestBody": "계약 기반 요청 본문",
            "successResponse": "success: boolean", "errorCodes": "400 — 요청 오류",
        },
    ], registry)

    assert {(item["method"], item["path"]) for item in endpoints} == {
        ("POST", "/api/v1/auth/login"),
    }
    assert not any("/auth/refresh" in issue for issue in issues)


def test_refresh_api_contract_adds_atomic_db_contract():
    item = _ensure_required_contracts({
        "featureId": "auth.refresh",
        "name": "세션 갱신",
        "actions": ["refresh"],
        "ownership": {"ownerEntity": "members"},
        "apiContract": [{"method": "POST", "path": "/api/v1/auth/refresh"}],
        "dbContract": {"tables": ["members"], "foreignKeys": []},
    })

    assert {"members", "refresh_tokens"} <= set(item["dbContract"]["tables"])
    assert {
        "table": "refresh_tokens",
        "column": "member_id",
        "references": {"table": "members", "column": "id"},
        "nullable": False,
    } in item["dbContract"]["foreignKeys"]


def test_truncated_api_repair_response_is_not_partially_applied():
    agent = object.__new__(ApiAgent)
    agent._call = AsyncMock(return_value=(
        '{"patches":[{"method":"GET","path":"/api/v1/items",'
        '"description":"부분 patch"}]'
    ))
    original = {
        "authentication": "공개 API",
        "endpoints": [{
            "method": "GET", "path": "/api/v1/items", "featureId": "items.read",
            "description": "목록 조회", "requestBody": "없음",
            "successResponse": "items: array", "errorCodes": "500 — 서버 오류",
        }],
    }
    state = PipelineState(
        api_spec=json.dumps(original, ensure_ascii=False),
        feature_list=["목록 조회"],
        feature_registry=[{
            "featureId": "items.read", "name": "목록 조회",
            "apiContract": [{"method": "GET", "path": "/api/v1/items"}],
        }],
    )

    result = asyncio.run(agent.repair(state, "description 수정"))

    assert result.api_spec == state.api_spec


def test_indexes_are_finalized_after_columns_and_fk_targets():
    tables = [
        {
            "name": "users",
            "columns": [{"name": "id", "type": "BIGINT", "constraints": "PRIMARY_KEY"}],
            "indexes": [],
        },
        {
            "name": "items",
            "columns": [
                {"name": "id", "type": "BIGINT", "constraints": "PRIMARY_KEY"},
                {"name": "user_id", "type": "BIGINT", "constraints": "FOREIGN_KEY REFERENCES users(id)"},
            ],
            "indexes": ["CREATE INDEX idx_items_missing ON items(missing_column)"],
        },
    ]

    finalized = finalize_table_indexes(tables)
    indexes = finalized[1]["indexes"]

    assert not any("missing_column" in index for index in indexes)
    assert any("(user_id)" in index for index in indexes)


def test_final_indexes_include_status_time_composite():
    tables = [{
        "name": "deliveries",
        "columns": [
            {"name": "id", "type": "BIGINT", "constraints": "PRIMARY_KEY"},
            {"name": "status", "type": "VARCHAR(30)", "constraints": "NOT_NULL"},
            {"name": "scheduled_at", "type": "TIMESTAMPTZ", "constraints": "NOT_NULL"},
        ],
        "indexes": [],
    }]

    finalize_table_indexes(tables)

    assert any("(status, scheduled_at)" in index for index in tables[0]["indexes"])


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


def test_api_targeted_repair_preserves_unrelated_endpoints():
    agent = ApiAgent(object())
    agent._call = lambda *_args, **_kwargs: _async_text(
        '{"patches":[{"method":"POST","path":"/api/v1/orders","description":"주문 생성 및 검증","requestBody":"order","successResponse":"order","errorCodes":"400"}]}'
    )
    state = PipelineState(api_spec='{"authentication":"JWT","endpoints":['
        '{"method":"GET","path":"/api/v1/orders","description":"주문 목록","successResponse":"list","errorCodes":"400"},'
        '{"method":"POST","path":"/api/v1/orders","description":"잘못된 설명","successResponse":"old","errorCodes":"500"}]}'
    )

    result = asyncio.run(agent.repair(state, "POST /api/v1/orders의 요청 형식이 잘못되었습니다"))
    endpoints = __import__("json").loads(result.api_spec)["endpoints"]

    assert len(endpoints) == 2
    assert endpoints[0]["description"] == "주문 목록"
    assert endpoints[1]["description"] == "주문 생성 및 검증"


async def _async_text(value):
    return value


def test_api_plan_soft_cap_scales_with_feature_count():
    cap = _endpoint_soft_cap(15)

    assert 15 <= cap <= 36
    assert cap < 40


def test_feature_registry_exposes_explicit_contract_fields_and_stable_ids():
    registry = normalize_feature_registry([{
        "featureId": "reservation.create",
        "name": "예약 생성",
        "actions": ["create"],
        "apiContract": [{"method": "POST", "path": "/api/v1/reservations"}],
        "dbContract": {"tables": ["reservations"], "foreignKeys": ["reservations.user_id -> users.id"]},
    }], ["예약 생성"])

    assert registry[0]["featureId"] == "reservation.create"
    assert registry[0]["apiContract"][0]["path"] == "/api/v1/reservations"
    assert registry[0]["dbContract"]["tables"] == ["reservations"]


def test_api_path_normalization_and_auth_semantic_dedupe_do_not_drop_refresh_logout():
    plan = [
        {"method": "POST", "path": "/api/v1/api/auth/login", "description": "로그인"},
        {"method": "POST", "path": "/api/auth/refresh", "description": "토큰 갱신"},
        {"method": "POST", "path": "/api/auth/logout", "description": "로그아웃"},
    ]
    normalized = _normalize_endpoints(_sanitize_plan_paths(plan))
    paths = {ep["path"] for ep in normalized}

    assert "/api/v1/auth/login" in paths
    assert "/api/v1/auth/refresh" in paths
    assert "/api/v1/auth/logout" in paths


def test_auth_logout_with_token_word_is_not_deduped_as_refresh():
    plan = _normalize_endpoints([
        {"method": "POST", "path": "/api/v1/auth/refresh", "description": "토큰 갱신"},
        {"method": "POST", "path": "/api/v1/auth/logout", "description": "토큰 폐기 및 로그아웃"},
    ])
    assert {ep["path"] for ep in plan} == {"/api/v1/auth/refresh", "/api/v1/auth/logout"}


def test_auth_actions_are_not_semantically_collapsed():
    endpoints = _normalize_endpoints([
        {"method": "POST", "path": "/api/v1/auth/refresh", "description": "토큰 갱신"},
        {"method": "POST", "path": "/api/v1/auth/signup", "description": "회원가입"},
        {"method": "POST", "path": "/api/v1/auth/verify", "description": "이메일 인증"},
    ])
    assert {ep["path"] for ep in endpoints} == {
        "/api/v1/auth/refresh", "/api/v1/auth/signup", "/api/v1/auth/verify",
    }


def test_dba_fk_constraints_keep_reference_target_and_type():
    tables = [
        {"name": "users", "columns": [{"name": "id", "type": "BIGINT", "constraints": "PRIMARY_KEY"}]},
        {"name": "reservations", "columns": [{"name": "id", "type": "BIGINT", "constraints": "PRIMARY_KEY"},
            {"name": "user_id", "type": "VARCHAR(255)", "constraints": "NOT_NULL"}]},
    ]
    reconcile_fk_types(tables)
    user_id = tables[1]["columns"][1]

    assert user_id["type"] == "BIGINT"
    assert "FOREIGN_KEY" in user_id["constraints"]
    assert "REFERENCES users(id)" in user_id["constraints"]


def test_dba_buyer_and_seller_ids_fall_back_to_users_when_no_special_table_exists():
    lookup = {
        "users": "users",
        "inquiries": "inquiries",
    }

    assert _fk_target("buyer_id", lookup) == "users"
    assert _fk_target("seller_id", lookup) == "users"


def test_dba_backstop_adds_missing_pk_and_unique_compound_fk_target():
    tables = [
        {"name": "pet_guardians", "columns": [{"name": "id", "type": "BIGINT", "constraints": ""},
            {"name": "guardian_id", "type": "BIGINT", "constraints": "FOREIGN_KEY"}]},
        {"name": "users", "columns": [{"name": "id", "type": "BIGINT", "constraints": "PRIMARY_KEY"}]},
        {"name": "grooming_records", "columns": [{"name": "id", "type": "BIGINT", "constraints": "PRIMARY_KEY"}]},
        {"name": "grooming_photos", "columns": [{"name": "id", "type": "BIGINT", "constraints": "PRIMARY_KEY"},
            {"name": "record_id", "type": "BIGINT", "constraints": "FOREIGN_KEY"}]},
    ]
    ensure_primary_keys(tables)
    reconcile_fk_types(tables)

    assert "PRIMARY_KEY" in tables[0]["columns"][0]["constraints"]
    photo_record = tables[3]["columns"][1]
    assert "REFERENCES grooming_records(id)" in photo_record["constraints"]


def test_dba_special_references_distinguish_external_polymorphic_and_self_fk():
    tables = [
        {"name": "refresh_tokens", "columns": [
            {"name": "id", "type": "BIGINT", "constraints": "PRIMARY_KEY"},
            {"name": "replaced_by_token_id", "type": "BIGINT", "constraints": "NULL"},
        ]},
        {"name": "notification_deliveries", "columns": [
            {"name": "id", "type": "BIGINT", "constraints": "PRIMARY_KEY"},
            {"name": "provider_message_id", "type": "VARCHAR(255)", "constraints": "FOREIGN_KEY"},
        ]},
        {"name": "audit_logs", "columns": [
            {"name": "id", "type": "BIGINT", "constraints": "PRIMARY_KEY"},
            {"name": "target_id", "type": "BIGINT", "constraints": "FOREIGN_KEY"},
        ]},
    ]

    _normalize_special_references(tables)
    reconcile_fk_types(tables)

    refresh = tables[0]["columns"][1]
    provider = tables[1]["columns"][1]
    audit_names = {column["name"] for column in tables[2]["columns"]}
    assert "REFERENCES refresh_tokens(id)" in refresh["constraints"]
    assert "FOREIGN_KEY" not in provider["constraints"]
    assert "target_type" in audit_names


def test_dba_request_id_is_correlation_id_without_requests_table():
    tables = [{"name": "audit_logs", "columns": [
        {"name": "id", "type": "BIGINT", "constraints": "PRIMARY_KEY"},
        {
            "name": "request_id",
            "type": "BIGINT",
            "constraints": "NULL FOREIGN_KEY REFERENCES audit_logs(id)",
        },
    ]}]

    checked, relationships, issues = dba_self_check(
        tables, ["audit_logs (1:N) audit_logs"], [],
    )

    request_id = checked[0]["columns"][1]
    assert "FOREIGN_KEY" not in request_id["constraints"]
    assert "REFERENCES" not in request_id["constraints"]
    assert relationships == []
    assert issues == []


def test_dba_request_id_keeps_fk_when_requests_table_exists():
    tables = [
        {"name": "requests", "columns": [
            {"name": "id", "type": "BIGINT", "constraints": "PRIMARY_KEY"},
        ]},
        {"name": "request_events", "columns": [
            {"name": "id", "type": "BIGINT", "constraints": "PRIMARY_KEY"},
            {"name": "request_id", "type": "BIGINT", "constraints": "NOT_NULL"},
        ]},
    ]

    checked, relationships, issues = dba_self_check(tables, [], [])

    request_id = checked[1]["columns"][1]
    assert "REFERENCES requests(id)" in request_id["constraints"]
    assert "requests (1:N) request_events" in relationships
    assert issues == []


def test_supporting_priority_uses_dependency_score_and_release_gate():
    priority, score, gate = _score_supporting("inventory validation", "required release blocker for data integrity")

    assert priority == "P1"
    assert score >= 5
    assert gate is True


def test_feature_spec_core_contract_expands_methods_from_requirements():
    item = {
        "featureId": "feature_001",
        "name": "식물 등록",
        "description": "사용자의 식물을 저장하고 목록으로 제공한다.",
        "requirements": ["등록한 식물을 조회하고 수정하며 삭제할 수 있어야 한다."],
        "actions": ["create"],
        "apiContract": [
            {"method": "POST", "path": "/api/v1/plants", "action": "create"},
        ],
        "dbContract": {"tables": ["plants"], "foreignKeys": []},
    }

    result = _ensure_required_contracts(item)
    contracts = {
        (contract["method"], contract["path"])
        for contract in result["apiContract"]
    }

    assert ("POST", "/api/v1/plants") in contracts
    assert ("GET", "/api/v1/plants") in contracts
    assert ("PATCH", "/api/v1/plants/{plantId}") in contracts
    assert ("DELETE", "/api/v1/plants/{plantId}") in contracts


def test_feature_spec_failure_preserves_source_and_blocks_downstream():
    agent = FeatureSpecAgent(object(), "test")
    state = PipelineState(
        feature_list=["재고 등록"],
        feature_registry=[{"featureId": "inventory.register", "name": "재고 등록", "apiContract": [],
                           "dbContract": {"tables": [], "foreignKeys": []}}],
        prd_document='{"coreFeatures":[{"name":"재고 등록"}]}',
    )

    result = asyncio.run(agent.execute(state))
    assert result.feature_registry[0]["featureId"] == "inventory.register"
    assert result.feature_registry == state.feature_registry
    assert result.generation_blockers
    assert not result.feature_spec_document


def test_feature_registry_preserves_feature_spec_supporting_items_for_downstream_agents():
    registry = normalize_feature_registry(
        [
            {"featureId": "reservation.create", "name": "예약 생성"},
            {"featureId": "auth.signup", "name": "회원가입", "source": "supporting"},
        ],
        ["예약 생성"],
        preserve_extra=True,
    )

    assert [item["featureId"] for item in registry] == ["reservation.create", "auth.signup"]


def test_feature_registry_deduplicates_extra_feature_ids():
    registry = normalize_feature_registry(
        [
            {"featureId": "feature_001", "name": "핵심"},
            {"featureId": "feature_001", "name": "보조"},
        ],
        ["핵심"],
        preserve_extra=True,
    )

    assert len({item["featureId"] for item in registry}) == len(registry)


def test_dba_table_aliases_follow_registry_and_get_feature_mapping():
    tables = [{
        "name": "organization_memberships",
        "columns": [{"name": "id", "type": "BIGINT", "constraints": "PRIMARY_KEY"}],
        "indexes": [],
    }]
    registry = [{
        "featureId": "membership.manage",
        "dbContract": {"tables": ["memberships"], "foreignKeys": []},
    }]

    tables, relationships = normalize_table_contract_names(
        tables, ["organizations (1:N) organization_memberships"], registry,
    )
    annotate_table_feature_ids(tables, registry)

    assert tables[0]["name"] == "memberships"
    assert tables[0]["featureIds"] == ["membership.manage"]
    assert relationships == ["organizations (1:N) memberships"]


def test_dba_auxiliary_table_inherits_feature_mapping_from_fk_target():
    tables = [
        {
            "name": "orders", "featureIds": [],
            "columns": [{"name": "id", "type": "BIGINT", "constraints": "PRIMARY_KEY"}],
        },
        {
            "name": "order_status_histories", "featureIds": [],
            "columns": [
                {"name": "id", "type": "BIGINT", "constraints": "PRIMARY_KEY"},
                {
                    "name": "order_id", "type": "BIGINT",
                    "constraints": "FOREIGN_KEY REFERENCES orders(id)",
                },
            ],
        },
    ]
    registry = [{
        "featureId": "orders.track",
        "dbContract": {"tables": ["orders"], "foreignKeys": []},
    }]

    annotate_table_feature_ids(tables, registry)

    assert tables[1]["featureIds"] == ["orders.track"]


def test_dba_manager_plan_accepts_valid_partial_plan_for_targeted_patch():
    agent = DbaAgent(object())
    calls = {"count": 0}

    async def call(*_args, **_kwargs):
        calls["count"] += 1
        return '{"plan": [{"name":"reservations","purpose":"예약 등록"},{"name":"users","purpose":"회원"},{"name":"payments","purpose":"결제"}]}'

    agent._call = call
    result = asyncio.run(agent._manager_plan_node({
        "ctx": {
            "feature_list": ["예약 등록", "회원정보 수정"],
            "feature_str": "- 예약 등록\\n- 회원정보 수정",
            "instruction": "",
            "context": "",
            "feature_registry": "[]",
        }
    }))

    assert calls["count"] == 1
    assert len(result["plan"]) == 3


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


def test_auth_semantic_keys_keep_refresh_and_password_reset_confirmation_distinct():
    assert _semantic_endpoint_key({
        "method": "POST", "path": "/api/v1/auth/refresh", "description": "토큰 갱신"
    }) == ("auth", "refresh")
    assert _semantic_endpoint_key({
        "method": "POST", "path": "/api/v1/auth/password-reset/confirm", "description": "비밀번호 재설정 확인"
    }) == ("auth", "password_reset_confirm")
    assert _semantic_endpoint_key({
        "method": "POST", "path": "/api/v1/auth/password-reset", "description": "비밀번호 재설정 요청"
    }) == ("auth", "password_reset_request")


def test_reconcile_core_features_caps_excessive_p0_priorities():
    core_features = [
        {"name": f"핵심 기능 {i}", "priority": "P0", "description": "설명", "requirements": ["요구사항"]}
        for i in range(10)
    ]

    result = reconcile_core_features(core_features, {"included": ["핵심 기능 0", "핵심 기능 1", "핵심 기능 2"]})

    assert sum(1 for item in result if item["priority"] == "P0") <= 5
    assert any(item["priority"] == "P1" for item in result)


def test_market_research_polish_removes_system_limitation_phrasing():
    text = "현재 웹 검색을 수행할 수 없어 최신 법규를 확인할 수 없습니다. 따라서 답변할 수 없습니다."

    polished = _polish_market_research_text(text)

    assert "웹 검색을 수행할 수 없어" not in polished
    assert "확인할 수 없습니다" not in polished
    assert "답변할 수 없습니다" not in polished
    assert "공개적으로 확인 가능한 자료 기준" in polished


def test_search_agent_output_never_leaks_system_limitation_phrasing():
    """정리 함수가 에이전트 산출 경로에 실제로 연결돼 있어야 한다.

    market_research는 PRD 프롬프트와 저장되는 marketResearch 산출물에 그대로
    실리므로, 모델이 드러낸 시스템 사정 문구가 통과하면 최종 결과물에 남는다.
    """
    import asyncio
    from types import SimpleNamespace

    from phase2.state import PipelineState

    leaky = "현재 웹 검색을 수행할 수 없어 최신 법규를 확인할 수 없습니다. 출처: https://example.go.kr/a 2026"

    class _Client:
        def __init__(self):
            self.responses = SimpleNamespace(create=self._responses)
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._chat))

        async def _responses(self, **_kwargs):
            return SimpleNamespace(output_text=leaky, model_dump=lambda: {"output": []})

        async def _chat(self, **_kwargs):
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="할 일 관리 협업"))]
            )

    agent = SearchAgent(_Client(), "stub-model", web_search_enabled=True)
    research = asyncio.run(agent.execute(PipelineState(user_query="할 일 정리"))).market_research

    assert "웹 검색을 수행할 수 없어" not in research
    assert "확인할 수 없습니다" not in research
    assert "공개적으로 확인 가능한 자료 기준" in research
    # 추적 가능한 출처는 보존해야 한다.
    assert "https://example.go.kr/a" in research


def test_qa_max_round_ends_without_false_approval():
    agent = QaAgent(object())
    check = agent._make_check_node("db")

    result = asyncio.run(check({"db_round": 1, "db_issues": ["결함"], "ctx": {}}))

    assert result["db_approved"] is False
    assert result["db_forced"] is True


def test_qa_quality_score_distinguishes_unapproved_result():
    agent = QaAgent(object())

    assert agent._compute_quality_score(False, 1) < agent._compute_quality_score(True, 1)


def test_qa_gate_blocks_contract_breaks_but_allows_quality_warnings():
    classified = QaAgent._classify_issues(
        ["DB 설명 문구 보강 필요", "DB 스키마: 참조 대상 테이블을 찾을 수 없습니다"],
        ["endpoint 설명이 간략함", "API 스펙: 기능에 대응하는 endpoint가 없어 보임"],
        [],
    )

    assert classified["db_warnings"] == ["DB 설명 문구 보강 필요"]
    assert classified["db_blockers"] == ["DB 스키마: 참조 대상 테이블을 찾을 수 없습니다"]
    assert classified["api_warnings"] == ["endpoint 설명이 간략함"]
    assert classified["api_blockers"] == ["API 스펙: 기능에 대응하는 endpoint가 없어 보임"]


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

    assert result.qa_approved is False
    assert result.status_message == "QA 경량 게이트 완료 — Phase 3 구조 검증으로 이관"


def test_qa_lightweight_gate_rejects_cross_artifact_entity_mismatch():
    agent = QaAgent(object())
    state = PipelineState(
        feature_list=["사용자 로그인"],
        db_schema='{"tables":[{"name":"users","columns":[{"name":"id","type":"BIGINT","constraints":"PRIMARY_KEY"}]}],"relationships":[]}',
        api_spec='{"authentication":"Bearer JWT","endpoints":[{"method":"POST","path":"/api/v1/login","description":"사용자 로그인","successResponse":"ok","errorCodes":"400"}]}',
        prd_document='{"coreFeatures":[{"name":"사용자 로그인"}],"techStack":{"database":"courses(id PK, academy_id FK→academies.id)"}}',
    )

    result = asyncio.run(agent.execute(state))

    assert result.qa_approved is False
    assert any("PRD에 정의된 테이블" in issue for issue in result.qa_prd_issues)


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


def test_selective_retry_does_not_fallback_to_execute():
    search = _Agent()
    pm = _Agent()
    prd = _Agent()
    dba = _Agent(lambda state: state.copy(db_schema='{"tables":[]}'))
    api = _Agent()
    qa = _Agent()
    graph = OrchestrationGraph(search, pm, prd, dba, api, qa, _Progress(), timeout_seconds=1)

    result = asyncio.run(graph.repair(PipelineState(), "DB 스키마: 관계 누락", "p1"))

    assert result.db_schema == ""
    assert result.generation_blockers
    assert (search.calls, pm.calls, prd.calls, api.calls) == (0, 0, 0, 0)
    assert (dba.calls, qa.calls) == (0, 1)


def test_prd_repair_without_patch_support_blocks_full_generation():
    search = _Agent()
    pm = _Agent()
    prd = _Agent()
    dba = _Agent(lambda state: state.copy(db_schema="{}", prd_feedback_from_dba=""))
    api = _Agent(lambda state: state.copy(api_spec="{}", prd_feedback_from_api=""))
    qa = _Agent()
    graph = OrchestrationGraph(search, pm, prd, dba, api, qa, _Progress())

    result = asyncio.run(graph.repair(PipelineState(), "PRD 문서 오류", "p1", repair_targets=["prd"]))

    assert (search.calls, pm.calls) == (0, 0)
    assert (prd.calls, dba.calls, api.calls, qa.calls) == (0, 0, 0, 1)
    assert result.generation_blockers


def test_pm_repair_without_patch_support_does_not_regenerate_dependencies():
    search = _Agent()
    pm = _Agent(lambda state: state.copy(feature_list=["로그인"]))
    prd = _Agent(lambda state: state.copy(prd_document="{}"))
    dba = _Agent(lambda state: state.copy(db_schema="{}", prd_feedback_from_dba=""))
    api = _Agent(lambda state: state.copy(api_spec="{}", prd_feedback_from_api=""))
    qa = _Agent()
    graph = OrchestrationGraph(search, pm, prd, dba, api, qa, _Progress())

    result = asyncio.run(graph.repair(PipelineState(), "featureList: 기능 목록 오류", "p1", repair_targets=["pm"]))

    assert result.feature_list == []
    assert result.generation_blockers
    assert search.calls == 0
    assert (pm.calls, prd.calls, dba.calls, api.calls, qa.calls) == (0, 0, 0, 0, 1)


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


def test_api_feedback_does_not_trigger_full_prd_regeneration_without_patch_support():
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

    assert prd.calls == 1
    assert api.calls == 1
    assert result.rollback_count == 1
    assert result.generation_blockers


def test_llm_runtime_retries_busy_gateway_then_succeeds():
    import httpx
    from openai import RateLimitError

    calls = {"n": 0}

    async def factory():
        calls["n"] += 1
        if calls["n"] < 3:
            response = httpx.Response(429, request=httpx.Request("POST", "https://gw.test/v1/chat/completions"))
            raise RateLimitError("busy", response=response, body=None)
        return "ok"

    runtime = LlmRuntime(max_concurrency=1, request_timeout_seconds=1, busy_retries=3, busy_backoff_seconds=0.001)
    assert asyncio.run(runtime.call(factory)) == "ok"
    assert calls["n"] == 3


def test_llm_runtime_gives_up_after_busy_retries():
    import httpx
    from openai import RateLimitError

    async def factory():
        response = httpx.Response(429, request=httpx.Request("POST", "https://gw.test/v1/chat/completions"))
        raise RateLimitError("busy", response=response, body=None)

    runtime = LlmRuntime(max_concurrency=1, request_timeout_seconds=1, busy_retries=2, busy_backoff_seconds=0.001)
    with pytest.raises(RateLimitError):
        asyncio.run(runtime.call(factory))
