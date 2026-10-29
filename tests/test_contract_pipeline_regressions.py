import asyncio
import json

import pytest

from phase2.agents.api_agent import ApiAgent


@pytest.mark.parametrize("path", [
    "/api/v0/auth/login", "/api/v1/api/v0/auth/login", "/api/v1/v0/auth/login",
])
def test_manager_review_canonicalizes_version_without_losing_registry_identity(path):
    agent = object.__new__(ApiAgent)
    result = asyncio.run(agent._manager_review_node({
        "ctx": {"feature_registry": json.dumps([{
            "featureId": "auth.login", "name": "Login", "actions": ["login"],
            "apiContract": [{"method": "POST", "path": "/api/v1/auth/login", "action": "login"}],
        }])},
        "endpoints": [{"method": "POST", "path": path, "featureId": "auth.login", "action": "login"}],
        "authentication": "JWT",
    }))
    endpoints = json.loads(result["api_spec"])["endpoints"]
    assert len(endpoints) == 1
    assert endpoints[0]["path"] == "/api/v1/auth/login"
    assert endpoints[0]["featureId"] == "auth.login"
    assert endpoints[0]["action"] == "login"


def test_manager_preserves_route_and_unmapped_dto_for_qa_instead_of_erasing_it():
    registry = [{
        "featureId": "document.create", "name": "Document create", "actions": ["create"],
        "apiContract": [{"method": "POST", "path": "/api/v1/documents", "action": "create"}],
        "dbContract": {"tables": ["work_items"], "foreignKeys": []},
    }]
    schema = json.dumps({"tables": [{
        "name": "work_items", "description": "Document storage",
        "columns": [
            {"name": "id", "type": "BIGINT", "constraints": "PRIMARY_KEY"},
            {"name": "title", "type": "TEXT", "constraints": "NOT_NULL"},
        ],
    }]})
    result = asyncio.run(object.__new__(ApiAgent)._manager_review_node({
        "ctx": {"feature_registry": json.dumps(registry), "db_schema": schema},
        "endpoints": [{"method": "POST", "path": "/api/v1/documents", "requestBody": "wrong: integer"}],
        "authentication": "",
    }))
    endpoint = json.loads(result["api_spec"])["endpoints"][0]
    assert endpoint["path"] == "/api/v1/documents"
    assert endpoint["featureId"] == "document.create"
    assert endpoint["action"] == "create"
    assert "title: string" in endpoint["requestBody"]
    # Unknown DTO fields require an explicit mapping or QA repair, not silent deletion.
    assert "wrong: integer" in endpoint["requestBody"]


def test_dba_manager_normalizes_postgresql_before_emitting_schema():
    from phase2.agents.dba_agent import DbaAgent
    result = asyncio.run(object.__new__(DbaAgent)._manager_review_node({
        "ctx": {"feature_list": ["업무 등록"], "feature_registry": "[]",
                "prd_document": json.dumps({"coreFeatures": [{"name": "업무 등록", "description": "제목과 마감일을 입력한다"}]}, ensure_ascii=False)},
        "tables": [{"name": "documents", "description": "업무 등록", "columns": [
            {"name": "id", "type": "BIGINT", "constraints": "PRIMARY_KEY AUTO_INCREMENT"},
            {"name": "created_at", "type": "DATETIME", "constraints": "NOT_NULL ON UPDATE CURRENT_TIMESTAMP"},
        ]}],
        "relationships": [],
    }))
    columns = {column["name"]: column for column in result["final_tables"][0]["columns"]}
    assert columns["created_at"]["type"] == "TIMESTAMPTZ"
    assert "ON UPDATE" not in columns["created_at"]["constraints"]
    assert "GENERATED_IDENTITY" in columns["id"]["constraints"]
    assert {"title", "due_date"} <= set(columns)


def test_feature_spec_failure_preserves_owner_without_publishing_fallback():
    from phase2.agents.feature_spec_agent import FeatureSpecAgent
    from phase2.state import PipelineState
    state = PipelineState(
        feature_list=["문서 등록"],
        feature_registry=[{
            "featureId": "document.create", "name": "문서 등록", "actions": ["create"],
            "ownership": {"scope": "USER", "ownerEntity": "members", "ownerKey": "member_id"},
            "apiContract": [{"method": "POST", "path": "/api/v0/documents", "action": "create"}],
            "dbContract": {"tables": ["documents"], "foreignKeys": []},
        }],
        prd_document=json.dumps({"coreFeatures": [{"name": "문서 등록", "requirements": ["문서를 저장한다"]}]}, ensure_ascii=False),
    )
    # Missing client must not invent a replacement contract.
    result = asyncio.run(FeatureSpecAgent(object(), "test").execute(state))
    spec = result.feature_registry[0]
    assert spec["featureId"] == "document.create"
    assert spec["ownership"]["ownerEntity"] == "members"
    assert spec["ownership"]["ownerKey"] == "member_id"
    assert spec["apiContract"][0]["path"] == "/api/v0/documents"
    assert spec == result.feature_registry[0]
    assert not result.feature_spec_document
    assert result.generation_blockers


@pytest.mark.parametrize("feature_name", ["상품 목록 조회", "상품 재고 수량 조회"])
def test_finalizer_keeps_explicit_public_catalog_table(feature_name):
    from phase2.agents.api_agent import finalize_api_contracts
    schema = json.dumps({"tables": [
        {"name": "products", "description": "상품 카탈로그", "columns": [
            {"name": "id", "type": "BIGINT", "constraints": "PRIMARY_KEY"},
            {"name": "name", "type": "TEXT", "constraints": "NOT_NULL"},
        ]},
        {"name": "stock_items", "description": "사용자별 상품 재고 수량", "columns": [
            {"name": "id", "type": "BIGINT", "constraints": "PRIMARY_KEY"},
            {"name": "product_id", "type": "BIGINT", "constraints": "REFERENCES products(id)"},
            {"name": "user_id", "type": "BIGINT", "constraints": "NOT_NULL"},
            {"name": "quantity", "type": "INTEGER", "constraints": "NOT_NULL"},
        ]},
    ]}, ensure_ascii=False)
    endpoint = {"method": "GET", "path": "/api/v1/products", "authRequired": False}
    registry = [{
        "featureId": "product.list", "name": feature_name, "actions": ["read"],
        "apiContract": [{"method": "GET", "path": "/api/v1/products", "action": "read"}],
        "dbContract": {"tables": ["products"], "foreignKeys": []},
    }]
    result = finalize_api_contracts([endpoint], schema, registry=registry)
    assert len(result) == 1
    assert result[0]["path"] == "/api/v1/products"
    assert result[0]["successResponse"].startswith("items: array<products>")
    assert result[0]["authRequired"] is False
    assert result[0]["featureId"] == "product.list"
    assert result[0]["action"] == "read"


def test_finalizer_does_not_infer_personal_stock_for_catalog_read_without_registry():
    from phase2.agents.api_agent import finalize_api_contracts
    schema = json.dumps({"tables": [
        {"name": "products", "description": "상품 목록 조회", "columns": [
            {"name": "id", "type": "BIGINT"}, {"name": "name", "type": "TEXT"},
        ]},
        {"name": "stock_items", "description": "개인 재고 관리", "columns": [
            {"name": "id", "type": "BIGINT"}, {"name": "user_id", "type": "BIGINT"},
            {"name": "product_id", "type": "BIGINT", "constraints": "REFERENCES products(id)"},
            {"name": "quantity", "type": "INTEGER"},
        ]},
    ]}, ensure_ascii=False)
    result = finalize_api_contracts([{
        "method": "GET", "path": "/api/v1/products", "description": "상품 목록 조회", "authRequired": True,
    }], schema)
    assert result[0]["path"] == "/api/v1/products"
    assert result[0]["successResponse"].startswith("items: array<products>")


@pytest.fixture
def favorites_and_repeatable_views():
    def table(name, description, columns):
        return {"name": name, "description": description, "columns": [
            {"name": "id", "type": "BIGINT", "constraints": "PRIMARY_KEY"},
            *[{"name": key, "type": kind, "constraints": constraint} for key, kind, constraint in columns],
        ], "indexes": []}
    pair = [("user_id", "BIGINT", "REFERENCES users(id)"), ("document_id", "BIGINT", "REFERENCES documents(id)")]
    return {"ctx": {
        "feature_list": ["즐겨찾기", "문서 조회 이력"], "feature_registry": "[]",
        "prd_document": json.dumps({"coreFeatures": [
            {"name": "즐겨찾기", "requirements": ["동일 사용자의 동일 문서 중복 즐겨찾기 방지"]},
            {"name": "문서 조회 이력", "requirements": ["같은 회원이 같은 문서를 볼 때마다 새 조회 이력을 저장한다"]},
        ]}, ensure_ascii=False),
    }, "tables": [
        table("users", "회원", []), table("documents", "문서", []),
        table("favorites", "즐겨찾기 중복 방지", pair),
        table("document_views", "문서 조회 이력: 같은 회원이 같은 문서를 볼 때마다 기록", [
            *pair, ("viewed_at", "TIMESTAMPTZ", "NOT_NULL"),
        ]),
    ], "relationships": []}


def test_dba_manager_limits_duplicate_rule_to_matching_feature(favorites_and_repeatable_views):
    from phase2.agents.dba_agent import DbaAgent
    result = asyncio.run(object.__new__(DbaAgent)._manager_review_node(favorites_and_repeatable_views))
    tables = {table["name"]: table for table in result["final_tables"]}
    assert sum("CREATE UNIQUE INDEX" in index for index in tables["favorites"]["indexes"]) == 1
    assert not any("CREATE UNIQUE INDEX" in index for index in tables["document_views"]["indexes"])


def test_dba_manager_does_not_make_views_depend_on_favorites(favorites_and_repeatable_views):
    from phase2.agents.dba_agent import DbaAgent
    result = asyncio.run(object.__new__(DbaAgent)._manager_review_node(favorites_and_repeatable_views))
    views = next(table for table in result["final_tables"] if table["name"] == "document_views")
    assert {column["name"] for column in views["columns"]} == {"id", "user_id", "document_id", "viewed_at"}
    assert not any("REFERENCES favorites(" in column["constraints"] for column in views["columns"])


def test_authorization_header_input_is_implied_for_authenticated_route_without_policy():
    # Live run (cafe-beans): non-auth features listed header.Authorization and
    # Feature Spec failed because only auth features carry a sessionPolicy.
    from phase2.contract_quality import operation_quality_issues
    operation = {"method": "GET", "path": "/api/v1/ingredients", "authRequired": True, "requiredInputs": ["header.Authorization"]}
    assert not any("API_ACCEPTANCE_FIELD_MISSING" in issue for issue in operation_quality_issues(operation))
    operation["sessionPolicy"] = {"transport": "cookie", "cookieName": "sid"}
    assert any("API_ACCEPTANCE_FIELD_MISSING" in issue for issue in operation_quality_issues(operation))


def test_draft_field_may_be_supplied_by_a_header():
    from phase2.feature_draft import DbFieldDraft
    assert DbFieldDraft.model_validate({"name": "idempotency_key", "type": "text", "source": "header"}).source == "header"


def test_null_and_omitted_optional_keys_are_not_contract_drift():
    # Live run (gym-pt): roleProvisioning had method/path null in the contract and
    # the API agent omitted them, which blocked Phase3 as sessionPolicy drift.
    from phase2.operation_contract import _without_nulls
    contract = {"strategy": "bearer_session", "roleProvisioning": [{"role": "trainer", "source": "seed", "method": None, "path": None}]}
    endpoint = {"strategy": "bearer_session", "roleProvisioning": [{"role": "trainer", "source": "seed"}]}
    assert _without_nulls(contract) == _without_nulls(endpoint)
    assert _without_nulls({**endpoint, "strategy": "cookie"}) != _without_nulls(contract)


def test_domain_named_counter_satisfies_decrement_requirement():
    from phase2.agents.qa_agent import QaAgent
    prd = {"coreFeatures": [{"name": "PT 차감", "description": "수업 완료 시 남은 PT 횟수를 차감한다."}]}
    db = {"tables": [
        {"name": "users", "columns": [{"name": "id"}]},
        {"name": "pt_passes", "columns": [{"name": "id"}, {"name": "member_user_id"}, {"name": "remaining_sessions"}],
         "foreignKeys": [{"column": "member_user_id", "references": {"table": "users", "column": "id"}}]},
    ]}
    def issues(api):
        result = QaAgent._check_cross_document_semantics(prd, db, api, [])
        return [item for part in result if isinstance(part, list) for item in part]

    decrement = {"endpoints": [{"method": "POST", "path": "/api/v1/sessions/{id}/complete",
                                "effects": [{"kind": "decrement", "table": "pt_passes", "column": "remaining_sessions"}]}]}
    # The decision follows the API's real mutation, not the column's wording.
    assert not any("상태 변경을 요구하지만" in str(item) for item in issues(decrement))
    assert any("상태 변경을 요구하지만" in str(item) for item in issues({"endpoints": []}))


def test_invite_acceptance_that_opens_a_session_is_a_public_session_entry():
    # Live run (academy-makeup): POST /auth/invites/accept was blocked as an
    # unauthenticated user-owned route although it issues the first session.
    from phase2.operation_contract import is_public_session_entry
    feature = {"sessionPolicy": {"strategy": "bearer_session", "sessionTable": "user_sessions"},
               "dbContract": {"tables": ["users", "user_sessions", "user_invites"]}}
    operation = {"method": "POST", "path": "/api/v1/auth/invites/accept", "action": "join_with_token", "authRequired": False,
                 "requestSchema": {"type": "object"}, "effects": [{"table": "users", "kind": "insert"},
                                                                 {"table": "user_sessions", "kind": "insert"}]}
    assert is_public_session_entry(operation, feature)
    operation["effects"] = [{"table": "users", "kind": "insert"}]
    assert not is_public_session_entry(operation, feature)


def test_list_404_is_allowed_only_for_an_identifier_filter_target():
    from phase2.agents.qa_agent import QaAgent

    def issues(parameters):
        endpoint = {"method": "GET", "path": "/api/v1/stock-events", "action": "list", "errorCodes": "401, 404 not_found",
                    "parameters": parameters, "responseSchema": {"type": "array", "items": {"type": "object"}}}
        result = QaAgent._check_cross_document_semantics({"coreFeatures": []}, {"tables": []}, {"endpoints": [endpoint]}, [])
        return [item for part in result if isinstance(part, list) for item in part]

    assert any("빈 결과에 404" in str(item) for item in issues([{"name": "limit", "in": "query"}]))
    assert not any("빈 결과에 404" in str(item) for item in issues([{"name": "materialId", "in": "query"}]))


def test_identity_table_is_not_required_to_reference_business_root():
    # Live run (cafe-beans): an account/invite feature matched users by name and the
    # association rule demanded users -> materials, which DB repair could not satisfy.
    from phase2.agents.qa_agent import QaAgent
    prd = {"coreFeatures": [{"name": "재료 등록"}, {"name": "직원 계정 지정"}]}
    db = {"tables": [
        {"name": "materials", "description": "재료 등록", "columns": [{"name": "id"}, {"name": "name"}]},
        {"name": "users", "description": "직원 계정 지정", "columns": [{"name": "id"}, {"name": "email"}, {"name": "password_hash"}]},
    ]}
    result = QaAgent._check_cross_document_semantics(prd, db, {"endpoints": []}, ["재료 등록", "직원 계정 지정"])
    issues = [str(item) for part in result if isinstance(part, list) for item in part]
    assert not any("users" in item and ("기능 관계 FK 누락" in item or "배정 주체 FK 누락" in item) for item in issues)


def test_unique_id_column_without_reference_is_not_an_inferred_fk():
    # Live run (cafe-beans): inventory_events.client_request_id (UNIQUE idempotency
    # key) was treated as an FK to a missing client_requests table.
    from phase2.agents.dba_agent import is_own_identifier_column
    from phase2.agents.qa_agent import QaAgent
    assert is_own_identifier_column({"name": "client_request_id", "constraints": "NOT_NULL UNIQUE"})
    assert not is_own_identifier_column({"name": "user_id", "constraints": "NOT_NULL UNIQUE REFERENCES users(id)"})
    assert not is_own_identifier_column({"name": "material_id", "constraints": "NOT_NULL"})
    db = {"tables": [{"name": "inventory_events", "columns": [
        {"name": "id", "type": "UUID", "constraints": "PRIMARY_KEY"},
        {"name": "client_request_id", "type": "UUID", "constraints": "NOT_NULL UNIQUE"}]}]}
    result = QaAgent._check_cross_document_semantics({"coreFeatures": []}, db, {"endpoints": []}, [])
    issues = [str(item) for part in result if isinstance(part, list) for item in part]
    assert not any("client_request_id의 FK 참조 대상" in item for item in issues)


def test_quota_exhaustion_is_not_retried_like_a_busy_gateway():
    # Live run: "insufficient_quota" was retried 5x per call and reported only as
    # RateLimitError, hiding that the account had no credits left.
    import httpx
    import openai

    from phase2.feature_generation import retryable_transport
    from phase2.llm_runtime import LlmRuntime

    request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")

    def error(code):
        response = httpx.Response(429, request=request)
        return openai.RateLimitError("rate", response=response, body={"error": {"code": code, "type": code}})

    assert not retryable_transport(error("insufficient_quota"))
    assert retryable_transport(error("rate_limit_exceeded"))
    calls = []

    async def factory():
        calls.append(1)
        raise error("insufficient_quota")

    runtime = LlmRuntime(1, 5, busy_retries=5, busy_backoff_seconds=0)
    with pytest.raises(openai.RateLimitError):
        asyncio.run(runtime.call(factory))
    assert len(calls) == 1
