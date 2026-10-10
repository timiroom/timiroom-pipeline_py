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


def test_manager_aligns_fields_with_declared_table_and_preserves_explicit_route():
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
    assert "wrong" not in endpoint["requestBody"]


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


def test_feature_spec_publishes_semantics_from_final_registry_without_changing_owner():
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
    # Missing client invokes the documented deterministic fallback, with no provider call.
    result = asyncio.run(FeatureSpecAgent(object(), "test").execute(state))
    spec = result.feature_specs[0]
    assert spec["featureId"] == "document.create"
    assert spec["ownership"]["ownerEntity"] == "members"
    assert spec["ownership"]["ownerKey"] == "member_id"
    assert spec["apiContract"][0]["path"] == "/api/v1/documents"
    assert spec == result.feature_registry[0]
    assert json.loads(result.feature_spec_document)["features"][0] == spec


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
