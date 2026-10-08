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
