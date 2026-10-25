import asyncio
import copy
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from phase2.agents.api_agent import ApiAgent
from phase2.agents.dba_agent import DbaAgent, _ensure_contract_tables, sanitize_tables
from phase2.agents.feature_spec_agent import FeatureSpecAgent
from phase2.agents.pm_agent import PmAgent
from phase2.agents.prd_agent import PrdAgent
from phase2.contract_guard import feature_contract_issues
from phase2.orchestration_graph import OrchestrationGraph
from phase2.repair_scope import artifact_hash, merge_columns
from phase2.state import PipelineState


def feature():
    return {
        "featureId": "assets.read", "name": "Asset list", "actions": ["read"],
        "description": "Read assets belonging to the current user.",
        "requirements": ["Return the user's saved assets."],
        "apiContract": [{"method": "GET", "path": "/api/v1/assets", "responseSchema": {
            "type": "object", "properties": {"items": {"type": "array", "items": {"type": "object"}}},
        }}],
        "dbContract": {"tables": ["assets"], "primaryTable": "assets", "foreignKeys": [],
                       "fields": [{"table": "assets", "name": "title", "type": "TEXT"}]},
    }


@pytest.mark.parametrize("field,value", [
    ("featureId", "intent.abc1231234"), ("featureId", "feature_001"),
    ("featureId", "de0b724e30s.create"), ("requirements", []),
    ("description", ""), ("dbContract", {"tables": ["assets"]}),
    ("apiContract", [{"method": "GET", "path": "/api/v1/abc1231234"}]),
])
def test_contract_guard_rejects_incomplete_or_placeholder_contracts(field, value):
    item = feature()
    item[field] = value
    assert feature_contract_issues([item])


def test_contract_guard_accepts_explicit_contract():
    assert feature_contract_issues([feature()]) == []


@pytest.mark.parametrize("content", ['{"features":', '{"features":[]}', '{"features":[{"name":"broken"}]}'])
def test_feature_generation_failure_preserves_existing_artifacts(content):
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=AsyncMock(
        return_value=SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content), finish_reason="stop")]),
    ))))
    state = PipelineState(feature_registry=[feature()], db_schema="original-db", api_spec="original-api")
    result = asyncio.run(FeatureSpecAgent(client, "test").execute(state))
    assert result.feature_registry == state.feature_registry
    assert result.db_schema == "original-db" and result.api_spec == "original-api"
    assert result.generation_blockers and result.qa_approved is False


def test_complete_model_contract_is_published_without_synthetic_subfeatures():
    item = feature()
    item.update(source="prd_core", sourceFeatureId="assets.read", parentFeatureId=None,
                priority="P0", reason="Core workflow", applicability={"ui": True, "api": True, "db": True},
                nonApplicabilityReasons={}, dependencies=[])
    item["apiContract"][0].update(action="read", featureId="assets.read", successStatus=200,
                                  contentType="application/json", fieldMappings=[])
    wire_feature = {key: item[key] for key in (
        "featureId", "sourceFeatureId", "source", "parentFeatureId", "name", "description", "requirements",
        "reason", "actions", "priority", "applicability", "nonApplicabilityReasons", "dependencies",
    )}
    wire_feature.update(ownership={"scope": "PUBLIC", "access": "Authenticated catalog readers"},
                        dbTables=["assets"], primaryTable="assets", apiContract=[{
        "method": "GET", "path": "/api/v1/assets", "action": "read", "successStatus": 200, "responseType": "Assets",
    }])
    payload = {"version": "contract-draft-v1", "features": [wire_feature],
               "tables": [{"name": "assets", "fields": [{"name": "title", "type": "TEXT"}]}],
               "schemas": [{"name": "Assets", "nodes": [
                   {"id": "root", "kind": "object"}, {"id": "items", "parent": "root", "name": "items", "kind": "array"},
                   {"id": "item", "parent": "items", "kind": "object"},
               ]}]}
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=AsyncMock(
        return_value=SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(payload)), finish_reason="stop")]),
    ))))
    state = PipelineState(prd_document=json.dumps({"coreFeatures": [item]}))
    result = asyncio.run(FeatureSpecAgent(client, "test").execute(state))
    assert not result.generation_blockers
    assert len(result.feature_registry) == 1
    assert result.feature_registry[0]["featureId"] == "assets.read"
    assert result.feature_registry[0]["dbContract"]["primaryTable"] == "assets"


@pytest.mark.parametrize("key,value", [
    ("description", "new text"), ("requirements", ["New requirement"]),
    ("acceptanceCriteria", ["New criterion"]), ("ownership", {"scope": "PROJECT"}),
    ("states", ["active", "archived"]),
])
def test_requirement_changes_trigger_feature_delta(key, value):
    old = feature()
    new = {**old, key: value}
    assert OrchestrationGraph._changed_prd_feature_ids(
        json.dumps({"coreFeatures": [old]}), json.dumps({"coreFeatures": [new]}),
    ) == {"assets.read"}


def test_registry_delta_removes_deleted_parent_and_children_but_keeps_siblings():
    old = PipelineState(feature_registry=[
        {"featureId": "assets.read"}, {"featureId": "assets.history", "parentFeatureId": "assets.read"},
        {"featureId": "users.read"},
    ])
    candidate = old.copy(feature_registry=[{"featureId": "users.read"}])
    result = OrchestrationGraph._retain_unchanged_registry(old, candidate, {"assets.read"})
    assert [item["featureId"] for item in result.feature_registry] == ["users.read"]


def test_registry_delta_accepts_new_child_and_preserves_unrelated_child():
    old = PipelineState(feature_registry=[{"featureId": "assets.read"}, {"featureId": "users.read"}])
    candidate = old.copy(feature_registry=[
        {"featureId": "assets.read", "description": "new"},
        {"featureId": "assets.history", "parentFeatureId": "assets.read"},
        {"featureId": "users.read", "description": "unauthorized"},
    ])
    result = OrchestrationGraph._retain_unchanged_registry(old, candidate, {"assets.read"})
    assert result.feature_registry[1] == old.feature_registry[1]
    assert result.feature_registry[-1]["featureId"] == "assets.history"


@pytest.mark.parametrize("name", ["oauth2_tokens", "de0b724e30s", "assets_2"])
def test_table_normalization_is_idempotent_and_keeps_valid_digits(name):
    registry = [{"featureId": "assets.read", "dbContract": {"tables": [name]}}]
    tables = [{"name": name, "columns": [{"name": "id", "type": "BIGINT", "constraints": "PRIMARY_KEY"}], "indexes": []}]
    first = _ensure_contract_tables(sanitize_tables(copy.deepcopy(tables)), registry)
    second = _ensure_contract_tables(sanitize_tables(copy.deepcopy(first)), registry)
    assert first == second
    assert [item["name"] for item in second] == [name]


def test_api_patch_rejects_unrelated_endpoint_and_preserves_sibling_exactly():
    agent = object.__new__(ApiAgent)
    target = {"method": "GET", "path": "/api/v1/assets", "description": "old"}
    sibling = {"method": "POST", "path": "/api/v1/assets", "custom": {"nested": [1]}, "description": "preserve"}
    state = PipelineState(api_spec=json.dumps({"endpoints": [target, sibling]}))
    agent._call = AsyncMock(return_value=json.dumps({"patches": [
        {**target, "description": "new"}, {**sibling, "description": "tampered"},
    ]}))
    result = asyncio.run(agent.repair(state, "GET /api/v1/assets description"))
    endpoints = json.loads(result.api_spec)["endpoints"]
    assert endpoints[1] == sibling
    assert endpoints[0]["description"] == "new"


def test_api_stale_hash_and_ambiguous_scope_do_not_apply():
    agent = object.__new__(ApiAgent)
    state = PipelineState(api_spec=json.dumps({"endpoints": [{"method": "GET", "path": "/api/v1/assets"}]}))
    agent._call = AsyncMock(return_value=json.dumps({"baseHash": "wrong", "patches": []}))
    result = asyncio.run(agent.repair(state, "GET /api/v1/assets"))
    assert result.api_spec == state.api_spec
    assert result.targeted_repair_results["api"] == "rejected_revision"
    agent._call.reset_mock()
    result = asyncio.run(agent.repair(state, "please repair"))
    agent._call.assert_not_called()
    assert result.targeted_repair_results["api"] == "rejected_scope"


def test_db_patch_keeps_unmentioned_columns_and_unrelated_tables():
    agent = object.__new__(DbaAgent)
    original = {"tables": [
        {"name": "assets", "columns": [
            {"name": "id", "type": "BIGINT", "constraints": "PRIMARY_KEY"},
            {"name": "title", "type": "TEXT", "constraints": "NOT_NULL"},
        ], "indexes": []},
        {"name": "users", "description": "preserve", "columns": [{"name": "id", "type": "UUID", "constraints": "PRIMARY_KEY"}]},
    ], "relationships": []}
    agent._call = AsyncMock(return_value=json.dumps({"patches": {
        "assets": {"columns": [{"name": "title", "type": "VARCHAR(255)"}, {"name": "id", "type": "TEXT"}]},
        "users": {"description": "tampered"},
    }}))
    state = PipelineState(db_schema=json.dumps(original))
    result = json.loads(asyncio.run(agent.repair(state, "assets.title type incorrect")).db_schema)
    assert result["tables"][1] == original["tables"][1]
    columns = {item["name"]: item for item in result["tables"][0]["columns"]}
    assert columns["id"] == original["tables"][0]["columns"][0]
    assert columns["title"]["type"] == "VARCHAR(255)"
    assert "NOT_NULL" in columns["title"]["constraints"]


def test_column_remove_is_explicit():
    columns = [{"name": "id", "type": "BIGINT"}, {"name": "obsolete", "type": "TEXT"}]
    assert merge_columns(columns, [{"name": "obsolete", "op": "remove"}]) == columns[:1]


def test_prd_patch_preserves_other_sections_and_unmentioned_features():
    agent = object.__new__(PrdAgent)
    original = {"goals": ["old"], "coreFeatures": [
        {"featureId": "assets.read", "name": "Asset list", "description": "old"},
        {"featureId": "users.read", "name": "Users", "description": "preserve"},
    ]}
    agent._call = AsyncMock(return_value=json.dumps({"patches": {
        "goals": ["tampered"], "coreFeatures": [
            {"featureId": "assets.read", "description": "new"},
            {"featureId": "users.read", "description": "tampered"},
        ],
    }}))
    state = PipelineState(prd_document=json.dumps(original), feature_registry=original["coreFeatures"])
    result = json.loads(asyncio.run(agent.repair(state, "assets.read description")).prd_document)
    assert result["goals"] == ["old"]
    assert result["coreFeatures"][0]["description"] == "new"
    assert result["coreFeatures"][1] == original["coreFeatures"][1]


def test_pm_multiple_removals_use_original_indices():
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=AsyncMock(
        return_value=SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps({"patches": [
            {"op": "remove", "index": 0}, {"op": "remove", "index": 1},
        ]})))])
    ))))
    state = PipelineState(feature_list=["Asset read", "Asset history", "Users"])
    result = asyncio.run(PmAgent(client, "test").repair(state, "Asset read, Asset history remove"))
    assert result.feature_list == ["Users"]


def test_hash_is_order_independent():
    assert artifact_hash({"b": 2, "a": 1}) == artifact_hash({"a": 1, "b": 2})


class Progress:
    def send(self, *_args):
        pass


class Agent:
    async def execute(self, state, _dump=None):
        return state

    async def repair(self, state, feedback, _dump=None):
        return state


def test_feature_generation_blocker_prevents_db_api_calls():
    dba, api = SimpleNamespace(execute=AsyncMock()), SimpleNamespace(execute=AsyncMock())
    spec = SimpleNamespace(execute=AsyncMock(return_value=PipelineState(generation_blockers=["FEATURE_SPEC_CONTRACT_BLOCKER: invalid"])))
    graph = OrchestrationGraph(Agent(), Agent(), Agent(), dba, api, Agent(), Progress(), feature_spec_agent=spec)
    result = asyncio.run(graph._run_prd_with_rollback(PipelineState(), None, prd_ready=True))
    assert result.generation_blockers
    dba.execute.assert_not_called()
    api.execute.assert_not_called()


def test_phase3_prd_repair_resyncs_changed_contract_without_full_generation():
    old = feature()
    new = copy.deepcopy(old)
    new["requirements"] = ["Read archived assets too."]
    new["apiContract"][0]["path"] = "/api/v1/assets/archive"
    state = PipelineState(prd_document=json.dumps({"coreFeatures": [old]}), feature_registry=[old],
                          db_schema='{"tables":[]}', api_spec='{"endpoints":[]}',
                          feature_specs=[old])
    prd = SimpleNamespace(repair=AsyncMock(return_value=state.copy(
        prd_document=json.dumps({"coreFeatures": [new]}), db_schema="", api_spec="",
    )))
    spec = SimpleNamespace(execute=AsyncMock(side_effect=lambda value, _dump=None: value.copy(feature_registry=[new], feature_specs=[new])))
    dba = SimpleNamespace(repair=AsyncMock(side_effect=lambda value, _feedback: value))
    api = SimpleNamespace(repair=AsyncMock(side_effect=lambda value, _feedback: value))
    graph = OrchestrationGraph(Agent(), Agent(), prd, dba, api, Agent(), Progress(), feature_spec_agent=spec)
    result = asyncio.run(graph.repair(state, "assets.read requirements", repair_targets=["prd"]))
    spec.execute.assert_awaited_once()
    api.repair.assert_awaited_once()
    dba.repair.assert_not_called()
    assert result.db_schema == state.db_schema
    assert result.api_spec == state.api_spec


def test_resync_timeout_preserves_baseline():
    old = feature()
    state = PipelineState(prd_document=json.dumps({"coreFeatures": [old]}), feature_registry=[old],
                          db_schema='{"tables":[]}', api_spec='{"endpoints":[]}')
    new = copy.deepcopy(old)
    new["apiContract"][0]["path"] = "/api/v1/assets/new"
    spec = SimpleNamespace(execute=AsyncMock(side_effect=lambda value, _dump=None: value.copy(feature_registry=[new])))

    async def slow(*_args):
        await asyncio.sleep(1)

    graph = OrchestrationGraph(Agent(), Agent(), Agent(), Agent(), SimpleNamespace(repair=slow), Agent(), Progress(),
                               feature_spec_agent=spec, api_resync_timeout_seconds=0.01)
    result, domains = asyncio.run(graph._resync_prd(
        state, state.copy(prd_document=json.dumps({"coreFeatures": [new]})), "assets.read", None,
    ))
    assert domains == {"api"}
    assert result.api_spec == state.api_spec and result.qa_api_blockers


def test_deleted_features_are_not_passed_back_to_feature_generation():
    old = feature()
    state = PipelineState(prd_document=json.dumps({"coreFeatures": [old]}), feature_registry=[old],
                          feature_specs=[old], feature_list=[old["name"]], db_schema="db", api_spec="api")
    received = []

    async def generate(value, _dump=None):
        received.append(value)
        return value

    graph = OrchestrationGraph(Agent(), Agent(), Agent(), Agent(), Agent(), Agent(), Progress(),
                               feature_spec_agent=SimpleNamespace(execute=generate))
    result, _ = asyncio.run(graph._resync_prd(
        state, state.copy(prd_document='{"coreFeatures":[]}'), "assets.read remove", None,
    ))
    assert received[0].feature_registry == []
    assert received[0].feature_list == []
    assert result.feature_registry == []


def test_feature_resync_timeout_becomes_blocker_and_keeps_artifacts():
    old = feature()
    state = PipelineState(prd_document=json.dumps({"coreFeatures": [old]}), feature_registry=[old],
                          db_schema="original-db", api_spec="original-api")

    async def slow(*_args):
        await asyncio.sleep(1)

    graph = OrchestrationGraph(Agent(), Agent(), Agent(), Agent(), Agent(), Agent(), Progress(),
                               feature_spec_agent=SimpleNamespace(execute=slow), repair_timeout_seconds=0.01)
    updated = {**old, "requirements": ["changed"]}
    result, _ = asyncio.run(graph._resync_prd(
        state, state.copy(prd_document=json.dumps({"coreFeatures": [updated]})), "assets.read", None,
    ))
    assert result.generation_blockers
    assert result.db_schema == state.db_schema and result.api_spec == state.api_spec
