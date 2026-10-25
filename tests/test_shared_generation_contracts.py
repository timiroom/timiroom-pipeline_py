import copy

import pytest

from phase2.agents.api_agent import (
    _normalize_endpoints,
    reconcile_auth_session_contracts,
)
from phase2.agents.dba_agent import project_worker_columns, registry_schema_fields
from phase2.feature_generation import (
    normalize_parameter_maps,
    normalize_response_array_maps,
)
from phase2.operation_contract import (
    bind_auth_registry_policy,
    bind_operation_contract,
    effect_schema_issues,
    operation_schema_issues,
    session_policy_issues,
)
from phase2.registry_contract_guard import shared_db_issues
from phase2.repair_scope import repair_targets
from phase2.state import PipelineState


def registry():
    return [
        {"featureId": "catalogs.create", "dbContract": {"primaryTable": "catalogs", "tables": ["catalogs"],
         "fields": [{"table": "catalogs", "name": "id", "type": "UUID", "generated": True},
                    {"table": "catalogs", "name": "code", "type": "TEXT"}], "foreignKeys": []}},
        {"featureId": "entries.create", "dbContract": {"primaryTable": "entries", "tables": ["entries"],
         "fields": [{"table": "entries", "name": "id", "type": "UUID", "generated": True},
                    {"table": "entries", "name": "catalog_id", "type": "UUID"}],
         "foreignKeys": [{"table": "entries", "column": "catalog_id", "references": {"table": "catalogs", "column": "id"}}]}},
    ]


def test_shared_fk_valid_without_redeclaring_target_table_in_every_feature():
    assert not shared_db_issues(registry())


def test_bad_reference_is_scoped_and_never_auto_adds_unique():
    features = registry()
    features[1]["dbContract"]["foreignKeys"][0]["references"]["column"] = "code"
    baseline = copy.deepcopy(features)
    issues = shared_db_issues(features)
    assert any("UNIQUE_REQUIRED" in issue and "/features/1/" in issue for issue in issues)
    assert features == baseline


def test_shared_columns_cannot_disagree_and_fk_source_must_be_declared():
    features = registry()
    features[1]["dbContract"]["fields"][1]["type"] = "BIGINT"
    assert any("TYPE_MISMATCH" in issue for issue in shared_db_issues(features))
    features[1]["dbContract"]["fields"].pop()
    assert any("SOURCE_INVALID" in issue for issue in shared_db_issues(features))


def test_unique_declared_by_later_feature_is_visible_to_earlier_fk():
    features = registry()
    features[1]["dbContract"]["foreignKeys"][0]["references"]["column"] = "code"
    features[1]["dbContract"]["fields"][1]["type"] = "TEXT"
    features[0]["dbContract"]["uniqueConstraints"] = [{"table": "catalogs", "columns": ["code"]}]
    assert not shared_db_issues(list(reversed(features)))


def test_shared_field_conflict_names_every_affected_feature():
    features = registry()
    features[1]["dbContract"]["fields"].append({"table": "catalogs", "name": "id", "type": "BIGINT"})
    issues = shared_db_issues(features)
    assert any("FIELD_CONFLICT" in issue and "/features/0/" in issue for issue in issues)
    assert any("FIELD_CONFLICT" in issue and "/features/1/" in issue for issue in issues)


def test_invalid_field_identifiers_are_blockers_not_python_type_errors():
    features = registry()
    features[0]["dbContract"]["fields"].append({"table": {}, "name": [], "type": "TEXT"})
    assert any("FIELD_SHAPE" in issue for issue in shared_db_issues(features))


def auth_features():
    return [{"featureId": "auth.login", "sessionPolicy": {"strategy": "bearer_session", "sessionTable": "sessions"},
             "apiContract": [{"method": "POST", "path": "/api/v1/auth/login"}],
             "dbContract": {"tables": ["sessions", "users"], "fields": [
                 {"table": "sessions", "name": "token_hash", "type": "TEXT"},
                 {"table": "sessions", "name": "expires_at", "type": "TIMESTAMPTZ"},
                 {"table": "users", "name": "id", "type": "UUID", "generated": True}]}}]


def test_declared_session_store_is_not_replaced_or_duplicated():
    features = auth_features()
    bind_auth_registry_policy(features)
    assert set(features[0]["dbContract"]["tables"]) == {"sessions", "users"}
    assert features[0]["sessionPolicy"]["sessionTable"] == "sessions"
    assert next(field for field in features[0]["dbContract"]["fields"] if field["name"] == "user_id")["type"] == "UUID"
    before = copy.deepcopy(features)
    bind_auth_registry_policy(features)
    assert features == before


def test_empty_feature_policy_does_not_hide_explicit_operation_policy():
    features = auth_features()
    features[0]["apiContract"][0]["sessionPolicy"] = features[0].pop("sessionPolicy")
    features[0]["sessionPolicy"] = {}
    bind_auth_registry_policy(features)
    assert features[0]["sessionPolicy"] == {"strategy": "bearer_session", "sessionTable": "sessions"}


def test_shared_guard_checks_http_contracts_before_downstream_generation():
    features = registry()
    features[0]["apiContract"] = [{"method": "GET", "path": "/api/v1/catalogs", "requestSchema": {"type": "object"}},
                                  {"method": "DELETE", "path": "/api/v1/catalogs/{id}", "successStatus": 204,
                                   "responseSchema": {"type": "object"}}]
    issues = shared_db_issues(features)
    assert any("API_BODYLESS_SCHEMA" in issue and "/features/0/apiContract/0" in issue for issue in issues)
    assert any("API_NO_CONTENT_RESPONSE_BODY" in issue and "/features/0/apiContract/1" in issue for issue in issues)


def test_shared_create_guard_preserves_server_sources_and_database_defaults():
    features = registry()
    fields = features[1]["dbContract"]["fields"]
    fields[1].update(required=True, source="auth")
    fields.append({"table": "entries", "name": "status", "type": "TEXT", "required": True, "default": "open"})
    features[1]["apiContract"] = [{"method": "POST", "path": "/api/v1/entries", "action": "create"}]
    assert not shared_db_issues(features)
    fields[1]["source"] = "request"
    assert any("API_CREATE_STORAGE_VALUE_MISSING: entries.catalog_id" in issue for issue in shared_db_issues(features))


def test_conflicting_session_stores_are_rejected_without_silent_rewrite():
    features = auth_features()
    other = copy.deepcopy(features[0])
    other["sessionPolicy"]["sessionTable"] = "other_sessions"
    features.append(other)
    before = copy.deepcopy(features)
    assert session_policy_issues(features)
    bind_auth_registry_policy(features)
    assert features == before


def test_worker_cannot_add_uncontracted_state_changes():
    operation = {"method": "POST", "path": "/api/v1/entries"}
    endpoint = {**operation, "effects": ["mark parent complete"], "transactionRules": "Atomic insert"}
    result = bind_operation_contract(endpoint, {"apiContract": [operation]})
    assert result["effects"] == []
    assert result["generationWarnings"] == ["API_UNDECLARED_EFFECTS_IGNORED"]
    assert result["transactionRules"] == "Atomic insert"


def test_auth_reconciliation_preserves_explicit_logout_input_contract():
    features = auth_features()
    operation = {"method": "POST", "path": "/api/v1/auth/logout", "requestSchema": {
        "type": "object", "properties": {"csrfToken": {"type": "string"}}, "required": ["csrfToken"]}}
    features[0]["apiContract"] = [operation]
    endpoint = bind_operation_contract({"method": "POST", "path": operation["path"]}, features[0])
    assert reconcile_auth_session_contracts([endpoint], features)[0]["requestSchema"] == operation["requestSchema"]


def test_effects_require_real_tables_columns_and_kind():
    tables = {"entries": {"columns": [{"name": "status"}]}}
    assert not effect_schema_issues({"effects": [{"table": "entries", "column": "status", "kind": "update"}]}, tables)
    assert effect_schema_issues({"effects": [{"table": "entries", "column": "unknown", "kind": "update"}]}, tables)


def test_row_effects_and_explicit_state_transition_use_same_rules_at_both_boundaries():
    tables = {"entries": {"columns": [{"name": "status"}]}}
    for effect in [{"table": "entries", "kind": "insert"},
                   {"table": "entries", "kind": "delete"},
                   {"table": "entries", "column": "status", "kind": "state_transition", "from": "open", "to": "closed"}]:
        endpoint = {"effects": [effect]}
        assert not effect_schema_issues(endpoint, tables)
        assert not operation_schema_issues(endpoint, tables)


def test_null_body_schema_is_absent_only_for_bodyless_operations():
    normalized = _normalize_endpoints([
        {"method": "GET", "path": "/api/v1/entries", "requestSchema": None},
        {"method": "POST", "path": "/api/v1/entries", "requestSchema": None},
        {"method": "GET", "path": "/api/v1/catalogs", "requestSchema": {"type": "object"}},
    ])
    assert "requestSchema" not in next(ep for ep in normalized if ep["method"] == "GET" and ep["path"].endswith("entries"))
    assert next(ep for ep in normalized if ep["method"] == "POST")["requestSchema"] is None
    assert next(ep for ep in normalized if ep["path"].endswith("catalogs"))["requestSchema"] == {"type": "object"}


def test_shape_pointer_is_never_parsed_as_a_new_endpoint_path():
    state = PipelineState(qa_blocker_details=[{"domain": "api", "artifactKey": "GET /api/v1/entries", "pointer": "/endpoints/0/requestSchema"}])
    current = {"endpoints": [{"method": "GET", "path": "/api/v1/entries"}]}
    assert repair_targets(state, "GET /api/v1/entries/endpoints/0/requestSchema: invalid", "api", current) == {"GET /api/v1/entries"}


def test_initial_worker_cannot_invent_required_inputs_or_alias_tables():
    features = registry()
    assert set(registry_schema_fields(features)) == {"catalogs", "entries"}
    table = {"name": "entries", "columns": [{"name": "catalog_id"}, {"name": "id"}, {"name": "created_at"},
                                               {"name": "uncontracted_required_input", "constraints": "NOT_NULL"}]}
    projected = project_worker_columns(table, features)
    assert [field["name"] for field in projected["columns"]] == ["catalog_id", "id", "created_at"]
    assert projected["generationWarnings"]


def test_incomplete_legacy_contracts_do_not_trim_worker_columns():
    features = registry()
    features[0]["dbContract"].pop("fields")
    table = {"name": "entries", "columns": [{"name": "real_legacy_field"}]}
    assert not registry_schema_fields(features)
    assert project_worker_columns(copy.deepcopy(table), features) == table


def test_valid_parameter_maps_convert_without_losing_schema_or_required_flags():
    features = [{"apiContract": [{"parameters": {
        "path": {"entryId": {"type": "string", "format": "uuid"}},
        "query": {"state": {"schema": {"type": "string", "enum": ["open", "closed"]}, "required": True}},
    }}]}]
    baseline = copy.deepcopy(features)
    normalized, changes = normalize_parameter_maps(features)
    parameters = normalized[0]["apiContract"][0]["parameters"]
    assert parameters == [{"in": "path", "name": "entryId", "schema": {"type": "string", "format": "uuid"}, "required": True},
                          {"in": "query", "name": "state", "schema": {"type": "string", "enum": ["open", "closed"]}, "required": True}]
    assert changes == ["/features/0/apiContract/0/parameters"]
    assert features == baseline


def test_ambiguous_parameter_maps_are_not_silently_dropped_or_guessed():
    features = [{"apiContract": [{"parameters": {"unknown": {"id": {"type": "string"}}}}]}]
    normalized, changes = normalize_parameter_maps(features)
    assert normalized == features and not changes


def test_response_array_wrapper_normalization_preserves_targets_and_input():
    features = [{"apiContract": [{"responseSchema": {"type": "object", "properties": {
        "entries": {"type": "array", "items": {"type": "object", "properties": {"id": {"type": "integer"}}}},
    }}, "fieldMappings": [{"source": "response", "field": "[].id", "table": "entries", "column": "id"}]}]}]
    baseline = copy.deepcopy(features)
    normalized, changes = normalize_response_array_maps(features)
    mapping = normalized[0]["apiContract"][0]["fieldMappings"][0]
    assert mapping == {"source": "response", "field": "entries[].id", "table": "entries", "column": "id"}
    assert features == baseline
    assert changes[0]["before"] == "[].id" and changes[0]["after"] == "entries[].id"
    assert normalize_response_array_maps(normalized) == (normalized, [])


@pytest.mark.parametrize("case", ["ambiguous", "missing", "body", "pointer", "root"])
def test_response_array_wrapper_is_not_guessed_for_ambiguous_or_other_locations(case):
    child = {"type": "array", "items": {"type": "object", "properties": {"id": {"type": "integer"}}}}
    schema = {"type": "object", "properties": {"entries": child}}
    mapping = {"source": "response", "field": "[].id", "table": "entries", "column": "id"}
    if case == "ambiguous":
        schema["properties"]["others"] = copy.deepcopy(child)
    elif case == "missing":
        mapping["field"] = "[].absent"
    elif case == "body":
        mapping["source"] = "body"
    elif case == "pointer":
        mapping["pointer"] = "/explicit/path"
    else:
        schema = child
    features = [{"apiContract": [{"responseSchema": schema, "fieldMappings": [mapping]}]}]
    assert normalize_response_array_maps(features) == (features, [])
