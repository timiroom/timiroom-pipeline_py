import asyncio
import json
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from phase2.agent_contract import normalize_feature_contract
from phase2.agents.api_agent import (
    ApiAgent,
    api_self_check,
    reconcile_auth_session_contracts,
)
from phase2.agents.dba_agent import project_explicit_schema
from phase2.agents.feature_spec_agent import (
    FeatureSpecAgent,
    _ensure_feature_narrative,
    _score_supporting,
)
from phase2.feature_draft import compile_document, merge_patch, repair_scope
from phase2.feature_generation import generate_document, source_cores
from phase2.llm_runtime import LlmRuntime
from phase2.openapi_spec import build_openapi_document, validate_openapi_document
from phase2.operation_contract import (
    bind_operation_contract,
    cross_contract_issues,
    operation_schema_issues,
    session_policy_issues,
)
from phase2.orchestration_graph import OrchestrationGraph
from phase2.registry_contract_guard import contract_artifacts
from phase2.state import PipelineState
from phase2.workflow_contract import (
    SessionPolicy,
    authentication_contract,
    bind_workflow_errors,
    normalize_boolean_workflows,
    workflow_check_constraints,
    workflow_contract_issues,
    workflow_metadata,
)


def auth_feature(transport="bearer"):
    policy = SessionPolicy(strategy="bearer_session", sessionTable="sessions", transport=transport,
                           csrfProtection="Synchronizer token on state-changing requests",
                           defaultRole="member", roleProvisioning=[{
                               "role": "operator", "source": "seed", "description": "Restricted deployment seed; no public role assignment",
                           }]).model_dump()
    return {
        "featureId": "auth.session", "name": "Account session", "roles": ["member", "operator"],
        "sessionPolicy": policy,
        "dbContract": {"primaryTable": "sessions", "tables": ["sessions"], "fields": [
            {"table": "sessions", "name": "id", "type": "UUID", "primaryKey": True, "generated": True},
            {"table": "sessions", "name": "token_hash", "type": "TEXT", "source": "server"},
            {"table": "sessions", "name": "expires_at", "type": "TIMESTAMPTZ", "source": "server"},
            {"table": "sessions", "name": "revoked_at", "type": "TIMESTAMPTZ", "source": "server"},
        ]},
        "apiContract": [{"method": "POST", "path": "/api/v1/auth/login", "action": "login", "authRequired": False,
                         "successStatus": 200, "requestSchema": {"type": "object", "properties": {"password": {"type": "string"}}},
                         "responseSchema": {"type": "object", "properties": {"accessToken": {"type": "string"}}, "required": ["accessToken"]},
                         "responseHeaders": {"Set-Cookie": "Secure; HttpOnly; SameSite=Lax"} if transport == "cookie" else {},
                         "effects": [{"table": "sessions", "kind": "insert"}], "sessionPolicy": deepcopy(policy)},
                        {"method": "POST", "path": "/api/v1/auth/logout", "action": "logout", "authRequired": True,
                         "successStatus": 204, "responseSchema": {}, "effects": [{"table": "sessions", "column": "revoked_at", "kind": "revoke"}],
                         "sessionPolicy": deepcopy(policy)}],
    }


def allocation_feature():
    def field(table, name, sql_type="UUID", **kwargs):
        return {"table": table, "name": name, "type": sql_type, **kwargs}

    return {
        "featureId": "allocation.manage", "roles": ["member"], "states": ["pending", "cancelled"],
        "dbContract": {"primaryTable": "allocations", "tables": ["allocations", "assets", "windows"], "fields": [
            field("allocations", "id", primaryKey=True, generated=True), field("allocations", "asset_id"),
            field("allocations", "starts_at", "TIMESTAMPTZ"), field("allocations", "ends_at", "TIMESTAMPTZ"),
            field("allocations", "state", "TEXT", default="pending"), field("assets", "id", primaryKey=True, generated=True),
            field("windows", "id", primaryKey=True, generated=True), field("windows", "asset_id"),
            field("windows", "starts_at", "TIMESTAMPTZ"), field("windows", "ends_at", "TIMESTAMPTZ"),
        ], "foreignKeys": []},
        "statePolicies": [{"table": "allocations", "column": "state", "initial": "pending", "states": ["pending", "cancelled"]}],
        "intervalPolicies": [{"table": "allocations", "resourceColumn": "asset_id", "startColumn": "starts_at", "endColumn": "ends_at",
                              "stateColumn": "state", "blockingStates": ["pending"], "blocksOnCreate": True,
                              "availabilityTable": "windows", "availabilityResourceColumn": "asset_id",
                              "availabilityStartColumn": "starts_at", "availabilityEndColumn": "ends_at",
                              "boundary": "half_open", "timeZone": "Asia/Seoul", "lockTable": "assets", "lockColumn": "id",
                              "mutationPolicy": "Lock the asset and reject window changes overlapping existing blocking allocations."}],
        "stateTransitions": [{"entity": "allocations", "from": "pending", "to": "cancelled"}],
        "apiContract": [{"method": "POST", "path": "/api/v1/allocations", "authRequired": True, "effects": [{"table": "allocations", "kind": "insert"}]},
                        {"method": "DELETE", "path": "/api/v1/allocations/{allocationId}", "authRequired": True,
                         "effects": [{"table": "allocations", "column": "state", "kind": "state_transition", "from": "pending", "to": "cancelled"}]}],
    }


def test_boolean_interval_policy_uses_real_boolean_update_and_sql_literals():
    feature = allocation_feature()
    feature["statePolicies"][0].update(initial=True, states=[True, False])
    feature["intervalPolicies"][0]["blockingStates"] = [True]
    feature["stateTransitions"] = []
    for field in feature["dbContract"]["fields"]:
        if field["table"] == "allocations" and field["name"] == "state":
            field.update(type="BOOLEAN", default=True)
    feature["apiContract"][1] = {"method": "PATCH", "path": "/api/v1/allocations/{allocationId}",
        "requestSchema": {"type": "object", "properties": {"active": {"type": "boolean"}}},
        "fieldMappings": [{"source": "body", "field": "active", "table": "allocations", "column": "state"}],
        "effects": [{"table": "allocations", "column": "state", "kind": "update"}]}
    assert not workflow_contract_issues([feature])
    assert "CHECK (state IN (TRUE, FALSE))" in workflow_check_constraints([feature])["allocations"]
    feature["apiContract"][1]["requestSchema"]["properties"]["active"]["const"] = True
    assert any("WORKFLOW_STATE_UNREACHABLE" in value for value in workflow_contract_issues([feature]))
    feature["apiContract"][1]["requestSchema"]["properties"]["active"] = {"type": "string"}
    assert any("WORKFLOW_STATE_UNREACHABLE" in value for value in workflow_contract_issues([feature]))


def test_boolean_set_from_request_and_json_external_keys_are_typed_state_evidence():
    feature = allocation_feature()
    feature["statePolicies"][0].update(initial=True, states=[True, False], externalStates={"false": "Restricted external retirement"})
    feature["intervalPolicies"][0]["blockingStates"] = [True]
    feature["stateTransitions"] = [{"entity": "allocations", "from": True, "to": False}]
    for field in feature["dbContract"]["fields"]:
        if field["table"] == "allocations" and field["name"] == "state":
            field.update(type="BOOLEAN", default=True)
    feature["apiContract"][1] = {"method": "PATCH", "path": "/api/v1/allocations/{allocationId}",
        "requestSchema": {"type": "object", "properties": {"active": {"type": "boolean"}}},
        "fieldMappings": [{"source": "body", "field": "active", "table": "allocations", "column": "state"}],
        "effects": [{"table": "allocations", "column": "state", "kind": "set"}]}
    assert not workflow_contract_issues([feature])


def test_role_specific_consumer_does_not_require_public_member_access_to_operator_lookup():
    feature = {"featureId": "resource.manage", "roles": ["member", "operator"],
               "readDependencies": [{"path": "/api/v1/resources", "responseField": "items[].id", "purpose": "Select the operator target"}],
               "apiContract": [
                   {"method": "GET", "path": "/api/v1/resources/me", "authRequired": True, "allowedRoles": ["member"]},
                   {"method": "GET", "path": "/api/v1/resources", "authRequired": True, "allowedRoles": ["operator"],
                    "responseSchema": {"type": "object", "properties": {"items": {"type": "array", "items": {"type": "object", "properties": {"id": {"type": "integer"}}}}}}},
                   {"method": "POST", "path": "/api/v1/resources/{resourceId}/status", "authRequired": True, "allowedRoles": ["operator"]},
               ]}
    assert not workflow_contract_issues([feature])
    feature["apiContract"][2]["allowedRoles"] = ["member"]
    assert any("AUTH_LOOKUP_ACCESS_DENIED" in value for value in workflow_contract_issues([feature]))


def test_boolean_policy_normalization_requires_a_declared_boolean_column():
    feature = allocation_feature()
    feature["statePolicies"][0].update(initial="true", states=["true", "false", "unexpected"])
    feature["intervalPolicies"][0]["blockingStates"] = ["true"]
    assert normalize_boolean_workflows([feature]) == ([feature], [])
    for field in feature["dbContract"]["fields"]:
        if field["table"] == "allocations" and field["name"] == "state":
            field.update(type="BOOLEAN", default=True)
    normalized, changes = normalize_boolean_workflows([feature])
    assert normalized[0]["statePolicies"][0]["states"] == [True, False, "unexpected"]
    assert normalized[0]["intervalPolicies"][0]["blockingStates"] == [True]
    assert feature["statePolicies"][0]["initial"] == "true" and len(changes) == 3
    assert normalize_boolean_workflows(normalized) == (normalized, [])


def lookup_draft():
    return {"version": "contract-draft-v1", "tables": [{"name": "assets", "fields": [
        {"name": "id", "type": "UUID", "primaryKey": True, "generated": True, "source": "server"},
    ]}], "schemas": [{"name": "Asset", "nodes": [{"id": "root", "kind": "object"},
                                                     {"id": "id", "parent": "root", "name": "id", "kind": "string", "format": "uuid", "required": True}]}],
        "features": [{"featureId": "asset.manage", "sourceFeatureId": "source.asset", "source": "prd_core", "parentFeatureId": None,
                      "name": "Asset management", "description": "Create an asset and discover its identifier.",
                      "requirements": ["Users can create and select assets."], "reason": "Core workflow", "actions": ["create"], "priority": "P0",
                      "ownership": {"scope": "PUBLIC", "access": "Public asset creation and selection"},
                      "applicability": {"ui": True, "api": True, "db": True}, "dbTables": ["assets"], "primaryTable": "assets",
                      "readDependencies": [{"path": "/api/v1/assets", "responseField": "[].id", "purpose": "Select an asset identifier"}],
                      "apiContract": [{"method": "POST", "path": "/api/v1/assets", "action": "create", "authRequired": False,
                                       "successStatus": 201, "responseType": "Asset", "effects": [{"table": "assets", "kind": "insert"}],
                                       "transactionRules": "Insert atomically and roll back on failure."}]}]}


def lookup_patch(path="/api/v1/assets"):
    return {"features": [{"featureId": "asset.manage", "apiContract": [
        {"method": "GET", "path": path, "action": "list", "authRequired": False, "successStatus": 200, "responseType": "AssetList"},
    ]}], "schemas": [{"name": "AssetList", "nodes": [{"id": "root", "kind": "array"},
                                                        {"id": "entry", "parent": "root", "kind": "ref", "ref": "Asset"}]}]}


def test_system_scope_retains_explicit_authorization_and_never_defaults_to_public():
    owner = {"scope": "SYSTEM", "access": "Authenticated operators only"}
    assert normalize_feature_contract({"ownership": owner}, True)["ownership"]["access"] == owner["access"]
    assert "공개" not in normalize_feature_contract({"ownership": {"scope": "SYSTEM"}}, True)["ownership"]["access"]
    item = {"ownership": {"scope": "SYSTEM", "access": "public access"}, "apiContract": [{"authRequired": True}]}
    assert any("AUTH_ACCESS_POLICY_CONFLICT" in value for value in workflow_contract_issues([item]))


def test_validated_workflow_narrative_is_not_overwritten_by_earlier_prd_prose():
    item = {"name": "Resource update", "source": "prd_core", "workflowContractVersion": 1,
            "description": "Update the current owned resource", "requirements": ["Only pending resources may be updated"]}
    before = deepcopy(item)
    prd = {"description": "Earlier unresolved policy", "requirements": ["Confirmed resources may also be updated"]}
    assert _ensure_feature_narrative(item, prd=prd) == before
    legacy = {"name": "Legacy", "source": "prd_core"}
    assert "Confirmed resources may also be updated" in _ensure_feature_narrative(legacy, prd=prd)["requirements"]


def test_validated_supporting_priority_is_not_demoted_by_keyword_scoring():
    feature = {"workflowContractVersion": 1, "priority": "P1"}
    assert _score_supporting("Resource window management", "Supports the main workflow", feature=feature) == ("P1", 10, True)
    feature["priority"] = "P2"
    assert _score_supporting("Resource reports", "Optional summary", feature=feature) == ("P2", 6, False)


@pytest.mark.parametrize("required", [False, True])
def test_token_delivery_requires_a_real_required_response_field(required):
    feature = auth_feature()
    response = feature["apiContract"][0]["responseSchema"]
    response["properties"].clear()
    response["required"] = []
    if required:
        response["properties"]["accessToken"] = {"type": "string"}
    assert any("AUTH_TOKEN_DELIVERY_MISSING" in value for value in workflow_contract_issues([feature]))
    assert not workflow_contract_issues([auth_feature()])


def test_actual_endpoint_token_drift_is_checked_instead_of_trusting_registry():
    feature = auth_feature()
    endpoints = deepcopy(feature["apiContract"])
    endpoints[0]["responseSchema"] = {"type": "object"}
    assert any("AUTH_TOKEN_DELIVERY_MISSING" in value for value in workflow_contract_issues([feature], endpoints))


def test_cookie_delivery_requires_header_and_explicit_csrf_policy():
    feature = auth_feature("cookie")
    assert not workflow_contract_issues([feature])
    feature["apiContract"][0]["responseHeaders"] = {}
    feature["sessionPolicy"]["csrfProtection"] = ""
    issues = workflow_contract_issues([feature])
    assert any("AUTH_COOKIE_DELIVERY_MISSING" in value for value in issues)
    assert any("AUTH_COOKIE_POLICY_REQUIRED" in value for value in issues)


@pytest.mark.parametrize("transport,source,field", [("bearer", "header", "Authorization"), ("bearer", "header", "authorization"),
                                                   ("cookie", "cookie", "session")])
def test_explicit_auth_transport_is_a_real_input_without_duplicate_parameter(transport, source, field):
    endpoint = {"method": "POST", "path": "/api/v1/auth/logout", "authRequired": True,
                "sessionPolicy": auth_feature(transport)["sessionPolicy"], "parameters": [],
                "fieldMappings": [{"source": source, "field": field, "table": "sessions", "column": "token_hash", "transform": "Hash the received credential"}]}
    tables = {"sessions": {"columns": [{"name": "token_hash", "type": "TEXT"}]}}
    assert not operation_schema_issues(endpoint, tables)
    endpoint["authRequired"] = False
    assert any("API_FIELD_MAPPING_INPUT_UNKNOWN" in value for value in operation_schema_issues(endpoint, tables))
    endpoint["authRequired"] = True
    endpoint["fieldMappings"][0]["field"] = "unlistedCredential"
    assert any("API_FIELD_MAPPING_INPUT_UNKNOWN" in value for value in operation_schema_issues(endpoint, tables))
    endpoint["fieldMappings"][0]["field"] = field
    endpoint.pop("sessionPolicy")
    assert any("API_FIELD_MAPPING_INPUT_UNKNOWN" in value for value in operation_schema_issues(endpoint, tables))


def test_role_supply_is_explicit_and_public_signup_cannot_choose_operator():
    feature = auth_feature()
    feature["sessionPolicy"]["roleProvisioning"] = []
    assert any("AUTH_ROLE_PROVISION_MISSING" in value for value in workflow_contract_issues([feature]))
    feature = auth_feature()
    feature["apiContract"][0]["requestSchema"]["properties"]["role"] = {"type": "string"}
    assert any("AUTH_SELF_ASSIGNMENT_FORBIDDEN" in value for value in workflow_contract_issues([feature]))
    feature["apiContract"][0]["requestSchema"]["properties"]["role"]["const"] = "member"
    assert not workflow_contract_issues([feature])


@pytest.mark.parametrize("source", ["seed", "external", "invite"])
def test_principal_role_state_uses_explicit_account_supply_only(source):
    feature = auth_feature()
    feature["dbContract"]["fields"].append({"table": "accounts", "name": "role", "type": "TEXT", "default": "member"})
    feature["dbContract"]["foreignKeys"] = [{"table": "sessions", "column": "account_id", "references": {"table": "accounts", "column": "id"}}]
    feature["statePolicies"] = [{"table": "accounts", "column": "role", "initial": "member", "states": ["member", "operator"]}]
    supply = feature["sessionPolicy"]["roleProvisioning"][0]
    supply.update(source=source, method="POST", path="/api/v1/invitations")
    if source == "invite":
        assert any("WORKFLOW_STATE_UNREACHABLE" in value for value in workflow_contract_issues([feature]))
        feature["apiContract"].append({"method": "POST", "path": "/api/v1/invitations", "authRequired": True})
    assert not workflow_contract_issues([feature])
    feature["dbContract"]["foreignKeys"][0]["references"]["table"] = "unrelated"
    assert any("WORKFLOW_STATE_UNREACHABLE" in value for value in workflow_contract_issues([feature]))


def test_invitation_supply_requires_its_actual_route():
    feature = auth_feature()
    feature["sessionPolicy"]["roleProvisioning"] = [{"role": "operator", "source": "invite", "description": "PM invitation",
                                                     "method": "POST", "path": "/api/v1/invitations"}]
    assert any("WORKFLOW_LOOKUP_MISSING" in value for value in workflow_contract_issues([feature]))
    feature["apiContract"].append({"method": "POST", "path": "/api/v1/invitations", "authRequired": True})
    assert not workflow_contract_issues([feature])


def test_mixed_transport_declarations_are_not_silently_reconciled():
    feature = auth_feature()
    feature["apiContract"][0]["sessionPolicy"]["transport"] = "cookie"
    assert session_policy_issues([feature])


def test_transport_conflicts_are_checked_for_non_auth_named_routes_too():
    feature = auth_feature()
    feature["apiContract"][0]["path"] = "/api/v1/identity/sessions"
    feature["apiContract"][1]["path"] = "/api/v1/identity/current"
    feature["apiContract"][0]["sessionPolicy"]["transport"] = "cookie"
    assert session_policy_issues([feature])


@pytest.mark.parametrize("transport,format_name", [("bearer", "opaque"), ("bearer", "JWT"), ("cookie", "opaque")])
def test_openapi_and_authentication_description_consume_the_same_transport(transport, format_name):
    feature = auth_feature(transport)
    feature["sessionPolicy"]["tokenFormat"] = format_name
    internal = {"endpoints": feature["apiContract"], **authentication_contract([feature])}
    document = build_openapi_document(internal)
    assert not validate_openapi_document(document)
    operation = document["paths"]["/api/v1/auth/logout"]["post"]
    if transport == "cookie":
        assert operation["security"] == [{"cookieAuth": []}]
        assert document["components"]["securitySchemes"]["cookieAuth"]["name"] == "session"
        assert "Set-Cookie" in document["paths"]["/api/v1/auth/login"]["post"]["responses"]["200"]["headers"]
        assert "JWT" not in internal["authentication"]
    else:
        assert operation["security"] == [{"bearerAuth": []}]
        scheme = document["components"]["securitySchemes"]["bearerAuth"]
        assert scheme.get("bearerFormat") == ("JWT" if format_name == "JWT" else None)
        assert format_name in internal["authentication"]


def test_typed_transport_reconciliation_preserves_declared_logout_and_headers():
    feature = auth_feature("cookie")
    expected = deepcopy(feature["apiContract"])
    assert reconcile_auth_session_contracts(deepcopy(expected), [feature]) == expected
    endpoint = {"method": "POST", "path": "/api/v1/auth/login"}
    bind_operation_contract(endpoint, feature)
    assert endpoint["responseHeaders"] == expected[0]["responseHeaders"]


def test_initial_state_and_unreachable_confirmation_are_checked_generically():
    feature = allocation_feature()
    assert not workflow_contract_issues([feature])
    feature["statePolicies"][0]["states"].append("confirmed")
    assert any("WORKFLOW_STATE_UNREACHABLE" in value for value in workflow_contract_issues([feature]))
    feature["apiContract"].append({"method": "PATCH", "path": "/api/v1/allocations/{id}/confirmation", "effects": [
        {"table": "allocations", "column": "state", "kind": "state_transition", "from": "pending", "to": "confirmed"}]})
    assert not workflow_contract_issues([feature])
    feature["statePolicies"][0]["initial"] = "confirmed"
    assert any("WORKFLOW_STATE_INITIAL" in value for value in workflow_contract_issues([feature]))


def test_declared_cancel_transition_must_have_its_effect_not_only_a_sentence():
    feature = allocation_feature()
    feature["statePolicies"][0]["states"].append("confirmed")
    feature["statePolicies"][0]["externalStates"] = {"confirmed": "Confirmed by an explicitly integrated external approval system"}
    feature["stateTransitions"].append({"entity": "allocations", "from": "confirmed", "to": "cancelled"})
    assert any("WORKFLOW_TRANSITION_MISSING" in value for value in workflow_contract_issues([feature]))
    feature["apiContract"][1]["effects"].append({"table": "allocations", "column": "state", "kind": "state_transition", "from": "confirmed", "to": "cancelled"})
    assert not workflow_contract_issues([feature])


def test_response_enum_cannot_promise_a_state_outside_the_state_policy():
    feature = allocation_feature()
    operation = feature["apiContract"][0]
    operation["responseSchema"] = {"type": "object", "properties": {"status": {"type": "string", "enum": ["pending", "confirmed"]}}}
    operation["fieldMappings"] = [{"field": "status", "table": "allocations", "column": "state", "source": "response"}]
    assert any("WORKFLOW_STATE_ENUM_CONFLICT" in value for value in workflow_contract_issues([feature]))


def test_omitting_transition_entity_does_not_bypass_primary_table_effect_check():
    feature = allocation_feature()
    feature["stateTransitions"] = [{"from": "pending", "to": "cancelled"}]
    feature["apiContract"][1]["effects"] = []
    assert any("WORKFLOW_TRANSITION_MISSING" in value for value in workflow_contract_issues([feature]))


def test_request_enum_update_is_a_real_state_producer():
    feature = allocation_feature()
    feature["statePolicies"][0]["states"].append("confirmed")
    feature["apiContract"].append({"method": "PATCH", "path": "/api/v1/allocations/{id}",
                                   "requestSchema": {"type": "object", "properties": {"state": {"type": "string", "enum": ["confirmed"]}}},
                                   "fieldMappings": [{"field": "state", "source": "body", "table": "allocations", "column": "state"}],
                                   "effects": [{"table": "allocations", "column": "state", "kind": "update"}]})
    assert not workflow_contract_issues([feature])


@pytest.mark.parametrize("kind", ["set", "update"])
def test_declared_transition_accepts_a_real_set_or_enum_update_effect(kind):
    feature = allocation_feature()
    effect = feature["apiContract"][1]["effects"][0]
    effect["kind"] = kind
    effect.pop("from")
    if kind == "update":
        effect.pop("to")
        feature["apiContract"][1].update(requestSchema={"type": "object", "properties": {"state": {"type": "string", "enum": ["cancelled"]}}},
                                         fieldMappings=[{"field": "state", "source": "body", "table": "allocations", "column": "state"}])
    assert not workflow_contract_issues([feature])
    if kind == "update":
        feature["apiContract"][1]["requestSchema"]["properties"]["state"].pop("enum")
        assert any("WORKFLOW_TRANSITION_MISSING" in value for value in workflow_contract_issues([feature]))


def test_read_only_consumer_reuses_the_existing_global_state_policy():
    feature = allocation_feature()
    consumer = {"workflowContractVersion": 1, "featureId": "allocation.read", "dbContract": deepcopy(feature["dbContract"]),
                "apiContract": [{"method": "GET", "path": "/api/v1/me/allocations"}]}
    assert not workflow_contract_issues([feature, consumer])
    assert any("WORKFLOW_STATE_POLICY_REQUIRED" in value for value in workflow_contract_issues([consumer]))


@pytest.mark.parametrize("change,code", [
    ({"blockingStates": ["cancelled"]}, "WORKFLOW_INTERVAL_INITIAL_UNPROTECTED"),
    ({"blockingStates": ["active"]}, "WORKFLOW_INTERVAL_STATE"),
    ({"lockColumn": "missing"}, "WORKFLOW_INTERVAL_DB_REFERENCE"),
    ({"startColumn": "missing"}, "WORKFLOW_INTERVAL_DB_REFERENCE"),
    ({"slotReferenceColumn": "window_id"}, "WORKFLOW_INTERVAL_SLOT_REFERENCE"),
])
def test_interval_policy_rejects_unsafe_occupancy_and_phantom_references(change, code):
    feature = allocation_feature()
    feature["intervalPolicies"][0].update(change)
    assert any(code in value for value in workflow_contract_issues([feature]))


def test_slot_reference_is_optional_but_must_be_real_when_declared():
    feature = allocation_feature()
    feature["intervalPolicies"][0]["slotReferenceColumn"] = "window_id"
    feature["dbContract"]["fields"].append({"table": "allocations", "name": "window_id", "type": "UUID"})
    feature["dbContract"]["foreignKeys"] = [{"table": "allocations", "column": "window_id", "references": {"table": "windows", "column": "id"}}]
    assert not workflow_contract_issues([feature])


def test_db_projection_uses_explicit_state_and_interval_checks_only():
    feature = allocation_feature()
    checks = workflow_check_constraints([feature])
    assert "CHECK (starts_at < ends_at)" in checks["allocations"]
    assert "CHECK (state IN ('pending', 'cancelled'))" in checks["allocations"]
    db, _ = contract_artifacts([feature])
    projected, _ = project_explicit_schema([], [feature])
    assert next(table for table in projected if table["name"] == "allocations")["constraints"] == checks["allocations"]
    assert not any(code == "DB_WORKFLOW_CHECK_MISSING" for code, _, _ in cross_contract_issues([feature], db, {"endpoints": feature["apiContract"]}, {}))
    db["tables"][0]["constraints"] = []
    assert any(code == "DB_WORKFLOW_CHECK_MISSING" for code, _, _ in cross_contract_issues([feature], db, {"endpoints": feature["apiContract"]}, {}))
    assert workflow_check_constraints([auth_feature()]) == {}


def test_lookup_requires_route_and_actual_response_field():
    feature = {"readDependencies": [{"path": "/api/v1/assets", "purpose": "Select an asset", "responseField": "[].id"}], "apiContract": []}
    assert any("WORKFLOW_LOOKUP_MISSING" in value for value in workflow_contract_issues([feature]))
    feature["apiContract"] = [{"method": "GET", "path": "/api/v1/assets", "responseSchema": {"type": "array", "items": {"type": "object"}}}]
    assert any("WORKFLOW_LOOKUP_OUTPUT" in value for value in workflow_contract_issues([feature]))
    feature["apiContract"][0]["responseSchema"]["items"]["properties"] = {"id": {"type": "string", "format": "uuid"}}
    assert not workflow_contract_issues([feature])


def test_identifier_discovery_cannot_be_a_self_referencing_detail_route():
    feature = {"workflowContractVersion": 1, "applicability": {"ui": True}, "apiContract": [
        {"method": "GET", "path": "/api/v1/assets/{assetId}", "parameters": [
            {"name": "assetId", "in": "path", "schema": {"type": "string", "format": "uuid"}}],
         "responseSchema": {"type": "object", "properties": {"id": {"type": "string"}}}}]}
    assert any("WORKFLOW_LOOKUP_MISSING" in value for value in workflow_contract_issues([feature]))
    feature["apiContract"].append({"method": "GET", "path": "/api/v1/assets", "responseSchema": {"type": "array", "items": {"type": "object", "properties": {"id": {"type": "string"}}}}})
    assert not workflow_contract_issues([feature])


def test_lookup_with_valid_dto_but_inaccessible_role_is_not_accepted():
    provider = {"featureId": "asset.manage", "roles": ["operator"], "apiContract": [
        {"method": "GET", "path": "/api/v1/assets", "authRequired": True, "allowedRoles": ["operator"],
         "responseSchema": {"type": "array", "items": {"type": "object", "properties": {"id": {"type": "string"}}}}}]}
    consumer = {"featureId": "allocation.apply", "roles": ["member"], "readDependencies": [
        {"path": "/api/v1/assets", "purpose": "Choose an asset", "responseField": "[].id"}]}
    issues = workflow_contract_issues([provider, consumer])
    assert any("AUTH_LOOKUP_ACCESS_DENIED: /features/0/apiContract/0" in value for value in issues)
    provider["apiContract"][0]["allowedRoles"].append("member")
    assert not workflow_contract_issues([provider, consumer])


def nested_lookup_feature():
    def parameter(name):
        return {"name": name, "in": "path", "schema": {"type": "integer"}}

    def response(table):
        return {"responseSchema": {"type": "array", "items": {"type": "object", "properties": {"id": {"type": "integer"}}}},
                "fieldMappings": [{"field": "[].id", "table": table, "column": "id", "source": "response"}]}

    feature = {"featureId": "window.manage", "workflowContractVersion": 1, "applicability": {"ui": True}, "roles": ["operator"],
               "dbContract": {"tables": ["assets", "windows"], "fields": [
                   {"table": table, "name": "id", "primaryKey": True} for table in ("assets", "windows")],
                   "foreignKeys": [{"table": "windows", "column": "asset_id", "references": {"table": "assets", "column": "id"}}]},
               "apiContract": [{"method": "GET", "path": "/api/v1/assets", **response("assets")},
                               {"method": "GET", "path": "/api/v1/assets/{assetId}/windows", "parameters": [parameter("assetId")], **response("windows")},
                               {"method": "PATCH", "path": "/api/v1/windows/{windowId}", "parameters": [parameter("windowId")],
                                "fieldMappings": [{"field": "windowId", "table": "windows", "column": "id", "source": "path"}]}]}
    feature["apiContract"][1]["fieldMappings"].append({"field": "assetId", "table": "windows", "column": "asset_id", "source": "path"})
    return feature


def test_nested_lookup_chain_resolves_child_ids_through_parent_fk():
    assert not workflow_contract_issues([nested_lookup_feature()])


def test_lookup_chain_without_root_or_with_wrong_permissions_stays_blocked():
    feature = nested_lookup_feature()
    feature["apiContract"].pop(0)
    assert any("WORKFLOW_LOOKUP_MISSING" in value for value in workflow_contract_issues([feature]))
    feature = nested_lookup_feature()
    feature["apiContract"][0].update(authRequired=True, allowedRoles=["administrator"])
    assert any("WORKFLOW_LOOKUP_MISSING" in value for value in workflow_contract_issues([feature]))


def test_shared_state_transition_can_be_implemented_by_another_feature():
    feature = allocation_feature()
    other = {"featureId": "allocation.cancel", "apiContract": [feature["apiContract"].pop()]}
    assert not workflow_contract_issues([feature, other])
    other["apiContract"][0]["effects"][0]["from"] = "other"
    assert any("WORKFLOW_TRANSITION_MISSING" in value for value in workflow_contract_issues([feature, other]))


def test_unreachable_state_marks_each_affected_response_and_query_consumer():
    feature = allocation_feature()
    feature["statePolicies"][0]["states"].append("completed")
    consumer = {"featureId": "allocation.list", "dbContract": {"primaryTable": "allocations"}, "apiContract": [
        {"method": "GET", "path": "/api/v1/me/allocations", "parameters": [{"name": "state", "in": "query", "schema": {"type": "string", "enum": ["pending", "completed"]}}]},
        {"method": "GET", "path": "/api/v1/operator/allocations", "responseSchema": {"type": "object", "properties": {"state": {"type": "string", "enum": ["pending", "completed"]}}},
         "fieldMappings": [{"field": "state", "table": "allocations", "column": "state", "source": "response"}]}]}
    issues = workflow_contract_issues([feature, consumer])
    assert any("WORKFLOW_STATE_ENUM_UNREACHABLE: /features/1/apiContract/0" in value for value in issues)
    assert any("WORKFLOW_STATE_ENUM_UNREACHABLE: /features/1/apiContract/1" in value for value in issues)


def test_lookup_patch_can_only_add_the_named_route_and_its_referenced_dto():
    payload = lookup_draft()
    features = compile_document(payload)
    assert features[0]["workflowContractVersion"] == 1
    issues = workflow_contract_issues(features)
    scope = repair_scope(payload, issues, [])
    assert scope["missingOperationKeys"] == {"asset.manage": ["GET /api/v1/assets"]}
    merged = merge_patch(payload, lookup_patch(), scope)
    assert merged["features"][0]["apiContract"][0] == payload["features"][0]["apiContract"][0]
    assert not workflow_contract_issues(compile_document(merged))
    with pytest.raises(ValueError, match="REPAIR_SCOPE"):
        merge_patch(payload, lookup_patch("/api/v1/unrelated"), scope)
    with pytest.raises(ValueError, match="promised read dependency"):
        merge_patch(payload, {"features": [{"featureId": "asset.manage", "readDependencies": []}]}, scope)


def test_sparse_read_dependency_patch_adds_without_deleting_existing_promises():
    payload = lookup_draft()
    scope = repair_scope(payload, workflow_contract_issues(compile_document(payload)), [])
    patch = {"features": [{"featureId": "asset.manage", "readDependencies": [{
        "path": "/api/v1/other-assets", "responseField": "items[].id", "purpose": "Another promised lookup",
    }]}]}
    merged = merge_patch(payload, patch, scope)
    assert merged["features"][0]["readDependencies"] == [*payload["features"][0]["readDependencies"], *patch["features"][0]["readDependencies"]]
    assert merge_patch(merged, patch, scope) == merged
    patch["features"][0]["readDependencies"] *= 2
    with pytest.raises(ValueError, match="duplicate read dependency"):
        merge_patch(payload, patch, scope)


def test_canonical_collection_id_read_resolves_a_mapped_path_identifier():
    feature = lookup_draft()
    features = compile_document(feature)
    features[0]["readDependencies"] = []
    features[0]["apiContract"] += [
        {"method": "GET", "path": "/api/v1/assets", "responseSchema": {"type": "object", "properties": {
            "items": {"type": "array", "items": {"type": "object", "properties": {"id": {"type": "string", "format": "uuid"}}}},
        }}},
        {"method": "PATCH", "path": "/api/v1/assets/{assetId}", "parameters": [{"in": "path", "name": "assetId", "schema": {"type": "string", "format": "uuid"}}],
         "fieldMappings": [{"source": "path", "field": "assetId", "table": "assets", "column": "id"}]},
    ]
    assert not workflow_contract_issues(features)
    features[0]["apiContract"][1]["responseSchema"]["properties"]["items"]["items"]["properties"] = {"name": {"type": "string"}}
    assert any("WORKFLOW_LOOKUP_MISSING" in value for value in workflow_contract_issues(features))


def test_missing_lookup_generation_uses_one_scoped_patch_without_recreating_siblings():
    def response(payload):
        return SimpleNamespace(choices=[SimpleNamespace(finish_reason="stop", message=SimpleNamespace(content=json.dumps(payload), refusal=None))], usage=None)

    payload = lookup_draft()
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=AsyncMock(side_effect=[response(payload), response(lookup_patch())]))))
    cores = source_cores({"coreFeatures": [{"sourceFeatureId": "source.asset"}]})
    features, diagnostics, issues = asyncio.run(generate_document(client, "test", None, "JSON", cores, draft=True))
    assert features and not issues and len(diagnostics) == 2
    assert client.chat.completions.create.await_count == 2
    assert len(features) == 1 and len(features[0]["apiContract"]) == 2


def test_unique_response_wrapper_is_normalized_before_preflight_without_model_repair():
    payload = lookup_draft()
    added = lookup_patch()
    added["schemas"][0]["nodes"] = [
        {"id": "root", "kind": "object"},
        {"id": "items", "parent": "root", "name": "items", "kind": "array", "required": True},
        {"id": "entry", "parent": "items", "kind": "ref", "ref": "Asset"},
    ]
    added["features"][0]["apiContract"][0]["fieldMappings"] = [{
        "source": "response", "field": "[].id", "table": "assets", "column": "id",
    }]
    payload["features"][0]["apiContract"] += added["features"][0]["apiContract"]
    payload["features"][0]["readDependencies"][0]["responseField"] = "items[].id"
    payload["schemas"] += added["schemas"]
    response = SimpleNamespace(choices=[SimpleNamespace(finish_reason="stop", message=SimpleNamespace(content=json.dumps(payload), refusal=None))], usage=None)
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=AsyncMock(return_value=response))))
    cores = source_cores({"coreFeatures": [{"sourceFeatureId": "source.asset"}]})
    features, diagnostics, issues = asyncio.run(generate_document(client, "test", None, "JSON", cores, draft=True))
    assert not issues and client.chat.completions.create.await_count == 1
    assert features[0]["apiContract"][1]["fieldMappings"][0]["field"] == "items[].id"
    assert diagnostics[0]["responseMapNormalizations"][0]["after"] == "items[].id"


@pytest.mark.parametrize("model,effort,expected", [("gpt-5.4-mini", "low", {"reasoning_effort": "low"}),
                                                  ("gpt-5.4-mini", "none", {"temperature": 0.2}),
                                                  ("test", "low", {"temperature": 0.2})])
def test_contract_reasoning_does_not_mix_unsupported_sampling_or_change_other_models(model, effort, expected):
    payload = lookup_draft()
    patch = lookup_patch()
    payload["features"][0]["apiContract"] += patch["features"][0]["apiContract"]
    payload["schemas"] += patch["schemas"]
    response = SimpleNamespace(model=model, choices=[SimpleNamespace(finish_reason="stop", message=SimpleNamespace(content=json.dumps(payload), refusal=None))], usage=None)
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=AsyncMock(return_value=response))))
    _, diagnostics, issues = asyncio.run(generate_document(client, model, None, "JSON", source_cores({"coreFeatures": [{"sourceFeatureId": "source.asset"}]}), draft=True, reasoning_effort=effort))
    assert not issues
    kwargs = client.chat.completions.create.await_args.kwargs
    assert {key: kwargs[key] for key in ("temperature", "reasoning_effort") if key in kwargs} == expected
    assert diagnostics[0]["providerModel"] == model


def test_contract_timeout_is_a_local_client_override_without_sdk_retries():
    payload = lookup_draft()
    patch = lookup_patch()
    payload["features"][0]["apiContract"] += patch["features"][0]["apiContract"]
    payload["schemas"] += patch["schemas"]
    response = SimpleNamespace(choices=[SimpleNamespace(finish_reason="stop", message=SimpleNamespace(content=json.dumps(payload), refusal=None))], usage=None)
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=AsyncMock(return_value=response))))
    options = []
    client.with_options = lambda **kwargs: options.append(kwargs) or client
    _, diagnostics, issues = asyncio.run(generate_document(client, "test", None, "JSON", source_cores({"coreFeatures": [{"sourceFeatureId": "source.asset"}]}), draft=True, timeout_seconds=240))
    assert not issues and options == [{"max_retries": 0, "timeout": 240}]
    assert client.chat.completions.create.await_count == 1
    assert diagnostics[0]["requestTimeoutSeconds"] == 240


def test_contract_timeout_also_overrides_shared_runtime_without_changing_its_default():
    async def scenario():
        runtime = LlmRuntime(1, 0.001)
        payload = lookup_draft()
        patch = lookup_patch()
        payload["features"][0]["apiContract"] += patch["features"][0]["apiContract"]
        payload["schemas"] += patch["schemas"]
        async def create(**_kwargs):
            await asyncio.sleep(0.02)
            return SimpleNamespace(choices=[SimpleNamespace(finish_reason="stop", message=SimpleNamespace(content=json.dumps(payload), refusal=None))], usage=None)
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=AsyncMock(side_effect=create))))
        _, diagnostics, issues = await generate_document(client, "test", runtime, "JSON", [{"sourceFeatureId": "source.asset"}], draft=True, timeout_seconds=1)
        assert not issues and len(diagnostics) == 1 and client.chat.completions.create.await_count == 1
        with pytest.raises(TimeoutError):
            await runtime.call(lambda: asyncio.sleep(0.02))
        with pytest.raises(ValueError, match="timeout_seconds"):
            await runtime.call(lambda: asyncio.sleep(0), timeout_seconds=0)
    asyncio.run(scenario())


def test_malformed_workflow_policy_is_a_scoped_error_not_an_exception():
    assert "WORKFLOW_POLICY_SHAPE" in workflow_contract_issues([{"statePolicies": [{"table": "missing"}]}])[0]
    assert workflow_check_constraints([{"statePolicies": [{"table": "missing"}]}]) == {}
    assert any(code == "WORKFLOW_POLICY_SHAPE" for code, _, _ in cross_contract_issues(
        [{"statePolicies": [{"table": "missing"}]}], {}, {}, {},
    ))


def test_workflow_metadata_and_role_restrictions_survive_api_projection():
    feature = auth_feature("cookie")
    feature["apiContract"][1]["allowedRoles"] = ["member", "operator"]
    for operation in feature["apiContract"]:
        operation["featureId"] = feature["featureId"]
    endpoints, issues = api_self_check(feature["apiContract"], [feature])
    assert not issues
    document = build_openapi_document({"endpoints": endpoints, **authentication_contract([feature]),
                                      "workflowPolicies": workflow_metadata([allocation_feature()])})
    assert document["paths"]["/api/v1/auth/logout"]["post"]["x-allowed-roles"] == ["member", "operator"]
    assert document["x-workflow-policies"][0]["intervalPolicies"][0]["blockingStates"] == ["pending"]


def test_explicit_role_and_interval_guards_have_corresponding_openapi_errors():
    feature = allocation_feature()
    operation = feature["apiContract"][0]
    operation.update(allowedRoles=["member"], errorCodes="400 — Invalid input", successStatus=201)
    bind_workflow_errors(operation, [feature])
    once = deepcopy(operation)
    assert bind_workflow_errors(operation, [feature]) == once
    responses = build_openapi_document({"endpoints": [operation]})["paths"]["/api/v1/allocations"]["post"]["responses"]
    assert {"400", "401", "403", "409", "500"} <= responses.keys()


def test_plain_read_has_no_invented_interval_or_role_errors():
    operation = {"method": "GET", "path": "/api/v1/assets"}
    assert bind_workflow_errors(operation, [allocation_feature()]) == operation
    assert "errorCodes" not in operation


def test_api_agent_final_output_does_not_keep_graphs_hardcoded_jwt_description():
    async def scenario():
        feature = auth_feature("cookie")
        feature.update(name="Session", actions=["login", "logout"], description="Authenticate a session", requirements=["Issue and revoke sessions"])
        for operation in feature["apiContract"]:
            operation["featureId"] = feature["featureId"]
        agent = ApiAgent(object())
        agent._graph = SimpleNamespace(ainvoke=AsyncMock(return_value={"api_spec": json.dumps({
            "authentication": "JWT Bearer", "endpoints": feature["apiContract"],
        })}))
        result = await agent.execute(PipelineState(feature_registry=[feature], feature_list=["Session"]))
        api = json.loads(result.api_spec)
        assert "JWT" not in api["authentication"]
        assert api["sessionPolicy"]["transport"] == "cookie"
        assert "cookieAuth" in json.loads(result.openapi_spec)["components"]["securitySchemes"]

    asyncio.run(scenario())


@pytest.mark.parametrize("policy,domains", [
    ("sessionPolicy", {"api"}), ("readDependencies", {"api"}),
    ("statePolicies", {"api", "db"}), ("intervalPolicies", {"api", "db"}),
])
def test_policy_only_change_triggers_affected_domains_but_not_text_only_change(policy, domains):
    old = {"featureId": "allocation.manage", "description": "Before"}
    updated = {**old, policy: {"changed": True}}
    assert OrchestrationGraph._changed_registry_domains([old], [updated], {"allocation.manage"}) == domains
    assert not OrchestrationGraph._changed_registry_domains([old], [{**old, "description": "After"}], {"allocation.manage"})


def test_feature_spec_publishes_read_contracts_to_prd_without_losing_them_in_normalization():
    payload = lookup_draft()
    scope = repair_scope(payload, workflow_contract_issues(compile_document(payload)), [])
    payload = merge_patch(payload, lookup_patch(), scope)
    core = compile_document(payload)[0]
    response = SimpleNamespace(choices=[SimpleNamespace(finish_reason="stop", message=SimpleNamespace(content=json.dumps(payload), refusal=None))], usage=None)
    provider = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=AsyncMock(return_value=response))))
    result = asyncio.run(FeatureSpecAgent(provider, "test").execute(PipelineState(prd_document=json.dumps({"coreFeatures": [core]}))))
    assert not result.generation_blockers
    assert json.loads(result.prd_document)["coreFeatures"][0]["readDependencies"] == result.feature_registry[0]["readDependencies"]
    assert provider.chat.completions.create.await_count == 1


def test_existing_lookup_route_is_repairable_in_place():
    # Live run (academy-makeup): GET /makeup-requests existed but was flagged as a
    # missing lookup; listing it only as an append key rejected the model's fix.
    from phase2.feature_draft import repair_scope
    payload = {"features": [{"featureId": "makeup.assignment", "apiContract": [
        {"method": "GET", "path": "/api/v1/makeup-requests"}, {"method": "POST", "path": "/api/v1/makeup-schedules"}]}]}
    issues = ["WORKFLOW_LOOKUP_MISSING: /features/0/readDependencies: GET /api/v1/makeup-requests: lookup",
              "WORKFLOW_LOOKUP_MISSING: /features/0/readDependencies: GET /api/v1/makeup-schedules: lookup"]
    scope = repair_scope(payload, issues, [])
    assert scope["operationIndices"]["makeup.assignment"] == [0]
    assert scope["missingOperationKeys"]["makeup.assignment"] == ["GET /api/v1/makeup-schedules"]


def test_list_route_returning_the_parameter_field_counts_as_lookup():
    from phase2.workflow_contract import _lists_parameter
    listing = {"method": "GET", "path": "/api/v1/absences", "authRequired": True, "allowedRoles": ["principal"], "parameters": [],
               "responseSchema": {"type": "object", "properties": {"items": {"type": "array", "items": {
                   "type": "object", "properties": {"absenceId": {"type": "string", "format": "uuid"}}}}}}}
    parameter = [{"name": "absenceId", "in": "path"}]
    assert _lists_parameter([listing], "/api/v1/absences", parameter, ["principal"])
    assert not _lists_parameter([listing], "/api/v1/absences", [{"name": "studentId", "in": "path"}], ["principal"])
    assert not _lists_parameter([listing], "/api/v1/absences", parameter, ["guardian"])


def test_lookup_missing_names_the_roles_that_must_discover_the_identifier():
    parameter = {"name": "assetId", "in": "path", "schema": {"type": "string", "format": "uuid"}}
    feature = {"workflowContractVersion": 1, "applicability": {"ui": True}, "apiContract": [
        {"method": "PATCH", "path": "/api/v1/assets/{assetId}", "authRequired": True, "allowedRoles": ["MANAGER"], "parameters": [parameter]},
        {"method": "GET", "path": "/api/v1/assets", "authRequired": True, "allowedRoles": ["MEMBER"],
         "responseSchema": {"type": "array", "items": {"type": "object", "properties": {"assetId": {"type": "string"}}}}}]}
    issues = [value for value in workflow_contract_issues([feature]) if "WORKFLOW_LOOKUP_MISSING" in value]
    assert issues and "['MANAGER']" in issues[0] and "extend its allowedRoles" in issues[0]
    feature["apiContract"][1]["allowedRoles"] = ["MEMBER", "MANAGER"]
    assert not any("WORKFLOW_LOOKUP_MISSING" in value for value in workflow_contract_issues([feature]))
