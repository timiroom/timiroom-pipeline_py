import json
from copy import deepcopy

import pytest

from phase2.agent_contract import normalize_feature_contract
from phase2.agents.api_agent import (
    _align_endpoints_to_db,
    api_self_check,
    finalize_api_contracts,
    reconcile_auth_session_contracts,
)
from phase2.agents.dba_agent import (
    _primary_contract_table,
    apply_explicit_db_contracts,
    dba_self_check,
)
from phase2.agents.feature_spec_agent import (
    FeatureSpecAgent,
    _ensure_feature_narrative,
    _ensure_required_contracts,
    _normalize_contracts,
    inherit_validation_contracts,
)
from phase2.agents.qa_agent import QaAgent
from phase2.feature_registry import normalize_feature_registry
from phase2.openapi_spec import build_openapi_document, validate_openapi_document
from phase2.operation_contract import (
    auxiliary_only_contract,
    bind_auth_registry_policy,
    bind_operation_contract,
    cross_contract_issues,
    operation_schema_issues,
)
from phase2.quality_rules import required_field_concepts
from phase3.schema_validator import SchemaValidator


def table(name, *columns):
    return {"name": name, "description": name, "columns": [
        {"name": "id", "type": "BIGINT", "constraints": "PRIMARY_KEY"},
        *[{"name": name, "type": "VARCHAR(255)", "constraints": "NULL"} for name in columns],
    ], "indexes": []}


def test_bodyless_typed_post_contract_removes_worker_null_schema_and_invented_body():
    contract = {"method": "POST", "path": "/api/v1/auth/logout", "effects": [], "successStatus": 204}
    endpoint = {**contract, "requestSchema": None, "requestBody": "refreshToken: string"}
    result = bind_operation_contract(endpoint, {"apiContract": [contract]})
    assert "requestSchema" not in result and result["requestBody"] == "없음"


def test_create_checks_all_insert_tables_and_custom_actions_not_just_primary():
    db = {name: {"name": name, "columns": [{"name": "required_value", "type": "TEXT", "source": "request", "constraints": "NOT_NULL"}]}
          for name in ("entries", "history")}
    endpoint = {"action": "record_action", "primaryTable": "entries", "effects": [
        {"kind": "insert", "table": "entries"}, {"kind": "insert", "table": "history"}]}
    errors = operation_schema_issues(endpoint, db)
    assert "API_CREATE_STORAGE_VALUE_MISSING: entries.required_value" in errors
    assert "API_CREATE_STORAGE_VALUE_MISSING: history.required_value" in errors
    db["history"]["columns"][0]["source"] = "server"
    assert "API_CREATE_STORAGE_VALUE_MISSING: history.required_value" not in operation_schema_issues(endpoint, db)


def test_explicit_request_source_does_not_gain_server_values_from_column_name():
    db = {"accounts": {"name": "accounts", "columns": [
        {"name": "password_hash", "type": "TEXT", "source": "request", "constraints": "NOT_NULL"}]}}
    endpoint = {"action": "register", "effects": [{"kind": "insert", "table": "accounts"}]}
    assert "API_CREATE_STORAGE_VALUE_MISSING: accounts.password_hash" in operation_schema_issues(endpoint, db)


@pytest.mark.parametrize(("response", "field"), [
    ({"type": "array", "items": {"type": "object", "properties": {"id": {"type": "integer"}}}}, "[].id"),
    ({"type": "object", "properties": {"items": {"type": "array", "items": {"type": "object", "properties": {"id": {"type": "integer"}}}}}}, "items[].id"),
])
def test_array_mapping_paths_follow_the_actual_dto_shape(response, field):
    endpoint = {"responseSchema": response, "fieldMappings": [{"source": "response", "field": field, "table": "entries", "column": "id"}]}
    assert not operation_schema_issues(endpoint, {"entries": table("entries")})


def test_bodyless_contract_removes_worker_invented_structured_body():
    contract = {"method": "DELETE", "path": "/api/v1/entries/{id}", "parameters": []}
    endpoint = {**contract, "requestSchema": {"type": "object"}, "requestBody": "id: integer"}
    bound = bind_operation_contract(endpoint, {"apiContract": [contract]})
    assert "requestSchema" not in bound and bound["requestBody"] == "없음"


def test_complete_physical_contract_preserves_nonstandard_owner_fk_and_removes_phantom_tables():
    registry = [{"featureId": "entries.create", "name": "Entries", "ownership": {
        "scope": "USER", "ownerEntity": "principals", "ownerKey": "created_by"},
        "dbContract": {"primaryTable": "entries", "tables": ["principals", "entries"], "fields": [
            {"table": name, "name": "id", "type": "UUID", "primaryKey": True, "required": True}
            for name in ["principals", "entries"]] + [
                {"table": "entries", "name": "created_by", "type": "UUID", "required": True, "source": "auth"}],
            "foreignKeys": [{"table": "entries", "column": "created_by", "references": {"table": "principals", "column": "id"}}]}}]
    workers = [table("principals"), table("entries", "status", "principal_id"), table("principal")]
    registry.append({**deepcopy(registry[0]), "featureId": "entries.read"})
    tables, relationships, errors = dba_self_check(workers, ["principal (1:N) entries"], registry)
    assert not errors and {item["name"] for item in tables} == {"principals", "entries"}
    entry = next(item for item in tables if item["name"] == "entries")
    assert {item["name"] for item in entry["columns"]} == {"id", "created_by"}
    assert all(item["type"] == "UUID" for item in entry["columns"])
    assert "REFERENCES principals(id)" in entry["columns"][1]["constraints"]
    assert relationships == ["principals (1:N) entries"]


def feature():
    return {
        "featureId": "photos.create", "name": "사진 등록", "source": "prd_core", "actions": ["create"],
        "apiContract": [{
            "method": "POST", "path": "/api/v1/photos", "action": "create", "successStatus": 201,
            "contentType": "multipart/form-data",
            "requestSchema": {"type": "object", "properties": {
                "file": {"type": "string", "format": "binary"},
                "caption": {"type": "string", "maxLength": 100},
            }, "required": ["file"], "additionalProperties": False},
            "responseSchema": {"type": "object", "properties": {
                "id": {"type": "integer"}, "guidance": {"type": "object", "properties": {
                    "action": {"type": "string"},
                }, "required": ["action"]},
            }, "required": ["id", "guidance"]},
            "fieldMappings": [{"field": "caption", "table": "photos", "column": "memo", "source": "request"}],
        }],
        "dbContract": {"tables": ["users", "photos"], "primaryTable": "photos", "foreignKeys": [],
                       "fields": [{"table": "photos", "name": "memo", "type": "TEXT", "required": True}],
                       "uniqueConstraints": [{"table": "photos", "columns": ["user_id", "memo"]}]},
    }


def test_contract_details_survive_both_normalizers_without_mutating_input():
    original = feature()
    saved = deepcopy(original)
    normalized = _ensure_required_contracts(_normalize_contracts(deepcopy(original)))
    registry = normalize_feature_registry([normalized], [original["name"]])
    assert registry[0]["dbContract"] == original["dbContract"]
    assert registry[0]["apiContract"][0]["requestSchema"] == original["apiContract"][0]["requestSchema"]
    assert original == saved


def test_dto_upload_and_nested_response_survive_db_finalization():
    item = feature()
    db = {"tables": [table("users", "email"), table("photos", "memo", "user_id")],
          "featureMappings": [{"featureId": item["featureId"], "table": "photos", "tables": ["users", "photos"]}]}
    endpoint = {"method": "POST", "path": "/api/v1/photos", "featureId": item["featureId"]}
    result = finalize_api_contracts([endpoint], json.dumps(db), registry=[item])[0]
    assert "file:" in result["requestBody"] and "caption:" in result["requestBody"]
    assert "user_id:" not in result["requestBody"] and "memo:" not in result["requestBody"]
    assert result["responseSchema"] == item["apiContract"][0]["responseSchema"]
    assert result["contentType"] == "multipart/form-data"
    assert not operation_schema_issues(result, {t["name"]: t for t in db["tables"]})


def test_openapi_preserves_binary_required_nested_fields_and_does_not_mutate():
    endpoint = bind_operation_contract({"method": "POST", "path": "/api/v1/photos"}, feature())
    original = deepcopy(endpoint)
    document = build_openapi_document({"endpoints": [endpoint]})
    operation = document["paths"]["/api/v1/photos"]["post"]
    schema = operation["requestBody"]["content"]["multipart/form-data"]["schema"]
    assert schema["required"] == ["file"]
    assert schema["properties"]["file"]["format"] == "binary"
    assert schema["properties"]["caption"]["maxLength"] == 100
    assert operation["responses"]["201"]["content"]["application/json"]["schema"]["properties"]["guidance"]["required"] == ["action"]
    assert validate_openapi_document(document) == []
    schema["required"].append("changed")
    assert endpoint == original


def test_openapi_honors_action_status_and_no_content():
    document = build_openapi_document({"endpoints": [
        {"method": "POST", "path": "/api/v1/checks", "successStatus": 200,
         "responseSchema": {"type": "boolean"}},
        {"method": "DELETE", "path": "/api/v1/checks/{id}", "successStatus": 204,
         "parameters": [{"name": "id", "in": "path", "type": "integer"}]},
    ]})
    assert "200" in document["paths"]["/api/v1/checks"]["post"]["responses"]
    assert "content" not in document["paths"]["/api/v1/checks/{id}"]["delete"]["responses"]["204"]
    assert not validate_openapi_document(document)


def test_openapi_checks_required_binary_and_component_references():
    document = build_openapi_document({"endpoints": [{
        "method": "POST", "path": "/api/v1/uploads",
        "requestSchema": {"type": "object", "properties": {"file": {"type": "string", "format": "binary"}}, "required": ["missing"]},
        "responseSchema": {"$ref": "#/components/schemas/Missing"},
    }]})
    errors = validate_openapi_document(document)
    assert any("required" in value for value in errors)
    assert any("multipart" in value for value in errors)
    assert any("참조 대상 없음" in value for value in errors)


def test_auxiliary_child_accepts_resource_parent_in_api_and_phase3():
    parent = {"featureId": "reminders.manage", "apiContract": [{"method": "POST", "path": "/api/v1/plants/{plantId}/reminders"}]}
    child = {"featureId": "reminders.status", "parentFeatureId": "reminders.manage",
             "apiContract": [{"method": "PATCH", "path": "/api/v1/reminders/{id}/status"}]}
    registry = [parent, child]
    assert not auxiliary_only_contract(child, registry)
    _, issues = api_self_check([], registry)
    assert not any("AUXILIARY_ONLY" in issue for issue in issues)
    errors = []
    SchemaValidator._check_registry_invariants(registry, lambda *args, **kwargs: errors.append(args[0]))
    assert "API_AUXILIARY_ONLY_CONTRACT" not in errors


@pytest.mark.parametrize("parent_id,parent_route", [(None, "/api/v1/reminders"), ("missing", "/api/v1/reminders"), ("auth.login", "/api/v1/auth/login")])
def test_orphan_auxiliary_is_not_silently_accepted(parent_id, parent_route):
    child = {"featureId": "reminders.status", "parentFeatureId": parent_id,
             "apiContract": [{"method": "PATCH", "path": "/api/v1/reminders/{id}/status"}]}
    assert auxiliary_only_contract(child, [child, {"featureId": "auth.login", "apiContract": [{"path": parent_route}]}])


def test_auxiliary_cycle_terminates_and_blocks():
    child = {"featureId": "reminders.status", "parentFeatureId": "reminders.status", "apiContract": [{"path": "/api/v1/reminders/status"}]}
    assert auxiliary_only_contract(child, [child])


def test_auxiliary_core_accepts_explicit_same_resource_dependency():
    core = {"featureId": "applications.manage", "apiContract": [{"method": "POST", "path": "/api/v1/applications"}]}
    status = {"featureId": "applications.status", "dependencies": ["applications.manage"],
              "apiContract": [{"method": "PATCH", "path": "/api/v1/applications/{id}/status"}]}
    assert not auxiliary_only_contract(status, [core, status])
    errors = []
    SchemaValidator._check_registry_invariants([core, status], lambda *args, **kwargs: errors.append(args[0]))
    assert "API_AUXILIARY_ONLY_CONTRACT" not in errors
    status["dependencies"] = ["auth.login"]
    assert auxiliary_only_contract(status, [core, status, {"featureId": "auth.login", "apiContract": [{"path": "/api/v1/auth/login"}]}])
    status["dependencies"] = ["applications.status"]
    assert auxiliary_only_contract(status, [core, status])


def test_primary_table_is_not_inferred_from_identity_order():
    tables = [table("users", "email"), table("plants", "name")]
    assert _primary_contract_table({"dbContract": {"primaryTable": "plants"}}, tables, ["users", "plants"])["name"] == "plants"
    assert _primary_contract_table({"dbContract": {"primaryTable": "missing"}}, tables, ["users"]) is None


def test_db_contract_reapplies_required_default_generated_and_composite_unique():
    tables = [table("users"), table("photos", "memo", "user_id")]
    registry = [feature()]
    registry[0]["dbContract"]["fields"].append({"table": "photos", "name": "status_history", "type": "TEXT", "required": True, "default": "[]", "generated": True, "source": "server"})
    result, _, issues = dba_self_check(tables, registry=registry)
    assert not issues
    photos = next(t for t in result if t["name"] == "photos")
    columns = {c["name"]: c for c in photos["columns"]}
    assert columns["memo"]["type"] == "TEXT"
    assert "NOT_NULL" in columns["memo"]["constraints"]
    assert "DEFAULT '[]'" in columns["status_history"]["constraints"]
    assert columns["status_history"]["generated"] is True
    assert any("UNIQUE" in value and "(user_id, memo)" in value for value in photos["indexes"])
    snapshot = deepcopy(result)
    assert not apply_explicit_db_contracts(result, registry)
    assert snapshot == result


def test_conflicting_db_contracts_are_reported_instead_of_last_writer_wins():
    first, second = feature(), feature()
    second["dbContract"]["fields"][0]["required"] = False
    issues = apply_explicit_db_contracts([table("photos", "memo", "user_id")], [first, second])
    assert "DB_FIELD_CONTRACT_CONFLICT: photos.memo" in issues


def test_prd_requirement_survives_llm_summary_before_method_expansion():
    item = {"featureId": "notes.record", "name": "기록 관리", "source": "prd_core", "description": "기록 등록", "requirements": ["메모를 저장한다"], "apiContract": [{"method": "POST", "path": "/api/v1/notes"}], "dbContract": {"tables": ["notes"]}}
    _ensure_feature_narrative(item, {}, {"description": "기록을 수정하고 삭제할 수 있다", "requirements": ["기록 삭제 시 소유권을 확인한다"]})
    result = _ensure_required_contracts(item)
    assert {"POST", "GET", "PATCH", "DELETE"} <= {c["method"] for c in result["apiContract"]}
    assert "기록 삭제 시 소유권을 확인한다" in result["requirements"]


def test_input_ownership_validation_folds_into_parent_unless_standalone():
    parent = {"featureId": "plants.create", "name": "식물 등록", "source": "prd_core", "requirements": []}
    child = {"featureId": "plants.create.validate", "name": "식물 등록 입력·소유권 검증", "source": "supporting", "parentFeatureId": "plants.create", "requirements": ["식물명은 필수"]}
    result = FeatureSpecAgent._merge_semantic_duplicates([parent, child])
    assert result == [parent] and parent["requirements"] == child["requirements"]
    child["standalone"] = True
    assert len(FeatureSpecAgent._merge_semantic_duplicates([parent, child])) == 2


def test_standalone_validation_inherits_resource_input_not_owner_columns():
    parent = feature()
    child = {"featureId": "photos.validate", "name": "사진 입력 검증", "parentFeatureId": parent["featureId"],
             "actions": ["validate"], "standalone": True,
             "apiContract": [{"method": "POST", "path": "/api/v1/photos/validation"}],
             "dbContract": {"tables": ["users"]}}
    inherit_validation_contracts([parent, child])
    assert child["dbContract"]["primaryTable"] == "photos"
    assert child["apiContract"][0]["requestSchema"] == parent["apiContract"][0]["requestSchema"]
    assert child["apiContract"][0]["path"] == "/api/v1/photos/validation"


def test_structured_dto_checks_storage_mapping_not_field_name_equality():
    item = feature()
    db = {"tables": [table("photos", "memo", "user_id")],
          "featureMappings": [{"featureId": item["featureId"], "table": "photos", "tables": ["photos"]}]}
    endpoint = bind_operation_contract({"method": "POST", "path": "/api/v1/photos", "featureId": item["featureId"], "action": "create"}, item)
    _, issues, _ = QaAgent(object())._check_cross_document_semantics({}, db, {"endpoints": [endpoint]}, [])
    assert not any("ERD와 불일치" in issue for issue in issues)
    endpoint["fieldMappings"][0]["column"] = "missing"
    _, issues, _ = QaAgent(object())._check_cross_document_semantics({}, db, {"endpoints": [endpoint]}, [])
    assert any("API_FIELD_MAPPING_TARGET_UNKNOWN" in issue for issue in issues)


def test_no_content_and_state_effects_are_checked_without_request_body():
    endpoint = {"method": "DELETE", "successStatus": 204, "responseSchema": {"type": "object"},
                "effects": [{"kind": "state_transition", "table": "missing", "column": "status"}]}
    issues = operation_schema_issues(endpoint, {})
    assert any("API_NO_CONTENT_RESPONSE_BODY" in issue for issue in issues)
    assert any("API_EFFECT_TARGET_UNKNOWN" in issue for issue in issues)
    assert any("API_STATE_TRANSITION_INCOMPLETE" in issue for issue in issues)


@pytest.mark.parametrize("text,expected", [("발생일을 입력한다", False), ("증상 발생일", False), ("생일을 입력한다", True), ("생년월일", True), ("출생일", True)])
def test_date_concepts_do_not_confuse_occurrence_with_birth(text, expected):
    assert ("birth_date" in required_field_concepts(text)) is expected


def test_fk_and_status_do_not_invent_parent_state_reversal():
    parent = table("plants", "status")
    child = table("photos", "status", "recorded_at")
    child["columns"].append({"name": "plant_id", "type": "BIGINT", "constraints": "FOREIGN_KEY REFERENCES plants(id)"})
    endpoint = {"method": "PATCH", "path": "/api/v1/photos/{id}", "requestBody": "status: string", "transactionRules": "사진만 수정한다."}
    result = _align_endpoints_to_db([endpoint], json.dumps({"tables": [parent, child]}))[0]
    assert result["transactionRules"] == "사진만 수정한다."


def test_legacy_transport_input_is_not_silently_deleted():
    endpoint = {"method": "POST", "path": "/api/v1/photos", "requestBody": "file: binary, caption: string"}
    result = _align_endpoints_to_db([endpoint], json.dumps({"tables": [table("photos", "caption")]}))[0]
    assert "file: binary" in result["requestBody"]


def test_session_logout_does_not_require_token_that_login_never_issued():
    endpoints = [{"method": "POST", "path": "/api/v1/auth/login", "successResponse": "accessToken: string"},
                 {"method": "POST", "path": "/api/v1/auth/logout", "requestBody": "refreshToken: string"}]
    result = reconcile_auth_session_contracts(endpoints, [])
    assert result[1]["requestBody"] == "없음"
    assert result[1]["authRequired"] is True
    assert "현재 세션" in result[1]["transactionRules"]
    assert len(result) == 2


def test_refresh_issuance_and_password_change_revoke_are_consistent():
    endpoints = [{"method": "POST", "path": "/api/v1/auth/login", "responseSchema": {"type": "object", "properties": {"accessToken": {"type": "string"}}}},
                 {"method": "POST", "path": "/api/v1/auth/refresh"},
                 {"method": "POST", "path": "/api/v1/auth/logout"},
                 {"method": "POST", "path": "/api/v1/auth/password-reset/confirm", "requestBody": "newPassword: string"}]
    result = reconcile_auth_session_contracts(endpoints, [])
    assert "refreshToken" in result[0]["responseSchema"]["required"]
    assert result[2]["requestSchema"]["required"] == ["refreshToken"]
    assert "기존 세션" in result[3]["transactionRules"]


def test_parallel_generators_receive_one_revocable_session_storage_contract():
    registry = [{"featureId": "auth.login", "apiContract": [{"method": "POST", "path": "/api/v1/auth/login"}], "dbContract": {"tables": ["users"]}},
                {"featureId": "auth.logout", "apiContract": [{"method": "POST", "path": "/api/v1/auth/logout"}]}]
    bind_auth_registry_policy(registry)
    saved = deepcopy(registry)
    bind_auth_registry_policy(registry)
    assert registry == saved
    for item in registry:
        assert item["sessionPolicy"]["strategy"] == "bearer_session"
        assert "auth_sessions" in item["dbContract"]["tables"]
        assert {field["name"] for field in item["dbContract"]["fields"]} == {"token_hash", "expires_at", "revoked_at"}
        assert item["dbContract"]["foreignKeys"][0]["references"]["table"] == "users"


def test_conflicting_auth_policies_and_missing_refresh_are_not_hidden():
    registry = [{"sessionPolicy": {"strategy": "refresh_token"}, "apiContract": []}]
    assert any(code == "AUTH_REFRESH_CONTRACT_MISSING" for code, _, _ in cross_contract_issues(registry, {}, {}, {}))
    registry.append({"sessionPolicy": {"strategy": "bearer_session"}, "apiContract": []})
    assert any(code == "AUTH_SESSION_POLICY_CONFLICT" for code, _, _ in cross_contract_issues(registry, {}, {}, {}))


def test_cross_contract_checks_prd_actions_db_constraints_and_response_drift():
    item = feature()
    db = {"tables": [table("users"), table("photos", "memo", "user_id")]}
    endpoint = {"method": "POST", "path": "/api/v1/photos", "featureId": item["featureId"]}
    prd = {"coreFeatures": [{"featureId": item["featureId"], "description": "사진 수정과 삭제를 제공한다"}]}
    issues = cross_contract_issues([item], db, {"endpoints": [endpoint]}, prd)
    codes = {value[0] for value in issues}
    assert {"PRD_ACTION_CONTRACT_MISSING", "API_OPERATION_CONTRACT_DRIFT", "DB_FIELD_REQUIRED_MISMATCH", "DB_UNIQUE_CONTRACT_MISSING"} <= codes
    db_issues, api_issues, prd_issues = QaAgent._check_registry_contracts([item], db, {"endpoints": [endpoint]}, prd)
    assert any("DB_UNIQUE_CONTRACT_MISSING" in value for value in db_issues)
    assert any("API_OPERATION_CONTRACT_DRIFT" in value for value in api_issues)
    assert any("PRD_ACTION_CONTRACT_MISSING" in value for value in prd_issues)


def test_file_metadata_alone_does_not_satisfy_upload_requirement():
    item = feature()
    item["requirements"] = ["사용자는 이미지 파일을 업로드한다"]
    endpoint = {"featureId": item["featureId"], "method": "POST", "path": "/api/v1/photos",
                "requestSchema": {"type": "object", "properties": {"filename": {"type": "string"}}}}
    issues = cross_contract_issues([item], {}, {"endpoints": [endpoint]}, {})
    assert any(code == "FEATURE_UPLOAD_CONTRACT_INCOMPLETE" for code, _, _ in issues)


def test_signed_upload_flow_can_be_split_across_parent_and_child():
    parent = {"featureId": "photos.record", "actions": ["upload"], "apiContract": []}
    child = {"featureId": "photos.record.complete", "parentFeatureId": parent["featureId"], "apiContract": []}
    endpoints = [
        {"featureId": parent["featureId"], "method": "POST", "path": "/api/v1/uploads",
         "responseSchema": {"type": "object", "properties": {"uploadUrl": {"type": "string", "format": "uri"}}}},
        {"featureId": child["featureId"], "method": "POST", "path": "/api/v1/photos",
         "requestSchema": {"type": "object", "properties": {"storageKey": {"type": "string"}}}},
    ]
    assert not cross_contract_issues([parent, child], {}, {"endpoints": endpoints}, {})


def test_required_output_is_checked_through_nested_component_schema():
    item = {"featureId": "symptoms.record", "outputRequirements": [{"field": "recommendedAction"}], "apiContract": []}
    endpoint = {"featureId": item["featureId"], "method": "POST", "path": "/api/v1/symptoms",
                "responseSchema": {"$ref": "#/components/schemas/Result"}}
    api = {"endpoints": [endpoint], "schemas": {"Result": {"type": "object", "properties": {
        "guidance": {"type": "object", "properties": {"recommendedAction": {"type": "string"}}},
    }}}}
    assert not cross_contract_issues([item], {}, api, {})
    api["schemas"]["Result"]["properties"].clear()
    assert any(code == "API_REQUIRED_OUTPUT_MISSING" for code, _, _ in cross_contract_issues([item], {}, api, {}))


@pytest.mark.parametrize("path", ["guidance.actions[]", "items[].message"])
def test_required_output_resolves_paths_and_rejects_wrong_parent(path):
    item = {"featureId": "notes.read", "outputRequirements": [{"field": path}], "apiContract": []}
    endpoint = {"featureId": item["featureId"], "method": "GET", "path": "/api/v1/notes", "responseSchema": {
        "type": "object", "properties": {
            "guidance": {"type": "object", "properties": {"actions": {"type": "array", "items": {"type": "string"}}}},
            "items": {"type": ["array", "null"], "items": {"type": "object", "properties": {"message": {"type": "string"}}}},
        }}}
    assert not cross_contract_issues([item], {}, {"endpoints": [endpoint]}, {})
    item["outputRequirements"] = [{"field": "wrongParent." + path.rsplit(".", 1)[-1]}]
    assert any(code == "API_REQUIRED_OUTPUT_MISSING" for code, _, _ in cross_contract_issues([item], {}, {"endpoints": [endpoint]}, {}))


def test_structured_transitions_are_not_stringified_by_feature_normalization():
    transition = {"from": "pending", "to": "completed", "role": "owner"}
    result = normalize_feature_contract({"name": "상태 변경", "states": ["pending", "completed"],
                                         "stateTransitions": [transition, transition]}, False)
    assert result["stateTransitions"] == [transition]


def test_session_entry_auth_exemption_uses_declared_lifecycle_not_resource_path():
    from phase2.operation_contract import is_public_session_entry

    operation = {"method": "POST", "path": "/api/v1/identity/sessions", "action": "login", "authRequired": False,
                 "requestSchema": {"type": "object"}, "effects": [{"kind": "insert", "table": "access_sessions"}]}
    item = {"sessionPolicy": {"strategy": "bearer_session", "sessionTable": "access_sessions"},
            "dbContract": {"tables": ["access_sessions"]}}
    assert is_public_session_entry(operation, item)
    assert is_public_session_entry({**operation, "action": "create"}, item)
    assert not is_public_session_entry({**operation, "action": "logout"}, item)
    assert not is_public_session_entry({**operation, "effects": [{"kind": "insert", "table": "orders"}]}, item)
    assert not is_public_session_entry(operation, {"dbContract": {"tables": ["access_sessions"]}})


def test_structured_read_and_cancel_actions_do_not_require_patch_for_change_history():
    for action, method in (("read", "GET"), ("cancel", "POST")):
        item = {"featureId": "bookings." + action, "apiContract": [{"method": method, "path": "/api/v1/bookings"}]}
        prd = {"coreFeatures": [{"featureId": item["featureId"], "actions": [action], "description": "예약 상태를 취소로 변경한 내역을 조회한다"}]}
        issues = cross_contract_issues([item], {}, {}, prd)
        assert not any(code == "PRD_ACTION_CONTRACT_MISSING" for code, _, _ in issues)


def test_auth_policy_binding_does_not_enrich_explicit_physical_contract():
    item = {"featureId": "identity.session", "ownership": {"ownerEntity": "principals"},
            "sessionPolicy": {"strategy": "bearer_session", "sessionTable": "sessions"},
            "apiContract": [{"method": "POST", "path": "/api/v1/auth/login"}],
            "dbContract": {"tables": ["principals", "sessions"], "primaryTable": "sessions",
                           "fields": [{"table": "principals", "name": "id", "type": "UUID", "primaryKey": True},
                                      {"table": "sessions", "name": "id", "type": "UUID", "primaryKey": True},
                                      {"table": "sessions", "name": "subject_id", "type": "UUID"}],
                           "foreignKeys": [{"table": "sessions", "column": "subject_id", "references": {"table": "principals", "column": "id"}}],
                           "uniqueConstraints": []}}
    before = deepcopy(item["dbContract"])
    bind_auth_registry_policy([item])
    assert item["dbContract"] == before


def test_invite_token_table_is_not_a_second_session_table():
    # Live run (academy-makeup): account_invites(token_hash, expires_at) next to the
    # declared user_sessions table blocked Feature Spec with an unrepairable conflict.
    from phase2.operation_contract import session_policy_issues
    policy = {"strategy": "bearer_session", "sessionTable": "user_sessions"}
    fields = [{"table": "user_sessions", "name": "session_token_hash"}, {"table": "user_sessions", "name": "expires_at"},
              {"table": "account_invites", "name": "token_hash"}, {"table": "account_invites", "name": "expires_at"}]
    feature = {"sessionPolicy": policy, "dbContract": {"fields": fields}, "apiContract": [
        {"method": "POST", "path": "/api/v1/auth/login", "authRequired": False, "effects": [{"table": "user_sessions", "kind": "insert"}]},
        {"method": "POST", "path": "/api/v1/auth/invitations", "authRequired": True, "effects": [{"table": "account_invites", "kind": "insert"}]},
    ]}
    assert session_policy_issues([feature]) == []
    # A second table that public login actually writes is still a conflict.
    feature["apiContract"][0]["effects"].append({"table": "account_invites", "kind": "insert"})
    assert session_policy_issues([feature])


def test_shared_route_unions_compatible_field_mappings():
    from phase2.operation_contract import operation_registry

    def feature(fid, mappings):
        return {"featureId": fid, "apiContract": [{"method": "GET", "path": "/api/v1/orders/{orderId}", "fieldMappings": mappings}]}

    path = {"field": "orderId", "table": "orders", "column": "id", "source": "path"}
    status = {"field": "status", "table": "orders", "column": "status", "source": "response"}
    quantity = {"field": "items[].quantity", "table": "order_items", "column": "quantity", "source": "response"}
    operations, conflicts = operation_registry([feature("a", [path, status]), feature("b", [path, quantity])])
    assert conflicts == []
    assert next(iter(operations.values()))["fieldMappings"] == [path, status, quantity]

    clash = {"field": "status", "table": "orders", "column": "state", "source": "response"}
    _, conflicts = operation_registry([feature("a", [path, status]), feature("b", [path, clash])])
    assert any("fieldMappings" in item for item in conflicts)


def test_camel_case_request_field_supplies_its_snake_case_column():
    db = {"bookings": {"name": "bookings", "columns": [
        {"name": "room_id", "type": "UUID", "source": "body", "constraints": "NOT_NULL"},
        {"name": "plan_id", "type": "UUID", "source": "body", "constraints": "NOT_NULL"}]}}
    endpoint = {"action": "create", "primaryTable": "bookings", "effects": [{"kind": "insert", "table": "bookings"}],
                "requestSchema": {"type": "object", "properties": {"roomId": {"type": "string"}, "membershipPlanId": {"type": "string"}}}}
    errors = operation_schema_issues(endpoint, db)
    assert "API_CREATE_STORAGE_VALUE_MISSING: bookings.room_id" not in errors
    # A different name still needs an explicit fieldMapping.
    assert "API_CREATE_STORAGE_VALUE_MISSING: bookings.plan_id" in errors


def test_child_insert_takes_parent_fk_from_bound_path_parameter():
    db = {"adjustments": {"name": "adjustments", "columns": [
        {"name": "source_record_id", "type": "UUID", "source": "path", "constraints": "NOT_NULL FOREIGN_KEY REFERENCES records(id)"}]}}
    endpoint = {"action": "create", "path": "/api/v1/records/{recordId}/adjustments", "primaryTable": "adjustments",
                "effects": [{"kind": "insert", "table": "adjustments"}],
                "fieldMappings": [{"field": "recordId", "table": "records", "column": "id", "source": "path"}]}
    assert "API_CREATE_STORAGE_VALUE_MISSING: adjustments.source_record_id" not in operation_schema_issues(endpoint, db)
    endpoint["fieldMappings"] = []
    assert "API_CREATE_STORAGE_VALUE_MISSING: adjustments.source_record_id" in operation_schema_issues(endpoint, db)
