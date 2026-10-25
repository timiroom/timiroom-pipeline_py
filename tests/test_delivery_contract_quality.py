from copy import deepcopy

from phase2.agents.dba_agent import finalize_table_indexes
from phase2.agents.qa_agent import QaAgent
from phase2.contract_quality import (
    database_checks,
    index_issues,
    normalize_generated_contracts,
    operation_primary_table,
    operation_quality_issues,
    public_read_contract,
    volatile_index,
)
from phase2.operation_contract import bind_operation_contract
from phase2.prd_quality import reconcile_prd_quality


def test_public_read_is_distinct_from_owner_only_write_and_private_profile():
    read = {"method": "GET", "path": "/api/v1/assets/{id}", "authRequired": False, "accessScope": "PUBLIC", "effects": []}
    feature = {"ownership": {"scope": "USER"}, "apiContract": [read]}
    assert public_read_contract(feature, read)
    assert not public_read_contract(feature, {**read, "method": "DELETE"})
    private = {**read, "path": "/api/v1/me", "accessScope": "OWNER"}
    assert not public_read_contract({**feature, "apiContract": [private]}, private)
    assert operation_quality_issues(private) == ["API_ACCESS_CONTRACT_CONFLICT: accessScope/authRequired"]


def test_prd_global_auth_heuristic_does_not_override_explicit_public_read():
    read = {"method": "GET", "path": "/api/v1/assets", "authRequired": False}
    prd = {"coreFeatures": [{"name": "물품 관리", "requirements": ["등록은 인증 사용자만 가능하고 목록은 공개한다"], "apiContract": [read]}]}
    _, errors, _ = QaAgent._check_cross_document_semantics(prd, {"tables": []}, {"endpoints": [read]}, ["물품 관리"])
    assert not any("인증 요구와" in error for error in errors)
    _, legacy_errors, _ = QaAgent._check_cross_document_semantics(
        {"coreFeatures": [{"name": "개인 기록", "requirements": ["로그인 후 본인만 자신의 기록을 조회한다"]}]},
        {"tables": []}, {"endpoints": [read]}, ["개인 기록"],
    )
    assert any("인증 요구와" in error for error in legacy_errors)


def test_acceptance_detects_missing_file_update_and_screen_outputs():
    op = {"requiredInputs": ["images"], "requiredOutputs": ["items[].title", "items[].counterpartyName"],
          "requiredEffectTables": ["attachments"], "requestSchema": {"type": "object", "properties": {"title": {"type": "string"}}},
          "responseSchema": {"type": "object", "properties": {"items": {"type": "array", "items": {"type": "object", "properties": {"id": {"type": "string"}}}}}},
          "effects": []}
    errors = operation_quality_issues(op)
    assert len(errors) == 4
    assert any("requiredInputs.images" in error for error in errors)
    assert any("counterpartyName" in error for error in errors)
    op["requestSchema"]["properties"]["images"] = {"type": "array"}
    op["responseSchema"]["properties"]["items"]["items"]["properties"].update(title={"type": "string"}, counterpartyName={"type": "string"})
    op["effects"] = [{"table": "attachments", "kind": "insert"}]
    assert not operation_quality_issues(op)


def test_acceptance_recognizes_real_path_query_and_body_inputs_without_a_get_body():
    op = {"requiredInputs": ["assetId", "path.assetId", "query.limit"],
          "parameters": [{"in": "path", "name": "assetId", "schema": {"type": "string"}},
                         {"in": "query", "name": "limit", "schema": {"type": "integer"}}]}
    assert not operation_quality_issues(op)
    op["requiredInputs"] = ["body.assetId", "query.unknown"]
    assert len(operation_quality_issues(op)) == 2
    op.update(requiredInputs=["header.Authorization"], authRequired=True, sessionPolicy={"transport": "bearer"})
    assert not operation_quality_issues(op)
    op["authRequired"] = False
    assert operation_quality_issues(op)


def test_primary_table_comes_from_operation_and_worker_policy_is_not_preserved():
    op = {"method": "POST", "path": "/api/v1/tasks/{id}/messages", "successStatus": 201,
          "effects": [{"table": "messages", "kind": "insert"}]}
    feature = {"dbContract": {"primaryTable": "tasks"}, "apiContract": [op]}
    result = bind_operation_contract({**op, "primaryTable": "tasks", "sessionPolicy": {"transport": "wrong"}}, feature)
    assert result["primaryTable"] == "messages"
    assert "sessionPolicy" not in result
    assert operation_primary_table({"fieldMappings": [{"table": "messages", "source": "path"}]}, "tasks") == "messages"


def test_storage_bounds_and_creation_rules_apply_to_patch_without_inventing_fields():
    create = {"method": "POST", "authRequired": True, "requestSchema": {"type": "object", "properties": {"name": {"type": "string", "minLength": 1}}},
              "fieldMappings": [{"source": "body", "field": "name", "table": "assets", "column": "name"}]}
    patch = deepcopy(create)
    patch.update(method="PATCH", requestSchema={"type": "object", "properties": {"name": {"type": "string"}}})
    feature = {"ownership": {"scope": "USER"}, "dbContract": {"primaryTable": "assets", "fields": [{"table": "assets", "name": "name", "type": "VARCHAR(40)"}]}, "apiContract": [create, patch]}
    normalize_generated_contracts([feature])
    assert patch["requestSchema"]["properties"]["name"] == {"type": "string", "minLength": 1, "maxLength": 40}
    assert patch["requestSchema"]["minProperties"] == 1
    assert set(patch["requestSchema"]["properties"]) == {"name"}
    assert patch["accessScope"] == "OWNER"


def test_collection_reads_get_bounded_pagination_without_changing_filters():
    op = {"method": "GET", "authRequired": False, "transactionRules": "읽기 전용, 날짜 오름차순.",
          "parameters": [{"name": "status", "in": "query", "schema": {"type": "string"}}],
          "responseSchema": {"type": "object", "properties": {"items": {"type": "array", "items": {"type": "object"}}}}}
    feature = {"dbContract": {}, "apiContract": [op]}
    normalize_generated_contracts([feature])
    normalize_generated_contracts([feature])
    assert [p["name"] for p in op["parameters"]] == ["status", "limit", "offset"]
    assert op["parameters"][1]["schema"]["maximum"] == 100
    assert "날짜 오름차순" in op["transactionRules"]


def test_embedded_attachment_array_does_not_turn_detail_into_a_paginated_collection():
    detail = {"method": "GET", "authRequired": False, "responseSchema": {"type": "object", "properties": {
        "id": {"type": "string"}, "title": {"type": "string"}, "images": {"type": "array", "items": {"type": "string"}},
    }}}
    root_array = {"method": "GET", "authRequired": False, "responseSchema": {"type": "array", "items": {"type": "object"}}}
    normalize_generated_contracts([{"dbContract": {}, "apiContract": [detail, root_array]}])
    assert not detail.get("pagination") and not detail.get("parameters")
    assert root_array["pagination"]["maxLimit"] == 100


def test_volatile_indexes_are_removed_without_silently_weakening_uniqueness():
    base = "CREATE INDEX ix_session ON access_sessions(user_id) WHERE revoked_at IS NULL AND expires_at > now()"
    assert volatile_index(base)
    assert not volatile_index("CREATE INDEX ix_constant ON audit(status) WHERE status = 'now()'")
    table = {"name": "access_sessions", "columns": [{"name": "user_id"}, {"name": "revoked_at"}, {"name": "expires_at"}], "indexes": [base]}
    finalize_table_indexes([table])
    assert base not in table["indexes"]
    assert not index_issues([table])
    table["indexes"] = [base.replace("CREATE INDEX", "CREATE UNIQUE INDEX")]
    finalize_table_indexes([table])
    assert index_issues([table])


def test_database_checks_only_use_declared_numeric_bounds_and_enum():
    fields = [{"table": "assets", "name": "cost", "minimum": 0}, {"table": "assets", "name": "state", "enum": ["open", "closed"]}]
    checks = database_checks([{"dbContract": {"fields": fields}}])
    assert checks["assets"] == ["CHECK (cost >= 0)", "CHECK (state IN ('open', 'closed'))"]


def test_prd_delivery_policies_are_visible_design_defaults_and_events_are_consistent():
    prd = {"projectOverview": "사진을 올리는 WEB 서비스", "coreFeatures": [{"name": "사진 등록"}],
           "releaseSchedule": [{"date": "1개월차"}], "techStack": {"messageQueue": "미사용"},
           "kpi": [{"metric": "검색", "measurement": {"event": "item_search"}},
                   {"metric": "재방문", "measurement": {"event": "search_items"}, "measurementMethod": "search_items 수행 수"}]}
    result = reconcile_prd_quality(prd)
    assert result["kpi"][1]["measurement"]["event"] == "item_search"
    assert result["kpi"][1]["measurementMethod"] == "item_search 수행 수"
    assert result["telemetryEvents"][0]["aliases"] == ["search_items"]
    assert result["uiAcceptanceCriteria"]
    assert result["fileLifecyclePolicy"]["status"] == "design_default"
    assert "backgroundJobs" in result["techStack"]
    assert result["releasePlanningAssumptions"]["teamSize"] is None
    assert "fileLifecyclePolicy" not in prd
    simple = reconcile_prd_quality({"projectOverview": "WEB 메모 서비스", "coreFeatures": [{"name": "메모 관리", "requirements": ["파일 업로드는 제외하며 외부 파일 정리 책임은 없다"]}]})
    assert "fileLifecyclePolicy" not in simple
