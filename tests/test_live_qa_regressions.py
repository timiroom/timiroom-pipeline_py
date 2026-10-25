import json

from phase2.agents.api_agent import _align_endpoints_to_db, _endpoint_quality_issues
from phase2.agents.dba_agent import (
    annotate_table_feature_ids,
    normalize_table_contract_names,
)
from phase2.agents.feature_spec_agent import FeatureSpecAgent
from phase2.agents.prd_agent import sync_core_features_with_registry
from phase2.agents.qa_agent import QaAgent
from phase2.feature_registry import build_feature_registry, normalize_feature_registry
from phase2.state import PipelineState
from phase3.schema_validator import SchemaValidator


def table(name, columns, **extra):
    return {"name": name, "description": name, "columns": [
        {"name": "id", "type": "BIGINT", "constraints": "PRIMARY_KEY"}, *columns,
    ], "indexes": [], **extra}


def test_positional_ids_are_replaced_and_parent_and_aliases_follow():
    rows = [
        {"featureId": "feature_001", "name": "예약 생성", "actions": ["create"],
         "apiContract": [{"method": "POST", "path": "/api/v1/bookings", "featureId": "feature_001"}]},
        {"featureId": "feature_001.cancel", "name": "예약 취소", "parentFeatureId": "feature_001",
         "actions": ["cancel"], "apiContract": [{"method": "POST", "path": "/api/v1/bookings/{id}/cancel"}]},
    ]
    result = normalize_feature_registry(rows, [row["name"] for row in rows])
    assert result[0]["featureId"] == "bookings.create"
    assert result[1]["parentFeatureId"] == "bookings.create"
    assert all(not row["featureId"].startswith(("feature_", "supporting.")) for row in result)
    for row in result:
        assert row["api"] == row["apiContract"]
        assert all(c["featureId"] == row["featureId"] for c in row["apiContract"])
    assert normalize_feature_registry(result, [row["name"] for row in rows]) == result


def test_initial_ids_do_not_depend_on_input_order():
    left = {row["name"]: row["featureId"] for row in build_feature_registry(["대여 등록", "반납 기록"])}
    right = {row["name"]: row["featureId"] for row in build_feature_registry(["반납 기록", "대여 등록"])}
    assert left == right


def test_auth_actions_with_distinct_routes_do_not_merge():
    rows = [{"featureId": f"auth.{action}", "name": name, "source": "supporting",
             "apiContract": [{"method": "POST", "path": f"/api/v1/auth/{action}"}]}
            for action, name in [("signup", "회원가입"), ("login", "로그인")]]
    result = FeatureSpecAgent._merge_semantic_duplicates(rows)
    assert len(result) == 2


def test_prd_sync_updates_ids_and_priority_even_without_new_features():
    state = PipelineState(
        feature_list=["대여 등록"],
        prd_document=json.dumps({"goals": ["목표"], "coreFeatures": [
            {"featureId": "feature_001", "name": "대여 등록", "priority": "P0", "description": "기존 본문"},
        ]}),
        feature_registry=[{"featureId": "rentals.create", "name": "대여 등록", "priority": "P1"}],
    )
    result = json.loads(sync_core_features_with_registry(state).prd_document)
    assert result["coreFeatures"][0]["featureId"] == "rentals.create"
    assert result["coreFeatures"][0]["priority"] == "P1"
    assert result["coreFeatures"][0]["description"] == "기존 본문"
    assert result["goals"] == ["목표"]


def test_nested_collection_is_an_array_and_path_id_uses_db_type():
    db = {"tables": [table("projects", []), table("notes", [])]}
    endpoint = {"method": "GET", "path": "/api/v1/projects/{projectId}/notes", "action": "list",
                "parameters": [{"in": "path", "name": "projectId", "type": "string"}],
                "successResponse": "item: notes"}
    result = _align_endpoints_to_db([endpoint], json.dumps(db))[0]
    assert "items: array<notes>" in result["successResponse"]
    assert "total: integer" in result["successResponse"]
    assert result["parameters"][0]["type"] == "integer"


def test_server_columns_are_not_promoted_to_writable_fields():
    columns = [{"name": name, "type": "VARCHAR(255)", "constraints": "NULL"}
               for name in ["caption", "storage_key", "external_deleted_at", "previous_status", "created_at"]]
    db = {"tables": [table("photos", columns)]}
    endpoint = {"method": "POST", "path": "/api/v1/photos", "requestBody":
                "caption: string — 사진 설명, storage_key: string — 내부 키, external_deleted_at: string — 삭제 시각"}
    result = _align_endpoints_to_db([endpoint], json.dumps(db))[0]
    assert result["requestBody"] == "caption: string — 사진 설명"


def test_user_fk_is_not_a_reason_to_reverse_user_state():
    users = table("users", [{"name": "email", "type": "TEXT"}, {"name": "status", "type": "TEXT"}])
    photos = table("photos", [
        {"name": "user_id", "type": "BIGINT", "constraints": "REFERENCES users(id)"},
        {"name": "status", "type": "TEXT"}, {"name": "completed_at", "type": "TIMESTAMP"},
    ])
    result = _align_endpoints_to_db([{"method": "DELETE", "path": "/api/v1/photos/{id}"}],
                                    json.dumps({"tables": [users, photos]}))[0]
    assert "users" not in result["transactionRules"]
    assert "역보정" not in result["transactionRules"]


def test_record_alias_is_renamed_without_creating_an_empty_duplicate():
    tables = [table("symptom_records", [{"name": "observed_at", "type": "TIMESTAMP"}]),
              table("attachments", [{"name": "symptom_id", "type": "BIGINT", "constraints": "REFERENCES symptom_records(id)"}])]
    registry = [{"featureId": "symptoms.record", "dbContract": {"tables": ["symptoms"]}}]
    result, relationships = normalize_table_contract_names(tables, ["symptom_records (1:N) attachments"], registry)
    assert [t["name"] for t in result] == ["symptoms", "attachments"]
    assert "REFERENCES symptoms(id)" in result[1]["columns"][1]["constraints"]
    assert relationships == ["symptoms (1:N) attachments"]


def test_fk_dependency_does_not_copy_every_feature_to_child():
    registry = [{"featureId": f"domain.action_{n}", "name": f"기능 {n}", "dbContract": {"tables": ["users"]}}
                for n in range(5)]
    tables = [table("users", []), table("audit_logs", [
        {"name": "user_id", "type": "BIGINT", "constraints": "REFERENCES users(id)"},
    ])]
    result = annotate_table_feature_ids(tables, registry)
    assert result[1]["featureIds"] == []
    assert len(result[1]["dependencyFeatureIds"]) == 5


def test_login_quality_is_checked_against_route_action():
    endpoint = {"method": "POST", "path": "/api/v1/auth/login", "featureName": "회원가입",
                "description": "로그인: 이메일과 비밀번호를 검증하고 인증 토큰을 발급한다.",
                "requestBody": "email: string, password: string", "successResponse": "accessToken: string",
                "errorCodes": "400, 401"}
    assert "담당 기능과 description 불일치" not in _endpoint_quality_issues(endpoint, "", endpoint, False)


def test_registry_validation_rejects_terminal_counter_ids_and_wrong_owners():
    errors = []
    SchemaValidator._check_registry_invariants([
        {"featureId": "feature_001", "priority": "P0", "apiContract": [
            {"method": "POST", "path": "/api/v1/items", "featureId": "other.create"},
        ]},
    ], lambda code, target, message, **kw: errors.append(code))
    assert "FEATURE_ID_NON_SEMANTIC" in errors
    assert "FEATURE_CONTRACT_OWNER_MISMATCH" in errors


def test_qa_accepts_path_identity_and_server_generated_storage_but_checks_business_inputs():
    db = {"tables": [table("photos", [
        {"name": "project_id", "type": "BIGINT", "constraints": "NOT_NULL"},
        {"name": "caption", "type": "TEXT", "constraints": "NOT_NULL"},
        {"name": "storage_url", "type": "TEXT", "constraints": "NOT_NULL"},
    ])]}
    endpoint = {"method": "POST", "path": "/api/v1/projects/{projectId}/photos", "requestBody": "caption: string"}
    _, errors, _ = QaAgent(object())._check_cross_document_semantics({}, db, {"endpoints": [endpoint]}, [])
    assert not any("POST 필수 필드" in error for error in errors)
    endpoint["requestBody"] = ""
    _, errors, _ = QaAgent(object())._check_cross_document_semantics({}, db, {"endpoints": [endpoint]}, [])
    assert any("caption" in error and "POST 필수 필드" in error for error in errors)


def test_api_adds_missing_required_business_input_without_adding_server_fields():
    db = {"tables": [table("notes", [
        {"name": "project_id", "type": "BIGINT", "constraints": "NOT_NULL"},
        {"name": "title", "type": "TEXT", "constraints": "NOT_NULL"},
        {"name": "content", "type": "TEXT", "constraints": "NOT_NULL"},
        {"name": "storage_url", "type": "TEXT", "constraints": "NOT_NULL"},
    ])]}
    result = _align_endpoints_to_db([{
        "method": "POST", "path": "/api/v1/projects/{projectId}/notes", "requestBody": "title: string",
    }], json.dumps(db))[0]
    assert "content: string" in result["requestBody"]
    assert "project_id:" not in result["requestBody"]
    assert "storage_url:" not in result["requestBody"]


def test_generic_inline_validation_is_a_parent_requirement_not_a_new_resource():
    result = FeatureSpecAgent._merge_semantic_duplicates([
        {"featureId": "notes.create", "name": "메모 등록", "source": "prd_core", "requirements": []},
        {"featureId": "validation_rules.create", "name": "메모 등록 · 입력·권한 검증 및 실행",
         "source": "supporting", "parentFeatureId": "notes.create", "requirements": ["소유권 검증"]},
    ])
    assert len(result) == 1
    assert result[0]["requirements"] == ["소유권 검증"]
