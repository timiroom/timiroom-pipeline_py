import json

import pytest

from phase2.agent_contract import (
    IssueSeverity,
    auth_feature_specs,
    classify_issue,
    feature_methods,
    feature_relation_kind,
    normalize_target_arrow,
    requires_auth,
    semantic_relevance,
)
from phase2.agents.api_agent import (
    _api_feature_mappings,
    _ensure_auth_endpoints,
    _ensure_feature_endpoint_groups,
    _invalid_paths,
    _sanitize_plan_paths,
)
from phase2.agents.dba_agent import (
    DbaAgent,
    _ensure_auth_contract_tables,
    build_feature_mappings,
    enforce_feature_relation_contracts,
    enforce_schema_contracts,
    ensure_domain_columns,
)
from phase2.agents.pm_agent import PmAgent
from phase2.agents.qa_agent import QaAgent
from phase2.state import PipelineState
from phase3.schema_validator import SchemaValidator, ValidationResult
from phase3.validation_service import ValidationService


def test_auth_is_activated_by_business_requirement_not_tech_stack():
    tech_only = {"techStack": {"auth": "JWT 역할 기반 인증"}}
    explicit = {"coreFeatures": [{"name": "계정 로그인", "description": "회원이 로그인한다"}]}
    public = {"coreFeatures": [{"name": "공개 조회", "description": "로그인 없이 누구나 조회한다"}]}
    assert not requires_auth(tech_only)
    assert requires_auth(explicit)
    assert not requires_auth(public)
    assert requires_auth({}, [], "사용자가 로그인한 뒤 자신의 할 일을 관리한다")
    assert PmAgent._normalize_contract({
        "name": "문서 등록", "rationale": "사용자가 문서를 등록하고 저장한다",
        "ownership": ["PUBLIC"], "transactionRules": ["원자 저장"],
    }, False)["ownership"]["scope"] == "USER"


def test_personal_scope_adds_canonical_identity_capabilities():
    specs = auth_feature_specs()
    assert [spec["name"] for spec in specs] == [
        "회원 가입 및 로그인", "인증 세션 관리", "내 정보 및 계정 관리", "사용자별 데이터 접근 제어",
    ]
    assert all(spec["ownership"]["scope"] == "USER" for spec in specs)
    assert all(spec["transactionRules"] for spec in specs)
    assert all(spec["stateTransitions"] for spec in specs)


def test_api_paths_are_pinned_to_the_supported_version():
    plan = [{"method": "POST", "path": "/api/v0/auth/login"}]
    assert _invalid_paths(plan) == ["/api/v0/auth/login"]
    assert _sanitize_plan_paths(plan)[0]["path"] == "/api/v1/auth/login"


def test_canonical_auth_contract_removes_noncanonical_profile_mutations():
    endpoints = _ensure_auth_endpoints([{
        "method": "PUT", "path": "/api/v1/users/me", "description": "사용자 수정",
    }, {
        "method": "POST", "path": "/api/v1/users/profile", "description": "프로필 생성",
    }], True)
    keys = {(item["method"], item["path"]) for item in endpoints}
    assert ("PUT", "/api/v1/users/me") not in keys
    assert ("POST", "/api/v1/users/profile") not in keys
    assert ("PATCH", "/api/v1/users/me") in keys


def test_db_and_api_mappings_follow_the_same_user_owned_feature_contract():
    spec = {
        "name": "문서 등록", "ownership": {
            "scope": "USER", "ownerEntity": "users", "ownerKey": "user_id",
        },
        "states": ["DRAFT", "PUBLISHED"],
        "stateTransitions": ["DRAFT → PUBLISHED: 발행"],
        "transactionRules": ["문서와 첨부 정보를 함께 저장한다"],
    }
    tables = _ensure_auth_contract_tables([{
        "name": "documents", "description": "문서 등록",
        "columns": [{"name": "id", "type": "BIGINT", "constraints": "PRIMARY_KEY"}],
        "indexes": [],
    }])
    db_mappings = build_feature_mappings(tables, [spec, *auth_feature_specs()])
    documents = next(table for table in tables if table["name"] == "documents")
    user_id = next(column for column in documents["columns"] if column["name"] == "user_id")
    assert "REFERENCES users(id)" in user_id["constraints"]
    assert next(item for item in db_mappings if item["featureName"] == "문서 등록")["table"] == "documents"

    endpoints = _ensure_auth_endpoints([{
        "method": "POST", "path": "/api/v1/documents", "featureName": "문서 등록",
        "description": "문서 등록: 생성", "authRequired": True,
        "requestBody": "title: string", "successResponse": "id: integer",
        "errorCodes": "400 — 입력 오류", "transactionRules": "한 트랜잭션으로 저장한다.",
    }], True)
    schema = json.dumps({"tables": tables, "featureMappings": db_mappings}, ensure_ascii=False)
    mappings = _api_feature_mappings(
        endpoints, schema, ["문서 등록", *[item["name"] for item in auth_feature_specs()]],
    )
    assert next(item for item in mappings if item["featureName"] == "문서 등록")["operations"]
    assert next(item for item in mappings if item["featureName"] == "사용자별 데이터 접근 제어")["operations"]


def test_phase3_rejects_missing_user_owner_fk_from_explicit_mapping():
    spec = {
        "name": "문서 등록", "ownership": {"scope": "USER", "ownerEntity": "users", "ownerKey": "user_id"},
        "states": [], "stateTransitions": [], "transactionRules": ["원자 저장"],
    }
    db = json.dumps({
        "tables": [
            {"name": "users", "columns": [{"name": "id", "type": "BIGINT", "constraints": "PRIMARY_KEY"}]},
            {"name": "refresh_tokens", "columns": [{"name": "id", "type": "BIGINT", "constraints": "PRIMARY_KEY"}]},
            {"name": "documents", "description": "문서 등록", "columns": [{"name": "id", "type": "BIGINT", "constraints": "PRIMARY_KEY"}]},
        ],
        "relationships": ["users (1:N) refresh_tokens"],
        "featureMappings": [{"featureName": "문서 등록", "table": "documents"}],
    }, ensure_ascii=False)
    endpoints = [{
        "method": "POST", "path": "/api/v1/documents", "featureName": "문서 등록",
        "description": "문서 등록: 생성", "authRequired": True, "requestBody": "title: string",
        "successResponse": "id: integer", "errorCodes": "400 — 입력 오류", "transactionRules": "원자 저장",
    }]
    api = json.dumps({
        "endpoints": endpoints,
        "featureMappings": [{"featureName": "문서 등록", "table": "documents", "operations": [{"method": "POST", "path": "/api/v1/documents"}]}],
    }, ensure_ascii=False)
    result = SchemaValidator().validate(["문서 등록"], db, api, [spec])
    assert not result.success
    assert "DB 사용자 소유 FK 누락" in result.errors_to_string()


@pytest.mark.parametrize(("feature", "expected"), [
    ("할 일 목록 조회", {"GET"}),
    ("할 일 등록", {"GET", "POST"}),
    ("담당자 배정", {"GET", "POST", "PATCH"}),
    ("담당자 지정", {"GET", "PATCH"}),
    ("예약 취소", {"GET", "DELETE"}),
    ("통계 분석", {"GET"}),
])
def test_methods_follow_explicit_action_contract(feature, expected):
    assert set(feature_methods(feature)) == expected


def test_identity_capabilities_use_action_contracts_not_generic_crud():
    assert feature_methods("회원 가입 및 로그인") == ("POST",)
    assert feature_methods("인증 세션 관리") == ("POST",)
    assert feature_methods("내 정보 및 계정 관리") == ("GET", "PATCH", "DELETE")
    assert feature_methods("사용자별 데이터 접근 제어") == ("GET",)


def test_semantic_contract_supports_synonyms_and_format_normalization():
    assert semantic_relevance("담당자 배정", "팀원을 작업에 할당한다") > 0
    assert normalize_target_arrow("0%에서 60%") == "0% → 60%"
    assert feature_relation_kind("시험 기간 감지") == "derived"
    assert feature_relation_kind("과제 우선순위 표시") == "derived"


def test_derived_feature_tables_reference_the_primary_aggregate_without_service_hardcoding():
    tables = [
        {"name": "source_records", "description": "원본 정보 수집", "columns": [
            {"name": "id", "type": "BIGINT", "constraints": "PRIMARY_KEY"},
        ], "indexes": []},
        {"name": "risk_scores", "description": "위험도 계산", "columns": [
            {"name": "id", "type": "BIGINT", "constraints": "PRIMARY_KEY"},
        ], "indexes": []},
    ]

    repaired = enforce_feature_relation_contracts(
        tables, ["원본 정보 수집", "위험도 계산"], auth_required=False,
    )
    derived_columns = {column["name"]: column for column in repaired[1]["columns"]}

    assert "source_record_id" in derived_columns
    assert "REFERENCES source_records(id)" in derived_columns["source_record_id"]["constraints"]


def test_issue_severity_separates_human_review_from_phase3_blockers():
    assert classify_issue("유효하지 않은 JSON") == IssueSeverity.BLOCKER
    assert classify_issue("KPI 근거가 약함") == IssueSeverity.WARNING
    assert classify_issue("업무 규칙 불일치") == IssueSeverity.ERROR
    assert classify_issue("다음 기능에 대응하는 엔드포인트가 없어 보임") == IssueSeverity.BLOCKER
    assert classify_issue("userPersonas가 2개뿐 — 최소 3개 필요") == IssueSeverity.ERROR


@pytest.mark.parametrize("feature", [
    "할 일 등록", "수강 예약", "장비 대여 신청", "반려동물 예방접종 기록", "작품 위탁 접수",
])
def test_five_service_topics_do_not_force_auth_or_full_crud(feature):
    table_name = "domain_records"
    schema = json.dumps({"tables": [{
        "name": table_name, "description": feature,
        "columns": [{"name": "id", "type": "BIGINT", "constraints": "PRIMARY_KEY"}],
        "indexes": [],
    }]}, ensure_ascii=False)
    endpoints = _ensure_feature_endpoint_groups([], schema, [feature], "{}")
    assert endpoints
    assert all(endpoint["authRequired"] is False for endpoint in endpoints)
    assert {endpoint["method"] for endpoint in endpoints} == set(feature_methods(feature))


def test_explicit_auth_and_relationship_features_build_physical_contracts():
    agent = object.__new__(DbaAgent)
    plan = agent._fallback_plan(
        ["할 일 등록", "담당자 배정", "완료 상태 기록"],
        ["tasks", "assigned_agents", "completed_status_records"],
        auth_required=True,
    )["plan"]
    by_name = {item["name"]: item for item in plan}
    assert "users" in by_name
    assert by_name["task_assignments"]["required_refs"] == ["tasks", "users"]
    assert by_name["completed_status_records"]["required_refs"] == ["tasks", "users"]


def test_domain_plan_descriptions_still_receive_actor_and_root_foreign_keys():
    tables = [
        {"name": "tasks", "description": "할 일의 제목과 상태", "columns": [
            {"name": "id", "type": "BIGINT", "constraints": "PRIMARY_KEY"},
            {"name": "created_by", "type": "BIGINT", "constraints": "NOT_NULL"},
        ], "indexes": []},
        {"name": "users", "description": "로그인 사용자", "columns": [
            {"name": "id", "type": "BIGINT", "constraints": "PRIMARY_KEY"},
            {"name": "credential_hash", "type": "VARCHAR(255)", "constraints": "NOT_NULL"},
        ], "indexes": []},
        {"name": "task_assignments", "description": "할 일과 사용자 사이 담당자 배정", "columns": [
            {"name": "id", "type": "BIGINT", "constraints": "PRIMARY_KEY"},
        ], "indexes": []},
    ]

    repaired = enforce_feature_relation_contracts(
        tables, ["할 일 등록", "담당자 배정"], auth_required=True,
    )
    columns = {column["name"]: column for column in repaired[2]["columns"]}

    assert "REFERENCES tasks(id)" in columns["task_id"]["constraints"]
    assert "REFERENCES users(id)" in columns["user_id"]["constraints"]
    task_columns = {column["name"]: column for column in repaired[0]["columns"]}
    assert "REFERENCES users(id)" in task_columns["created_by"]["constraints"]


def test_qa_distinguishes_plain_identifiers_from_declared_foreign_keys():
    qa = object.__new__(QaAgent)
    schema = {"tables": [{
        "name": "users", "description": "로그인 사용자",
        "columns": [
            {"name": "id", "type": "BIGINT", "constraints": "PRIMARY_KEY"},
            {"name": "login_id", "type": "VARCHAR(255)", "constraints": "NOT_NULL UNIQUE"},
        ], "indexes": [],
    }, {
        "name": "tasks", "description": "할 일",
        "columns": [
            {"name": "id", "type": "BIGINT", "constraints": "PRIMARY_KEY"},
            {"name": "external_id", "type": "VARCHAR(255)", "constraints": "NULL"},
            {"name": "owner_id", "type": "BIGINT", "constraints": "FOREIGN_KEY REFERENCES missing_users(id)"},
        ], "indexes": [],
    }], "relationships": ["users (1:N) tasks"]}
    issues = qa._check_db_completeness(schema, ["할 일 등록"])
    assert not any("users.login_id" in issue or "tasks.external_id" in issue for issue in issues)
    assert any("tasks.owner_id" in issue for issue in issues)


def test_manager_contract_materializes_required_relations_and_repairs_semantic_type():
    tables = [{
        "name": "users", "description": "로그인 주체",
        "columns": [{"name": "login_id", "type": "VARCHAR(255)", "constraints": "UNIQUE"}],
    }, {
        "name": "tasks", "description": "할 일 등록",
        "columns": [{"name": "id", "type": "BIGINT", "constraints": "PRIMARY_KEY"}],
    }, {
        "name": "task_assignments", "description": "담당자 배정; 필수 관계",
        "columns": [{"name": "assigned_by", "type": "BIGINT", "constraints": "NOT_NULL"}],
    }, {
        "name": "completion_records", "description": "완료 상태 기록; 필수 관계",
        "columns": [{"name": "notes_updated", "type": "TIMESTAMPTZ", "constraints": "NULL"}],
    }]
    ensure_domain_columns(tables)
    enforce_feature_relation_contracts(
        tables, ["할 일 등록", "담당자 배정", "완료 상태 기록"], True,
    )
    by_name = {table["name"]: table for table in tables}
    assignment = {column["name"]: column for column in by_name["task_assignments"]["columns"]}
    completion = {column["name"]: column for column in by_name["completion_records"]["columns"]}
    assert "REFERENCES tasks(id)" in assignment["task_id"]["constraints"]
    assert "REFERENCES users(id)" in assignment["user_id"]["constraints"]
    assert "REFERENCES users(id)" in assignment["assigned_by"]["constraints"]
    assert "REFERENCES tasks(id)" in completion["task_id"]["constraints"]
    assert completion["notes_updated"]["type"] == "TEXT"


def test_identity_principal_is_not_selected_as_business_field_owner_before_credentials_exist():
    tables = [{
        "name": "users", "description": "사용자 인증과 프로필", "columns": [
            {"name": "id", "type": "BIGINT", "constraints": "PRIMARY_KEY"},
            {"name": "email", "type": "VARCHAR(255)", "constraints": "NOT_NULL UNIQUE"},
            {"name": "username", "type": "VARCHAR(100)", "constraints": "NOT_NULL UNIQUE"},
        ], "indexes": [],
    }, {
        "name": "work_items", "description": "업무 등록과 진행 상태 관리", "columns": [
            {"name": "id", "type": "BIGINT", "constraints": "PRIMARY_KEY"},
        ], "indexes": [],
    }]
    prd = json.dumps({"coreFeatures": [{
        "name": "업무 등록",
        "description": "제목과 설명, 마감일, 상태를 입력하여 업무를 등록한다.",
        "requirements": ["제목은 필수이며 상태와 마감일을 저장한다"],
    }]}, ensure_ascii=False)

    repaired = enforce_schema_contracts(tables, ["업무 등록"], prd)
    by_name = {table["name"]: table for table in repaired}
    principal_fields = {column["name"] for column in by_name["users"]["columns"]}
    aggregate_fields = {column["name"] for column in by_name["work_items"]["columns"]}

    assert not principal_fields & {"title", "description", "due_date", "status"}
    assert aggregate_fields & {"title", "due_date", "status"}


def test_event_feature_references_containing_aggregate_and_identity_principal():
    tables = [{
        "name": "users", "description": "사용자 인증과 프로필", "columns": [
            {"name": "id", "type": "BIGINT", "constraints": "PRIMARY_KEY"},
            {"name": "email", "type": "VARCHAR(255)", "constraints": "NOT_NULL UNIQUE"},
        ], "indexes": [],
    }, {
        "name": "work_items", "description": "업무 등록, 배정, 상태 관리", "columns": [
            {"name": "id", "type": "BIGINT", "constraints": "PRIMARY_KEY"},
        ], "indexes": [],
    }, {
        "name": "status_events", "description": "상태 변경 기록", "columns": [
            {"name": "id", "type": "BIGINT", "constraints": "PRIMARY_KEY"},
            {"name": "status", "type": "VARCHAR(50)", "constraints": "NOT_NULL"},
        ], "indexes": [],
    }]

    repaired = enforce_feature_relation_contracts(
        tables, ["업무 등록", "상태 변경 기록"], auth_required=True,
    )
    event_columns = {
        column["name"]: column for column in repaired[2]["columns"]
    }

    assert "REFERENCES work_items(id)" in event_columns["work_item_id"]["constraints"]
    assert "REFERENCES users(id)" in event_columns["user_id"]["constraints"]


def test_downstream_notification_is_not_selected_as_aggregate_root():
    tables = [{
        "name": "users", "description": "사용자 인증", "columns": [
            {"name": "id", "type": "BIGINT", "constraints": "PRIMARY_KEY"},
            {"name": "email", "type": "VARCHAR(255)", "constraints": "NOT_NULL UNIQUE"},
        ], "indexes": [],
    }, {
        "name": "documents", "description": "문서를 등록하고 내용을 보관하는 핵심 엔티티", "columns": [
            {"name": "id", "type": "BIGINT", "constraints": "PRIMARY_KEY"},
            {"name": "title", "type": "TEXT", "constraints": "NOT_NULL"},
        ], "indexes": [],
    }, {
        "name": "notifications", "description": "문서 등록 완료 알림", "columns": [
            {"name": "id", "type": "BIGINT", "constraints": "PRIMARY_KEY"},
        ], "indexes": [],
    }, {
        "name": "change_history", "description": "변경 이력 기록", "columns": [
            {"name": "id", "type": "BIGINT", "constraints": "PRIMARY_KEY"},
        ], "indexes": [],
    }]

    repaired = enforce_feature_relation_contracts(
        tables, ["문서 등록", "변경 이력 기록"], auth_required=True,
    )
    history_columns = {
        column["name"]: column for column in repaired[3]["columns"]
    }

    assert "REFERENCES documents(id)" in history_columns["document_id"]["constraints"]
    assert "notification_id" not in history_columns


def test_downstream_delivery_does_not_own_referenced_aggregate_quantity():
    tables = [{
        "name": "items", "description": "사용자 보유 항목", "columns": [
            {"name": "id", "type": "BIGINT", "constraints": "PRIMARY_KEY"},
            {"name": "quantity", "type": "INTEGER", "constraints": "NOT_NULL"},
        ], "indexes": [],
    }, {
        "name": "notifications", "description": "항목 상태 알림", "columns": [
            {"name": "id", "type": "BIGINT", "constraints": "PRIMARY_KEY"},
            {"name": "item_id", "type": "BIGINT", "constraints": "FOREIGN_KEY REFERENCES items(id)"},
            {"name": "quantity", "type": "INTEGER", "constraints": "NOT_NULL"},
            {"name": "status", "type": "VARCHAR(20)", "constraints": "NOT_NULL"},
        ], "indexes": [],
    }]
    repaired = enforce_schema_contracts(tables, [], "")
    by_name = {table["name"]: {column["name"] for column in table["columns"]} for table in repaired}
    assert "quantity" in by_name["items"]
    assert "quantity" not in by_name["notifications"]
    assert "status" in by_name["notifications"]


class _AlwaysOkValidator:
    def validate(self, **_kwargs):
        return ValidationResult.ok()


def test_phase3_blocks_error_and_blocker_qa_issues():
    service = ValidationService(_AlwaysOkValidator())
    warning_state = PipelineState(qa_issue_details=[{
        "severity": "WARNING", "reason": "KPI 근거가 약함",
    }])
    assert service.validate(warning_state).validated

    error_state = PipelineState(qa_issue_details=[{
        "severity": "ERROR", "reason": "API와 DB 계약이 일치하지 않음",
    }])
    error_result = service.validate(error_state)
    assert not error_result.validated
    assert "QA ERROR" in error_result.last_validation_error

    blocker_state = PipelineState(qa_issue_details=[{
        "severity": "BLOCKER", "reason": "유효하지 않은 JSON",
    }])
    result = service.validate(blocker_state)
    assert not result.validated
    assert "QA BLOCKER" in result.last_validation_error
