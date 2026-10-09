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


def _contract_tables(*names: str) -> list[dict]:
    tables = [{
        "name": name, "description": "",
        "columns": [{"name": "id", "type": "BIGINT", "constraints": "PRIMARY_KEY"}], "indexes": [],
    } for name in names]
    users = next(table for table in tables if table["name"] == "users")
    users["columns"] += [
        {"name": "email", "type": "VARCHAR(320)", "constraints": "NOT_NULL UNIQUE"},
        {"name": "password_hash", "type": "VARCHAR(255)", "constraints": "NOT_NULL"},
    ]
    return tables


def _mapped_table(mappings: list[dict], feature_id: str) -> str:
    return next(item for item in mappings if item["featureId"] == feature_id)["table"]


def test_feature_mapping_selects_the_resource_table_not_the_first_declared_owner():
    # 운영 재현: dbContract.tables가 소유자·상위 테이블까지 모두 나열하면 스키마 순서상 첫 테이블(users)이
    # 모든 기능의 대표 테이블이 되어 QA가 /tasks 요청 필드를 users 컬럼과 대조했다.
    ownership = {"scope": "USER", "ownerEntity": "users", "ownerKey": "user_id"}
    tables = _contract_tables(
        "users", "teams", "team_members", "tasks", "task_assignees", "task_status_histories",
    )
    specs = [{
        "name": "할 일 등록", "featureId": "feature_001", "ownership": ownership,
        "apiContract": [{"method": "POST", "path": "/api/v1/tasks"}],
        "dbContract": {"tables": ["users", "teams", "team_members", "tasks"], "foreignKeys": [
            {"table": "tasks", "column": "created_by_user_id", "references": {"table": "users", "column": "id"}},
            {"table": "tasks", "column": "team_id", "references": {"table": "teams", "column": "id"}},
        ]},
    }, {
        "name": "담당자 배정", "featureId": "feature_002", "ownership": ownership,
        "apiContract": [{"method": "PUT", "path": "/api/v1/tasks/{taskid}/assignees"}],
        "dbContract": {"tables": ["users", "teams", "team_members", "tasks", "task_assignees"], "foreignKeys": []},
    }, {
        "name": "완료 상태 기록", "featureId": "feature_003", "ownership": ownership,
        "apiContract": [{"method": "PATCH", "path": "/api/v1/tasks/{taskid}/status"}],
        "dbContract": {"tables": ["users", "teams", "tasks", "task_status_histories"], "foreignKeys": []},
    }, {
        "name": "소규모 팀 멤버 관리", "featureId": "feature_002.team_members", "ownership": ownership,
        "apiContract": [
            {"method": "GET", "path": "/api/v1/teams/{teamid}/members"},
            {"method": "POST", "path": "/api/v1/teams/{teamid}/members"},
        ],
        "dbContract": {"tables": ["users", "teams", "team_members"], "foreignKeys": []},
    }, {
        "name": "회원가입", "featureId": "auth.signup", "ownership": ownership,
        "apiContract": [{"method": "POST", "path": "/api/v1/auth/signup"}],
        "dbContract": {"tables": ["users"], "foreignKeys": []},
    }]

    mappings = build_feature_mappings(tables, specs)

    assert _mapped_table(mappings, "feature_001") == "tasks"
    assert _mapped_table(mappings, "feature_002") == "task_assignees"
    assert _mapped_table(mappings, "feature_003") == "tasks"
    assert _mapped_table(mappings, "feature_002.team_members") == "team_members"
    assert _mapped_table(mappings, "auth.signup") == "users"
    # 대표 테이블에 추가되는 소유자 FK는 QA의 FK 인덱스 규칙도 함께 만족해야 한다.
    tasks = next(table for table in tables if table["name"] == "tasks")
    assert any(column["name"] == "user_id" for column in tasks["columns"])
    assert any("tasks(user_id)" in index.replace(" ", "") for index in tasks["indexes"])


def test_feature_mapping_without_path_match_uses_the_dependent_contract_table():
    ownership = {"scope": "USER", "ownerEntity": "users", "ownerKey": "user_id"}
    tables = _contract_tables("users", "calendars", "user_calendar_preferences")
    specs = [{
        "name": "기본 캘린더 설정", "featureId": "profile.preferences", "ownership": ownership,
        "apiContract": [{"method": "PATCH", "path": "/api/v1/profile"}],
        "dbContract": {"tables": ["users", "calendars", "user_calendar_preferences"], "foreignKeys": [
            {"table": "user_calendar_preferences", "column": "user_id", "references": {"table": "users", "column": "id"}},
            {"table": "user_calendar_preferences", "column": "calendar_id", "references": {"table": "calendars", "column": "id"}},
        ]},
    }, {
        "name": "회원정보 수정", "featureId": "profile.manage", "ownership": ownership,
        "apiContract": [{"method": "PATCH", "path": "/api/v1/profile"}],
        "dbContract": {"tables": ["users", "user_sessions_missing"], "foreignKeys": []},
    }]

    mappings = build_feature_mappings(tables, specs)

    assert _mapped_table(mappings, "profile.preferences") == "user_calendar_preferences"
    assert _mapped_table(mappings, "profile.manage") == "users"


def test_registry_contract_and_api_output_use_the_same_update_method():
    # 운영 재현: API 출력은 경로 변수가 있는 PUT을 PATCH로 바꾸는데 Registry 계약은 PUT으로 남아
    # "apiContract PUT ... 누락"이 재생성으로도 해소되지 않았다.
    from phase2.agents.api_agent import _normalize_endpoints
    from phase2.feature_registry import (
        missing_api_contract_features,
        normalize_feature_registry,
    )

    registry = normalize_feature_registry([{
        "id": "feature_002", "name": "담당자 배정",
        "apiContract": [
            {"method": "PUT", "path": "/tasks/{taskid}/assignees", "action": "assign"},
            {"method": "PUT", "path": "/api/v1/settings", "action": "replace"},
        ],
    }], ["담당자 배정"])
    contracts = [(item["method"], item["path"]) for item in registry[0]["apiContract"]]
    endpoints = _normalize_endpoints([
        {"method": "PUT", "path": "/api/v1/tasks/{taskid}/assignees", "description": "담당자 배정: 지정"},
        {"method": "PUT", "path": "/api/v1/settings", "description": "설정 교체"},
    ])

    assert contracts == [("PATCH", "/api/v1/tasks/{taskid}/assignees"), ("PUT", "/api/v1/settings")]
    assert [(item["method"], item["path"]) for item in endpoints] == contracts
    assert missing_api_contract_features(registry, endpoints) == []


def test_api_request_fields_follow_the_same_feature_table_as_qa():
    # 운영 재현: 계약에 테이블이 여러 개면 API는 경로 첫 구간(tasks)의 컬럼으로 본문을 채우고
    # QA는 기능 매핑(task_assignees)과 대조해 같은 엔드포인트를 서로 다른 테이블로 판정했다.
    from phase2.agents.api_agent import finalize_api_contracts
    from phase2.agents.qa_agent import _endpoint_table, _request_field_names

    ownership = {"scope": "USER", "ownerEntity": "users", "ownerKey": "user_id"}
    tables = _contract_tables("users", "tasks", "task_assignees")
    next(table for table in tables if table["name"] == "tasks")["columns"] += [
        {"name": "title", "type": "VARCHAR(200)", "constraints": "NOT_NULL"},
    ]
    next(table for table in tables if table["name"] == "task_assignees")["columns"] += [
        {"name": "task_id", "type": "BIGINT", "constraints": "NOT_NULL FOREIGN_KEY REFERENCES tasks(id)"},
        {"name": "assigned_at", "type": "TIMESTAMPTZ", "constraints": "NOT_NULL DEFAULT_NOW"},
    ]
    registry = [{
        "name": "담당자 배정", "featureId": "feature_002", "id": "feature_002", "ownership": ownership,
        "apiContract": [{"method": "PUT", "path": "/api/v1/tasks/{taskid}/assignees", "featureId": "feature_002"}],
        "dbContract": {"tables": ["users", "tasks", "task_assignees"], "foreignKeys": []},
    }]
    mappings = build_feature_mappings(tables, registry)
    schema = json.dumps({"tables": tables, "featureMappings": mappings}, ensure_ascii=False)

    endpoints = finalize_api_contracts([{
        "method": "PUT", "path": "/api/v1/tasks/{taskid}/assignees", "featureId": "feature_002",
        "description": "담당자 배정: 지정", "requestBody": "title: string",
    }], schema, registry=registry)

    endpoint = next(item for item in endpoints if item["path"] == "/api/v1/tasks/{taskid}/assignees")
    table_name, table = _endpoint_table(endpoint, {item["name"]: item for item in tables}, mappings)
    assert table_name == "task_assignees"
    fields = _request_field_names(endpoint["requestBody"])
    assert "title" not in fields
    assert fields <= {column["name"] for column in table["columns"]}
    assert endpoint["method"] == "PATCH"


def test_pre_authentication_routes_are_not_required_to_be_authenticated():
    # 운영 재현: 회원가입(/auth/register)과 비밀번호 재설정은 로그인 전에 호출되는데
    # 사용자 소유 기능이라는 이유로 authRequired를 요구해 재생성으로 해소할 수 없었다.
    from phase3.schema_validator import SchemaValidator

    ownership = {"scope": "USER", "ownerEntity": "users", "ownerKey": "user_id"}
    public = [
        ("POST", "/api/v1/auth/register"), ("POST", "/api/v1/auth/password-reset-requests"),
        ("POST", "/api/v1/auth/password-resets"),
    ]
    protected = [("POST", "/api/v1/auth/logout"), ("PATCH", "/api/v1/profile")]
    specs = [{
        "name": "계정", "featureId": "auth.account", "ownership": ownership,
        "transactionRules": ["한 트랜잭션으로 처리한다"],
        "apiContract": [{"method": method, "path": path} for method, path in public + protected],
    }]
    endpoints = [{
        "method": method, "path": path, "authRequired": False, "requestBody": "email: string",
        "successResponse": "success: boolean", "errorCodes": "400 — 입력 오류",
        "transactionRules": "한 트랜잭션으로 처리한다.",
    } for method, path in public + protected]
    db = json.dumps({"tables": _contract_tables("users"), "featureMappings": [
        {"featureName": "계정", "featureId": "auth.account", "table": "users"},
    ]}, ensure_ascii=False)
    api = json.dumps({"endpoints": endpoints, "featureMappings": [{
        "featureName": "계정", "operations": [{"method": method, "path": path} for method, path in public + protected],
    }]}, ensure_ascii=False)

    errors = SchemaValidator()._check_feature_contracts(specs, db, api)

    auth_errors = sorted(error for error in errors if "인증 누락" in error)
    assert auth_errors == [
        "API 사용자 소유 기능 인증 누락: PATCH /api/v1/profile",
        "API 사용자 소유 기능 인증 누락: POST /api/v1/auth/logout",
    ]


def test_contract_name_normalization_never_creates_duplicate_table_names():
    # 운영 재현: 계약 이름(password_reset_tokens)이 이미 스키마에 있는데 끝 단어만 같은 다른 테이블을
    # 같은 이름으로 바꿔 "중복 테이블명" 차단이 발생했다.
    from phase2.agents.dba_agent import normalize_table_contract_names

    registry = [{"featureId": "auth.password_reset", "dbContract": {"tables": ["users", "password_reset_tokens"]}}]
    tables = [
        {"name": name, "columns": [{"name": "id", "type": "BIGINT", "constraints": "PRIMARY_KEY"}], "indexes": []}
        for name in ("users", "password_reset_tokens", "email_verification_tokens", "invitation_tokens")
    ]
    renamed, _ = normalize_table_contract_names(tables, [], registry)
    assert [table["name"] for table in renamed] == [
        "users", "password_reset_tokens", "email_verification_tokens", "invitation_tokens",
    ]

    missing = [
        {"name": name, "columns": [{"name": "id", "type": "BIGINT", "constraints": "PRIMARY_KEY"}], "indexes": []}
        for name in ("users", "reset_tokens", "invitation_tokens")
    ]
    renamed, _ = normalize_table_contract_names(missing, [], registry)
    names = [table["name"] for table in renamed]
    assert len(names) == len(set(names))
    assert names.count("password_reset_tokens") == 1


def test_prd_core_features_follow_required_features_confirmed_after_prd():
    # 운영 재현: Feature Spec이 PRD 작성 뒤에 P0 기능을 확정하면 "PRD coreFeatures에 featureId/name 누락"이
    # 발생했고, 이를 LLM 보정에 맡기자 기존 기능의 이름이 featureId 조각(manage)으로 바뀌었다.
    from phase2.agents.prd_agent import sync_core_features_with_registry

    original = {
        "featureId": "feature_001", "name": "할 일 등록", "priority": "P0",
        "description": "사용자가 새 할 일을 입력하면 시스템이 저장한다.", "requirements": ["제목은 필수다."],
    }
    state = PipelineState(
        feature_list=["할 일 등록"],
        prd_document=json.dumps({"coreFeatures": [original], "goals": ["목표"]}, ensure_ascii=False),
        feature_registry=[
            {"featureId": "feature_001", "name": "할 일 등록", "priority": "P0", "source": "prd_core"},
            {"featureId": "profile.manage", "name": "회원정보 수정", "priority": "P0", "source": "prd_core",
             "apiContract": [{"method": "PATCH", "path": "/api/v1/profile"}]},
            {"featureId": "feature_001.validation", "name": "입력 검증", "priority": "P1", "source": "supporting"},
        ],
    )

    synced = json.loads(sync_core_features_with_registry(state).prd_document)

    assert synced["goals"] == ["목표"]
    assert synced["coreFeatures"][0] == original
    assert [(item["featureId"], item["name"]) for item in synced["coreFeatures"]] == [
        ("feature_001", "할 일 등록"), ("profile.manage", "회원정보 수정"),
    ]
    assert sync_core_features_with_registry(state.copy(prd_document="")).prd_document == ""


def test_parent_state_entity_with_completion_time_is_not_a_history_table():
    # 운영 재현: status와 completed_at을 가진 tasks가 이력 테이블로 분류되어
    # "이력 엔티티 tasks가 실제 상태 엔티티를 참조하지 않음"으로 차단됐다.
    def table(name, *columns):
        return {"name": name, "indexes": [], "columns": [
            {"name": "id", "type": "BIGINT", "constraints": "PRIMARY_KEY"},
            *[{"name": column, "type": "BIGINT" if ref else "VARCHAR(40)",
               "constraints": f"NOT_NULL FOREIGN_KEY REFERENCES {ref}(id)" if ref else "NOT_NULL"}
              for column, ref in columns],
        ]}

    db = {"tables": [
        table("users", ("email", None)),
        table("workspaces", ("name", None)),
        table("workspace_members", ("user_id", "users"), ("workspace_id", "workspaces"), ("status", None)),
        table("tasks", ("user_id", "users"), ("workspace_id", "workspaces"), ("status", None), ("completed_at", None)),
        table("task_status_histories", ("task_id", "tasks"), ("status", None), ("recorded_at", None)),
    ]}
    prd = {"coreFeatures": [{"name": "할 일 등록", "description": "할 일을 등록한다.", "requirements": []}]}

    db_issues, _api_issues, _prd_issues = QaAgent(client=None)._check_cross_document_semantics(
        prd, db, {"endpoints": []}, ["할 일 등록"],
    )

    assert not [issue for issue in db_issues if "이력 엔티티 tasks" in issue]


def test_put_contract_is_matched_by_every_gate_after_canonicalization():
    # 운영 재현: Registry를 거치지 않은 feature_specs의 PUT 계약이 최종 게이트에서 다시 누락으로 판정됐다.
    from phase2.feature_registry import missing_api_contract_features

    contract = {"method": "PUT", "path": "/api/v1/tasks/{taskid}/assignees"}
    endpoint = {
        "method": "PATCH", "path": "/api/v1/tasks/{taskid}/assignees", "authRequired": True,
        "requestBody": "user_id: integer", "successResponse": "success: boolean",
        "errorCodes": "400 — 입력 오류", "transactionRules": "한 트랜잭션으로 처리한다.",
    }
    spec = {
        "name": "담당자 배정", "featureId": "feature_002", "id": "feature_002",
        "transactionRules": ["한 트랜잭션으로 처리한다"], "apiContract": [contract],
    }
    assert missing_api_contract_features([spec], [endpoint]) == []

    db = json.dumps({"tables": _contract_tables("users"), "featureMappings": [
        {"featureName": "담당자 배정", "featureId": "feature_002", "table": "users"},
    ]}, ensure_ascii=False)
    api = json.dumps({"endpoints": [endpoint], "featureMappings": [
        {"featureName": "담당자 배정", "operations": [{"method": "PATCH", "path": endpoint["path"]}]},
    ]}, ensure_ascii=False)
    errors = SchemaValidator()._check_feature_contracts([spec], db, api)
    assert not [error for error in errors if "엔드포인트 누락" in error]

    _db_issues, api_issues, _prd_issues = QaAgent._check_cross_artifacts(
        {"tables": _contract_tables("users")}, {"endpoints": [endpoint]},
        {"coreFeatures": [spec]},
    )
    assert not [issue for issue in api_issues if "apiContract" in issue]


def test_self_reference_with_a_singular_table_name_resolves_its_target():
    # 운영 재현: auth_sessions.replaced_by_auth_session_id가 복수형 테이블명과 맞지 않아 FK 대상 없음으로 차단됐다.
    db = {"tables": [
        {"name": "users", "columns": [{"name": "id", "type": "BIGINT", "constraints": "PRIMARY_KEY"}]},
        {"name": "auth_sessions", "columns": [
            {"name": "id", "type": "BIGINT", "constraints": "PRIMARY_KEY"},
            {"name": "user_id", "type": "BIGINT", "constraints": "NOT_NULL FOREIGN_KEY REFERENCES users(id)"},
            {"name": "replaced_by_auth_session_id", "type": "BIGINT", "constraints": "NULL FOREIGN_KEY"},
            {"name": "approved_by_manager_id", "type": "BIGINT", "constraints": "NULL FOREIGN_KEY"},
            # DBA는 teacher/instructor 같은 역할 FK를 users로 연결한다. QA도 같은 기준이어야 한다.
            {"name": "teacher_id", "type": "BIGINT", "constraints": "NOT_NULL FOREIGN_KEY"},
        ]},
    ]}
    # 운영 실패 원문: refresh_tokens.replaced_by_token_id (토큰 회전 자기참조)
    db["tables"].append({"name": "refresh_tokens", "columns": [
        {"name": "id", "type": "BIGINT", "constraints": "PRIMARY_KEY"},
        {"name": "user_id", "type": "BIGINT", "constraints": "NOT_NULL FOREIGN_KEY REFERENCES users(id)"},
        {"name": "replaced_by_token_id", "type": "BIGINT", "constraints": "NULL FOREIGN_KEY"},
    ]})
    db_issues, _api_issues, _prd_issues = QaAgent._check_cross_artifacts(db, {"endpoints": []}, {})
    fk_issues = [issue for issue in db_issues if "FK 참조 대상" in issue]
    assert fk_issues == ["auth_sessions.approved_by_manager_id의 FK 참조 대상을 찾을 수 없습니다"]


def test_each_endpoint_of_a_feature_uses_the_table_its_own_route_names():
    # 한 기능이 두 리소스(/invitations, /members/{id}/role)를 다루면 기능 대표 테이블 하나로는
    # 한쪽 엔드포인트 본문이 다른 테이블 필드로 채워진다. API와 QA가 경로별로 같은 테이블을 골라야 한다.
    from phase2.agents.api_agent import finalize_api_contracts
    from phase2.agents.qa_agent import _endpoint_table, _request_field_names

    ownership = {"scope": "USER", "ownerEntity": "users", "ownerKey": "user_id"}
    tables = _contract_tables("users", "workspaces", "workspace_members", "workspace_invitations")
    next(t for t in tables if t["name"] == "workspace_members")["columns"] += [
        {"name": "role", "type": "VARCHAR(20)", "constraints": "NOT_NULL DEFAULT 'MEMBER'"},
    ]
    next(t for t in tables if t["name"] == "workspace_invitations")["columns"] += [
        {"name": "invitee_email", "type": "VARCHAR(320)", "constraints": "NOT_NULL"},
    ]
    registry = [{
        "name": "작업공간 역할 관리", "featureId": "feature_002.workspace_roles",
        "id": "feature_002.workspace_roles", "ownership": ownership,
        "apiContract": [
            {"method": "POST", "path": "/api/v1/workspaces/{workspaceid}/invitations"},
            {"method": "PATCH", "path": "/api/v1/workspaces/{workspaceid}/members/{memberid}/role"},
        ],
        "dbContract": {"tables": ["users", "workspaces", "workspace_members", "workspace_invitations"], "foreignKeys": []},
    }]
    mappings = build_feature_mappings(tables, registry)
    assert mappings[0]["table"] == "workspace_invitations"
    schema = json.dumps({"tables": tables, "featureMappings": mappings}, ensure_ascii=False)

    endpoints = finalize_api_contracts([
        {"method": "POST", "path": "/api/v1/workspaces/{workspaceid}/invitations",
         "featureId": "feature_002.workspace_roles", "description": "초대"},
        {"method": "PATCH", "path": "/api/v1/workspaces/{workspaceid}/members/{memberid}/role",
         "featureId": "feature_002.workspace_roles", "description": "역할 변경"},
    ], schema, registry=registry)

    by_name = {item["name"]: item for item in tables}
    resolved = {item["path"].rsplit("/", 1)[-1]: _endpoint_table(item, by_name, mappings)[0] for item in endpoints}
    assert resolved == {"invitations": "workspace_invitations", "role": "workspace_members"}
    role = next(item for item in endpoints if item["path"].endswith("/role"))
    assert "role" in _request_field_names(role["requestBody"])
    assert "invitee_email" not in _request_field_names(role["requestBody"])


def test_route_segment_shared_by_two_contract_tables_prefers_the_parent_named_in_the_route():
    ownership = {"scope": "USER", "ownerEntity": "users", "ownerKey": "user_id"}
    tables = _contract_tables("users", "teams", "team_members", "workspace_members")
    spec = {
        "name": "팀 멤버 관리", "featureId": "team.members", "ownership": ownership,
        "apiContract": [{"method": "POST", "path": "/api/v1/teams/{teamid}/members"}],
        "dbContract": {"tables": ["users", "teams", "team_members", "workspace_members"], "foreignKeys": []},
    }
    assert _mapped_table(build_feature_mappings(tables, [spec]), "team.members") == "team_members"


def test_put_contract_does_not_add_a_duplicate_of_the_published_patch_route():
    from phase2.agents.api_agent import (
        _annotate_feature_ids,
        _ensure_contract_endpoints,
    )

    registry = [{"featureId": "feature_002", "name": "담당자 배정", "apiContract": [
        {"method": "PUT", "path": "/api/v1/tasks/{taskid}/assignees", "action": "assign"},
    ]}]
    plan = [{"method": "PATCH", "path": "/api/v1/tasks/{taskid}/assignees", "description": "담당자 지정"}]

    ensured = _annotate_feature_ids(_ensure_contract_endpoints(plan, registry), registry)

    assert [(item["method"], item["path"], item.get("featureId")) for item in ensured] == [
        ("PATCH", "/api/v1/tasks/{taskid}/assignees", "feature_002"),
    ]


def test_pre_auth_routes_cover_common_recovery_and_verification_names():
    from phase2.agent_contract import is_pre_auth_route

    for path in ("/api/v1/auth/forgot-password", "/auth/email-verifications", "/api/v1/auth/verify-email",
                 "/api/v1/auth/password-reset-requests", "/api/v1/auth/register"):
        assert is_pre_auth_route(path), path
    for path in ("/api/v1/auth/logout", "/api/v1/auth/password-change", "/api/v1/profile", "/api/v1/auth/me"):
        assert not is_pre_auth_route(path), path


def test_duplicate_key_is_required_only_on_the_table_the_feature_route_names():
    # 운영 재현: '중복 대여 차단' 기능이 dbContract에 함께 적은 상위 테이블(tools)과,
    # 같은 대상을 가리키는 두 FK(created_by_user_id, user_id)에 UNIQUE가 요구되어 생성이 차단됐다.
    from phase2.quality_rules import scoped_unique_columns

    def table(name, *columns):
        return {"name": name, "description": "", "indexes": [], "columns": [
            {"name": "id", "type": "BIGINT", "constraints": "PRIMARY_KEY"},
            *[{"name": column, "type": "BIGINT", "constraints": f"NOT_NULL FOREIGN_KEY REFERENCES {ref}(id)"}
              for column, ref in columns],
        ]}

    tools = table("tools", ("created_by_user_id", "users"), ("category_id", "categories"))
    same_target = table("reservations", ("class_schedule_id", "class_schedules"), ("schedule_id", "class_schedules"))
    rentals = table("rentals", ("tool_id", "tools"), ("member_id", "users"))
    tables = [table("users"), table("categories"), table("class_schedules"), tools, same_target, rentals]
    rental = {
        "featureId": "feature_002", "name": "공구 대여",
        "description": "회원이 공구를 대여하면 시스템은 중복 대여를 차단한다.", "requirements": [],
        "apiContract": [{"method": "POST", "path": "/api/v1/rentals"}],
        "dbContract": {"tables": ["users", "tools", "rentals"], "foreignKeys": []},
    }
    reservation = {
        "featureId": "feature_003", "name": "수강 예약",
        "description": "동일 일정에 중복 예약하지 못하도록 방지한다.", "requirements": [],
        "apiContract": [{"method": "POST", "path": "/api/v1/reservations"}],
        "dbContract": {"tables": ["reservations", "class_schedules"], "foreignKeys": []},
    }
    features = [rental, reservation]

    assert scoped_unique_columns(rentals, tables, features) == ("tool_id", "member_id")
    assert scoped_unique_columns(tools, tables, features) == ()
    assert scoped_unique_columns(same_target, tables, features) == ()
