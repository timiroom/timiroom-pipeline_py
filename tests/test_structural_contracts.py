import json
import re

from phase2.agents.api_agent import (
    _align_endpoints_to_db,
    _ensure_feature_endpoint_groups,
    _is_garbage_endpoint,
)
from phase2.agents.dba_agent import (
    _synthesize_relationships,
    enforce_schema_contracts,
    ensure_domain_columns,
    reconcile_fk_types,
    sanitize_tables,
)
from phase2.agents.prd_agent import PrdAgent
from phase2.agents.qa_agent import QaAgent
from phase2.feature_coverage import uncovered_features
from phase3.schema_validator import SchemaValidator
from routers.chat import _extract_question_from_raw, _is_valid_question


def _table(name, columns):
    return {
        "name": name,
        "description": name,
        "columns": [
            {"name": column_name, "type": column_type, "constraints": constraints}
            for column_name, column_type, constraints in columns
        ],
        "indexes": [],
    }


def _schema_tables():
    return [
        _table("users", [("id", "BIGINT", "PRIMARY_KEY"), ("password_hash", "VARCHAR(255)", "NULL")]),
        _table("ingredients", [
            ("id", "BIGINT", "PRIMARY_KEY"), ("name", "VARCHAR(255)", "NOT_NULL"),
            ("quantity", "DECIMAL(10,2)", "NOT_NULL"), ("unit", "VARCHAR(30)", "NOT_NULL"),
            ("expiry_date", "DATE", "NULL"),
        ]),
        _table("inventory_items", [
            ("id", "BIGINT", "PRIMARY_KEY"), ("user_id", "BIGINT", "NOT_NULL"),
            ("ingredient_id", "BIGINT", "NOT_NULL"), ("quantity", "DECIMAL(10,2)", "NOT_NULL"),
            ("unit", "VARCHAR(30)", "NOT_NULL"), ("expiry_date", "DATE", "NULL"),
        ]),
        _table("consumption_records", [
            ("id", "BIGINT", "PRIMARY_KEY"), ("user_id", "BIGINT", "NOT_NULL"),
            ("ingredient_id", "BIGINT", "NOT_NULL"), ("quantity_used", "DECIMAL(10,2)", "NOT_NULL"),
        ]),
        _table("notifications", [
            ("id", "BIGINT", "PRIMARY_KEY"), ("user_id", "BIGINT", "NOT_NULL"),
            ("status", "VARCHAR(30)", "NOT_NULL"), ("scheduled_at", "TIMESTAMPTZ", "NOT_NULL"),
        ]),
    ]


def test_user_owned_date_range_resource_is_not_collapsed_into_referencing_aggregate():
    schema = json.dumps({"tables": [
        _table("exam_periods", [
            ("id", "BIGINT", "PRIMARY_KEY"),
            ("user_id", "BIGINT", "NOT_NULL FOREIGN_KEY REFERENCES users(id)"),
            ("start_date", "DATE", "NOT_NULL"),
            ("end_date", "DATE", "NOT_NULL"),
        ]),
        _table("scheduled_collections", [
            ("id", "BIGINT", "PRIMARY_KEY"),
            ("user_id", "BIGINT", "NOT_NULL FOREIGN_KEY REFERENCES users(id)"),
            ("exam_period_id", "BIGINT", "NULL FOREIGN_KEY REFERENCES exam_periods(id)"),
            ("status", "VARCHAR(30)", "NOT_NULL"),
        ]),
    ]}, ensure_ascii=False)
    endpoints = _align_endpoints_to_db([{
        "method": "GET",
        "path": "/api/v1/exam-periods",
        "description": "시험 기간 감지: 목록 조회",
        "featureName": "시험 기간 감지",
        "authRequired": False,
    }], schema)

    assert endpoints[0]["path"] == "/api/v1/exam-periods"


def test_final_db_sanitizer_drops_malformed_and_duplicate_indexes_and_normalizes_boolean():
    table = _table("task_priorities", [
        ("id", "BIGINT", "PRIMARY_KEY"),
        ("task_priority_level", "INTEGER", "NOT_NULL CHECK (task_priority_priority_level >= 1)"),
        ("is_exam", "INTEGER", "NOT_NULL CHECK (is_exam >= 0 OR is_exam = 1)"),
    ])
    table["indexes"] = [
        "CREATE INDEX idx_bad ON task_priorities(id) WHERE status('auto'",
        "CREATE INDEX idx_priority ON task_priorities(task_priority_level)",
        "CREATE INDEX idx_priority ON task_priorities(is_exam)",
    ]

    repaired = sanitize_tables([table])[0]
    columns = {column["name"]: column for column in repaired["columns"]}

    assert repaired["indexes"] == ["CREATE INDEX idx_priority ON task_priorities(task_priority_level)"]
    assert "task_priority_priority_level" not in columns["task_priority_level"]["constraints"]
    assert columns["is_exam"]["type"] == "BOOLEAN"


def test_schema_contracts_separate_catalog_state_and_event_target():
    tables = reconcile_fk_types(enforce_schema_contracts(
        _schema_tables(), ["유통기한 임박 푸시 알림"], "웹 푸시 알림을 제공한다",
    ))
    by_name = {table["name"]: table for table in tables}
    ingredient_columns = {column["name"] for column in by_name["ingredients"]["columns"]}
    event_columns = {column["name"] for column in by_name["consumption_records"]["columns"]}

    assert not {"quantity", "unit", "expiry_date"} & ingredient_columns
    assert "inventory_item_id" in event_columns
    assert "ingredient_id" not in event_columns
    assert not {"refresh_tokens", "notification_preferences", "push_devices"} & set(by_name)
    assert next(c for c in by_name["users"]["columns"] if c["name"] == "password_hash")["constraints"] == "NULL"
    assert any("(status, scheduled_at)" in index for index in by_name["notifications"]["indexes"])


def test_notification_schema_is_not_retargeted_without_explicit_contract():
    tables = _schema_tables()
    notifications = next(table for table in tables if table["name"] == "notifications")
    notifications["columns"].append({"name": "ingredient_id", "type": "BIGINT", "constraints": "NOT_NULL"})
    notifications["indexes"] = [
        "CREATE INDEX INDEX broken ON wrong_table(user_id)",
        "CREATE INDEX idx_notifications_user_id ON notifications(user_id)",
    ]
    tables = reconcile_fk_types(enforce_schema_contracts(sanitize_tables(tables), ["알림"], ""))
    notifications = next(table for table in tables if table["name"] == "notifications")
    names = {column["name"] for column in notifications["columns"]}
    assert "inventory_item_id" not in names
    assert "ingredient_id" in names
    assert all("wrong_table" not in index for index in notifications["indexes"])


def test_notification_preference_contract_adds_canonical_enabled_field():
    table = _table("notifications", [
        ("id", "BIGINT", "PRIMARY_KEY"),
        ("user_id", "BIGINT", "NOT_NULL"),
        ("is_received", "BOOLEAN", "DEFAULT_TRUE"),
    ])
    table["description"] = "사용자별 알림 전송과 수신 동의를 관리한다"
    prd = json.dumps({"coreFeatures": [{
        "name": "알림 수신 설정", "description": "사용자가 알림 수신 설정을 변경한다",
        "requirements": ["알림 수신 설정 저장"],
    }]}, ensure_ascii=False)

    result = enforce_schema_contracts([table], ["알림 수신 설정"], prd)
    names = {column["name"] for column in result[0]["columns"]}

    assert "is_enabled" in names


def test_compound_table_accepts_unique_short_fk_aliases():
    tables = [
        _table("lecture_schedules", [("id", "BIGINT", "PRIMARY_KEY")]),
        _table("attendance_records", [
            ("id", "BIGINT", "PRIMARY_KEY"),
            ("schedule_id", "INTEGER", "NOT_NULL"),
            ("lecture_id", "INTEGER", "NOT_NULL"),
        ]),
    ]
    result = reconcile_fk_types(tables)
    columns = {column["name"]: column for column in result[1]["columns"]}

    assert {"schedule_id", "lecture_id"} <= set(columns)
    assert all("REFERENCES lecture_schedules(id)" in columns[name]["constraints"] for name in ("schedule_id", "lecture_id"))


def test_reconcile_preserves_actor_ids_and_blocks_inferred_fk_cycle():
    tables = [
        _table("lecture_schedules", [
            ("id", "BIGINT", "PRIMARY_KEY"),
            ("course_id", "BIGINT", "NOT_NULL"),
            ("instructor_id", "BIGINT", "NOT_NULL"),
        ]),
        _table("course_reservations", [
            ("id", "BIGINT", "PRIMARY_KEY"),
            ("lecture_schedule_id", "BIGINT", "FOREIGN_KEY REFERENCES lecture_schedules(id)"),
            ("student_id", "BIGINT", "NOT_NULL"),
        ]),
    ]

    result = reconcile_fk_types(tables)
    by_name = {table["name"]: table for table in result}
    schedule_columns = {column["name"]: column for column in by_name["lecture_schedules"]["columns"]}
    reservation_columns = {column["name"]: column for column in by_name["course_reservations"]["columns"]}

    assert "course_id" in schedule_columns
    assert "instructor_id" in schedule_columns
    assert "student_id" in reservation_columns
    assert not {"students", "instructors"} & set(by_name)


def test_api_contract_remaps_catalog_and_defines_empty_list_and_compensation():
    tables = reconcile_fk_types(enforce_schema_contracts(_schema_tables(), [], ""))
    schema = json.dumps({"tables": tables}, ensure_ascii=False)
    endpoints = _align_endpoints_to_db([
        {"method": "GET", "path": "/api/v1/ingredients", "description": "재료 추가 목록", "authRequired": True},
        {"method": "POST", "path": "/api/v1/consumption-records", "description": "소비 기록 생성", "authRequired": True},
        {"method": "PATCH", "path": "/api/v1/consumption-records", "description": "소비 기록 수정", "authRequired": True},
        {"method": "DELETE", "path": "/api/v1/consumption-records", "description": "소비 기록 삭제", "authRequired": True},
    ], schema)
    get_endpoint = endpoints[0]
    assert get_endpoint["path"] == "/api/v1/inventory-items"
    assert "[]" in get_endpoint["successResponse"]
    assert "404" not in get_endpoint["errorCodes"]
    for endpoint in endpoints[1:]:
        assert endpoint.get("transactionRules")
    assert endpoints[2]["path"].endswith("/{id}")
    realigned = _align_endpoints_to_db(endpoints, schema)
    assert [endpoint["path"] for endpoint in realigned] == [endpoint["path"] for endpoint in endpoints]


def test_api_contract_reconciles_different_english_translations_via_feature_meaning():
    table = _table("expiration_notifications", [
        ("id", "BIGINT", "PRIMARY_KEY"), ("scheduled_at", "TIMESTAMPTZ", "NOT_NULL"),
        ("status", "VARCHAR(30)", "NOT_NULL"),
    ])
    table["description"] = "유통기한 알림"
    schema = json.dumps({"tables": [table]}, ensure_ascii=False)
    endpoint = _align_endpoints_to_db([{
        "method": "GET", "path": "/api/v1/expiry-notifications",
        "description": "유통기한 알림: 목록 조회", "authRequired": False,
    }], schema)[0]

    assert endpoint["path"] == "/api/v1/expiration-notifications"
    assert "[]" in endpoint["successResponse"]


def test_cross_document_qa_detects_semantic_contract_violations():
    qa = object.__new__(QaAgent)
    tables = reconcile_fk_types(enforce_schema_contracts(_schema_tables(), [], ""))
    prd = {"releaseSchedule": [{"description": "사용자의 번거로움을 해소하는."}]}
    api = {"endpoints": [{
        "method": "GET", "path": "/api/v1/inventory-items",
        "successResponse": "items: array", "errorCodes": "404 — 없음",
    }]}
    db_issues, api_issues, prd_issues = qa._check_cross_document_semantics(
        prd, {"tables": tables}, api, [],
    )
    assert not db_issues
    assert any("404" in issue for issue in api_issues)
    assert any("빈 배열" in issue for issue in api_issues)
    assert not any("기준 엔티티 inventory_items" in issue for issue in api_issues)
    assert any("미완결" in issue for issue in prd_issues)


def test_cross_document_qa_allows_mutating_referenced_aggregate_root():
    qa = object.__new__(QaAgent)
    tables = [
        _table("lecture_schedules", [
            ("id", "BIGINT", "PRIMARY_KEY"),
            ("title", "VARCHAR(255)", "NOT_NULL"),
            ("start_time", "TIMESTAMPTZ", "NOT_NULL"),
        ]),
        _table("course_reservations", [
            ("id", "BIGINT", "PRIMARY_KEY"),
            ("lecture_schedule_id", "BIGINT", "FOREIGN_KEY REFERENCES lecture_schedules(id)"),
            ("status", "VARCHAR(30)", "NOT_NULL"),
        ]),
    ]
    api = {"endpoints": [{
        "method": "PATCH",
        "path": "/api/v1/lecture-schedules/{id}",
        "requestBody": "title: string, start_time: string",
        "successResponse": "id: integer",
        "errorCodes": "400, 404, 500",
    }]}

    _, api_issues, _ = qa._check_cross_document_semantics(
        {"releaseSchedule": []}, {"tables": tables}, api, [],
    )

    assert not any("상태 변경 API" in issue for issue in api_issues)


def test_cross_document_qa_recognizes_common_notification_preference_columns():
    qa = object.__new__(QaAgent)
    tables = [_table("notifications", [
        ("id", "BIGINT", "PRIMARY_KEY"),
        ("user_id", "BIGINT", "NOT_NULL"),
        ("is_enabled", "BOOLEAN", "NOT_NULL DEFAULT_TRUE"),
    ])]
    prd = {
        "projectOverview": "사용자는 알림 수신 설정을 변경할 수 있다.",
        "releaseSchedule": [],
    }

    db_issues, _, _ = qa._check_cross_document_semantics(
        prd, {"tables": tables}, {"endpoints": []}, [],
    )

    assert not any("사용자별 수신 설정 구조가 없음" in issue for issue in db_issues)


def test_cross_document_qa_rejects_state_field_written_to_catalog_path():
    qa = object.__new__(QaAgent)
    tables = [
        _table("products", [("id", "BIGINT", "PRIMARY_KEY"), ("name", "TEXT", "NOT_NULL")]),
        _table("stock_items", [
            ("id", "BIGINT", "PRIMARY_KEY"),
            ("product_id", "BIGINT", "FOREIGN_KEY REFERENCES products(id)"),
            ("quantity", "INTEGER", "NOT_NULL"),
        ]),
    ]
    api = {"endpoints": [{
        "method": "PATCH",
        "path": "/api/v1/products/{id}",
        "requestBody": "name: string, quantity: integer",
        "successResponse": "id: integer",
        "errorCodes": "400, 404, 500",
    }]}

    _, api_issues, _ = qa._check_cross_document_semantics(
        {"releaseSchedule": []}, {"tables": tables}, api, [],
    )

    assert any("quantity" in issue for issue in api_issues)


def test_cross_document_qa_detects_fk_cycle_and_missing_action_subject():
    qa = object.__new__(QaAgent)
    tables = [
        _table("schedules", [
            ("id", "BIGINT", "PRIMARY_KEY"),
            ("reservation_id", "BIGINT", "FOREIGN_KEY REFERENCES reservations(id)"),
        ]),
        _table("reservations", [
            ("id", "BIGINT", "PRIMARY_KEY"),
            ("schedule_id", "BIGINT", "FOREIGN_KEY REFERENCES schedules(id)"),
            ("status", "VARCHAR(20)", "NOT_NULL"),
        ]),
    ]

    db_issues, _, _ = qa._check_cross_document_semantics(
        {"releaseSchedule": []}, {"tables": tables}, {"endpoints": []}, [],
    )

    assert any("상호 참조 순환 FK" in issue for issue in db_issues)
    assert not any("행위 주체 식별 컬럼 누락" in issue for issue in db_issues)


def test_general_semantic_contracts_project_fields_unique_and_predecessor_fk():
    tables = [
        _table("bicycles", [("id", "BIGINT", "PRIMARY_KEY"), ("name", "TEXT", "NOT_NULL")]),
        _table("bicycle_rentals", [
            ("id", "BIGINT", "PRIMARY_KEY"), ("user_id", "BIGINT", "NOT_NULL"),
            ("bicycle_id", "BIGINT", "NOT_NULL"),
        ]),
        _table("return_records", [
            ("id", "BIGINT", "PRIMARY_KEY"), ("user_id", "BIGINT", "NOT_NULL"),
            ("bicycle_id", "BIGINT", "NOT_NULL"),
        ]),
    ]
    tables[1]["description"] = "자전거 대여"
    tables[2]["description"] = "자전거 반납 기록"
    prd = json.dumps({"coreFeatures": [
        {"name": "자전거 대여", "description": "중복 대여를 방지합니다", "requirements": ["대여 시작 시간과 종료 시간을 입력합니다"]},
        {"name": "자전거 반납", "description": "대여 후 반납 상태를 기록합니다", "requirements": []},
    ]}, ensure_ascii=False)

    result = enforce_schema_contracts(tables, ["자전거 대여", "자전거 반납"], prd)
    by_name = {table["name"]: table for table in result}
    rental_names = {column["name"] for column in by_name["bicycle_rentals"]["columns"]}
    return_names = {column["name"] for column in by_name["return_records"]["columns"]}

    assert {"start_time", "end_time"} <= rental_names
    assert any("CREATE UNIQUE INDEX" in index for index in by_name["bicycle_rentals"]["indexes"])
    assert "bicycle_rental_id" in return_names
    return_fk = next(
        column for column in by_name["return_records"]["columns"]
        if column["name"] == "bicycle_rental_id"
    )
    assert "REFERENCES bicycle_rentals(id)" in return_fk["constraints"]


def test_api_alignment_applies_prd_auth_and_business_transaction_rules():
    rentals = _table("bicycle_rentals", [
        ("id", "BIGINT", "PRIMARY_KEY"), ("user_id", "BIGINT", "NOT_NULL"),
        ("bicycle_id", "BIGINT", "NOT_NULL"),
    ])
    rentals["indexes"] = [
        "CREATE UNIQUE INDEX uq_bicycle_rentals_bicycle_id_user_id ON bicycle_rentals(bicycle_id, user_id)"
    ]
    schema = json.dumps({"tables": [rentals]}, ensure_ascii=False)
    prd = json.dumps({"techStack": {"auth": "JWT 역할 기반 인증"}}, ensure_ascii=False)

    endpoint = _align_endpoints_to_db([{
        "method": "POST", "path": "/api/v1/bicycle-rentals",
        "description": "자전거 대여", "authRequired": False,
    }], schema, prd)[0]

    assert endpoint["authRequired"] is False
    assert "401" not in endpoint["errorCodes"]
    assert "중복" in endpoint["transactionRules"]
    assert "409" in endpoint["errorCodes"]


def test_api_alignment_adds_operation_contract_for_plain_mutation():
    table = _table("tasks", [
        ("id", "BIGINT", "PRIMARY_KEY"),
        ("title", "TEXT", "NOT_NULL"),
    ])
    endpoint = _align_endpoints_to_db([{
        "method": "PATCH", "path": "/api/v1/tasks/{id}",
        "description": "할 일 제목 수정", "authRequired": True,
    }], json.dumps({"tables": [table]}, ensure_ascii=False))[0]

    assert "단일 DB 트랜잭션" in endpoint["transactionRules"]
    assert "401" in endpoint["errorCodes"]
    assert "404" in endpoint["errorCodes"]
    assert "title: string" in endpoint["requestBody"]


def test_api_repair_maps_domain_behavior_tables_and_normalizes_put():
    from phase2.agents.api_agent import (
        _ensure_feature_endpoint_groups,
        _normalize_endpoints,
    )

    tables = [
        _table("tasks", [("id", "BIGINT", "PRIMARY_KEY")]),
        _table("task_assignments", [
            ("id", "BIGINT", "PRIMARY_KEY"),
            ("task_id", "BIGINT", "FOREIGN_KEY REFERENCES tasks(id)"),
        ]),
        _table("completion_records", [
            ("id", "BIGINT", "PRIMARY_KEY"),
            ("task_id", "BIGINT", "FOREIGN_KEY REFERENCES tasks(id)"),
        ]),
    ]
    tables[1]["description"] = "할 일과 사용자 사이 담당자 배정"
    tables[2]["description"] = "완료 상태 기록과 변경 이력"
    schema = json.dumps({"tables": tables}, ensure_ascii=False)
    existing = _normalize_endpoints([{
        "method": "PUT", "path": "/api/v1/tasks/{id}/assignee",
        "description": "담당자 배정", "authRequired": True,
    }])
    repaired = _ensure_feature_endpoint_groups(
        existing, schema, ["담당자 배정", "완료 상태 기록"], "{}",
    )

    assert existing[0]["method"] == "PATCH"
    assignment_methods = {
        endpoint["method"] for endpoint in repaired
        if "담당자 배정" in endpoint.get("description", "")
    }
    completion_methods = {
        endpoint["method"] for endpoint in repaired
        if "완료 상태 기록" in endpoint.get("description", "")
    }
    assert {"GET", "POST", "PATCH"} <= assignment_methods
    assert {"GET", "POST", "PATCH"} <= completion_methods


def test_existing_behavior_endpoint_is_tagged_for_qa_coverage():
    table = _table("task_assignments", [
        ("id", "BIGINT", "PRIMARY_KEY"),
        ("task_id", "BIGINT", "FOREIGN_KEY REFERENCES tasks(id)"),
    ])
    table["description"] = "할 일과 사용자 사이 담당자 배정"
    endpoints = [{
        "method": "PATCH", "path": "/api/v1/task-assignments/{id}",
        "description": "관계 일부 수정", "authRequired": True,
    }]

    repaired = _ensure_feature_endpoint_groups(
        endpoints, json.dumps({"tables": [table]}, ensure_ascii=False), ["담당자 배정"], "{}",
    )

    patch_endpoint = next(endpoint for endpoint in repaired if endpoint["method"] == "PATCH")
    assert patch_endpoint["featureName"] == "담당자 배정"


def test_api_repair_uses_prd_feature_evidence_and_avoids_principal_table():
    users = _table("users", [("id", "BIGINT", "PRIMARY_KEY")])
    settings = _table("notification_settings", [
        ("id", "BIGINT", "PRIMARY_KEY"),
        ("user_id", "BIGINT", "NOT_NULL"),
        ("expiration_alert_enabled", "BOOLEAN", "NOT_NULL"),
    ])
    settings["description"] = "사용자의 알림 수신 여부와 알림 유형 설정을 저장한다"
    feature = "알림 수신 설정 관리"
    prd = json.dumps({"coreFeatures": [{
        "name": feature,
        "parentFeature": "유통기한 알림",
        "rationale": "사용자가 알림 수신 여부를 직접 제어하기 위해 필요하다",
        "actions": ["알림 수신 설정을 조회하고 변경한다"],
        "dataRequirements": ["알림 활성화 여부", "사용자 식별자"],
    }]}, ensure_ascii=False)

    repaired = _ensure_feature_endpoint_groups(
        [], json.dumps({"tables": [users, settings]}, ensure_ascii=False), [feature], prd,
    )

    assert {endpoint["method"] for endpoint in repaired} == {"GET", "PATCH"}
    assert all("/notification-settings" in endpoint["path"] for endpoint in repaired)


def test_api_repair_does_not_reuse_endpoint_tagged_for_another_feature():
    records = _table("activity_records", [
        ("id", "BIGINT", "PRIMARY_KEY"), ("status", "VARCHAR(20)", "NOT_NULL"),
    ])
    records["description"] = "활동 기록 누락 감지와 활동 기록 조회"
    existing = [{
        "method": "GET", "path": "/api/v1/activity-records",
        "description": "활동 기록: 목록 조회", "featureName": "활동 기록",
        "authRequired": True,
    }]

    repaired = _ensure_feature_endpoint_groups(
        existing, json.dumps({"tables": [records]}, ensure_ascii=False),
        ["활동 기록 누락 감지"], "{}",
    )

    assert any(
        endpoint.get("featureName") == "활동 기록 누락 감지"
        and "활동 기록 누락 감지" in endpoint.get("description", "")
        for endpoint in repaired
    )


def test_phase3_api_coverage_accepts_explicit_feature_tag():
    api = json.dumps({"endpoints": [{
        "method": "GET", "path": "/api/v1/activity-records",
        "featureName": "활동 기록 누락 감지",
        "description": "감지 결과 목록 조회",
    }]}, ensure_ascii=False)

    assert SchemaValidator()._check_api_coverage(api, ["활동 기록 누락 감지"]) == []


def test_business_endpoint_does_not_bind_to_credential_table_by_path_alone():
    users = _table("users", [
        ("id", "BIGINT", "PRIMARY_KEY"),
        ("login_id", "VARCHAR(255)", "NOT_NULL"),
        ("credential_hash", "VARCHAR(255)", "NOT_NULL"),
    ])
    tasks = _table("tasks", [
        ("id", "BIGINT", "PRIMARY_KEY"),
        ("title", "TEXT", "NOT_NULL"),
        ("status", "VARCHAR(20)", "NOT_NULL"),
    ])
    tasks["description"] = "할 일 등록과 상태 관리"
    endpoint = _align_endpoints_to_db([{
        "method": "POST", "path": "/api/v1/users",
        "description": "할 일 등록: 새 업무 생성", "authRequired": True,
    }], json.dumps({"tables": [users, tasks]}, ensure_ascii=False))[0]

    assert endpoint["path"] == "/api/v1/tasks"
    assert "title: string" in endpoint["requestBody"]


def test_relationship_roles_are_normalized_and_api_contract_is_concrete():
    tables = [
        _table("tasks", [
            ("id", "BIGINT", "PRIMARY_KEY"),
            ("title", "TEXT", "NOT_NULL"),
            ("assigned_staff_id", "BIGINT", "FOREIGN_KEY REFERENCES assigned_staff(id)"),
        ]),
        _table("assigned_staff", [
            ("id", "BIGINT", "PRIMARY_KEY"),
            ("staff_id", "BIGINT", "FOREIGN_KEY REFERENCES staffs(id)"),
            ("assigned_by", "BIGINT", "NOT_NULL"),
        ]),
        _table("staffs", [("id", "BIGINT", "PRIMARY_KEY")]),
    ]
    repaired = reconcile_fk_types(tables)
    by_name = {table["name"]: table for table in repaired}
    task_columns = {column["name"]: column for column in by_name["tasks"]["columns"]}
    assignment_columns = {column["name"]: column for column in by_name["assigned_staff"]["columns"]}
    assert "REFERENCES assigned_staff(id)" in task_columns["assigned_staff_id"]["constraints"]
    assert "task_id" not in assignment_columns
    assert "REFERENCES" not in assignment_columns["assigned_by"]["constraints"]

    schema = json.dumps({"tables": repaired}, ensure_ascii=False)
    endpoint = _align_endpoints_to_db([{
        "method": "POST", "path": "/api/v1/assignees",
        "description": "담당자 배정", "authRequired": True,
        "requestBody": "payload: object — 기능별 입력 필드",
    }], schema)[0]
    assert endpoint["path"] == "/api/v1/assigned-staff"
    assert "payload: object" not in endpoint["requestBody"]
    assert "staff_id" in endpoint["requestBody"]


def test_qa_detects_relationship_direction_and_concrete_api_field_mismatch():
    qa = object.__new__(QaAgent)
    tables = [
        _table("tasks", [("id", "BIGINT", "PRIMARY_KEY"), ("title", "TEXT", "NOT_NULL")]),
        _table("task_records", [
            ("id", "BIGINT", "PRIMARY_KEY"),
            ("task_id", "BIGINT", "FOREIGN_KEY REFERENCES tasks(id)"),
        ]),
    ]
    db_issues, api_issues, _ = qa._check_cross_document_semantics(
        {"releaseSchedule": []},
        {"tables": tables, "relationships": ["task_records (1:N) tasks"]},
        {"endpoints": [{
            "method": "POST", "path": "/api/v1/tasks", "authRequired": False,
            "requestBody": "payload: object — 기능별 입력 필드",
            "successResponse": "id: integer", "errorCodes": "400, 500",
        }]},
        [],
    )
    assert any("역할·방향" in issue for issue in db_issues)
    assert any("payload" in issue for issue in api_issues)
    assert any("필수 필드" in issue for issue in api_issues)


def test_reconcile_removes_root_to_event_reverse_fk_and_keeps_assignment_actor():
    tables = [
        _table("tasks", [
            ("id", "BIGINT", "PRIMARY_KEY"),
            ("assignee_id", "BIGINT", "FOREIGN_KEY REFERENCES assignees(id)"),
            ("status_id", "BIGINT", "FOREIGN_KEY REFERENCES completed_statuses(id)"),
            ("status", "VARCHAR(30)", "NOT_NULL"),
        ]),
        _table("assigned_personnel", [
            ("id", "BIGINT", "PRIMARY_KEY"),
            ("task_id", "BIGINT", "FOREIGN_KEY REFERENCES tasks(id)"),
        ]),
        _table("completed_statuses", [
            ("id", "BIGINT", "PRIMARY_KEY"),
            ("task_id", "BIGINT", "FOREIGN_KEY REFERENCES tasks(id)"),
            ("status", "VARCHAR(30)", "NOT_NULL"),
        ]),
        _table("assignees", [("id", "BIGINT", "PRIMARY_KEY")]),
    ]
    repaired = reconcile_fk_types(tables)
    by_name = {table["name"]: table for table in repaired}
    task_names = {column["name"] for column in by_name["tasks"]["columns"]}
    assignment = {column["name"]: column for column in by_name["assigned_personnel"]["columns"]}
    completed = {column["name"] for column in by_name["completed_statuses"]["columns"]}
    assert not ({"status_id"} <= task_names and {"task_id"} <= completed)
    assert "personnel_id" not in assignment


def test_reconcile_removes_completion_status_naming_variant_reverse_fk():
    tables = [
        _table("tasks", [
            ("id", "BIGINT", "PRIMARY_KEY"),
            ("completion_status_id", "BIGINT", "FOREIGN_KEY REFERENCES completion_statuses(id)"),
        ]),
        _table("completion_statuses", [
            ("id", "BIGINT", "PRIMARY_KEY"),
            ("task_id", "BIGINT", "FOREIGN_KEY REFERENCES tasks(id)"),
            ("status", "VARCHAR(30)", "NOT_NULL"),
        ]),
    ]

    repaired = reconcile_fk_types(tables)
    by_name = {table["name"]: table for table in repaired}
    task_columns = {column["name"] for column in by_name["tasks"]["columns"]}

    assert not (
        "completion_status_id" in task_columns
        and "tasks (1:N) completion_statuses" in _synthesize_relationships(repaired)
    )


def test_actor_principal_is_not_invented_from_table_name():
    tables = [
        _table("tasks", [
            ("id", "BIGINT", "PRIMARY_KEY"),
            ("assigned_personnel_id", "BIGINT", "FOREIGN_KEY REFERENCES assigned_personnel(id)"),
        ]),
        _table("assigned_personnel", [("id", "BIGINT", "PRIMARY_KEY")]),
    ]
    repaired = reconcile_fk_types(tables)
    by_name = {table["name"]: table for table in repaired}
    assert "personnel" not in by_name
    assignment = {column["name"]: column for column in by_name["assigned_personnel"]["columns"]}
    assert "task_id" not in assignment
    assert "personnel_id" not in assignment
    tasks = {column["name"]: column for column in by_name["tasks"]["columns"]}
    assert "REFERENCES assigned_personnel(id)" in tasks["assigned_personnel_id"]["constraints"]
    relationships = _synthesize_relationships(repaired)
    assert "personnel (1:N) tasks" not in relationships
    assert "assigned_personnel (1:N) tasks" in relationships


def test_reconcile_breaks_explicit_root_assignment_cycle():
    tables = [
        _table("tasks", [
            ("id", "BIGINT", "PRIMARY_KEY"),
            ("assigned_personnel_id", "BIGINT", "FOREIGN_KEY REFERENCES assigned_personnel(id)"),
        ]),
        _table("assigned_personnel", [
            ("id", "BIGINT", "PRIMARY_KEY"),
            ("task_id", "BIGINT", "FOREIGN_KEY REFERENCES tasks(id)"),
            ("assigned_user_id", "BIGINT", "FOREIGN_KEY REFERENCES users(id)"),
        ]),
        _table("users", [("id", "BIGINT", "PRIMARY_KEY")]),
    ]

    repaired = reconcile_fk_types(tables)
    by_name = {table["name"]: table for table in repaired}
    task_columns = {column["name"]: column for column in by_name["tasks"]["columns"]}
    relationships = _synthesize_relationships(repaired)

    assert "assigned_personnel_id" not in task_columns
    assignment_columns = {
        column["name"]: column for column in by_name["assigned_personnel"]["columns"]
    }
    assert "REFERENCES users(id)" in assignment_columns["assigned_user_id"]["constraints"]
    assert "tasks (1:N) assigned_personnel" in relationships
    assert "assigned_personnel (1:N) tasks" not in relationships


def test_reconcile_does_not_create_missing_actor_principals():
    tables = [
        _table("tasks", [
            ("id", "BIGINT", "PRIMARY_KEY"),
            ("assignee_name", "VARCHAR(100)", "NOT_NULL"),
        ]),
        _table("assigned_personnel", [
            ("id", "BIGINT", "PRIMARY_KEY"),
            ("task_id", "BIGINT", "FOREIGN_KEY REFERENCES tasks(id)"),
            ("assignee_id", "BIGINT", "FOREIGN_KEY REFERENCES assignees(id)"),
        ]),
        _table("completed_status_records", [
            ("id", "BIGINT", "PRIMARY_KEY"),
            ("assigned_personnel_id", "BIGINT", "FOREIGN_KEY REFERENCES assigned_personnel(id)"),
            ("recorded_by", "BIGINT", "FOREIGN_KEY REFERENCES users(id)"),
        ]),
        _table("assignees", [("id", "BIGINT", "PRIMARY_KEY")]),
    ]

    repaired = reconcile_fk_types(tables)
    by_name = {table["name"]: table for table in repaired}

    assert "users" not in by_name
    assert "personnel" not in by_name
    assert "assignee_name" in {
        column["name"] for column in by_name["tasks"]["columns"]
    }
    recorded_by = next(column for column in by_name["completed_status_records"]["columns"] if column["name"] == "recorded_by")
    assert "REFERENCES users(id)" in recorded_by["constraints"]


def test_qa_detects_reference_to_missing_table():
    qa = object.__new__(QaAgent)
    db_issues, _, _ = qa._check_cross_document_semantics(
        {"coreFeatures": []},
        {"tables": [_table("events", [
            ("id", "BIGINT", "PRIMARY_KEY"),
            ("recorded_by", "BIGINT", "FOREIGN_KEY REFERENCES users(id)"),
        ])], "relationships": []},
        {"endpoints": []},
        [],
    )
    assert any("존재하지 않는 FK 대상" in issue for issue in db_issues)


def test_audit_by_column_is_not_bound_to_a_fixed_actor():
    tables = [
        _table("assignments", [
            ("id", "BIGINT", "PRIMARY_KEY"),
            ("assigned_by", "BIGINT", "NOT_NULL"),
            ("assignee_id", "BIGINT", "FOREIGN_KEY REFERENCES assignees(id)"),
        ]),
        _table("users", [("id", "BIGINT", "PRIMARY_KEY")]),
        _table("assignees", [("id", "BIGINT", "PRIMARY_KEY")]),
    ]

    repaired = reconcile_fk_types(tables)
    assignment = next(table for table in repaired if table["name"] == "assignments")
    assigned_by = next(column for column in assignment["columns"] if column["name"] == "assigned_by")

    assert "REFERENCES" not in assigned_by["constraints"]


def test_missing_feature_endpoint_group_is_built_from_matching_erd_table():
    schema = json.dumps({"tables": [
        {**_table("tasks", [("id", "BIGINT", "PRIMARY_KEY"), ("title", "TEXT", "NOT_NULL")]),
         "description": "할 일 등록"},
        {**_table("assigned_staffs", [
            ("id", "BIGINT", "PRIMARY_KEY"),
            ("task_id", "BIGINT", "FOREIGN_KEY REFERENCES tasks(id)"),
            ("staff_id", "BIGINT", "FOREIGN_KEY REFERENCES staffs(id)"),
        ]), "description": "담당자 배정"},
    ]}, ensure_ascii=False)
    existing = [{
        "method": "GET", "path": "/api/v1/tasks", "description": "할 일 등록: 목록 조회",
        "authRequired": True, "requestBody": "없음", "successResponse": "items: array",
        "errorCodes": "400, 401, 500",
    }]

    repaired = _ensure_feature_endpoint_groups(existing, schema, ["할 일 등록", "담당자 배정"])
    assignment_methods = {
        endpoint["method"] for endpoint in repaired
        if endpoint["path"].startswith("/api/v1/assigned-staffs")
    }

    assert assignment_methods == {"GET", "POST", "PATCH"}


def test_feature_description_overlap_does_not_replace_association_resource_contract():
    feature = "구성원 배정"
    schema = json.dumps({"tables": [
        {**_table("projects", [
            ("id", "BIGINT", "PRIMARY_KEY"),
            ("owner_id", "BIGINT", "FOREIGN_KEY REFERENCES users(id)"),
        ]), "description": "프로젝트 생성과 구성원 배정 현황 조회"},
        {**_table("project_memberships", [
            ("id", "BIGINT", "PRIMARY_KEY"),
            ("project_id", "BIGINT", "FOREIGN_KEY REFERENCES projects(id)"),
            ("user_id", "BIGINT", "FOREIGN_KEY REFERENCES users(id)"),
        ]), "description": feature},
        _table("users", [("id", "BIGINT", "PRIMARY_KEY")]),
    ]}, ensure_ascii=False)
    existing = [
        {
            "method": method,
            "path": "/api/v1/projects",
            "description": f"프로젝트 {method}: 구성원 배정 정보를 포함",
            "featureName": "",
            "authRequired": True,
        }
        for method in ("GET", "POST")
    ]

    repaired = _ensure_feature_endpoint_groups(existing, schema, [feature])
    membership_methods = {
        endpoint["method"] for endpoint in repaired
        if endpoint["path"].startswith("/api/v1/project-memberships")
    }

    assert membership_methods == {"GET", "POST", "PATCH"}


def test_collection_endpoint_owns_feature_when_detail_and_nested_actions_exist():
    feature = "프로젝트 등록"
    schema = json.dumps({"tables": [
        {**_table("projects", [
            ("id", "BIGINT", "PRIMARY_KEY"),
            ("title", "TEXT", "NOT_NULL"),
        ]), "description": feature},
    ]}, ensure_ascii=False)
    existing = [
        {"method": "GET", "path": "/api/v1/projects", "description": "프로젝트 목록"},
        {"method": "GET", "path": "/api/v1/projects/{projectId}", "description": "프로젝트 상세"},
        {"method": "POST", "path": "/api/v1/projects", "description": "프로젝트 생성"},
        {"method": "POST", "path": "/api/v1/projects/{projectId}/members", "description": "구성원 추가"},
    ]

    repaired = _ensure_feature_endpoint_groups(existing, schema, [feature])
    owners = {
        (endpoint["method"], endpoint["path"]): endpoint.get("featureName")
        for endpoint in repaired
    }

    assert owners[("GET", "/api/v1/projects")] == feature
    assert owners[("POST", "/api/v1/projects")] == feature
    assert not owners.get(("GET", "/api/v1/projects/{projectId}"))
    assert not owners.get(("POST", "/api/v1/projects/{projectId}/members"))


def test_put_with_named_path_parameter_is_normalized_to_patch():
    from phase2.agents.api_agent import _normalize_endpoints

    normalized = _normalize_endpoints([{
        "method": "PUT",
        "path": "/api/v1/resources/{resourceId}",
        "description": "리소스 수정",
    }])

    assert normalized[0]["method"] == "PATCH"


def test_postgresql_type_constraint_combinations_are_normalized():
    table = _table("events", [
        ("id", "BIGINT", "PRIMARY_KEY GENERATED IDENTITY"),
        ("starts_at", "TIMESTAMPTZ", "WITHOUT TIME ZONE NOT_NULL"),
        ("updated_at", "TIMESTAMP", "WITH TIME ZONE DEFAULT NOW()"),
    ])

    result = ensure_domain_columns([table])[0]
    columns = {column["name"]: column for column in result["columns"]}

    assert "GENERATED_IDENTITY" in columns["id"]["constraints"]
    assert "WITHOUT TIME ZONE" not in columns["starts_at"]["constraints"]
    assert columns["updated_at"]["type"] == "TIMESTAMPTZ"
    assert "WITH TIME ZONE" not in columns["updated_at"]["constraints"]


def test_invalid_constraint_markers_fall_back_by_column_semantics():
    repaired = sanitize_tables([_table("assignments", [
        ("updated_at", "TIMESTAMPTZ", "both"),
        ("assigned_to", "TIMESTAMPTZ", "?"),
        ("description", "TEXT", ""),
    ])])[0]
    columns = {column["name"]: column for column in repaired["columns"]}

    assert columns["updated_at"]["constraints"] == "NOT_NULL DEFAULT_NOW"
    assert columns["assigned_to"]["constraints"] == "NULL"
    assert columns["description"]["constraints"] == "NULL"


def test_constraint_prose_is_removed_and_duplicate_columns_are_collapsed():
    repaired = sanitize_tables([_table("tasks", [
        ("id", "BIGINT", "PRIMARY_KEY GENERATED_IDENTITY"),
        ("id", "BIGINT", "PRIMARY_KEY GENERATED_IDENTITY"),
        ("title_title", "VARCHAR(255)", "NOT_NULL"),
        ("title", "VARCHAR(255)", "NOT_NULL"),
        ("status", "VARCHAR(20)", "NOT_NULL DEFAULT 'pending' CHECK (status IN ('pending', 'completed')) lose ownership here"),
        ("due_date", "TIMESTAMPTZ", "NULLABLE CHECK (due_date >= NOW()) OR IS NOT_NULL TRUE) -- prose"),
        ("updated_at", "TIMESTAMPTZ", "NOT_NULL DEFAULT_NOW AT"),
    ])])[0]
    columns = {column["name"]: column for column in repaired["columns"]}

    assert [column["name"] for column in repaired["columns"]].count("id") == 1
    assert [column["name"] for column in repaired["columns"]].count("title") == 1
    assert columns["status"]["constraints"] == "NOT_NULL DEFAULT 'pending' CHECK (status IN ('pending', 'completed'))"
    assert columns["due_date"]["constraints"] == "NOT_NULL CHECK (due_date >= NOW())"
    assert columns["updated_at"]["constraints"] == "NOT_NULL DEFAULT_NOW"


def test_numeric_id_gets_identity_without_name_based_alias_rewrite():
    tables = [
        _table("assigned_personnel", [
            ("id", "BIGINT", "PRIMARY_KEY"),
            ("person_id", "BIGINT", "UNIQUE FOREIGN_KEY REFERENCES personnel(id)"),
            ("personnel_id", "BIGINT", "FOREIGN_KEY REFERENCES personnel(id)"),
            ("assigned_personnel_id", "BIGINT", "FOREIGN_KEY REFERENCES personnel(id)"),
        ]),
        _table("personnel", [("id", "BIGINT", "PRIMARY_KEY")]),
    ]

    repaired = reconcile_fk_types(sanitize_tables(tables))
    by_name = {table["name"]: table for table in repaired}
    assignment_names = {column["name"] for column in by_name["assigned_personnel"]["columns"]}
    personnel_id = next(column for column in by_name["personnel"]["columns"] if column["name"] == "id")

    assert assignment_names & {"person_id", "assigned_personnel_id"} == {"person_id", "assigned_personnel_id"}
    assert "personnel_id" in assignment_names
    assert "GENERATED_IDENTITY" in personnel_id["constraints"]


def test_invalid_ascii_column_identifier_is_normalized():
    repaired = sanitize_tables([_table("events", [
        ("updated.", "TIMESTAMPTZ", "NULL"),
        ("event-title", "TEXT", "NOT_NULL"),
    ])])[0]
    names = {column["name"] for column in repaired["columns"]}

    assert "updated_at" in names
    assert "event_title" in names
    assert all(re.match(r'^[a-z][a-z0-9_]*$', name) for name in names)


def test_feature_coverage_understands_general_action_synonyms():
    missing = uncovered_features(
        ["할 일 등록", "담당자 배정", "상태 기록"],
        [
            "작업 생성과 수정 정보를 저장한다",
            "담당자를 작업에 할당한다",
            "상태 변경 이력과 로그를 저장한다",
        ],
    )

    assert missing == []


def test_malformed_duplicate_reference_is_canonicalized_and_text_actor_is_allowed():
    tables = sanitize_tables([_table("vaccination_records", [
        ("id", "BIGINT", "PRIMARY_KEY"),
        (
            "vaccination_schedule_id",
            "BIGINT",
            "NOT resident REFERENCES FOREIGN_KEY REFERENCES vaccination_schedules(id)",
        ),
        ("administered_by", "VARCHAR(100)", "NOT_NULL"),
    ])])
    constraint = tables[0]["columns"][1]["constraints"]
    assert constraint == "FOREIGN_KEY REFERENCES vaccination_schedules(id)"

    qa = object.__new__(QaAgent)
    db_issues, _, _ = qa._check_cross_document_semantics(
        {},
        {
            "tables": tables + [_table("vaccination_schedules", [("id", "BIGINT", "PRIMARY_KEY")])],
            "relationships": ["vaccination_schedules (1:N) vaccination_records"],
        },
        {"endpoints": []},
        [],
    )
    assert not any("administered_by" in issue for issue in db_issues)
    assert not any("relationship" in issue for issue in db_issues)


def test_explicit_pet_requirement_fields_get_general_contracts():
    pet = _table("pets", [
        ("id", "BIGINT", "PRIMARY_KEY"),
        ("created_at", "TIMESTAMPTZ", "NOT_NULL"),
        ("updated_at", "TIMESTAMPTZ", "NOT_NULL"),
    ])
    pet["description"] = "반려동물 등록 정보"
    tables = ensure_domain_columns([pet])

    prd = json.dumps({"coreFeatures": [{
        "name": "반려동물 등록",
        "description": "반려동물을 등록합니다.",
        "requirements": ["이름, 품종, 생년월일, 성별, 사진을 입력하여 저장합니다."],
    }]}, ensure_ascii=False)
    repaired = enforce_schema_contracts(tables, ["반려동물 등록"], prd)
    names = {column["name"] for column in repaired[0]["columns"]}
    assert {"name", "species", "birth_date", "gender", "image_url"} <= names


def test_long_single_line_concrete_api_contract_is_not_dropped_as_contamination():
    body = ", ".join(
        f"business_field_{index}: string — application_submissions 필드"
        for index in range(8)
    )
    assert len(body) > 400
    assert not _is_garbage_endpoint({
        "method": "POST",
        "path": "/api/v1/application-submissions",
        "description": "참여 신청 생성",
        "requestBody": body,
        "successResponse": "id: integer — 생성된 ID",
        "errorCodes": "400 — 검증 실패, 500 — 서버 오류",
    })


def test_db_completeness_prefers_explicit_fk_target_over_column_name_inference():
    qa = object.__new__(QaAgent)
    schema = {
        "tables": [
            _table("equipment_registrations", [("id", "BIGINT", "PRIMARY_KEY")]),
            _table("equipment_reservations", [
                ("id", "BIGINT", "PRIMARY_KEY"),
                ("equipment_id", "BIGINT", "FOREIGN_KEY REFERENCES equipment_registrations(id)"),
            ]),
        ],
        "relationships": ["equipment_registrations (1:N) equipment_reservations"],
    }
    issues = qa._check_db_completeness(schema, [])
    assert not any("equipment_id" in issue for issue in issues)


def test_downstream_check_does_not_invent_upstream_link_and_cleans_sql_splice():
    tables = [
        _table("volunteer_recruitments", [("id", "BIGINT", "PRIMARY_KEY")]),
        _table("application_submissions", [
            ("id", "BIGINT", "PRIMARY_KEY"),
            ("volunteer_recruitment_id", "BIGINT", "FOREIGN_KEY REFERENCES volunteer_recruitments(id)"),
            ("participant_id", "BIGINT", "FOREIGN_KEY REFERENCES participants(id)"),
        ]),
        _table("activity_checks", [
            ("id", "BIGINT", "PRIMARY_KEY"),
            ("volunteer_recruitment_id", "BIGINT", "FOREIGN_KEY REFERENCES volunteer_recruitments(id)"),
        ]),
        _table("participants", [("id", "BIGINT", "PRIMARY_KEY")]),
    ]
    repaired = reconcile_fk_types(enforce_schema_contracts(tables, [], ""))
    checks = next(table for table in repaired if table["name"] == "activity_checks")
    assert "application_submission_id" not in {column["name"] for column in checks["columns"]}

    polluted = sanitize_tables([_table("items", [
        ("updated_at", "TIMESTAMPTZ", "DEFAULT CURRENT_TIMESTAMP ? update SET updated_at = WHERE id ?;"),
    ])])
    assert polluted[0]["columns"][0]["constraints"] == "DEFAULT CURRENT_TIMESTAMP"


def test_generic_application_does_not_invent_business_links_from_names():
    tables = [
        _table("meeting_schedules", [("id", "BIGINT", "PRIMARY_KEY")]),
        _table("applications", [
            ("id", "BIGINT", "PRIMARY_KEY"),
            ("user_id", "BIGINT", "FOREIGN_KEY REFERENCES users(id)"),
        ]),
        _table("reading_records", [
            ("id", "BIGINT", "PRIMARY_KEY"),
            ("meeting_id", "BIGINT", "FOREIGN_KEY REFERENCES meeting_schedules(id)"),
        ]),
        _table("users", [("id", "BIGINT", "PRIMARY_KEY")]),
    ]
    repaired = reconcile_fk_types(enforce_schema_contracts(tables, [], ""))
    by_name = {table["name"]: table for table in repaired}
    application_names = {column["name"] for column in by_name["applications"]["columns"]}
    record_names = {column["name"] for column in by_name["reading_records"]["columns"]}
    assert "meeting_schedule_id" not in application_names
    assert "application_id" not in record_names


def test_event_api_ignores_other_event_targets_when_building_atomic_rule():
    schema = json.dumps({"tables": [
        _table("consigned_products", [
            ("id", "BIGINT", "PRIMARY_KEY"),
            ("status", "VARCHAR(20)", "NOT_NULL"),
        ]),
        _table("sales_records", [
            ("id", "BIGINT", "PRIMARY_KEY"),
            ("consigned_product_id", "BIGINT", "FOREIGN_KEY REFERENCES consigned_products(id)"),
            ("amount", "NUMERIC(10,2)", "NOT_NULL"),
        ]),
        _table("settlement_records", [
            ("id", "BIGINT", "PRIMARY_KEY"),
            ("consigned_product_id", "BIGINT", "FOREIGN_KEY REFERENCES consigned_products(id)"),
            ("sale_record_id", "BIGINT", "FOREIGN_KEY REFERENCES sales_records(id)"),
        ]),
    ]}, ensure_ascii=False)
    endpoint = _align_endpoints_to_db([{
        "method": "POST", "path": "/api/v1/settlement-records",
        "description": "정산 처리", "authRequired": True,
    }], schema)[0]
    assert endpoint.get("transactionRules")
    assert "잔량 부족" not in endpoint["errorCodes"]


def test_event_api_prefers_root_over_stateful_assignment_for_atomic_rule():
    schema = json.dumps({"tables": [
        _table("tasks", [
            ("id", "BIGINT", "PRIMARY_KEY"),
            ("status", "VARCHAR(20)", "NOT_NULL"),
        ]),
        _table("assigned_staffs", [
            ("id", "BIGINT", "PRIMARY_KEY"),
            ("task_id", "BIGINT", "FOREIGN_KEY REFERENCES tasks(id)"),
            ("status", "VARCHAR(20)", "NOT_NULL"),
        ]),
        _table("completed_status_records", [
            ("id", "BIGINT", "PRIMARY_KEY"),
            ("task_id", "BIGINT", "FOREIGN_KEY REFERENCES tasks(id)"),
            ("assigned_staff_id", "BIGINT", "FOREIGN_KEY REFERENCES assigned_staffs(id)"),
            ("status", "VARCHAR(20)", "NOT_NULL"),
        ]),
    ]}, ensure_ascii=False)
    endpoints = _align_endpoints_to_db([
        {
            "method": method,
            "path": "/api/v1/completed-status-records/{id}",
            "description": "완료 상태 기록 변경",
            "authRequired": True,
        }
        for method in ("PATCH", "DELETE")
    ], schema)
    assert all(endpoint.get("transactionRules") for endpoint in endpoints)
    assert all("tasks" in endpoint["transactionRules"] for endpoint in endpoints)


def test_chat_recovers_unlabeled_question_without_retry():
    raw = "다음 단계에서 가장 중요하게 해결하려는 문제는 무엇인가요?\nSUGGESTION: 예시"
    assert _extract_question_from_raw(raw) == "다음 단계에서 가장 중요하게 해결하려는 문제는 무엇인가요?"
    assert not _is_valid_question("다음 단계에서 가장 중요하게 해결하려는 문제")


def test_prd_audit_replaces_language_leaks_and_out_of_range_persona():
    sections = {
        "projectOverview": "사용자가 반복 확인 없이 최신 상태를 파악하고 필요한 작업을 빠르게 완료하도록 돕는 웹 서비스입니다.",
        "background": "사용자는 여러 위치의 정보를 반복 확인해야 하고 기록 누락도 자주 발생합니다ology. " * 3,
        "goals": [
            "기능을 통해 반복 확인을 줄이고 일상 관리 효율을 높인다",
            "기능을 통해 반복 확인을 줄이며 일상 관리 효율을 높인다",
        ],
        "kpi": [], "coreFeatures": [],
        "userPersonas": [{"name": "A", "age": "40대 이상", "job": "관리자", "techLevel": "낮음", "goal": "관리", "painPoint": "불편", "usagePattern": "매일"}],
        "mvpScope": {}, "techStack": {},
        "releaseSchedule": [{"description": "정식 출시를 완료한다. untolerable"}] * 6,
    }
    market = """SOURCE 1
provider: KOSIS
url: https://kosis.kr/statHtml/statHtml.do?orgId=101&tblId=DT_TEST
identifier: 101/DT_TEST
evidence: KOSIS 실제 통계값 — 2025 1인가구: 8244308가구
"""
    audited = PrdAgent._audit_sections(sections, ["상태 관리"], "20-30대 사용자를 위한 서비스", market)
    assert "ology" not in audited["background"]
    assert "KOSIS 101/DT_TEST" in audited["background"]
    assert all(item["age"] in {"20대", "30대"} for item in audited["userPersonas"])
    assert all("untolerable" not in item["description"] for item in audited["releaseSchedule"])
    assert len(audited["goals"]) == 3
    assert not any("다른 언어 조각" in issue for issue in object.__new__(QaAgent)._check_prd_completeness(audited, 1))
