import asyncio
import json

from phase2.state import PipelineState
from phase3.retry_service import RetryService
from phase3.schema_validator import SchemaValidator
from phase3.validation_service import ValidationService

FEATURES = ["사용자 로그인"]


def _db() -> dict:
    return {
        "tables": [
            {
                "name": "users",
                "description": "사용자 로그인 계정",
                "columns": [{"name": "id", "type": "BIGINT", "constraints": "PRIMARY_KEY"}],
            },
            {
                "name": "sessions",
                "description": "사용자 로그인 세션",
                "columns": [{"name": "id", "type": "BIGINT", "constraints": "PRIMARY_KEY"}],
            },
        ],
        "relationships": ["users (1:N) sessions"],
    }


def _api() -> dict:
    return {
        "authentication": "Bearer JWT",
        "endpoints": [
            {
                "method": "POST",
                "path": "/api/v1/login",
                "description": "사용자 로그인",
                "authRequired": True,
                "requestBody": "email, password",
                "successResponse": "accessToken",
                "errorCodes": "401 — 인증 실패",
            }
        ],
    }


def _prd() -> dict:
    return {
        "projectOverview": "로그인 서비스",
        "background": "사용자 인증이 필요합니다",
        "goals": ["안전한 로그인"],
        "kpi": [{"metric": "로그인 성공률"}],
        "userPersonas": [{"name": "사용자"}],
        "mvpScope": {"included": FEATURES},
        "techStack": {"backend": "FastAPI"},
        "releaseSchedule": [{"date": "1차"}],
        "coreFeatures": [{"name": "사용자 로그인"}],
    }


def _state(**overrides) -> PipelineState:
    values = {
        "feature_list": FEATURES,
        "db_schema": json.dumps(_db(), ensure_ascii=False),
        "api_spec": json.dumps(_api(), ensure_ascii=False),
        "prd_document": json.dumps(_prd(), ensure_ascii=False),
        "qa_approved": True,
    }
    values.update(overrides)
    return PipelineState(**values)


def test_empty_objects_are_rejected():
    result = SchemaValidator().validate(FEATURES, "{}", "{}", "{}")

    assert result.success is False
    assert {"DB_TABLES_REQUIRED", "API_ENDPOINTS_REQUIRED", "PRD_FIELDS_REQUIRED"} <= set(result.error_codes)


def test_malformed_array_items_are_rejected_without_crashing():
    result = SchemaValidator().validate(
        FEATURES,
        '{"tables":["bad"],"relationships":[]}',
        '{"endpoints":["bad"]}',
        json.dumps(_prd(), ensure_ascii=False),
    )

    assert result.success is False
    assert {"DB_TABLE_INVALID", "API_ENDPOINT_INVALID"} <= set(result.error_codes)


def test_repairable_json_is_canonicalized_in_state():
    state = _state(
        db_schema=json.dumps(_db(), ensure_ascii=False)[:-1] + ",}",
        api_spec=json.dumps(_api(), ensure_ascii=False)[:-1] + ",}",
    )

    result = ValidationService(SchemaValidator()).validate(state)

    assert result.validated is True
    assert json.loads(result.db_schema) == _db()
    assert json.loads(result.api_spec) == _api()


def test_api_registry_contract_is_validated_without_description_keyword_match():
    registry = [{
        "featureId": "plot.schedule",
        "name": "재배 일정 관리",
        "apiContract": [{"method": "POST", "path": "/api/v1/growing-schedules"}],
    }]
    api = {
        "authentication": "Bearer JWT",
        "endpoints": [{
            "featureId": "plot.schedule",
            "method": "POST",
            "path": "/api/v1/growing-schedules",
            "description": "Create a schedule",
            "successResponse": "ok",
            "errorCodes": "400",
        }],
    }
    result = SchemaValidator().validate(
        ["재배 일정 관리"], json.dumps(_db()), json.dumps(api), json.dumps(_prd()), registry,
    )

    assert "API_FEATURE_COVERAGE" not in result.error_codes
    assert "API_CONTRACT_MISSING" not in result.error_codes


def test_api_registry_contract_missing_is_a_structured_blocker():
    registry = [{
        "featureId": "plot.schedule",
        "name": "재배 일정 관리",
        "apiContract": [{"method": "POST", "path": "/api/v1/growing-schedules"}],
    }]
    api = {
        "authentication": "Bearer JWT",
        "endpoints": [{
            "featureId": "plot.schedule",
            "method": "GET",
            "path": "/api/v1/growing-schedules",
            "description": "List schedules",
            "successResponse": "ok",
            "errorCodes": "400",
        }],
    }
    result = SchemaValidator().validate(
        ["재배 일정 관리"], json.dumps(_db()), json.dumps(api), json.dumps(_prd()), registry,
    )

    assert "API_CONTRACT_MISSING" in result.error_codes


def test_prd_required_sections_are_validated():
    result = SchemaValidator().validate(
        FEATURES,
        json.dumps(_db(), ensure_ascii=False),
        json.dumps(_api(), ensure_ascii=False),
        '{"coreFeatures":[]}',
    )

    assert result.success is False
    assert "prd" in result.repair_targets
    assert "PRD_FIELDS_REQUIRED" in result.error_codes


def test_phase2_qa_rejection_blocks_phase3():
    state = _state(qa_approved=False, qa_api_issues=["응답 스키마 누락"])

    result = ValidationService(SchemaValidator()).validate(state)

    assert result.validated is False
    assert "QA_NOT_APPROVED" not in result.validation_error_codes
    assert result.validation_repair_targets == ["api"]


def test_validation_tracks_current_and_resolved_blocker_bundle():
    state = _state(
        qa_approved=True,
        validation_unresolved_blockers=["이전 repair blocker"],
        validation_blockers=["이전 repair blocker"],
    )

    result = ValidationService(SchemaValidator()).validate(state)

    assert result.validated is True
    assert result.validation_unresolved_blockers == []
    assert result.validation_blockers == []
    assert result.validation_resolved_blockers == ["이전 repair blocker"]


def test_validation_deduplicates_qa_and_schema_blockers():
    state = _state(
        qa_approved=False,
        qa_db_blockers=["DB 스키마: users에 PRIMARY_KEY가 없습니다"],
        qa_db_issues=["DB 스키마: users에 PRIMARY_KEY가 없습니다"],
    )

    result = ValidationService(SchemaValidator()).validate(state)

    assert result.validated is False
    assert result.validation_unresolved_blockers == [
        "DB 스키마: users에 PRIMARY_KEY가 없습니다"
    ]


def test_structural_and_cross_artifact_errors_are_detected():
    db = _db()
    db["tables"][1]["columns"].append(
        {"name": "ghost_id", "type": "BIGINT", "constraints": "FOREIGN_KEY"}
    )
    api = _api()
    api["authentication"] = ""
    api["endpoints"].append(dict(api["endpoints"][0]))
    api["endpoints"].append({
        **api["endpoints"][0],
        "method": "GET",
        "path": "/api/v1/orders",
        "description": "사용자 로그인 주문",
    })

    result = SchemaValidator().validate(
        FEATURES,
        json.dumps(db, ensure_ascii=False),
        json.dumps(api, ensure_ascii=False),
        json.dumps(_prd(), ensure_ascii=False),
    )

    assert {
        "DB_FOREIGN_KEY_TARGET_MISSING",
        "API_AUTH_DEFINITION_REQUIRED",
        "API_ENDPOINT_DUPLICATED",
    } <= set(result.error_codes)
    assert {"db", "api"} <= set(result.repair_targets)


def test_feature_coverage_gap_is_rejected_for_backend_features():
    result = SchemaValidator().validate(
        ["사용자 로그인", "결제 환불"],
        json.dumps(_db(), ensure_ascii=False),
        json.dumps(_api(), ensure_ascii=False),
        json.dumps(_prd(), ensure_ascii=False),
    )

    assert result.success is False
    assert "DB_FEATURE_COVERAGE" in result.error_codes
    assert "API_FEATURE_COVERAGE" in result.error_codes


def test_strict_feature_coverage_does_not_accept_one_generic_token():
    api = _api()
    api["endpoints"][0]["description"] = "관리 기능"
    result = SchemaValidator().validate(
        ["사용자 로그인", "출결 자동 대조"],
        json.dumps(_db(), ensure_ascii=False),
        json.dumps(api, ensure_ascii=False),
        json.dumps(_prd(), ensure_ascii=False),
    )

    assert "API_FEATURE_COVERAGE" in result.error_codes


def test_prd_db_entity_mismatch_is_rejected():
    prd = _prd()
    prd["techStack"] = {"database": "courses(id PK, academy_id FK→academies.id)"}
    result = SchemaValidator().validate(
        FEATURES,
        json.dumps(_db(), ensure_ascii=False),
        json.dumps(_api(), ensure_ascii=False),
        json.dumps(prd, ensure_ascii=False),
    )

    assert result.success is False
    assert "PRD_DB_ENTITY_MISMATCH" in result.error_codes


def test_duplicate_relationship_is_rejected():
    db = _db()
    db["relationships"].append("users (1:N) sessions")
    result = SchemaValidator().validate(
        FEATURES,
        json.dumps(db, ensure_ascii=False),
        json.dumps(_api(), ensure_ascii=False),
        json.dumps(_prd(), ensure_ascii=False),
    )

    assert result.success is False
    assert "DB_RELATIONSHIP_DUPLICATED" in result.error_codes


def test_frontend_only_feature_does_not_require_db_or_api_coverage():
    prd = _prd()
    prd["coreFeatures"].append({"name": "반응형 웹 화면 지원(PC·태블릿)", "description": "PC와 태블릿에 맞는 반응형 화면"})
    result = SchemaValidator().validate(
        ["반응형 웹 화면 지원(PC·태블릿)"],
        json.dumps(_db(), ensure_ascii=False),
        json.dumps(_api(), ensure_ascii=False),
        json.dumps(prd, ensure_ascii=False),
    )

    assert result.success is True


def test_common_fk_aliases_and_self_relationships_are_allowed():
    db = _db()
    db["tables"].append({
        "name": "files",
        "description": "첨부 파일",
        "columns": [{"name": "id", "type": "BIGINT", "constraints": "PRIMARY_KEY"}],
    })
    db["tables"][0]["columns"].append(
        {"name": "created_by_user_id", "type": "BIGINT", "constraints": "FOREIGN_KEY"}
    )
    db["tables"][1]["columns"].extend([
        {"name": "pdf_file_id", "type": "BIGINT", "constraints": "FOREIGN_KEY"},
        {"name": "source_session_id", "type": "BIGINT", "constraints": "FOREIGN_KEY"},
    ])
    db["relationships"].extend([
        "users (1:N) files",
        "sessions (1:N) sessions (source_session_id)",
    ])

    result = SchemaValidator().validate(
        FEATURES,
        json.dumps(db, ensure_ascii=False),
        json.dumps(_api(), ensure_ascii=False),
        json.dumps(_prd(), ensure_ascii=False),
    )

    assert result.success is True
    assert "DB_FOREIGN_KEY_TARGET_MISSING" not in result.error_codes
    assert "DB_RELATIONSHIP_INVALID" not in result.error_codes


def test_success_clears_stale_validation_state():
    state = _state(
        last_validation_error="과거 오류",
        validation_error_codes=["OLD"],
        validation_repair_targets=["db"],
    )

    result = ValidationService(SchemaValidator()).validate(state)

    assert result.validated is True
    assert result.last_validation_error == ""
    assert result.validation_error_codes == []
    assert result.validation_repair_targets == []


def test_retry_stops_at_limit_when_same_failure_repeats():
    class Orchestration:
        def __init__(self):
            self.calls = 0

        async def repair(self, state, _error, _pipeline_id, repair_targets=None):
            self.calls += 1
            return state

    class Validation:
        def validate(self, state):
            return state.copy(validated=False, last_validation_error="DB 스키마: 동일 오류")

    orchestration = Orchestration()
    state = _state(
        validated=False,
        last_validation_error="DB 스키마: 동일 오류",
        validation_repair_targets=["db"],
    )

    result = asyncio.run(RetryService(max_retry=3).retry_with(state, orchestration, Validation()))

    assert orchestration.calls == 1
    assert result.validated is False
    assert "동일 blocker 재발" in result.status_message


def test_retry_repeats_full_blocker_bundle_until_retry_limit():
    class Orchestration:
        def __init__(self):
            self.calls = 0

        async def repair(self, state, _error, _pipeline_id, repair_targets=None):
            self.calls += 1
            return state

    class Validation:
        def validate(self, state):
            return state.copy(
                validated=False,
                last_validation_error="표현만 바뀐 오류",
                validation_blockers=["DB FK 대상 누락", "API endpoint 누락"],
                validation_unresolved_blockers=["DB FK 대상 누락", "API endpoint 누락"],
            )

    state = _state(
        validated=False,
        last_validation_error="이전 오류",
        validation_repair_targets=["db", "api"],
        validation_blockers=["DB FK 대상 누락", "API endpoint 누락"],
        validation_unresolved_blockers=["DB FK 대상 누락", "API endpoint 누락"],
    )
    orchestration = Orchestration()

    result = asyncio.run(RetryService(max_retry=3).retry_with(state, orchestration, Validation()))

    assert orchestration.calls == 1
    assert result.validation_unresolved_blockers == ["DB FK 대상 누락", "API endpoint 누락"]
    assert "동일 blocker 재발" in result.status_message


def test_retry_timeout_consumes_remaining_retry_budget():
    class Orchestration:
        def __init__(self):
            self.calls = 0

        async def repair(self, state, _error, _pipeline_id, repair_targets=None):
            self.calls += 1
            raise TimeoutError("simulated timeout")

    orchestration = Orchestration()
    state = _state(
        validated=False,
        last_validation_error="DB FK 대상 누락",
        validation_repair_targets=["db"],
        validation_blockers=["DB FK 대상 누락"],
        validation_unresolved_blockers=["DB FK 대상 누락"],
    )

    result = asyncio.run(RetryService(max_retry=3).retry_with(state, orchestration, ValidationService(SchemaValidator())))

    assert orchestration.calls == 1
    assert "동일 blocker 재발" in result.status_message
    assert "REPAIR_TIMEOUT" in result.last_validation_error


def test_retry_keeps_resolved_blockers_when_one_of_two_is_fixed():
    class Orchestration:
        async def repair(self, state, _error, _pipeline_id, repair_targets=None):
            return state

    class Validation:
        def validate(self, state):
            return state.copy(
                validated=False,
                validation_blockers=["API endpoint 누락"],
                validation_unresolved_blockers=["API endpoint 누락"],
                validation_resolved_blockers=["DB FK 대상 누락"],
            )

    state = _state(
        validated=False,
        validation_repair_targets=["db", "api"],
        validation_blockers=["DB FK 대상 누락", "API endpoint 누락"],
        validation_unresolved_blockers=["DB FK 대상 누락", "API endpoint 누락"],
    )

    result = asyncio.run(RetryService(max_retry=3).retry_with(state, Orchestration(), Validation()))

    assert result.validation_resolved_blockers == ["DB FK 대상 누락"]
    assert result.validation_unresolved_blockers == ["API endpoint 누락"]


def test_retry_feedback_does_not_grow_across_attempts():
    marker = "\n\n[PHASE3_VALIDATION_FEEDBACK]\n"
    state = _state(context_prompt=f"원본 컨텍스트{marker}과거 오류")
    state = state.copy(last_validation_error="새 오류")

    rebuilt = RetryService._build_retry_prompt(state)

    assert rebuilt.count(marker) == 1
    assert rebuilt.startswith("원본 컨텍스트")
