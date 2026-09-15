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


def test_phase2_qa_rejection_is_advisory_to_phase3():
    state = _state(qa_approved=False, qa_api_issues=["응답 스키마 누락"])

    result = ValidationService(SchemaValidator()).validate(state)

    assert result.validated is True
    assert "QA_NOT_APPROVED" not in result.validation_error_codes
    assert result.validation_repair_targets == []


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
        "API_DB_RESOURCE_MISMATCH",
    } <= set(result.error_codes)
    assert {"db", "api"} <= set(result.repair_targets)


def test_any_feature_coverage_gap_is_rejected():
    result = SchemaValidator().validate(
        ["사용자 로그인", "결제 환불"],
        json.dumps(_db(), ensure_ascii=False),
        json.dumps(_api(), ensure_ascii=False),
        json.dumps(_prd(), ensure_ascii=False),
    )

    assert result.success is False
    assert {"DB_FEATURE_COVERAGE", "API_FEATURE_COVERAGE", "PRD_FEATURE_COVERAGE"} <= set(result.error_codes)


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


def test_retry_stops_when_same_failure_repeats():
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
    assert "동일 검증 실패" in result.status_message


def test_retry_feedback_does_not_grow_across_attempts():
    marker = "\n\n[PHASE3_VALIDATION_FEEDBACK]\n"
    state = _state(context_prompt=f"원본 컨텍스트{marker}과거 오류")
    state = state.copy(last_validation_error="새 오류")

    rebuilt = RetryService._build_retry_prompt(state)

    assert rebuilt.count(marker) == 1
    assert rebuilt.startswith("원본 컨텍스트")
