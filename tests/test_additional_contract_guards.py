import asyncio
import copy
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from phase2.agents.api_agent import ApiAgent, _normalize_parameters, api_self_check
from phase2.agents.dba_agent import DbaAgent, reconcile_fk_types
from phase2.agents.qa_agent import QaAgent
from phase2.artifact_shapes import shape_issues
from phase2.contract_guard import feature_contract_issues
from phase2.feature_registry import missing_db_contract_features
from phase2.fk_validation import foreign_key_issues
from phase2.mvp_policy import dependency_issues, normalize_priorities, select_mvp
from phase2.openapi_spec import build_openapi_document
from phase2.operation_contract import operation_registry, operation_schema_issues
from phase2.orchestration_graph import OrchestrationGraph
from phase2.repair_budget import merge_attempts
from phase2.state import PipelineState
from phase2.validation_contract import artifact_revision, issue_detail
from phase3.retry_service import RetryService
from phase3.schema_validator import SchemaValidator, ValidationResult
from phase3.validation_service import ValidationService


def column(name, sql_type="BIGINT", constraints=""):
    return {"name": name, "type": sql_type, "constraints": constraints}


def database():
    return {"tables": [
        {"name": "users", "columns": [column("id", constraints="PRIMARY_KEY"), column("email", "VARCHAR(255)", "UNIQUE")]},
        {"name": "orders", "columns": [column("id", constraints="PRIMARY_KEY"), column("buyer", constraints="REFERENCES users(id)")]},
    ]}


def test_get_mapping_without_body_checks_storage_target():
    issues = operation_schema_issues({"method": "GET", "fieldMappings": [
        {"field": "id", "source": "server", "table": "ghost", "column": "id"},
    ]}, {})
    assert any("API_FIELD_MAPPING_TARGET_UNKNOWN" in issue for issue in issues)


@pytest.mark.parametrize("source,field", [("path", "orderId"), ("query", "page"), ("header", "X-Tenant"), ("response", "items[].id")])
def test_mapping_sources_and_nested_response_pointers(source, field):
    endpoint = {"method": "GET", "path": "/api/v1/orders/{orderId}", "parameters": [
        {"in": "query", "name": "page", "schema": {"type": "integer"}},
        {"in": "header", "name": "X-Tenant", "schema": {"type": "integer"}},
    ], "responseSchema": {"type": "object", "properties": {"items": {"type": "array", "items": {
        "type": "object", "properties": {"id": {"type": "integer"}},
    }}}}, "fieldMappings": [{"source": source, "field": field, "table": "orders", "column": "id"}]}
    assert not operation_schema_issues(endpoint, {"orders": database()["tables"][1]})


def test_nested_mapping_does_not_accept_same_leaf_from_another_object():
    endpoint = {"method": "POST", "requestSchema": {"type": "object", "properties": {
        "buyer": {"type": "object", "properties": {"id": {"type": "integer"}}},
    }}, "fieldMappings": [{"source": "body", "field": "seller.id", "table": "orders", "column": "id"}]}
    assert any("INPUT_UNKNOWN" in issue for issue in operation_schema_issues(endpoint, {"orders": database()["tables"][1]}))


def test_mapping_type_mismatch_requires_transform():
    endpoint = {"method": "POST", "requestSchema": {"type": "object", "properties": {"id": {"type": "string"}}},
                "fieldMappings": [{"field": "id", "source": "request", "table": "orders", "column": "id"}]}
    assert any("TYPE_MISMATCH" in issue for issue in operation_schema_issues(endpoint, {"orders": database()["tables"][1]}))
    endpoint["fieldMappings"][0]["transform"] = "parse validated decimal integer"
    assert not operation_schema_issues(endpoint, {"orders": database()["tables"][1]})


def test_parameter_schema_and_serialization_survive_projection():
    schema = {"type": "array", "items": {"type": "string", "enum": ["a", "b"]}, "minItems": 1}
    endpoint = {"method": "GET", "path": "/api/v1/orders/{id}", "parameters": [
        {"name": "id", "in": "query", "schema": schema, "style": "form", "explode": False},
        {"name": "id", "in": "path", "schema": {"type": "string", "format": "uuid"}, "required": False},
    ]}
    _normalize_parameters(endpoint)
    _normalize_parameters(endpoint)
    assert len(endpoint["parameters"]) == 2
    assert endpoint["parameters"][0]["schema"] == schema
    params = build_openapi_document({"endpoints": [endpoint]})["paths"]["/api/v1/orders/{id}"]["get"]["parameters"]
    assert params[0]["schema"] == schema and params[0]["explode"] is False
    assert params[1]["required"] is True and params[1]["schema"]["format"] == "uuid"


@pytest.mark.parametrize("field,alternate", [("responseSchema", {"type": "string"}), ("contentType", "multipart/form-data"), ("successStatus", 202)])
def test_shared_operation_conflicts_are_not_silently_overwritten(field, alternate):
    original = {"method": "POST", "path": "/api/v1/orders", "responseSchema": {"type": "object"},
                "contentType": "application/json", "successStatus": 201}
    registry = [{"featureId": "orders.create", "apiContract": [original]},
                {"featureId": "orders.confirm", "apiContract": [{**original, field: alternate}]}]
    operations, issues = operation_registry(registry)
    assert any("CONFLICT" in issue for issue in issues)
    assert next(iter(operations.values()))[field] == original[field]
    endpoints, self_issues = api_self_check([{**original, "featureIds": ["orders.create", "orders.confirm"]}], registry)
    assert any("CONFLICT" in issue for issue in self_issues)
    assert endpoints[0][field] == original[field]


def test_shared_operation_merges_nonconflicting_details():
    operations, issues = operation_registry([
        {"featureId": "orders.read", "apiContract": [{"method": "GET", "path": "/api/v1/orders", "responseSchema": {"type": "object"}}]},
        {"featureId": "reports.read", "apiContract": [{"method": "GET", "path": "/api/v1/orders", "successStatus": 200}]},
    ])
    assert not issues
    operation = next(iter(operations.values()))
    assert set(operation["featureIds"]) == {"orders.read", "reports.read"}
    assert operation["responseSchema"] == {"type": "object"} and operation["successStatus"] == 200


@pytest.mark.parametrize("constraints,code", [
    ("REFERENCES ghost(id)", "DB_FOREIGN_KEY_TARGET_MISSING"),
    ("REFERENCES users(missing)", "DB_FOREIGN_KEY_COLUMN_MISSING"),
    ("FOREIGN_KEY", "DB_FOREIGN_KEY_REFERENCE_REQUIRED"),
    ("FOREIGN_KEY REFERENCES users", "DB_FOREIGN_KEY_REFERENCE_INVALID"),
    ("REFERENCES users(email)", "DB_FOREIGN_KEY_TYPE_MISMATCH"),
    ("NOT_NULL REFERENCES users(id) ON DELETE SET NULL", "DB_FOREIGN_KEY_NULLABILITY_CONFLICT"),
    ("NOT_NULL REFERENCES users(id) ON DELETE SET DEFAULT", "DB_FOREIGN_KEY_DEFAULT_REQUIRED"),
])
def test_fk_precise_failures(constraints, code):
    db = database()
    db["tables"][1]["columns"][1]["constraints"] = constraints
    assert code in {issue["code"] for issue in foreign_key_issues(db)}


def test_non_pk_unique_reference_is_preserved_and_type_is_reconciled():
    db = database()
    ref = db["tables"][1]["columns"][1]
    ref["constraints"] = "REFERENCES users(email)"
    reconcile_fk_types(db["tables"])
    assert ref["type"] == "VARCHAR(255)"
    assert ref["constraints"] == "REFERENCES users(email)"
    assert not foreign_key_issues(db)


def test_composite_fk_requires_matching_arity_order_and_unique_key():
    db = {"tables": [
        {"name": "parents", "columns": [column("tenant"), column("code")], "indexes": ["CREATE UNIQUE INDEX idx_key ON parents(tenant,code)"]},
        {"name": "children", "columns": [column("tenant"), column("code")], "foreignKeys": [
            {"columns": ["tenant", "code"], "references": {"table": "parents", "columns": ["tenant", "code"]}},
        ]},
    ]}
    assert not foreign_key_issues(db)
    db["tables"][1]["foreignKeys"][0]["references"]["columns"] = ["code"]
    assert any(issue["code"] == "DB_FOREIGN_KEY_SOURCE_INVALID" for issue in foreign_key_issues(db))


def test_fk_cannot_reference_nonunique_or_partial_unique_column():
    db = database()
    db["tables"][0]["columns"][1]["constraints"] = ""
    db["tables"][0]["indexes"] = ["CREATE UNIQUE INDEX idx_email ON users(email) WHERE active = true"]
    db["tables"][1]["columns"][1] = column("buyer", "VARCHAR(255)", "REFERENCES users(email)")
    assert any(issue["code"] == "DB_FOREIGN_KEY_UNIQUE_REQUIRED" for issue in foreign_key_issues(db))


def test_unknown_fk_is_not_deleted_to_hide_failure():
    db = database()
    db["tables"][1]["columns"].append(column("missing_id", constraints="FOREIGN_KEY REFERENCES unknown(id)"))
    reconcile_fk_types(db["tables"])
    assert any(item["name"] == "missing_id" for item in db["tables"][1]["columns"])
    assert any(issue["code"] == "DB_FOREIGN_KEY_TARGET_MISSING" for issue in foreign_key_issues(db))


@pytest.mark.parametrize("domain,document", [
    ("db", {"tables": None}), ("db", {"tables": [{"name": "items", "columns": None}]}),
    ("api", {"endpoints": [{"parameters": "wrong"}]}),
    ("api", {"endpoints": [{"requestSchema": {"properties": []}}]}),
    ("registry", [{"name": "Feature", "sessionPolicy": "wrong"}]),
    ("registry", [{"name": "Feature", "apiContract": [{"responseSchema": {"required": {}}}]}]),
])
def test_shape_errors_have_pointer_without_crashing(domain, document):
    errors = shape_issues(domain, document)
    assert errors and errors[0]["pointer"]


def test_schema_validator_returns_shape_blocker_instead_of_typeerror():
    result = SchemaValidator().validate(["Items"], '{"tables":null}', '{"endpoints":[{"parameters":null}]}', '{}')
    assert not result.success and "ARTIFACT_SHAPE_INVALID" in result.error_codes


def test_ui_only_feature_requires_no_api_or_database_and_no_llm_call():
    ui = {"featureId": "layout.responsive", "name": "반응형 화면", "description": "기기별 화면 배치를 제공한다.",
          "requirements": ["모바일 화면에서 내용을 읽을 수 있다."], "priority": "P1", "source": "prd_core",
          "applicability": {"ui": True, "api": False, "db": False}, "apiContract": [], "dbContract": {}}
    assert not feature_contract_issues([ui])
    api, dba = object.__new__(ApiAgent), object.__new__(DbaAgent)
    state = PipelineState(feature_registry=[ui], feature_specs=[ui], feature_list=[ui["name"]])
    api_result = asyncio.run(api.execute(state))
    db_result = asyncio.run(dba.execute(state))
    prd = {key: [] for key in ("goals", "kpi", "userPersonas", "releaseSchedule")}
    prd.update(projectOverview="UI", background="UI", techStack={}, mvpScope={}, coreFeatures=[ui])
    result = SchemaValidator().validate(state.feature_list, db_result.db_schema, api_result.api_spec,
                                        json.dumps(prd), [ui], [ui])
    assert result.success, result.errors


def test_coverage_requires_all_declared_tables_even_with_feature_tag():
    assert missing_db_contract_features([
        {"featureId": "orders.create", "dbContract": {"tables": ["users", "orders"]}},
    ], [{"name": "users", "featureIds": ["orders.create"]}]) == ["orders.create"]


def test_shared_qa_and_phase3_issue_identity_matches_current_revision():
    state = PipelineState(feature_list=["Items"], feature_registry=[{
        "featureId": "items.read", "name": "Items", "apiContract": [{"method": "GET", "path": "/api/v1/items"}],
        "dbContract": {"tables": ["items"]},
    }], db_schema='{"tables":[]}', api_spec='{"endpoints":[]}', prd_document='{"coreFeatures":[]}')
    qa = asyncio.run(QaAgent(object()).execute(state))
    checked = ValidationService(SchemaValidator()).validate(qa)
    qa_keys = {(item["code"], item["artifactKey"], item["pointer"]) for item in qa.qa_blocker_details}
    checked_keys = {(item["code"], item["artifactKey"], item["pointer"]) for item in checked.qa_blocker_details}
    assert qa_keys == checked_keys
    assert qa.qa_validation_revision == checked.qa_validation_revision


def test_stale_qa_failure_is_not_a_gate_after_current_checker_passes():
    checker = SimpleNamespace(validate=lambda **_kwargs: ValidationResult.ok())
    state = PipelineState(qa_approved=False, qa_validation_revision="old", qa_api_blockers=["stale"],
                          qa_issue_details=[{"severity": "BLOCKER", "reason": "stale"}])
    result = ValidationService(checker).validate(state)
    assert result.validated and not result.qa_api_blockers


def test_fingerprint_ignores_order_wording_and_revision_with_structured_identity():
    first = issue_detail("API_CONTRACT_MISSING", "api", "old wording", artifactKey="GET /api/v1/items", pointer="/responseSchema")
    second = issue_detail("DB_FK", "db", "other", artifactKey="items", pointer="/columns/user_id")
    before = PipelineState(validation_repair_targets=["api", "db"], qa_blocker_details=[first, second])
    after = before.copy(qa_blocker_details=[{**second, "revision": "new"}, {**first, "message": "new wording"}])
    assert RetryService._failure_fingerprint(before) == RetryService._failure_fingerprint(after)


def test_exhausted_domain_does_not_prevent_unspent_domain_repair():
    seen = []
    async def repair(state, _feedback, _pipeline_id, repair_targets=None):
        seen.extend(repair_targets)
        return state
    service = SimpleNamespace(validate=lambda state: state.copy(validated=True))
    state = PipelineState(validation_repair_targets=["api", "db"], targeted_repair_attempts={"db": 1})
    result = asyncio.run(RetryService().retry_with(state, SimpleNamespace(repair=repair), service))
    assert result.validated and seen == ["api"]


def test_domain_budget_counts_actual_dispatch_once_and_rejects_second_call():
    agent = SimpleNamespace(repair=AsyncMock(side_effect=lambda state, _feedback: state.copy(api_spec="updated")))
    graph = OrchestrationGraph(None, None, None, None, agent, None, None)
    first = asyncio.run(graph._bounded_repair("api", agent, PipelineState(), "feedback"))
    second = asyncio.run(graph._bounded_repair("api", agent, first, "feedback"))
    assert first.targeted_repair_attempts == {"api": 1}
    assert second.targeted_repair_results["api"] == "budget_exhausted"
    agent.repair.assert_awaited_once()


def test_no_dispatch_scope_rejection_does_not_consume_budget():
    agent = SimpleNamespace(repair=AsyncMock(side_effect=lambda state, _feedback: state.copy(targeted_repair_results={"api": "rejected_scope"})))
    graph = OrchestrationGraph(None, None, None, None, agent, None, None)
    result = asyncio.run(graph._bounded_repair("api", agent, PipelineState(), "ambiguous"))
    assert not result.targeted_repair_attempts


def test_previous_rejection_cannot_hide_a_later_dispatched_noop():
    agent = SimpleNamespace(repair=AsyncMock(side_effect=lambda state, _feedback: state))
    graph = OrchestrationGraph(None, None, None, None, agent, None, None)
    state = PipelineState(targeted_repair_results={"api": "rejected_scope"})
    result = asyncio.run(graph._bounded_repair("api", agent, state, "specific endpoint"))
    assert result.targeted_repair_results["api"] == "noop"
    assert result.targeted_repair_attempts == {"api": 1}


def test_mvp_policy_preserves_explicit_priorities_and_excludes_wont():
    features = [{"featureId": "items.read", "name": "Read", "source": "prd_core", "priority": "P2"},
                {"featureId": "items.delete", "name": "Delete", "source": "prd_core", "priority": "WONT"}]
    before = copy.deepcopy(features)
    normalize_priorities(features)
    selected = select_mvp(features)
    assert selected[0]["priority"] == before[0]["priority"]
    assert len(selected) == 1 and features[1]["releaseScope"] == "excluded"


def test_mvp_selection_is_dependency_closed_and_capped_at_twelve():
    features = [{"featureId": f"items.f{index}", "name": str(index), "source": "prd_core", "priority": "P0"} for index in range(15)]
    features.append({"featureId": "items.child", "name": "Child", "source": "supporting", "priority": "P1", "priorityScore": 99,
                     "parentFeatureId": "items.f14", "dependencies": ["items.f13"]})
    selected = select_mvp(features)
    ids = {item["featureId"] for item in selected}
    assert len(selected) <= 12
    assert sum(item["priority"] == "P0" for item in selected) <= 5
    assert "items.child" in ids and {"items.f13", "items.f14"} <= ids


def test_dependency_cycles_and_missing_ids_are_explicit_issues():
    assert dependency_issues([{"featureId": "items.read", "dependencies": ["unknown.read"]}])
    assert dependency_issues([{"featureId": "items.read", "parentFeatureId": "items.edit"},
                              {"featureId": "items.edit", "parentFeatureId": "items.read"}])


def test_revision_is_json_format_independent():
    assert artifact_revision('{"a":1}') == artifact_revision('{ "a" : 1 }')


def test_distinct_mapping_failures_are_not_collapsed_by_issue_deduplication():
    from phase2.validation_contract import dedupe_issues
    details = [issue_detail("API_FIELD_MAPPING_TARGET_UNKNOWN", "api", f"GET /api/v1/items: API_FIELD_MAPPING_TARGET_UNKNOWN: items.{field}")
               for field in ("missing_a", "missing_b")]
    assert len(dedupe_issues([*details, details[0]])) == 2


def test_parallel_budget_merges_cannot_reset_sibling_attempts():
    assert merge_attempts({"api": 1, "db": 0}, {"api": 0, "db": 1}) == {"api": 1, "db": 1}


def test_phase1_and_feature_spec_share_moscow_priority_mapping():
    from phase2.mvp_policy import apply_project_priorities
    features = [{"featureId": "items.read", "name": "Read", "source": "prd_core"},
                {"featureId": "items.search", "name": "Search", "source": "supporting"},
                {"featureId": "items.export", "name": "Export", "source": "supporting"}]
    apply_project_priorities(features, {"priorities": {"must": ["Read"], "should": ["Search"], "could": ["Export"]}})
    assert [item["priority"] for item in features] == ["P0", "P1", "P2"]


def test_stateless_api_does_not_require_fake_database_mapping():
    feature = {"featureId": "rates.convert", "name": "환율 조회", "source": "prd_core", "priority": "P1",
               "description": "외부 환율을 조회하여 변환 결과를 제공한다.", "requirements": ["조회 결과를 반환한다."],
               "applicability": {"api": True, "db": False}, "dataMode": "stateless", "dbContract": {},
               "transactionRules": ["읽기 전용 외부 호출이며 저장하지 않는다."], "apiContract": [
                   {"method": "GET", "path": "/api/v1/rates", "responseSchema": {"type": "object", "properties": {"rate": {"type": "number"}}}},
               ]}
    endpoint = {**feature["apiContract"][0], "featureId": feature["featureId"], "featureIds": [feature["featureId"]],
                "description": "외부 환율 조회", "authRequired": False, "successResponse": "rate: number", "errorCodes": "502 외부 조회 실패"}
    api = {"endpoints": [endpoint], "featureMappings": [{"featureName": feature["name"], "featureId": feature["featureId"],
            "operations": [{"method": "GET", "path": "/api/v1/rates"}]}]}
    prd = {"coreFeatures": [feature], "projectOverview": "환율 조회", "background": "외부 환율 조회", "goals": [],
           "kpi": [], "userPersonas": [], "mvpScope": {}, "techStack": {}, "releaseSchedule": []}
    assert not feature_contract_issues([feature])
    result = SchemaValidator().validate([feature["name"]], '{"tables":[]}', json.dumps(api), json.dumps(prd), [feature], [feature])
    assert result.success, result.errors
