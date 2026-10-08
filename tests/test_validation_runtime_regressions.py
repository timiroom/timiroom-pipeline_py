import asyncio
import json

from phase2.agents.qa_agent import QaAgent
from phase2.orchestration_graph import OrchestrationGraph
from phase2.state import PipelineState
from phase3.schema_validator import SchemaValidator, ValidationResult
from phase3.validation_service import ValidationService


class _Progress:
    def send(self, *_args, **_kwargs):
        pass


def test_runtime_qa_reports_semantic_mismatch_without_rewriting_input():
    state = PipelineState(
        db_schema=json.dumps({"tables": [{
            "name": "documents", "columns": [
                {"name": "id", "type": "BIGINT", "constraints": "PRIMARY_KEY"},
                {"name": "title", "type": "TEXT", "constraints": "NOT_NULL"},
            ],
        }], "relationships": []}, indent=2),
        api_spec=json.dumps({"endpoints": [{
            "method": "POST", "path": "/api/v1/documents",
            "requestBody": "unknown_field: string", "successResponse": "id: integer",
            "errorCodes": "400, 500",
        }]}),
        prd_document='{"coreFeatures": []}',
    )

    result = asyncio.run(QaAgent(object()).execute(state))

    assert result.db_schema == state.db_schema
    assert result.api_spec == state.api_spec
    assert result.prd_document == state.prd_document
    assert any("unknown_field" in issue for issue in result.qa_api_blockers)
    assert any(
        item["agent"] == "API" and "unknown_field" in item["reason"]
        for item in result.qa_repair_issues
    )


def test_deterministic_quality_warning_does_not_become_a_contract_blocker():
    warning = "KPI 근거 검증 실패: 외부 KPI 근거에 원문 URL 없음"
    error = "API 요청 필드가 ERD와 불일치: POST /api/v1/documents ['unknown_field']"

    result = QaAgent._classify_issues([], [error], [warning], {warning, error})

    assert result["prd_blockers"] == []
    assert result["prd_warnings"] == [warning]
    assert result["api_blockers"] == [error]


def test_phase3_repairs_only_blocking_domains_when_quality_warnings_remain():
    class ValidSchema:
        def validate(self, **_kwargs):
            return ValidationResult.ok()

    warning = "KPI 근거 검증 실패: 외부 KPI 근거에 원문 URL 없음"
    error = "API 요청 필드가 ERD와 불일치"
    state = PipelineState(
        qa_approved=False,
        qa_api_issues=[error], qa_api_blockers=[error],
        qa_prd_issues=[warning], qa_prd_warnings=[warning],
        qa_issue_details=[
            {"agent": "API", "reason": error, "severity": "BLOCKER"},
            {"agent": "PRD", "reason": warning, "severity": "WARNING"},
        ],
    )

    result = ValidationService(ValidSchema()).validate(state)

    assert result.validation_repair_targets == ["api"]
    assert warning not in result.validation_unresolved_blockers


def test_public_repair_handles_all_targets_and_passes_repaired_prd_to_api():
    class Agent:
        async def execute(self, state, _dump=None):
            return state

    class Prd(Agent):
        async def repair(self, state, _feedback):
            return state.copy(prd_document='{"version":"repaired"}')

    class Api(Agent):
        async def repair(self, state, _feedback):
            return state.copy(api_spec=json.dumps({"prd": json.loads(state.prd_document)}))

    graph = OrchestrationGraph(Agent(), Agent(), Prd(), Agent(), Api(), Agent(), _Progress())
    state = PipelineState(prd_document='{"version":"old"}')

    result = asyncio.run(graph.repair(state, "PRD/API contract errors", repair_targets=["prd", "api"]))

    assert json.loads(result.api_spec) == {"prd": {"version": "repaired"}}


def test_exhausted_legacy_prd_rollback_preserves_latest_downstream_documents():
    class Agent:
        async def execute(self, state, _dump=None):
            return state

    class Dba(Agent):
        async def execute(self, state, _dump=None):
            return state.copy(db_schema='{"tables":[{"name":"documents"}]}')

    class Api(Agent):
        async def execute(self, state, _dump=None):
            return state.copy(api_spec='{"endpoints":[]}', prd_feedback_from_api="PRD contract missing")

    graph = OrchestrationGraph(Agent(), Agent(), Agent(), Dba(), Api(), Agent(), _Progress())

    result = asyncio.run(graph._run_prd_with_rollback(PipelineState(), None))

    assert json.loads(result.db_schema)["tables"][0]["name"] == "documents"
    assert json.loads(result.api_spec) == {"endpoints": []}
    assert result.prd_feedback_from_api == "PRD contract missing"


def test_parallel_drafts_are_aligned_to_final_db_before_qa():
    class Agent:
        async def execute(self, state, _dump=None):
            return state

    class Dba(Agent):
        async def execute(self, state, _dump=None):
            return state.copy(db_schema=json.dumps({"tables": [{
                "name": "documents", "description": "문서 등록",
                "columns": [
                    {"name": "id", "type": "BIGINT", "constraints": "PRIMARY_KEY"},
                    {"name": "title", "type": "TEXT", "constraints": "NOT_NULL"},
                ], "indexes": [],
            }], "relationships": []}))

    class Api(Agent):
        async def execute(self, state, _dump=None):
            return state.copy(api_spec=json.dumps({"authentication": "공개 API", "endpoints": [{
                "featureId": "document.create", "action": "create",
                "method": "POST", "path": "/api/v1/documents", "description": "문서 등록",
                "requestBody": "payload: object", "successResponse": "id: integer",
                "errorCodes": "400, 500", "authRequired": False,
            }]}))

    state = PipelineState(feature_list=["문서 등록"], feature_registry=[{
        "featureId": "document.create", "name": "문서 등록",
        "apiContract": [{"method": "POST", "path": "/api/v1/documents", "action": "create"}],
        "dbContract": {"tables": ["documents"]},
    }], prd_document='{}')
    graph = OrchestrationGraph(Agent(), Agent(), Agent(), Dba(), Api(), Agent(), _Progress())

    result = asyncio.run(graph.run(state))

    api = json.loads(result.api_spec)
    endpoint = api["endpoints"][0]
    assert "title" in endpoint["requestBody"]
    assert "payload" not in endpoint["requestBody"]
    assert endpoint["featureId"] == "document.create"
    assert endpoint["action"] == "create"
    assert api["authentication"] == "공개 API"


def test_item_repair_consumes_domain_budget_and_does_not_repeat_provider_failure():
    class Agent:
        async def execute(self, state, _dump=None):
            return state

    class Api(Agent):
        async def repair(self, state, _feedback):
            return state.copy(api_spec='{"repaired":true}')

    graph = OrchestrationGraph(Agent(), Agent(), Agent(), Agent(), Api(), Agent(), _Progress())
    state = PipelineState(qa_repair_issues=[{"agent": "API", "reason": "contract missing"}])

    result = asyncio.run(graph._repair_once(state, None))
    assert result.targeted_repair_attempts == {"api": 1}

    exhausted = state.copy(targeted_repair_attempts={"api": 1})
    assert asyncio.run(graph._repair_once(exhausted, None)).api_spec == ""

    failed = state.copy(generation_blockers=["API_GENERATION_BLOCKER: provider unavailable"])
    assert asyncio.run(graph._repair_once(failed, None)).api_spec == ""


def test_feature_contract_validates_the_declared_identity_owner():
    spec = {
        "name": "문서 등록", "ownership": {
            "scope": "USER", "ownerEntity": "members", "ownerKey": "member_id",
        }, "transactionRules": ["원자 저장"],
    }
    db = {
        "tables": [
            {"name": "members", "columns": [{"name": "id", "type": "BIGINT", "constraints": "PRIMARY_KEY"}]},
            {"name": "documents", "description": "문서 등록", "columns": [
                {"name": "id", "type": "BIGINT", "constraints": "PRIMARY_KEY"},
                {"name": "member_id", "type": "BIGINT", "constraints": "FOREIGN_KEY REFERENCES members(id)"},
            ]},
        ], "relationships": ["members (1:N) documents"],
        "featureMappings": [{"featureName": "문서 등록", "table": "documents"}],
    }
    api = {
        "authentication": "External identity gateway",
        "endpoints": [{
            "method": "POST", "path": "/api/v1/documents", "description": "문서 등록",
            "authRequired": True, "requestBody": "title: string", "successResponse": "id: integer",
            "errorCodes": "400, 500", "transactionRules": "원자 저장",
        }],
        "featureMappings": [{"featureName": "문서 등록", "table": "documents", "operations": [
            {"method": "POST", "path": "/api/v1/documents"},
        ]}],
    }

    result = SchemaValidator().validate(["문서 등록"], json.dumps(db), json.dumps(api), [spec])

    assert result.success, result.errors
