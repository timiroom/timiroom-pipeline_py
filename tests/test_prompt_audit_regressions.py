import asyncio
import json
from types import SimpleNamespace

import pytest

from phase2.agents.api_agent import ApiAgent
from phase2.agents.dba_agent import TABLE_SPEC_PROMPT
from phase2.agents.feature_spec_agent import PROMPT
from phase2.agents.pm_agent import PM_PROMPT, PmAgent
from phase2.agents.prd_agent import PrdAgent
from phase2.agents.qa_agent import _REVIEW_PROMPTS
from phase2.agents.search_agent import SearchAgent
from phase2.feature_generation import (
    GeneratedDocument,
    output_schema_prompt,
    response_format,
)
from phase2.orchestration_graph import OrchestrationGraph
from phase2.registry_contract_guard import shared_db_issues
from phase2.state import PipelineState


def test_default_output_mode_prefers_schema_guidance_with_explicit_json_object_compatibility():
    from config.settings import Settings

    assert Settings.model_fields["feature_spec_output_mode"].default == "auto"
    assert response_format("auto")["type"] == "json_schema"
    assert response_format("json_object") == {"type": "json_object"}
    assert json.loads(output_schema_prompt()) == GeneratedDocument.model_json_schema()


def test_feature_prompt_and_provider_use_actual_validation_schema():
    schema = json.loads(output_schema_prompt())
    assert schema == GeneratedDocument.model_json_schema()
    assert schema == response_format("json_schema")["json_schema"]["schema"]
    prompt = PROMPT.format(project_plan="{}", prd="{}", registry="[]", output_contract=output_schema_prompt())
    assert output_schema_prompt() in prompt
    assert '"inventory.create"' not in prompt


def test_pm_example_has_all_priority_buckets_and_no_physical_contract_demand():
    prompt = PM_PROMPT.format(context="requirements", form_features_hint="")
    start, end = prompt.index("{"), prompt.index("\n}\n") + 2
    example = json.loads(prompt[start:end])
    assert set(example["projectPlan"]["priorities"]) == {"must", "should", "could", "wont"}
    assert "기능별 HTTP method/path 매핑을 포함" not in prompt
    assert "모든 *_id 참조의 실제 REFERENCES 대상을 명시" not in prompt


def test_dba_prompt_preserves_uuid_and_actual_keys_and_uses_valid_index_syntax():
    prompt = TABLE_SPEC_PROMPT.format(name="resource", purpose="purpose", all_table_names="resource", feature_registry="[]", context="")
    assert "UUID" in prompt and "JSONB" in prompt and "TIMESTAMPTZ" in prompt
    assert "CREATE INDEX idx_resource_x ON resource(x)" in prompt
    assert '"id BIGINT PRIMARY_KEY AUTO_INCREMENT"' not in prompt
    assert "실제_대상컬럼" in prompt


@pytest.mark.parametrize("domain", ["db", "api", "prd"])
def test_legacy_qa_requests_only_scoped_patches(domain):
    prompt = _REVIEW_PROMPTS[domain].format(
        draft="{}", sibling="{}", feature_str="", feedback_section="", completeness_hint="",
        leniency="", prd_summary="", db_sibling="{}", api_sibling="{}",
    )
    assert '"patches"' in prompt
    assert '"fixed"' not in prompt


def test_complete_api_contract_plan_never_calls_model_or_truncates_to_soft_cap():
    contracts = [{"method": "GET", "path": f"/api/v1/resources/{i}", "action": "read"} for i in range(30)]
    registry = [{"featureId": "resources.read", "name": "read resources", "apiContract": contracts}]
    agent = ApiAgent(object())
    result = asyncio.run(agent._manager_plan_node({"ctx": {
        "feature_list": ["read resources"], "feature_registry": json.dumps(registry),
    }}))
    assert len(result["plan"]) == 30
    assert {item["path"] for item in result["plan"]} == {item["path"] for item in contracts}
    assert not result["generation_blocker"]


def test_shared_contract_route_retains_both_feature_ids():
    operation = {"method": "GET", "path": "/api/v1/items", "action": "read"}
    registry = [{"featureId": fid, "name": fid, "apiContract": [operation]} for fid in ["items.read", "items.filter"]]
    result = asyncio.run(ApiAgent(object())._manager_plan_node({"ctx": {
        "feature_list": [item["name"] for item in registry], "feature_registry": registry,
    }}))
    assert len(result["plan"]) == 1
    assert set(result["plan"][0]["featureIds"]) == {"items.read", "items.filter"}


def test_shared_field_conflict_reports_all_participants_for_one_scoped_repair():
    features = [{"dbContract": {"tables": ["items"], "fields": [
        {"table": "items", "name": "id", "type": "UUID", "default": default, "primaryKey": True},
    ]}} for default in ["gen_random_uuid()", None, None, None]]
    issues = shared_db_issues(features)
    assert all(any(f"/features/{index}/dbContract/fields" in issue for issue in issues) for index in range(4))


def test_projected_api_plan_preserves_public_auth_and_declared_parameters():
    operation = {"method": "GET", "path": "/api/v1/catalog", "action": "read", "authRequired": False,
                 "parameters": [{"in": "query", "name": "cursor", "required": False, "schema": {"type": "string"}}]}
    registry = [{"featureId": "catalog.read", "name": "catalog", "apiContract": [operation]}]
    result = asyncio.run(ApiAgent(object())._manager_plan_node({"ctx": {
        "feature_list": ["catalog"], "feature_registry": registry,
    }}))
    assert result["plan"][0]["authRequired"] is False
    assert result["plan"][0]["parameters"] == operation["parameters"]


def test_prd_transport_error_is_not_empty_output_or_format_retry():
    calls = []

    async def fail(**_kwargs):
        calls.append(1)
        raise TimeoutError("private upstream diagnostic")

    agent = PrdAgent(SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=fail))))
    with pytest.raises(TimeoutError):
        asyncio.run(agent._call("prompt", 100, "system"))
    assert len(calls) == 1


def test_prd_transport_preserves_baseline_and_reports_real_failed_stage():
    async def fail(_input):
        raise TimeoutError("private upstream diagnostic")

    agent = PrdAgent(object())
    agent._graph = SimpleNamespace(ainvoke=fail)
    state = PipelineState(prd_document='{"coreFeatures":[]}', last_completed_stage="PM")
    result = asyncio.run(agent.execute(state))
    assert result.prd_document == state.prd_document
    assert result.failed_stage == "PRD"
    assert result.last_completed_stage == "PM"
    assert result.generation_blockers == ["PRD_TRANSPORT: upstream request failed"]
    assert "private" not in result.status_message


def test_pm_transport_stops_without_empty_plan_success():
    async def fail(**_kwargs):
        raise TimeoutError("private upstream diagnostic")

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=fail)))
    result = asyncio.run(PmAgent(client).execute(PipelineState(last_completed_stage="SEARCH")))
    assert result.failed_stage == "PM"
    assert result.last_completed_stage == "SEARCH"
    assert result.generation_blockers


def test_orchestration_prd_consumes_pm_plan_and_failure_skips_downstream():
    observed = []

    class Agent:
        def __init__(self, stage):
            self.stage = stage

        async def execute(self, state, _dump=None):
            observed.append(self.stage)
            if self.stage == "PM":
                return state.copy(project_plan={"overview": "approved plan"})
            if self.stage == "PRD":
                assert state.project_plan["overview"] == "approved plan"
                return state.copy(generation_blockers=["PRD_TRANSPORT"], failed_stage="PRD")
            return state

    graph = OrchestrationGraph(*[Agent(name) for name in ["SEARCH", "PM", "PRD", "DBA", "API", "QA"]],
                               progress_service=SimpleNamespace(send=lambda *_args: None))
    result = asyncio.run(graph.run(PipelineState()))
    assert observed == ["SEARCH", "PM", "PRD"]
    assert result.last_completed_stage == "PM"


def test_search_separates_sources_unavailable_and_failures_without_exception_evidence():
    agent = SearchAgent(object(), "test")
    prompts = []

    async def query(prompt):
        prompts.append(prompt)
        if len(prompts) == 1:
            raise TimeoutError("DO_NOT_USE_AS_EVIDENCE")
        if len(prompts) == 2:
            return "unsupported 99% statistic"
        return "Source (2026): https://example.org/report"

    agent._query = query
    text, statuses = asyncio.run(agent._collect_report("plant diary", "plant care"))
    assert [item["status"] for item in statuses] == ["failed", "unavailable", "sourced", "sourced", "sourced"]
    assert "DO_NOT_USE_AS_EVIDENCE" not in text
    assert "99%" not in text
    assert "https://example.org/report" in text
    assert all("5개 이상" not in prompt for prompt in prompts)
    assert statuses[0]["errorType"] == "TimeoutError"
