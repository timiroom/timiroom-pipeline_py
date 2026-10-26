import sys
from pathlib import Path
import asyncio
import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from openai import BadRequestError

from phase1.document_ingestion import DocumentIngestionService
from phase2.agents.feature_spec_agent import FeatureSpecAgent, _score_supporting
from phase2.feature_generation import (
    GeneratedDocument,
    _schema_without_titles,
    bind_sources,
    document_issues,
    generate_document,
    output_schema_prompt,
    response_format,
    safe_text,
    source_cores,
    strict_document,
)
from phase2.mvp_policy import dependency_issues, select_mvp
from phase2.prd_quality import reconcile_prd_quality
from phase2.state import PipelineState
from phase3.schema_validator import SchemaValidator
from phase3.validation_service import ValidationService
from phase4.kafka_producer import KafkaProducerService


def feature(name="Item create", key="items.create", source="source.items"):
    return {
        "featureId": key, "sourceFeatureId": source, "source": "prd_core",
        "parentFeatureId": None, "name": name, "description": "Validate and store the requested item.",
        "requirements": ["Store a valid name and return the new identifier; reject empty names."],
        "actions": ["create"], "priority": "P0", "reason": "User-confirmed core workflow",
        "applicability": {"ui": True, "api": True, "db": True}, "nonApplicabilityReasons": {}, "dependencies": [],
        "apiContract": [{"method": "POST", "path": "/api/v1/items", "action": "create", "featureId": key,
                         "contentType": "application/json", "successStatus": 201,
                         "requestSchema": {"type": "object", "properties": {"name": {"type": "string"}}},
                         "responseSchema": {"type": "object", "properties": {"id": {"type": "integer"}}},
                         "fieldMappings": [{"field": "name", "table": "items", "column": "name", "source": "request"}]}],
        "dbContract": {"primaryTable": "items", "tables": ["items"], "foreignKeys": [],
                       "fields": [{"table": "items", "name": "id", "type": "BIGINT", "source": "server", "generated": True},
                                  {"table": "items", "name": "name", "type": "TEXT", "source": "request"}]},
    }


def response(features=None, raw=None, finish="stop"):
    return SimpleNamespace(choices=[SimpleNamespace(finish_reason=finish, message=SimpleNamespace(
        content=raw if raw is not None else json.dumps({"features": features}), refusal=None))],
        usage=SimpleNamespace(prompt_tokens=20, completion_tokens=30, total_tokens=50))


def client(*responses):
    create = AsyncMock(side_effect=list(responses))
    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))


def run(provider, cores=None, **kwargs):
    return asyncio.run(generate_document(provider, "test", None, "JSON", cores or source_cores({"coreFeatures": [feature()]}), **kwargs))


def test_session_priority_uses_physical_contract_even_without_auth_keywords():
    item = {"sessionPolicy": {"strategy": "bearer_session", "sessionTable": "sessions"},
            "dbContract": {"tables": ["sessions"]}, "apiContract": [{
                "method": "POST", "path": "/api/v1/identity/sessions", "action": "login", "authRequired": False,
                "requestSchema": {"type": "object"}, "effects": [{"kind": "insert", "table": "sessions"}]}]}
    assert _score_supporting("Entry", "Identify the caller", feature=item) == ("P1", 10, True)
    assert _score_supporting("Entry", "Identify the caller", {"priorities": {"wont": ["Entry"]}}, feature=item)[0] == "EXCLUDE"


@pytest.mark.parametrize("raw", ['{"features":[', '```json\n{"features":[]}\n```', '{"features":[],"features":[]}', '{"features":NaN}', '{"features":null}'])
def test_strict_parse_never_partially_recovers(raw):
    features, issues = strict_document(raw)
    assert features is None and issues


def test_provider_schema_and_local_shape_agree():
    assert response_format("auto")["json_schema"]["schema"] == GeneratedDocument.model_json_schema()
    GeneratedDocument.model_validate({"features": [feature()]})
    assert document_issues([feature()], source_cores({"coreFeatures": [feature()]})) == []


def test_prompt_schema_drops_only_titles_not_constraints_or_named_properties():
    schema = {"title": "Root", "type": "object", "required": ["title"], "additionalProperties": False,
              "properties": {"title": {"title": "Title", "type": "string", "minLength": 1}},
              "default": {"title": "literal title"}, "enum": [{"title": "literal enum"}]}
    result = _schema_without_titles(schema)
    assert "title" not in result and "title" in result["properties"]
    assert result["properties"]["title"] == {"type": "string", "minLength": 1}
    assert result["required"] == ["title"] and result["additionalProperties"] is False
    assert result["default"] == schema["default"] and result["enum"] == schema["enum"]
    assert json.loads(output_schema_prompt(True))["required"] == ["version", "features", "tables", "schemas"]


def test_auto_invalid_json_uses_existing_retry_in_json_object_mode_without_recovery():
    provider = client(response(raw='{"features":['), response([feature()]))
    features, diagnostics, errors = run(provider)
    assert features and not errors
    calls = provider.chat.completions.create.call_args_list
    assert len(calls) == 2
    assert calls[0].kwargs["response_format"]["type"] == "json_schema"
    assert calls[1].kwargs["response_format"]["type"] == "json_object"
    assert all(not item.get("partialRecoveryUsed", False) for item in diagnostics)


def test_patch_prompt_names_the_exact_feature_in_addition_to_its_array_index():
    invalid = feature()
    invalid["states"] = {"items": ["open", "closed"]}
    fixed = feature()
    fixed["states"] = ["open", "closed"]
    provider = client(response([invalid]), response([fixed]))
    features, diagnostics, errors = run(provider)
    assert features and not errors and len(diagnostics) == 2
    repair_message = provider.chat.completions.create.call_args_list[1].kwargs["messages"][-1]["content"]
    assert '"repairScope"' in repair_message and '"featureId": "items.create"' in repair_message
    assert '"sourceFeatureId": "source.items"' in repair_message


def test_renamed_core_uses_source_identity_not_name():
    cores = source_cores({"coreFeatures": [feature()]})
    renamed = feature(name="Renamed workflow")
    assert document_issues([renamed], cores) == []
    assert bind_sources([renamed], cores)[0]["sourceFeatureId"] == "source.items"


def test_unbound_core_is_not_accepted_on_semantic_guess():
    unknown = feature(name="Different", key="other.create", source="unknown")
    assert any("CORE_COVERAGE" in error for error in document_issues([unknown], source_cores({"coreFeatures": [feature()]})))


@pytest.mark.parametrize("field,value", [("description", ""), ("apiContract", {}), ("requirements", []),
                                         ("sourceFeatureId", "wrong"), ("actions", []), ("priority", "P9")])
def test_malformed_contract_fields_are_located(field, value):
    item = feature()
    item[field] = value
    assert document_issues([item], source_cores({"coreFeatures": [feature()]}))


def test_ui_only_requires_a_reason_without_fake_tables():
    item = feature()
    item.update(applicability={"ui": True, "api": False, "db": False}, apiContract=[],
                dbContract={"primaryTable": None, "tables": [], "fields": [], "foreignKeys": []})
    assert any("SCOPE" in error for error in document_issues([item], source_cores({"coreFeatures": [feature()]})))
    item["nonApplicabilityReasons"] = {"api": "Local UI preference", "db": "Browser-only state"}
    assert not document_issues([item], source_cores({"coreFeatures": [feature()]}))


def test_one_generation_and_one_targeted_patch_preserve_valid_sibling():
    good = feature()
    broken = feature("Order create", "orders.create", "source.orders")
    broken["description"] = ""
    fixed = {**broken, "description": "Validate and store an order."}
    provider = client(response([good, broken]), response([fixed]))
    result, diagnostics, issues = run(provider, source_cores({"coreFeatures": [good, fixed]}))
    assert not issues and result[0] == good and result[1] == fixed
    assert len(diagnostics) == 2
    assert provider.chat.completions.create.await_count == 2
    assert "only replacements" in provider.chat.completions.create.call_args.kwargs["messages"][-1]["content"]


def test_repair_cannot_modify_valid_siblings():
    good = feature()
    broken = feature("Order create", "orders.create", "source.orders")
    broken["description"] = ""
    tampered = {**good, "description": "Unauthorized modification"}
    result, _, issues = run(client(response([good, broken]), response([tampered])), source_cores({"coreFeatures": [good, broken]}))
    assert result is None and any("REPAIR_SCOPE" in error for error in issues)


def test_missing_core_can_be_added_without_replacing_valid_core():
    good = feature()
    second = feature("Order create", "orders.create", "source.orders")
    result, _, issues = run(client(response([good]), response([second])), source_cores({"coreFeatures": [good, second]}))
    assert not issues and result == [good, second]


def test_invalid_json_uses_only_one_bounded_regeneration():
    provider = client(response(raw='{"features":['), response([feature()]))
    result, diagnostics, issues = run(provider)
    assert not issues and result
    assert diagnostics[0]["issues"][0].startswith("FEATURE_SPEC_JSON")
    assert diagnostics[0]["partialRecoveryUsed"] is False


def test_finish_length_is_not_silently_accepted():
    provider = client(response([feature()], finish="length"), response([feature()], finish="length"))
    result, _, issues = run(provider)
    assert result is None and issues == ["FEATURE_SPEC_FINISH: length"]
    assert provider.chat.completions.create.await_count == 2


def test_schema_unsupported_falls_back_only_on_explicit_capability_error():
    request = httpx.Request("POST", "https://example.test")
    exc = BadRequestError("response_format json_schema not supported", response=httpx.Response(400, request=request), body={})
    provider = client(exc, response([feature()]))
    result, diagnostics, issues = run(provider)
    assert result and not issues
    assert diagnostics[0]["code"] == "STRUCTURED_OUTPUT_UNSUPPORTED"
    assert provider.chat.completions.create.call_args.kwargs["response_format"]["type"] == "json_object"


def test_auth_error_does_not_retry_or_downgrade():
    exc = SimpleError()
    provider = client(exc)
    result, _, issues = run(provider)
    assert result is None and issues
    assert provider.chat.completions.create.await_count == 1


class SimpleError(Exception):
    status_code = 401


def test_diagnostics_are_hashes_and_opt_in_files_redact_tokens(tmp_path):
    raw = json.dumps({"features": [feature()], "api_key": "sk-secretvalue", "password": "private"})
    _, diagnostics, issues = run(client(response(raw=raw)), diagnostics_dir=str(tmp_path))
    assert not issues and diagnostics[-1]["usage"]["total_tokens"] == 50
    assert "sk-secretvalue" not in json.dumps(diagnostics)
    saved = next(tmp_path.iterdir()).read_text(encoding="utf-8")
    assert "sk-secretvalue" not in saved and '"private"' not in saved
    assert "xoxb-token" not in safe_text("xoxb-token")


def test_feature_failure_keeps_baseline_and_only_reports_root_cause():
    state = PipelineState(prd_document=json.dumps({"coreFeatures": [feature()]}), feature_registry=[feature()])
    result = asyncio.run(FeatureSpecAgent(client(response(raw="broken"), response(raw="broken")), "test").execute(state))
    assert result.feature_registry == state.feature_registry and result.failed_stage == "FEATURE_SPEC"
    validated = ValidationService(SchemaValidator()).validate(result)
    assert validated.validation_error_codes == ["FEATURE_SPEC_CONTRACT_BLOCKER"]
    assert not validated.qa_db_blockers and not validated.qa_api_blockers
    assert not validated.validation_repair_targets


def test_generated_feature_rename_updates_prd_with_same_source(monkeypatch):
    original = feature()
    renamed = feature(name="Renamed workflow")
    async def generated(*_args, **kwargs):
        assert kwargs["draft"] is True
        return [renamed], [], []

    monkeypatch.setattr("phase2.agents.feature_spec_agent.generate_document", generated)
    state = PipelineState(prd_document=json.dumps({"coreFeatures": [original]}), feature_registry=[original])
    result = asyncio.run(FeatureSpecAgent(client(response([renamed])), "test").execute(state))
    assert not result.generation_blockers
    assert json.loads(result.prd_document)["coreFeatures"][0]["name"] == "Renamed workflow"


def test_prd_auth_retention_and_persona_are_requirement_derived():
    prd = {"coreFeatures": [{"name": "식물 등록", "requirements": ["사용자별 식물만 조회한다."]}],
           "techStack": {"auth": "인증이 명시된 경우에만"},
           "kpi": [{"metric": "7일 재방문율", "measurementMethod": "7일 이내 재방문"}, {"metric": "30일 리텐션"}],
           "userPersonas": [{"goal": "일반 업무", "usagePattern": "웹 사용"}],
           "_qualityFallbacks": ["techStack:worker_validation_failed"]}
    out = reconcile_prd_quality(prd)
    assert "소유권" in out["techStack"]["auth"]
    assert out["kpi"][0]["measurementDefinition"]["kind"] == "cumulative_return"
    assert out["kpi"][1]["measurementDefinition"]["windowStartDay"] == 30
    assert "식물 등록" in out["userPersonas"][0]["usagePattern"]
    assert out["qualityMetadata"]["fallbacks"]
    assert "measurementDefinition" not in prd["kpi"][0]


def test_public_service_is_not_forced_to_authenticate():
    prd = {"coreFeatures": [{"name": "공개 조회", "requirements": ["누구나 조회한다."]}], "techStack": {"auth": "인증 없음"}}
    assert reconcile_prd_quality(prd)["techStack"]["auth"] == "인증 없음"


def test_storage_confirmation_requires_actual_receipt():
    state = PipelineState(project_name="Demo", feature_list=["Item"], prd_document="{}", api_spec="{}", db_schema="{}")
    ingestion = SimpleNamespace(event_storage_status=AsyncMock(return_value={"status": "stored", "expectedChunks": 3, "storedChunks": 3}))
    producer = KafkaProducerService("unused", "unused", "unused", ingestion_service=ingestion)
    receipt = asyncio.run(producer.confirm_storage(state, "pipeline"))
    assert receipt["status"] == "stored" and receipt["kafkaPublished"]
    assert {meta["type"] for _, meta in ingestion.event_storage_status.call_args.args[0]} == {"prd", "erd", "api", "features"}


def test_storage_timeout_is_pending_not_completed():
    state = PipelineState(project_name="Demo", feature_list=["Item"], prd_document="{}", api_spec="{}", db_schema="{}")
    ingestion = SimpleNamespace(event_storage_status=AsyncMock(return_value={"status": "pending", "storedChunks": 0}))
    producer = KafkaProducerService("unused", "unused", "unused", ingestion_service=ingestion, storage_timeout_seconds=0.01)
    receipt = asyncio.run(producer.confirm_storage(state, "pipeline"))
    assert receipt["status"] == "pending" and receipt["reason"] == "STORAGE_CONFIRMATION_TIMEOUT"


def test_partial_or_other_pipeline_chunks_do_not_confirm_storage():
    async def scenario():
        ingestion = object.__new__(DocumentIngestionService)
        ingestion._db_semaphore = asyncio.Semaphore(1)
        documents = [("x" * 1601, {"pipeline_id": "current", "type": "prd"})]
        ingestion._stored_source_keys = lambda keys: {keys[0], "unrelated-pipeline-key"}
        pending = await ingestion.event_storage_status(documents)
        ingestion._stored_source_keys = set
        stored = await ingestion.event_storage_status(documents)
        return pending, stored
    pending, stored = asyncio.run(scenario())
    assert pending == {"status": "pending", "expectedChunks": 3, "storedChunks": 1}
    assert stored == {"status": "stored", "expectedChunks": 3, "storedChunks": 3}


def test_prd_quality_is_idempotent():
    prd = {"coreFeatures": [], "kpi": [{"metric": "7일 재방문율", "measurementMethod": "7일 이내 재방문"}]}
    first = reconcile_prd_quality(prd)
    assert reconcile_prd_quality(first) == first


def test_parent_group_is_not_an_execution_dependency():
    parent = {"featureId": "items.create", "name": "Create", "priority": "P0", "source": "prd_core", "dependencies": ["auth.login"]}
    child = {"featureId": "auth.login", "name": "Login", "priority": "P1", "source": "supporting", "parentFeatureId": "items.create"}
    assert not dependency_issues([parent, child])
    assert {item["featureId"] for item in select_mvp([parent, child])} == {"items.create", "auth.login"}
    child["dependencies"] = ["items.create"]
    assert dependency_issues([parent, child])


def test_dependency_cycle_repair_has_feature_scope():
    first = feature()
    second = feature("Order create", "orders.create", "source.orders")
    first["dependencies"] = ["orders.create"]
    second["dependencies"] = ["items.create"]
    fixed = {**second, "dependencies": []}
    provider = client(response([first, second]), response([fixed]))
    result, _, issues = run(provider, source_cores({"coreFeatures": [first, second]}))
    assert result and not issues


def test_extra_attempts_patch_the_previous_draft_until_contracts_pass():
    """Feature Spec은 fallback이 없으므로 보정 기회가 모자라면 그대로 하드 블록이다.

    보정은 이전 draft 위의 targeted patch라서, 시도를 늘리면 남은 위반이
    줄어든다. 실측에서 초안(231s/61KB)의 1/15 비용(15s/4KB)으로 3건을 고쳤다.
    """
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    from phase2.feature_draft import compile_document
    from phase2.feature_generation import document_issues, generate_document

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from test_flat_feature_draft import draft, response

    good = draft()
    cores = [{"sourceFeatureId": compile_document(good)[0]["sourceFeatureId"]}]
    broken = {"version": "contract-draft-v1", "features": [], "tables": [], "schemas": []}

    # 2회로는 모자라고 3회째에 통과하는 경우.
    def run(max_attempts):
        create = AsyncMock(side_effect=[
            response(broken), response(broken), response(good),
        ])
        provider = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        result, _diagnostics, issues = asyncio.run(generate_document(
            provider, "test", None, "draft", cores, draft=True, max_attempts=max_attempts,
        ))
        return result, issues, create.await_count

    _result, issues, calls = run(2)
    assert issues and calls == 2, (issues, calls)

    result, issues, calls = run(3)
    assert not issues and calls == 3, (issues, calls)
    assert not document_issues(result, cores, preflight=True)
