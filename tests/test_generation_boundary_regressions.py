import copy
import json

from phase2.agents.dba_agent import _clean_constraint, dba_self_check
from phase2.repair_scope import repair_targets
from phase2.state import PipelineState
from phase3.schema_validator import SchemaValidator
from phase3.validation_service import ValidationService


def test_generation_failure_details_preserve_individual_producer_and_operation_pointer():
    state = PipelineState(failed_stage="FEATURE_SPEC", generation_blockers=["FEATURE_SPEC_CONTRACT_BLOCKER: bundled"],
                          generation_diagnostics=[{"issues": ["API_BAD: /features/0/apiContract/2: wrong"],
                                                   "featureRefs": [{"index": 0, "featureId": "entries.create"}]}])
    result = ValidationService(SchemaValidator()).validate(state)
    detail = result.qa_blocker_details[0]
    assert detail["code"] == "API_BAD" and detail["agent"] == "FEATURE_SPEC"
    assert detail["pointer"] == "/features/0/apiContract/2" and detail["featureId"] == "entries.create"
    assert detail["repairable"] is False and result.validation_repair_targets == []


def test_constraint_cleaning_keeps_declared_non_id_and_composite_keys():
    assert "REFERENCES catalogs(code)" in _clean_constraint("FOREIGN_KEY REFERENCES catalogs(code)")
    assert "REFERENCES catalogs(tenant_id,code)" in _clean_constraint("REFERENCES catalogs(tenant_id, code)")


def test_self_check_reconciles_fk_after_explicit_target_type_and_reference():
    tables = [
        {"name": "catalogs", "columns": [{"name": "id", "type": "BIGINT", "constraints": "PRIMARY_KEY GENERATED_IDENTITY"},
                                            {"name": "code", "type": "TEXT", "constraints": "UNIQUE"}], "indexes": []},
        {"name": "entries", "columns": [{"name": "id", "type": "BIGINT", "constraints": "PRIMARY_KEY"},
                                           {"name": "catalog_id", "type": "BIGINT", "constraints": "REFERENCES catalogs(id)"},
                                           {"name": "catalog_code", "type": "TEXT", "constraints": "REFERENCES catalogs(id)"}], "indexes": []},
    ]
    registry = [{"featureId": "entries.create", "dbContract": {
        "tables": ["catalogs", "entries"], "primaryTable": "entries",
        "fields": [{"table": "catalogs", "name": "id", "type": "UUID"}],
        "foreignKeys": [{"table": "entries", "column": "catalog_code", "references": {"table": "catalogs", "column": "code"}}],
    }}]
    result, _, issues = dba_self_check(copy.deepcopy(tables), [], registry)
    columns = {table["name"]: {column["name"]: column for column in table["columns"]} for table in result}
    assert columns["entries"]["catalog_id"]["type"] == "UUID"
    assert "REFERENCES catalogs(code)" in columns["entries"]["catalog_code"]["constraints"]
    assert columns["entries"]["catalog_code"]["type"] == "TEXT"
    assert "GENERATED_IDENTITY" not in columns["catalogs"]["id"]["constraints"]
    assert not any("TYPE_MISMATCH" in issue for issue in issues)


def test_repair_scope_resolves_api_and_db_json_pointers_without_names():
    state = PipelineState(qa_blocker_details=[{"domain": "api", "pointer": "/endpoints/1/effects/0", "message": "Invalid dictionary"}])
    api = {"endpoints": [{"method": "GET", "path": "/api/v1/catalogs"}, {"method": "POST", "path": "/api/v1/entries"}]}
    assert repair_targets(state, "Invalid dictionary", "api", api) == {"POST /api/v1/entries"}
    state.qa_blocker_details = [{"domain": "db", "pointer": "/tables/0/columns/1", "message": "Invalid column"}]
    assert repair_targets(state, "Invalid column", "db", {"tables": [{"name": "entries"}]}) == {"entries"}


def test_shape_diagnostics_retain_endpoint_keys_and_distinct_messages():
    endpoints = [{"method": "POST", "path": path, "effects": ["unstructured effect"]}
                 for path in ("/api/v1/catalogs", "/api/v1/entries")]
    result = SchemaValidator().validate(["Catalog"], '{"tables":[]}', json.dumps({"endpoints": endpoints}), '{}')
    details = [item for item in result.blocker_details if item["code"] == "ARTIFACT_SHAPE_INVALID"]
    assert {item["artifactKey"] for item in details} == {"POST /api/v1/catalogs", "POST /api/v1/entries"}
    assert len({item["message"] for item in details}) == 2
