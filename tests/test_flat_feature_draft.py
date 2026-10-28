import asyncio
import copy
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from phase2.agents.feature_spec_agent import FeatureSpecAgent
from phase2.feature_draft import (
    DraftDocument,
    SchemaDraft,
    compile_document,
    compile_schemas,
    merge_patch,
    repair_scope,
)
from phase2.feature_generation import (
    _ownership_repair_candidates,
    document_issues,
    generate_document,
    generation_input,
    source_cores,
    strict_payload,
)
from phase2.openapi_spec import build_openapi_document
from phase2.state import PipelineState


def draft():
    return {
        "version": "contract-draft-v1",
        "tables": [{"name": "items", "fields": [
            {"name": "id", "type": "BIGINT", "generated": True, "primaryKey": True, "source": "server"},
            {"name": "name", "type": "TEXT", "required": True},
        ]}],
        "schemas": [
            {"name": "CreateItem", "nodes": [{"id": "root", "kind": "object"},
                                             {"id": "name", "parent": "root", "name": "name", "kind": "string", "required": True}]},
            {"name": "Item", "nodes": [{"id": "root", "kind": "object"},
                                       {"id": "id", "parent": "root", "name": "id", "kind": "integer", "required": True},
                                       {"id": "name", "parent": "root", "name": "name", "kind": "string", "required": True}]},
        ],
        "features": [{
            "featureId": "items.create", "sourceFeatureId": "source.items", "source": "prd_core",
            "parentFeatureId": None, "name": "Item create", "description": "Validate and store an item.",
            "requirements": ["Store a valid name; reject an empty name and return the new item."],
            "reason": "Core workflow", "actions": ["create"], "priority": "P0",
            "ownership": {"scope": "PUBLIC", "access": "Public item creation"},
            "applicability": {"ui": True, "api": True, "db": True}, "dbTables": ["items"], "primaryTable": "items",
            "apiContract": [{"method": "POST", "path": "/api/v1/items", "action": "create", "authRequired": False,
                             "transactionRules": "Insert the item atomically; roll back on failure.",
                             "successStatus": 201, "requestType": "CreateItem", "responseType": "Item",
                             "fieldMappings": [{"field": "name", "table": "items", "column": "name", "source": "body"}]}],
        }],
    }


def response(value):
    return SimpleNamespace(choices=[SimpleNamespace(finish_reason="stop", message=SimpleNamespace(
        content=value if isinstance(value, str) else json.dumps(value), refusal=None))], usage=None)


def test_preflight_shared_route_conflicts_scope_both_owners_before_api_generation():
    payload = draft()
    sibling = copy.deepcopy(payload["features"][0])
    sibling.update(featureId="items.secondary", sourceFeatureId="source.secondary")
    sibling["apiContract"][0]["successStatus"] = 202
    payload["features"].append(sibling)
    cores = [{"sourceFeatureId": "source.items"}, {"sourceFeatureId": "source.secondary"}]
    issues = document_issues(compile_document(payload), cores, preflight=True)
    conflicts = [issue for issue in issues if issue.startswith("API_OPERATION_CONTRACT_CONFLICT:")]
    assert any("/features/0/apiContract/0" in issue for issue in conflicts)
    assert any("/features/1/apiContract/0" in issue for issue in conflicts)
    scope = repair_scope(payload, issues, cores)
    assert scope["operationIndices"] == {"items.create": [0], "items.secondary": [0]}
    merged = merge_patch(payload, {"features": [{"featureId": "items.secondary", "apiContract": [
        {"method": "POST", "path": "/api/v1/items", "successStatus": 201},
    ]}]}, scope)
    assert not document_issues(compile_document(merged), cores, preflight=True)


def test_scoped_existing_dto_can_introduce_a_new_referenced_dto_only():
    payload = draft()
    scope = repair_scope(payload, ["API_REQUIRED_OUTPUT_MISSING: /features/0: items.create: detail"], [])
    modified = copy.deepcopy(payload["schemas"][1])
    modified["nodes"].append({"id": "detail", "parent": "root", "name": "detail", "kind": "ref", "ref": "ItemDetail", "required": True})
    detail = {"name": "ItemDetail", "nodes": [{"id": "root", "kind": "object"},
                                             {"id": "description", "parent": "root", "name": "description", "kind": "string"}]}
    patch = {"schemas": [modified, detail]}
    merged = merge_patch(payload, patch, scope)
    assert "detail" in compile_document(merged)[0]["apiContract"][0]["responseSchema"]["properties"]
    patch["schemas"].append({**copy.deepcopy(detail), "name": "UnrelatedDetail"})
    with pytest.raises(ValueError, match="unrequested schemas patch UnrelatedDetail"):
        merge_patch(payload, patch, scope)


def test_feature_wide_patch_merges_sparse_operations_by_route_without_losing_siblings():
    payload = draft()
    sibling = {**copy.deepcopy(payload["features"][0]["apiContract"][0]), "path": "/api/v1/other-items"}
    payload["features"][0]["apiContract"].append(sibling)
    scope = repair_scope(payload, ["FEATURE_OWNER_TABLE_UNKNOWN: /features/0/ownership: missing owner"], [])
    patch = {"features": [{"featureId": "items.create", "apiContract": [{
        "method": "POST", "path": "/api/v1/items", "transactionRules": "Corrected atomic insert policy",
    }]}]}
    merged = merge_patch(payload, patch, scope)
    operations = merged["features"][0]["apiContract"]
    assert len(operations) == 2 and operations[1] == sibling
    assert operations[0] == {**payload["features"][0]["apiContract"][0], "transactionRules": "Corrected atomic insert policy"}
    compile_document(merged)


def test_partial_output_repair_exposes_protected_shared_dto_and_accepts_new_projection():
    payload = draft()
    sibling = copy.deepcopy(payload["features"][0])
    sibling.update(featureId="items.other", sourceFeatureId="source.other")
    sibling["apiContract"][0]["path"] = "/api/v1/other-items"
    payload["features"].append(sibling)
    scope = repair_scope(payload, ["API_REQUIRED_OUTPUT_MISSING: /features/0: items.create: extra"], [])
    assert "Item" in scope["protectedSchemaNames"] and "Item" not in scope["schemaNames"]
    extra = copy.deepcopy(payload["schemas"][1])
    extra["name"] = "DetailedItem"
    extra["nodes"].append({"id": "extra", "parent": "root", "name": "extra", "kind": "string"})
    patch = {"features": [{"featureId": "items.create", "apiContract": [{"method": "POST", "path": "/api/v1/items", "responseType": "DetailedItem"}]}],
             "schemas": [extra]}
    merged = merge_patch(payload, patch, scope)
    assert merged["features"][1] == sibling
    assert compile_document(merged)[0]["apiContract"][0]["responseSchema"]["properties"].get("extra")
    patch["schemas"][0]["name"] = "Item"
    with pytest.raises(ValueError, match="unrelated endpoint schema changed: Item"):
        merge_patch(payload, patch, scope)


def test_draft_compiles_to_existing_contract_and_passes_unchanged_guards():
    payload = draft()
    DraftDocument.model_validate(payload)
    features = compile_document(payload)
    assert features[0]["apiContract"][0]["requestSchema"]["required"] == ["name"]
    assert features[0]["apiContract"][0]["authRequired"] is False
    assert document_issues(features, source_cores({"coreFeatures": [{"sourceFeatureId": "source.items"}]})) == []
    assert "requestType" not in features[0]["apiContract"][0]


def test_draft_preserves_explicit_nonstandard_owner_keys():
    payload = draft()
    ownership = {"scope": "SHARED", "ownerEntity": "organizations", "ownerKey": "organization_id", "access": "Verify current membership"}
    payload["features"][0]["ownership"] = ownership
    assert compile_document(payload)[0]["ownership"] == ownership


def test_shared_table_is_defined_once_and_projected_identically_without_aliasing():
    payload = draft()
    second = copy.deepcopy(payload["features"][0])
    second.update(featureId="items.read", sourceFeatureId="source.read", actions=["read"])
    second["apiContract"] = [{"method": "GET", "path": "/api/v1/items", "action": "read", "successStatus": 200, "responseType": "Item"}]
    payload["features"].append(second)
    features = compile_document(payload)
    assert features[0]["dbContract"]["fields"] == features[1]["dbContract"]["fields"]
    features[0]["dbContract"]["fields"][0]["type"] = "UUID"
    assert features[1]["dbContract"]["fields"][0]["type"] == "BIGINT"


def test_flat_nodes_compile_nested_array_nullable_enum_binary_and_ref():
    definitions = [SchemaDraft.model_validate(value) for value in [
        {"name": "File", "nodes": [{"id": "root", "kind": "string", "format": "binary"}]},
        {"name": "Payload", "nodes": [
            {"id": "root", "kind": "object", "additionalProperties": False},
            {"id": "entries", "parent": "root", "name": "entries", "kind": "array", "required": True},
            {"id": "entry", "parent": "entries", "kind": "object"},
            {"id": "status", "parent": "entry", "name": "status", "kind": "string", "nullable": True, "enum": ["new", "done", None]},
            {"id": "file", "parent": "entry", "name": "file", "kind": "ref", "ref": "File"},
        ]},
    ]]
    schema = compile_schemas(definitions)["Payload"]
    assert schema["required"] == ["entries"]
    assert schema["properties"]["entries"]["items"]["properties"]["status"]["type"] == ["string", "null"]
    assert schema["properties"]["entries"]["items"]["properties"]["file"]["format"] == "binary"
    assert schema["additionalProperties"] is False


@pytest.mark.parametrize("nodes", [
    [{"id": "root", "kind": "array"}],
    [{"id": "root", "kind": "string"}, {"id": "child", "parent": "root", "name": "bad", "kind": "integer"}],
    [{"id": "root", "kind": "object"}, {"id": "bad", "parent": "missing", "name": "name", "kind": "string"}],
    [{"id": "root", "kind": "object"}, {"id": "a", "parent": "b", "name": "a", "kind": "object"}, {"id": "b", "parent": "a", "name": "b", "kind": "object"}],
])
def test_invalid_flat_graph_is_rejected_not_guessed(nodes):
    with pytest.raises(ValueError):
        compile_schemas([SchemaDraft.model_validate({"name": "Invalid", "nodes": nodes})])


def test_schema_reference_cycle_is_rejected():
    definitions = [SchemaDraft.model_validate({"name": name, "nodes": [{"id": "root", "kind": "ref", "ref": target}]})
                   for name, target in [("A", "B"), ("B", "A")]]
    with pytest.raises(ValueError, match="cyclic"):
        compile_schemas(definitions)


def test_scalar_mapping_refuses_csv_columns_at_generation_boundary():
    payload = draft()
    payload["features"][0]["apiContract"][0]["fieldMappings"][0]["column"] = "id,name"
    with pytest.raises(ValueError, match="fieldMappings/0/column"):
        compile_document(payload)


def test_declared_fk_dependencies_are_projected_from_shared_table_definitions():
    payload = draft()
    payload["tables"][0]["fields"].append({"name": "category_id", "type": "BIGINT"})
    payload["tables"][0]["foreignKeys"] = [{"column": "category_id", "references": {"table": "categories", "column": "id"}}]
    payload["tables"].append({"name": "categories", "fields": [{"name": "id", "type": "BIGINT", "primaryKey": True}]})
    result = compile_document(payload)
    assert result[0]["dbContract"]["tables"] == ["items", "categories"]
    assert any(field["table"] == "categories" and field["name"] == "id" for field in result[0]["dbContract"]["fields"])


def test_bounded_draft_shape_repair_merges_only_invalid_feature():
    initial = draft()
    initial["features"][0]["apiContract"][0]["fieldMappings"][0]["column"] = "name,id"
    repaired = {"version": "contract-draft-v1", "features": draft()["features"], "tables": [], "schemas": []}
    create = AsyncMock(side_effect=[response(initial), response(repaired)])
    provider = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    result, diagnostics, issues = asyncio.run(generate_document(provider, "test", None, "draft",
        source_cores({"coreFeatures": [{"sourceFeatureId": "source.items"}]}), draft=True))
    assert result and not issues and create.await_count == 2
    assert len(diagnostics) == 2
    assert '"items.create"' in create.call_args.kwargs["messages"][-1]["content"]


def test_repair_rejects_changes_to_an_unrelated_endpoint_inside_same_feature():
    payload = draft()
    payload["features"][0]["apiContract"].append({"method": "GET", "path": "/api/v1/items", "action": "read", "successStatus": 200, "responseType": "Item"})
    scope = repair_scope(payload, ["SHAPE: /features/0/apiContract/0/fieldMappings: bad mapping"], [])
    replacement = copy.deepcopy(payload["features"][0])
    replacement["apiContract"][1]["path"] = "/api/v1/unrelated"
    with pytest.raises(ValueError, match="unrelated endpoint"):
        merge_patch(payload, {"version": "contract-draft-v1", "features": [replacement]}, scope)


def test_schema_repair_scope_is_exact_and_does_not_match_short_names_in_error_codes():
    payload = draft()
    payload["schemas"].append({"name": "A", "nodes": [{"id": "root", "kind": "string"}]})
    scope = repair_scope(payload, ["FEATURE_DRAFT_SCHEMA_ITEMS: /schemas/0/nodes: root"], [])
    assert scope["schemaNames"] == ["CreateItem"]


def test_union_schema_and_invalid_bounds():
    definition = SchemaDraft.model_validate({"name": "Value", "nodes": [
        {"id": "root", "kind": "oneOf"}, {"id": "s", "parent": "root", "kind": "string"},
        {"id": "n", "parent": "root", "kind": "integer"},
    ]})
    assert compile_schemas([definition])["Value"]["oneOf"] == [{"type": "string"}, {"type": "integer"}]
    schema = compile_schemas([definition])["Value"]
    api = build_openapi_document({"endpoints": [{"method": "GET", "path": "/api/v1/value", "responseSchema": schema}]})
    assert api["paths"]["/api/v1/value"]["get"]["responses"]["200"]["content"]["application/json"]["schema"] == schema
    bad = SchemaDraft.model_validate({"name": "Bad", "nodes": [{"id": "root", "kind": "string", "minLength": 5, "maxLength": 2}]})
    with pytest.raises(ValueError, match="BOUNDS"):
        compile_schemas([bad])


def test_draft_generation_has_one_call_and_publishes_canonical_only():
    create = AsyncMock(return_value=response(draft()))
    provider = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    cores = source_cores({"coreFeatures": [{"sourceFeatureId": "source.items"}]})
    features, diagnostics, issues = asyncio.run(generate_document(provider, "test", None, "draft", cores, draft=True))
    assert not issues and len(diagnostics) == 1 and create.await_count == 1
    assert "requestSchema" in features[0]["apiContract"][0]
    assert "tables" in create.call_args.kwargs["response_format"]["json_schema"]["schema"]["properties"]


def test_scoped_draft_repair_keeps_shared_tables_and_valid_siblings():
    payload = draft()
    second = copy.deepcopy(payload["features"][0])
    second.update(featureId="items.other", sourceFeatureId="source.other")
    payload["features"].append(second)
    scope = repair_scope(payload, ["FEATURE_SPEC_REQUIRED: /features/0/name: blank"], [])
    replacement = copy.deepcopy(payload["features"][0])
    replacement["name"] = "Changed"
    patch = {"version": "contract-draft-v1", "features": [replacement], "tables": [], "schemas": []}
    result = merge_patch(payload, patch, scope)
    assert result["features"][1] == second
    assert result["tables"] == payload["tables"]
    patch["features"] = [second]
    with pytest.raises(ValueError, match="unrequested"):
        merge_patch(payload, patch, scope)


def test_invalid_json_still_has_only_one_retry_and_no_partial_recovery():
    create = AsyncMock(side_effect=[response('{"features":['), response('{"features":[')])
    provider = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    result, diagnostics, issues = asyncio.run(generate_document(provider, "test", None, "draft", [], draft=True, mode="json_object"))
    assert result is None and issues and create.await_count == 2
    assert all(not item["partialRecoveryUsed"] for item in diagnostics)


def test_empty_initial_draft_has_no_valid_siblings_to_preserve_and_retries_once():
    create = AsyncMock(side_effect=[response({"version": "contract-draft-v1", "features": [], "tables": [], "schemas": []}), response(draft())])
    provider = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    result, _, issues = asyncio.run(generate_document(provider, "test", None, "draft",
        source_cores({"coreFeatures": [{"sourceFeatureId": "source.items"}]}), draft=True))
    assert result and not issues and create.await_count == 2


@pytest.mark.parametrize(("collection", "value", "pointer"), [
    ("parameters", 3, "/features/0/apiContract/0/parameters"),
    ("apiContract", 3, "/features/0/apiContract"),
    ("dbTables", 3, "/features/0/dbTables"),
    ("tables", [None, {"name": "items", "foreignKeys": [{"references": 3}, None]}], "/tables/0"),
    ("schemas", [None], "/schemas/0"),
])
def test_malformed_draft_values_do_not_crash_repair_scope(collection, value, pointer):
    payload = draft()
    if collection == "parameters":
        payload["features"][0]["apiContract"][0][collection] = value
    elif collection in {"apiContract", "dbTables"}:
        payload["features"][0][collection] = value
    else:
        payload[collection] = value
    scope = repair_scope(payload, [f"FEATURE_DRAFT_SHAPE: {pointer}: invalid", "FK_ERROR: /features/0/dbContract: invalid"], [])
    assert scope["featureIds"] == ["items.create"]


def test_malformed_parameter_array_is_repaired_once_without_escaping_generation_boundary():
    initial = draft()
    initial["features"][0]["apiContract"][0]["parameters"] = 3
    patch = {"version": "contract-draft-v1", "features": draft()["features"]}
    create = AsyncMock(side_effect=[response(initial), response(patch)])
    provider = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    result, _, issues = asyncio.run(generate_document(provider, "test", None, "draft",
        source_cores({"coreFeatures": [{"sourceFeatureId": "source.items"}]}), draft=True))
    assert result and not issues and create.await_count == 2


def test_invalid_baseline_collection_yields_classified_error_not_type_error():
    payload = draft()
    payload["schemas"] = 3
    scope = repair_scope(payload, ["FEATURE_DRAFT_SHAPE: /schemas: list required"], [])
    with pytest.raises(ValueError, match="FEATURE_SPEC_REPAIR_SCOPE: /schemas"):
        merge_patch(payload, {"version": "contract-draft-v1", "schemas": []}, scope)


def test_unrepairable_collection_remains_a_classified_blocker_with_bounded_calls():
    initial = draft()
    initial["schemas"] = 3
    create = AsyncMock(side_effect=[response(initial), response({"version": "contract-draft-v1", "features": [], "schemas": []})])
    provider = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    result, diagnostics, issues = asyncio.run(generate_document(provider, "test", None, "draft", [], draft=True))
    assert result is None and create.await_count == 2
    assert len(diagnostics) == 2
    assert issues == ["FEATURE_DRAFT_SHAPE: /tables: Field required"]
    assert diagnostics[0]["retryKind"] == "UNUSABLE_DRAFT_RETRY"


def missing_core_patch():
    template = draft()
    feature = template["features"][0]
    feature.update(featureId="orders.create", sourceFeatureId="source.orders", dbTables=["orders"], primaryTable="orders")
    operation = feature["apiContract"][0]
    operation.update(path="/api/v1/orders", responseType="Order")
    operation["fieldMappings"][0]["table"] = "orders"
    table = template["tables"][0]
    table.update(name="orders", foreignKeys=[{"column": "category_id", "nullable": True,
                                             "references": {"table": "categories", "column": "id"}}])
    table["fields"].append({"name": "category_id", "type": "BIGINT"})
    dto = copy.deepcopy(template["schemas"][1])
    dto["name"] = "Order"
    dto["nodes"][1].update(kind="ref", ref="OrderId")
    return {"version": "contract-draft-v1", "features": [feature],
            "tables": [table, {"name": "categories", "fields": [copy.deepcopy(table["fields"][0])]}],
            "schemas": [dto, {"name": "OrderId", "nodes": [{"id": "root", "kind": "integer"}]}]}


def test_missing_core_can_add_only_its_reachable_new_tables_and_dtos():
    payload, patch = draft(), missing_core_patch()
    cores = [{"sourceFeatureId": "source.items"}, {"sourceFeatureId": "source.orders"}]
    scope = repair_scope(payload, document_issues(compile_document(payload), cores), cores)
    merged = merge_patch(payload, patch, scope)
    assert merged["features"][0] == payload["features"][0]
    assert merged["tables"][0] == payload["tables"][0]
    assert merged["schemas"][:2] == payload["schemas"]
    assert document_issues(compile_document(merged), cores) == []


@pytest.mark.parametrize("collection", ["tables", "schemas"])
def test_missing_core_repair_cannot_add_unrelated_or_overwrite_existing_resources(collection):
    payload, patch = draft(), missing_core_patch()
    scope = repair_scope(payload, ["FEATURE_SPEC_CORE_COVERAGE: source.orders: missing"], [{"sourceFeatureId": "source.orders"}])
    extra = copy.deepcopy(payload[collection][0])
    patch[collection].append(extra)
    with pytest.raises(ValueError, match="unrequested"):
        merge_patch(payload, patch, scope)
    extra["name"] = "unrelated" if collection == "tables" else "Unrelated"
    with pytest.raises(ValueError, match="unrequested"):
        merge_patch(payload, patch, scope)


def test_new_resource_permissions_require_missing_core_in_the_same_patch():
    payload, patch = draft(), missing_core_patch()
    patch["features"] = []
    scope = repair_scope(payload, ["FEATURE_SPEC_CORE_COVERAGE: source.orders: missing"], [{"sourceFeatureId": "source.orders"}])
    with pytest.raises(ValueError, match="unrequested"):
        merge_patch(payload, patch, scope)


def test_missing_core_and_dependencies_are_merged_in_one_bounded_generation_retry():
    cores = [{"sourceFeatureId": "source.items"}, {"sourceFeatureId": "source.orders"}]
    create = AsyncMock(side_effect=[response(draft()), response(missing_core_patch())])
    provider = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    result, _, issues = asyncio.run(generate_document(provider, "test", None, "draft", cores, draft=True))
    assert result and not issues and create.await_count == 2
    assert len(result) == 2


def test_missing_response_schema_has_exact_operation_pointer_and_preserves_valid_sibling():
    payload = draft()
    payload["features"][0]["apiContract"].append({"method": "GET", "path": "/api/v1/items", "action": "read",
                                                "successStatus": 200, "responseType": "Item"})
    payload["features"][0]["apiContract"][0].pop("responseType")
    cores = [{"sourceFeatureId": "source.items"}]
    errors = document_issues(compile_document(payload), cores)
    assert any("/features/0/apiContract/0" in issue and "responseSchema required" in issue for issue in errors)
    scope = repair_scope(payload, errors, cores)
    assert scope["operationIndices"] == {"items.create": [0]}
    replacement = copy.deepcopy(payload["features"][0])
    replacement["apiContract"][0]["responseType"] = "Item"
    patch = {"version": "contract-draft-v1", "features": [replacement]}
    result = merge_patch(payload, patch, scope)
    assert not document_issues(compile_document(result), cores)
    replacement["apiContract"][1]["path"] = "/api/v1/changed"
    with pytest.raises(ValueError, match="unrelated endpoint"):
        merge_patch(payload, patch, scope)


def test_operation_repair_cannot_change_shared_dto_used_by_healthy_endpoint():
    payload = draft()
    payload["features"][0]["apiContract"].append({"method": "GET", "path": "/api/v1/items", "action": "read",
                                                "successStatus": 200, "responseType": "Item"})
    scope = repair_scope(payload, ["INVALID: /features/0/apiContract/0/fieldMappings: bad"], [])
    dto = copy.deepcopy(payload["schemas"][1])
    dto["nodes"][1]["kind"] = "string"
    with pytest.raises(ValueError, match="unrelated endpoint schema"):
        merge_patch(payload, {"version": "contract-draft-v1", "schemas": [dto]}, scope)


def test_operation_index_protection_allows_replacing_a_malformed_operation_object():
    payload = draft()
    payload["features"][0]["apiContract"][0] = None
    scope = repair_scope(payload, ["FEATURE_DRAFT_SHAPE: /features/0/apiContract/0: object required"], [])
    result = merge_patch(payload, {"version": "contract-draft-v1", "features": draft()["features"]}, scope)
    assert result["features"] == draft()["features"]


def test_operation_owner_error_uses_full_feature_and_operation_indices():
    payload = draft()
    features = compile_document(payload)
    second = copy.deepcopy(features[0])
    second.update(featureId="items.other", sourceFeatureId="source.other")
    features.append(second)
    errors = document_issues(features, [{"sourceFeatureId": "source.items"}, {"sourceFeatureId": "source.other"}])
    assert any("FEATURE_SPEC_OPERATION_OWNER: /features/1/apiContract/0:" in issue for issue in errors)


@pytest.mark.parametrize("identifier", ["intent.a123.create", "supporting.auth", "feature_001.create"])
def test_transport_rejects_nonsemantic_final_feature_ids(identifier):
    payload = draft()
    payload["features"][0]["featureId"] = identifier
    with pytest.raises(ValueError, match="featureId"):
        compile_document(payload)


def test_generation_context_preserves_source_identity_without_copying_temporary_final_id():
    items = [{"featureId": "intent.a123.create", "name": "Original"}, {"featureId": "items.create", "name": "Canonical"}]
    context = generation_input(items)
    assert context[0]["sourceFeatureId"] == "intent.a123.create" and "featureId" not in context[0]
    assert context[1] == items[1] and items[0]["featureId"] == "intent.a123.create"


def test_duplicate_keys_are_rejected_with_exact_escaped_pointer():
    value, issues = strict_payload('{"features":[{"a/b~c":1,"a/b~c":1}]}')
    assert value is None and issues == ["FEATURE_SPEC_JSON: duplicate key: /features/0/a~1b~0c"]


def test_composite_unique_transport_uses_named_objects_and_preserves_legacy_input():
    payload = draft()
    payload["tables"][0]["uniqueConstraints"] = [{"columns": ["id", "name"]}]
    expected = [{"table": "items", "columns": ["id", "name"]}]
    assert compile_document(payload)[0]["dbContract"]["uniqueConstraints"] == expected
    payload["tables"][0]["uniqueConstraints"] = [["id", "name"]]
    assert compile_document(payload)[0]["dbContract"]["uniqueConstraints"] == expected
    schema = DraftDocument.model_json_schema()
    assert schema['$defs']['TableDraft']['properties']['uniqueConstraints']['items'] == {'$ref': '#/$defs/UniqueDraft'}


def test_composite_unique_on_undeclared_column_is_dropped_not_propagated():
    # Live run (club-dues): file_cleanup_jobs UNIQUE(stored_file_id, deletion_version)
    # named a sibling table's column, so DBA dropped it and Phase3 failed forever.
    payload = draft()
    payload["tables"][0]["uniqueConstraints"] = [{"columns": ["id", "deletion_version"]}, {"columns": ["id", "name"]}]
    assert compile_document(payload)[0]["dbContract"]["uniqueConstraints"] == [{"table": "items", "columns": ["id", "name"]}]


def test_state_transition_effect_takes_from_and_to_from_declared_transitions():
    from phase2.workflow_contract import normalize_state_transition_effects
    feature = {"dbContract": {"primaryTable": "requests"},
               "stateTransitions": [{"from": "requested", "to": "cancelled"}, {"from": "scheduled", "to": "cancelled"},
                                    {"from": "requested", "to": "scheduled"}],
               "apiContract": [{"effects": [{"kind": "state_transition", "table": "requests", "column": "status", "to": "cancelled"}]},
                               {"effects": [{"kind": "state_transition", "table": "requests", "column": "status", "from": "requested"}]}]}
    result, changes = normalize_state_transition_effects([feature])
    cancel, schedule = (op["effects"] for op in result[0]["apiContract"])
    # Many sources become one typed effect per source.
    assert [(item["from"], item["to"]) for item in cancel] == [("requested", "cancelled"), ("scheduled", "cancelled")]
    # requested -> {cancelled, scheduled} is ambiguous, so it stays for the model to repair.
    assert schedule == feature["apiContract"][1]["effects"] and len(changes) == 1


def test_invalid_json_retry_does_not_replay_unusable_assistant_output():
    broken = '{"features":[INVALID'
    create = AsyncMock(side_effect=[response(broken), response(draft())])
    provider = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    result, _, issues = asyncio.run(generate_document(provider, "test", None, "draft",
        [{"sourceFeatureId": "source.items"}], draft=True))
    assert result and not issues and create.await_count == 2
    assert all(message["content"] != broken for message in create.call_args.kwargs["messages"])


@pytest.mark.parametrize("kind", ["update", "revoke", "expire", "state_transition"])
def test_effect_columns_are_required_at_the_generation_boundary(kind):
    payload = draft()
    payload["features"][0]["apiContract"][0]["effects"] = [{"kind": kind, "table": "items"}]
    issues = document_issues(compile_document(payload), [{"sourceFeatureId": "source.items"}], preflight=True)
    assert any("API_EFFECT_COLUMN_UNKNOWN" in issue for issue in issues)


def test_effect_semantics_and_other_contract_errors_are_collected_in_one_preflight():
    payload = draft()
    payload["features"][0]["apiContract"][0]["effects"] = [{"kind": "state_transition", "table": "items", "column": "name"}]
    payload["features"][0]["ownership"] = {"scope": "SHARED", "ownerEntity": "unknown", "access": "Restricted to members"}
    issues = document_issues(compile_document(payload), [{"sourceFeatureId": "source.items"}], preflight=True)
    assert any("API_STATE_TRANSITION_INCOMPLETE" in issue for issue in issues)
    assert any("FEATURE_OWNER_TABLE_UNKNOWN" in issue for issue in issues)
    repaired = {"features": [{"featureId": "items.create", "ownership": {"scope": "PUBLIC", "ownerEntity": "", "access": "Public creation"},
                             "apiContract": [{"method": "POST", "path": "/api/v1/items", "effects": [{"kind": "insert", "table": "items"}]}]}]}
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=AsyncMock(side_effect=[response(payload), response(repaired)]))))
    features, diagnostics, failures = asyncio.run(generate_document(client, "test", None, "JSON", [{"sourceFeatureId": "source.items"}], draft=True))
    assert features and not failures and client.chat.completions.create.await_count == 2
    assert any("API_STATE_TRANSITION_INCOMPLETE" in issue for issue in diagnostics[0]["issues"])
    assert any("FEATURE_OWNER_TABLE_UNKNOWN" in issue for issue in diagnostics[0]["issues"])


def test_preflight_catches_ownership_action_output_and_transaction_before_agents_run():
    payload = draft()
    item = payload["features"][0]
    item["ownership"] = {"scope": "USER", "ownerEntity": "item", "ownerKey": "id", "access": "Verify current owner"}
    item["actions"].append("update")
    item["outputRequirements"] = [{"field": "missingResult", "description": "Required result"}]
    item["apiContract"][0]["transactionRules"] = ""
    issues = document_issues(compile_document(payload), [{"sourceFeatureId": "source.items"}], preflight=True)
    assert any("FEATURE_OWNER_TABLE_UNKNOWN" in issue for issue in issues)
    assert any("FEATURE_ACTION_CONTRACT_MISSING" in issue for issue in issues)
    assert any("API_REQUIRED_OUTPUT_MISSING" in issue for issue in issues)
    assert any("FEATURE_TRANSACTION_REQUIRED" in issue for issue in issues)


def test_primary_key_contract_has_required_semantics_without_fabricating_defaults():
    field = compile_document(draft())[0]["dbContract"]["fields"][0]
    assert field["primaryKey"] and field["required"] and field["default"] is None


def test_missing_action_repair_appends_only_required_method_and_keeps_existing_operations():
    payload = draft()
    scope = repair_scope(payload, ["FEATURE_ACTION_CONTRACT_MISSING: /features/0/actions: update requires PATCH"], [])
    replacement = copy.deepcopy(payload["features"][0])
    replacement["apiContract"].append({"method": "PATCH", "path": "/api/v1/items/{id}", "action": "update", "successStatus": 200, "requestType": "PatchItem"})
    patch = {"version": "contract-draft-v1", "features": [replacement],
             "schemas": [{"name": "PatchItem", "nodes": [{"id": "root", "kind": "object"}]}]}
    merged = merge_patch(payload, patch, scope)
    assert merged["features"][0]["apiContract"][0] == payload["features"][0]["apiContract"][0]
    assert merged["schemas"][-1]["name"] == "PatchItem"
    replacement["apiContract"][1]["method"] = "DELETE"
    with pytest.raises(ValueError, match="unrelated endpoint"):
        merge_patch(payload, patch, scope)


def test_sparse_feature_patch_preserves_omitted_source_operations_and_nested_fields():
    payload = draft()
    payload["features"][0]["ownership"] = {"scope": "USER", "ownerEntity": "users", "ownerKey": "id", "access": "owner"}
    scope = repair_scope(payload, ["OWNER: /features/0/ownership: invalid"], [])
    patch = {"version": "contract-draft-v1", "features": [{"featureId": "items.create", "ownership": {"ownerKey": "user_id"}}]}
    merged = merge_patch(payload, patch, scope)
    expected = copy.deepcopy(payload)
    expected["features"][0]["ownership"]["ownerKey"] = "user_id"
    assert merged == expected and payload["features"][0]["ownership"]["ownerKey"] == "id"


@pytest.mark.parametrize("source", [None, "other.source"])
def test_sparse_patch_cannot_explicitly_clear_or_change_source_identity(source):
    payload = draft()
    scope = repair_scope(payload, ["OWNER: /features/0/ownership: invalid"], [])
    patch = {"version": "contract-draft-v1", "features": [{"featureId": "items.create", "sourceFeatureId": source}]}
    with pytest.raises(ValueError, match="source identity changed"):
        merge_patch(payload, patch, scope)


def test_sparse_patch_inherits_envelope_version_but_cannot_change_it():
    payload = draft()
    scope = repair_scope(payload, ["SHAPE: /features/0/reason: missing"], [])
    patch = {"features": [{"featureId": "items.create", "reason": "Repair only this field"}]}
    merged = merge_patch(payload, patch, scope)
    assert merged["version"] == payload["version"] and merged["features"][0]["reason"] == "Repair only this field"
    with pytest.raises(ValueError, match="invalid draft patch envelope"):
        merge_patch(payload, {**patch, "version": "unexpected-v2"}, scope)
    with pytest.raises(ValueError, match="invalid draft patch envelope"):
        merge_patch(payload, {**patch, "unrequested": []}, scope)


def test_misnested_schema_patch_is_rejected_with_exact_feature_identity_pointer():
    payload = draft()
    scope = repair_scope(payload, ["BAD: /features/0/reason: missing"], [])
    patch = {"features": [{"schemas": [{"name": "Item", "nodes": []}]}]}
    with pytest.raises(ValueError, match=r"/features/0/featureId: invalid or duplicate patch key"):
        merge_patch(payload, patch, scope)


@pytest.mark.parametrize("broken", ["missing_tables", "nested_document", "duplicate_identity"])
def test_unaddressable_initial_draft_uses_one_clean_retry_not_an_impossible_sparse_patch(broken):
    initial = draft()
    if broken == "missing_tables":
        initial.pop("tables")
    elif broken == "nested_document":
        initial = {"version": "contract-draft-v1", "features": [copy.deepcopy(initial)]}
    else:
        initial["features"].append(copy.deepcopy(initial["features"][0]))
    create = AsyncMock(side_effect=[response(initial), response(draft())])
    provider = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    result, diagnostics, issues = asyncio.run(generate_document(
        provider, "test", None, "draft", [{"sourceFeatureId": "source.items"}], draft=True,
    ))
    assert result and not issues and create.await_count == 2
    assert diagnostics[0]["retryKind"] == "UNUSABLE_DRAFT_RETRY"
    retry_messages = create.await_args_list[1].kwargs["messages"]
    assert not any(message["role"] == "assistant" for message in retry_messages)
    assert "one complete shallow draft" in retry_messages[-1]["content"]


def test_authenticated_stored_feature_repairs_missing_ownership_before_legacy_defaults():
    initial = draft()
    initial["features"][0].pop("ownership")
    initial["features"][0]["apiContract"][0]["authRequired"] = True
    patch = {"features": [{"featureId": "items.create", "ownership": {
        "scope": "PUBLIC", "access": "Authenticated operators manage the shared catalog.",
    }}]}
    create = AsyncMock(side_effect=[response(initial), response(patch)])
    provider = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    result, diagnostics, issues = asyncio.run(generate_document(
        provider, "test", None, "draft", [{"sourceFeatureId": "source.items"}], draft=True,
    ))
    assert diagnostics[0]["issues"] == [
        "FEATURE_DRAFT_SHAPE: /features/0/ownership: Field required",
    ]
    assert result and not issues and create.await_count == 2
    assert result[0]["ownership"]["scope"] == "PUBLIC"
    assert result[0]["ownership"]["access"] == patch["features"][0]["ownership"]["access"]


def test_ownership_repair_hints_use_declared_scoped_foreign_keys_and_tolerate_invalid_shape():
    payload = draft()
    payload["features"][0].pop("ownership")
    payload["tables"][0]["foreignKeys"] = [
        {"column": "owner_id", "references": {"table": "accounts", "column": "id"}},
        {"column": "invalid", "references": None}, None,
    ]
    original = copy.deepcopy(payload)
    candidates = _ownership_repair_candidates(payload, {"featureIds": ["items.create"]})
    assert candidates == [{"featureId": "items.create", "primaryTable": "items", "existingOwnership": None,
                           "foreignKeys": [{"ownerKey": "owner_id", "ownerEntity": "accounts"}]}]
    assert payload == original
    assert _ownership_repair_candidates(payload, {"featureIds": []}) == []
    payload["tables"][0]["foreignKeys"] = 3
    assert _ownership_repair_candidates(payload, {"featureIds": ["items.create"]})[0]["foreignKeys"] == []


def test_missing_metadata_does_not_mask_missing_dto_in_the_single_repair():
    payload = draft()
    payload["features"][0].pop("reason")
    payload["features"][0]["apiContract"][0]["responseType"] = "MissingDto"
    with pytest.raises(ValueError) as error:
        compile_document(payload)
    assert "/features/0/reason" in str(error.value)
    assert "/features/0/apiContract/0/responseType: MissingDto" in str(error.value)
    schema = copy.deepcopy(payload["schemas"][1])
    schema["name"] = "MissingDto"
    patch = {"features": [{"featureId": "items.create", "reason": "Required journey"}], "schemas": [schema]}
    create = AsyncMock(side_effect=[response(payload), response(patch)])
    provider = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    result, _, issues = asyncio.run(generate_document(provider, "test", None, "draft", [{"sourceFeatureId": "source.items"}], draft=True))
    assert result and not issues and create.await_count == 2


def test_sparse_operation_patch_keeps_siblings_order_and_unmodified_fields():
    payload = draft()
    payload["features"][0]["apiContract"].insert(0, {"method": "GET", "path": "/api/v1/items", "action": "read"})
    scope = repair_scope(payload, ["BAD_MAPPING: /features/0/apiContract/1/fieldMappings"], [])
    patch = {"features": [{"featureId": "items.create", "apiContract": [{"method": "POST", "path": "/api/v1/items", "transactionRules": "Changed atomic write"}]}]}
    merged = merge_patch(payload, patch, scope)
    expected = copy.deepcopy(payload)
    expected["features"][0]["apiContract"][1]["transactionRules"] = "Changed atomic write"
    assert merged == expected
    patch["features"][0]["apiContract"][0]["method"] = "GET"
    with pytest.raises(ValueError, match="unrelated endpoint"):
        merge_patch(payload, patch, scope)


def test_single_operation_sparse_fields_preserve_dto_and_parameter_contracts():
    payload = draft()
    scope = repair_scope(payload, ["BAD: /features/0/apiContract/0"], [])
    patch = {"features": [{"featureId": "items.create", "apiContract": [{"method": "POST", "path": "/api/v1/items", "transactionRules": "Patched"}]}]}
    result = merge_patch(payload, patch, scope)
    assert result["features"][0]["apiContract"][0] == {**payload["features"][0]["apiContract"][0], "transactionRules": "Patched"}


@pytest.mark.parametrize("collection", ["schemas", "tables"])
def test_resource_only_draft_patch_does_not_need_unchanged_features_array(collection):
    initial = draft()
    fixed = copy.deepcopy(initial[collection][0])
    if collection == "schemas":
        initial[collection][0]["nodes"][0]["kind"] = "string"
    else:
        initial[collection][0]["fields"][0]["type"] = ""
    create = AsyncMock(side_effect=[response(initial), response({collection: [fixed]})])
    provider = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    result, diagnostics, issues = asyncio.run(generate_document(
        provider, "test", None, "draft", [{"sourceFeatureId": "source.items"}], draft=True,
    ))
    assert diagnostics[0]["issues"] and result and not issues
    assert result[0]["featureId"] == "items.create" and create.await_count == 2
    assert result[0]["apiContract"][0]["path"] == "/api/v1/items"


def test_sparse_payload_mode_is_explicit_and_preserves_array_and_duplicate_key_checks():
    assert strict_payload('{"schemas": []}')[0] is None
    assert strict_payload('{"schemas": []}', allow_patch=True) == ({"schemas": []}, [])
    assert strict_payload('{"features": null, "schemas": []}', allow_patch=True)[0] is None
    payload, issues = strict_payload('{"schemas": [], "schemas": []}', allow_patch=True)
    assert payload is None and issues == ["FEATURE_SPEC_JSON: duplicate key: /schemas"]


def test_missing_method_does_not_hide_existing_response_repair_in_same_feature():
    initial = draft()
    initial["features"][0]["actions"].append("read")
    initial["features"][0]["outputRequirements"] = [{"field": "status", "description": "Creation outcome"}]
    dto = copy.deepcopy(initial["schemas"][1])
    dto["name"] = "ItemWithStatus"
    dto["nodes"].append({"id": "status", "parent": "root", "name": "status", "kind": "string", "required": True})
    patch = {"features": [{"featureId": "items.create", "apiContract": [
        {"method": "POST", "path": "/api/v1/items", "responseType": "ItemWithStatus"},
        {"method": "GET", "path": "/api/v1/items", "action": "read", "authRequired": False,
         "successStatus": 200, "responseType": "ItemWithStatus"},
    ]}], "schemas": [dto]}
    create = AsyncMock(side_effect=[response(initial), response(patch)])
    provider = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    result, diagnostics, issues = asyncio.run(generate_document(
        provider, "test", None, "draft", [{"sourceFeatureId": "source.items"}], draft=True,
    ))
    assert any("ACTION_CONTRACT_MISSING" in issue for issue in diagnostics[0]["issues"])
    assert any("OUTPUT_MISSING" in issue for issue in diagnostics[0]["issues"])
    assert result and not issues and create.await_count == 2
    operations = result[0]["apiContract"]
    assert [operation["method"] for operation in operations] == ["POST", "GET"]
    assert operations[0]["requestSchema"]["properties"]["name"]["type"] == "string"


def empty_api_patch():
    payload = draft()
    payload["features"][0].update(apiContract=[], actions=["validate"])
    request, response_dto = copy.deepcopy(payload["schemas"])
    request["name"], response_dto["name"] = "ValidationRequest", "ValidationResponse"
    operation = {"method": "POST", "path": "/api/v1/items/validate", "action": "validate", "authRequired": False,
                 "requestType": "ValidationRequest", "responseType": "ValidationResponse", "successStatus": 200,
                 "transactionRules": "Read only validation; no persistence effects."}
    patch = {"features": [{"featureId": "items.create", "apiContract": [operation]}], "schemas": [request, response_dto]}
    return payload, patch


def test_empty_api_contract_can_declare_its_own_new_request_and_response_dtos():
    initial, patch = empty_api_patch()
    create = AsyncMock(side_effect=[response(initial), response(patch)])
    provider = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    result, _, issues = asyncio.run(generate_document(
        provider, "test", None, "draft", [{"sourceFeatureId": "source.items"}], draft=True,
    ))
    assert result and not issues and create.await_count == 2
    assert result[0]["apiContract"][0]["requestSchema"]["properties"]["name"]["type"] == "string"


@pytest.mark.parametrize("extra", ["schema", "table", "detached"])
def test_empty_api_contract_does_not_grant_unrelated_or_detached_resource_writes(extra):
    initial, patch = empty_api_patch()
    scope = repair_scope(initial, ["FEATURE_SPEC_CONTRACT: /features/0: items.create: apiContract required"], [])
    if extra == "schema":
        patch["schemas"].append({"name": "Unrelated", "nodes": [{"id": "root", "kind": "string"}]})
    elif extra == "table":
        patch["tables"] = [{"name": "unrelated", "fields": [{"name": "id", "type": "UUID"}]}]
    else:
        patch["features"] = []
    with pytest.raises(ValueError, match="unrequested"):
        merge_patch(initial, patch, scope)


def duplicate_core_draft():
    initial = draft()
    child = copy.deepcopy(initial["features"][0])
    child.update(featureId="items.list", name="Item list", actions=["read"], requirements=["Read the saved items."], apiContract=[{
        "method": "GET", "path": "/api/v1/items", "action": "read", "authRequired": False,
        "responseType": "Item", "successStatus": 200, "transactionRules": "Read only.",
    }])
    initial["features"].append(child)
    patch = {"features": [{"featureId": "items.list", "source": "supporting", "priority": "P1",
                           "parentFeatureId": "items.create", "reason": "Read the data saved by the core journey."}]}
    return initial, patch


def test_duplicate_core_is_not_missing_and_can_be_reclassified_without_new_features():
    initial, patch = duplicate_core_draft()
    cores = [{"sourceFeatureId": "source.items"}]
    scope = repair_scope(initial, document_issues(compile_document(initial), cores), cores)
    assert scope["missingSourceFeatureIds"] == []
    assert scope["duplicateCoreSources"] == {"source.items": ["items.create", "items.list"]}
    assert scope["featureIds"] == ["items.create", "items.list"]
    merged = merge_patch(initial, patch, scope)
    assert merged["features"][0] == initial["features"][0]
    assert merged["features"][1]["sourceFeatureId"] == "source.items"
    assert merged["tables"] == initial["tables"] and merged["schemas"] == initial["schemas"]
    assert not document_issues(compile_document(merged), cores, preflight=True)
    third = copy.deepcopy(initial["features"][0])
    third["featureId"] = "items.replacement"
    with pytest.raises(ValueError, match="unrequested features"):
        merge_patch(initial, {"features": [third]}, scope)


def test_duplicate_core_repair_stays_supporting_after_agent_normalization():
    initial, patch = duplicate_core_draft()
    prd_core = compile_document(draft())[0]
    create = AsyncMock(side_effect=[response(initial), response(patch)])
    provider = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    state = PipelineState(prd_document=json.dumps({"coreFeatures": [prd_core]}))
    result = asyncio.run(FeatureSpecAgent(provider, "test").execute(state))
    assert not result.generation_blockers and create.await_count == 2
    by_id = {feature["featureId"]: feature for feature in result.feature_registry}
    assert set(by_id) == {"items.create", "items.list"}
    assert by_id["items.create"]["source"] == "prd_core"
    assert by_id["items.list"]["source"] == "supporting"
    assert by_id["items.list"]["sourceFeatureId"] == "source.items"
    assert by_id["items.list"]["priority"] != "P0"
    assert len(json.loads(result.prd_document)["coreFeatures"]) == 1


def test_draft_repair_includes_referenced_dto_and_does_not_protect_fully_affected_features():
    payload = draft()
    payload["schemas"].append({"name": "Name", "nodes": [{"id": "root", "kind": "string"}]})
    payload["schemas"][1]["nodes"][2] = {"id": "name", "parent": "root", "name": "name", "kind": "ref", "ref": "Name"}
    other = copy.deepcopy(payload["features"][0])
    other.update(featureId="items.other", sourceFeatureId="source.other")
    payload["features"].append(other)
    issues = ["FEATURE_OWNERSHIP_REQUIRED: /features/0/ownership: explicit policy required",
              "API_REQUIRED_OUTPUT_MISSING: /features/1: items.other: name"]
    scope = repair_scope(payload, issues, [])
    assert "Name" in scope["schemaNames"]
    replacement = {"name": "Name", "nodes": [{"id": "root", "kind": "string", "minLength": 1}]}
    updated = merge_patch(payload, {"schemas": [replacement]}, scope)
    assert updated["schemas"][-1] == replacement
    assert updated["features"] == payload["features"]


def test_supporting_type_requires_a_real_parent_and_reason_before_compilation():
    payload = draft()
    payload["features"][0].update(source="supporting", reason="", parentFeatureId=None)
    with pytest.raises(ValueError) as error:
        compile_document(payload)
    assert "/features/0/parentFeatureId" in str(error.value)
    assert "/features/0/reason" in str(error.value)
    assert "/features/0/supporting/" not in str(error.value)


def test_scope_rejected_patch_keeps_draft_baseline_and_retries_patch():
    payload = draft()
    payload["features"][0]["apiContract"][0]["effects"] = [{"kind": "state_transition", "table": "items", "column": "name"}]
    payload["features"][0]["ownership"] = {"scope": "SHARED", "ownerEntity": "unknown", "access": "Restricted to members"}
    fixed = {"method": "POST", "path": "/api/v1/items", "effects": [{"kind": "insert", "table": "items"}]}
    # Appending a route nobody asked for violates the repair scope.
    rogue = {"features": [{"featureId": "items.create", "apiContract": [fixed, {"method": "GET", "path": "/api/v1/unrelated"}]}]}
    repaired = {"features": [{"featureId": "items.create", "ownership": {"scope": "PUBLIC", "ownerEntity": "", "access": "Public creation"},
                              "apiContract": [fixed]}]}
    create = AsyncMock(side_effect=[response(payload), response(rogue), response(repaired)])
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    features, diagnostics, failures = asyncio.run(generate_document(
        client, "test", None, "JSON", [{"sourceFeatureId": "source.items"}], draft=True, max_attempts=3))
    assert features and not failures and create.await_count == 3
    assert diagnostics[1]["retryKind"] == "SCOPE_REJECTED_PATCH_RETRY"
    # The retry stays a patch over the original draft, not a full regeneration.
    last_prompt = create.call_args.kwargs["messages"][-1]["content"]
    assert "FEATURE_OWNER_TABLE_UNKNOWN" in last_prompt and "unusable" not in last_prompt


def test_unparseable_patch_keeps_baseline_issues_and_retries_patch():
    payload = draft()
    payload["features"][0]["apiContract"][0]["effects"] = [{"kind": "state_transition", "table": "items", "column": "name"}]
    payload["features"][0]["ownership"] = {"scope": "SHARED", "ownerEntity": "unknown", "access": "Restricted to members"}
    fixed = {"method": "POST", "path": "/api/v1/items", "effects": [{"kind": "insert", "table": "items"}]}
    repaired = {"features": [{"featureId": "items.create", "ownership": {"scope": "PUBLIC", "ownerEntity": "", "access": "Public creation"},
                              "apiContract": [fixed]}]}
    broken = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='{"features":[{}]}\n}]', refusal=None),
                                                      finish_reason="stop")], model="test", usage=None)
    create = AsyncMock(side_effect=[response(payload), broken, response(repaired)])
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    features, diagnostics, failures = asyncio.run(generate_document(
        client, "test", None, "JSON", [{"sourceFeatureId": "source.items"}], draft=True, max_attempts=3))
    assert features and not failures and create.await_count == 3
    assert any(item.get("retryKind") == "UNPARSEABLE_PATCH_RETRY" for item in diagnostics)
    last_prompt = create.call_args.kwargs["messages"][-1]["content"]
    assert "FEATURE_OWNER_TABLE_UNKNOWN" in last_prompt and "unusable" not in last_prompt
