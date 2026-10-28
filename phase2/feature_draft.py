"""Shallow generation transport compiled into the existing feature contracts."""

import logging
from copy import deepcopy
from typing import Annotated, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)
from pydantic.json_schema import SkipJsonSchema

from phase2.contract_guard import semantic_identifier
from phase2.workflow_contract import (
    IntervalPolicy,
    ReadDependency,
    SessionPolicy,
    StatePolicy,
    bind_workflow_errors,
)

Scalar = str | int | float | bool | None


logger = logging.getLogger(__name__)


class DraftModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, populate_by_name=True)


class SchemaNode(DraftModel):
    id: str = Field(min_length=1)
    parent: str | None = None
    name: str | None = None
    kind: Literal["object", "array", "string", "integer", "number", "boolean", "null", "ref", "oneOf", "anyOf", "allOf"]
    required: bool = False
    nullable: bool = False
    format: str | None = None
    description: str | None = None
    enum: list[Scalar] | None = None
    ref: str | None = None
    default: Scalar = None
    minLength: int | None = Field(default=None, ge=0)
    maxLength: int | None = Field(default=None, ge=0)
    minimum: int | float | None = None
    maximum: int | float | None = None
    minItems: int | None = Field(default=None, ge=0)
    maxItems: int | None = Field(default=None, ge=0)
    pattern: str | None = None
    additionalProperties: bool | None = None
    minProperties: int | None = Field(default=None, ge=0)


class SchemaDraft(DraftModel):
    name: str = Field(min_length=1, pattern=r"^[A-Za-z][A-Za-z0-9_]*$")
    nodes: list[SchemaNode] = Field(min_length=1, max_length=512)


class MappingDraft(DraftModel):
    field: str = Field(min_length=1)
    table: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    column: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    source: Literal["body", "path", "query", "header", "cookie", "response", "auth", "server", "derived", "request"]
    transform: str | None = None
    pointer: str | None = None


class ParameterDraft(DraftModel):
    name: str = Field(min_length=1)
    location: Literal["path", "query", "header", "cookie"] = Field(alias="in")
    typeRef: str
    required: bool = False
    description: str = ""


class EffectDraft(DraftModel):
    table: str
    column: str | None = None
    kind: Literal["insert", "create", "update", "set", "delete", "increment", "decrement", "revoke", "expire", "state_transition"]
    from_value: Scalar = Field(default=None, alias="from")
    to: Scalar = None

class SessionDraft(SessionPolicy):
    pass


class OperationDraft(DraftModel):
    method: Literal["GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"]
    path: str = Field(pattern=r"^/api/v1/")
    action: str = Field(min_length=1)
    authRequired: bool = True
    requestType: str | None = None
    responseType: str | None = None
    successStatus: int = Field(ge=100, le=599)
    contentType: str = "application/json"
    parameters: list[ParameterDraft] = Field(default_factory=list)
    fieldMappings: list[MappingDraft] = Field(default_factory=list)
    effects: list[EffectDraft] = Field(default_factory=list)
    transactionRules: str = ""
    allowedRoles: list[str] = Field(default_factory=list)
    responseHeaders: dict[str, str] = Field(default_factory=dict)
    errorCodes: str | None = None
    accessScope: Literal["PUBLIC", "AUTHENTICATED", "OWNER", "PARTICIPANT"] | None = None
    primaryTable: str | None = None
    requiredInputs: list[str] = Field(default_factory=list, description="Concrete inputs promised for THIS operation. Body fields must exist in requestType; path/query/header/cookie fields must be declared parameters. Optional location prefix: body.images or path.resourceId.")
    requiredOutputs: list[str] = Field(default_factory=list, description="Every promised screen value as an actual response field path, including names, timestamps, counterpart labels and ownership indicators.")
    requiredEffectTables: list[str] = Field(default_factory=list, description="Every table this operation promises to mutate, including replaced file metadata.")


class DbFieldDraft(DraftModel):
    name: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    type: str = Field(min_length=1)
    required: bool = False
    primaryKey: bool = False
    generated: bool = False
    unique: bool = False
    # header covers values such as an Idempotency-Key stored for replay checks;
    # without it one such column made the whole first draft unusable.
    source: Literal["request", "body", "path", "query", "header", "auth", "server", "derived"] = "request"
    default: Scalar = None
    minimum: int | float | None = None
    maximum: int | float | None = None
    enum: list[Scalar] | None = None

    @model_validator(mode="after")
    def primary_key_is_required(self):
        if self.primaryKey:
            self.required = True
        return self


class ReferenceDraft(DraftModel):
    table: str
    column: str = ""
    columns: list[str] = Field(default_factory=list)


class ForeignKeyDraft(DraftModel):
    column: str = ""
    columns: list[str] = Field(default_factory=list)
    references: ReferenceDraft
    nullable: bool = False
    onDelete: Literal["NO ACTION", "RESTRICT", "CASCADE", "SET NULL", "SET DEFAULT"] = "NO ACTION"


class UniqueDraft(DraftModel):
    columns: list[str] = Field(min_length=1)


class TableDraft(DraftModel):
    name: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    fields: list[DbFieldDraft] = Field(min_length=1)
    foreignKeys: list[ForeignKeyDraft] = Field(default_factory=list)
    uniqueConstraints: list[UniqueDraft | SkipJsonSchema[list[str]]] = Field(default_factory=list)


class ScopeDraft(DraftModel):
    ui: bool
    api: bool
    db: bool


class OutputDraft(DraftModel):
    field: str
    description: str


class TransitionDraft(DraftModel):
    entity: str | None = None
    from_value: str | bool = Field(alias="from")
    to: str | bool


class OwnershipDraft(DraftModel):
    scope: Literal["USER", "SHARED", "PUBLIC", "SYSTEM"]
    ownerEntity: str = ""
    ownerKey: str = ""
    access: str = Field(min_length=1)


class FeatureDraft(DraftModel):
    featureId: str = Field(min_length=1, pattern=r"^[a-z][a-z0-9_]*(\.[a-z][a-z0-9_]*)+$",
                           description="Semantic domain.action ID, never intent.*, supporting.*, feature_001 or hash IDs. sourceFeatureId alone preserves the source ID.")
    sourceFeatureId: str | None
    source: Literal["prd_core", "supporting"]
    parentFeatureId: str | None
    name: str = Field(min_length=1)
    description: str = Field(min_length=1)
    requirements: list[str] = Field(min_length=1)
    reason: str
    actions: list[str] = Field(min_length=1)
    priority: Literal["P0", "P1", "P2", "P3"]
    priorityScore: int = Field(default=0, ge=0, le=13)
    releaseGate: bool = False
    applicability: ScopeDraft
    nonApplicabilityReasons: dict[str, str] = Field(default_factory=dict)
    dependencies: list[str] = Field(default_factory=list)
    roles: list[str] = Field(default_factory=list)
    ownership: OwnershipDraft = Field(description="Explicit access policy for every feature. Use PUBLIC for public reads, SYSTEM for privileged operations, USER/SHARED with real owner references. Never omit authenticated resource access.")
    dbTables: list[str]
    primaryTable: str | None
    apiContract: list[OperationDraft] = Field(description="Always present. api=true requires real operations; api=false uses an explicit empty array. Never create a separate middleware-only authentication feature.")
    sessionPolicy: SessionDraft | None = None
    states: list[str] = Field(default_factory=list)
    stateTransitions: list[TransitionDraft] = Field(default_factory=list)
    transactionRules: list[str] = Field(default_factory=list)
    outputRequirements: list[OutputDraft] = Field(default_factory=list)
    readDependencies: list[ReadDependency] = Field(default_factory=list)
    statePolicies: list[StatePolicy] = Field(default_factory=list)
    intervalPolicies: list[IntervalPolicy] = Field(default_factory=list)

    @field_validator("featureId")
    @classmethod
    def validate_feature_id(cls, value):
        if not semantic_identifier(value):
            raise ValueError("semantic domain.action featureId required; preserve temporary IDs only in sourceFeatureId")
        return value


class CoreFeatureDraft(FeatureDraft):
    source: Literal["prd_core"]
    sourceFeatureId: str = Field(min_length=1)
    parentFeatureId: None


class SupportingFeatureDraft(FeatureDraft):
    source: Literal["supporting"]
    parentFeatureId: str = Field(min_length=1, description="Existing core semantic featureId, never null")
    reason: str = Field(min_length=1)


class DraftDocument(DraftModel):
    version: Literal["contract-draft-v1"]
    features: list[Annotated[CoreFeatureDraft | SupportingFeatureDraft, Field(discriminator="source")]] = Field(max_length=12)
    tables: list[TableDraft]
    schemas: list[SchemaDraft] = Field(description="Each schema name is defined exactly once, including primitive identifier schemas. Reuse names through typeRef/ref rather than repeating their definitions.")


def _unique(items, key, pointer):
    result = {}
    for index, item in enumerate(items):
        name = getattr(item, key)
        if name in result:
            raise ValueError(f"FEATURE_DRAFT_DUPLICATE: {pointer}/{index}: {name}")
        result[name] = item
    return result


def compile_schemas(definitions):
    """Resolve flat parent links; never repair syntax or guess missing nodes."""
    drafts = _unique(definitions, "name", "/schemas")
    compiled, visiting = {}, set()

    def definition(name, depth=0):
        if name in compiled:
            return deepcopy(compiled[name])
        if name not in drafts:
            raise ValueError(f"FEATURE_DRAFT_SCHEMA_REFERENCE: {name}: missing schema")
        pointer = f"/schemas/{list(drafts).index(name)}"
        if name in visiting or depth > 32:
            raise ValueError(f"FEATURE_DRAFT_SCHEMA_REFERENCE: {pointer}: cyclic or too deep")
        visiting.add(name)
        nodes = _unique(drafts[name].nodes, "id", f"/schemas/{list(drafts).index(name)}/nodes")
        roots = [node for node in nodes.values() if node.parent is None]
        if len(roots) != 1:
            raise ValueError(f"FEATURE_DRAFT_SCHEMA_ROOT: {pointer}: exactly one root required")
        children = {key: [] for key in nodes}
        for node in nodes.values():
            if node.parent is not None:
                if node.parent not in nodes:
                    raise ValueError(f"FEATURE_DRAFT_SCHEMA_PARENT: {pointer}/nodes: {node.id}")
                children[node.parent].append(node)
        reached = set()

        def build(node, level):
            if node.id in reached or level > 32:
                raise ValueError(f"FEATURE_DRAFT_SCHEMA_CYCLE: {pointer}/nodes: {node.id}")
            reached.add(node.id)
            descendants = children[node.id]
            for lower, upper in (("minLength", "maxLength"), ("minItems", "maxItems"), ("minimum", "maximum")):
                low, high = getattr(node, lower), getattr(node, upper)
                if low is not None and high is not None and low > high:
                    raise ValueError(f"FEATURE_DRAFT_SCHEMA_BOUNDS: {pointer}/nodes: {node.id}: {lower}>{upper}")
            if node.kind == "ref":
                if not node.ref or descendants or node.ref not in drafts:
                    raise ValueError(f"FEATURE_DRAFT_SCHEMA_REF: {pointer}/nodes: {node.id}: reference missing or invalid")
                out = definition(node.ref, level + 1)
            elif node.kind == "object":
                names = [child.name for child in descendants]
                if any(not value for value in names) or len(names) != len(set(names)):
                    raise ValueError(f"FEATURE_DRAFT_SCHEMA_PROPERTY: {pointer}/nodes: {node.id}")
                out = {"type": "object", "properties": {child.name: build(child, level + 1) for child in descendants}}
                required = [child.name for child in descendants if child.required]
                if required:
                    out["required"] = required
            elif node.kind == "array":
                if len(descendants) != 1 or descendants[0].name is not None:
                    raise ValueError(f"FEATURE_DRAFT_SCHEMA_ITEMS: {pointer}/nodes: {node.id}: one unnamed item node required")
                out = {"type": "array", "items": build(descendants[0], level + 1)}
            elif node.kind in {"oneOf", "allOf", "anyOf"}:
                if not descendants:
                    raise ValueError(f"FEATURE_DRAFT_SCHEMA_UNION: {pointer}/nodes: {node.id}")
                out = {node.kind: [build(child, level + 1) for child in descendants]}
            else:
                if descendants:
                    raise ValueError(f"FEATURE_DRAFT_SCHEMA_LEAF: {pointer}/nodes: {node.id}")
                out = {"type": node.kind}
            for key in ("format", "description", "enum", "default", "minLength", "maxLength", "minimum", "maximum", "minItems", "maxItems", "pattern", "additionalProperties", "minProperties"):
                value = getattr(node, key)
                if value is not None:
                    out[key] = deepcopy(value)
            if node.nullable:
                if isinstance(out.get("type"), str):
                    out["type"] = list(dict.fromkeys([out["type"], "null"]))
                else:
                    out = {"anyOf": [out, {"type": "null"}]}
            return out

        out = build(roots[0], depth)
        if reached != set(nodes):
            raise ValueError(f"FEATURE_DRAFT_SCHEMA_ORPHAN: {pointer}")
        visiting.remove(name)
        compiled[name] = out
        return deepcopy(out)

    for name in drafts:
        definition(name)
    return compiled


def _draft_reference_issues(payload):
    if not isinstance(payload, dict):
        return []
    schemas = {item["name"] for item in _array(payload.get("schemas"))
               if isinstance(item, dict) and isinstance(item.get("name"), str)}
    tables = {item["name"] for item in _array(payload.get("tables"))
              if isinstance(item, dict) and isinstance(item.get("name"), str)}
    issues = []
    for index, feature in enumerate(_array(payload.get("features"))):
        if not isinstance(feature, dict):
            continue
        for table in _array(feature.get("dbTables")):
            if isinstance(table, str) and table not in tables:
                issues.append(f"FEATURE_DRAFT_TABLE_UNKNOWN: /features/{index}/dbTables: {table}")
        for op_index, operation in enumerate(_array(feature.get("apiContract"))):
            if not isinstance(operation, dict):
                continue
            references = [(key, operation.get(key)) for key in ("requestType", "responseType")]
            references.extend((f"parameters/{param_index}/typeRef", parameter.get("typeRef"))
                              for param_index, parameter in enumerate(_array(operation.get("parameters"))) if isinstance(parameter, dict))
            for location, name in references:
                if isinstance(name, str) and name not in schemas:
                    issues.append(f"FEATURE_DRAFT_SCHEMA_REFERENCE: /features/{index}/apiContract/{op_index}/{location}: {name}")
    return issues


def compile_document(payload):
    """Every repeated physical field is projected from one typed table definition."""
    references = _draft_reference_issues(payload)
    try:
        draft = DraftDocument.model_validate(payload)
    except ValidationError as exc:
        def pointer(location):
            parts = list(location)
            if len(parts) > 2 and parts[0] == "features" and isinstance(parts[1], int) and parts[2] in {"prd_core", "supporting"}:
                parts.pop(2)
            return "/" + "/".join(map(str, parts))

        raise ValueError("; ".join([
            f"FEATURE_DRAFT_SHAPE: {pointer(error['loc'])}: {error['msg']}"
            for error in exc.errors(include_url=False, include_input=False)] + references)) from None
    if references:
        raise ValueError("; ".join(references))
    tables = _unique(draft.tables, "name", "/tables")
    schemas = compile_schemas(draft.schemas)
    _unique(draft.features, "featureId", "/features")
    for index, table in enumerate(draft.tables):
        _unique(table.fields, "name", f"/tables/{index}/fields")
        # A composite UNIQUE naming a column that this table does not define
        # (e.g. a sibling table's version column) can never be built. DBA then
        # drops the index and Phase3 repeats DB_UNIQUE_CONTRACT_MISSING until
        # the run fails, so keep only constraints the table can implement.
        declared = {field.name for field in table.fields}
        kept = []
        for unique in table.uniqueConstraints:
            columns = unique.columns if isinstance(unique, UniqueDraft) else unique
            if set(columns) <= declared:
                kept.append(unique)
            else:
                logger.warning("Feature draft UNIQUE 제거 — %s.%s: 테이블에 없는 컬럼 %s",
                               table.name, list(columns), sorted(set(columns) - declared))
        table.uniqueConstraints = kept
    features = []
    for index, feature in enumerate(draft.features):
        data = feature.model_dump(by_alias=True, exclude_none=True)
        data["workflowContractVersion"] = 1
        names = data.pop("dbTables")
        primary = data.pop("primaryTable", None)
        if len(names) != len(set(names)) or any(name not in tables for name in names):
            raise ValueError(f"FEATURE_DRAFT_TABLE_UNKNOWN: /features/{index}/dbTables: {','.join(names)}")
        # Explicit FK dependencies are part of the physical contract even when
        # the feature only lists its business table. No new table is inferred.
        for name in names:
            for fk in tables[name].foreignKeys:
                target = fk.references.table
                if target not in tables:
                    raise ValueError(f"FEATURE_DRAFT_TABLE_UNKNOWN: /features/{index}/dbTables: {target}")
                if target not in names:
                    names.append(target)
        db = {"tables": names, "primaryTable": primary, "fields": [], "foreignKeys": [], "uniqueConstraints": []}
        for name in names:
            table = tables[name]
            db["fields"].extend({"table": name, **field.model_dump()} for field in table.fields)
            db["foreignKeys"].extend({"table": name, **fk.model_dump(by_alias=True, exclude_none=True)} for fk in table.foreignKeys)
            db["uniqueConstraints"].extend({"table": name, "columns": columns.columns if isinstance(columns, UniqueDraft) else columns}
                                           for columns in table.uniqueConstraints)
        data["dbContract"] = db
        data["sourceFeatureId"] = feature.sourceFeatureId
        data["parentFeatureId"] = feature.parentFeatureId
        operations = []
        for op_index, operation in enumerate(feature.apiContract):
            op = operation.model_dump(by_alias=True, exclude_none=True)
            for ref_key, schema_key in (("requestType", "requestSchema"), ("responseType", "responseSchema")):
                ref = op.pop(ref_key, None)
                if ref is not None and ref not in schemas:
                    raise ValueError(f"FEATURE_DRAFT_SCHEMA_REFERENCE: /features/{index}/apiContract/{op_index}: {ref}")
                if ref is not None:
                    op[schema_key] = deepcopy(schemas[ref])
            op.setdefault("responseSchema", {})
            for parameter in op["parameters"]:
                ref = parameter.pop("typeRef")
                if ref not in schemas:
                    raise ValueError(f"FEATURE_DRAFT_SCHEMA_REFERENCE: /features/{index}/apiContract/{op_index}/parameters: {ref}")
                parameter["schema"] = deepcopy(schemas[ref])
            op["featureId"] = feature.featureId
            if feature.sessionPolicy:
                op["sessionPolicy"] = feature.sessionPolicy.model_dump()
            if op.get("authRequired") and not op.get("allowedRoles") and feature.roles:
                op["allowedRoles"] = list(feature.roles)
            operations.append(op)
        data["apiContract"] = operations
        features.append(data)
    for feature in features:
        for operation in feature["apiContract"]:
            bind_workflow_errors(operation, features)
    from phase2.contract_quality import normalize_generated_contracts

    normalize_generated_contracts(features)
    return features


def _array(value):
    return value if isinstance(value, list) else []


def repair_scope(payload, issues, cores):
    """Translate canonical feature errors back to explicitly named draft assets."""
    import re

    features = _array(payload.get("features"))
    indices = {int(value) for issue in issues for value in re.findall(r"/features/(\d+)", issue)}
    scope = {"featureIds": [], "tableNames": [], "schemaNames": [], "missingSourceFeatureIds": [],
             "duplicateCoreSources": {}, "emptyApiFeatureIds": [],
             "operationKeys": {}, "operationIndices": {}, "missingOperationMethods": {}, "missingOperationKeys": {}}
    for core in cores:
        source = core["sourceFeatureId"]
        if not any(f"CORE_COVERAGE: {source}:" in issue or "FEATURE_SPEC_EMPTY" in issue for issue in issues):
            continue
        matches = [index for index, item in enumerate(features) if isinstance(item, dict)
                   and item.get("sourceFeatureId") == source and item.get("source", "prd_core") == "prd_core"]
        if not matches:
            scope["missingSourceFeatureIds"].append(source)
        elif len(matches) > 1:
            indices.update(matches)
            scope["duplicateCoreSources"][source] = [features[index].get("featureId") for index in matches]
    for index in sorted(indices):
        if index >= len(features) or not isinstance(features[index], dict):
            continue
        feature = features[index]
        scope["featureIds"].append(feature.get("featureId"))
        scoped = " ".join(issue for issue in issues if re.search(rf"/features/{index}(?:/|:|$)", issue))
        if isinstance(feature.get("apiContract"), list) and not feature["apiContract"] and "apiContract required" in scoped:
            scope["emptyApiFeatureIds"].append(feature.get("featureId"))
        missing_methods = sorted({method for issue in issues if re.search(rf"/features/{index}(?:/|:|$)", issue)
                                  and any(code in issue for code in ("FEATURE_ACTION_CONTRACT_MISSING", "PRD_ACTION_CONTRACT_MISSING"))
                                  for method in re.findall(r"\b(?:GET|POST|PATCH|PUT|DELETE|HEAD|OPTIONS)\b", issue)})
        if missing_methods and isinstance(feature.get("featureId"), str):
            scope["missingOperationMethods"][feature["featureId"]] = missing_methods
        missing_keys = sorted({f"{method} {path}" for issue in issues
                               if issue.startswith("WORKFLOW_LOOKUP_MISSING:") and re.search(rf"/features/{index}(?:/|:|$)", issue)
                               for method, path in re.findall(r"\b(GET|POST|PUT|PATCH) (/api/v1/[^\s:]+)", issue)})
        op_indices = {int(value) for issue in issues for value in re.findall(rf"/features/{index}/apiContract/(\d+)", issue)}
        # A lookup route that already exists (wrong output field or roles) has to
        # be corrected in place. Listing it only as an append key made the merge
        # reject the model's correct fix as "unrelated endpoint changed".
        for op_index, operation in enumerate(_array(feature.get("apiContract"))):
            key = f"{operation.get('method')} {operation.get('path')}" if isinstance(operation, dict) else None
            if key in missing_keys:
                missing_keys.remove(key)
                op_indices.add(op_index)
        if missing_keys:
            scope["missingOperationKeys"][feature["featureId"]] = missing_keys
        if any(issue.startswith("API_REQUIRED_OUTPUT_MISSING:") and re.search(rf"/features/{index}(?:\s*:|$)", issue)
               for issue in issues):
            # A feature-level output defect also permits repairing its current
            # response contracts, even when a separate missing-method issue exists.
            op_indices.update(range(len(_array(feature.get("apiContract")))))
        if (op_indices or missing_methods or missing_keys) and isinstance(feature.get("featureId"), str):
            scope["operationIndices"][feature["featureId"]] = sorted(op_indices)
            scope["operationKeys"][feature["featureId"]] = [
                f"{op.get('method')} {op.get('path')}" for op_index, op in enumerate(_array(feature.get("apiContract")))
                if op_index in op_indices and isinstance(op, dict)
            ]
        if any(token in scoped for token in ("DB_", "dbContract", "dbTables", "STORAGE", "FK_", "MAPPING_TARGET", "WORKFLOW_STATE_", "WORKFLOW_INTERVAL_")):
            scope["tableNames"].extend(_array(feature.get("dbTables")))
            if "FK_" in scoped:
                for table in _array(payload.get("tables")):
                    if isinstance(table, dict) and table.get("name") in _array(feature.get("dbTables")):
                        scope["tableNames"].extend(
                            fk["references"].get("table") for fk in _array(table.get("foreignKeys"))
                            if isinstance(fk, dict) and isinstance(fk.get("references"), dict))
        for op_index, operation in enumerate(_array(feature.get("apiContract"))):
            if not isinstance(operation, dict) or (op_indices and op_index not in op_indices):
                continue
            scope["schemaNames"].extend(ref for ref in (operation.get("requestType"), operation.get("responseType")) if ref)
            scope["schemaNames"].extend(p.get("typeRef") for p in _array(operation.get("parameters")) if isinstance(p, dict))
    for collection, target in (("tables", "tableNames"), ("schemas", "schemaNames")):
        for issue in issues:
            for value in re.findall(rf"/{collection}/(\d+)", issue):
                entries = _array(payload.get(collection))
                if int(value) < len(entries) and isinstance(entries[int(value)], dict):
                    scope[target].append(entries[int(value)].get("name"))
    definitions = {item.get("name"): item for item in _array(payload.get("schemas")) if isinstance(item, dict)}
    pending, reached = list(scope["schemaNames"]), set()
    while pending:
        name = pending.pop()
        if not isinstance(name, str) or name in reached:
            continue
        reached.add(name)
        pending.extend(node.get("ref") for node in _array(definitions.get(name, {}).get("nodes"))
                       if isinstance(node, dict) and node.get("kind") == "ref")
    pending = []
    if scope["featureIds"]:
        for feature in features:
            if not isinstance(feature, dict):
                continue
            fid = feature.get("featureId")
            if fid in scope["featureIds"] and fid not in scope["operationIndices"]:
                continue
            affected = scope["operationIndices"].get(fid, [])
            for position, operation in enumerate(_array(feature.get("apiContract"))):
                if position in affected or not isinstance(operation, dict):
                    continue
                pending.extend(name for name in (operation.get("requestType"), operation.get("responseType")) if isinstance(name, str))
                pending.extend(parameter["typeRef"] for parameter in _array(operation.get("parameters"))
                               if isinstance(parameter, dict) and isinstance(parameter.get("typeRef"), str))
    protected = set()
    while pending:
        name = pending.pop()
        if name in protected:
            continue
        protected.add(name)
        pending.extend(node.get("ref") for node in _array(definitions.get(name, {}).get("nodes"))
                       if isinstance(node, dict) and node.get("kind") == "ref" and isinstance(node.get("ref"), str))
    scope["schemaNames"] = list(reached - protected)
    scope["protectedSchemaNames"] = list(protected)
    for key, values in scope.items():
        if isinstance(values, list):
            scope[key] = sorted({value for value in values if isinstance(value, str) and value})
    return scope


def _new_resource_scope(baseline, patch, scope):
    """Allow new assets only through explicitly scoped missing contracts."""
    previous_features = _array(baseline.get("features"))
    previous_ids = {item.get("featureId") for item in previous_features
                    if isinstance(item, dict) and isinstance(item.get("featureId"), str)}
    previous_sources = {item.get("sourceFeatureId") for item in previous_features
                        if isinstance(item, dict) and item.get("source", "prd_core") == "prd_core"
                        and isinstance(item.get("sourceFeatureId"), str)}
    missing = set(scope["missingSourceFeatureIds"]) - previous_sources
    additions = [item for item in _array(patch.get("features")) if isinstance(item, dict)
                 and isinstance(item.get("sourceFeatureId"), str) and item["sourceFeatureId"] in missing
                 and item.get("source") == "prd_core" and isinstance(item.get("featureId"), str)
                 and item["featureId"] not in previous_ids]
    roots = {"tables": set(), "schemas": set(scope.get("schemaNames", []))}
    for feature in additions:
        roots["tables"].update(name for name in _array(feature.get("dbTables")) if isinstance(name, str))
        for operation in _array(feature.get("apiContract")):
            if not isinstance(operation, dict):
                continue
            roots["schemas"].update(value for value in (operation.get("requestType"), operation.get("responseType")) if isinstance(value, str))
            roots["schemas"].update(parameter["typeRef"] for parameter in _array(operation.get("parameters"))
                                    if isinstance(parameter, dict) and isinstance(parameter.get("typeRef"), str))
    for replacement in _array(patch.get("features")):
        if not isinstance(replacement, dict):
            continue
        eligible = [item for item in previous_features if isinstance(item, dict) and item.get("featureId") in scope["featureIds"]]
        matches = [item for item in eligible if item.get("featureId") == replacement.get("featureId")]
        if not matches and replacement.get("sourceFeatureId"):
            matches = [item for item in eligible if item.get("sourceFeatureId") == replacement["sourceFeatureId"]]
        if len(matches) != 1:
            continue
        previous = matches[0]
        affected = scope.get("operationIndices", {}).get(previous.get("featureId"))
        missing_methods = scope.get("missingOperationMethods", {}).get(previous.get("featureId"), [])
        missing_keys = scope.get("missingOperationKeys", {}).get(previous.get("featureId"), [])
        previous_ops = _array(previous.get("apiContract"))
        if affected is None:
            affected = list(range(len(previous_ops)))
        empty_contract = not previous_ops and previous.get("featureId") in scope.get("emptyApiFeatureIds", [])
        for index, operation in enumerate(_array(replacement.get("apiContract"))):
            matching = [position for position, old in enumerate(previous_ops) if isinstance(old, dict) and isinstance(operation, dict)
                        and (old.get("method"), old.get("path")) == (operation.get("method"), operation.get("path"))]
            full_slot = len(_array(replacement.get("apiContract"))) >= len(previous_ops) and index in affected
            if not isinstance(operation, dict) or not (empty_contract or any(position in affected for position in matching) or full_slot or
                    (not matching and (operation.get("method") in missing_methods or
                                      f"{operation.get('method')} {operation.get('path')}" in missing_keys))):
                continue
            roots["schemas"].update(value for value in (operation.get("requestType"), operation.get("responseType")) if isinstance(value, str))
            roots["schemas"].update(parameter["typeRef"] for parameter in _array(operation.get("parameters"))
                                    if isinstance(parameter, dict) and isinstance(parameter.get("typeRef"), str))
    allowed = {}
    for collection, names in roots.items():
        existing = {item["name"] for item in _array(baseline.get(collection))
                    if isinstance(item, dict) and isinstance(item.get("name"), str)}
        definitions = {item["name"]: item for item in _array(patch.get(collection))
                       if isinstance(item, dict) and isinstance(item.get("name"), str)}
        pending, reached = list(names), set()
        while pending:
            name = pending.pop()
            if name in reached or (name in existing and not (collection == "schemas" and name in scope.get("schemaNames", []))):
                continue
            reached.add(name)
            definition = definitions.get(name, {})
            if collection == "tables":
                pending.extend(fk["references"]["table"] for fk in _array(definition.get("foreignKeys"))
                               if isinstance(fk, dict) and isinstance(fk.get("references"), dict)
                               and isinstance(fk["references"].get("table"), str))
            else:
                pending.extend(node["ref"] for node in _array(definition.get("nodes"))
                               if isinstance(node, dict) and node.get("kind") == "ref" and isinstance(node.get("ref"), str))
        allowed[collection] = reached
    return allowed


def merge_patch(baseline, patch, scope):
    result = deepcopy(baseline)

    def merge_feature(previous, changes):
        merged = deepcopy(previous)
        for key, value in changes.items():
            if isinstance(value, dict) and isinstance(merged.get(key), dict):
                merged[key] = merge_feature(merged[key], value)
            else:
                merged[key] = deepcopy(value)
        return merged

    def merge_operation(previous, changes):
        for key in ("requiredInputs", "requiredOutputs", "requiredEffectTables"):
            if key in changes and not set(previous.get(key) or []) <= set(changes.get(key) or []):
                raise ValueError("FEATURE_SPEC_REPAIR_SCOPE: acceptance requirement removed: " + key)
        try:
            OperationDraft.model_validate(previous)
        except ValidationError:
            try:
                OperationDraft.model_validate(changes)
            except ValidationError:
                pass
            else:
                # A complete valid replacement may omit optional malformed
                # fields and use their model defaults; sparse edits may not.
                return deepcopy(changes)
        return merge_feature(previous, changes)

    def feature_replacement(previous, changes):
        merged = merge_feature(previous, changes)
        reads = changes.get("readDependencies")
        if isinstance(reads, list) and reads and all(isinstance(item, dict) for item in reads):
            def read_key(item):
                return item.get("method", "GET"), item.get("path"), item.get("responseField", "")

            combined = {read_key(item): deepcopy(item) for item in _array(previous.get("readDependencies")) if isinstance(item, dict)}
            if len({read_key(item) for item in reads}) != len(reads):
                raise ValueError("FEATURE_SPEC_REPAIR_SCOPE: duplicate read dependency patch")
            combined.update((read_key(item), deepcopy(item)) for item in reads)
            merged["readDependencies"] = list(combined.values())
        affected = scope.get("operationIndices", {}).get(previous.get("featureId"))
        if "apiContract" not in changes:
            return merged
        old_ops, patch_ops = _array(previous.get("apiContract")), changes["apiContract"]
        if affected is None:
            affected = list(range(len(old_ops)))
        if not isinstance(patch_ops, list) or any(not isinstance(op, dict) for op in patch_ops):
            return merged
        # Full images retain index semantics; sparse images address known routes.
        if len(patch_ops) >= len(old_ops) and all(old == patch_ops[index] for index, old in enumerate(old_ops) if index not in affected):
            merged["apiContract"] = [merge_operation(old_ops[index], operation) if index < len(old_ops) and isinstance(old_ops[index], dict)
                                     else deepcopy(operation) for index, operation in enumerate(patch_ops)]
            return merged
        operations = deepcopy(old_ops)
        missing = list(scope.get("missingOperationMethods", {}).get(previous.get("featureId"), []))
        missing_keys = list(scope.get("missingOperationKeys", {}).get(previous.get("featureId"), []))
        seen = set()
        for operation in patch_ops:
            key = (operation.get("method"), operation.get("path"))
            if key in seen:
                raise ValueError("FEATURE_SPEC_REPAIR_SCOPE: duplicate endpoint patch")
            seen.add(key)
            matches = [index for index, old in enumerate(old_ops) if isinstance(old, dict) and (old.get("method"), old.get("path")) == key]
            if len(matches) == 1:
                index = matches[0]
                replacement = merge_operation(old_ops[index], operation)
                if index not in affected and replacement != old_ops[index]:
                    raise ValueError("FEATURE_SPEC_REPAIR_SCOPE: unrelated endpoint changed")
                operations[index] = replacement
            elif not matches and operation.get("method") in missing:
                missing.remove(operation["method"])
                operations.append(deepcopy(operation))
            elif not matches and f"{key[0]} {key[1]}" in missing_keys:
                missing_keys.remove(f"{key[0]} {key[1]}")
                operations.append(deepcopy(operation))
            else:
                raise ValueError("FEATURE_SPEC_REPAIR_SCOPE: unrelated endpoint changed")
        merged["apiContract"] = operations
        return merged

    for collection in ("features", "tables", "schemas"):
        if collection in baseline and not isinstance(baseline[collection], list):
            raise ValueError(f"FEATURE_SPEC_REPAIR_SCOPE: /{collection}: baseline array required")

    def preserve_operations(previous, replacement):
        for name, keys in (("statePolicies", ("table", "column")), ("intervalPolicies", ("table",))):
            required = {tuple(item.get(key) for key in keys) for item in _array(previous.get(name)) if isinstance(item, dict)}
            actual = {tuple(item.get(key) for key in keys) for item in _array(replacement.get(name)) if isinstance(item, dict)}
            if not required <= actual:
                raise ValueError("FEATURE_SPEC_REPAIR_SCOPE: promised workflow policy removed")
        required_reads = {(item.get("method", "GET"), item.get("path"), item.get("responseField", ""))
                          for item in _array(previous.get("readDependencies")) if isinstance(item, dict)}
        replacement_reads = {(item.get("method", "GET"), item.get("path"), item.get("responseField", ""))
                             for item in _array(replacement.get("readDependencies")) if isinstance(item, dict)}
        if not required_reads <= replacement_reads:
            raise ValueError("FEATURE_SPEC_REPAIR_SCOPE: promised read dependency removed")
        allowed_indices = scope.get("operationIndices", {}).get(previous.get("featureId"))
        allowed_ops = scope.get("operationKeys", {}).get(previous.get("featureId"))
        if allowed_ops is None:
            return
        if not isinstance(replacement.get("apiContract"), list) or any(not isinstance(op, dict) for op in replacement["apiContract"]):
            raise ValueError("FEATURE_SPEC_REPAIR_SCOPE: apiContract object array required")
        if allowed_indices is not None:
            previous_ops = _array(previous.get("apiContract"))
            additional = replacement["apiContract"][len(previous_ops):]
            missing_methods = scope.get("missingOperationMethods", {}).get(previous.get("featureId"), [])
            missing_keys = scope.get("missingOperationKeys", {}).get(previous.get("featureId"), [])
            if len(replacement["apiContract"]) < len(previous_ops) or len(additional) > len(missing_methods) + len(missing_keys) or any(
                operation.get("method") not in missing_methods and f"{operation.get('method')} {operation.get('path')}" not in missing_keys
                for operation in additional
            ) or any(
                operation != replacement["apiContract"][index] for index, operation in enumerate(previous_ops) if index not in allowed_indices
            ):
                raise ValueError("FEATURE_SPEC_REPAIR_SCOPE: unrelated endpoint changed")
            return
        unaffected = {f"{op.get('method')} {op.get('path')}": op for op in _array(previous.get("apiContract"))
                      if isinstance(op, dict) and f"{op.get('method')} {op.get('path')}" not in allowed_ops}
        candidate_ops = {f"{op.get('method')} {op.get('path')}": op for op in replacement["apiContract"]}
        if any(candidate_ops.get(key) != operation for key, operation in unaffected.items()) or len(set(candidate_ops) - set(unaffected)) > len(allowed_ops):
            raise ValueError("FEATURE_SPEC_REPAIR_SCOPE: unrelated endpoint changed")
    if set(patch) - {"version", "features", "tables", "schemas"} or baseline.get("version") != "contract-draft-v1" or patch.get("version", baseline["version"]) != baseline["version"]:
        raise ValueError("FEATURE_SPEC_REPAIR_SCOPE: invalid draft patch envelope")
    new_resources = _new_resource_scope(baseline, patch, scope)
    for collection, identity, allowed in (("features", "featureId", "featureIds"), ("tables", "name", "tableNames"), ("schemas", "name", "schemaNames")):
        entries = patch.get(collection, [])
        if not isinstance(entries, list):
            raise ValueError("FEATURE_SPEC_REPAIR_SCOPE: patch array required")  # noqa: TRY004 - classified contract failure
        existing = {item[identity]: i for i, item in enumerate(result.get(collection) or [])
                    if isinstance(item, dict) and isinstance(item.get(identity), str)}
        seen = set()
        for entry_index, item in enumerate(entries):
            if not isinstance(item, dict) or not isinstance(item.get(identity), str) or item[identity] in seen:
                raise ValueError(f"FEATURE_SPEC_REPAIR_SCOPE: /{collection}/{entry_index}/{identity}: invalid or duplicate patch key")
            key = item[identity]
            seen.add(key)
            if collection == "schemas" and key in scope.get("protectedSchemaNames", []) and key in existing:
                if item == result[collection][existing[key]]:
                    continue
                raise ValueError(f"FEATURE_SPEC_REPAIR_SCOPE: unrelated endpoint schema changed: {key}")
            if collection == "features" and key not in existing and item.get("sourceFeatureId"):
                matches = [i for i, previous in enumerate(result[collection])
                           if isinstance(previous, dict) and previous.get("featureId") in scope[allowed]
                           and previous.get("sourceFeatureId") == item["sourceFeatureId"]]
                if len(matches) == 1:
                    previous = result[collection][matches[0]]
                    replacement = feature_replacement(previous, item)
                    preserve_operations(previous, replacement)
                    result[collection][matches[0]] = replacement
                    continue
            missing = collection == "features" and item.get("sourceFeatureId") in scope["missingSourceFeatureIds"]
            new_resource = collection in new_resources and key not in existing and key in new_resources[collection]
            if key not in scope[allowed] and not missing and not new_resource:
                raise ValueError(f"FEATURE_SPEC_REPAIR_SCOPE: unrequested {collection} patch {key}")
            if key in existing:
                previous = result[collection][existing[key]]
                if collection == "features" and "sourceFeatureId" in item and previous.get("sourceFeatureId") != item["sourceFeatureId"]:
                    raise ValueError("FEATURE_SPEC_REPAIR_SCOPE: source identity changed")
                if collection == "features":
                    replacement = feature_replacement(previous, item)
                    preserve_operations(previous, replacement)
                else:
                    replacement = deepcopy(item)
                result[collection][existing[key]] = replacement
            else:
                result.setdefault(collection, []).append(deepcopy(item))
    if scope.get("operationIndices"):
        before = {item["name"]: item for item in _array(baseline.get("schemas"))
                  if isinstance(item, dict) and isinstance(item.get("name"), str)}
        after = {item["name"]: item for item in _array(result.get("schemas"))
                 if isinstance(item, dict) and isinstance(item.get("name"), str)}
        pending = []
        for feature in _array(baseline.get("features")):
            if not isinstance(feature, dict) or not isinstance(feature.get("featureId"), str):
                continue
            if feature["featureId"] in scope["featureIds"] and feature["featureId"] not in scope["operationIndices"]:
                continue
            affected = scope["operationIndices"].get(feature["featureId"], [])
            for index, operation in enumerate(_array(feature.get("apiContract"))):
                if index in affected or not isinstance(operation, dict):
                    continue
                pending.extend(value for value in (operation.get("requestType"), operation.get("responseType")) if isinstance(value, str))
                pending.extend(parameter["typeRef"] for parameter in _array(operation.get("parameters"))
                               if isinstance(parameter, dict) and isinstance(parameter.get("typeRef"), str))
        checked = set()
        while pending:
            name = pending.pop()
            if name in checked:
                continue
            checked.add(name)
            if before.get(name) != after.get(name):
                raise ValueError(f"FEATURE_SPEC_REPAIR_SCOPE: unrelated endpoint schema changed: {name}")
            pending.extend(node["ref"] for node in _array(before.get(name, {}).get("nodes"))
                           if isinstance(node, dict) and node.get("kind") == "ref" and isinstance(node.get("ref"), str))
    return result
