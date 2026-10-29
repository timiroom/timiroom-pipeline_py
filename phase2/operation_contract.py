"""Business operation contracts shared by generation, projection and validation."""

from __future__ import annotations

import re
from copy import deepcopy

from phase2.agent_contract import (
    canonical_api_method,
    normalize_api_path,
    path_input_fields,
    server_managed_fields,
)
from phase2.contract_quality import (
    database_checks,
    index_issues,
    operation_primary_table,
    operation_quality_issues,
)
from phase2.feature_scope import artifact_required


def _without_nulls(value):
    if isinstance(value, dict):
        return {key: _without_nulls(item) for key, item in value.items() if item is not None}
    if isinstance(value, list):
        return [_without_nulls(item) for item in value]
    return value


def route_key(operation: dict) -> tuple[str, str]:
    path = normalize_api_path(operation.get("path"))
    return canonical_api_method(operation.get("method"), path), re.sub(r"\{[^}]+\}", "{}", path)


def normalize_db_contract(raw: dict | None) -> dict:
    contract = deepcopy(raw) if isinstance(raw, dict) else {}
    names = []
    for value in contract.get("tables") or []:
        name = str(value.get("name") or value.get("table") or "") if isinstance(value, dict) else str(value)
        if name and name not in names:
            names.append(name)
    contract["tables"] = names
    for key in ("foreignKeys", "fields", "uniqueConstraints"):
        if not isinstance(contract.get(key), list):
            contract[key] = []
    return contract


def _auth_features(registry):
    return [(index, item) for index, item in enumerate(registry) if
            isinstance(item.get("sessionPolicy"), dict) and item["sessionPolicy"].get("strategy") or any(
                isinstance(operation, dict) and (
                    normalize_api_path(operation.get("path")).startswith("/api/v1/auth/") or
                    isinstance(operation.get("sessionPolicy"), dict) and operation["sessionPolicy"].get("strategy")
                ) for operation in item.get("apiContract") or []
            )]


def session_tables(registry):
    explicit, inferred = set(), set()
    fields, issued = {}, set()
    for _, feature in _auth_features(registry):
        for operation in feature.get("apiContract") or []:
            if isinstance(operation, dict) and operation.get("authRequired") is False:
                issued.update(effect.get("table") for effect in operation.get("effects") or []
                              if isinstance(effect, dict) and effect.get("kind") in {"insert", "create"})
        policies = [feature.get("sessionPolicy"), *(op.get("sessionPolicy") for op in feature.get("apiContract") or [] if isinstance(op, dict))]
        for policy in policies:
            if isinstance(policy, dict) and (policy.get("sessionTable") or policy.get("table")):
                explicit.add(str(policy.get("sessionTable") or policy["table"]))
        for field in (feature.get("dbContract") or {}).get("fields") or []:
            if isinstance(field, dict) and isinstance(field.get("table"), str) and isinstance(field.get("name"), str):
                fields.setdefault(field.get("table"), set()).add(field.get("name"))
    inferred = {name for name, names in fields.items() if name and {"token_hash", "expires_at"} <= names}
    # Invite/reset tokens share token_hash+expires_at but are written by an
    # authenticated issuer, not by the public login/signup that opens a session.
    # Counting them made a single declared sessionTable an unrepairable conflict.
    if issued:
        inferred &= issued
    return explicit, inferred


def session_policy_issues(registry):
    auth = _auth_features(registry)
    explicit, inferred = session_tables(registry)
    strategies = {policy.get("strategy") for _, item in auth
                  for policy in [item.get("sessionPolicy"), *(op.get("sessionPolicy") for op in item.get("apiContract") or [] if isinstance(op, dict))]
                  if isinstance(policy, dict) and policy.get("strategy")}
    conflict = len(explicit) > 1 or len(inferred) > 1 or bool(explicit and inferred - explicit) or len(strategies) > 1
    for key in ("transport", "tokenFormat", "tokenField", "cookieName", "expiresInSeconds", "defaultRole"):
        values = {str(policy[key]) for _, feature in auth for policy in [feature.get("sessionPolicy"), *(
            op.get("sessionPolicy") for op in feature.get("apiContract") or [] if isinstance(op, dict)
        )] if isinstance(policy, dict) and key in policy}
        conflict = conflict or len(values) > 1
    if conflict:
        return [f"AUTH_SESSION_CONTRACT_CONFLICT: /features/{index}/sessionPolicy: one strategy and one declared sessionTable required; tables={sorted(explicit | inferred)}"
                for index, _ in auth]
    return []


def is_public_session_entry(operation: dict, feature: dict) -> bool:
    """Recognize explicitly declared session issuance, not arbitrary public writes."""
    policy = operation.get("sessionPolicy") or feature.get("sessionPolicy") or {}
    table = policy.get("sessionTable")
    return (
        operation.get("authRequired") is False
        and str(operation.get("method")).upper() == "POST"
        # Decided by structure, not by the action's wording: any unauthenticated
        # POST that inserts into the declared session table opens a session
        # (login, invite acceptance, magic link...). Verbs that end a session or
        # change existing data are excluded.
        and not re.search(r"log_?out|sign_?out|revoke|update|edit|patch|delete|remove",
                          str(operation.get("action") or ""), re.IGNORECASE)
        and policy.get("strategy") in {"bearer_session", "refresh_token"}
        and bool(table)
        and table in (feature.get("dbContract") or {}).get("tables", [])
        and bool(operation.get("requestSchema"))
        and any(effect.get("table") == table and effect.get("kind") in {"insert", "create"}
                for effect in operation.get("effects") or [] if isinstance(effect, dict))
    )


def bind_auth_registry_policy(registry: list[dict]) -> None:
    """Publish one session lifecycle to both parallel generators."""
    auth = [item for _, item in _auth_features(registry)]
    if not auth:
        return
    policies = [policy for item in auth for policy in [item.get("sessionPolicy"),
                *(operation.get("sessionPolicy") for operation in item.get("apiContract") or [] if isinstance(operation, dict))]
                if isinstance(policy, dict) and policy]
    explicit = next((policy for policy in policies if policy.get("strategy")), policies[0] if policies else None)
    refresh = any(normalize_api_path(operation.get("path")).endswith("/auth/refresh")
                  for item in auth for operation in item.get("apiContract") or [] if isinstance(operation, dict))
    policy = {"strategy": "refresh_token" if refresh else "bearer_session", **(explicit or {})}
    declared, inferred = session_tables(registry)
    if session_policy_issues(registry):
        return
    canonical_table = next(iter(declared or inferred), "refresh_tokens" if policy.get("strategy") == "refresh_token" else "auth_sessions")
    policy = {**policy, "sessionTable": canonical_table}
    all_fields = [field for feature in registry for field in (feature.get("dbContract") or {}).get("fields") or [] if isinstance(field, dict)]
    for item in auth:
        # Preserve contradictory declarations so QA can reject them rather
        # than silently blessing whichever agent output arrived first.
        item.setdefault("sessionPolicy", deepcopy(policy))
        effective = item["sessionPolicy"]
        effective.setdefault("strategy", policy["strategy"])
        effective.setdefault("sessionTable", canonical_table)
        session_table = canonical_table
        db = normalize_db_contract(item.get("dbContract"))
        uses_session = session_table in db["tables"] or any(
            str(operation.get("path") or "").rstrip("/").rsplit("/", 1)[-1] in {"login", "logout", "refresh"}
            for operation in item.get("apiContract") or []
        )
        for operation in item.get("apiContract") or []:
            if isinstance(operation, dict) and normalize_api_path(operation.get("path")).startswith("/api/v1/auth/"):
                operation.setdefault("sessionPolicy", deepcopy(effective))
        if not uses_session:
            continue
        if db["tables"] and all(any(field.get("table") == name and field.get("primaryKey") for field in db["fields"])
                                for name in db["tables"]):
            # A typed physical contract is authoritative. Legacy enrichment
            # must not invent a second principal FK or change its columns.
            item["dbContract"] = db
            continue
        owner = str((item.get("ownership") or {}).get("ownerEntity") or "users")
        db["tables"] = list(dict.fromkeys([*db["tables"], owner, session_table]))
        for name, sql_type, required in (("token_hash", "VARCHAR(255)", True), ("expires_at", "TIMESTAMPTZ", True), ("revoked_at", "TIMESTAMPTZ", False)):
            if not any(isinstance(field, dict) and field.get("table") == session_table and field.get("name") == name for field in db["fields"]):
                db["fields"].append({"table": session_table, "name": name, "type": sql_type, "required": required, "generated": True, "source": "server"})
        unique = {"table": session_table, "columns": ["token_hash"]}
        if unique not in db["uniqueConstraints"]:
            db["uniqueConstraints"].append(unique)
        owner_column = f"{owner.removesuffix('s')}_id"
        owner_id = next((field for field in all_fields if field.get("table") == owner and field.get("name") == "id"), {})
        if owner_id.get("type") and not any(field.get("table") == session_table and field.get("name") == owner_column for field in db["fields"]):
            db["fields"].append({"table": session_table, "name": owner_column, "type": owner_id["type"], "required": True, "source": "auth"})
        if not any(isinstance(fk, dict) and fk.get("table") == session_table and fk.get("column") == owner_column for fk in db["foreignKeys"]):
            db["foreignKeys"].append({"table": session_table, "column": owner_column, "references": {"table": owner, "column": "id"}, "nullable": False})
        item["dbContract"] = db
        for operation in item.get("apiContract") or []:
            if isinstance(operation, dict) and normalize_api_path(operation.get("path")).startswith("/api/v1/auth/"):
                operation.setdefault("sessionPolicy", deepcopy(effective))


def auxiliary_only_contract(feature: dict, registry: list[dict]) -> bool:
    """An auxiliary operation needs a declared parent or dependency journey.

    A parent pointer alone is insufficient: an unrelated authentication journey
    must not legitimize a status-only business resource. Cycles are not followed.
    """
    auxiliary = re.compile(r"/(?:retry|history|audit|status)(?:/|$)")
    contracts = [item for item in feature.get("apiContract") or [] if isinstance(item, dict)]
    if not contracts or any(not auxiliary.search(normalize_api_path(item.get("path"))) for item in contracts):
        return False
    def resource(path):
        parts = [part for part in normalize_api_path(path).split("/")
                 if part and not part.startswith("{") and part not in {"api", "v1", "retry", "history", "audit", "status"}]
        return parts[-1] if parts else ""

    resources = {resource(item.get("path")) for item in contracts} - {""}
    by_id = {str(item.get("featureId") or item.get("id") or ""): item for item in registry if isinstance(item, dict)}
    visited = {str(feature.get("featureId") or feature.get("id") or "")}
    def related_ids(item):
        dependencies = item.get("dependencies") or []
        return [str(item.get("parentFeatureId") or ""),
                *[value for value in dependencies if isinstance(value, str)]]

    pending = related_ids(feature)
    while pending:
        parent_id = pending.pop()
        if not parent_id or parent_id in visited or parent_id not in by_id:
            continue
        visited.add(parent_id)
        parent = by_id[parent_id]
        for operation in parent.get("apiContract") or []:
            if not isinstance(operation, dict):
                continue
            path = normalize_api_path(operation.get("path"))
            if not auxiliary.search(path) and resource(path) in resources:
                return False
        pending.extend(related_ids(parent))
    return True


def schema_summary(schema: dict) -> str:
    """Keep legacy document renderers readable without discarding JSON Schema."""
    return ", ".join(
        f"{name}: {value.get('type', 'object')}" + (f" — {value['description']}" if value.get("description") else "")
        for name, value in (schema.get("properties") or {}).items() if isinstance(value, dict)
    ) or "result: object"


def _schema_nodes(schema: dict, schemas: dict | None = None, seen: frozenset = frozenset()):
    if not isinstance(schema, dict):
        return
    yield schema
    reference = schema.get("$ref")
    if isinstance(reference, str) and reference.startswith("#/components/schemas/") and reference not in seen:
        name = reference.rsplit("/", 1)[-1].replace("~1", "/").replace("~0", "~")
        yield from _schema_nodes((schemas or {}).get(name), schemas, seen | {reference})
    for child in (schema.get("properties") or {}).values():
        yield from _schema_nodes(child, schemas, seen)
    yield from _schema_nodes(schema.get("items"), schemas, seen)
    for key in ("allOf", "anyOf", "oneOf"):
        for child in schema.get(key) or []:
            yield from _schema_nodes(child, schemas, seen)


def _schema_property_names(schema: dict, schemas: dict | None = None) -> set[str]:
    return {name for node in _schema_nodes(schema, schemas) for name in node.get("properties") or {}}


def _has_binary(schema: dict, schemas: dict | None = None) -> bool:
    return any(node.get("format") == "binary" for node in _schema_nodes(schema, schemas))


def bind_operation_contract(endpoint: dict, feature: dict) -> dict:
    contract = next((item for item in feature.get("apiContract") or []
                     if isinstance(item, dict) and route_key(item) == route_key(endpoint)), {})
    primary = operation_primary_table(contract, (feature.get("dbContract") or {}).get("primaryTable"))
    if primary:
        endpoint["primaryTable"] = primary
    if contract:
        if "effects" in contract and "successStatus" in contract and "sessionPolicy" not in contract:
            endpoint.pop("sessionPolicy", None)
        declared_effects = contract.get("effects") or []
        if endpoint.get("effects") and endpoint["effects"] != declared_effects:
            endpoint.setdefault("generationWarnings", []).append("API_UNDECLARED_EFFECTS_IGNORED")
        endpoint["effects"] = deepcopy(declared_effects)
        bodyless = str(contract.get("method") or endpoint.get("method")).upper() in {"GET", "DELETE", "HEAD"}
        typed_contract = "effects" in contract and "successStatus" in contract
        if (bodyless or typed_contract) and "requestSchema" not in contract:
            endpoint.pop("requestSchema", None)
            endpoint["requestBody"] = "없음"
    for key in ("requestSchema", "responseSchema", "contentType", "successStatus", "fieldMappings",
                "effects", "sessionPolicy", "transactionRules", "responses", "parameters", "authRequired",
                "responseHeaders", "allowedRoles", "errorCodes", "accessScope", "primaryTable",
                "requiredInputs", "requiredOutputs", "requiredEffectTables", "pagination"):
        if key in contract and not (key == "transactionRules" and not contract[key]):
            endpoint[key] = deepcopy(contract[key])
    for schema_key, legacy_key in (("requestSchema", "requestBody"), ("responseSchema", "successResponse")):
        if isinstance(endpoint.get(schema_key), dict):
            endpoint[legacy_key] = schema_summary(endpoint[schema_key])
    return endpoint


def _field_schema(schema, field, schemas=None, seen=frozenset()):
    if not isinstance(schema, dict):
        return None
    reference = schema.get("$ref")
    if reference and reference not in seen:
        name = str(reference).rsplit("/", 1)[-1]
        resolved = _field_schema((schemas or {}).get(name), field, schemas, seen | {reference})
        if resolved is not None:
            return resolved
    if not field:
        return schema
    parts = [part.replace("~1", "/").replace("~0", "~") for part in
             re.sub(r"\[\]", ".*", str(field)).replace("/", ".").strip(".").split(".")]
    head, tail = parts[0], ".".join(parts[1:])
    schema_types = schema.get("type")
    if schema_types == "array" or isinstance(schema_types, list) and "array" in schema_types:
        return _field_schema(schema.get("items"), tail if head in {"*", "items", "0"} else field, schemas, seen)
    child = (schema.get("properties") or {}).get(head)
    if child is not None:
        return _field_schema(child, tail, schemas, seen)
    for key in ("allOf", "anyOf", "oneOf"):
        for branch in schema.get(key) or []:
            found = _field_schema(branch, field, schemas, seen)
            if found is not None:
                return found
    return None


def effect_schema_issues(endpoint, tables):
    issues = []
    for index, effect in enumerate(endpoint.get("effects") or []):
        if not isinstance(effect, dict):
            issues.append(f"API_EFFECT_SHAPE_INVALID: effects/{index}")
            continue
        table_name = effect.get("table")
        table = tables.get(table_name) if isinstance(table_name, str) else None
        kind = str(effect.get("kind") or "").lower()
        if table is None:
            issues.append(f"API_EFFECT_TABLE_UNKNOWN: effects/{index}")
        elif kind not in {"insert", "create", "update", "set", "delete", "increment", "decrement", "revoke", "expire", "state_transition"}:
            issues.append(f"API_EFFECT_KIND_INVALID: effects/{index}")
        elif kind not in {"insert", "create", "delete"} and effect.get("column") not in {
            column.get("name") for column in table.get("columns") or []
        }:
            issues.append(f"API_EFFECT_COLUMN_UNKNOWN: effects/{index}")
    return issues


def operation_schema_issues(endpoint: dict, tables: dict[str, dict], schemas: dict | None = None) -> list[str]:
    """Validate body/path/query/response mappings independently of body presence."""
    issues = [*effect_schema_issues(endpoint, tables), *operation_quality_issues(endpoint)]
    primary = endpoint.get("primaryTable")
    if primary and primary not in tables:
        issues.append(f"API_PRIMARY_TABLE_UNKNOWN: {primary}")
    targets = {effect.get("table") for effect in endpoint.get("effects") or []}
    if primary and targets and primary not in targets:
        issues.append(f"API_PRIMARY_TABLE_MISMATCH: {primary} vs {sorted(targets)}")
    request = endpoint.get("requestSchema")
    if str(endpoint.get("successStatus")) == "204" and endpoint.get("responseSchema"):
        issues.append("API_NO_CONTENT_RESPONSE_BODY: 204")
    if isinstance(request, dict) and str(endpoint.get("method") or "GET").upper() in {"GET", "DELETE", "HEAD"}:
        issues.append("API_BODYLESS_SCHEMA: requestSchema")
    for node in _schema_nodes(request, schemas):
        for name in node.get("required") or []:
            if _field_schema(node, name, schemas) is None:
                issues.append(f"API_SCHEMA_REQUIRED_UNKNOWN: {name}")
    for effect in endpoint.get("effects") or []:
        if not isinstance(effect, dict):
            issues.append("API_EFFECT_INVALID: object required")
            continue
        columns = {item.get("name") for item in tables.get(str(effect.get("table")), {}).get("columns") or []}
        if effect.get("table") not in tables or (effect.get("kind") not in {"insert", "create", "delete"} and effect.get("column") not in columns):
            issues.append(f"API_EFFECT_TARGET_UNKNOWN: {effect.get('table')}.{effect.get('column')}")
        if effect.get("kind") == "state_transition" and (effect.get("from") is None or effect.get("to") is None):
            issues.append("API_STATE_TRANSITION_INCOMPLETE: from/to")
    for mapping in endpoint.get("fieldMappings") or []:
        if not isinstance(mapping, dict):
            issues.append("API_FIELD_MAPPING_INVALID: object required")
            continue
        field = str(mapping.get("field") or "")
        source = str(mapping.get("source") or "request").lower()
        source = {"request": "body", "request.body": "body", "request.path": "path",
                  "request.query": "query", "request.header": "header"}.get(source, source)
        value_schema = None
        if source in {"body", "response"}:
            schema = request if source == "body" else endpoint.get("responseSchema")
            value_schema = _field_schema(schema, mapping.get("pointer") or field, schemas)
            if value_schema is None and not (isinstance(schema, dict) and "$ref" in schema and schemas is None):
                issues.append(f"API_FIELD_MAPPING_INPUT_UNKNOWN: {source}.{field}")
        elif source in {"path", "query", "header", "cookie"}:
            parameter = next((item for item in endpoint.get("parameters") or []
                              if item.get("in") == source and (str(item.get("name")).casefold() == field.casefold()
                                  if source == "header" else item.get("name") == field)), None)
            policy = endpoint.get("sessionPolicy")
            security_input = endpoint.get("authRequired") is True and isinstance(policy, dict) and (
                source == "header" and field.casefold() == "authorization" and policy.get("transport") == "bearer" or
                source == "cookie" and policy.get("transport") == "cookie" and field == policy.get("cookieName")
            )
            if parameter is None and not security_input and not (source == "path" and field in re.findall(r"\{([^}]+)\}", endpoint.get("path") or "")):
                issues.append(f"API_FIELD_MAPPING_INPUT_UNKNOWN: {source}.{field}")
            value_schema = (parameter or {}).get("schema") or {"type": "string" if security_input else (parameter or {}).get("type")}
        elif source not in {"auth", "server", "derived"}:
            issues.append(f"API_FIELD_MAPPING_SOURCE_INVALID: {source}")
        table, column = str(mapping.get("table") or ""), str(mapping.get("column") or "")
        target = next((item for item in tables.get(table, {}).get("columns") or [] if item.get("name") == column), None)
        if (table or column) and target is None:
            issues.append(f"API_FIELD_MAPPING_TARGET_UNKNOWN: {table}.{column}")
        if target and value_schema and not mapping.get("transform"):
            sql_type = str(target.get("type") or "").upper()
            expected = "integer" if re.match(r"(?:BIGINT|SMALLINT|INTEGER|INT|SERIAL)", sql_type) else (
                "number" if re.match(r"(?:NUMERIC|DECIMAL|REAL|DOUBLE|FLOAT)", sql_type) else
                "boolean" if sql_type.startswith("BOOL") else "string" if re.match(r"(?:TEXT|VARCHAR|CHAR|UUID|DATE|TIME)", sql_type) else None)
            actual = value_schema.get("type")
            actual_types = actual if isinstance(actual, list) else [actual]
            if expected and actual and expected not in actual_types and not (expected == "number" and "integer" in actual_types):
                issues.append(f"API_FIELD_MAPPING_TYPE_MISMATCH: {source}.{field} -> {table}.{column}")
    if _has_binary(request, schemas) and endpoint.get("contentType") != "multipart/form-data":
        issues.append("API_FILE_TRANSPORT_REQUIRED: binary input requires multipart/form-data")
    created_tables = {effect.get("table") for effect in endpoint.get("effects") or []
                      if isinstance(effect, dict) and effect.get("kind") in {"insert", "create"} and isinstance(effect.get("table"), str)}
    if not created_tables and endpoint.get("action") in {"create", "register", "apply", "record", "upload"}:
        created_tables.add(str(endpoint.get("primaryTable") or ""))
    for table_name in sorted(created_tables):
        primary = tables.get(table_name)
        if not primary:
            continue
        columns = primary.get("columns") or []
        managed = server_managed_fields(columns)
        # Typed declarations override legacy name heuristics. A column named
        # user_id/password_hash is not proof that a server value was supplied.
        for column in columns:
            if "source" in column and column.get("source") not in {"server", "auth", "derived"} and not (
                column.get("generated") or column.get("readOnly")
            ):
                managed.discard(column.get("name"))
        managed |= path_input_fields(endpoint, columns)
        # A child row created under /records/{recordId} takes its parent FK from
        # the path parameter already bound to that parent table.
        path_tables = {item.get("table") for item in endpoint.get("fieldMappings") or []
                       if isinstance(item, dict) and item.get("source") == "path"}
        for column in columns:
            reference = (column.get("references") or {}).get("table") if isinstance(column.get("references"), dict) else None
            reference = reference or (re.search(r"REFERENCES\s+(\w+)", str(column.get("constraints") or ""), re.IGNORECASE) or [None, None])[1]
            if reference and reference in path_tables:
                managed.add(column.get("name"))
        # roomId in the request DTO is the room_id column; only the casing differs.
        supplied = {name for prop in _schema_property_names(request, schemas)
                    for name in (prop, re.sub(r"(?<!^)(?=[A-Z])", "_", prop).lower())} | {
            str(item.get("column")) for item in endpoint.get("fieldMappings") or []
            if isinstance(item, dict) and item.get("table") == primary.get("name")}
        for column in columns:
            if column.get("name") not in managed | supplied and re.search(r"NOT[_\s]+NULL", str(column.get("constraints")), re.IGNORECASE) and "DEFAULT" not in str(column.get("constraints")).upper():
                issues.append(f"API_CREATE_STORAGE_VALUE_MISSING: {primary['name']}.{column.get('name')}")
    return list(dict.fromkeys(issues))


def _merge_field_mappings(current, incoming) -> list | None:
    """Union field bindings; None when one field/source is bound to two storage targets."""
    if not isinstance(current, list) or not isinstance(incoming, list):
        return None
    merged, bound = deepcopy(current), {}
    for item in current:
        if isinstance(item, dict):
            bound[(item.get("field"), item.get("source"))] = (item.get("table"), item.get("column"))
    for item in incoming:
        if not isinstance(item, dict):
            return None
        key, target = (item.get("field"), item.get("source")), (item.get("table"), item.get("column"))
        if key in bound:
            if bound[key] != target:
                return None
            continue
        bound[key] = target
        merged.append(deepcopy(item))
    return merged


def operation_registry(registry: list[dict]) -> tuple[dict, list[str]]:
    """A route has one contract; disjoint declarations merge, conflicts never overwrite."""
    operations, issues = {}, []
    fields = ("requestSchema", "responseSchema", "contentType", "successStatus", "responses",
              "parameters", "fieldMappings", "effects", "sessionPolicy", "authRequired", "responseHeaders", "allowedRoles", "errorCodes",
              "accessScope", "primaryTable", "requiredInputs", "requiredOutputs", "requiredEffectTables", "pagination")
    for feature in registry:
        fid = str(feature.get("featureId") or feature.get("id") or "")
        if not artifact_required(feature, "api"):
            continue
        for contract in feature.get("apiContract") or []:
            key = route_key(contract)
            entry = operations.setdefault(key, {"method": contract.get("method"), "path": contract.get("path"), "featureIds": []})
            entry["featureIds"].append(fid)
            for field in fields:
                if field not in contract:
                    continue
                if field in {"requiredInputs", "requiredOutputs", "requiredEffectTables"}:
                    entry[field] = list(dict.fromkeys([*entry.get(field, []), *contract[field]]))
                    continue
                if field == "fieldMappings" and field in entry:
                    merged = _merge_field_mappings(entry[field], contract[field])
                    if merged is not None:
                        entry[field] = merged
                        continue
                if field in entry and entry[field] != contract[field]:
                    issues.append(f"API_OPERATION_CONTRACT_CONFLICT: {key[0]} {key[1]} {field}; featureIds={','.join(entry['featureIds'])}")
                else:
                    entry[field] = deepcopy(contract[field])
    return operations, list(dict.fromkeys(issues))


def cross_contract_issues(registry: list[dict], db: dict, api: dict, prd: dict) -> list[tuple[str, str, str]]:
    """Return code/domain/detail for semantic drift shared by QA and Phase3."""
    from phase2.workflow_contract import (
        workflow_check_constraints,
        workflow_contract_issues,
    )

    issues = []
    tables = {str(item.get("name")): item for item in db.get("tables") or [] if isinstance(item, dict)}
    issues.extend(("DB_INDEX_VOLATILE_EXPRESSION", "db", value) for value in index_issues(list(tables.values())))
    endpoints = [item for item in api.get("endpoints") or [] if isinstance(item, dict)]
    for value in workflow_contract_issues(registry, endpoints):
        index = re.search(r"/features/(\d+)", value)
        fid = registry[int(index[1])].get("featureId", "") if index else ""
        issues.append((value.split(":", 1)[0], "prd", f"{fid}: {value}"))
    checks_by_table = workflow_check_constraints(registry)
    for table, checks in database_checks(registry).items():
        checks_by_table.setdefault(table, []).extend(checks)
    for table, checks in checks_by_table.items():
        existing = {re.sub(r"\s+", "", str(check)).casefold() for check in tables.get(table, {}).get("constraints") or []}
        for check in checks:
            if re.sub(r"\s+", "", check).casefold() not in existing:
                issues.append(("DB_WORKFLOW_CHECK_MISSING", "db", f"{table}: {check}"))
    schemas = api.get("schemas") or {}
    _, conflicts = operation_registry(registry)
    issues.extend(("API_OPERATION_CONTRACT_CONFLICT", "prd", message) for message in conflicts)
    policies = {str(item["sessionPolicy"].get("strategy")) for item in registry if isinstance(item.get("sessionPolicy"), dict)}
    if len(policies) > 1:
        issues.append(("AUTH_SESSION_POLICY_CONFLICT", "prd", ", ".join(sorted(policies))))
    if policies == {"refresh_token"} and not any(normalize_api_path(item.get("path")).endswith("/auth/refresh") for item in endpoints):
        issues.append(("AUTH_REFRESH_CONTRACT_MISSING", "prd", "refresh_token strategy requires a refresh operation"))
    for endpoint in endpoints:
        issues.extend((value.split(":", 1)[0], "api", f"{endpoint.get('method')} {endpoint.get('path')}: {value}")
                      for value in operation_schema_issues(endpoint, tables, schemas))
    core = [item for item in prd.get("coreFeatures") or [] if isinstance(item, dict)]
    for feature in registry:
        fid = str(feature.get("featureId") or feature.get("id") or "")
        api_required = artifact_required(feature, "api")
        if not api_required and not artifact_required(feature, "db"):
            continue
        feature_endpoints = [item for item in endpoints if fid in [item.get("featureId"), *(item.get("featureIds") or [])]]
        related_ids = {fid}
        while True:
            expanded = related_ids | {str(item.get("featureId") or item.get("id") or "")
                                      for item in registry if item.get("parentFeatureId") in related_ids}
            if expanded == related_ids:
                break
            related_ids = expanded
        related_endpoints = [item for item in endpoints if related_ids.intersection(
            [item.get("featureId"), *(item.get("featureIds") or [])]
        )]
        narrative = str(feature.get("description") or "") + " " + str(feature.get("requirements") or "")
        upload_required = "upload" in (feature.get("actions") or []) or (
            "업로드" in narrative and any(word in narrative for word in ("파일", "이미지", "사진"))
        )
        if upload_required and api_required:
            multipart = any(item.get("contentType") == "multipart/form-data" and _has_binary(item.get("requestSchema"), schemas) for item in related_endpoints)
            issued_url = any({name.replace("_", "").lower() for name in _schema_property_names(item.get("responseSchema"), schemas)}
                             & {"uploadurl", "presignedurl", "signeduploadurl"} for item in related_endpoints)
            completed = any(str(item.get("method")).upper() in {"POST", "PATCH"} and
                            {name.replace("_", "").lower() for name in _schema_property_names(item.get("requestSchema"), schemas)}
                            & {"storagekey", "objectkey", "uploadid"} for item in related_endpoints)
            if not multipart and not (issued_url and completed):
                issues.append(("FEATURE_UPLOAD_CONTRACT_INCOMPLETE", "prd", f"{fid}: multipart binary or upload URL/completion contract required"))
        response_fields = set().union(*(_schema_property_names(item.get("responseSchema"), schemas) for item in related_endpoints))
        for output in (feature.get("outputRequirements") or []) if api_required else []:
            name = str(output.get("field") or "") if isinstance(output, dict) else str(output)
            resolved = any(_field_schema(item.get("responseSchema"), name, schemas) is not None for item in related_endpoints)
            legacy_leaf = not any(token in name for token in (".", "/", "[")) and name in response_fields
            if name and not resolved and not legacy_leaf:
                issues.append(("API_REQUIRED_OUTPUT_MISSING", "api", f"{fid}: {name}"))
        original = next((item for item in core if item.get("featureId") == fid or item.get("name") == feature.get("name")), None)
        if original is not None and api_required:
            requirements = original.get("requirements") or []
            if isinstance(requirements, str):
                requirements = [requirements]
            text = " ".join([str(original.get("name") or ""), str(original.get("description") or ""), *map(str, requirements)])
            # Only explicit mutation promises are hard requirements here;
            # feature_methods' inferred read/create defaults are not evidence.
            promised = set()
            actions = set(original.get("actions") or []) - {"manage"}
            if actions:
                if actions & {"update", "edit", "change"}:
                    promised.add("PATCH")
                if actions & {"delete", "remove"}:
                    promised.add("DELETE")
            else:
                if any(word in text.lower() for word in ("수정", "변경", "편집", "갱신", "update", "edit", "change")):
                    promised.add("PATCH")
                if any(word in text.lower() for word in ("삭제", "제거", "delete", "remove")):
                    promised.add("DELETE")
            declared = {str(item.get("method")).upper() for item in feature.get("apiContract") or [] if isinstance(item, dict)}
            if "PUT" in declared:
                promised.discard("PATCH")
            for method in sorted(promised - declared):
                issues.append(("PRD_ACTION_CONTRACT_MISSING", "prd", f"{fid}: {method}"))
        for contract in feature.get("apiContract") or []:
            if not isinstance(contract, dict):
                continue
            endpoint = next((item for item in feature_endpoints if route_key(item) == route_key(contract)), None)
            if endpoint is not None:
                for key in ("requestSchema", "responseSchema", "contentType", "successStatus", "fieldMappings", "effects",
                            "sessionPolicy", "allowedRoles", "responseHeaders", "authRequired", "accessScope", "primaryTable", "pagination"):
                    # Agents serialize absent optional values either as null or by
                    # omitting the key; only a real value difference is drift.
                    if key in contract and _without_nulls(endpoint.get(key)) != _without_nulls(contract[key]):
                        issues.append(("API_OPERATION_CONTRACT_DRIFT", "api", f"{fid}: {route_key(contract)} {key}"))
        if not artifact_required(feature, "db"):
            continue
        contract = normalize_db_contract(feature.get("dbContract"))
        primary = str(contract.get("primaryTable") or "")
        if primary and primary not in contract["tables"]:
            issues.append(("DB_PRIMARY_TABLE_UNDECLARED", "prd", f"{fid}: {primary}"))
        for field in contract["fields"]:
            if not isinstance(field, dict):
                continue
            table_name = str(field.get("table") or primary)
            name = str(field.get("name") or "")
            column = next((item for item in tables.get(table_name, {}).get("columns") or [] if item.get("name") == name), None)
            if column is None:
                issues.append(("DB_FIELD_CONTRACT_MISSING", "db", f"{fid}: {table_name}.{name}"))
                continue
            required = bool(re.search(r"NOT[_\s]+NULL|PRIMARY_KEY", str(column.get("constraints")), re.IGNORECASE))
            if "required" in field and bool(field["required"]) != required:
                issues.append(("DB_FIELD_REQUIRED_MISMATCH", "db", f"{fid}: {table_name}.{name}"))
            if field.get("type") and re.sub(r"\s", "", str(field["type"])).upper() != re.sub(r"\s", "", str(column.get("type"))).upper():
                issues.append(("DB_FIELD_TYPE_MISMATCH", "db", f"{fid}: {table_name}.{name}"))
        for unique in contract["uniqueConstraints"]:
            if not isinstance(unique, dict):
                continue
            table_name = str(unique.get("table") or primary)
            names = unique.get("columns") or []
            table = tables.get(table_name, {})
            definitions = [str(index) for index in table.get("indexes") or [] if re.search(r"\bUNIQUE\b", str(index), re.IGNORECASE)]
            present = any(re.search(r"\(\s*" + r"\s*,\s*".join(map(re.escape, names)) + r"\s*\)", definition) for definition in definitions) if names else False
            if len(names) == 1:
                present = present or any(item.get("name") == names[0] and "UNIQUE" in str(item.get("constraints")).upper() for item in table.get("columns") or [])
            if not present:
                issues.append(("DB_UNIQUE_CONTRACT_MISSING", "db", f"{fid}: {table_name}.{names}"))
    return list(dict.fromkeys(issues))
