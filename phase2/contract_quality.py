"""Operation-level access, concrete acceptance contracts and storage bounds."""

import re
from copy import deepcopy


def declared_operation(feature, endpoint):
    from phase2.operation_contract import route_key

    return next((op for op in feature.get("apiContract") or [] if route_key(op) == route_key(endpoint)), None)


def public_read_contract(feature, endpoint):
    op = declared_operation(feature, endpoint)
    return bool(op and op.get("authRequired") is False
                and str(op.get("method")).upper() in {"GET", "HEAD"} and not op.get("effects")
                and (op.get("accessScope") == "PUBLIC" or (feature.get("ownership") or {}).get("scope") == "PUBLIC"))


def _public_lookup_sibling(feature):
    """같은 기능이 선언한 공개 조회 수단 (접수번호 조회 등)."""
    return any(
        isinstance(op, dict)
        and op.get("authRequired") is False
        and op.get("accessScope") == "PUBLIC"
        and str(op.get("method")).upper() in {"GET", "HEAD"}
        for op in feature.get("apiContract") or []
    )


def public_write_contract(feature, endpoint):
    """비회원 접수(게스트 제출) 생성만 허용한다. 무심한 공개 쓰기는 계속 막는다.

    배달·수리 접수나 비회원 예약처럼 제출자가 계정을 갖지 않는 흐름이 실제로
    존재하고 draft 계약은 accessScope로 이를 표현할 수 있다. 다만 공개 쓰기는
    보안에 민감하므로 아래를 모두 만족할 때만 인정한다.

    - operation에 accessScope=PUBLIC이 **명시**되어 있다
      (feature.ownership.scope의 상속으로는 인정하지 않는다)
    - 생성(POST)이다 — 기존 자원의 수정·삭제는 공개로 열지 않는다
    - effects가 insert/create뿐이다 — 기존 행을 바꾸지 않는다
    - 같은 기능이 공개 조회 수단을 함께 선언했다
      (제출만 받고 확인할 길이 없는 반쪽 흐름을 거른다)
    """
    op = declared_operation(feature, endpoint)
    if not op or op.get("authRequired") is not False:
        return False
    if op.get("accessScope") != "PUBLIC" or str(op.get("method")).upper() != "POST":
        return False
    effects = op.get("effects") or []
    if not all(isinstance(item, dict) and item.get("kind") in {"insert", "create"} for item in effects):
        return False
    return _public_lookup_sibling(feature)


def operation_primary_table(operation, fallback=None):
    if operation.get("primaryTable"):
        return operation["primaryTable"]
    targets = {item.get("table") for item in operation.get("effects") or [] if item.get("table")}
    if not targets:
        targets = {item.get("table") for item in operation.get("fieldMappings") or []
                   if item.get("table") and item.get("source") not in {"auth", "header", "cookie"}}
    if len(targets) == 1:
        return next(iter(targets))
    return fallback if fallback in targets or not targets else None


def normalize_generated_contracts(features):
    """Complete only declared metadata; never fabricate missing business fields."""
    from phase2.operation_contract import _field_schema

    fields = {(c.get("table") or f.get("dbContract", {}).get("primaryTable"), c.get("name")): c
              for f in features for c in f.get("dbContract", {}).get("fields") or []}
    creation_bounds = {}
    for feature in features:
        for op in feature.get("apiContract") or []:
            if op.get("method") != "POST":
                continue
            for mapping in op.get("fieldMappings") or []:
                if mapping.get("source") not in {"body", "request"} or mapping.get("transform"):
                    continue
                schema = _field_schema(op.get("requestSchema"), mapping.get("field"))
                if schema:
                    creation_bounds.setdefault((mapping.get("table"), mapping.get("column")), []).append(schema)
    for feature in features:
        for op in feature.get("apiContract") or []:
            op.setdefault("accessScope", "PUBLIC" if not op.get("authRequired") else
                          {"USER": "OWNER", "SHARED": "PARTICIPANT"}.get((feature.get("ownership") or {}).get("scope"), "AUTHENTICATED"))
            primary = operation_primary_table(op, feature.get("dbContract", {}).get("primaryTable"))
            if primary:
                op["primaryTable"] = primary
            for mapping in op.get("fieldMappings") or []:
                if mapping.get("source") not in {"body", "request"} or mapping.get("transform"):
                    continue
                schema = _field_schema(op.get("requestSchema"), mapping.get("field"))
                key = mapping.get("table"), mapping.get("column")
                field = fields.get(key, {})
                if not schema:
                    continue
                length = re.fullmatch(r"(?:VAR)?CHAR\s*\(\s*(\d+)\s*\)", str(field.get("type", "")).upper())
                if length and schema.get("type") == "string":
                    schema["maxLength"] = min(schema.get("maxLength", int(length[1])), int(length[1]))
                if op.get("method") == "PATCH":
                    # A PATCH changes field presence, not the valid values of a supplied field.
                    for name in ("minLength", "minimum", "maximum"):
                        values = [s[name] for s in creation_bounds.get(key, []) if name in s]
                        if values and name not in schema:
                            schema[name] = max(values) if name == "maximum" else min(values)
                for name in ("minimum", "maximum", "enum"):
                    if field.get(name) is not None:
                        schema.setdefault(name, deepcopy(field[name]))
            request = op.get("requestSchema") or {}
            if op.get("method") == "PATCH" and request.get("type") == "object" and request.get("properties"):
                request.setdefault("minProperties", 1)
            response = op.get("responseSchema") or {}
            properties = response.get("properties", {})
            arrays = {name for name, value in properties.items() if value.get("type") == "array"}
            page_fields = {"total", "totalCount", "count", "hasMore", "nextCursor", "page", "pageSize", "limit", "offset", "pagination", "meta"}
            collection = response.get("type") == "array" or bool(arrays and set(properties) <= arrays | page_fields)
            if op.get("method") == "GET" and collection:
                parameters = op.setdefault("parameters", [])
                names = {p.get("name") for p in parameters if p.get("in") == "query"}
                if not names.intersection({"limit", "pageSize", "size", "take", "cursor", "page", "offset"}):
                    parameters.extend([
                        {"name": "limit", "in": "query", "required": False, "description": "한 번에 조회할 개수 (최대 100)",
                         "schema": {"type": "integer", "minimum": 1, "maximum": 100, "default": 20}},
                        {"name": "offset", "in": "query", "required": False, "description": "건너뛸 개수; 다음 조회는 offset+limit",
                         "schema": {"type": "integer", "minimum": 0, "default": 0}},
                    ])
                    op["pagination"] = {"strategy": "offset", "defaultLimit": 20, "maxLimit": 100,
                                        "limitParameter": "limit", "offsetParameter": "offset"}
                    op["transactionRules"] = (op.get("transactionRules") or "읽기 전용.") + " 기존 필터·소유권·정렬을 유지하고 PK 보조 정렬로 순서를 고정하며 limit/offset을 적용한다."


def operation_quality_issues(operation):
    from phase2.operation_contract import _field_schema

    issues = []
    scope = operation.get("accessScope")
    if scope and (scope == "PUBLIC") == (operation.get("authRequired") is True):
        issues.append("API_ACCESS_CONTRACT_CONFLICT: accessScope/authRequired")
    for key, schema_key in (("requiredInputs", "requestSchema"), ("requiredOutputs", "responseSchema")):
        for field in operation.get(key) or []:
            found = _field_schema(operation.get(schema_key), field)
            if key == "requiredInputs" and found is None:
                location, separator, name = field.partition(".")
                if separator and location in {"body", "request"}:
                    found = _field_schema(operation.get("requestSchema"), name)
                else:
                    found = next((p.get("schema") or {"type": p.get("type")} for p in operation.get("parameters") or []
                                  if p.get("name") == (name if separator else field)
                                  and (not separator or p.get("in") == location)), None)
                    policy = operation.get("sessionPolicy") or {}
                    actual = name if separator else field
                    # Non-auth features carry no sessionPolicy of their own; an
                    # authenticated route still receives the bearer header then.
                    credential = (actual.lower() == "authorization" and policy.get("transport", "bearer") == "bearer"
                                  and (not separator or location == "header")) or (
                        actual == policy.get("cookieName") and policy.get("transport") == "cookie"
                        and (not separator or location == "cookie"))
                    actor = any(m.get("source") == "auth" and m.get("field") == actual for m in operation.get("fieldMappings") or [])
                    if found is None and operation.get("authRequired") is True and (credential or actor and (not separator or location == "auth")):
                        found = {"type": "string"}
            if found is None:
                issues.append(f"API_ACCEPTANCE_FIELD_MISSING: {key}.{field}")
    actual = {effect.get("table") for effect in operation.get("effects") or []}
    for table in operation.get("requiredEffectTables") or []:
        if table not in actual:
            issues.append(f"API_ACCEPTANCE_EFFECT_MISSING: {table}")
    return issues


def database_checks(features):
    result = {}
    for feature in features:
        for field in feature.get("dbContract", {}).get("fields") or []:
            table = field.get("table") or feature.get("dbContract", {}).get("primaryTable")
            name = field.get("name", "")
            if not table or not re.fullmatch(r"[a-z][a-z0-9_]*", name):
                continue
            predicates = []
            for key, sign in (("minimum", ">="), ("maximum", "<=")):
                value = field.get(key)
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    predicates.append(f"{name} {sign} {value}")
            values = field.get("enum")
            if values:
                literals = ["'" + value.replace("'", "''") + "'" if isinstance(value, str) else
                            str(value).upper() if isinstance(value, bool) else str(value) for value in values if value is not None]
                if literals:
                    predicates.append(f"{name} IN ({', '.join(literals)})")
            checks = result.setdefault(table, [])
            for predicate in predicates:
                check = f"CHECK ({predicate})"
                if check not in checks:
                    checks.append(check)
    return result


def volatile_index(statement):
    # Mask quoted values: status='now()' is a constant, not a volatile call.
    sql = re.sub(r"'(?:''|[^'])*'", "''", str(statement))
    return bool(re.search(r"\b(?:now|clock_timestamp|statement_timestamp|transaction_timestamp|timeofday|random|nextval|setval|gen_random_uuid)\s*\(|\b(?:current_timestamp|current_date|current_time|localtimestamp|localtime)\b", sql, re.IGNORECASE))


def index_issues(tables):
    return [f"DB_INDEX_VOLATILE_EXPRESSION: {table.get('name')}: {index}" for table in tables
            for index in table.get("indexes") or [] if volatile_index(index)]
