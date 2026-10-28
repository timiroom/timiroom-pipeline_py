"""Explicit workflow policies shared by draft generation and cross-document QA."""

from copy import deepcopy
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

WORKFLOW_GENERATION_RULES = """
[업무 계약 완결성: 설명과 실제 API/DB가 같은 정책을 소비]
- sessionPolicy는 strategy/sessionTable뿐 아니라 transport(bearer 또는 cookie), tokenFormat(opaque 또는 JWT), tokenField/cookieName, expiresInSeconds를 확정하세요. bearer_session은 기본적으로 불투명 토큰입니다. JWT가 필요할 때만 tokenFormat=JWT를 선언하세요.
- Bearer 로그인·가입·갱신 중 세션을 발급하는 응답에는 tokenField 이름의 필수 string을 넣고 토큰 원문 반환/해시 저장 책임을 구분하세요. cookie 방식은 responseHeaders의 Set-Cookie 설명과 HttpOnly/Secure/SameSite/CSRF 정책을 명시하세요. 쿠키와 Bearer 설명을 혼용하지 마세요.
- 일반 가입은 sessionPolicy.defaultRole로 제한하세요. 사용되는 다른 역할마다 roleProvisioning에 role/source(seed,invite,external)/description을 선언하세요. 운영자 공급이 없다면 제한된 초기 seed 절차를 정하고 임의 권한 선택 가입은 금지하세요. invite 방식은 실제 method/path의 operation이 있어야 합니다.
- SYSTEM은 데이터 소유 범위이지 공개 접근이 아닙니다. ownership.access에는 인증/운영 역할을 보존하고 각 비공개 operation.allowedRoles를 작성하세요. 역할별 거절 403, 충돌 409 등 실제 실패 결과를 API에 포함하세요.
- UI에서 자원을 선택·관리해야 하면 기존 기능 안에 목록/상세 GET을 포함하고 readDependencies에 필요한 GET 경로와 responseField/purpose를 선언하세요. 생성 응답의 ID를 영구히 기억한다고 가정하지 마세요. path의 부모 ID나 body의 FK를 고르는 조회 진입점도 포함하세요. 새 기능 카드를 늘리지 말고 기존 기능에 operation을 추가하세요.
- 상태 컬럼은 statePolicies에 table/column/initial/states를 선언하고 DB default와 일치시키세요. 모든 상태는 초기 상태 또는 실제 operation의 state_transition effect로 도달해야 합니다. 외부 공급 상태만 externalStates에 공급 주체와 근거를 적으세요. 승인·확정 작업이 없으면 해당 상태를 만들지 말고 자동 완료의 초기 상태로 통일하세요.
- BOOLEAN 활성 플래그를 구간의 stateColumn으로 사용하면 statePolicies.initial/states와 intervalPolicies.blockingStates에도 실제 JSON boolean true/false를 사용하세요. 문자열 "true"/"false"와 섞지 마세요. 해당 필드에 매핑된 boolean 요청과 update effect로 활성/비활성 전이를 정의할 수 있습니다.
- stateTransitions.from/to와 effects.from/to도 해당 statePolicies.states의 실제 타입과 값을 그대로 사용하세요. boolean 컬럼의 전이를 active/inactive 같은 새 문자열 이름으로 적지 마세요.
- expires_at/revoked_at/deleted_at 같은 시각 컬럼에 active/revoked 문자열 enum을 붙이지 마세요. 만료·폐기는 시각/null과 명시된 sessionPolicy로 표현합니다. enum 상태가 실제 필요한 경우에만 별도의 문자열 상태 컬럼과 DB default를 선언하세요. 같은 물리 테이블의 상태 정책은 공통 계약을 재사용하고 기능별로 다시 만들지 마세요.
- stateTransitions에 약속한 모든 from/to는 같은 기능의 실제 operation effect에 작성하세요. 여러 상태에서 취소할 수 있으면 각 from 상태의 effect를 명시하세요. requirements와 상태 기계에 다른 허용 상태를 쓰지 마세요.
- 예약·배정 등 시간 구간을 점유하는 서비스만 intervalPolicies를 선언하세요. table/resourceColumn/startColumn/endColumn/stateColumn/blockingStates, availabilityTable과 availabilityResourceColumn/availabilityStartColumn/availabilityEndColumn, lockTable/lockColumn, timeZone/mutationPolicy를 확정하세요. blocksOnCreate=true이면 초기 상태도 blockingStates에 포함합니다. 다른 도메인에 예약 정책을 강제하지 마세요.
- 구간은 [start,end)로 비교하며 기존.start < 요청.end AND 기존.end > 요청.start인 경우 충돌입니다. 같은 자원 잠금·검사·쓰기를 단일 트랜잭션으로 묶으세요. 빈 시간 조회도 같은 blockingStates로 점유 구간을 제외하세요. 시간대의 수정/삭제와 기존 점유 보호 조건을 mutationPolicy로 선언하세요.
- 슬롯 직접 참조 방식은 slotReferenceColumn과 실제 FK를 함께 선언하세요. 직접 참조가 없으면 장비/자원 키와 구간 포함 관계로 예약 연관성을 판정한다고 명시하세요. 존재하지 않는 슬롯 FK로 삭제 조건을 설명하지 마세요.
- 단순 최신 상태 조회와 변경 이력 조회를 구분하세요. 이력을 약속할 때만 이력 데이터/operation/응답을 정의하고 단순 updated_at으로 과거 내역을 제공한다고 쓰지 마세요.
"""


class PolicyModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class RoleProvision(PolicyModel):
    role: str = Field(min_length=1)
    source: Literal["seed", "invite", "external"]
    description: str = Field(min_length=1)
    method: Literal["POST", "PUT", "PATCH"] | None = None
    path: str | None = None


class SessionPolicy(PolicyModel):
    strategy: Literal["bearer_session", "refresh_token"]
    sessionTable: str
    transport: Literal["bearer", "cookie"] = "bearer"
    tokenFormat: Literal["opaque", "JWT"] = "opaque"
    tokenField: str = "accessToken"
    cookieName: str = "session"
    secure: bool = True
    httpOnly: bool = True
    sameSite: Literal["Strict", "Lax", "None"] = "Lax"
    csrfProtection: str = ""
    expiresInSeconds: int = Field(default=3600, gt=0)
    defaultRole: str = "user"
    roleProvisioning: list[RoleProvision] = Field(default_factory=list)


class ReadDependency(PolicyModel):
    method: Literal["GET"] = "GET"
    path: str = Field(pattern=r"^/api/v1/")
    responseField: str = ""
    purpose: str = Field(min_length=1)
    roles: list[str] = Field(default_factory=list)


class StatePolicy(PolicyModel):
    table: str
    column: str
    initial: str | bool
    states: list[str | bool] = Field(min_length=1)
    externalStates: dict[str, str] = Field(default_factory=dict)


class IntervalPolicy(PolicyModel):
    table: str
    resourceColumn: str
    startColumn: str
    endColumn: str
    stateColumn: str
    blockingStates: list[str | bool] = Field(min_length=1)
    blocksOnCreate: bool = True
    availabilityTable: str
    availabilityResourceColumn: str
    availabilityStartColumn: str
    availabilityEndColumn: str
    slotReferenceColumn: str | None = None
    boundary: Literal["half_open"] = "half_open"
    timeZone: str = Field(min_length=1)
    lockTable: str
    lockColumn: str
    mutationPolicy: str = Field(min_length=1)


def authentication_contract(registry):
    policies = [feature["sessionPolicy"] for feature in registry
                if isinstance(feature.get("sessionPolicy"), dict) and feature["sessionPolicy"]]
    if not policies:
        return {"authentication": "인증 여부와 전달 방식은 각 endpoint의 명시적 계약을 따른다."}
    policy = deepcopy(policies[0])
    if policy.get("transport") == "cookie":
        description = f"세션 쿠키 {policy.get('cookieName', 'session')} 기반 인증. 로그인 응답의 Set-Cookie를 사용한다."
    else:
        token_format = policy.get("tokenFormat") or ("JWT" if policy.get("strategy") == "refresh_token" else "opaque")
        description = f"{token_format} Bearer 인증. 로그인 응답의 {policy.get('tokenField', 'accessToken')}을 Authorization: Bearer <token>으로 전송한다."
    return {"authentication": description, "sessionPolicy": policy}


def workflow_metadata(registry):
    return [{"featureId": feature["featureId"], **{key: deepcopy(feature[key]) for key in
             ("statePolicies", "intervalPolicies", "readDependencies") if feature.get(key)}}
            for feature in registry if any(feature.get(key) for key in ("statePolicies", "intervalPolicies", "readDependencies"))]


def bind_workflow_errors(operation, registry):
    import re

    required = {}
    if operation.get("authRequired") and operation.get("allowedRoles"):
        required[403] = "명시된 역할의 접근 권한이 없음"
    occupied_tables = {policy["table"] for feature in registry for policy in _valid_policies(feature, "intervalPolicies", IntervalPolicy)}
    if any(effect.get("kind") == "state_transition" or effect.get("table") in occupied_tables
           for effect in operation.get("effects") or []) and operation.get("method") in {"POST", "PUT", "PATCH", "DELETE"}:
        required[409] = "업무 상태 또는 자원 점유 충돌"
    if required:
        if operation.get("authRequired"):
            required[401] = "인증 정보가 없거나 만료됨"
        required[500] = "서버 내부 오류"
    text = str(operation.get("errorCodes") or "")
    for status, description in required.items():
        if not re.search(rf"\b{status}\b", text):
            text = (text + ", " if text else "") + f"{status} — {description}"
    if text:
        operation["errorCodes"] = text
    return operation


def workflow_check_constraints(features):
    import re

    constraints = {}

    def add(table, columns, expression):
        if all(isinstance(name, str) and re.fullmatch(r"[a-z][a-z0-9_]*", name) for name in [table, *columns]):
            constraints.setdefault(table, set()).add(f"CHECK ({expression})")

    for feature in features:
        for policy in _valid_policies(feature, "statePolicies", StatePolicy):
            values = ", ".join(("TRUE" if value else "FALSE") if isinstance(value, bool) else "'" + value.replace("'", "''") + "'" for value in policy["states"])
            add(policy["table"], [policy["column"]], f"{policy['column']} IN ({values})")
        for policy in _valid_policies(feature, "intervalPolicies", IntervalPolicy):
            for table, start, end in ((policy["table"], policy["startColumn"], policy["endColumn"]),
                                      (policy["availabilityTable"], policy["availabilityStartColumn"], policy["availabilityEndColumn"])):
                add(table, [start, end], f"{start} < {end}")
    return {table: sorted(values) for table, values in constraints.items()}


def _valid_policies(feature, key, model):
    values = feature.get(key) or []
    for value in values if isinstance(values, list) else []:
        try:
            model.model_validate(value)
        except ValidationError:
            continue
        yield value


def _operations(features):
    return [(index, op_index, operation) for index, feature in enumerate(features)
            for op_index, operation in enumerate(feature.get("apiContract") or []) if isinstance(operation, dict)]


def _state_targets(operation, effect):
    from phase2.operation_contract import _field_schema

    if effect.get("kind") in {"set", "state_transition"} and effect.get("to") is not None:
        return {effect["to"]}
    if effect.get("kind") not in {"update", "set"}:
        return set()
    targets = set()
    for mapping in operation.get("fieldMappings") or []:
        if mapping.get("table") != effect.get("table") or mapping.get("column") != effect.get("column") or mapping.get("source") not in {"body", "request"}:
            continue
        schema = _field_schema(operation.get("requestSchema"), mapping.get("field")) or {}
        targets.update(schema.get("enum") or ({schema["const"]} if "const" in schema else {True, False} if schema.get("type") == "boolean" else []))
    return targets


def normalize_boolean_workflows(features):
    """Canonicalize only explicit true/false literals for declared SQL boolean fields."""
    result, changes = deepcopy(features), []
    flags = {(field.get("table") or feature.get("dbContract", {}).get("primaryTable"), field.get("name"))
             for feature in result for field in (feature.get("dbContract") or {}).get("fields") or []
             if str(field.get("type", "")).upper() in {"BOOL", "BOOLEAN"}}
    for index, feature in enumerate(result):
        for collection in ("statePolicies", "intervalPolicies"):
            for position, policy in enumerate(feature.get(collection) or []):
                column = policy.get("column") if collection == "statePolicies" else policy.get("stateColumn")
                if (policy.get("table"), column) not in flags:
                    continue
                for name in (("initial", "states") if collection == "statePolicies" else ("blockingStates",)):
                    old = policy.get(name)
                    def literal(value):
                        return {"true": True, "false": False}.get(value, value) if isinstance(value, str) else value
                    new = [literal(value) for value in old] if isinstance(old, list) else literal(old)
                    if old != new:
                        policy[name] = new
                        changes.append({"pointer": f"/features/{index}/{collection}/{position}/{name}", "before": old, "after": new})
    return result, changes


def normalize_state_transition_effects(features):
    """Fill a state_transition effect's missing from/to from the declared stateTransitions.

    The model often writes the transition once in stateTransitions and leaves the
    effect half-typed (e.g. cancel from several states). Patch rounds kept failing on
    API_STATE_TRANSITION_INCOMPLETE, so only a value the declarations determine
    uniquely is filled; a many-source transition becomes one effect per source.
    """
    result, changes = deepcopy(features), []

    def declared(feature):
        primary = (feature.get("dbContract") or {}).get("primaryTable")
        return [(item.get("entity") or primary, item.get("from"), item.get("to"))
                for item in feature.get("stateTransitions") or [] if isinstance(item, dict)]

    everywhere = [transition for feature in result for transition in declared(feature)]
    for index, feature in enumerate(result):
        own = declared(feature)
        for op_index, operation in enumerate(feature.get("apiContract") or []):
            if not isinstance(operation, dict) or not isinstance(operation.get("effects"), list):
                continue
            effects = []
            for effect in operation["effects"]:
                if not (isinstance(effect, dict) and effect.get("kind") == "state_transition"
                        and (effect.get("from") is None or effect.get("to") is None)):
                    effects.append(effect)
                    continue
                pairs = []
                for transitions in (own, everywhere):
                    pairs = list(dict.fromkeys(
                        (source, target) for table, source, target in transitions
                        if table == effect.get("table")
                        and effect.get("from") in {None, source} and effect.get("to") in {None, target}
                    ))
                    if pairs:
                        break
                # Several targets mean the declarations do not decide this effect.
                if not pairs or len({target for _, target in pairs}) != 1:
                    effects.append(effect)
                    continue
                expanded = [{**effect, "from": source, "to": target} for source, target in pairs]
                effects.extend(expanded)
                changes.append({"pointer": f"/features/{index}/apiContract/{op_index}/effects",
                                "before": effect, "after": expanded})
            operation["effects"] = effects
    return result, changes


def _identifier_target(operation, parameter, fields, foreign_keys):
    targets = set()
    for mapping in operation.get("fieldMappings") or []:
        if mapping.get("source") != "path" or mapping.get("field") != parameter.get("name"):
            continue
        key = (mapping.get("table"), mapping.get("column"))
        if (fields.get(key) or {}).get("primaryKey"):
            targets.add("table:" + key[0])
        targets.update("table:" + fk["references"]["table"] for fk in foreign_keys
                       if fk.get("table") == key[0] and fk.get("column") == key[1])
    if len(targets) == 1:
        return next(iter(targets))
    return "route:" + operation.get("path", "").split("{", 1)[0].rstrip("/")


def _discoverable_identifiers(endpoints, fields, foreign_keys, roles):
    from phase2.operation_contract import _field_schema, _schema_nodes

    available = set()
    reads = [op for op in endpoints if op.get("method") == "GET" and not (
        op.get("authRequired") and op.get("allowedRoles") and set(roles) - set(op["allowedRoles"])
    )]
    while True:
        expanded = set(available)
        for op in reads:
            needed = {_identifier_target(op, parameter, fields, foreign_keys)
                      for parameter in op.get("parameters") or [] if parameter.get("in") == "path"}
            if not needed <= available:
                continue
            for mapping in op.get("fieldMappings") or []:
                key = (mapping.get("table"), mapping.get("column"))
                if mapping.get("source") in {"response", "derived", "server"} and (fields.get(key) or {}).get("primaryKey") and _field_schema(
                    op.get("responseSchema"), mapping.get("field")
                ) is not None:
                    expanded.add("table:" + key[0])
            if not needed and any("id" in (node.get("properties") or {}) for node in _schema_nodes(op.get("responseSchema"))):
                expanded.add("route:" + op.get("path", ""))
        if expanded == available:
            return available
        available = expanded


def _lists_parameter(endpoints, collection, parameters, roles):
    """A role-visible GET {collection} that returns the path parameter's own field (absenceId).

    The UI can pass that value straight into {absenceId}. Only an exact "id"
    property or a PK fieldMapping counted before, so a correct list route named
    after the parameter stayed WORKFLOW_LOOKUP_MISSING through every repair.
    """
    from phase2.operation_contract import _schema_nodes

    names = {parameter.get("name") for parameter in parameters if parameter.get("name")}
    for op in endpoints:
        if op.get("method") != "GET" or op.get("path", "").rstrip("/") not in {
            collection, collection.replace("/api/v1/", "/api/v1/me/", 1)
        } or any(parameter.get("in") == "path" for parameter in op.get("parameters") or []):
            continue
        if op.get("authRequired") and op.get("allowedRoles") and set(roles) - set(op["allowedRoles"]):
            continue
        if any(names & set(node.get("properties") or {}) for node in _schema_nodes(op.get("responseSchema"))):
            return True
    return False


def workflow_contract_issues(features, endpoints=None):
    # Local imports avoid a cycle: operation contracts also invoke this checker.
    from phase2.operation_contract import _field_schema, route_key

    malformed = []
    for index, feature in enumerate(features):
        for key, model in (("statePolicies", StatePolicy), ("intervalPolicies", IntervalPolicy), ("readDependencies", ReadDependency)):
            values = feature.get(key) or []
            if not isinstance(values, list):
                malformed.append(f"WORKFLOW_POLICY_SHAPE: /features/{index}/{key}: array required")
                continue
            for position, value in enumerate(values):
                try:
                    model.model_validate(value)
                except ValidationError:
                    malformed.append(f"WORKFLOW_POLICY_SHAPE: /features/{index}/{key}/{position}: valid typed policy required")
        policy = feature.get("sessionPolicy")
        if policy is not None and not isinstance(policy, dict):
            malformed.append(f"WORKFLOW_POLICY_SHAPE: /features/{index}/sessionPolicy: object required")
        if isinstance(policy, dict) and policy.get("transport"):
            try:
                SessionPolicy.model_validate(policy)
            except ValidationError:
                malformed.append(f"WORKFLOW_POLICY_SHAPE: /features/{index}/sessionPolicy: valid transport policy required")
    if malformed:
        return malformed
    operations = _operations(features)
    actual = [operation for _, _, operation in operations] if endpoints is None else endpoints
    routes = {route_key(operation): operation for operation in actual}
    fields = {(field.get("table") or feature.get("dbContract", {}).get("primaryTable"), field.get("name")): field
              for feature in features for field in (feature.get("dbContract") or {}).get("fields") or []}
    foreign_keys = [fk for feature in features for fk in (feature.get("dbContract") or {}).get("foreignKeys") or []
                    if isinstance(fk.get("references"), dict) and fk["references"].get("table")]
    issues = []

    def issue(code, index, location, message):
        issues.append(f"{code}: /features/{index}/{location}: {message}")

    policies = {}
    for index, feature in enumerate(features):
        for policy in feature.get("statePolicies") or []:
            key = (policy["table"], policy["column"])
            previous = policies.get(key)
            signature = {**StatePolicy.model_validate(policy).model_dump(), "states": sorted(policy["states"], key=repr)}
            if previous and {**StatePolicy.model_validate(previous[1]).model_dump(), "states": sorted(previous[1]["states"], key=repr)} != signature:
                for owner in {previous[0], index}:
                    issue("WORKFLOW_STATE_CONFLICT", owner, "statePolicies", f"{key}: one shared state policy required")
            policies[key] = (index, policy)
            column = fields.get(key)
            states = set(policy["states"])
            if not column or policy["initial"] not in states or column.get("default") != policy["initial"]:
                issue("WORKFLOW_STATE_INITIAL", index, "statePolicies", f"{key}: initial state must match a declared DB default and enum")
            boolean_states = all(isinstance(value, bool) for value in states)
            external = {({"true": True, "false": False}.get(state, state) if boolean_states else state)
                        for state, reason in policy.get("externalStates", {}).items() if reason.strip()}
            reached = {policy["initial"], *external}
            # A declared account seed is supply evidence, not an application transition.
            for account_feature in features:
                session = account_feature.get("sessionPolicy") or {}
                principals = {fk["references"]["table"] for fk in foreign_keys
                              if fk.get("table") == session.get("sessionTable")}
                if key[1] != "role" or key[0] not in principals or session.get("defaultRole") != policy["initial"]:
                    continue
                reached.update(supply["role"] for supply in session.get("roleProvisioning") or []
                               if supply.get("description", "").strip() and
                               (supply.get("source") in {"seed", "external"} or
                                supply.get("source") == "invite" and route_key(supply) in routes))
            edges = []
            for _, _, op in operations:
                for effect in op.get("effects") or []:
                    if effect.get("table") != key[0] or effect.get("column") != key[1]:
                        continue
                    sources = {effect["from"]} if effect.get("from") is not None else states
                    edges.extend((source, target) for source in sources for target in _state_targets(op, effect))
            unknown = ({value for edge in edges for value in edge if value is not None} | reached) - states
            if unknown:
                issue("WORKFLOW_STATE_UNKNOWN", index, "statePolicies", f"{key}: undeclared states {sorted(unknown, key=repr)}")
            for owner, position, operation in operations:
                for mapping in operation.get("fieldMappings") or []:
                    if mapping.get("table") != key[0] or mapping.get("column") != key[1]:
                        continue
                    value = _field_schema(operation.get("responseSchema"), mapping.get("field")) or {}
                    if set(value.get("enum") or []) - states:
                        issue("WORKFLOW_STATE_ENUM_CONFLICT", owner, f"apiContract/{position}", f"{key}: response enum must use the declared state policy")
            while True:
                expanded = reached | {target for source, target in edges if source in reached}
                if expanded == reached:
                    break
                reached = expanded
            if states - reached:
                issue("WORKFLOW_STATE_UNREACHABLE", index, "statePolicies", f"{key}: no operation or explicit external supply for {sorted(states - reached, key=repr)}")
                for owner, position, operation in operations:
                    schemas = [_field_schema(operation.get("responseSchema"), mapping.get("field")) or {}
                               for mapping in operation.get("fieldMappings") or []
                               if mapping.get("table") == key[0] and mapping.get("column") == key[1]]
                    if (features[owner].get("dbContract") or {}).get("primaryTable") == key[0]:
                        schemas.extend(parameter.get("schema") or {} for parameter in operation.get("parameters") or []
                                       if parameter.get("name") == key[1])
                    if any(set(schema.get("enum") or []) & (states - reached) for schema in schemas):
                        issue("WORKFLOW_STATE_ENUM_UNREACHABLE", owner, f"apiContract/{position}", f"{key}: operation shares an unreachable state enum")

    for index, feature in enumerate(features):
        if feature.get("workflowContractVersion") == 1:
            primary = (feature.get("dbContract") or {}).get("primaryTable")
            declared_tables = {table for table, _ in policies}
            if any(table == primary and column in {"status", "state"} and field.get("default") is not None
                   for (table, column), field in fields.items()) and primary not in declared_tables:
                issue("WORKFLOW_STATE_POLICY_REQUIRED", index, "statePolicies", "declare the primary resource's initial state and reachable states")
            if (feature.get("applicability") or {}).get("ui"):
                for _, _, operation in [entry for entry in operations if entry[0] == index]:
                    available = _discoverable_identifiers(actual, fields, foreign_keys, operation.get("allowedRoles") or feature.get("roles") or [])
                    path = operation.get("path", "")
                    identifier_parameters = [parameter for parameter in operation.get("parameters") or []
                                             if parameter.get("in") == "path" and (
                                                 (parameter.get("schema") or {}).get("format") == "uuid" or
                                                 (parameter.get("schema") or {}).get("type") == "integer")]
                    if "{" not in path or not identifier_parameters:
                        continue
                    collection = path.split("{", 1)[0].rstrip("/")
                    if collection.count("/") != 3:
                        continue
                    targets = {_identifier_target(operation, parameter, fields, foreign_keys) for parameter in identifier_parameters}
                    caller_roles = operation.get("allowedRoles") or feature.get("roles") or []
                    if _lists_parameter(actual, collection, identifier_parameters, caller_roles):
                        continue
                    if not targets <= available and "route:" + collection not in available and "route:" + collection.replace("/api/v1/", "/api/v1/me/", 1) not in available:
                        # Name the roles that must see the list: a lookup added for
                        # another role (VOLUNTEER only) never satisfied a manager action.
                        visible = f" visible to roles {sorted(caller_roles)}" if caller_roles else ""
                        existing = [op for op in actual if op.get("method") == "GET" and op.get("path", "").rstrip("/") == collection]
                        hint = (f"; the existing GET {collection} must allow those roles (extend its allowedRoles) and return the identifier"
                                if existing else "")
                        issue("WORKFLOW_LOOKUP_MISSING", index, "readDependencies", f"GET {collection}: UI needs an identifier discovery route{visible} before addressing this resource; add this GET operation to apiContract (a readDependencies entry alone does not create it){hint}")
        ownership = feature.get("ownership") or {}
        if ownership.get("scope") == "SYSTEM" and any(op.get("authRequired") for op in feature.get("apiContract") or []) and any(
            text in ownership.get("access", "").casefold() for text in ("인증 없이", "unauthenticated", "public access")
        ):
            issue("AUTH_ACCESS_POLICY_CONFLICT", index, "ownership", "SYSTEM ownership is not unauthenticated public access")
        policy = feature.get("sessionPolicy") or {}
        if policy.get("transport"):
            roles = {role for item in features if any(op.get("authRequired") for op in item.get("apiContract") or [])
                     for role in item.get("roles") or []} - {policy.get("defaultRole")}
            supplied = {entry.get("role") for entry in policy.get("roleProvisioning") or [] if entry.get("description")}
            if roles - supplied:
                issue("AUTH_ROLE_PROVISION_MISSING", index, "sessionPolicy", f"explicit seed/invite/external supply required for roles {sorted(roles - supplied)}")
            for supply in policy.get("roleProvisioning") or []:
                if supply.get("source") == "invite" and route_key(supply) not in routes:
                    issue("WORKFLOW_LOOKUP_MISSING", index, "sessionPolicy", f"{supply.get('method')} {supply.get('path')}: role invitation endpoint required")
            if policy["transport"] == "cookie" and (not policy.get("csrfProtection") or not policy.get("httpOnly") or
                                                       (policy.get("sameSite") == "None" and not policy.get("secure"))):
                issue("AUTH_COOKIE_POLICY_REQUIRED", index, "sessionPolicy", "cookie transport requires HttpOnly, explicit CSRF protection and Secure for SameSite=None")
            for op_index, operation in enumerate(feature.get("apiContract") or []):
                if not any(effect.get("table") == policy.get("sessionTable") and effect.get("kind") in {"insert", "create"}
                           for effect in operation.get("effects") or []):
                    continue
                endpoint = routes.get(route_key(operation), operation)
                if operation.get("authRequired") is False:
                    role_input = (endpoint.get("requestSchema") or {}).get("properties", {}).get("role")
                    if role_input and role_input.get("const") != policy.get("defaultRole") and role_input.get("enum") != [policy.get("defaultRole")]:
                        issue("AUTH_SELF_ASSIGNMENT_FORBIDDEN", index, f"apiContract/{op_index}", "public session entry must not accept arbitrary privileged roles")
                if policy["transport"] == "bearer":
                    name = policy.get("tokenField", "accessToken")
                    schema = endpoint.get("responseSchema") or {}
                    token = _field_schema(schema, name)
                    if not token or token.get("type") != "string" or name not in (schema.get("required") or []):
                        issue("AUTH_TOKEN_DELIVERY_MISSING", index, f"apiContract/{op_index}", f"required response string {name} must deliver the issued Bearer token")
                elif not any(name.casefold() == "set-cookie" for name in endpoint.get("responseHeaders") or {}) and not any(
                    str(status).startswith("2") and any(name.casefold() == "set-cookie" for name in response.get("headers") or {})
                    for status, response in (endpoint.get("responses") or {}).items()
                ):
                    issue("AUTH_COOKIE_DELIVERY_MISSING", index, f"apiContract/{op_index}", "Set-Cookie response header required")

        for dependency in feature.get("readDependencies") or []:
            target = routes.get(route_key(dependency))
            consumers = [op for op in feature.get("apiContract") or []
                         if str(op.get("path", "")).startswith(dependency["path"].rstrip("/") + "/{")]
            consumer_roles = dependency.get("roles") or sorted({role for op in consumers for role in op.get("allowedRoles") or []}) or feature.get("roles") or []
            if target is None:
                issue("WORKFLOW_LOOKUP_MISSING", index, "readDependencies", f"GET {dependency['path']}: declared read has no operation; add this GET operation to apiContract returning {dependency.get('responseField') or 'the identifier'} ({dependency['purpose']})")
            elif dependency.get("responseField") and _field_schema(target.get("responseSchema"), dependency["responseField"]) is None:
                issue("WORKFLOW_LOOKUP_OUTPUT", index, "readDependencies", f"GET {dependency['path']}: response field {dependency['responseField']} required")
            elif target.get("authRequired") and target.get("allowedRoles") and set(consumer_roles) - set(target["allowedRoles"]):
                for owner, position, operation in operations:
                    if route_key(operation) == route_key(dependency):
                        issue("AUTH_LOOKUP_ACCESS_DENIED", owner, f"apiContract/{position}", f"GET {dependency['path']}: lookup must be accessible to dependent feature {feature.get('featureId')} roles")

        for transition in feature.get("stateTransitions") or []:
            if not isinstance(transition, dict):
                continue
            entity = transition.get("entity") or (feature.get("dbContract") or {}).get("primaryTable")
            applicable = [state for (table, _), (_, state) in policies.items() if table == entity]
            if not applicable:
                continue
            if not any(transition.get("from") in state["states"] and transition.get("to") in state["states"] for state in applicable):
                issue("WORKFLOW_TRANSITION_STATE_UNKNOWN", index, "stateTransitions", f"{transition}: from/to must use actual declared states {[(state['column'], state['states']) for state in applicable]}; keep the promised transition but correct its typed values")
                continue
            if not any(effect.get("table") == entity and effect.get("column") == state["column"]
                       and transition.get("from") in state["states"] and transition.get("to") in state["states"]
                       and effect.get("from") in {None, transition.get("from")}
                       and transition.get("to") in _state_targets(op, effect)
                       for state in applicable for _, _, op in operations for effect in op.get("effects") or []):
                issue("WORKFLOW_TRANSITION_MISSING", index, "stateTransitions", f"{transition}: matching operation effect required")

        for interval in feature.get("intervalPolicies") or []:
            checks = [
                (interval["table"], [interval[key] for key in ("resourceColumn", "startColumn", "endColumn", "stateColumn")]),
                (interval["availabilityTable"], [interval[key] for key in ("availabilityResourceColumn", "availabilityStartColumn", "availabilityEndColumn")]),
                (interval["lockTable"], [interval["lockColumn"]]),
            ]
            for table, columns in checks:
                if any((table, column) not in fields for column in columns):
                    issue("WORKFLOW_INTERVAL_DB_REFERENCE", index, "intervalPolicies", f"{table}: explicit interval/lock columns must exist")
            state = policies.get((interval["table"], interval["stateColumn"]), (None, {}))[1]
            blocking = set(interval["blockingStates"])
            if not state or not blocking <= set(state.get("states") or []):
                issue("WORKFLOW_INTERVAL_STATE", index, "intervalPolicies", "blockingStates must belong to the declared state policy")
            elif interval.get("blocksOnCreate", True) and state["initial"] not in blocking:
                issue("WORKFLOW_INTERVAL_INITIAL_UNPROTECTED", index, "intervalPolicies", "initial state must block overlapping requests when blocksOnCreate=true")
            slot = interval.get("slotReferenceColumn")
            if slot and ((interval["table"], slot) not in fields or not any(fk.get("table") == interval["table"] and fk.get("column") == slot and
                                (fk.get("references") or {}).get("table") == interval["availabilityTable"]
                                for item in features for fk in (item.get("dbContract") or {}).get("foreignKeys") or [])):
                issue("WORKFLOW_INTERVAL_SLOT_REFERENCE", index, "intervalPolicies", "slotReferenceColumn must have a real FK to the availability table; otherwise use interval containment")

    return list(dict.fromkeys(issues))
