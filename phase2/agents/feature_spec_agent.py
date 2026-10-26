"""PRD coreFeatures를 구현 가능한 기능 계약으로 확정하는 Agent."""

from __future__ import annotations

import json
import logging
import re
from copy import deepcopy
from typing import Any

from openai import AsyncOpenAI

from phase2.agent_contract import (
    feature_methods,
    normalize_api_path,
    normalize_feature_contract,
    requires_auth,
    route_contract_table,
)
from phase2.artifact_shapes import shape_issues
from phase2.contract_guard import feature_contract_issues, semantic_identifier
from phase2.feature_generation import (
    generate_document,
    generation_input,
    output_schema_prompt,
    source_cores,
)
from phase2.feature_registry import normalize_feature_registry
from phase2.json_utils import try_parse_json
from phase2.llm_runtime import LlmRuntime
from phase2.mvp_policy import apply_project_priorities, dependency_issues, select_mvp
from phase2.operation_contract import (
    bind_auth_registry_policy,
    is_public_session_entry,
    normalize_db_contract,
)
from phase2.registry_contract_guard import registry_contract_issues
from phase2.state import PipelineState
from phase2.workflow_contract import WORKFLOW_GENERATION_RULES

logger = logging.getLogger(__name__)


def _normalize_contracts(item: dict) -> dict:
    """Feature Spec 계약을 downstream Agent가 소비하는 단일 JSON 형태로 고정한다."""
    api = item.get("apiContract", item.get("api"))
    normalized_api = []
    for contract in api if isinstance(api, list) else []:
        if not isinstance(contract, dict):
            continue
        value = dict(contract)
        value["method"] = str(value.get("method") or "GET").upper()
        path = normalize_api_path(value.get("path")) if value.get("path") else ""
        value["path"] = path
        value["featureId"] = str(value.get("featureId") or item.get("featureId") or "").strip()
        value["action"] = str(value.get("action") or "manage").strip()
        if path:
            normalized_api.append(value)

    db = item.get("dbContract", item.get("db"))
    db = db if isinstance(db, dict) else {}
    tables = []
    for table in db.get("tables") if isinstance(db.get("tables"), list) else []:
        if isinstance(table, dict):
            table = table.get("name") or table.get("table")
        name = str(table or "").strip()
        generic = re.fullmatch(r"feature_?(\d+)s?", name.casefold())
        if generic:
            name = f"feature_{int(generic.group(1)):03d}_records"
        if name and name not in tables:
            tables.append(name)
    foreign_keys = []
    for fk in db.get("foreignKeys") if isinstance(db.get("foreignKeys"), list) else []:
        if isinstance(fk, dict):
            column = str(fk.get("column") or "").strip()
            references = fk.get("references") if isinstance(fk.get("references"), dict) else {}
            target = str(references.get("table") or "").strip()
            target_column = str(references.get("column") or "id").strip()
            if column and target:
                foreign_keys.append({
                    **fk,
                    "table": str(fk.get("table") or "").strip(),
                    "column": column,
                    "references": {"table": target, "column": target_column},
                })
        elif str(fk).strip():
            foreign_keys.append(str(fk).strip())
    item["apiContract"] = normalized_api
    item["api"] = normalized_api
    item["dbContract"] = {**db, "tables": tables, "foreignKeys": foreign_keys}
    item["db"] = item["dbContract"]
    return item


def _contract_domain(item: dict) -> str:
    feature_id = str(item.get("featureId") or "feature").strip().lower()
    name = str(item.get("name") or "").casefold()
    # Domain nouns take precedence over the word "registration". Otherwise
    # "sports facility and class registration" is incorrectly treated as auth.
    if any(term in name for term in ("facility", "시설", "workshop", "공방", "club", "클럽")):
        return "facilities"
    if any(term in name for term in ("class", "클래스")):
        return "classes"
    if any(term in name for term in ("login", "로그인", "register", "registration", "회원가입", "profile", "프로필")):
        return "user_profiles" if any(term in name for term in ("profile", "프로필")) else "users"
    if any(term in name for term in ("booking", "reservation", "예약")):
        return "bookings"
    if any(term in name for term in ("class", "클래스")):
        return "classes"
    if any(term in name for term in ("workshop", "공방")):
        return "workshops"
    if any(term in name for term in ("notification", "알림")):
        return "notifications"
    if any(term in name for term in ("audit", "감사", "이력")):
        return "audit_logs"
    if any(term in name for term in ("validation", "검증")):
        return "validation_rules"
    if any(term in name for term in ("search", "filter", "검색", "필터")):
        return "searches"
    db_contract = item.get("dbContract") if isinstance(item.get("dbContract"), dict) else {}
    declared_tables = []
    for raw in db_contract.get("tables") or []:
        table = raw.get("name") or raw.get("table") if isinstance(raw, dict) else raw
        table = str(table or "").strip().casefold()
        if table and not re.fullmatch(r"(?:feature_?\d+_records|supporting_features)", table):
            declared_tables.append(table)
    if declared_tables:
        return declared_tables[0]
    # 의미 없는 순번 ID를 URL에 노출하지 않는다. 구체적인 DB 계약이나 공통
    # 도메인을 찾지 못한 supporting 기능은 공유 workflow 리소스에 연결한다.
    if feature_id.startswith(("supporting.", "supporting-")):
        return "workflows"
    # PM이 의미 없는 순번 ID만 제공한 경우에도 API/DB 계약은 필요하다.
    # 숫자를 제거한 feature001s 같은 이름은 sanitize_tables와 충돌하므로
    # 안정적인 fallback 식별자를 사용하고, 의미 있는 Registry 계약이 있으면
    # 아래에서 항상 그 계약을 우선한다.
    generic_feature = re.match(r"^(feature_\d+)(?:\.|$)", feature_id)
    if generic_feature:
        return "workflows"
    parts = [re.sub(r"[^a-z0-9]+", "", part) for part in feature_id.split(".")]
    parts = [part for part in parts if part]
    domain = parts[-2] if len(parts) > 1 else (parts[0] if parts else "feature")
    if domain in {"auth", "account", "member", "membership"}:
        return "users"
    if domain in {"profile", "setting", "settings"}:
        return "user_profiles"
    if not domain.endswith("s"):
        domain += "s"
    return domain


def _contract_action(item: dict) -> list[str]:
    actions = [str(value).strip().lower() for value in item.get("actions") or [] if str(value).strip()]
    name = str(item.get("name") or "").casefold()
    inferred = []
    for terms, action in (
        (("register", "registration", "회원가입"), "register"),
        (("login", "로그인"), "login"),
        (("logout", "로그아웃"), "logout"),
        (("profile", "프로필", "수정", "edit"), "update"),
        (("search", "detail", "조회", "검색"), "read"),
        (("booking", "reservation", "예약", "신청"), "apply"),
        (("approve", "승인"), "approve"),
        (("cancel", "cancellation", "취소"), "cancel"),
        (("notification", "알림"), "notify"),
    ):
        if any(term in name for term in terms) and action not in inferred:
            inferred.append(action)
    if inferred and (not actions or actions == ["manage"]):
        return inferred
    return actions or ["manage"]


def _is_supporting_feature(name: str) -> bool:
    text = str(name or "").casefold()
    return any(term in text for term in (
        "user registration", "user login", "user logout", "회원가입", "로그인", "로그아웃",
        "profile", "프로필", "password", "비밀번호", "role", "permission", "권한",
        "validation", "검증", "notification", "알림", "audit", "감사", "이력",
    ))


def _semantic_contracts(item: dict, domain: str) -> tuple[list[dict], list[str]]:
    feature_id = str(item.get("featureId") or "").strip()
    name = str(item.get("name") or "").casefold()
    if any(term in name for term in ("facility", "시설", "workshop", "공방", "club", "클럽")) and any(term in name for term in ("class", "클래스")):
        return [
            {"method": "POST", "path": "/api/v1/clubs/{clubId}/facilities", "featureId": feature_id, "action": "create"},
            {"method": "POST", "path": "/api/v1/clubs/{clubId}/classes", "featureId": feature_id, "action": "create"},
        ], ["clubs", "facilities", "classes"]
    base = normalize_api_path(domain)
    if domain == "users":
        base = "/api/v1/auth"
    contracts = []
    for action in _contract_action(item):
        if action in {"register", "signup"}:
            method, path = "POST", "/api/v1/auth/register"
        elif action in {"login", "signin"}:
            method, path = "POST", "/api/v1/auth/login"
        elif action in {"logout", "signout"}:
            method, path = "POST", "/api/v1/auth/logout"
        elif action in {"update", "edit", "change"} and domain == "user_profiles":
            method, path = "PATCH", "/api/v1/me"
        elif action in {"read", "search", "list", "view"}:
            method, path = "GET", "/api/v1/me" if domain == "user_profiles" else base
        elif action in {"approve", "cancel", "confirm"} and domain == "bookings":
            method, path = "POST", f"{base}/{{bookingId}}/{action}"
        elif action == "notify" and domain == "notifications":
            method, path = "POST", base
        elif action == "retry" and domain == "notifications":
            method, path = "POST", f"{base}/{{notificationId}}/retry"
        elif action == "apply" and domain == "bookings":
            method, path = "POST", base
        else:
            method, path = "POST", base
        if (method, path) not in {(c["method"], c["path"]) for c in contracts}:
            contracts.append({"method": method, "path": path, "featureId": feature_id, "action": action})
    return contracts, [domain]


def _ensure_required_contracts(item: dict) -> dict:
    """Fill omitted contracts deterministically from featureId/action.

    LLM output may omit contracts even when the feature itself is present. Downstream
    agents must never receive an empty contract for a backend feature.
    """
    feature_id = str(item.get("featureId") or "").strip()
    if not feature_id:
        return item
    domain = _contract_domain(item)
    name = str(item.get("name") or "").casefold()
    parts = feature_id.lower().split(".")
    namespace = parts[-2] if len(parts) > 1 else ""
    base_path = normalize_api_path(f"auth/{parts[-1]}") if namespace == "auth" else normalize_api_path(domain)
    contracts = item.get("apiContract") if isinstance(item.get("apiContract"), list) else []
    # A source registry item can carry a stale auth contract after PM/PRD
    # repair. Do not let that contract turn an unrelated generic supporting
    # feature (for example supporting.1) into an implicit login feature.
    auth_feature = any(term in name for term in (
        "login", "signin", "로그인", "register", "회원가입", "signup",
        "logout", "로그아웃", "profile", "프로필", "password", "비밀번호",
        "refresh", "session renewal", "token renewal", "세션 갱신", "토큰 갱신",
    ))
    if not auth_feature:
        contracts = [
            contract for contract in contracts
            if not re.search(r"/auth/(?:login|signup|register|logout|refresh)(?:/|$)", str(contract.get("path") or ""), re.I)
        ]
        item["apiContract"] = contracts
    generic_contract = any(
        re.search(r"/api/v1/(?:feature[-_]\d+[-_]records|supporting[-_]features)(?:/|$)", str(c.get("path") or ""))
        for c in contracts if isinstance(c, dict)
    )
    generic_table = any(
        re.fullmatch(r"feature_\d+_records", str(t))
        for t in (item.get("dbContract") or {}).get("tables", [])
    )
    if _contract_action(item) == ["manage"]:
        # 관리형 기능도 생성/조회 두 사용자 여정을 분리해 계약을 만든다.
        item["actions"] = ["create", "read"]
    if (
        not contracts
        or (domain not in {"feature", "feature_records"} and (generic_contract or generic_table))
    ):
        contracts, semantic_tables = _semantic_contracts(item, domain)
        item["apiContract"] = contracts
        item["dbContract"] = {"tables": semantic_tables, "foreignKeys": []}
    if not item.get("apiContract"):
        generated = []
        for action in _contract_action(item):
            if action in {"create", "register", "signup", "apply", "manage"}:
                method, path = "POST", base_path
            elif action in {"read", "search", "list", "view"}:
                method, path = "GET", base_path
            elif action in {"update", "edit", "change"}:
                method, path = "PATCH", f"{base_path}/{{id}}"
            elif action in {"delete", "remove"}:
                method, path = "DELETE", f"{base_path}/{{id}}"
            elif action in {"cancel", "confirm", "approve"}:
                method, path = "POST", f"{base_path}/{{id}}/{action}"
            else:
                method, path = "POST", base_path
            generated.append({
                "method": method,
                "path": path,
                "featureId": feature_id,
                "action": action,
            })
        item["apiContract"] = generated

    # Core features are persisted user journeys. Expand their Registry contract
    # from the PRD prose so downstream API generation cannot silently omit a
    # read/update/delete action that the requirements explicitly promise.
    if item.get("source") == "prd_core" or re.fullmatch(r"feature_\d+", feature_id):
        behavior_text = " ".join([
            str(item.get("name") or ""),
            str(item.get("description") or ""),
            *[str(value) for value in item.get("requirements") or []],
        ])
        required_methods = feature_methods(behavior_text)
        action_methods = {
            "create": "POST", "register": "POST", "record": "POST", "upload": "POST",
            "read": "GET", "list": "GET", "search": "GET",
            "update": "PATCH", "edit": "PATCH", "delete": "DELETE", "remove": "DELETE",
        }
        required_methods = tuple(dict.fromkeys([
            *required_methods,
            *[action_methods[action] for action in item.get("actions") or [] if action in action_methods],
        ]))
        contracts = [
            contract for contract in item.get("apiContract") or []
            if isinstance(contract, dict)
        ]
        existing_methods = {
            str(contract.get("method") or "GET").upper()
            for contract in contracts
        }
        anchor = next(
            (contract for contract in contracts if str(contract.get("method") or "").upper() == "POST"),
            contracts[0] if contracts else None,
        )
        anchor_path = normalize_api_path((anchor or {}).get("path") or base_path)
        for method in required_methods:
            if method in existing_methods:
                continue
            path = anchor_path
            action = {
                "GET": "read",
                "POST": "create",
                "PATCH": "update",
                "DELETE": "delete",
            }.get(method, "manage")
            if method in {"PATCH", "DELETE"} and "{" not in path.rsplit("/", 1)[-1]:
                resource = path.rstrip("/").rsplit("/", 1)[-1]
                singular = resource[:-3] + "y" if resource.endswith("ies") else resource.removesuffix("s")
                words = [word for word in singular.split("-") if word]
                parameter = words[0] + "".join(word.title() for word in words[1:]) + "Id"
                path = f"{path}/{{{parameter}}}"
            contracts.append({
                "method": method,
                "path": path,
                "featureId": feature_id,
                "action": action,
            })
            existing_methods.add(method)
        item["apiContract"] = contracts

    db = item.get("dbContract") if isinstance(item.get("dbContract"), dict) else {}
    tables = db.get("tables") if isinstance(db.get("tables"), list) else []
    if not tables:
        tables = [domain]
    item["dbContract"] = {
        **db,
        "tables": tables,
        "foreignKeys": db.get("foreignKeys") if isinstance(db.get("foreignKeys"), list) else [],
    }
    refresh_declared = any(
        isinstance(contract, dict)
        and str(contract.get("method") or "GET").upper() == "POST"
        and normalize_api_path(contract.get("path")) == "/api/v1/auth/refresh"
        for contract in item.get("apiContract") or []
    )
    if refresh_declared:
        # Refresh is not a standalone HTTP detail: expiry and revocation need a
        # durable DB contract. Keep both sides in the same Registry item so the
        # parallel API/DBA agents cannot interpret the flow differently.
        owner = str((item.get("ownership") or {}).get("ownerEntity") or "users").strip()
        owner = owner if re.fullmatch(r"[a-z][a-z0-9_]*", owner) else "users"
        db_contract = item["dbContract"]
        db_contract["tables"] = list(dict.fromkeys([*db_contract["tables"], owner, "refresh_tokens"]))
        owner_column = f"{owner[:-1] if owner.endswith('s') else owner}_id"
        refresh_fk = {
            "table": "refresh_tokens",
            "column": owner_column,
            "references": {"table": owner, "column": "id"},
            "nullable": False,
        }
        existing_fks = db_contract["foreignKeys"]
        if not any(
            isinstance(fk, dict)
            and fk.get("table") == "refresh_tokens"
            and fk.get("column") == owner_column
            for fk in existing_fks
        ):
            existing_fks.append(refresh_fk)
    return _normalize_contracts(item)


def _ensure_feature_narrative(item: dict, source: dict | None = None, prd: dict | None = None) -> dict:
    """Feature Spec 설명/요구사항을 보존하고 비어 있으면 결정론적으로 보강한다."""
    source = source if isinstance(source, dict) else {}
    prd = prd if isinstance(prd, dict) else {}
    name = str(item.get("name") or source.get("name") or prd.get("name") or "기능").strip()
    description = str(item.get("description") or source.get("description") or prd.get("description") or "").strip()
    if not description:
        description = f"사용자가 {name}을 요청하면 시스템이 입력과 권한을 검증한 뒤 처리 결과를 저장하고 API 응답으로 제공한다."
    requirements = item.get("requirements")
    if isinstance(requirements, str):
        requirements = [requirements]
    if not isinstance(requirements, list):
        requirements = []
    requirements = [str(value).strip() for value in requirements if str(value).strip()]
    if prd and item.get("source") == "prd_core" and item.get("workflowContractVersion") != 1:
        inherited = prd.get("requirements") or []
        if isinstance(inherited, str):
            inherited = [inherited]
        requirements = list(dict.fromkeys([*requirements, *map(str, inherited)]))
        original_description = str(prd.get("description") or "").strip()
        if original_description and original_description not in description:
            description += " " + original_description
    if not requirements:
        inherited = prd.get("requirements") or source.get("requirements") or []
        if isinstance(inherited, str):
            inherited = [inherited]
        requirements = [str(value).strip() for value in inherited if str(value).strip()]
    if not requirements:
        requirements = [f"{name} 요청 시 입력과 권한을 검증하고 정상 결과를 API와 데이터베이스에 반영하며 실패 원인을 반환한다."]
    item["name"] = name
    item["description"] = description
    item["requirements"] = requirements
    return item


def inherit_validation_contracts(features: list[dict]) -> None:
    """Standalone preflight validation consumes its parent's input, not its actor row."""
    by_id = {item.get("featureId"): item for item in features}
    for item in features:
        if "validate" not in (item.get("actions") or []):
            continue
        parent = by_id.get(item.get("parentFeatureId"))
        if not parent:
            continue
        parent_db = normalize_db_contract(parent.get("dbContract"))
        primary = parent_db.get("primaryTable") or next((
            target for operation in parent.get("apiContract") or []
            if (target := route_contract_table(operation.get("path"), parent_db["tables"]))
        ), None)
        if not primary:
            continue
        child_db = normalize_db_contract(item.get("dbContract"))
        child_db["primaryTable"] = primary
        child_db["tables"] = list(dict.fromkeys([*child_db["tables"], primary]))
        item["dbContract"] = child_db
        source = next((operation for operation in parent.get("apiContract") or []
                       if operation.get("method") == "POST" and isinstance(operation.get("requestSchema"), dict)), {})
        for operation in item.get("apiContract") or []:
            for key in ("requestSchema", "fieldMappings", "contentType"):
                if key not in operation and key in source:
                    operation[key] = deepcopy(source[key])


_LEGACY_PROMPT = """당신은 Feature Spec 아키텍트입니다.
PRD의 coreFeatures를 모두 기능명세로 변환하고, 구현에 필수적인 supporting 기능만 추가하세요.
JSON만 출력하세요.

규칙:
- PRD coreFeatures는 모두 source='prd_core'로 유지하되 PRD/PM의 P0/P1/P2 우선순위를 그대로 보존
- supporting 기능은 핵심 기능을 실제 서비스로 운영하기 위해 필요한 일반 기능도 검토하세요.
  적용 가능한 경우 회원가입, 로그인·로그아웃, 회원정보 수정, 비밀번호 재설정,
  운영자 계정, 역할·권한 관리, 입력 검증, 상태 전이, 알림 처리, 알림 재시도,
  이력/감사 로그를 추가하세요. 모든 프로젝트에 무조건 넣지는 말고 사용자 식별,
  개인정보, 권한, 예약·주문·결제·협업 데이터 등 실제 요구사항과 연결될 때만 추가하세요.
- supporting 기능은 source='supporting', parentFeatureId, reason을 반드시 작성
- priority는 PM projectPlan의 MoSCoW 우선순위를 우선 반영하세요.
  MUST는 핵심이면 P0, supporting이면 P1, SHOULD는 P1, COULD는 P2로 매핑하고,
  projectPlan의 priorities에 없는 supporting 기능만 의존성/출시 차단/장애 영향/사용 빈도/보안·무결성 점수로 결정하세요.
- priorities.must/should/could/wont에 명시되거나 의미상 일치하는 기능을 찾아 우선순위를 연결하세요.
- WONT 또는 scope.out에 해당하는 기능은 기능명세에 추가하지 마세요.
- priorityScore는 0~13 정수: 핵심 의존성(0~3)+출시 차단(0~3)+장애 영향(0~2)+사용 영향(0~2)+보안·무결성(0~3)
- 10~13=P1, 5~9=P2, 0~4=P3
- 보안·법규·데이터 무결성은 releaseGate=true로 지정
- 각 기능에는 featureId, name, source, parentFeatureId, reason, actions, priority, priorityScore,
  releaseGate, description, requirements, apiContract, dbContract를 포함
- applicability={{"ui":true,"api":true,"db":true}}로 실제 산출물 적용 영역을 명시하세요. UI-only 기능은 api/db=false와 빈 계약을 사용하고, 무상태·외부 조회 기능은 db=false로 설정하세요. 억지 endpoint/table을 만들지 마세요.
- dependencies에는 먼저 구현되어야 하는 featureId만 작성하고, parentFeatureId 및 dependencies가 실제 기능을 참조하도록 하세요. 순환 의존성은 금지합니다.
- parentFeatureId는 기능 분류/출처 관계이며 실행 선행조건이 아닙니다. 인증은 핵심 기능의 supporting으로 분류될 수 있지만 인증 실행이 핵심 기능 완료에 의존해서는 안 됩니다. dependencies에는 실제 구현/실행에 필요한 선행 기능만 작성하세요.
- description은 사용자 상황 → 시스템 처리 → 결과를 설명하는 문장으로 반드시 작성하세요.
- requirements는 최소 1개 이상이며 역할, 선행조건, 입력, 정상 처리, 실패 조건, 결과를 포함하세요.
- PRD coreFeatures는 한 항목도 누락하지 말고 각각 하나의 feature object로 변환하세요. 각 core의 sourceFeatureId를 그대로 복사하여 원본을 추적하세요. 기능명은 명확하게 다듬어도 원본 참조는 바꾸지 마세요.
- 모든 backend 기능은 apiContract에 실제 method/path/action/featureId를 최소 하나 포함하세요. 저장·상태·이력·권한 데이터가 필요한 기능은 dbContract.tables를 최소 하나 포함하세요.
- featureId는 plants.create, watering_records.record처럼 영문 도메인.동작으로 작성하세요. feature_001, supporting.14, 순번 ID는 금지합니다. intent.*는 입력 단계의 임시 ID이므로 실제 resource/action ID로 교체하세요.
- 각 기능의 requirements에는 정상 입력과 성공 결과, 입력 오류·권한 실패, 해당 기능의 상태·트랜잭션 또는 파일 처리 조건을 서로 구분한 검증 가능한 요구사항을 작성하세요.
- 회원가입·로그인·비밀번호 재설정은 서로 다른 사용자 작업이므로 각각 별도 기능 계약과 설명·요구사항을 유지하세요.
- dbContract.foreignKeys는 문자열 설명만 쓰지 말고 반드시 {{"table":"source_table","column":"...","references":{{"table":"...","column":"id"}}}} 객체로 작성하세요. source table은 dbContract.tables 중 하나여야 하며 실제 테이블 목록에 없는 대상을 참조하지 마세요.
- API path는 반드시 /api/v1/로 시작하는 영문 소문자 REST 경로여야 하며, 같은 method/path를 중복 생성하지 마세요.
- dbContract.primaryTable은 실제 업무 대상이며 소유권 확인용 테이블과 구분하세요. fields에는 table/name/type/required/default/generated/source를, uniqueConstraints에는 table/columns를 명시하세요. source는 request/auth/server/derived 중 값의 출처입니다.
- dbContract.fields.type은 BIGINT/TEXT/UUID/TIMESTAMPTZ 같은 PostgreSQL 타입입니다. requestSchema/responseSchema의 integer/string 등 JSON Schema 타입과 혼동하지 마세요.
- 모든 FK의 원본/대상 컬럼은 전체 features의 fields 안에 선언하고 두 타입을 일치시키세요. 각 기능에서 참조 대상 테이블의 필드를 중복 정의할 필요는 없습니다. 참조 대상은 실제 PK(primaryKey=true) 또는 uniqueConstraints에 선언한 고유키여야 합니다. 반복 기록의 분류값에 임의 UNIQUE를 추가하지 말고 올바른 기준 엔터티와 참조 방향을 설계하세요.
- 물리 컬럼 정의는 읽기/쓰기 기능에 따라 바뀌지 않습니다. 여러 기능이 같은 테이블/컬럼을 반복 선언하면 type/required/default/generated/primaryKey를 모두 동일하게 복사하세요. 읽는 기능이라는 이유로 UUID PK의 default를 null로 바꾸거나 generated=false로 바꾸지 마세요. 중복이 불필요한 참조 대상 필드는 소유 기능 한 곳에서만 정의하세요.
- 인증 기능은 하나의 sessionPolicy.strategy와 sessionPolicy.sessionTable을 공유하고 별도 이름의 세션 테이블을 중복 생성하지 마세요.
- apiContract에는 requestSchema/responseSchema(JSON Schema), contentType, successStatus, fieldMappings를 작성하세요. fieldMappings는 field/table/column/source 객체 배열이고 DTO와 저장 필드가 다르면 transform도 적으세요. 계산 결과를 저장 컬럼으로 강제하지 마세요.
- fieldMappings 각 객체는 실제 스칼라 필드 하나와 컬럼 하나만 연결합니다. column에 "id,name" 같은 쉼표 목록을 쓰거나 items 객체 전체를 여러 컬럼에 연결하지 마세요. 배열 응답이면 field="items[].id", column="id", source="response"처럼 원소의 필드별 객체를 작성하세요. 필터·경로 값은 query/path, 서버 생성값은 server로 구분하세요.
- GET/DELETE/HEAD에는 requestSchema를 넣지 말고 입력은 parameters의 path/query에 작성하세요. 경로의 plantId 같은 필드에 대한 fieldMappings.source는 body가 아니라 path여야 합니다. 204 응답이면 responseSchema={{}}로 하고, 응답 객체를 반환하려면 successStatus=200 등 본문을 허용하는 코드를 사용하세요.
- parameters는 반드시 [{{"in":"path","name":"resourceId","required":true,"schema":{{"type":"integer"}}}}] 배열 형식으로 작성하세요. path/query 키를 가진 객체 형식으로 만들지 마세요.
- PRD의 수정·삭제·조회 동작을 actions와 API 계약에서 생략하지 마세요. 기능 카드 제한으로 endpoint를 삭제하지 마세요. 안내·추천 결과는 responseSchema에 구체적인 객체와 필드로 포함하세요.
- 사용자에게 반드시 반환할 값은 outputRequirements의 field/description 객체 배열에도 명시하고, 해당 field를 API responseSchema에 실제로 포함하세요. 저장 테이블에 필드가 있다는 것만으로 응답 구현이 된 것으로 보지 마세요.
- 파일은 multipart/form-data + string/binary 필드 또는 업로드 URL 발급·완료 API 흐름을 명시하세요. 메타데이터 JSON만으로 파일 업로드를 대체하지 마세요.
- 입력·소유권 검증은 부모 요구사항에 포함하세요. 별도 검증 API가 실제 제품 요구일 때만 standalone=true로 분리하고 부모 입력과 primaryTable을 사용하세요.
- 상태 변경은 states/stateTransitions와 effects의 table/column/kind/from/to를 명시하세요. 단순 FK 참조로 부모 상태 변경을 추측하지 마세요.
- states와 stateTransitions는 객체 map이 아니라 배열이어야 합니다. 예: states=["pending","completed"], stateTransitions=[{{"from":"pending","to":"completed"}}]. 여러 엔터티의 상태는 배열 안 객체의 entity 필드로 구분하세요.
- 인증은 동일한 sessionPolicy.strategy(bearer_session 또는 refresh_token)를 사용하세요. refresh_token이면 발급·갱신·회수·DB 만료/회수를 함께 정의하고 비밀번호 변경 완료 시 기존 세션과 토큰을 폐기하세요.
- Registry에 이미 apiContract/dbContract가 있으면 재해석하거나 이름을 바꾸지 말고 그대로 보존하세요. 누락된 계약만 요구사항과 PRD를 근거로 보완하세요.
- supporting 기능은 실제 핵심 기능의 실행·보안·무결성에 필요한 경우에만 추가하고, 각 supporting 기능에 parentFeatureId와 reason을 작성하세요. supporting 기능을 coreFeatures 또는 P0으로 승격하지 마세요.
- prd_core의 조회·수정·삭제는 같은 기능의 actions/apiContract/requirements에 포함하세요. 독립적인 사용자 목적이 있을 때만 supporting으로 분리하세요. 검증 코드를 별도 기능 카드로 늘리지 마세요.
- 하위 기능은 부모 기능의 featureId를 parentFeatureId로 사용하고, 부모와 다른 구체적인 featureId/name/actions/apiContract/dbContract/description/requirements를 가져야 합니다. 단순히 "기타"나 "관리"로 뭉뚱그리지 마세요.
- 전체 기능은 최대 12개이며 최소 개수를 채우기 위한 기능 생성은 금지합니다. 핵심 기능 3~5개와 실행에 필요한 supporting만 작성하세요. schema/description을 중복 복사하지 말고 설명은 간결하게, 요구사항은 검증 가능하게 작성하세요.
- 이미 의미형인 core featureId는 유지하고 intent.* 같은 임시 ID만 실제 도메인.동작으로 교체하세요. supporting의 parentFeatureId/dependencies에는 최종 의미형 ID를 사용하세요.
- applicability=false인 영역은 nonApplicabilityReasons에 영역별 사유를 작성하세요. UI-only 기능이나 외부 무상태 조회에도 임의 테이블을 생성하지 마세요.
- 회원가입·로그인·회원정보 기능은 독립적인 auth.* 또는 profile.* ID를 사용하고, 핵심 기능에 대한 parentFeatureId와 추가 이유를 명시하세요.

[PM projectPlan]
{project_plan}

[PRD]
{prd}

[PM Registry 참고]
{registry}

출력은 아래 JSON Schema의 인스턴스입니다. schema 자체를 반환하지 마세요.
이 스키마는 실제 로컬 검증 모델에서 생성되며 required/배열/객체 구분을 그대로 따르세요.
필수 필드와 실제 적용되는 선택 필드만 출력하세요. featureId와 중복인 id, 무의미한 빈 배열·객체를 반복하지 마세요.
중첩 responseSchema의 properties/items 객체를 모두 닫은 뒤 apiContract의 다음 endpoint를 작성하세요.
{output_contract}
"""


PROMPT = """당신은 Feature Spec 아키텍트입니다. 입력 PRD의 핵심 여정을 구현 가능한 MVP 기능으로 정의하세요.
출력은 contract-draft-v1 JSON 한 객체입니다. 깊은 JSON Schema 대신 평탄한 typed draft를 작성하고 코드가 계약을 조립합니다.
최상위 구조는 {{"version":"contract-draft-v1","features":[기능 객체들],"tables":[테이블 객체들],"schemas":[DTO 객체들]}} 입니다. features 배열을 닫은 뒤 같은 최상위의 tables와 schemas를 작성하세요. 기능 배열 마지막에 tables/schemas를 담은 객체를 추가하면 전체 출력이 거절됩니다.

역할·범위:
- PRD coreFeatures마다 source=prd_core 기능을 정확히 하나 만들고 sourceFeatureId를 복사하세요. 입력 추적용 sourceFeatureId를 출력 featureId로 복사하지 마세요. featureId는 기능의 영문 도메인.동작으로 새로 확정합니다. intent.*, supporting.*, feature_001, 해시 ID는 모두 금지합니다. 인증 supporting도 auth.session 같은 실제 도메인 ID를 사용하세요. parentFeatureId/dependencies는 최종 featureId끼리 연결합니다. 이미 의미형인 기존 featureId만 보존하세요.
- 설명은 사용자 상황→처리→결과를, requirements는 입력·권한·정상/실패·파일/트랜잭션 조건을 검증 가능한 문장으로 작성하세요. core와 supporting 모두 reason이 필수이며 이 기능이 사용자 여정에 필요한 이유를 한 문장으로 작성하세요.
- supporting은 핵심 여정 실행에 실제로 필요한 인증·프로필·운영 기능만 만들고 parentFeatureId/reason을 명시하세요. 회원가입·로그인 등 독립 작업을 핵심 기능으로 승격하지 마세요.
- 계정 생성·로그인·로그아웃·세션·역할 검사는 하나의 응집된 인증 supporting 기능에 포함하세요. 인증 미들웨어/권한 검사만 담당하는 auth.session을 auth.account와 별도 기능으로 만들지 마세요. 각 업무 operation의 authRequired/allowedRoles와 같은 sessionPolicy를 소비하는 것으로 표현합니다. 사용자 입력에 별도 외부 인증 여정이 명시된 경우에만 실제 API 계약이 있는 독립 기능을 검토하세요.
- 입력에 조직·팀·다중 테넌트 요구가 없으면 임의 조직/소속 모델을 추가하지 마세요. 역할 구분과 개인 소유권으로 충분한 흐름을 테넌트로 확대하지 마세요. 실제로 소속별 접근이 필요한 경우 사용자·소속 연결과 소속 생성/초기 공급 경계를 함께 정의하세요. auth에서 읽는 소속 식별자는 실제 세션·사용자·멤버십 계약에서 확인할 수 있어야 합니다.
- 전체 최대 12개이며 개수를 채우기 위한 생성은 금지합니다. 조회·수정·삭제는 같은 기능의 actions와 apiContract에 포함하고 독립 목적이 아니면 별도 카드로 분해하지 마세요.
- PM must/should/could는 핵심 P0/보조 P1/P2에 반영하고 wont/scope.out 기능은 만들지 마세요. supporting은 P0이 아닙니다. dependencies는 실제 선행조건만 작성하며 분류용 parentFeatureId와 혼동하지 마세요.
- applicability와 nonApplicabilityReasons로 UI-only/무상태 기능의 API·DB 비적용 사유를 명시하고 임의 테이블/API를 만들지 마세요.

공통 자산은 한 번 정의:
- tables는 전역 테이블 목록입니다. name과 fields의 물리 타입·PK·generated·default·제약은 한 번만 정의하세요. PostgreSQL 타입을 사용하세요.
- 단일 컬럼 UNIQUE는 fields의 unique=true로 작성합니다. 복합 UNIQUE만 uniqueConstraints에 columns 배열을 가진 객체로 작성합니다. 배열 안에 컬럼 배열을 직접 중첩하지 마세요.
- 변경 가능한 상태값을 복합 UNIQUE에 넣어 과거 취소·완료 이력의 반복 생성을 막지 마세요. 시간 구간 중복은 동일 시작/종료 값의 UNIQUE만으로 막을 수 없습니다. 활성 구간의 충돌 검증과 동일 자원 잠금을 transactionRules에 명시하고, 취소 후 재생성·재취소 및 이력 보존이 가능해야 합니다.
- features에는 dbContract 대신 dbTables(이름 배열)와 primaryTable만 작성하세요. 참조 대상 테이블도 전역 tables에 존재해야 합니다. 같은 공유 PK를 기능마다 다시 정의하지 마세요.
- foreignKeys는 원본 column/columns와 references.table/column/columns로 실제 PK 또는 고유키를 참조하세요. 외부 식별자·polymorphic ID를 이름만 보고 내부 FK로 만들지 마세요.
- sessionPolicy는 인증 기능 전체가 같은 strategy와 sessionTable을 사용해야 하며 만료·폐기 저장소도 tables에 정의하세요.
- 로그인만 선언하고 계정 생성 경로를 빠뜨리지 마세요. 사용자가 이메일·비밀번호를 쓰는 서비스이고 입력에 기존 계정·외부 인증·관리자 계정 발급이 명시되지 않았으면 같은 인증 supporting 기능 안에 회원가입 operation, 비밀번호 해시 저장, 중복 식별자 오류를 포함하세요. 외부 인증/기존 계정이 실제 입력에 명시된 경우에는 그 공급 경계를 requirements에 적고 불필요한 회원가입은 만들지 마세요.
- ownership에는 실제 USER/SHARED/PUBLIC/SYSTEM 범위와 ownerEntity/ownerKey/access를 선언하세요. 소유권을 확인하는 실제 컬럼과 FK를 tables에 정의하고 이름만 보고 user_id를 가정하지 마세요.
- ownerEntity는 논리적 단수명이 아닌 tables.name과 정확히 같은 물리 이름입니다. ownerKey는 primaryTable에서 ownerEntity를 실제로 참조하는 FK 컬럼입니다. 자신의 PK id를 다른 테이블 소유권 FK로 쓰지 마세요.
- 자식 기록이 부모 엔터티를 통해 사용자 소유권을 확인하면 ownerEntity는 실제 부모 테이블, ownerKey는 자식의 부모 FK로 선언하고 access에 부모에서 사용자까지의 검증 경로를 적으세요. 부모에만 있는 owner_id를 자식 테이블의 ownerKey로 복사하지 마세요. 기존 테이블을 읽어서 계산하는 기능도 DB가 필요하므로 db=true이며 primaryTable은 실제 조회 기준 테이블을 사용하고 불필요한 새 테이블은 만들지 마세요.

평탄한 DTO 정의:
- schemas에는 고유한 name과 nodes 배열을 작성하세요. 각 노드는 고유 id, parent(부모 id 또는 루트 null), name(객체 속성명), kind를 가집니다. JSON properties/items 객체를 직접 출력하지 마세요.
- DTO마다 parent=null인 루트는 정확히 하나입니다. object의 자식은 서로 다른 name을 가지며 required로 필수 여부를 정합니다. array에는 name=null인 자식 노드 하나만 두세요. 그 자식이 object이면 그 아래에 속성 노드를 연결하세요.
- kind=ref는 ref에 다른 schemas.name을 넣습니다. 순환 참조·고아 노드는 금지합니다. nullable/enum/format과 제약은 노드의 typed 필드로만 작성하세요. 파일 노드는 kind=string, format=binary입니다.
- apiContract에는 requestSchema/responseSchema 대신 requestType/responseType에 schemas.name을 작성하세요. 본문이 없으면 requestType=null, 204 응답이면 responseType=null입니다.
- GET/DELETE/HEAD의 입력은 parameters에 두고 requestType=null로 하세요. parameters는 in/name/typeRef/required이며 typeRef 역시 schemas.name을 참조합니다. path 파라미터는 required=true입니다.
- path는 /api/v1/ 아래 의미형 영문 REST 경로이며 method/path 계약을 모두 보존하세요. 공유 route는 모순된 DTO/인증/응답을 중복 설계하지 마세요.
- fieldMappings는 DTO의 스칼라 field 하나와 table의 실제 column 하나만 연결하세요. column에 쉼표 목록은 금지합니다. 루트 배열 응답은 field=[].id, items 속성 안의 배열은 field=items[].id입니다. DTO에 없는 items wrapper를 매핑 경로에 만들지 마세요.
- 반대로 응답이 object이고 그 안의 items가 array이면 field=[].id가 아니라 items[].id입니다. DTO 루트와 wrapper 이름을 먼저 확인하고 모든 매핑과 readDependencies.responseField에 실제 전체 경로를 쓰세요.
- source는 body/path/query/header/cookie/response/auth/server/derived로 실제 출처를 구분하고 DTO와 저장 타입이 다르면 transform을 명시하세요. 서버 생성 컬럼을 사용자 필수 입력으로 만들지 마세요.
- field는 저장 컬럼의 별칭이 아니라 실제 source의 필드 경로입니다. 요청에 status만 있으면 이력의 to_status도 field=status로 매핑하며 존재하지 않는 body.toStatus를 만들지 마세요. Bearer 원문은 source=header, field=Authorization이며 임의 sessionToken 헤더 이름을 사용하지 마세요. 쿠키는 sessionPolicy.cookieName과 같은 이름을 사용합니다.
- source=path의 field는 실제 중괄호 경로 변수명과 정확히 같아야 합니다. 경로가 {{equipmentId}}이면 field=equipmentId, 저장 column=id로 연결하며 field=id로 줄이지 마세요.
- 모든 insert/create effect의 테이블마다 필수 컬럼의 값 출처를 완성하세요. camelCase 요청과 snake_case 컬럼은 fieldMappings로 연결합니다. 예: field=startedAt, column=started_at, source=body. path/auth 입력도 매핑하고, 비밀번호 해시·토큰 해시·만료일은 서버 생성/변환 책임을 명시합니다. tables.fields.source는 실제 request/auth/server/derived 출처로 쓰세요. 서로 다른 테이블을 함께 생성하면 각 테이블 값을 모두 선언하세요.
- 파일 전송은 multipart/form-data 또는 업로드 URL 발급/완료 계약입니다. 사용자에게 필요한 안내·추천 결과를 DTO와 outputRequirements에 실제로 정의하세요.
- effects는 선언된 상태 변경만 typed 객체로 작성하고 단순 FK에서 부모 상태 변경을 추론하지 마세요. insert/create/delete 외의 update/set/revoke/expire/increment/decrement/state_transition은 실제 column이 필수입니다. PATCH는 바뀌는 컬럼별 effect를, 로그아웃은 세션 폐기 컬럼과 서버 생성 시각을 명시하세요. 상태 전이에는 from/to와 해당 전이를 수행하는 operation effect를 작성하세요. 구현하지 않는 상태 전이를 나열하지 마세요.
- 파일의 MIME 유형·최대 용량·실패 시 파일 정리 정책을 검증 가능한 요구사항으로 정하세요. 알림은 전경 알림인지 백그라운드 push인지 범위를 명시하고 필요한 읽음 처리·구독·일정 실행/중복방지 책임을 계약에 포함하세요.
- PRD/기능 actions에서 약속한 조회·수정·삭제는 실제 endpoint 계약을 모두 작성하세요. actions에 update/delete를 쓰고 PATCH/DELETE를 생략하지 마세요. outputRequirements.field는 실제 성공 응답 DTO 필드명만 작성하며 오류 안내·UI 설명은 requirements에 둡니다.
- 모든 POST/PATCH/PUT/DELETE에는 비어 있지 않은 transactionRules로 쓰기 대상·소유권 조건·실패 롤백·외부 파일 보상 경계를 명시하세요. GET에는 읽기 전용임을 명시할 수 있습니다.
- 새 사용자별 테이블은 조회·수정 API만 있고 처음 데이터를 만드는 경로가 없는 상태로 두지 마세요. 최초 생성 operation과 insert effect 또는 상위 생성 operation의 원자적 insert effect를 정의하세요. 자동 생성이면 입력값·주기·실행 주체·중복 방지를 requirements와 해당 생성 operation에 명시하고, 정적 참조 데이터만 명시적인 seed 공급 경계를 사용하세요. 입력에 없는 기존 데이터가 있다고 가정하지 마세요.
- 상태 enum에는 초기 생성 상태와 실제 operation/effect로 도달 가능한 상태만 포함하세요. 제공하지 않는 승인·확정 상태를 미래 값처럼 나열하지 마세요. 요구사항에 역할별 조회·승인·이력 확인을 약속했으면 해당 역할이 사용할 실제 operation과 접근 범위도 정의하세요.
- 부모 데이터 삭제로 외부 파일 메타데이터가 함께 삭제되면 부모 삭제 operation에도 파일 정리·보상 또는 재처리 경계를 명시하세요. 화면에서 명칭·일시·안내를 표시한다고 한 요구사항은 실제 응답 DTO 또는 명시된 조합 조회 계약으로 충족시키세요.
- 각 객체 키는 정확히 한 번만 출력하세요. dbTables/primaryTable은 features의 지정 위치에 한 번만 선언하며 테이블 정의를 반복하지 마세요.
- 기능·테이블·DTO 객체마다 줄바꿈하세요. 빈 선택 배열/기본값은 생략하세요. 마지막 테이블 객체를 닫은 뒤 tables 배열을 닫고 schemas를 시작하세요.
- required로 표시된 배열은 비어 있어도 생략하지 마세요. 모든 기능에 apiContract와 dbTables를 반드시 작성하고 api=true이면 실제 operation이 있어야 합니다. api=false인 UI 전용 기능만 apiContract=[]와 비적용 사유를 명시합니다. required 배열과 선택 배열을 혼동하지 마세요.
- 최종 출력 전에 내부적으로 기능→operation→DTO 이름 목록을 대조하고, 동일 schemas.name을 두 번 정의하지 않았는지 확인하세요. JSON을 한 줄로 압축하지 말고 각 features/tables/schemas 원소를 별도 줄에 작성하며, 다음 원소 전의 쉼표와 마지막 괄호를 확인하세요. 이 점검 설명은 출력하지 마세요.

[PM projectPlan]
{project_plan}
[PRD]
{prd}
[기존 Registry 참고: 의미형 계약은 보존]
{registry}

아래 스키마는 실제 draft 검증 모델에서 생성됐습니다. 스키마 자체가 아닌 그 인스턴스를 출력하세요.
선택 필드의 불필요한 반복은 생략하고 version/features/tables/schemas를 포함하세요.
{output_contract}
"""


PROMPT += WORKFLOW_GENERATION_RULES
_LEGACY_PROMPT += WORKFLOW_GENERATION_RULES
PROMPT += """

구체적 수용 계약:
- operation별 accessScope는 PUBLIC/AUTHENTICATED/OWNER/PARTICIPANT 중 하나로 작성하세요. 공개 조회와 작성자 전용 변경을 같은 기능 안에서 구분하고 authRequired와 일치시키세요. 비공개 프로필·거래·메시지를 공개 조회로 완화하지 마세요.
- 모든 operation에 요구한 입력 필드 경로 requiredInputs, 화면 표시값 requiredOutputs, 변경 테이블 requiredEffectTables를 구체적으로 작성하고 실제 DTO·effects와 일치시키세요. 빈 배열로 상세 요구를 숨기지 마세요. 원래 요구사항에 이름·일시·상대방·등록시각을 표시한다면 ID만 반환해서 충족시키지 마세요. requiredOutputs는 해당 operation의 실제 응답 경로입니다.
- 사진/파일을 수정한다고 약속하면 해당 수정 operation의 requestType에도 실제 파일 추가·교체·삭제 입력을 정의하고 파일 메타데이터 효과·외부 파일 보상도 선언하세요. 생성 operation만 업로드를 지원하는 것은 수정 구현이 아닙니다. multipart의 실제 필드명·배열 직렬화를 requirements에 명시하세요.
- 본인 물품 관리와 같은 화면은 공개 응답의 canEdit/isOwner 또는 본인 전용 목록으로 진입·편집 여부를 판단할 계약을 정의하세요. 공개 API에 개인 연락처를 노출하지 마세요.
- operation.primaryTable은 실제 처리 자원입니다. 한 기능에서 서로 다른 자원을 처리할 때 기능의 대표 테이블을 모든 operation에 복사하지 마세요.
- tables.fields의 숫자 minimum/maximum 및 enum을 DTO의 불변식과 일치시켜 DB CHECK로 투영할 수 있게 하세요. 모든 입력의 길이·빈 문자열·선택 필드 null 정책을 물리 저장 타입과 맞추세요.
- 목록은 limit/offset 또는 명시적 cursor/page 기반의 유한한 페이지 크기와 안정 정렬을 사용하세요. PRD uiAcceptanceCriteria, fileLifecyclePolicy, telemetryEvents의 책임을 보존하세요. 파일 정리 worker를 약속하면 영속 작업 상태의 생성·재시도 경계를 계약으로 완성하세요.
"""


def _project_priority(name: str, project_plan: dict[str, Any]) -> tuple[str | None, bool]:
    priorities = project_plan.get("priorities") if isinstance(project_plan, dict) else {}
    if not isinstance(priorities, dict):
        return None, False
    normalized = str(name or "").casefold().strip()
    for key, priority, gate in (("must", "P1", True), ("should", "P1", False), ("could", "P2", False)):
        values = priorities.get(key) or []
        if any(normalized == str(value).casefold().strip() or normalized in str(value).casefold() or str(value).casefold() in normalized for value in values):
            return priority, gate
    wont = priorities.get("wont") or []
    if any(normalized == str(value).casefold().strip() or normalized in str(value).casefold() or str(value).casefold() in normalized for value in wont):
        return "EXCLUDE", False
    return None, False


def _score_supporting(name: str, reason: str, project_plan: dict[str, Any] | None = None, *, feature: dict | None = None) -> tuple[str, int, bool]:
    mapped, gate = _project_priority(name, project_plan or {})
    if mapped == "EXCLUDE":
        return "EXCLUDE", 0, False
    if feature and any(is_public_session_entry(operation, feature) for operation in feature.get("apiContract") or []):
        return "P1", 10, True
    if mapped:
        return mapped, {"P1": 10, "P2": 6, "P3": 3}[mapped], gate
    if feature and feature.get("workflowContractVersion") == 1 and feature.get("priority") in {"P1", "P2", "P3"}:
        priority = feature["priority"]
        return priority, int(feature.get("priorityScore") or {"P1": 10, "P2": 6, "P3": 3}[priority]), priority == "P1"
    text = f"{name} {reason}"
    lowered = text.casefold()
    if any(k in lowered for k in ("login", "logout", "회원가입", "로그인", "권한", "permission", "password", "비밀번호")):
        return "P1", 10, True
    if any(k in lowered for k in ("notification", "알림", "cancellation", "취소", "audit", "감사", "status", "상태")):
        return "P2", 6, False
    score = 0
    score += 3 if any(k in text for k in ("인증", "권한", "검증", "무결성", "상태", "auth", "permission", "validation", "integrity")) else 1
    score += 3 if any(k in text for k in ("필수", "출시", "차단", "동작", "required", "release", "blocker")) else 0
    score += 2 if any(k in text for k in ("오류", "장애", "손실", "잘못", "error", "failure", "loss", "invalid")) else 0
    score += 2 if any(k in text for k in ("사용자", "등록", "조회", "user", "create", "read")) else 0
    score += 3 if any(k in text for k in ("보안", "권한", "개인정보", "무결성", "FK", "security", "privacy")) else 0
    score = min(score, 13)
    gate = any(k in text for k in ("보안", "권한", "개인정보", "무결성", "FK", "security", "privacy", "integrity"))
    release_blocker = any(k in text for k in ("필수", "출시", "차단", "동작", "required", "release", "blocker"))
    priority = "P1" if score >= 10 or (gate and release_blocker) else "P2" if score >= 5 else "P3"
    return priority, score, priority == "P1"


class FeatureSpecAgent:
    def __init__(self, client: AsyncOpenAI, model: str, runtime: LlmRuntime | None = None):
        self._client, self._model, self._runtime = client, model, runtime

    @staticmethod
    def _apply_mvp_budget(features: list[dict]) -> list[dict]:
        """Select a dependency-closed release using the shared PRD policy."""
        return select_mvp(features)

    @staticmethod
    def _merge_semantic_duplicates(features: list[dict]) -> list[dict]:
        """Collapse generic validation and duplicate authentication journeys."""
        by_id = {
            str(item.get("featureId") or ""): item
            for item in features if isinstance(item, dict)
        }
        result: list[dict] = []
        auth_owners: dict[tuple, dict] = {}
        for item in features:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name") or "").casefold()
            if item.get("source") != "prd_core" and not item.get("standalone") and (
                any(token in name for token in ("입력 검증", "입력·권한 검증 및 실행", "오류 처리", "validation and error"))
                or ("입력" in name and "검증" in name)
            ):
                parent = by_id.get(str(item.get("parentFeatureId") or ""))
                if parent is not None:
                    parent["requirements"] = list(dict.fromkeys([
                        *(parent.get("requirements") or []),
                        *(item.get("requirements") or []),
                    ]))
                    continue
            contracts = [
                contract for contract in item.get("apiContract") or []
                if isinstance(contract, dict)
            ]
            is_auth = (
                str(item.get("featureId") or "").startswith("auth.")
                or any("/auth/" in str(contract.get("path") or "") for contract in contracts)
                or any(token in name for token in ("로그인", "회원가입", "login", "signup"))
            )
            auth_key = tuple(sorted(
                (str(contract.get("method") or "GET"), str(contract.get("path") or ""))
                for contract in contracts
            ))
            auth_owner = auth_owners.get(auth_key) if auth_key else None
            if is_auth and auth_owner is not None and item.get("source") != "prd_core":
                auth_owner["requirements"] = list(dict.fromkeys([
                    *(auth_owner.get("requirements") or []), *(item.get("requirements") or []),
                ]))
                auth_owner["actions"] = list(dict.fromkeys([
                    *(auth_owner.get("actions") or []), *(item.get("actions") or []),
                ]))
                auth_owner["apiContract"] = list({
                    (str(contract.get("method")), str(contract.get("path"))): contract
                    for contract in [
                        *(auth_owner.get("apiContract") or []), *contracts,
                    ] if isinstance(contract, dict)
                }.values())
                owner_db = auth_owner.setdefault("dbContract", {"tables": [], "foreignKeys": []})
                item_db = item.get("dbContract") or {}
                owner_db["tables"] = list(dict.fromkeys([
                    *(owner_db.get("tables") or []), *(item_db.get("tables") or []),
                ]))
                foreign_keys = [
                    *(owner_db.get("foreignKeys") or []), *(item_db.get("foreignKeys") or []),
                ]
                owner_db["foreignKeys"] = list({
                    json.dumps(value, ensure_ascii=False, sort_keys=True): value
                    for value in foreign_keys
                }.values())
                continue
            if is_auth and auth_key:
                auth_owners[auth_key] = item
            result.append(item)
        return result

    @staticmethod
    def _plan_supporting_specs(state: PipelineState, existing: list[dict]) -> list[dict]:
        """LLM이 생략한 projectPlan SHOULD/COULD 기능을 기능명세에 결정론적으로 추가한다."""
        plan = state.project_plan if isinstance(state.project_plan, dict) else {}
        priorities = plan.get("priorities") if isinstance(plan.get("priorities"), dict) else {}
        known = {str(item.get("name") or "").strip().casefold() for item in existing if isinstance(item, dict)}
        core = next((item for item in existing if item.get("source") == "prd_core"), None)
        parent = str((core or {}).get("featureId") or "").strip() or None
        result: list[dict] = []

        # 모델이 coreFeatures만 반환하는 경우에도 실제 서비스에 공통으로 필요한
        # 최소 운영 기능을 한 번 더 검토한다. 이 목록은 모든 프로젝트에 일괄
        # 삽입하지 않고, 사용자/도메인 신호가 있을 때만 추가한다.
        project_text = " ".join([
            str(state.project_name or ""),
            str(state.context_prompt or ""),
            " ".join(str(item.get("name") or "") for item in existing if isinstance(item, dict)),
        ]).casefold()
        inferred: list[tuple[str, str, str, int, bool]] = []
        # Input validation is a requirement of every feature, not a standalone
        # business capability. Publishing it as a feature creates a generic API
        # and table that cannot be connected meaningfully in the knowledge graph.
        user_domain = (
            state.target_users
            or any(term in project_text for term in (
                "사용자", "회원", "예약", "여행", "거래", "주문", "수업", "추천", "공유",
                "user", "booking", "travel", "commerce", "order",
            ))
        )
        has_auth = any(
            str(item.get("featureId") or "").startswith(("auth.", "profile."))
            or any(
                "/auth/" in str(contract.get("path") or "")
                for contract in item.get("apiContract") or [] if isinstance(contract, dict)
            )
            or any(term in str(item.get("name") or "").casefold() for term in ("로그인", "회원가입", "login", "signup"))
            for item in existing if isinstance(item, dict)
        )
        if user_domain and not has_auth:
            inferred.extend([
                (
                    "회원가입 및 로그인",
                    "사용자 식별과 핵심 기능 접근 권한을 제공하기 위한 supporting 기능",
                    "P1", 10, True,
                ),
                (
                    "회원정보 수정",
                    "사용자가 자신의 기본 정보와 서비스 설정을 관리하기 위한 supporting 기능",
                    "P2", 6, False,
                ),
            ])
        if any(term in project_text for term in (
            "예약", "주문", "결제", "상태", "알림", "여행", "booking", "order", "payment", "status",
        )):
            inferred.append((
                "상태 변경 및 알림 처리",
                "핵심 업무의 상태 전이와 사용자에게 결과를 전달하기 위한 supporting 기능",
                "P2", 6, False,
            ))

        # Chat/FormData가 명시한 공통 기능은 LLM이 누락해도 supporting
        # 계약으로 반드시 보존한다.
        for raw_name in state.supporting_feature_names or []:
            name = str(raw_name or "").strip()
            if not name or name.casefold() in known:
                continue
            slug = re.sub(r"[^a-z0-9]+", ".", name.casefold()).strip(".") or str(len(result) + 1)
            result.append({
                "featureId": f"supporting.{slug}",
                "name": name,
                "source": "supporting",
                "parentFeatureId": parent,
                "reason": "Chat에서 명시된 공통 기능을 핵심 기능의 실행 기반으로 제공",
                "actions": ["manage"],
                "priority": "P1",
                "priorityScore": 10,
                "releaseGate": True,
            })
            known.add(name.casefold())
        for bucket, priority, score in (("should", "P2", 6), ("could", "P3", 3)):
            for raw_name in priorities.get(bucket) or []:
                name = str(raw_name or "").strip()
                if not name or name.casefold() in known:
                    continue
                slug = re.sub(r"[^a-z0-9]+", ".", name.casefold()).strip(".") or str(len(result) + 1)
                result.append({
                    "featureId": f"supporting.{slug}",
                    "name": name,
                    "source": "supporting",
                    "parentFeatureId": parent,
                    "reason": f"projectPlan의 {bucket.upper()} 우선순위로 핵심 기능을 보완",
                    "actions": ["manage"],
                    "priority": priority,
                    "priorityScore": score,
                    "releaseGate": False,
                })
                known.add(name.casefold())

        # 마지막으로 모델이 생략한 공통 운영 기능을 보강한다. 명시된 Chat/FormData
        # 및 projectPlan 항목을 먼저 보존하므로 사용자의 명세가 우선한다.
        for name, reason, priority, score, gate in inferred:
            if name.casefold() in known:
                continue
            slug = re.sub(r"[^a-z0-9]+", ".", name.casefold()).strip(".") or str(len(result) + 1)
            result.append({
                "featureId": f"supporting.{slug}",
                "name": name,
                "source": "supporting",
                "parentFeatureId": parent,
                "reason": reason,
                "actions": ["manage"],
                "priority": priority,
                "priorityScore": score,
                "releaseGate": gate,
            })
            known.add(name.casefold())
        return result

    @staticmethod
    def _derive_missing_subfeatures(existing: list[dict]) -> list[dict]:
        """LLM이 부모 기능만 반환할 때 구현 가능한 하위 기능을 최소 보강한다."""
        cores = [item for item in existing if isinstance(item, dict) and item.get("source") == "prd_core"]
        names = {str(item.get("name") or "").casefold() for item in existing if isinstance(item, dict)}
        derived: list[dict] = []
        for core in cores:
            parent_id = str(core.get("featureId") or "").strip()
            if not parent_id:
                continue
            child_count = sum(
                1 for item in existing
                if isinstance(item, dict) and str(item.get("parentFeatureId") or "").strip() == parent_id
            )
            if child_count >= 2:
                continue
            core_name = str(core.get("name") or "기능").strip()
            actions = {str(action).casefold() for action in core.get("actions") or []}
            if actions and actions <= {"read", "search", "list", "view"}:
                candidates = [
                    ("검색·필터 및 상세 조회", ["read"], "검색 조건을 검증하고 목록·상세 결과를 제공"),
                    ("조회 결과 저장 및 이력", ["manage"], "사용자별 조회 결과와 변경 이력을 보존"),
                ]
            else:
                candidates = [
                    ("입력·권한 검증 및 실행", ["create"], "입력값과 권한을 검증한 뒤 핵심 처리를 실행"),
                    ("상태 전이 및 처리 이력", ["manage"], "처리 상태를 전이하고 성공·실패 이력을 보존"),
                ]
            for suffix, sub_actions, behavior in candidates[: max(0, 2 - child_count)]:
                name = f"{core_name} · {suffix}"
                if name.casefold() in names:
                    continue
                slug = re.sub(r"[^a-z0-9]+", ".", suffix.casefold()).strip(".") or "subfeature"
                derived.append({
                    "featureId": f"{parent_id}.{slug}",
                    "name": name,
                    "source": "supporting",
                    "parentFeatureId": parent_id,
                    "reason": f"{core_name}을 실제 서비스 흐름으로 구현하기 위한 하위 기능",
                    "actions": sub_actions,
                    "priority": "P1" if "검증" in suffix or "권한" in suffix else "P2",
                    "priorityScore": 10 if "검증" in suffix or "권한" in suffix else 6,
                    "releaseGate": "검증" in suffix or "권한" in suffix,
                    "description": f"사용자가 {core_name}을 이용할 때 {behavior}하여 결과를 제공한다.",
                    "requirements": [
                        f"{core_name}의 선행조건과 입력값을 확인하고 정상 처리 결과와 실패 사유를 반환한다.",
                    ],
                })
                names.add(name.casefold())
        return derived

    async def execute(self, state: PipelineState, dump=None) -> PipelineState:
        from config.settings import settings

        prd = try_parse_json(state.prd_document) or {}
        invalid = shape_issues("prd", prd) + shape_issues("registry", state.feature_registry)
        if invalid:
            blocker = "FEATURE_SPEC_CONTRACT_BLOCKER: " + "; ".join(f"{item['pointer']}: {item['message']}" for item in invalid)
            return state.copy(generation_blockers=[*state.generation_blockers, blocker], qa_approved=False,
                              failed_stage="FEATURE_SPEC", last_completed_stage="PRD",
                              skipped_stages=["DBA_API", "QA", "PHASE3", "PHASE4"])
        source_registry = normalize_feature_registry(
            state.feature_registry, state.feature_list, preserve_extra=True,
        )
        cores = source_cores(prd)
        prd = {**prd, "coreFeatures": cores}
        prompt = PROMPT.format(
            project_plan=json.dumps(state.project_plan or {}, ensure_ascii=False),
            prd=json.dumps({**prd, "coreFeatures": generation_input(cores)}, ensure_ascii=False),
            registry=json.dumps(generation_input(source_registry), ensure_ascii=False),
            output_contract=output_schema_prompt(draft=True),
        )
        features, diagnostics, input_issues = await generate_document(
            self._client, self._model, self._runtime, prompt, cores,
            mode=settings.feature_spec_output_mode,
            diagnostics_dir=settings.feature_spec_diagnostics_dir,
            max_tokens=settings.feature_spec_max_completion_tokens,
            reasoning_effort=settings.feature_spec_reasoning_effort,
            timeout_seconds=settings.feature_spec_request_timeout_seconds,
            max_attempts=settings.feature_spec_max_attempts,
            draft=True,
        )
        state = state.copy(generation_diagnostics=[*state.generation_diagnostics, *diagnostics])
        if input_issues:
            blocker = "FEATURE_SPEC_CONTRACT_BLOCKER: " + "; ".join(input_issues)
            return state.copy(
                generation_blockers=list(dict.fromkeys([*state.generation_blockers, blocker])),
                qa_prd_blockers=[*state.qa_prd_blockers, blocker], qa_approved=False,
                failed_stage="FEATURE_SPEC", last_completed_stage="PRD",
                skipped_stages=["DBA_API", "QA", "PHASE3", "PHASE4"],
                status_message="Feature Spec 계약 검증 실패 — downstream 차단",
            )
        source_by_name = {str(item.get("name") or "").strip(): item for item in source_registry}
        source_by_id = {str(item.get("featureId") or "").strip(): item for item in source_registry}
        core_by_source = {item["sourceFeatureId"]: item for item in cores}
        prd_by_name = {
            str(item.get("name") or "").strip(): item
            for item in (prd.get("coreFeatures") or [])
            if isinstance(item, dict) and str(item.get("name") or "").strip()
        }
        used_ids = {str(item.get("featureId") or "").strip() for item in source_registry if item.get("featureId")}
        output_ids: set[str] = set()
        normalized = []
        for item in features:
            if not isinstance(item, dict) or not str(item.get("name") or "").strip():
                continue
            item = dict(item)
            name = str(item.get("name") or "").strip()
            explicit_supporting = str(item.get("source") or "").casefold() in {"supporting", "support"}
            prd_item = {} if explicit_supporting else core_by_source.get(item.get("sourceFeatureId"), prd_by_name.get(name, {}))
            source = source_by_id.get(prd_item.get("featureId")) or source_by_name.get(name) or source_by_id.get(str(item.get("featureId") or "").strip())
            if source:
                # Feature Spec may add priority/reason, but PM's contract remains canonical.
                if semantic_identifier(source.get("featureId")):
                    item["featureId"] = source["featureId"]
                if not isinstance(item.get("apiContract"), list) or not item.get("apiContract"):
                    item["apiContract"] = source.get("apiContract", [])
                if not isinstance(item.get("dbContract"), dict) or not item.get("dbContract", {}).get("tables"):
                    item["dbContract"] = source.get("dbContract", {"tables": [], "foreignKeys": []})
            else:
                candidate_id = str(item.get("featureId") or "").strip()
                if not candidate_id or candidate_id in used_ids:
                    parent = str(item.get("parentFeatureId") or "").strip()
                    slug = re.sub(r"[^a-z0-9]+", ".", name.lower()).strip(".") or f"supporting.{len(normalized)+1}"
                    candidate_id = f"{parent}.{slug}" if parent else f"supporting.{slug}"
                    base_id = candidate_id
                    suffix = 2
                    while candidate_id in used_ids:
                        candidate_id = f"{base_id}.{suffix}"
                        suffix += 1
                item["featureId"] = candidate_id
                used_ids.add(candidate_id)
            # LLM/fallback 출력이 supporting 접두사를 중복해도 canonical ID로
            # 수렴시켜 API/DB/그래프에 supporting.supporting.*가 노출되지 않게 한다.
            item["featureId"] = re.sub(r"^supporting\.supporting\.", "supporting.", str(item["featureId"]))
            if item["featureId"] in output_ids:
                if any(existing.get("name") == name for existing in normalized):
                    continue
                base_id = item["featureId"]
                suffix = 2
                while item["featureId"] in output_ids:
                    item["featureId"] = f"{base_id}.{suffix}"
                    suffix += 1
            output_ids.add(item["featureId"])
            # Preserve an explicit supporting classification from the LLM or
            # derived subfeature pipeline. Generic account/profile detection
            # is only a fallback; otherwise a domain subfeature such as
            # inspection-record can be incorrectly promoted to PRD core/P0.
            explicit_source = str(
                item.get("source")
                or (source or {}).get("source")
                or ""
            ).strip().casefold()
            is_account_feature = any(term in name.casefold() for term in (
                "회원가입", "로그인", "로그아웃", "비밀번호", "프로필", "signup", "login", "password", "profile",
            ))
            if is_account_feature or explicit_source in {"supporting", "support"} or (not prd_item and _is_supporting_feature(name)):
                item["source"] = "supporting"
                if not item.get("parentFeatureId"):
                    item["parentFeatureId"] = next(
                        (
                            str(candidate.get("featureId") or candidate.get("id"))
                            for candidate in normalized
                            if isinstance(candidate, dict)
                            and candidate.get("source") == "prd_core"
                        ),
                        None,
                    )
                item.setdefault("reason", "핵심 도메인 기능을 사용하기 위한 공통 계정·운영 기능")
                priority, score, gate = _score_supporting(
                    str(item.get("name")), str(item.get("reason")), state.project_plan, feature=item,
                )
                if priority == "EXCLUDE":
                    continue
                item["priority"], item["priorityScore"], item["releaseGate"] = priority, score, gate
            else:
                item["source"] = "prd_core"
                canonical_priority = str(
                    prd_item.get("priority")
                    or (source or {}).get("priority")
                    or item.get("priority")
                    or "P1"
                ).upper()
                if canonical_priority not in {"P0", "P1", "P2", "P3"}:
                    canonical_priority = "P1"
                item["priority"] = canonical_priority
                item["priorityScore"] = int(
                    prd_item.get("priorityScore")
                    or (source or {}).get("priorityScore")
                    or item.get("priorityScore")
                    or {"P0": 13, "P1": 10, "P2": 6, "P3": 3}[canonical_priority]
                )
            item.setdefault("actions", ["manage"])
            item.setdefault("apiContract", [])
            item.setdefault("dbContract", {"tables": [], "foreignKeys": []})
            _ensure_feature_narrative(item, source, prd_item)
            _normalize_contracts(item)
            normalized.append(item)

        # Publish validated model contracts only; missing journeys are not
        # replaced with synthetic tables, routes or untyped subfeatures.
        # Feature Spec가 LLM/derived supporting 항목을 추가한 뒤에도 Registry
        # 전역 계약 중복 규칙을 다시 적용한다. 이 단계가 빠지면 PRD의
        # supporting 항목이 같은 auth route를 공유해 API graph에서 한쪽이
        # 고립된다.
        normalized = normalize_feature_registry(
            normalized,
            [str(item.get("name") or "") for item in normalized],
            preserve_extra=False,
        )
        # The final Feature Spec registry owns the semantic and physical
        # contracts consumed by DBA, API and validation.
        core_by_name = {
            str(item.get("name") or ""): item for item in prd.get("coreFeatures") or []
            if isinstance(item, dict)
        }
        auth_required = requires_auth(prd, state.feature_list, state.user_query)
        normalized = [normalize_feature_contract({
            **core_by_name.get(str(item.get("name") or ""), {}),
            **source_by_name.get(str(item.get("name") or ""), {}),
            **item,
        }, auth_required) for item in normalized]
        normalized = self._merge_semantic_duplicates(normalized)
        inherit_validation_contracts(normalized)
        dependency_errors = dependency_issues(normalized)
        if dependency_errors:
            return state.copy(generation_blockers=[*state.generation_blockers, *dependency_errors], qa_approved=False,
                              failed_stage="FEATURE_SPEC", last_completed_stage="PRD",
                              skipped_stages=["DBA_API", "QA", "PHASE3", "PHASE4"])
        apply_project_priorities(normalized, state.project_plan)
        normalized = self._apply_mvp_budget(normalized)
        final_by_source = {item.get("sourceFeatureId"): item for item in normalized if item.get("source") == "prd_core"}
        prd_core = [{**item, **{key: final_by_source[item["sourceFeatureId"]][key]
                              for key in ("featureId", "name", "actions", "apiContract", "dbContract", "readDependencies",
                                          "statePolicies", "intervalPolicies", "sessionPolicy", "ownership", "roles",
                                          "stateTransitions", "transactionRules", "description", "requirements", "outputRequirements") if key in final_by_source[item["sourceFeatureId"]]}}
                    for item in cores if item["sourceFeatureId"] in final_by_source]
        prd = {**prd, "coreFeatures": prd_core}
        bind_auth_registry_policy(normalized)
        normalized = normalize_feature_registry(
            normalized, [item["name"] for item in normalized], preserve_extra=True,
        )
        document = json.dumps({"features": normalized}, ensure_ascii=False)
        missing_contracts = [*feature_contract_issues(normalized), *registry_contract_issues(normalized)]
        blockers = [*state.generation_blockers]
        if missing_contracts:
            blockers.append(
                "FEATURE_SPEC_CONTRACT_BLOCKER: API/DB 계약이 없는 기능 — "
                + ", ".join(missing_contracts)
            )
            return state.copy(
                generation_blockers=list(dict.fromkeys(blockers)),
                qa_prd_blockers=[*state.qa_prd_blockers, blockers[-1]], qa_approved=False,
                status_message="Feature Spec 최종 계약 검증 실패 — 기존 산출물 보존",
                failed_stage="FEATURE_SPEC", last_completed_stage="PRD",
                skipped_stages=["DBA_API", "QA", "PHASE3", "PHASE4"],
            )
        return state.copy(
            feature_registry=normalized,
            feature_specs=normalized,
            feature_spec_document=document,
            feature_list=[item["name"] for item in normalized],
            prd_document=json.dumps(prd, ensure_ascii=False),
            generation_blockers=list(dict.fromkeys(blockers)),
            status_message="Feature Spec 에이전트 완료 — 기능 계약 확정",
        )
