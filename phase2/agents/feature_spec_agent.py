"""PRD coreFeatures를 구현 가능한 기능 계약으로 확정하는 Agent."""

from __future__ import annotations

import json
import re
from typing import Any

from openai import AsyncOpenAI

from phase2.feature_registry import normalize_feature_registry
from phase2.json_utils import try_parse_json
from phase2.llm_runtime import LlmRuntime
from phase2.state import PipelineState


def _normalize_contracts(item: dict) -> dict:
    """Feature Spec 계약을 downstream Agent가 소비하는 단일 JSON 형태로 고정한다."""
    api = item.get("apiContract", item.get("api"))
    normalized_api = []
    for contract in api if isinstance(api, list) else []:
        if not isinstance(contract, dict):
            continue
        value = dict(contract)
        value["method"] = str(value.get("method") or "GET").upper()
        path = str(value.get("path") or "").strip()
        if path and not path.startswith("/api/v1/"):
            path = "/api/v1/" + path.lstrip("/")
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
                    "table": str(fk.get("table") or "").strip(),
                    "column": column,
                    "references": {"table": target, "column": target_column},
                })
        elif str(fk).strip():
            foreign_keys.append(str(fk).strip())
    item["apiContract"] = normalized_api
    item["api"] = normalized_api
    item["dbContract"] = {"tables": tables, "foreignKeys": foreign_keys}
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
    # 한국어 supporting 기능명은 slug가 숫자/빈 문자열로 떨어질 수 있다.
    # supporting.1.1 같은 ID를 도메인으로 해석하면 1s 테이블이 생기므로
    # 이름 기반 도메인이 없는 경우 공용 supporting 테이블로 수렴시킨다.
    if feature_id.startswith(("supporting.", "supporting-")):
        return "supporting_features"
    # PM이 의미 없는 순번 ID만 제공한 경우에도 API/DB 계약은 필요하다.
    # 숫자를 제거한 feature001s 같은 이름은 sanitize_tables와 충돌하므로
    # 안정적인 fallback 식별자를 사용하고, 의미 있는 Registry 계약이 있으면
    # 아래에서 항상 그 계약을 우선한다.
    generic_feature = re.match(r"^(feature_\d+)(?:\.|$)", feature_id)
    if generic_feature:
        return f"{generic_feature.group(1)}_records"
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
    base = "/api/v1/" + domain
    if domain == "users":
        base = "/api/v1/auth"
    if domain == "supporting_features" or re.fullmatch(r"feature_\d+_records", domain):
        suffix = re.sub(r"[^a-z0-9]+", "-", feature_id.casefold()).strip("-") or "feature"
        base = f"{base}/{suffix}"
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
        elif action in {"notify", "retry"} and domain == "notifications":
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
    base_path = f"/api/v1/auth/{parts[-1]}" if namespace == "auth" else f"/api/v1/{domain}"
    # supporting/subfeature 기능은 모두 공용 테이블을 사용하더라도 API 계약은
    # 기능별로 고유해야 한다. 같은 /supporting_features 경로를 공유하면 API
    # dedupe 단계에서 한 기능만 남아 나머지 Registry 노드가 고립된다.
    unique_contract_domain = domain == "supporting_features" or bool(
        re.fullmatch(r"feature_\d+_records", domain)
    )
    if unique_contract_domain:
        suffix = re.sub(r"[^a-z0-9]+", "-", feature_id.casefold()).strip("-") or "feature"
        base_path = f"{base_path}/{suffix}"
    contracts = item.get("apiContract") if isinstance(item.get("apiContract"), list) else []
    # A source registry item can carry a stale auth contract after PM/PRD
    # repair. Do not let that contract turn an unrelated generic supporting
    # feature (for example supporting.1) into an implicit login feature.
    auth_feature = any(term in name for term in (
        "login", "signin", "로그인", "register", "회원가입", "signup",
        "logout", "로그아웃", "profile", "프로필", "password", "비밀번호",
    ))
    if not auth_feature:
        contracts = [
            contract for contract in contracts
            if not re.search(r"/auth/(?:login|signup|register|logout|refresh)(?:/|$)", str(contract.get("path") or ""), re.I)
        ]
        item["apiContract"] = contracts
    generic_contract = any(
        re.fullmatch(r"/api/v1/feature_\d+_records/?", str(c.get("path") or ""))
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
        or unique_contract_domain
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

    db = item.get("dbContract") if isinstance(item.get("dbContract"), dict) else {}
    tables = db.get("tables") if isinstance(db.get("tables"), list) else []
    if not tables:
        tables = [domain]
    item["dbContract"] = {
        "tables": tables,
        "foreignKeys": db.get("foreignKeys") if isinstance(db.get("foreignKeys"), list) else [],
    }
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


PROMPT = """당신은 Feature Spec 아키텍트입니다.
PRD의 coreFeatures를 모두 기능명세로 변환하고, 구현에 필수적인 supporting 기능만 추가하세요.
JSON만 출력하세요.

규칙:
- PRD coreFeatures는 모두 source='prd_core', priority='P0'으로 유지
- supporting 기능은 핵심 기능을 실제 서비스로 운영하기 위해 필요한 일반 기능도 검토하세요.
  적용 가능한 경우 회원가입, 로그인·로그아웃, 회원정보 수정, 비밀번호 재설정,
  운영자 계정, 역할·권한 관리, 입력 검증, 상태 전이, 알림 처리, 알림 재시도,
  이력/감사 로그를 추가하세요. 모든 프로젝트에 무조건 넣지는 말고 사용자 식별,
  개인정보, 권한, 예약·주문·결제·협업 데이터 등 실제 요구사항과 연결될 때만 추가하세요.
- supporting 기능은 source='supporting', parentFeatureId, reason을 반드시 작성
- priority는 PM projectPlan의 MoSCoW 우선순위를 우선 반영하세요.
  MUST는 핵심이면 P0, supporting이면 P1, SHOULD는 P2, COULD는 P3으로 매핑하고,
  projectPlan의 priorities에 없는 supporting 기능만 의존성/출시 차단/장애 영향/사용 빈도/보안·무결성 점수로 결정하세요.
- priorities.must/should/could/wont에 명시되거나 의미상 일치하는 기능을 찾아 우선순위를 연결하세요.
- WONT 또는 scope.out에 해당하는 기능은 기능명세에 추가하지 마세요.
- priorityScore는 0~13 정수: 핵심 의존성(0~3)+출시 차단(0~3)+장애 영향(0~2)+사용 영향(0~2)+보안·무결성(0~3)
- 10~13=P1, 5~9=P2, 0~4=P3
- 보안·법규·데이터 무결성은 releaseGate=true로 지정
- 각 기능에는 featureId, name, source, parentFeatureId, reason, actions, priority, priorityScore,
  releaseGate, description, requirements, apiContract, dbContract를 포함
- description은 사용자 상황 → 시스템 처리 → 결과를 설명하는 문장으로 반드시 작성하세요.
- requirements는 최소 1개 이상이며 역할, 선행조건, 입력, 정상 처리, 실패 조건, 결과를 포함하세요.
- PRD coreFeatures는 한 항목도 누락하지 말고 각각 하나의 feature object로 변환하세요. core 기능의 featureId/name/actions/apiContract/dbContract는 PM Registry와 PRD 사이에서 일치해야 합니다.
- 모든 backend 기능은 apiContract에 실제 method/path/action/featureId를 최소 하나 포함하세요. 저장·상태·이력·권한 데이터가 필요한 기능은 dbContract.tables를 최소 하나 포함하세요.
- dbContract.foreignKeys는 문자열 설명만 쓰지 말고 반드시 {{"table":"source_table","column":"...","references":{{"table":"...","column":"id"}}}} 객체로 작성하세요. source table은 dbContract.tables 중 하나여야 하며 실제 테이블 목록에 없는 대상을 참조하지 마세요.
- API path는 반드시 /api/v1/로 시작하는 영문 소문자 REST 경로여야 하며, 같은 method/path를 중복 생성하지 마세요.
- Registry에 이미 apiContract/dbContract가 있으면 재해석하거나 이름을 바꾸지 말고 그대로 보존하세요. 누락된 계약만 요구사항과 PRD를 근거로 보완하세요.
- supporting 기능은 실제 핵심 기능의 실행·보안·무결성에 필요한 경우에만 추가하고, 각 supporting 기능에 parentFeatureId와 reason을 작성하세요. supporting 기능을 coreFeatures 또는 P0으로 승격하지 마세요.
- 각 prd_core 기능을 이름 하나로 끝내지 말고, 실제 구현에 필요한 하위 기능을 2~4개까지 세분화하세요. 예: 입력/권한 검증, 생성·변경 처리, 목록·상세 조회, 상태 전이·이력, 알림·실패 재처리 중 해당 기능과 관련된 항목을 선택합니다.
- 하위 기능은 부모 기능의 featureId를 parentFeatureId로 사용하고, 부모와 다른 구체적인 featureId/name/actions/apiContract/dbContract/description/requirements를 가져야 합니다. 단순히 "기타"나 "관리"로 뭉뚱그리지 마세요.
- 전체 기능 수가 coreFeatures 수와 같아지지 않도록 하며, coreFeatures가 5개라면 일반적으로 core + supporting을 합쳐 10개 이상이 되도록 충분히 분해하세요. 단, 제품 범위와 무관한 기능을 억지로 추가하지 마세요.
- core 기능의 featureId는 PM Registry와 일치시키고, supporting 기능은 부모 ID 뒤에 안정적인 suffix를 붙이세요
- 회원가입·로그인·회원정보 기능은 독립적인 auth.* 또는 profile.* ID를 사용하고, 핵심 기능에 대한 parentFeatureId와 추가 이유를 명시하세요.

[PM projectPlan]
{project_plan}

[PRD]
{prd}

[PM Registry 참고]
{registry}

출력:
{{"features":[{{"featureId":"inventory.register","name":"원두 재고 등록","source":"prd_core","parentFeatureId":null,"reason":"","actions":["create"],"priority":"P0","priorityScore":13,"releaseGate":false,"description":"사용자가 원두 재고를 등록하면 시스템이 입력값을 검증하고 재고 수량을 저장하여 현재 재고를 조회할 수 있게 한다.","requirements":["관리자는 원두명과 수량을 입력하고 시스템은 필수값과 권한을 검증한 뒤 저장하며 오류 시 원인을 반환한다."],"apiContract":[],"dbContract":{{"tables":[],"foreignKeys":[]}}}}]}}
"""


def _project_priority(name: str, project_plan: dict[str, Any]) -> tuple[str | None, bool]:
    priorities = project_plan.get("priorities") if isinstance(project_plan, dict) else {}
    if not isinstance(priorities, dict):
        return None, False
    normalized = str(name or "").casefold().strip()
    for key, priority, gate in (("must", "P1", True), ("should", "P2", False), ("could", "P3", False)):
        values = priorities.get(key) or []
        if any(normalized == str(value).casefold().strip() or normalized in str(value).casefold() or str(value).casefold() in normalized for value in values):
            return priority, gate
    wont = priorities.get("wont") or []
    if any(normalized == str(value).casefold().strip() or normalized in str(value).casefold() or str(value).casefold() in normalized for value in wont):
        return "EXCLUDE", False
    return None, False


def _score_supporting(name: str, reason: str, project_plan: dict[str, Any] | None = None) -> tuple[str, int, bool]:
    mapped, gate = _project_priority(name, project_plan or {})
    if mapped == "EXCLUDE":
        return "EXCLUDE", 0, False
    if mapped:
        return mapped, {"P1": 10, "P2": 6, "P3": 3}[mapped], gate
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
        """P0 -> P1 -> P2 순서로 최대 12개 MVP 계약만 남긴다."""
        core = [item for item in features if item.get("source") == "prd_core"]
        supporting = [item for item in features if item.get("source") != "prd_core"]

        # PM 핵심 기능은 최대 5개까지만 P0 계약으로 유지한다.
        active_core = core[:5]

        # P1은 최대 6개, P2는 최대 3개까지 계약화한다. 각 bucket 안에서는
        # priorityScore를 우선해 LLM 응답 순서에 따른 변동을 줄인다.
        def ranked(priority: str) -> list[dict]:
            items = [
                item for item in supporting
                if str(item.get("priority") or "P2").upper() == priority
            ]
            return [item for _, item in sorted(
                enumerate(items),
                key=lambda pair: (-int(pair[1].get("priorityScore") or 0), pair[0]),
            )]

        p1 = ranked("P1")[:6]
        p2 = ranked("P2")[:3]
        active = (active_core + p1 + p2)[:12]
        return active

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
        if existing:
            inferred.append((
                "입력 검증 및 오류 처리",
                "핵심 기능의 입력값·권한·실패 응답을 일관되게 검증하기 위한 supporting 기능",
                "P1", 10, True,
            ))
        user_domain = (
            state.target_users
            or any(term in project_text for term in (
                "사용자", "회원", "예약", "여행", "거래", "주문", "수업", "추천", "공유",
                "user", "booking", "travel", "commerce", "order",
            ))
        )
        if user_domain:
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
        children_by_parent: dict[str, int] = {}
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
        prd = try_parse_json(state.prd_document) or {}
        source_registry = normalize_feature_registry(
            state.feature_registry, state.feature_list, preserve_extra=True,
        )
        prompt = PROMPT.format(
            project_plan=json.dumps(state.project_plan or {}, ensure_ascii=False),
            prd=json.dumps(prd, ensure_ascii=False),
            registry=json.dumps(source_registry, ensure_ascii=False),
        )
        data = None
        try:
            async def request():
                return await self._client.chat.completions.create(
                    model=self._model, temperature=0.2, max_completion_tokens=12000,
                    messages=[{"role": "system", "content": "JSON만 출력하세요."}, {"role": "user", "content": prompt}],
                )
            response = await self._runtime.call(request) if self._runtime else await request()
            data = try_parse_json(response.choices[0].message.content or "")
        except Exception:
            data = None

        features = data.get("features") if isinstance(data, dict) else None
        if not isinstance(features, list):
            features = []
            core = prd.get("coreFeatures") if isinstance(prd, dict) else []
            by_name = {item.get("name"): item for item in source_registry}
            for item in core or []:
                if not isinstance(item, dict):
                    continue
                name = str(item.get("name") or "").strip()
                ref = by_name.get(name, {})
                features.append({
                    "featureId": ref.get("featureId") or ref.get("id") or f"feature_{len(features)+1:03d}",
                    "name": name, "source": "prd_core", "parentFeatureId": None, "reason": "",
                    "actions": ["manage"], "priority": "P0", "priorityScore": 13,
                    "releaseGate": False, "apiContract": ref.get("apiContract", []),
                    "dbContract": ref.get("dbContract", {"tables": [], "foreignKeys": []}),
                })

        source_by_name = {str(item.get("name") or "").strip(): item for item in source_registry}
        source_by_id = {str(item.get("featureId") or "").strip(): item for item in source_registry}
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
            source = source_by_name.get(name) or source_by_id.get(str(item.get("featureId") or "").strip())
            prd_item = prd_by_name.get(name, {})
            if source:
                # Feature Spec may add priority/reason, but PM's contract remains canonical.
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
            if explicit_source in {"supporting", "support"} or _is_supporting_feature(name):
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
                    str(item.get("name")), str(item.get("reason")), state.project_plan,
                )
                if priority == "EXCLUDE":
                    continue
                item["priority"], item["priorityScore"], item["releaseGate"] = priority, score, gate
            else:
                item["source"], item["priority"], item["priorityScore"] = "prd_core", "P0", 13
            item.setdefault("actions", ["manage"])
            item.setdefault("apiContract", [])
            item.setdefault("dbContract", {"tables": [], "foreignKeys": []})
            _normalize_contracts(item)
            _ensure_required_contracts(item)
            _ensure_feature_narrative(item, source, prd_item)
            normalized.append(item)

        # LLM 응답이 coreFeatures만 담고 supporting 하위 기능을 생략해도
        # 기능명세가 5~6개 카드로 축소되지 않도록 부모별 최소 하위 흐름을 보장한다.
        for supporting in self._derive_missing_subfeatures(normalized):
            supporting = _ensure_required_contracts(_normalize_contracts(supporting))
            normalized.append(_ensure_feature_narrative(supporting, {}, {}))

        # supporting 누락은 LLM 산출물의 선택사항이 되면 안 된다. PM 계획에 명시된
        # SHOULD/COULD 여정은 Feature Spec의 최소 계약으로 항상 승격한다.
        for supporting in self._plan_supporting_specs(state, normalized):
            supporting = _ensure_required_contracts(_normalize_contracts(supporting))
            normalized.append(_ensure_feature_narrative(supporting, {}, {}))
        # LLM이 core/supporting 일부를 누락해도 PM registry 계약은 사라지지 않게 한다.
        known_ids = {str(item.get("featureId") or "") for item in normalized}
        for source in source_registry:
            if source.get("featureId") in known_ids:
                continue
            source_item = {
                **source,
                "source": "prd_core" if source.get("featureId") in {
                    str(item.get("featureId") or "") for item in (prd.get("coreFeatures") or []) if isinstance(item, dict)
                } else "supporting",
                "priority": source.get("priority", "P0"),
                "priorityScore": source.get("priorityScore", 13),
                "releaseGate": source.get("releaseGate", False),
            }
            source_item = _ensure_required_contracts(_normalize_contracts(source_item))
            normalized.append(_ensure_feature_narrative(
                source_item, source, prd_by_name.get(source_item.get("name"), {}),
            ))
        normalized = self._apply_mvp_budget(normalized)
        # Feature Spec가 LLM/derived supporting 항목을 추가한 뒤에도 Registry
        # 전역 계약 중복 규칙을 다시 적용한다. 이 단계가 빠지면 PRD의
        # supporting 항목이 같은 auth route를 공유해 API graph에서 한쪽이
        # 고립된다.
        normalized = normalize_feature_registry(
            normalized, state.feature_list, preserve_extra=True,
        )
        for item in normalized:
            if (
                not item.get("apiContract")
                or not isinstance(item.get("dbContract"), dict)
                or not item.get("dbContract", {}).get("tables")
            ):
                _ensure_required_contracts(_normalize_contracts(item))
        document = json.dumps({"features": normalized}, ensure_ascii=False)
        missing_contracts = [
            str(item.get("featureId") or item.get("name") or "unknown")
            for item in normalized
            if not item.get("apiContract")
            or not isinstance(item.get("dbContract"), dict)
            or not item.get("dbContract", {}).get("tables")
        ]
        blockers = [*state.generation_blockers]
        if missing_contracts:
            blockers.append(
                "FEATURE_SPEC_CONTRACT_BLOCKER: API/DB 계약이 없는 기능 — "
                + ", ".join(missing_contracts)
            )
        return state.copy(
            feature_registry=normalized,
            feature_spec_document=document,
            generation_blockers=list(dict.fromkeys(blockers)),
            qa_prd_blockers=(
                [*state.qa_prd_blockers, blockers[-1]]
                if missing_contracts else state.qa_prd_blockers
            ),
            qa_approved=False if missing_contracts else state.qa_approved,
            status_message="Feature Spec 에이전트 완료 — 기능 계약 확정",
        )
