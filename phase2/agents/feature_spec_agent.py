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
    # PM이 의미 없는 순번 ID만 제공한 경우에도 API/DB 계약은 필요하다.
    # 숫자를 제거한 feature001s 같은 이름은 sanitize_tables와 충돌하므로
    # 안정적인 fallback 식별자를 사용하고, 의미 있는 Registry 계약이 있으면
    # 아래에서 항상 그 계약을 우선한다.
    if re.fullmatch(r"feature_\d+", feature_id):
        return f"{feature_id}_records"
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
    parts = feature_id.lower().split(".")
    namespace = parts[-2] if len(parts) > 1 else ""
    base_path = f"/api/v1/auth/{parts[-1]}" if namespace == "auth" else f"/api/v1/{domain}"
    contracts = item.get("apiContract") if isinstance(item.get("apiContract"), list) else []
    generic_contract = any(
        re.fullmatch(r"/api/v1/feature_\d+_records/?", str(c.get("path") or ""))
        for c in contracts if isinstance(c, dict)
    )
    generic_table = any(
        re.fullmatch(r"feature_\d+_records", str(t))
        for t in (item.get("dbContract") or {}).get("tables", [])
    )
    if not contracts or (domain not in {"feature", "feature_records"} and (generic_contract or generic_table)):
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
  releaseGate, apiContract, dbContract를 포함
- PRD coreFeatures는 한 항목도 누락하지 말고 각각 하나의 feature object로 변환하세요. core 기능의 featureId/name/actions/apiContract/dbContract는 PM Registry와 PRD 사이에서 일치해야 합니다.
- 모든 backend 기능은 apiContract에 실제 method/path/action/featureId를 최소 하나 포함하세요. 저장·상태·이력·권한 데이터가 필요한 기능은 dbContract.tables를 최소 하나 포함하세요.
- dbContract.foreignKeys는 문자열 설명만 쓰지 말고 반드시 {{"table":"source_table","column":"...","references":{{"table":"...","column":"id"}}}} 객체로 작성하세요. source table은 dbContract.tables 중 하나여야 하며 실제 테이블 목록에 없는 대상을 참조하지 마세요.
- API path는 반드시 /api/v1/로 시작하는 영문 소문자 REST 경로여야 하며, 같은 method/path를 중복 생성하지 마세요.
- Registry에 이미 apiContract/dbContract가 있으면 재해석하거나 이름을 바꾸지 말고 그대로 보존하세요. 누락된 계약만 요구사항과 PRD를 근거로 보완하세요.
- supporting 기능은 실제 핵심 기능의 실행·보안·무결성에 필요한 경우에만 추가하고, 각 supporting 기능에 parentFeatureId와 reason을 작성하세요. supporting 기능을 coreFeatures 또는 P0으로 승격하지 마세요.
- core 기능의 featureId는 PM Registry와 일치시키고, supporting 기능은 부모 ID 뒤에 안정적인 suffix를 붙이세요
- 회원가입·로그인·회원정보 기능은 독립적인 auth.* 또는 profile.* ID를 사용하고, 핵심 기능에 대한 parentFeatureId와 추가 이유를 명시하세요.

[PM projectPlan]
{project_plan}

[PRD]
{prd}

[PM Registry 참고]
{registry}

출력:
{{"features":[{{"featureId":"inventory.register","name":"원두 재고 등록","source":"prd_core","parentFeatureId":null,"reason":"","actions":["create"],"priority":"P0","priorityScore":13,"releaseGate":false,"apiContract":[],"dbContract":{{"tables":[],"foreignKeys":[]}}}}]}}
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
        used_ids = {str(item.get("featureId") or "").strip() for item in source_registry if item.get("featureId")}
        output_ids: set[str] = set()
        normalized = []
        for item in features:
            if not isinstance(item, dict) or not str(item.get("name") or "").strip():
                continue
            item = dict(item)
            name = str(item.get("name") or "").strip()
            source = source_by_name.get(name) or source_by_id.get(str(item.get("featureId") or "").strip())
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
            if item["featureId"] in output_ids:
                if any(existing.get("name") == name for existing in normalized):
                    continue
                base_id = item["featureId"]
                suffix = 2
                while item["featureId"] in output_ids:
                    item["featureId"] = f"{base_id}.{suffix}"
                    suffix += 1
            output_ids.add(item["featureId"])
            # Common account/profile capabilities support the requested domain
            # feature; do not let an LLM promote them to P0 core features.
            if _is_supporting_feature(name):
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
            normalized.append(item)
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
            normalized.append(_ensure_required_contracts(_normalize_contracts(source_item)))
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
