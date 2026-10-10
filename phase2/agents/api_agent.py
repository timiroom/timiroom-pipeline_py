import asyncio
import operator
import json
import logging
import re
from typing import Annotated, TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.types import Send
from openai import AsyncOpenAI, InternalServerError, APITimeoutError, APIConnectionError

from phase2.agent_contract import canonical_api_method, endpoint_feature_table, feature_methods, feature_relation_kind, normalize_api_path, requires_auth
from phase2.quality_rules import contamination_reasons, has_placeholder, relevance_score
from phase2.feature_coverage import uncovered_features, undercovered_features, missing_features_note, strictly_uncovered_features
from phase2.feature_scope import backend_features
from phase2.feature_registry import normalize_feature_registry, registry_text
from phase2.json_utils import try_parse_json, has_suspicious_script
from phase2.llm_runtime import LlmRuntime
from phase2.state import PipelineState

# EXAONE이 가끔 오타로 내는 필드명 -> 정규화
_ENDPOINT_KEY_ALIASES = {
    "errorequestCodes": "errorCodes",
    "errorCode": "errorCodes",
    "successresponse": "successResponse",
    "requestbody": "requestBody",
}

logger = logging.getLogger(__name__)

# 생성 샘플링 파라미터
_TEMPERATURE = 1.0
_TOP_P = 0.95
_PRESENCE_PENALTY = 0.0
# 엔드포인트 목록(plan) 생성은 개수·커버리지 일관성이 중요한 구조 생성 단계 —
# temp=1.0은 실행마다 편차를 키우므로 이 단계만 낮춰 완성도/재현성을 높인다.
_PLAN_TEMPERATURE = 0.4

# plan 생성 배치 크기 — 한 번에 담당할 기능 수.
# 기능 전체(21개)를 한 프롬프트에 넣으면 EXAONE이 요구량의 1/3 수준에서 멈춘다(실측 12/42).
# Registry가 기능 경계를 제공하므로 초기 manager plan은 한 번만 만든다.
_PLAN_BATCH_SIZE = 8
_ENDPOINTS_PER_FEATURE = 2
_AUTH_ENDPOINT_COUNT = 4  # 회원가입/로그인/토큰갱신/로그아웃
_MAX_ENDPOINT_SOFT_CAP = 36

_AUTHENTICATION_DESC = (
    "JWT Bearer 토큰 방식. 로그인 시 발급받은 accessToken을 "
    "Authorization: Bearer <token> 헤더로 전송한다."
)

WORKER_SYSTEM = "JSON만 출력하세요. 설명·인사말·마크다운 코드블록 금지. { 로 시작해서 } 로 끝납니다."

MANAGER_SYSTEM = """당신은 시니어 백엔드 아키텍트 겸 API manager입니다.
JSON만 출력하세요. 설명·인사말·마크다운 코드블록 금지. { 로 시작해서 } 로 끝납니다."""

TARGETED_REPAIR_PROMPT = """현재 API 명세에서 검증 피드백이 지적한 엔드포인트만 수정하세요.
문제없는 엔드포인트는 절대 다시 작성하거나 삭제하지 마세요.
patches에는 수정된 엔드포인트 전체 객체를 넣고 method와 path를 식별자로 사용하세요.
피드백에 [STRUCTURED_REPAIR_TARGETS]가 있으면 artifactKey와 featureId가 일치하는 항목만 패치하세요.
각 patch는 기존 endpoint의 featureId와 action을 반드시 보존하거나 명시하세요.
쓰기 작업의 transactionRules는 원자적 변경 범위와 실패 시 롤백 규칙을 보존하거나 보완하세요.
새 엔드포인트가 정말 필요한 경우에만 추가하세요.
requestBody, successResponse, errorCodes에 '계약 기반', 'success: boolean',
'Registry 계약 endpoint' 같은 placeholder를 넣지 말고 실제 필드와 오류를 작성하세요.
JSON만 출력하세요.

[검증 피드백]
{feedback}

[현재 API 명세]
{api_spec}

출력 형식:
{{"patches": [{{"method":"PATCH","path":"/api/v1/example","description":"...","authRequired":true,"requestBody":"...","successResponse":"...","errorCodes":"..."}}]}}
"""

PLAN_PROMPT = """당신은 시니어 백엔드 아키텍트입니다.
아래 지시사항과 컨텍스트를 바탕으로, **담당 기능들**에 필요한 REST API 엔드포인트 목록(스켈레톤)을
설계하세요. 상세 스펙(requestBody/successResponse/errorCodes)은 이후 단계에서 채울 것이므로
지금은 엔드포인트의 method/path/description/인증 필요 여부만 결정하세요.

설계 규칙:
{auth_rule}- RESTful 설계: 명사형 복수형 경로, 반드시 영문 소문자·숫자·하이픈(-)만 사용 (예: /api/v1/book-clubs)
- path에 한글, 공백, 괄호(), 콜론(:), 쉼표 등은 절대 사용 금지 — 기능명이 한글이어도
  의미를 압축한 영문 리소스명으로 직접 번역해서 사용 (예: "재료 등록(바코드 스캔)" 기능 → /api/v1/ingredients, 절대 /api/v1/재료-등록-(바코드-스캔) 처럼 쓰지 말 것)
- 리소스별 CRUD 세트를 기계적으로 만들지 말고, 사용자의 핵심 워크플로우를 수행하는 엔드포인트만 설계
- 기능 목록에 생성·조회·수정·삭제·취소·신청·신청취소 등 서로 다른 행동이 있으면 각 행동에 대응하는 endpoint를 별도로 설계
- 한 endpoint의 description에 여러 행동을 나열하여 다른 endpoint를 대체하지 말 것
- 목록/상세/대시보드/히스토리처럼 비슷한 조회 API만 overview 또는 query parameter로 통합
- description은 반드시 "기능명: 행동 — 설명" 형태로 시작해 어느 기능의 어떤 행동에 대응하는지 드러낼 것
- Registry의 featureId를 각 endpoint skeleton에 반드시 그대로 넣고, action도 반드시 기록하세요. 하나의 endpoint는 하나의 featureId/action만 담당할 것
- Registry의 각 apiContract 항목은 누락 없이 1:1로 계획에 반영하세요. Registry에 계약이 있으면 새 해석보다 method/path/featureId/action 계약을 우선하세요.
- 모든 plan 항목은 featureId와 action을 가져야 하며, featureId 없는 endpoint는 출력하지 마세요. 계약을 만족시키는 endpoint가 부족하면 다른 기능과 합치지 말고 누락된 endpoint를 추가하세요.
- 상태 변경 endpoint는 정상 상태 전이, 선행 조건, 권한, 중복·기간·정원 초과 오류를 반영
- 목록 조회 엔드포인트는 페이지네이션이 필요함을 description에 명시
- 담당 기능은 {feature_count}개입니다. 권장 엔드포인트 수는 {min_endpoints}개 이상, {max_endpoints}개 이하입니다
- {max_endpoints}개를 넘기지 마세요. 초과가 필요하면 낮은 우선순위 CRUD/조회 파생 API를 합치거나 제외하세요
- 담당 기능 밖의 엔드포인트는 만들지 마세요 (다른 담당자가 설계합니다)
- 출력 직전에 담당 기능을 하나씩 대조하세요. 각 기능에 최소 1개의 endpoint가 있어야 하며,
  기능명에 독립 행동이 여러 개 포함되면 각 행동의 endpoint가 모두 있어야 합니다.
- 기존 endpoint를 설명만 고쳐서 여러 행동을 커버한 것으로 간주하지 마세요. 누락 행동은 별도 method/path로 추가하세요.
{existing_note}
응답 형식 (JSON만):
{{
  "plan": [
    {{
      "method": "GET 또는 POST 또는 PUT 또는 DELETE 또는 PATCH",
      "path": "/api/v1/영문-리소스명 (한글 금지, 예: /api/v1/ingredients)",
      "featureId": "PM Registry의 featureId",
      "action": "Registry apiContract의 action",
      "description": "기능명: 이 API가 하는 일 한 줄 설명",
      "authRequired": true 또는 false
    }}
  ]
}}

지시사항:
{instruction}

컨텍스트 (DB 스키마 포함):
{context}

PM 기능 계약 Registry (각 featureId/action의 apiContract를 빠짐없이 반영):
{feature_registry}

담당 기능 목록 ({feature_count}개):
{feature_str}

PRD 요구사항 문서 (Registry와 충돌하면 Registry의 featureId/action 계약을 우선):
{prd_document}
"""

_AUTH_RULE = (
    "- 회원가입, 로그인, 토큰 갱신, 로그아웃 등 인증 흐름에 필요한 엔드포인트를 반드시 포함\n"
)


def _existing_paths_note(paths: list[str] | None) -> str:
    """이미 설계된 'METHOD path' 목록을 프롬프트에 주입 — 배치 간 중복 설계를 줄인다.
    배치들이 서로 모르는 채 같은 리소스를 설계하면 병합 시 중복 제거로 총 개수가 깎인다
    (실측: 배치 3개 목표 24개 → 병합 후 11개)."""
    if not paths:
        return ""
    listed = "\n".join(f"  - {p}" for p in paths)
    return (
        "\n- ⚠️ 아래는 이미 설계된 엔드포인트입니다. 똑같은 method+path를 다시 만들지 말고,\n"
        "  담당 기능에 필요한데 아직 없는 것만 새로 설계하세요:\n" + listed + "\n"
    )


def _parse_endpoint_text(raw: str, skeleton: dict) -> dict | None:
    fields = {
        "METHOD": "method", "PATH": "path", "DESCRIPTION": "description",
        "AUTH_REQUIRED": "authRequired", "REQUEST_BODY": "requestBody",
        "SUCCESS_RESPONSE": "successResponse", "ERROR_CODES": "errorCodes",
    }
    data = {}
    current = None
    clean_raw = re.sub(r"^\s*SELF_CHECK\s*:\s*PASS\s*$", "", raw or "", flags=re.I | re.M)
    for line in clean_raw.replace("\r", "").splitlines():
        line = line.strip().strip("`*- ")
        if not line:
            continue
        matched = False
        upper = line.upper()
        for marker, field in fields.items():
            marker_match = upper.startswith(marker + ":")
            if marker == "REQUEST_BODY":
                marker_match = marker_match or (":" in line and upper.startswith("REQUEST_BODY"))
            elif marker == "ERROR_CODES":
                marker_match = marker_match or (":" in line and upper.startswith("ERROR_C"))
            if marker_match:
                data[field] = line.split(":", 1)[1].strip()
                current = field
                matched = True
                break
        if not matched and current:
            data[current] = f"{data[current]} {line}".strip()
    if not data:
        return None
    data["method"] = str(skeleton.get("method") or data.get("method") or "GET").upper()
    data["path"] = str(skeleton.get("path") or data.get("path") or "/api/v1/unknown")
    data["description"] = data.get("description") or skeleton.get("description") or ""
    auth = str(data.get("authRequired", skeleton.get("authRequired", True))).lower()
    data["authRequired"] = auth in {"true", "1", "yes", "필요"}
    if not data.get("description"):
        return None
    data["requestBody"] = data.get("requestBody") or ("없음" if data["method"] == "GET" else "요청 데이터")
    data["successResponse"] = data.get("successResponse") or "성공 응답"
    data["errorCodes"] = data.get("errorCodes") or "400 - 잘못된 요청, 500 - 서버 오류"
    for field in ("featureId", "action", "featureName"):
        if field in skeleton:
            data[field] = skeleton[field]
    return data


def _feature_resource_slug(feature: str, index: int) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", str(feature or "").lower()).strip("-")
    return f"{slug[:40]}-{index + 1}" if slug else f"feature-{index + 1}"


def _feature_methods(feature: str) -> tuple[str, ...]:
    return tuple(feature_methods(feature))


def _api_feature_mappings(endpoints: list[dict], db_schema: str, feature_list: list[str]) -> list[dict]:
    db = try_parse_json(db_schema or "{}") or {}
    db_map = {
        str(item.get("featureName") or ""): str(item.get("table") or "")
        for item in db.get("featureMappings") or [] if isinstance(item, dict)
    }
    result = []
    for feature in feature_list or []:
        if str(feature) == "사용자별 데이터 접근 제어":
            selected = [ep for ep in endpoints if ep.get("authRequired") and not str(ep.get("path") or "").startswith("/api/v1/auth/")]
        else:
            selected = [ep for ep in endpoints if str(ep.get("featureName") or "") == str(feature)]
            table_name = db_map.get(str(feature), "")
            if not selected and table_name:
                prefix = f"/api/v1/{table_name.replace('_', '-')}"
                selected = [ep for ep in endpoints if str(ep.get("path") or "").startswith(prefix)]
        result.append({
            "featureName": str(feature), "table": db_map.get(str(feature), ""),
            "operations": [
                {"method": str(ep.get("method") or ""), "path": str(ep.get("path") or "")}
                for ep in selected
            ],
        })
    return result


def _api_type(db_type: str) -> str:
    value = str(db_type or "").upper()
    if value in {"BIGINT", "INT", "INTEGER", "SMALLINT"}:
        return "integer"
    if value.startswith(("DECIMAL", "NUMERIC", "REAL", "DOUBLE")):
        return "number"
    if value == "BOOLEAN":
        return "boolean"
    if value == "JSONB":
        return "object"
    return "string"


def _align_endpoints_to_db(
    endpoints: list[dict], db_schema: str, prd_document: str = "", requirement_text: str = "",
    *, declared_table_name: str | None = None,
) -> list[dict]:
    """API manager owns the final contract and derives field names from the finalized ERD."""
    db = try_parse_json(db_schema or "{}") or {}
    auth_required_by_prd = requires_auth(prd_document, requirement_text=requirement_text)
    tables = {str(t.get("name") or ""): t for t in db.get("tables") or [] if isinstance(t, dict)}
    aligned = []
    seen: set[tuple[str, str]] = set()
    for endpoint in endpoints or []:
        ep = dict(endpoint)
        path = str(ep.get("path") or "")
        if path.startswith("/api/v1/auth/"):
            auth_ep = _deterministic_endpoint(ep)
            if path.endswith("/refresh"):
                auth_ep["transactionRules"] = "기존 Refresh Token의 해시·만료·회수 상태를 검증한 뒤 새 토큰으로 원자적으로 교체한다."
            elif path.endswith("/logout"):
                auth_ep["transactionRules"] = "전달된 Refresh Token 해시의 revoked_at을 기록하고 이후 재사용을 거부한다."
            key = (auth_ep["method"], auth_ep["path"])
            if key not in seen:
                seen.add(key)
                aligned.append(auth_ep)
            continue
        if path == "/api/v1/users/me":
            own_ep = _deterministic_endpoint(ep)
            own_ep["authRequired"] = True
            key = (own_ep["method"], own_ep["path"])
            if key not in seen:
                seen.add(key)
                aligned.append(own_ep)
            continue
        slug = path.removeprefix("/api/v1/").split("/", 1)[0]
        declared_table = tables.get(declared_table_name or "")
        table = declared_table or tables.get(slug.replace("-", "_"))
        feature_label = str(ep.get("featureName") or ep.get("description") or "").split(":", 1)[0].strip()
        if table and not declared_table and (_column_names(table) & {"credential_hash", "password_hash", "login_id"}):
            if not any(token in feature_label.lower() for token in ("로그인", "회원", "계정", "auth", "user account")):
                table = None
        if not table:
            relation_kind = feature_relation_kind(feature_label)
            role_candidates = []
            for candidate in tables.values():
                candidate_columns = _column_names(candidate)
                candidate_refs = _referenced_tables(candidate)
                if candidate_columns & {"credential_hash", "password_hash", "login_id"}:
                    continue
                if relation_kind == "association" and len(candidate_refs) < 2:
                    continue
                if relation_kind == "event" and not (
                    candidate_refs and candidate_columns & {"status", "state", "recorded_at", "occurred_at", "event_at"}
                ):
                    continue
                if relation_kind == "aggregate" and len(candidate_refs) >= 2:
                    continue
                score = max(
                    relevance_score(feature_label, f"{candidate.get('name', '')} {candidate.get('description', '')}"),
                    relevance_score(str(candidate.get("description") or ""), feature_label),
                )
                role_candidates.append((score, candidate))
            role_best = max((score for score, _candidate in role_candidates), default=0.0)
            role_winners = [candidate for score, candidate in role_candidates if score == role_best and score >= 0.2]
            if len(role_winners) == 1:
                table = role_winners[0]
            feature_matches = [
                candidate for candidate in tables.values()
                if feature_label and feature_label in str(candidate.get("description") or "")
            ]
            if not table and len(feature_matches) == 1:
                table = feature_matches[0]
            slug_stems = _english_resource_stems(slug)
            lexical = [
                (len(slug_stems & _english_resource_stems(name)), candidate)
                for name, candidate in tables.items()
            ]
            lexical_best = max((score for score, _candidate in lexical), default=0)
            lexical_winners = [candidate for score, candidate in lexical if score == lexical_best and score > 0]
            if not table and len(lexical_winners) == 1:
                table = lexical_winners[0]
            # PRD/API/DBA run independently, so two accurate English translations
            # can still differ (expiry vs expiration). Match through the original
            # Korean feature text retained in both descriptions, then normalize the
            # API path to the ERD table identifier.
            if not table:
                scored = [
                    (max(
                        relevance_score(str(candidate.get("description") or ""), str(ep.get("description") or "")),
                        relevance_score(str(ep.get("description") or ""), str(candidate.get("description") or "")),
                    ), candidate)
                    for candidate in tables.values()
                ]
                best_score = max((score for score, _candidate in scored), default=0.0)
                winners = [candidate for score, candidate in scored if score == best_score and score >= 0.25]
                if len(winners) == 1:
                    table = winners[0]
            if table:
                normalized_slug = str(table.get("name") or "").replace("_", "-")
                path = path.replace(f"/api/v1/{slug}", f"/api/v1/{normalized_slug}", 1)
                ep["path"] = path
                slug = normalized_slug
        # A catalog endpoint must not accidentally own user-specific mutable state.
        # A declared table remains authoritative. Inference needs endpoint intent;
        # an incoming FK alone does not turn a catalog read into personal data.
        table_columns = _column_names(table) if table else set()
        is_stateful_resource = bool(table and "user_id" in table_columns and table_columns & _STATE_FIELD_NAMES)
        if table and not declared_table and not is_stateful_resource and "user_id" not in table_columns:
            aggregate = _stateful_aggregate_for_catalog(str(table.get("name") or ""), tables)
            endpoint_meaning = str(ep.get("featureName") or ep.get("description") or "")
            catalog_score = max(
                relevance_score(str(table.get("description") or ""), endpoint_meaning),
                relevance_score(endpoint_meaning, str(table.get("description") or "")),
            )
            aggregate_score = max(
                relevance_score(str(aggregate.get("description") or ""), endpoint_meaning),
                relevance_score(endpoint_meaning, str(aggregate.get("description") or "")),
            ) if aggregate else 0.0
            requested_fields = set(re.findall(r"\b([a-z][a-z0-9_]*)\s*:", str(ep.get("requestBody") or "")))
            aggregate_state_fields = (_column_names(aggregate) - table_columns) & _STATE_FIELD_NAMES if aggregate else set()
            stateful_intent = (
                aggregate_score > catalog_score
                or bool(requested_fields & aggregate_state_fields)
                or (bool(ep.get("authRequired")) and bool(re.search(
                    r"등록|추가|저장|수정|변경|삭제|소비|\b(?:add|create|save|update|delete|consume)\b",
                    endpoint_meaning, re.I,
                )))
            )
            if aggregate and stateful_intent:
                old_slug = slug
                table = aggregate
                slug = str(table["name"]).replace("_", "-")
                path = path.replace(f"/api/v1/{old_slug}", f"/api/v1/{slug}", 1)
                ep["path"] = path
        if not table:
            fallback_ep = _deterministic_endpoint(ep)
            if auth_required_by_prd:
                fallback_ep["authRequired"] = True
            if str(fallback_ep.get("method") or "").upper() == "GET" and "{" not in str(fallback_ep.get("path") or ""):
                fallback_ep["successResponse"] = "items: array<object> — 조회 결과가 없으면 [], total: integer — 전체 개수"
                fallback_ep["errorCodes"] = _error_contract(bool(fallback_ep.get("authRequired")))
            if str(fallback_ep.get("method") or "").upper() in {"POST", "PUT", "PATCH", "DELETE"}:
                fallback_ep.setdefault(
                    "transactionRules",
                    "대상 존재·권한·업무 상태를 검증하고 변경을 단일 트랜잭션으로 처리하며 실패 시 롤백한다.",
                )
            aligned.append(fallback_ep)
            continue
        if auth_required_by_prd:
            ep["authRequired"] = True
        columns = [c for c in table.get("columns") or [] if isinstance(c, dict) and c.get("name")]
        business = [c for c in columns if c["name"] not in {"id", "created_at", "updated_at"}]
        # Actor identity is derived from the configured authentication context,
        # never invented from a technology suggestion in the PRD.
        endpoint_feature = str(ep.get("featureName") or ep.get("description") or "").split(":", 1)[0]
        association_contract = feature_relation_kind(endpoint_feature) == "association"
        # A stored credential hash is derived on the server; a client never submits it.
        writable = [
            c for c in business
            if c["name"] not in {"password_hash", "credential_hash"}
            and (c["name"] != "user_id" or association_contract)
        ]
        method = str(ep.get("method") or "GET").upper()
        if method in {"PATCH", "PUT", "DELETE"} and "{" not in path:
            path = path.rstrip("/") + "/{id}"
            ep["path"] = path
        if method in _BODYLESS_METHODS:
            ep["requestBody"] = "없음"
        else:
            selected = writable
            ep["requestBody"] = ", ".join(
                f"{c['name']}: {_api_type(c.get('type'))} — {table['name']} 필드" for c in selected
            ) or "payload: object — 기능 입력"
        if method == "GET":
            is_collection = "{" not in path
            if is_collection:
                ep["successResponse"] = f"items: array<{table['name']}> — 조회 결과가 없으면 [], total: integer — 전체 개수"
            else:
                ep["successResponse"] = f"item: {table['name']} — 단일 조회 결과"
        elif method == "DELETE":
            ep["successResponse"] = "success: boolean — 삭제 성공 여부, deletedId: integer — 삭제된 ID"
        elif method == "PATCH":
            ep["successResponse"] = "id: integer — 수정된 ID, updated_at: string — 수정 시각"
        else:
            ep["successResponse"] = "id: integer — 생성된 ID, created_at: string — 생성 시각"
        collection_get = method == "GET" and "{" not in path
        ep["errorCodes"] = _error_contract(
            bool(ep.get("authRequired")), target_required="{" in path,
        )
        if method == "POST":
            ep.setdefault(
                "transactionRules",
                f"{table['name']} 입력 검증과 생성을 단일 DB 트랜잭션으로 처리하고 실패 시 전체를 롤백한다.",
            )
        elif method == "PATCH":
            ep.setdefault(
                "transactionRules",
                f"{table['name']} 대상 존재와 현재 상태를 확인한 뒤 변경을 단일 DB 트랜잭션으로 처리한다.",
            )
        elif method == "DELETE":
            ep.setdefault(
                "transactionRules",
                f"{table['name']} 대상 존재와 참조 무결성을 확인한 뒤 삭제를 단일 DB 트랜잭션으로 처리한다.",
            )

        table_name = str(table.get("name") or "")
        table_columns_set = _column_names(table)
        occurrence_fields = {
            "occurred_at", "recorded_at", "event_at", "completed_at", "processed_at",
            "started_at", "ended_at", "effective_at",
        }
        event_like = bool(_referenced_tables(table)) and (
            bool(table_columns_set & occurrence_fields)
            or any(re.search(r"(?:^delta_|_delta$|_change$|_used$|^previous_|^new_)", name)
                   for name in table_columns_set)
            or (
                len(_referenced_tables(table)) >= 2
                and bool(table_columns_set & {"status", "state"})
                and any(
                    _column_names(tables[target]) & {"status", "state"}
                    for target in _referenced_tables(table) if target in tables
                )
            )
        )
        if event_like:
            aggregate_targets = [
                target for target in _referenced_tables(table)
                if (target in tables
                    and (_column_names(tables[target]) & _STATE_FIELD_NAMES))
            ]
            # An event can reference both a root aggregate and a stateful child
            # assignment. Prefer the ancestor referenced by another candidate.
            if len(aggregate_targets) > 1:
                root_targets = [
                    candidate for candidate in aggregate_targets
                    if any(
                        candidate in _referenced_tables(tables[other])
                        for other in aggregate_targets
                        if other != candidate
                    )
                ]
                if len(root_targets) == 1:
                    aggregate_targets = root_targets
            if len(aggregate_targets) == 1 and method in {"POST", "PATCH", "DELETE"}:
                aggregate_name = aggregate_targets[0]
                if method == "POST":
                    rule = f"{table_name} 생성과 {aggregate_name} 상태 반영을 단일 DB 트랜잭션으로 처리하고 허용되지 않은 상태는 409로 거부한다."
                elif method == "PATCH":
                    rule = f"기존 기록의 {aggregate_name} 반영분을 먼저 되돌린 뒤 변경값을 다시 반영하며 전체를 단일 트랜잭션으로 처리한다."
                else:
                    rule = f"삭제 전 기록이 반영한 {aggregate_name} 상태를 역보정한 뒤 기록을 삭제하며 전체를 단일 트랜잭션으로 처리한다."
                ep["transactionRules"] = rule
                ep["errorCodes"] = ep["errorCodes"].replace("500 —", "409 — 상태 또는 선행 업무 충돌, 500 —")
        unique_indexes = [str(index) for index in table.get("indexes") or [] if re.search(r"CREATE\s+UNIQUE\s+INDEX", str(index), re.I)]
        stateful_targets = [
            target for target in _referenced_tables(table)
            if target in tables and (_column_names(tables[target]) & _STATE_FIELD_NAMES)
        ]
        if method in {"POST", "PATCH", "DELETE"} and (unique_indexes or stateful_targets):
            rules = []
            if unique_indexes:
                rules.append("동일한 주체와 대상의 중복 요청을 UNIQUE 제약으로 차단하고 충돌 시 409를 반환한다")
            if stateful_targets:
                rules.append(f"참조 엔티티 {stateful_targets[0]}의 존재와 현재 상태를 확인한 뒤 단일 트랜잭션으로 처리한다")
            existing_rule = str(ep.get("transactionRules") or "").strip()
            existing_sentences = {
                sentence.strip().rstrip(".") for sentence in re.split(r"[.;]", existing_rule) if sentence.strip()
            }
            new_rules = [rule.rstrip(".") for rule in rules if rule.rstrip(".") not in existing_sentences]
            supplemental = "; ".join(new_rules)
            existing_part = f"{existing_rule.rstrip('.')}." if existing_rule else ""
            ep["transactionRules"] = " ".join(
                part for part in (existing_part, f"{supplemental}." if supplemental else "") if part
            ).strip()
            if "409" not in ep["errorCodes"]:
                ep["errorCodes"] = ep["errorCodes"].replace("500 —", "409 — 업무 상태 또는 중복 충돌, 500 —")
        _normalize_parameters(ep)
        key = (method, str(ep.get("path") or ""))
        if key not in seen:
            seen.add(key)
            aligned.append(ep)
    return aligned


def _ensure_auth_endpoints(endpoints: list[dict], auth_required: bool) -> list[dict]:
    if not auth_required:
        return endpoints
    required = [
        {
            "method": "POST", "path": "/api/v1/auth/signup", "featureName": "회원 가입 및 로그인",
            "description": "회원 가입 및 로그인: 신규 계정 생성", "authRequired": False,
            "requestBody": "email: string — 로그인 식별자, password: string — 평문 전송 후 서버에서 해시",
            "successResponse": "id: integer — 사용자 ID, status: string — 계정 상태, created_at: string — 가입 시각",
            "errorCodes": "400 — 입력 검증 실패, 409 — 로그인 식별자 중복, 500 — 서버 오류",
            "transactionRules": "사용자와 자격 증명을 한 DB 트랜잭션으로 생성하고 실패 시 전체를 롤백한다.",
        },
        {
            "method": "POST", "path": "/api/v1/auth/login", "featureName": "회원 가입 및 로그인",
            "description": "회원 가입 및 로그인: 자격 증명 검증 및 토큰 발급", "authRequired": False,
            "requestBody": "email: string — 로그인 식별자, password: string — 자격 증명",
            "successResponse": "accessToken: string — 접근 토큰, refreshToken: string — 갱신 토큰, expiresIn: integer — 만료 초",
            "errorCodes": "400 — 입력 검증 실패, 401 — 자격 증명 불일치, 403 — 비활성 계정, 500 — 서버 오류",
            "transactionRules": "자격 증명 검증 후 Refresh Token 해시 저장과 토큰 발급을 일관되게 처리한다.",
        },
        {
            "method": "POST", "path": "/api/v1/auth/refresh", "featureName": "인증 세션 관리",
            "description": "인증 세션 관리: 접근 토큰 갱신", "authRequired": False,
            "requestBody": "refreshToken: string — 기존 갱신 토큰",
            "successResponse": "accessToken: string — 새 접근 토큰, refreshToken: string — 교체된 갱신 토큰, expiresIn: integer — 만료 초",
            "errorCodes": "400 — 입력 검증 실패, 401 — 만료·회수·재사용 토큰, 500 — 서버 오류",
            "transactionRules": "기존 Refresh Token을 회수하고 새 토큰 해시를 한 트랜잭션으로 저장한다.",
        },
        {
            "method": "POST", "path": "/api/v1/auth/logout", "featureName": "인증 세션 관리",
            "description": "인증 세션 관리: 현재 갱신 토큰 회수", "authRequired": True,
            "requestBody": "refreshToken: string — 회수할 갱신 토큰",
            "successResponse": "success: boolean — 로그아웃 성공 여부",
            "errorCodes": "400 — 입력 검증 실패, 401 — 인증 실패, 403 — 세션 소유자 불일치, 500 — 서버 오류",
            "transactionRules": "토큰 해시의 revoked_at을 원자적으로 기록하고 이후 재사용을 거부한다.",
        },
        {
            "method": "GET", "path": "/api/v1/users/me", "featureName": "내 정보 및 계정 관리",
            "description": "내 정보 및 계정 관리: 인증된 본인 정보 조회", "authRequired": True,
            "requestBody": "없음", "successResponse": "item: users — 인증된 사용자 정보",
            "errorCodes": "401 — 인증 실패, 404 — 사용자 없음, 500 — 서버 오류",
        },
        {
            "method": "PATCH", "path": "/api/v1/users/me", "featureName": "내 정보 및 계정 관리",
            "description": "내 정보 및 계정 관리: 본인 정보 수정", "authRequired": True,
            "requestBody": "email: string — 변경할 로그인 식별자", "successResponse": "id: integer — 사용자 ID, updated_at: string — 수정 시각",
            "errorCodes": "400 — 입력 검증 실패, 401 — 인증 실패, 409 — 로그인 식별자 중복, 500 — 서버 오류",
            "transactionRules": "인증 주체와 대상 사용자가 같은지 확인한 뒤 한 트랜잭션으로 수정한다.",
        },
        {
            "method": "DELETE", "path": "/api/v1/users/me", "featureName": "내 정보 및 계정 관리",
            "description": "내 정보 및 계정 관리: 회원 탈퇴와 세션 회수", "authRequired": True,
            "requestBody": "없음", "successResponse": "success: boolean — 탈퇴 완료 여부, deletedId: integer — 사용자 ID",
            "errorCodes": "401 — 인증 실패, 409 — 이미 탈퇴한 계정, 500 — 서버 오류",
            "transactionRules": "계정 상태를 WITHDRAWN으로 전환하고 모든 활성 세션을 같은 트랜잭션에서 회수한다.",
        },
    ]
    canonical = {(ep["method"], ep["path"]): ep for ep in required}
    identity_features = {
        "회원 가입 및 로그인", "인증 세션 관리", "내 정보 및 계정 관리", "사용자별 데이터 접근 제어",
    }
    noncanonical_identity_slugs = (
        "/user-authentication", "/authentication-sessions", "/user-management",
        "/user-data-access-control", "/user-profiles",
    )
    result = []
    seen = set()
    for endpoint in endpoints:
        if not isinstance(endpoint, dict):
            continue
        key = (str(endpoint.get("method") or "").upper(), str(endpoint.get("path") or ""))
        if key not in canonical and (
            str(endpoint.get("featureName") or "") in identity_features
            or any(slug in key[1] for slug in noncanonical_identity_slugs)
            or bool(re.match(r"^/api/v\d+/auth/", key[1]))
            or bool(re.match(r"^/api/v1/users/(?:register|login|logout|refresh-token)$", key[1]))
            or key[1] in {"/api/v1/users/me", "/api/v1/users/profile"}
        ):
            continue
        result.append({**endpoint, **canonical.get(key, {})})
        seen.add(key)
    result.extend(ep for key, ep in canonical.items() if key not in seen)
    return result


def _ensure_feature_endpoint_groups(
    endpoints: list[dict], db_schema: str, feature_list: list[str], prd_document: str = "",
    requirement_text: str = "",
) -> list[dict]:
    """Add only missing minimal endpoint groups from the finalized ERD.

    Parallel API generation can omit an entire feature batch. A selective repair
    cannot align endpoints that do not exist, so reconstruct GET/POST/PATCH
    skeletons for the uniquely matching ERD resource and let the normal contract
    aligner fill concrete fields. No domain-specific table name is assumed.
    """
    db = try_parse_json(db_schema or "{}") or {}
    prd = try_parse_json(prd_document or "{}") or {}
    feature_specs = {
        str(item.get("name") or ""): item
        for item in (prd.get("coreFeatures") or [])
        if isinstance(item, dict) and item.get("name")
    } if isinstance(prd, dict) else {}
    tables = [table for table in db.get("tables") or [] if isinstance(table, dict) and table.get("name")]
    mapped_tables = {
        str(item.get("featureName") or ""): str(item.get("table") or "")
        for item in db.get("featureMappings") or [] if isinstance(item, dict) and item.get("table")
    }
    result = [dict(endpoint) for endpoint in (endpoints or []) if isinstance(endpoint, dict)]
    for feature in feature_list or []:
        if str(feature) in {
            "회원 가입 및 로그인", "인증 세션 관리", "내 정보 및 계정 관리", "사용자별 데이터 접근 제어",
        }:
            continue
        spec = feature_specs.get(str(feature), {})
        feature_context = " ".join([
            str(feature), str(spec.get("parentFeature") or ""),
            str(spec.get("rationale") or ""),
            " ".join(str(value) for value in spec.get("actions") or []),
            " ".join(str(value) for value in spec.get("dataRequirements") or []),
        ])
        identity_scope = any(token in feature_context.lower() for token in (
            "회원", "계정", "로그인", "인증", "사용자 관리",
            "member", "account", "login", "authentication", "user management",
        ))
        kind = feature_relation_kind(str(feature))
        explicitly_mapped = next(
            (table for table in tables if str(table.get("name") or "") == mapped_tables.get(str(feature))),
            None,
        )
        scored = []
        for table in tables:
            table_name = str(table.get("name") or "")
            singular_name = table_name.rstrip("s")
            if not identity_scope and singular_name in {"user", "member", "account", "customer", "principal"}:
                continue
            table_text = f"{table.get('name', '')} {table.get('description', '')}"
            referenced_tables = _referenced_tables(table)
            column_names = _column_names(table)
            score = max(
                relevance_score(feature_context, table_text),
                relevance_score(table_text, feature_context),
            )
            if str(feature) and str(feature) in str(table.get("description") or ""):
                score = max(score, 1.0)
            if kind == "association":
                # Association resources are identified primarily by topology, not
                # by a scenario-specific English table name. Naming is only a
                # tie-breaker when the ERD contains multiple two-sided resources.
                if len(referenced_tables) >= 2:
                    score += 1.0
                if any(token in table_name for token in (
                    "assign", "member", "link", "mapping", "join", "relation",
                )):
                    score += 0.75
                if any(token in table_name for token in ("record", "history", "log", "event")):
                    score -= 0.5
            if kind == "event" and any(
                token in table_name for token in ("record", "history", "log", "event")
            ):
                score += 1.0
            if kind == "event" and referenced_tables and column_names & {
                "status", "state", "recorded_at", "occurred_at", "event_at",
            }:
                score += 0.5
            scored.append((score, table))
        best = max((score for score, _table in scored), default=0.0)
        winners = [explicitly_mapped] if explicitly_mapped else [table for score, table in scored if score == best and score >= 0.25]
        if len(winners) != 1:
            continue
        table = winners[0]
        slug = str(table["name"]).replace("_", "-")
        collection_path = f"/api/v1/{slug}"
        normalized_feature = " ".join(str(feature).lower().split())
        related = []
        for endpoint in result:
            endpoint_path = str(endpoint.get("path") or "")
            explicit_feature = " ".join(
                str(endpoint.get("featureName") or "").lower().split()
            )
            if (
                explicit_feature == normalized_feature
                or (not explicit_feature and endpoint_path.startswith(collection_path))
            ):
                related.append(endpoint)
        existing_methods = {str(endpoint.get("method") or "").upper() for endpoint in related}
        for method in _feature_methods(str(feature)):
            if method in existing_methods:
                matching = [
                    endpoint for endpoint in related
                    if str(endpoint.get("method") or "").upper() == method
                ]
                if method in {"GET", "POST"}:
                    preferred = [
                        endpoint for endpoint in matching
                        if str(endpoint.get("path") or "").rstrip("/") == collection_path
                    ]
                else:
                    preferred = [
                        endpoint for endpoint in matching
                        if str(endpoint.get("path") or "").startswith(collection_path + "/")
                        and "{" in str(endpoint.get("path") or "")
                    ]
                owner = preferred[0] if len(preferred) == 1 else (
                    matching[0] if len(matching) == 1 else None
                )
                if owner is not None and not str(owner.get("featureName") or "").strip():
                    owner["featureName"] = str(feature)
                    if str(feature) not in str(owner.get("description") or ""):
                        owner["description"] = f"{feature}: {owner.get('description', '')}".strip()
                continue
            path = collection_path if method in {"GET", "POST"} else f"{collection_path}/{{id}}"
            result.append(_deterministic_endpoint({
                "method": method,
                "path": path,
                "description": f"{feature}: {'목록 조회' if method == 'GET' else '생성 또는 실행' if method == 'POST' else '부분 수정'}",
                "featureName": str(feature),
                "authRequired": requires_auth(prd_document, [str(feature)], requirement_text),
            }))
            logger.warning("API 누락 기능 계약 결정적 보강 — %s %s (%s)", method, path, feature)
    return result

ENDPOINT_SPEC_PROMPT = """당신은 시니어 백엔드 개발자입니다.
아래 엔드포인트 스켈레톤 1개에 대한 상세 REST API 스펙을 JSON으로 작성하세요.

담당 엔드포인트:
- featureId: {feature_id}
- action: Registry 계약의 action
- method: {method}
- path: {path}
- description: {description}
- authRequired: {auth_required}

설계 규칙:
- requestBody와 successResponse는 반드시 문자열로 작성 (객체 금지).
  각 필드를 "필드명: 타입 — 설명" 형태로 쉼표로 이어서 나열한다.
- 쿼리 파라미터·경로 파라미터는 requestBody가 아니라 parameters 배열에 넣는다.
  GET·DELETE처럼 본문이 없는 메서드의 requestBody는 정확히 "없음" 이라고만 쓴다.
- 목록 조회 엔드포인트라면 parameters에 page, size를 반드시 포함하고,
  정렬·필터가 필요하면 그것도 파라미터로 추가한다
- 경로에 {{id}} 같은 자리표시자가 있으면 parameters에 in="path"로 반드시 포함
- successResponse는 DB 스키마의 실제 컬럼명과 어긋나지 않게 작성
- errorCodes는 "코드 — 설명" 형식으로 2개 이상 작성
- transactionRules에는 쓰기 작업의 원자적 변경 범위와 실패 시 롤백 규칙을 명시한다.
- 위 지시문의 예시 문구("없으면 없음", "필드명" 등)를 값에 그대로 옮겨 쓰지 말 것 — 실제 내용만

응답 형식 (JSON만, method/path/description/authRequired는 위 값 그대로 유지):
{{
  "method": "{method}",
  "path": "{path}",
  "featureId": "{feature_id}",
  "action": "Registry 계약의 action",
  "description": "{description}",
  "authRequired": {auth_required},
  "parameters": [
    {{"in": "query 또는 path", "name": "파라미터명", "type": "string/integer/boolean", "required": true 또는 false, "description": "설명"}}
  ],
  "requestBody": "본문이 없으면 없음",
  "successResponse": "반환 필드들",
  "errorCodes": "401 — 인증 실패, 404 — 리소스 없음",
  "transactionRules": "이 작업의 원자적 변경 범위와 실패 처리"
}}

컨텍스트 (DB 스키마 포함):
{context}
"""

MANAGER_REVIEW_PROMPT = """아래는 sub-agent들이 작성한 API 엔드포인트 상세 스펙 전체입니다.

=== 엔드포인트 목록 ===
{endpoints_json}
=================

=== PM 기능 계약 (source of truth) ===
{feature_registry}
=====================================

=== 검증 기준 ===
- 기능 목록의 모든 기능에 대응하는 엔드포인트가 존재하는가?
- 각 기능의 생성·조회·수정·삭제·취소·신청·확정 등 독립 행동이 모두 별도 endpoint로 존재하는가?
- path가 서로 중복되지 않는가?
- requestBody/successResponse/errorCodes가 비어있거나 placeholder가 남아있지 않은가?
- DB 스키마(컨텍스트)와 필드명이 어긋나지 않는가?
- path는 반드시 영문 소문자·숫자·하이픈(-)만 사용 (한글/공백/괄호/콜론 절대 금지)

기능 목록: {feature_str}
컨텍스트 (DB 스키마 포함): {context}
{missing_note}

위 기준을 충족하지 못했거나 내용이 이상한 엔드포인트가 있으면 "method path"를 키로 하여
해당 엔드포인트만 새로 작성해 patches에 담으세요. 문제 없는 엔드포인트는 patches에 포함하지 마세요.
[결정론적 커버리지 검사]에 나열된 기능이 있다면 반드시 그 기능을 위한 엔드포인트를 새로 만들어 patches에 추가하세요.

JSON:
{{
  "prdIssues": "API 설계에 필요한데 PRD에 누락되거나 모순된 요구사항. 없으면 빈 문자열",
  "patches": {{
    "METHOD /api/v1/path": {{"method":"...","path":"...","description":"...","authRequired":true,"requestBody":"...","successResponse":"...","errorCodes":"..."}}
  }}
}}

PRD 자체에 문제가 없으면 prdIssues는 빈 문자열로 두세요.
문제되는 엔드포인트가 하나도 없으면 patches는 빈 객체로 응답하세요.
"""


def finalize_api_contracts(
    endpoints: list[dict], db_schema: str, prd_document: str = "",
    feature_list: list[str] | None = None, registry: list[dict] | None = None,
    requirement_text: str = "",
) -> list[dict]:
    """Align completed drafts to the ERD while retaining explicit registry routes.

    DBA and API workers can finish concurrently. This deterministic pass is also
    called after their results are joined, when the final schema is available.
    """
    registry = registry or []
    prepared = _normalize_endpoints([dict(endpoint) for endpoint in endpoints if isinstance(endpoint, dict)])
    if registry:
        prepared = _annotate_feature_ids(_ensure_contract_endpoints(prepared, registry), registry)
    schema = try_parse_json(db_schema or "{}")
    if not isinstance(schema, dict) or not schema.get("tables"):
        return prepared
    table_names = {str(table.get("name") or "") for table in schema["tables"] if isinstance(table, dict)}
    by_id = {str(item.get("featureId") or item.get("id") or ""): item for item in registry}
    # QA validates each endpoint against its feature's mapped table, so fields must come from the same table.
    feature_mappings = {
        str(item.get("featureId") or ""): item
        for item in schema.get("featureMappings") or [] if isinstance(item, dict) and item.get("featureId")
    }
    aligned = []
    for endpoint in prepared:
        original_path = _output_api_path(endpoint.get("path"))
        original_method = canonical_api_method(endpoint.get("method"), original_path)
        spec = by_id.get(str(endpoint.get("featureId") or ""), {})
        explicit_route = any(
            canonical_api_method(contract.get("method"), contract.get("path")) == original_method
            and _route_shape(contract.get("path")) == _route_shape(original_path)
            for contract in spec.get("apiContract") or [] if isinstance(contract, dict)
        )
        declared = (spec.get("dbContract") or {}).get("tables", [])
        declared_names = [
            str(table.get("name") or table.get("table") or "") if isinstance(table, dict) else str(table)
            for table in declared
        ]
        candidates = [name for name in declared_names if name in table_names]
        draft = dict(endpoint)
        if spec.get("name"):
            draft["featureName"] = spec["name"]
        # Route vocabulary and physical table names may legitimately differ.
        # Use an unambiguous declared table for fields, then restore the route.
        identity_route = original_path.startswith("/api/v1/auth/") or original_path == "/api/v1/users/me"
        feature_table = endpoint_feature_table(
            original_path, feature_mappings.get(str(endpoint.get("featureId") or "")), table_names,
        )
        if feature_table in table_names and not identity_route:
            declared_table = feature_table
        elif explicit_route and len(candidates) == 1:
            declared_table = candidates[0]
        else:
            declared_table = None
        if explicit_route and declared_table and not identity_route:
            resource, _, suffix = original_path.removeprefix("/api/v1/").partition("/")
            draft["path"] = f"/api/v1/{declared_table.replace('_', '-')}" + (f"/{suffix}" if suffix else "")
        contracts = _align_endpoints_to_db(
            [draft], db_schema, prd_document, requirement_text,
            declared_table_name=declared_table,
        )
        for contract in contracts:
            if explicit_route:
                contract["path"] = original_path
                contract["method"] = original_method
            for key in ("featureId", "action"):
                if key in endpoint:
                    contract[key] = endpoint[key]
            aligned.append(contract)
    return _normalize_endpoints(aligned)


class _ApiGraphState(TypedDict):
    plan: list
    authentication: str
    endpoints: Annotated[list, operator.add]
    api_spec: str
    prd_issues: str
    ctx: dict
    generation_blocker: str


class ApiAgent:

    def __init__(
        self,
        client: AsyncOpenAI,
        model: str = "gpt-5.4-mini",
        runtime: LlmRuntime | None = None,
    ):
        self._client = client
        self._model = model
        self._runtime = runtime
        self._graph = self._build_graph()

    def _build_graph(self):
        graph = StateGraph(_ApiGraphState)
        graph.add_node("manager_plan", self._manager_plan_node)
        graph.add_node("worker", self._worker_node)
        graph.add_node("manager_review", self._manager_review_node)
        graph.add_edge(START, "manager_plan")
        graph.add_conditional_edges("manager_plan", self._dispatch, ["worker"])
        graph.add_edge("worker", "manager_review")
        graph.add_edge("manager_review", END)
        return graph.compile()

    async def execute(self, state: PipelineState, dump=None) -> PipelineState:
        logger.info("API 에이전트 시작 (manager plan + 동적 fan-out 서브그래프)")

        context = state.context_prompt or ""
        instruction = state.api_instruction or ""
        # Feature Spec이 확정한 ID를 downstream에서 다시 순번화하지 않는다.
        # normalize_feature_registry를 여기서 재호출하면 supporting 항목이
        # feature_007처럼 새 ID를 받아 Phase3의 canonical Registry와 분리된다.
        source_registry = [
            dict(item) for item in (state.feature_registry or [])
            if isinstance(item, dict) and str(item.get("featureId") or item.get("id") or "").strip()
        ]
        if not source_registry:
            source_registry = normalize_feature_registry(
                state.feature_registry, state.feature_list, preserve_extra=True,
            )
        registry_features = [str(item.get("name")) for item in source_registry if item.get("name")]
        # Feature Spec의 모든 기능은 지식 그래프/API 계약의 노드가 된다.
        # backend_features로 범위를 줄이면 하위 supporting 기능의 Registry 계약이
        # API 산출물에서 사라져 Phase3에서 고립 노드가 된다.
        scoped_features = registry_features or list(state.feature_list or [])
        feature_str = "- " + "\n- ".join(scoped_features) if scoped_features else "(API 대상 기능 없음)"
        registry = source_registry

        graph_input = {
            "plan": [],
            "authentication": "",
            "endpoints": [],
            "api_spec": "",
            "prd_issues": "",
            "ctx": {
                "context": context,
                "instruction": instruction,
                "feature_str": feature_str,
                "feature_list": scoped_features,
                "feature_registry": registry_text(registry),
                "prd_document": state.prd_document or "{}",
                "db_schema": state.db_schema or "{}",
                "max_endpoints": _endpoint_soft_cap(len(scoped_features), include_auth=True),
                "dump": dump,
            },
            "generation_blocker": "",
        }

        try:
            result = await self._graph.ainvoke(graph_input)
            blocker = str(
                result.get("generation_blocker") or graph_input["ctx"].get("generation_blocker") or ""
            ).strip()
            if not blocker and not try_parse_json(result.get("api_spec")):
                blocker = "API_GENERATION_BLOCKER: upstream API 응답 후 유효한 산출물이 생성되지 않았습니다"
            if blocker:
                api_spec = json.dumps(
                    {
                        "endpoints": _registry_fallback_plan(registry),
                        "authentication": _AUTHENTICATION_DESC,
                    },
                    ensure_ascii=False,
                )
            else:
                api_spec = result["api_spec"]
            prd_issues = result.get("prd_issues", "")
        except Exception as e:
            logger.error("API 에이전트 실패: %s", e)
            blocker = f"API_GENERATION_BLOCKER: upstream API 호출 실패 ({type(e).__name__}: {e})"
            return state.copy(
                api_spec="{}",
                generation_blockers=[*state.generation_blockers, blocker],
                qa_api_blockers=[*state.qa_api_blockers, blocker],
                qa_approved=False,
                status_message=f"API 에이전트 실패: {e}",
            )

        # Graph worker 결과를 최종 산출물로 채택하기 직전에 계약을 다시 투영한다.
        # 병렬 worker 병합이나 manager patch가 featureId를 잃어도 Registry 계약이
        # 단일 기준이 되도록 보정하고, 계약 밖 orphan endpoint는 남기지 않는다.
        parsed_spec = try_parse_json(api_spec)
        self_check_issues: list[str] = []
        if isinstance(parsed_spec, dict) and isinstance(parsed_spec.get("endpoints"), list):
            endpoints = finalize_api_contracts(
                parsed_spec["endpoints"], state.db_schema, state.prd_document,
                scoped_features, registry, state.user_query or state.context_prompt,
            )
            endpoints = _ensure_contract_endpoints(endpoints, registry)
            endpoints = _annotate_feature_ids(endpoints, registry)
            endpoints, self_check_issues = api_self_check(endpoints, registry)
            endpoints = [ep for ep in endpoints if isinstance(ep, dict) and str(ep.get("featureId") or "").strip()]
            parsed_spec["endpoints"] = _normalize_endpoints(endpoints)
            parsed_spec["featureMappings"] = _api_feature_mappings(endpoints, state.db_schema, scoped_features)
            api_spec = json.dumps(parsed_spec, ensure_ascii=False)

        api_blockers = ([blocker] if blocker else []) + self_check_issues
        if self_check_issues:
            logger.warning("API self-check blocker: %s", self_check_issues)

        logger.info("API 에이전트 완료")
        return state.copy(
            api_spec=api_spec,
            prd_feedback_from_api=prd_issues,
            # upstream 호출/파싱 실패만 generation blocker로 남긴다. 계약 경로,
            # featureId, placeholder 같은 산출물 결함은 Phase3 targeted repair가
            # 처리해야 하므로 QA blocker에만 기록한다.
            generation_blockers=(
                [*state.generation_blockers, blocker]
                if blocker else state.generation_blockers
            ),
            qa_api_blockers=(
                [*state.qa_api_blockers, *api_blockers]
                if api_blockers else state.qa_api_blockers
            ),
            qa_approved=False if api_blockers else state.qa_approved,
            status_message="API 에이전트 완료 — API 스펙 생성",
        )

    async def repair(self, state: PipelineState, feedback: str, dump=None) -> PipelineState:
        """기존 API 명세를 보존하고 검증 피드백에 해당하는 endpoint만 patch한다."""
        current = try_parse_json(state.api_spec)
        if not isinstance(current, dict) or not isinstance(current.get("endpoints"), list):
            blocker = "TARGETED_REPAIR_BLOCKER: API 기존 산출물이 없어 전체 재생성을 차단했습니다"
            logger.error(blocker)
            return state.copy(
                prd_feedback_from_api=blocker,
                qa_api_blockers=[*state.qa_api_blockers, blocker],
                qa_approved=False,
                status_message="API targeted repair 차단 — 기존 산출물 없음",
            )

        prompt = TARGETED_REPAIR_PROMPT.format(
            feedback=str(feedback or "")[:8000],
            api_spec=json.dumps(current, ensure_ascii=False),
        )
        try:
            raw = await self._call(prompt, max_tokens=7000, system=MANAGER_SYSTEM)
            if dump:
                dump.log_raw("API_TARGETED_REPAIR", 1, raw)
            result = try_parse_json(raw)
            patches = result.get("patches") if isinstance(result, dict) else None
            if not isinstance(patches, list) or not patches:
                logger.warning("API targeted repair — 유효한 patches 없음, 원본 유지")
                return state

            by_key = {
                (str(ep.get("method", "GET")).upper(), str(ep.get("path", ""))): ep
                for ep in current["endpoints"] if isinstance(ep, dict)
            }
            applied = 0
            for patch in patches:
                if not isinstance(patch, dict):
                    continue
                patch = _normalize_endpoint_keys(dict(patch))
                method = str(patch.get("method", "GET")).upper()
                path = str(patch.get("path", ""))
                if not path:
                    continue
                patch["method"] = method
                key = (method, path)
                if key in by_key and not str(patch.get("transactionRules") or "").strip():
                    patch.pop("transactionRules", None)
                by_key[key] = {**by_key[key], **patch} if key in by_key else patch
                applied += 1

            if not applied:
                logger.warning("API targeted repair — 적용 가능한 endpoint 없음, 원본 유지")
                return state
            registry = [
                dict(item) for item in (state.feature_registry or [])
                if isinstance(item, dict) and str(item.get("featureId") or item.get("id") or "").strip()
            ]
            if not registry:
                registry = normalize_feature_registry(
                    state.feature_registry, state.feature_list, preserve_extra=True,
                )
            endpoints = _normalize_endpoints(list(by_key.values()))
            scoped_features = backend_features([item["name"] for item in registry] or state.feature_list)
            endpoints = finalize_api_contracts(
                endpoints, state.db_schema, state.prd_document, scoped_features, registry,
                state.user_query or state.context_prompt,
            )
            endpoints = _annotate_feature_ids(endpoints, registry)
            endpoints, self_check_issues = api_self_check(endpoints, registry)
            repaired = {
                **current, "endpoints": _normalize_endpoints(endpoints),
                "featureMappings": _api_feature_mappings(endpoints, state.db_schema, scoped_features),
            }
            logger.info("API targeted repair — %d개 endpoint만 패치", applied)
            return state.copy(
                api_spec=json.dumps(repaired, ensure_ascii=False),
                prd_feedback_from_api="",
                # API domain repair가 성공하면 이전 API blocker는 제거한다.
                # 다른 domain의 blocker는 orchestration/QA state가 별도로 보존한다.
                qa_api_blockers=self_check_issues,
                qa_api_issues=self_check_issues,
                qa_approved=False if self_check_issues else state.qa_approved,
                status_message="API 에이전트 완료 — 지적 endpoint만 수정",
            )
        except Exception as e:
            logger.warning("API targeted repair 실패 — 원본 유지: %s", e)
            return state

    async def _plan_batch(
        self, ctx: dict, batch: list[str], batch_idx: int, with_auth: bool,
        existing_paths: list[str] | None = None,
    ) -> list[dict]:
        """기능 몇 개만 담당하는 plan 배치 하나를 생성한다.

        기능 21개에 엔드포인트 42개를 한 번에 요구하면 EXAONE이 12개쯤에서 멈춰
        (실측) 레시피·알림·장보기 같은 기능군이 통째로 빠진 채 통과됐다. 담당 범위를
        좁히면 요구 개수가 배치당 8~10개로 내려가 실제로 채워진다."""
        min_endpoints = len(batch) * _ENDPOINTS_PER_FEATURE + (_AUTH_ENDPOINT_COUNT if with_auth else 0)
        max_endpoints = _endpoint_soft_cap(len(batch), include_auth=with_auth)
        prompt = PLAN_PROMPT.format(
            instruction=ctx["instruction"],
            context=ctx["context"],
            feature_str="- " + "\n- ".join(batch),
            feature_count=len(batch),
            min_endpoints=min_endpoints,
            max_endpoints=max_endpoints,
            auth_rule=_AUTH_RULE if with_auth else "",
            existing_note=_existing_paths_note(existing_paths),
            feature_registry=ctx.get("feature_registry", "[]"),
            prd_document=ctx.get("prd_document", "{}"),
        )
        label = f"API_MANAGER_PLAN_{batch_idx}"

        best: list[dict] = []
        for attempt in range(1):
            raw = await self._call(
                prompt, max_tokens=8192, system=MANAGER_SYSTEM,
                temperature=_PLAN_TEMPERATURE,
            )
            if ctx.get("dump"):
                ctx["dump"].log_raw(label, attempt + 1, raw)
            candidate = try_parse_json(raw)
            if not (candidate and isinstance(candidate.get("plan"), list)) or has_suspicious_script(candidate):
                logger.warning("%s 파싱 실패/오염 (attempt %d) — 재생성", label, attempt + 1)
                continue
            plan_list = [ep for ep in candidate["plan"] if isinstance(ep, dict)]
            plan_list = _prune_plan(plan_list, batch, max_endpoints)
            missing = uncovered_features(batch, [ep.get("description", "") for ep in plan_list])
            if len(plan_list) > len(best):
                best = plan_list
            if len(plan_list) >= min_endpoints and not _invalid_paths(plan_list) and not missing:
                return plan_list
            logger.warning(
                "%s 개수 부족/경로 오류/커버리지 미달 (attempt %d) — %d/%d개, 미반영 기능: %s — 재생성",
                label, attempt + 1, len(plan_list), min_endpoints, missing,
            )
        logger.warning("%s 재생성 한도 도달 — 최선의 결과(%d개)로 진행", label, len(best))
        return best

    @staticmethod
    def _merge_plans(batches: list[list[dict]]) -> list[dict]:
        """배치별 plan을 'METHOD path' 기준으로 중복 제거하며 합친다."""
        merged, seen = [], set()
        for plan in batches:
            for ep in plan:
                if not isinstance(ep, dict):
                    continue
                key = (str(ep.get("method", "")).upper(), str(ep.get("path", "")).rstrip("/"))
                if key in seen:
                    continue
                seen.add(key)
                merged.append(ep)
        return merged

    async def _manager_plan_node(self, state: dict) -> dict:
        ctx = state["ctx"]
        feature_list = ctx["feature_list"] or []
        min_endpoints = max(4, len(feature_list) * _ENDPOINTS_PER_FEATURE)
        max_endpoints = _endpoint_soft_cap(len(feature_list), include_auth=True)

        # 기능을 배치로 쪼개 병렬 설계 — 배치당 요구 개수가 작아야 EXAONE이 실제로 채운다
        batches = [
            feature_list[i:i + _PLAN_BATCH_SIZE]
            for i in range(0, len(feature_list), _PLAN_BATCH_SIZE)
        ] or [[]]
        logger.info("API manager_plan — 기능 %d개를 %d개 배치로 분할", len(feature_list), len(batches))

        results = await asyncio.gather(*[
            self._plan_batch(ctx, batch, i + 1, with_auth=(i == 0))
            for i, batch in enumerate(batches)
        ])
        plan = self._merge_plans(results)
        plan = _prune_plan(plan, feature_list, max_endpoints)

        # 배치들이 서로 모르는 채 같은 리소스를 설계해 병합 시 중복 제거로 개수가 깎인다.
        # 흔적이 없는 기능(uncovered)뿐 아니라 엔드포인트가 부족한 기능(undercovered)까지
        # 모아, 이미 설계된 경로를 알려주고 '없는 것만' 추가로 받아낸다. 최대 2라운드.
        # 누락 보충은 Phase3 targeted repair로 이관한다.
        for round_ in range(1):
            descriptions = [ep.get("description", "") for ep in plan]
            missing = uncovered_features(feature_list, descriptions)
            thin = undercovered_features(feature_list, descriptions, _ENDPOINTS_PER_FEATURE)
            need = missing + [f for f in thin if f not in missing]
            if not need or len(plan) >= min_endpoints or len(plan) >= max_endpoints:
                break
            logger.warning(
                "API manager_plan 보충 %d라운드 — 현재 %d/%d개, 미반영 %d개·부족 %d개: %s",
                round_ + 1, len(plan), min_endpoints, len(missing), len(thin), need,
            )
            topup = await self._plan_batch(
                ctx, need, len(batches) + round_ + 1, with_auth=False,
                existing_paths=[f"{ep.get('method')} {ep.get('path')}" for ep in plan],
            )
            merged = self._merge_plans([plan, topup])
            merged = _prune_plan(merged, feature_list, max_endpoints)
            if len(merged) == len(plan):
                logger.warning("API manager_plan 보충 %d라운드 — 새 엔드포인트 없음, 중단", round_ + 1)
                break
            plan = merged

        generation_blocker = ""
        registry_items = _registry_items(ctx.get("feature_registry", "[]"))
        if not plan:
            generation_blocker = (
                "API_GENERATION_BLOCKER: upstream API 응답 실패로 Registry 계약 기반 fallback을 사용했습니다"
            )
            logger.error("%s", generation_blocker)
            ctx["generation_blocker"] = generation_blocker
            plan = _registry_fallback_plan(registry_items)
        elif _invalid_paths(plan):
            logger.warning("API manager_plan — 경로 형식 오류가 남아있어 강제 살균 적용")
            plan = _sanitize_plan_paths(plan)
            plan = _prune_plan(plan, feature_list, max_endpoints)

        if len(plan) < min_endpoints:
            logger.warning("API manager_plan — 엔드포인트 %d/%d개로 목표 미달", len(plan), min_endpoints)

        plan = _ensure_contract_endpoints(plan, registry_items)
        plan = _annotate_feature_ids(plan, registry_items)
        logger.info("API manager_plan 완료 — 엔드포인트 %d개 계획 (목표 %d개, soft cap %d개)", len(plan), min_endpoints, max_endpoints)
        return {
            "plan": plan,
            "authentication": _AUTHENTICATION_DESC,
            "generation_blocker": generation_blocker,
        }

    def _fallback_plan(self, feature_list: list[str]) -> dict:
        plan = [
            {"method": "POST", "path": "/api/v1/auth/signup", "description": "회원가입", "authRequired": False},
            {"method": "POST", "path": "/api/v1/auth/login", "description": "로그인", "authRequired": False},
        ]
        for f in feature_list:
            # 기능명(한글 포함)을 path에 그대로 쓰지 않도록 나머지 항목과 함께 뒤에서 일괄 슬러그화한다
            plan.append({"method": "GET", "path": f"/api/v1/{f}", "description": f"{f} 목록 조회", "authRequired": True})
            plan.append({"method": "POST", "path": f"/api/v1/{f}", "description": f"{f} 생성", "authRequired": True})
        plan = _sanitize_plan_paths(plan)
        return {"plan": plan, "authentication": "JWT Bearer 토큰"}

    def _dispatch(self, state: dict) -> list[Send]:
        if state.get("generation_blocker"):
            return []
        ctx = state["ctx"]
        sends = []
        for skeleton in state["plan"]:
            sends.append(Send("worker", {"skeleton": skeleton, "context": ctx["context"], "dump": ctx.get("dump")}))
        return sends

    async def _worker_node(self, state: dict) -> dict:
        skeleton = state["skeleton"]
        context = state["context"]
        dump = state.get("dump")

        method = skeleton.get("method", "GET")
        path = skeleton.get("path", "/api/v1/unknown")
        description = skeleton.get("description", "")
        auth_required = bool(skeleton.get("authRequired", True))

        prompt = ENDPOINT_SPEC_PROMPT.format(
            feature_id=skeleton.get("featureId", "미지정"),
            method=method,
            path=path,
            description=description,
            auth_required=str(auth_required).lower(),
            context=context,
        )

        label = f"API_ENDPOINT_{method}_{path}"
        data = None
        for attempt in range(1):
            raw = await self._call(prompt, max_tokens=1200, system=WORKER_SYSTEM)
            if dump:
                dump.log_raw(label, attempt + 1, raw)
            data = try_parse_json(raw)
            if data and isinstance(data, dict):
                data = _normalize_endpoint_keys(data)
                if has_suspicious_script(data):
                    logger.warning("%s 스크립트 오염 감지 (attempt %d) — 재생성", label, attempt + 1)
                    data = None
                    continue
                break
            logger.warning("%s 파싱 실패 (attempt %d) — 재생성", label, attempt + 1)
            data = None

        if not data:
            logger.error("%s 최종 파싱 실패 — 스켈레톤 기본값으로 최소 스펙 구성", label)
            data = {
                "method": method,
                "path": path,
                "description": description,
                "authRequired": auth_required,
                "featureId": skeleton.get("featureId", ""),
                "requestBody": "없음",
                "successResponse": "success: boolean",
                "errorCodes": "500 — 서버 오류",
            }
        else:
            data.setdefault("method", method)
            data.setdefault("path", path)
            data.setdefault("description", description)
            data.setdefault("authRequired", auth_required)
            if skeleton.get("featureId"):
                data["featureId"] = skeleton["featureId"]
            if not data.get("requestBody"):
                data["requestBody"] = "없음"
            if not data.get("successResponse"):
                data["successResponse"] = "success: boolean"
            if not data.get("errorCodes"):
                data["errorCodes"] = "500 — 서버 오류"

        return {"endpoints": [data]}

    async def _manager_review_node(self, state: dict) -> dict:
        ctx = state["ctx"]
        endpoints = state["endpoints"]
        authentication = state["authentication"]
        # Review는 전체 endpoint를 다시 LLM에 보내는 생성 단계가 아니다.
        # 계약/경로/필수 필드 보강은 결정론적으로 처리하고 남은 결함은 Phase3가
        # featureId 단위 targeted repair 대상으로 전달한다.
        registry_items = _registry_items(ctx.get("feature_registry", "[]"))
        endpoints = _annotate_feature_ids(
            _ensure_contract_endpoints(
                _normalize_endpoints(_sanitize_plan_paths(endpoints)),
                registry_items,
            ),
            registry_items,
        )
        endpoints = finalize_api_contracts(
            endpoints, ctx.get("db_schema", "{}"), ctx.get("prd_document", ""),
            ctx.get("feature_list", []), registry_items, ctx.get("context", ""),
        )
        return {
            "api_spec": json.dumps(
                {"endpoints": endpoints, "authentication": authentication},
                ensure_ascii=False,
            ),
            "prd_issues": "",
        }

    async def _call(self, user_prompt: str, max_tokens: int, system: str, temperature: float = _TEMPERATURE) -> str:
        for attempt in range(1):
            try:
                async def request():
                    return await self._client.chat.completions.create(
                        model=self._model,
                        temperature=temperature,
                        top_p=_TOP_P,
                        presence_penalty=_PRESENCE_PENALTY,
                        max_completion_tokens=max_tokens,
                        frequency_penalty=0.5,
                        messages=[
                            {"role": "system", "content": system},
                            {"role": "user", "content": user_prompt},
                        ],
                    )

                response = await self._runtime.call(request) if self._runtime else await request()
                return response.choices[0].message.content or ""
            except (InternalServerError, APITimeoutError, APIConnectionError, TimeoutError) as e:
                logger.warning("API 호출 일시 오류 (attempt %d): %s — 재시도", attempt + 1, e)
                if attempt < 2:
                    await asyncio.sleep(5 * (attempt + 1))
                else:
                    logger.error("API 호출 최종 실패")
                    return ""
        return ""


_VALID_PATH_RE = re.compile(r'^/[A-Za-z0-9/_\-{}]*$')
_PATH_STRIP_RE = re.compile(r'[^A-Za-z0-9\-{}/]+')
_PATH_DASH_COLLAPSE_RE = re.compile(r'-{2,}')


def _endpoint_soft_cap(feature_count: int, include_auth: bool = True) -> int:
    """기능 수에 따라 API plan 상한을 유연하게 잡는다.

    고정 개수 제한이 아니라 MVP 산출물에서 과도한 CRUD 확장을 막기 위한 상한이다.
    """
    auth = _AUTH_ENDPOINT_COUNT if include_auth else 0
    if feature_count <= 0:
        return max(4, auth)
    return max(8, min(_MAX_ENDPOINT_SOFT_CAP, int(feature_count * 1.6) + auth))


def _endpoint_priority(ep: dict, feature_list: list[str]) -> tuple[int, int, int]:
    path = str(ep.get("path") or "").lower()
    method = str(ep.get("method") or "GET").upper()
    description = str(ep.get("description") or "")
    auth_rank = 0 if any(token in path for token in ("/auth/", "/login", "/logout", "/token", "/signup", "/register")) else 1
    action_rank = {"POST": 0, "PATCH": 1, "PUT": 2, "GET": 3, "DELETE": 4}.get(method, 5)
    derived_penalty = sum(
        token in path
        for token in ("dashboard", "history", "overview", "stats", "metrics", "archive", "search")
    )
    covered = 0 if any(feature in description for feature in feature_list if isinstance(feature, str)) else 1
    return auth_rank, covered + derived_penalty, action_rank


def _prune_plan(plan: list[dict], feature_list: list[str], max_endpoints: int) -> list[dict]:
    """soft cap 초과 시 인증/핵심 액션 API를 우선 보존하고 파생 조회 API를 줄인다."""
    unique = _sanitize_plan_paths(plan)
    if len(unique) <= max_endpoints:
        return unique
    indexed = list(enumerate(unique))
    # soft cap 적용 전에 기능별 대표 endpoint를 먼저 보존한다. 그렇지 않으면
    # 인증/저우선순위 정렬 때문에 특정 기능의 유일한 endpoint가 잘릴 수 있다.
    required: list[tuple[int, dict]] = []
    remaining: list[tuple[int, dict]] = []
    for item in indexed:
        ep_text = json.dumps(item[1], ensure_ascii=False)
        if any(not strictly_uncovered_features([feature], [ep_text]) for feature in feature_list):
            required.append(item)
        else:
            remaining.append(item)
    if len(required) > max_endpoints:
        required = sorted(required, key=lambda item: (*_endpoint_priority(item[1], feature_list), item[0]))[:max_endpoints]
        selected = required
    else:
        remaining.sort(key=lambda item: (*_endpoint_priority(item[1], feature_list), item[0]))
        selected = required + remaining[:max_endpoints - len(required)]
    selected = sorted(selected, key=lambda item: item[0])
    pruned = [ep for _, ep in selected]
    logger.warning("API manager_plan — soft cap 적용: %d개 → %d개", len(unique), len(pruned))
    return pruned


def _invalid_paths(plan: list) -> list[str]:
    """한글/공백/괄호 등 REST 경로에 쓸 수 없는 문자가 섞인 path 목록 (EXAONE이 기능명을
    그대로 슬러그로 써버리는 경우 방어)."""
    bad = []
    for ep in plan:
        path = ep.get("path") if isinstance(ep, dict) else None
        if (
            not isinstance(path, str) or not path or not _VALID_PATH_RE.match(path)
            or not path.startswith("/api/v1/")
        ):
            bad.append(path)
    return bad


def _canonical_api_path(path: str) -> str:
    """Compare registry and public paths using the same canonical resource key."""
    return _output_api_path(path).removeprefix("/api/v1") or "/"


def _output_api_path(path: str) -> str:
    return normalize_api_path(path)


def _route_shape(path: str) -> str:
    """Compare REST routes independent of LLM-chosen parameter names."""
    return re.sub(r"\{[^}]+\}", "{}", _canonical_api_path(path))


def _annotate_feature_ids(plan: list, registry: list[dict] | None) -> list:
    """모든 endpoint에 Registry의 featureId를 결정론적으로 부착한다.

    계약 경로가 있는 endpoint는 exact match를 사용하고, LLM이 새 supporting endpoint를
    추가한 경우에는 설명/기능명/action을 보조 기준으로 사용한다. Registry 밖의 임의
    featureId는 남기지 않아 QA가 추적 불가능한 endpoint를 즉시 발견할 수 있게 한다.
    """
    contracts: dict[tuple[str, str], str] = {}
    valid_ids: set[str] = set()
    names: list[tuple[str, str, set[str]]] = []
    route_prefixes: list[tuple[str, str]] = []
    for item in registry or []:
        feature_id = str(item.get("featureId") or item.get("id") or "").strip()
        if feature_id:
            valid_ids.add(feature_id)
            terms = {str(item.get("name") or "").casefold()}
            terms.update(str(action).casefold() for action in item.get("actions") or [])
            names.append((feature_id, str(item.get("name") or ""), {term for term in terms if term}))
        for contract in item.get("apiContract") or item.get("api") or []:
            if not isinstance(contract, dict) or not feature_id:
                continue
            method = canonical_api_method(contract.get("method"), contract.get("path"))
            path = _canonical_api_path(contract.get("path"))
            if path:
                contracts[(method, path)] = feature_id
                route_prefixes.append((path.rsplit("/", 1)[0] or "/", feature_id))
    for ep in plan:
        if not isinstance(ep, dict):
            continue
        ep["path"] = _output_api_path(ep.get("path"))
        key = (canonical_api_method(ep.get("method"), ep.get("path")), _canonical_api_path(ep.get("path")))
        if key in contracts:
            ep["featureId"] = contracts[key]
            continue
        current = str(ep.get("featureId") or "").strip()
        if current not in valid_ids:
            current = ""
        haystack = " ".join(str(ep.get(key) or "") for key in ("description", "summary", "operationId", "action")).casefold()
        candidates = [fid for fid, _name, terms in names if any(term in haystack for term in terms)]
        if not current:
            route = _canonical_api_path(ep.get("path"))
            candidates.extend(fid for prefix, fid in route_prefixes if route.startswith(prefix + "/"))
        if not current and len(candidates) == 1:
            current = candidates[0]
        if current:
            ep["featureId"] = current
        else:
            # Never assign an unrelated feature just to silence QA. An orphan
            # endpoint must remain visible as a mapping blocker for repair.
            ep.pop("featureId", None)
    return plan


def api_self_check(
    endpoints: list[dict] | None,
    registry: list[dict] | None,
) -> tuple[list[dict], list[str]]:
    """API 산출물과 Feature Registry 계약을 결정론적으로 검증한다.

    LLM이 만든 endpoint의 설명이 그럴듯한지만 확인하지 않고, 동일한
    ``featureId``의 method/path 계약과 정확히 연결되는지 검사한다. 반환된
    endpoint는 검사 전에 기존 정규화 규칙을 적용하므로 호출자는 이 결과를
    최종 API 문서로 그대로 사용할 수 있다.
    """
    registry_items = [item for item in registry or [] if isinstance(item, dict)]
    raw_keys = [
        (
            str(item.get("method") or "GET").upper(),
            _canonical_api_path(item.get("path")),
        )
        for item in endpoints or []
        if isinstance(item, dict)
    ]
    normalized = _ensure_contract_endpoints(
        _normalize_endpoints(list(endpoints or [])), registry_items,
    )
    normalized = _annotate_feature_ids(_normalize_endpoints(normalized), registry_items)
    if not registry_items:
        return normalized, []
    issues: list[str] = []
    duplicate_raw = sorted({key for key in raw_keys if raw_keys.count(key) > 1})
    if duplicate_raw:
        logger.warning("API self-check — 중복 endpoint %d개를 정규화로 제거", len(duplicate_raw))
    registry_by_id = {
        str(item.get("featureId") or item.get("id") or "").strip(): item
        for item in registry_items
        if str(item.get("featureId") or item.get("id") or "").strip()
    }
    expected: dict[str, set[tuple[str, str]]] = {}
    for feature_id, item in registry_by_id.items():
        expected[feature_id] = {
            (str(contract.get("method") or "GET").upper(), _canonical_api_path(contract.get("path")))
            for contract in (item.get("apiContract") or item.get("api") or [])
            if isinstance(contract, dict) and str(contract.get("path") or "").strip()
        }

    # Registry에 없는 endpoint와 계약 경로가 아닌 endpoint는 LLM이 추가한
    # orphan으로 간주하고 제거한다. Registry 계약 endpoint는 위의
    # _ensure_contract_endpoints가 이미 보장하므로, 제거 후에도 누락은
    # 아래의 계약별 검사에서 다시 검출된다.
    filtered: list[dict] = []
    seen: set[tuple[str, str]] = set()
    for endpoint in normalized:
        if not isinstance(endpoint, dict):
            continue
        method = str(endpoint.get("method") or "GET").upper()
        path = str(endpoint.get("path") or "").strip()
        key = (method, _canonical_api_path(path))
        feature_id = str(endpoint.get("featureId") or "").strip()
        if key in seen:
            logger.warning("API self-check — 중복 endpoint 제거: %s %s", method, path)
            continue
        seen.add(key)
        if not feature_id:
            issues.append(f"API_FEATURE_ID_REQUIRED: {method} {path}")
            continue
        if feature_id not in registry_by_id:
            issues.append(f"API_FEATURE_ID_UNKNOWN: {feature_id} ({method} {path})")
            continue
        if _endpoint_has_placeholder_contract(endpoint):
            issues.append(f"API_CONTRACT_DETAIL_PLACEHOLDER: {feature_id} {method} {path}")
        if key not in expected.get(feature_id, set()):
            expected_paths = sorted(
                f"{candidate_method} {candidate_path}"
                for candidate_method, candidate_path in expected.get(feature_id, set())
            )
            issues.append(
                "API_CONTRACT_PATH_MISMATCH: "
                f"{feature_id} {method} {path} -> expected: "
                f"{', '.join(expected_paths) or '(none)'}"
            )
            # canonical 계약 endpoint는 아래 projection에서 유지한다. LLM이
            # 만든 다른 path를 같이 보존하면 실제 계약과 중복 route가 생긴다.
            continue
        filtered.append(endpoint)

    # 최종 문서는 Registry 계약의 projection으로 확정한다. 같은 featureId의
    # 잘못된 경로·순번 ID endpoint는 버리고, 계약에만 있는 항목은 최소 계약
    # 객체로 생성해 API와 DB/Feature Spec 그래프가 반드시 같은 노드를 가리키게 한다.
    by_contract = {
        (
            str(endpoint.get("featureId") or "").strip(),
            str(endpoint.get("method") or "GET").upper(),
            _canonical_api_path(endpoint.get("path")),
        ): endpoint
        for endpoint in filtered
    }
    projected: list[dict] = []
    for feature_id, contracts in expected.items():
        feature = registry_by_id[feature_id]
        name = str(feature.get("name") or feature_id)
        for method, path in sorted(contracts):
            key = (feature_id, method, path)
            endpoint = by_contract.get(key)
            if endpoint is None:
                endpoint = {
                    "featureId": feature_id,
                    "action": "manage",
                    "method": method,
                    "path": _output_api_path(path),
                    "description": f"{name}: Registry 계약 endpoint",
                    "authRequired": not path.startswith("/auth/"),
                    "requestBody": "없음" if method in _BODYLESS_METHODS else "계약 기반 요청 본문",
                    "successResponse": "success: boolean",
                    "errorCodes": "400 — 요청 오류; 500 — 서버 오류",
                }
            projected.append(endpoint)

    return _normalize_endpoints(projected), list(dict.fromkeys(issues))


def _registry_items(raw: str | list[dict] | None) -> list[dict]:
    if isinstance(raw, list):
        return [item for item in raw if isinstance(item, dict)]
    try:
        parsed = json.loads(raw or "[]")
    except (TypeError, ValueError):
        parsed = try_parse_json(raw or "[]")
    return [item for item in parsed if isinstance(item, dict)] if isinstance(parsed, list) else []


def _registry_fallback_plan(registry: list[dict] | None) -> list[dict]:
    """Upstream 장애 시에도 Registry 계약만 투영하고 임의 endpoint는 만들지 않는다."""
    plan: list[dict] = []
    for item in registry or []:
        if not isinstance(item, dict):
            continue
        feature_id = str(item.get("featureId") or item.get("id") or "").strip()
        name = str(item.get("name") or feature_id).strip()
        for contract in item.get("apiContract") or item.get("api") or []:
            if not isinstance(contract, dict) or not feature_id:
                continue
            method = str(contract.get("method") or "GET").upper()
            path = _output_api_path(contract.get("path"))
            if not path or path == "/api/v1":
                continue
            action = str(contract.get("action") or "manage")
            plan.append({
                "featureId": feature_id,
                "action": action,
                "method": method,
                "path": path,
                "description": f"{name}: {action} — Registry 계약 fallback",
                "authRequired": not path.startswith("/api/v1/auth/"),
                "requestBody": "계약 기반 요청 본문",
                "successResponse": "계약 기반 성공 응답",
                "errorCodes": "400 — 요청 오류; 500 — 서버 오류",
            })
    return _normalize_endpoints(plan)


def _ensure_contract_endpoints(plan: list, registry: list[dict] | None) -> list:
    """PM이 명시한 endpoint는 LLM plan 누락 여부와 무관하게 최종 plan에 보존한다."""
    existing = {
        (canonical_api_method(ep.get("method"), ep.get("path")), _canonical_api_path(ep.get("path"))): ep
        for ep in plan if isinstance(ep, dict)
    }
    existing_shapes = {
        (method, _route_shape(path)): ep for (method, path), ep in existing.items()
    }
    for item in registry or []:
        fid = str(item.get("featureId") or item.get("id") or "")
        name = str(item.get("name") or fid)
        for contract in item.get("apiContract") or []:
            if not isinstance(contract, dict):
                continue
            path = str(contract.get("path", "")).strip()
            if not path:
                continue
            path = _output_api_path(path)
            method = canonical_api_method(contract.get("method"), path)
            key = (method, _canonical_api_path(path))
            shape_key = (method, _route_shape(path))
            if key in existing:
                existing[key]["featureId"] = fid
                existing[key]["action"] = contract.get("action") or existing[key].get("action", "manage")
                continue
            if shape_key in existing_shapes:
                endpoint = existing_shapes[shape_key]
                endpoint["path"] = path
                endpoint["featureId"] = fid
                endpoint["action"] = contract.get("action") or endpoint.get("action", "manage")
                existing[key] = endpoint
                continue
            plan.append({
                "featureId": fid, "method": method, "path": path,
                "description": f"{name}: {contract.get('action', '계약된 행동')} — PM 계약 endpoint",
                "authRequired": True,
            })
            existing[key] = plan[-1]
            existing_shapes[shape_key] = plan[-1]
    return plan


def _sanitize_plan_paths(plan: list) -> list:
    """유효하지 않은 path를 ASCII 슬러그로 강제 변환 (재생성 3회 실패 시 최후 수단)."""
    seen: set[tuple[str, str]] = set()
    for i, ep in enumerate(plan):
        if not isinstance(ep, dict):
            continue
        path = _output_api_path(ep.get("path"))
        if not path.lower().startswith("/api/v1/") or not _VALID_PATH_RE.match(path):
            rest = path[len("/api/v1/"):] if path.lower().startswith("/api/v1/") else path.lstrip("/")
            slug = _PATH_STRIP_RE.sub("-", rest)
            slug = _PATH_DASH_COLLAPSE_RE.sub("-", slug).strip("-").lower()
            path = f"/api/v1/{slug}" if slug else f"/api/v1/resource-{i}"
        path = re.sub(r"/{2,}", "/", path)
        ep["path"] = path
        method = ep.get("method", "GET")
        key = (method, ep["path"])
        n = 2
        while key in seen:
            ep["path"] = f"{path}-{n}"
            key = (method, ep["path"])
            n += 1
        seen.add(key)
    return plan


def _normalize_endpoint_keys(data: dict) -> dict:
    """EXAONE이 가끔 오타로 내는 필드명(예: errorequestCodes)을 정규화."""
    for wrong, correct in _ENDPOINT_KEY_ALIASES.items():
        if wrong in data and correct not in data:
            data[correct] = data.pop(wrong)
    return data


# EXAONE이 엔드포인트 스펙 대신 자기 추론/메타 설명을 필드에 흘려넣을 때 나타나는 어휘
_ENDPOINT_META_PHRASES = ("커버리지", "누락", "패치", "오타", "결정론", "엔드포인트", "오류 있음", "삭제됨", "수정해야")
_SPEC_FIELD_MAX = 1200  # 정상 스펙 한 필드의 상한; 긴 도메인 응답을 오탐 삭제하지 않음
_CONTRACT_PLACEHOLDER_PHRASES = (
    "계약 기반 요청 본문",
    "계약 기반 성공 응답",
    "Registry 계약 endpoint",
    "Registry 계약 fallback",
    "success: boolean",
)


def _is_garbage_endpoint(ep: dict) -> bool:
    """구조는 유효하지만 내용이 EXAONE 추론·메타텍스트로 오염된 엔드포인트 탐지.
    (예: description에 '결정론적 커버리지 검사...누락...' 서술이 통째로 들어가고
    path가 /api/v1/api/v-{material}/{id} 처럼 중복·오염된 경우)"""
    path = str(ep.get("path") or "")
    if re.search(r'/api/v\d+/api/', path) or 'v-{' in path or path.count('{') > 2:
        return True
    for f in ("description", "requestBody", "successResponse", "errorCodes"):
        v = str(ep.get(f) or "")
        if len(v) > _SPEC_FIELD_MAX:
            return True
    desc = str(ep.get("description") or "")
    if sum(p in desc for p in _ENDPOINT_META_PHRASES) >= 2:
        return True
    return False


def _endpoint_has_placeholder_contract(ep: dict) -> bool:
    """계약 연결만 맞고 실제 API 상세가 비어 있는 endpoint를 찾는다."""
    values = (
        str(ep.get("description") or ""),
        str(ep.get("requestBody") or ""),
        str(ep.get("successResponse") or ""),
        str(ep.get("errorCodes") or ""),
    )
    return any(
        phrase.casefold() in value.casefold()
        for value in values
        for phrase in _CONTRACT_PLACEHOLDER_PHRASES
    )


_PATH_PARAM_RE = re.compile(r'\{(\w+)\}')
_BODYLESS_METHODS = {"GET", "DELETE", "HEAD"}
# 프롬프트 응답 형식 예시가 통째로 복사돼 나온 값 — 내용이 없으므로 기본값으로 교체
_SPEC_ECHO_EXACT = {
    "field1: 타입 (설명) — 없으면 없음",
    "field1: 타입 — 반환 데이터 설명",
    "필드명: 타입 — 설명",
    "본문이 없으면 없음",
    "반환 필드들",
    "없으면 없음",
}
# 실제 내용 뒤에 예시 문구만 눌어붙은 경우 (실측: "barcode: 문자열 (바코드 번호 — 스캔 시 입력, 없으면 없음)")
# — 문구만 떼어내고 나머지 내용은 살린다
_SPEC_ECHO_TAIL_RE = re.compile(r'[,;]?\s*(?:—\s*)?없으면\s*없음')
_EMPTY_PARENS_RE = re.compile(r'\(\s*\)')


def _normalize_parameters(ep: dict) -> None:
    """parameters 배열을 정규화하고, path에 있는 {id} 자리표시자를 빠짐없이 채운다.
    프론트(ApiSpecPanel)가 이미 이 구조를 렌더링하는데 백엔드가 한 번도 채우지 않아
    쿼리 파라미터가 requestBody 문자열에 뭉뚱그려져 있었다."""
    raw = ep.get("parameters")
    params: list[dict] = []
    seen: set[str] = set()
    for p in raw if isinstance(raw, list) else []:
        if isinstance(p, str):
            p = {"name": p}
        if not isinstance(p, dict) or not str(p.get("name") or "").strip():
            continue
        name = str(p["name"]).strip()
        if name in seen:
            continue
        seen.add(name)
        location = str(p.get("in") or "").strip().lower()
        params.append({
            "in": location if location in ("query", "path", "header") else "query",
            "name": name,
            "type": str(p.get("type") or "string").strip(),
            "required": bool(p.get("required", False)),
            "description": str(p.get("description") or "").strip(),
        })

    for name in _PATH_PARAM_RE.findall(str(ep.get("path") or "")):
        if name in seen:
            continue
        seen.add(name)
        params.append({
            "in": "path", "name": name,
            "type": "integer" if name.lower().endswith("id") else "string",
            "required": True, "description": f"{name} 값",
        })

    if params:
        ep["parameters"] = params
    else:
        ep.pop("parameters", None)


# "page: 정수 — 페이지 번호" 처럼 나열된 한 항목에서 이름과 타입을 뽑는다
_BODY_FIELD_RE = re.compile(r'^\s*([A-Za-z_][A-Za-z0-9_]{0,39})\s*[:：]\s*([^—(]*)')
_MAX_SALVAGED_PARAMS = 8


def _param_type(raw: str) -> str:
    t = raw.strip().lower()
    if any(k in t for k in ("정수", "숫자", "int", "number", "long")):
        return "integer"
    if any(k in t for k in ("불리언", "bool")):
        return "boolean"
    return "string"


def _body_to_query_params(text: str) -> list[dict]:
    """본문 없는 메서드의 requestBody에 뭉뚱그려진 필드 나열을 쿼리 파라미터로 회수.
    (실측: GET /api/v1/ingredients 의 page·size·category 가 requestBody 문자열에 들어 있었다)
    형식을 못 읽는 항목은 조용히 건너뛴다 — 최선 노력 복구."""
    salvaged = []
    for chunk in text.split(","):
        match = _BODY_FIELD_RE.match(chunk)
        if not match:
            continue
        salvaged.append({
            "in": "query",
            "name": match.group(1),
            "type": _param_type(match.group(2)),
            "required": False,
            "description": chunk.strip(),
        })
        if len(salvaged) >= _MAX_SALVAGED_PARAMS:
            break
    return salvaged


def _clean_spec_field(value, fallback: str) -> str:
    """프롬프트 예시 문구가 그대로 복사된 값을 걸러낸다.
    값 전체가 예시면 기본값으로 바꾸고, 실제 내용 뒤에 예시 문구만 붙었으면 그 부분만 떼어낸다."""
    text = str(value or "").strip()
    if not text or text in _SPEC_ECHO_EXACT:
        return fallback
    cleaned = _SPEC_ECHO_TAIL_RE.sub("", text)
    cleaned = _EMPTY_PARENS_RE.sub("", cleaned)
    cleaned = re.sub(r'\s{2,}', ' ', cleaned).strip(" ,;—-")
    return cleaned or fallback


def _normalize_endpoints(endpoints: list) -> list:
    """manager 패치/QA 재작성으로 유입된 불완전 엔드포인트에 필수 필드 기본값을 채우고,
    내용이 통째로 오염된(추론/메타텍스트 유출) 엔드포인트는 드롭한다.
    worker의 채움 로직은 patch 경로를 타지 않아 method/path만 있는 엔드포인트가 최종 산출물에
    새어나갔다 — 이를 결정론적으로 보강 (worker _worker_node의 기본값과 동일 기준)."""
    out = []
    for ep in endpoints or []:
        if not isinstance(ep, dict):
            continue
        ep = _normalize_endpoint_keys(ep)
        if _is_garbage_endpoint(ep):
            logger.warning("API 오염 엔드포인트 드롭: %s %s", ep.get("method"), str(ep.get("path"))[:60])
            continue
        ep.setdefault("method", "GET")
        ep.setdefault("path", "/api/v1/unknown")
        ep["path"] = _output_api_path(ep["path"])
        ep["method"] = canonical_api_method(ep["method"], ep["path"])
        ep.setdefault("authRequired", True)
        if not str(ep.get("description") or "").strip():
            ep["description"] = f"{ep.get('method')} {ep.get('path')}"
        body = _clean_spec_field(ep.get("requestBody"), "없음")
        if str(ep["method"]).upper() in _BODYLESS_METHODS and body != "없음":
            # 본문 없는 메서드에 필드가 나열돼 있으면 쿼리 파라미터로 회수한 뒤 본문은 비운다
            salvaged = _body_to_query_params(body)
            if salvaged:
                existing = ep.get("parameters") if isinstance(ep.get("parameters"), list) else []
                ep["parameters"] = list(existing) + salvaged
                logger.info(
                    "API %s %s — requestBody의 필드 %d개를 쿼리 파라미터로 회수",
                    ep["method"], ep["path"], len(salvaged),
                )
            body = "없음"
        ep["requestBody"] = body
        _normalize_parameters(ep)
        ep["successResponse"] = _clean_spec_field(ep.get("successResponse"), "success: boolean")
        ep["errorCodes"] = _clean_spec_field(ep.get("errorCodes"), "401 — 인증 실패, 500 — 서버 오류")
        out.append(ep)
    deduped = _dedupe_semantic_endpoints(out)
    return sorted(
        deduped,
        key=lambda ep: (
            str(ep.get("featureId") or "~"),
            str(ep.get("method") or "GET"),
            str(ep.get("path") or ""),
        ),
    )


def _semantic_endpoint_key(ep: dict) -> tuple[str, str] | None:
    """path만 다른 동일 의미 endpoint를 하나로 묶기 위한 보수적 키.

    LLM manager review가 /auth/signup 과 /auth/registrations 처럼 같은 인증 동작을
    다른 REST 스타일로 다시 추가하는 경우가 있어, 최종 산출물 직전에 의미 중복을 제거한다.
    """
    method = str(ep.get("method") or "GET").upper()
    path = str(ep.get("path") or "").lower().rstrip("/")
    desc = str(ep.get("description") or "").lower()
    text = f"{path} {desc}"

    # featureId가 있으면 서로 다른 계약을 의미 중복으로 합치지 않는다.
    feature_id = str(ep.get("featureId") or "").strip()
    if feature_id:
        return ("feature", f"{feature_id}:{method}:{path}")

    if "/auth/" in path:
        # 인증 공통 prefix만으로 분류하지 않는다. 이메일 인증·OAuth callback·비밀번호
        # 재설정은 signup/signin/refresh와 별도 계약이며, dedupe에서 삭제되면 안 된다.
        # 경로에 동작이 명시되어 있으면 LLM description보다 우선한다.
        if "/refresh" in path:
            return ("auth", "refresh")
        if any(token in path for token in ("/logout", "/signout")):
            return ("auth", "signout")
        if any(token in path for token in ("/verify", "/verification")):
            return ("auth", "verify_email")
        if any(token in path for token in ("/signup", "/register", "/registrations")):
            return ("auth", "signup")
        if any(token in path for token in ("/login", "/signin")):
            return ("auth", "signin")
        if any(token in path for token in ("/verify", "/verification")) or any(token in text for token in ("verify email", "email verification", "이메일 인증", "인증 코드")):
            return ("auth", "verify_email")
        if "/oauth/" in path or "oauth" in text:
            return ("auth", "oauth_callback")
        if "password-reset" in path or any(token in text for token in ("password reset", "비밀번호 재설정", "비밀번호 찾기")):
            # 요청과 확인은 서로 다른 계약이다. path/설명에 confirm이 있으면
            # 첫 단계와 같은 semantic key를 쓰지 않아 후속 endpoint가 삭제되지 않는다.
            suffix = "confirm" if any(token in text for token in ("confirm", "확인", "완료")) else "request"
            return ("auth", f"password_reset_{suffix}")
        if any(token in path for token in ("/signup", "/register", "/registrations")) or any(token in text for token in ("signup", "register", "registration", "회원가입", "가입")):
            return ("auth", "signup")
        # logout 설명에 '토큰'이 함께 들어가도 refresh로 분류하지 않는다.
        if any(token in path for token in ("/logout", "/signout")) or any(
            token in text for token in ("signout", "logout", "sessions/current", "로그아웃")
        ):
            return ("auth", "signout")
        if any(token in path for token in ("/refresh", "/token/refresh")) or any(token in text for token in ("refresh", "token-refresh", "토큰 갱신", "갱신 토큰")):
            return ("auth", "refresh")
        if any(token in text for token in ("signin", "login", "session", "로그인", "인증")) and method == "POST":
            return ("auth", "signin")

    return None


def _dedupe_semantic_endpoints(endpoints: list[dict]) -> list[dict]:
    """METHOD+path 중복보다 한 단계 높은 의미 중복을 제거한다."""
    out: list[dict] = []
    seen_exact: set[tuple[str, str]] = set()
    seen_semantic: set[tuple[str, str]] = set()
    for ep in endpoints:
        method = str(ep.get("method") or "GET").upper()
        path = str(ep.get("path") or "").rstrip("/")
        exact = (method, path)
        if exact in seen_exact:
            logger.warning("API 중복 엔드포인트 드롭: %s %s", method, path)
            continue
        semantic = _semantic_endpoint_key(ep)
        if semantic and semantic in seen_semantic:
            logger.warning("API 의미 중복 엔드포인트 드롭: %s %s (%s)", method, path, semantic)
            continue
        seen_exact.add(exact)
        if semantic:
            seen_semantic.add(semantic)
        out.append(ep)
    return out


def _error_contract(auth_required: bool, target_required: bool = False) -> str:
    errors = ["400 — 요청 값 검증 실패"]
    if auth_required:
        errors.append("401 — 인증 실패")
    if target_required:
        errors.append("404 — 대상 없음")
    errors.append("500 — 서버 내부 오류")
    return ", ".join(errors)


def _endpoint_quality_issues(data: dict | None, raw: str, skeleton: dict, require_self_check: bool = True) -> list[str]:
    if not isinstance(data, dict):
        return ["필수 라벨 파싱 실패"]
    issues = [reason for reason in contamination_reasons(data) if reason != "mixed-language splice"]
    if has_placeholder(data):
        issues.append("placeholder 계약 값 포함")
    if str(data.get("method") or "").upper() != str(skeleton.get("method") or "").upper():
        issues.append("담당 method 변경")
    if str(data.get("path") or "") != str(skeleton.get("path") or ""):
        issues.append("담당 path 변경")
    feature = str(skeleton.get("featureName") or "")
    if feature and relevance_score(feature, str(data.get("description") or "")) < 0.2:
        issues.append("담당 기능과 description 불일치")
    method = str(data.get("method") or "GET").upper()
    body = str(data.get("requestBody") or "")
    if method in _BODYLESS_METHODS and body != "없음":
        issues.append("본문 없는 method에 requestBody 존재")
    if method in {"POST", "PUT", "PATCH"} and ":" not in body:
        issues.append("requestBody에 필드명: 타입 계약 없음")
    if ":" not in str(data.get("successResponse") or ""):
        issues.append("successResponse에 필드명: 타입 계약 없음")
    if len(set(_ERROR_CODE_RE.findall(str(data.get("errorCodes") or "")))) < 2:
        issues.append("표준 4xx/5xx 오류 코드가 2개 미만")
    return list(dict.fromkeys(issues))


def _deterministic_endpoint(skeleton: dict) -> dict:
    method = str(skeleton.get("method") or "GET").upper()
    path = str(skeleton.get("path") or "/api/v1/unknown")
    auth = bool(skeleton.get("authRequired", True))
    if path == "/api/v1/auth/signup":
        body, success = "email: string — 이메일, password: string — 비밀번호", "id: integer — 사용자 ID, email: string — 이메일"
    elif path == "/api/v1/auth/login":
        body, success = "loginId: string — 로그인 식별자, credential: string — 인증 자격 증명", "accessToken: string — 서버가 발급한 접근 토큰"
    elif path == "/api/v1/auth/refresh":
        body, success = "refreshToken: string — 갱신 토큰", "accessToken: string — 새 접근 토큰"
    elif path == "/api/v1/auth/logout":
        body, success = "refreshToken: string — 폐기할 갱신 토큰", "success: boolean — 로그아웃 성공 여부"
    elif method in _BODYLESS_METHODS:
        body, success = "없음", "items: array — 조회 결과, total: integer — 전체 개수"
    else:
        body, success = "payload: object — 기능별 입력 필드", "id: integer — 처리 대상 ID, success: boolean — 처리 성공 여부"
    result = {
        "method": method, "path": path, "description": str(skeleton.get("description") or f"{method} {path}"),
        "featureName": str(skeleton.get("featureName") or ""),
        "authRequired": auth, "requestBody": body, "successResponse": success,
        "errorCodes": _error_contract(auth, "{" in path),
    }
    for field in ("featureId", "action"):
        if field in skeleton:
            result[field] = skeleton[field]
    # Deterministic canonical contracts (auth/me and already ERD-aligned repairs)
    # must survive normalization instead of falling back to payload: object.
    for field in ("requestBody", "successResponse", "errorCodes", "transactionRules"):
        if str(skeleton.get(field) or "").strip():
            result[field] = skeleton[field]
    if not result.get("transactionRules"):
        known_transactions = {
            ("POST", "/api/v1/auth/signup"): "사용자와 자격 증명을 한 DB 트랜잭션으로 생성하고 실패 시 전체를 롤백한다.",
            ("POST", "/api/v1/auth/login"): "자격 증명을 검증한 뒤 인증 상태 변경과 토큰 발급을 일관되게 처리하고 실패 시 변경을 롤백한다.",
            ("PATCH", "/api/v1/users/me"): "인증 주체와 대상 사용자가 같은지 확인한 뒤 한 트랜잭션으로 수정한다.",
            ("PUT", "/api/v1/users/me"): "인증 주체와 대상 사용자가 같은지 확인한 뒤 한 트랜잭션으로 수정한다.",
            ("DELETE", "/api/v1/users/me"): "계정 삭제와 관련 인증 상태 정리를 원자적으로 처리하고 실패 시 롤백한다.",
        }
        if (method, path) in known_transactions:
            result["transactionRules"] = known_transactions[(method, path)]
    if "{id}" in path:
        result["parameters"] = [{"in": "path", "name": "id", "type": "integer", "required": True, "description": "대상 ID"}]
    elif method == "GET":
        result["parameters"] = [
            {"in": "query", "name": "page", "type": "integer", "required": False, "description": "페이지 번호"},
            {"in": "query", "name": "size", "type": "integer", "required": False, "description": "페이지 크기"},
        ]
    return result


def _column_names(table: dict) -> set[str]:
    return {str(column.get("name")) for column in table.get("columns") or [] if isinstance(column, dict)}


def _referenced_tables(table: dict) -> set[str]:
    targets = set()
    for column in table.get("columns") or []:
        if not isinstance(column, dict):
            continue
        match = re.search(r"REFERENCES\s+([a-z][a-z0-9_]*)", str(column.get("constraints") or ""), re.I)
        if match:
            targets.add(match.group(1).lower())
    return targets


def _english_resource_stems(value: str) -> set[str]:
    stems = set()
    for token in re.findall(r"[a-z]+", str(value or "").lower().replace("-", "_")):
        for suffix in ("ments", "ment", "ees", "ee", "ed", "ing", "ies", "s"):
            if token.endswith(suffix) and len(token) > len(suffix) + 2:
                token = token[:-len(suffix)]
                break
        stems.add(token)
    return stems


def _stateful_aggregate_for_catalog(catalog: str, tables: dict[str, dict]) -> dict | None:
    candidates = []
    for table in tables.values():
        table_name = str(table.get("name") or "")
        if any(token in table_name for token in (
            "reservation", "booking", "rental", "loan", "order", "application",
            "request", "ticket", "record", "history", "log", "transaction",
        )):
            continue
        columns = _column_names(table)
        if "user_id" in columns and columns & _STATE_FIELD_NAMES and catalog in _referenced_tables(table):
            candidates.append(table)
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        ownership_fields = {
            "quantity", "amount", "balance", "unit", "expiry_date", "stock", "remaining_count", "position", "progress",
        }
        scores = {str(table.get("name")): len(_column_names(table) & ownership_fields) for table in candidates}
        best = max(scores.values(), default=0)
        winners = [table for table in candidates if scores[str(table.get("name"))] == best]
        if best and len(winners) == 1:
            return winners[0]
    return None


_ERROR_CODE_RE = re.compile(r"\b[45]\d{2}\b")


_STATE_FIELD_NAMES = {
    "quantity", "amount", "balance", "status", "state", "unit", "expiry_date",
    "expires_at", "scheduled_at", "position", "progress", "stock", "remaining_count",
}
