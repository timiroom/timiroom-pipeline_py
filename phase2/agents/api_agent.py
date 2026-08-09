import asyncio
import operator
import json
import logging
import re
from typing import Annotated, TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.types import Send
from openai import AsyncOpenAI, InternalServerError, APITimeoutError, APIConnectionError

from phase2.feature_coverage import uncovered_features, undercovered_features, missing_features_note
from phase2.json_utils import try_parse_json, has_suspicious_script
from phase2.state import PipelineState
from phase2.llm_concurrency import llm_slot
from phase2.quality_rules import contamination_reasons, has_placeholder, relevance_score, retry_prompt, self_check_passed
from phase2.agent_contract import (
    contract_prompt, feature_methods, feature_relation_kind, requires_auth,
)

# EXAONE이 가끔 오타로 내는 필드명 -> 정규화
_ENDPOINT_KEY_ALIASES = {
    "errorequestCodes": "errorCodes",
    "errorCode": "errorCodes",
    "successresponse": "successResponse",
    "requestbody": "requestBody",
}

logger = logging.getLogger(__name__)

# EXAONE 모델 카드 권장 샘플링 파라미터
# https://huggingface.co/LGAI-EXAONE/K-EXAONE-236B-A23B
_TEMPERATURE = 1.0
_TOP_P = 0.95
_PRESENCE_PENALTY = 0.0
# 엔드포인트 목록(plan) 생성은 개수·커버리지 일관성이 중요한 구조 생성 단계 —
# temp=1.0은 실행마다 편차를 키우므로 이 단계만 낮춰 완성도/재현성을 높인다.
_PLAN_TEMPERATURE = 0.4

# plan 생성 배치 크기 — 한 번에 담당할 기능 수.
# 기능 전체(21개)를 한 프롬프트에 넣으면 EXAONE이 요구량의 1/3 수준에서 멈춘다(실측 12/42).
_PLAN_BATCH_SIZE = 4
_ENDPOINTS_PER_FEATURE = 2
_AUTH_ENDPOINT_COUNT = 4  # 회원가입/로그인/토큰갱신/로그아웃

_AUTHENTICATION_DESC = (
    "요구사항에 명시된 로그인 주체를 검증하고 서버가 발급한 세션 또는 접근 토큰을 "
    "보호 API 요청에 전달한다. 구체 방식은 배포 보안 정책에서 결정한다."
)

WORKER_SYSTEM = "JSON만 출력하세요. 설명·인사말·마크다운 코드블록 금지. { 로 시작해서 } 로 끝납니다."

MANAGER_SYSTEM = """당신은 시니어 백엔드 아키텍트 겸 API manager입니다.
JSON만 출력하세요. 설명·인사말·마크다운 코드블록 금지. { 로 시작해서 } 로 끝납니다."""

PLAN_PROMPT = """당신은 시니어 백엔드 아키텍트입니다.
아래 지시사항과 컨텍스트를 바탕으로, **담당 기능들**에 필요한 REST API 엔드포인트 목록(스켈레톤)을
설계하세요. 상세 스펙(requestBody/successResponse/errorCodes)은 이후 단계에서 채울 것이므로
지금은 엔드포인트의 method/path/description/인증 필요 여부만 결정하세요.

설계 규칙:
{auth_rule}- RESTful 설계: 명사형 복수형 경로, 반드시 영문 소문자·숫자·하이픈(-)만 사용 (예: /api/v1/book-clubs)
- path에 한글, 공백, 괄호(), 콜론(:), 쉼표 등은 절대 사용 금지 — 기능명이 한글이어도
  의미를 압축한 영문 리소스명으로 직접 번역해서 사용 (예: "재료 등록(바코드 스캔)" 기능 → /api/v1/ingredients, 절대 /api/v1/재료-등록-(바코드-스캔) 처럼 쓰지 말 것)
- **담당 기능 하나하나마다** 실제로 필요한 조회·생성·수정·삭제 엔드포인트를 빠짐없이 설계
- description은 반드시 "기능명: 설명" 형태로 시작해 어느 기능에 대응하는지 드러낼 것
- 목록 조회 엔드포인트는 페이지네이션이 필요함을 description에 명시
- 담당 기능은 {feature_count}개이므로 최소 {min_endpoints}개 이상의 엔드포인트가 나와야 합니다
- 담당 기능 밖의 엔드포인트는 만들지 마세요 (다른 담당자가 설계합니다)
{existing_note}
응답 형식 (JSON만):
{{
  "plan": [
    {{
      "method": "GET 또는 POST 또는 PUT 또는 DELETE 또는 PATCH",
      "path": "/api/v1/영문-리소스명 (한글 금지, 예: /api/v1/ingredients)",
      "description": "기능명: 이 API가 하는 일 한 줄 설명",
      "authRequired": true 또는 false
    }}
  ]
}}

지시사항:
{instruction}

컨텍스트 (DB 스키마 포함):
{context}

담당 기능 목록 ({feature_count}개):
{feature_str}
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

ENDPOINT_SPEC_PROMPT = """당신은 시니어 백엔드 개발자입니다.
아래 엔드포인트 스켈레톤 1개에 대한 상세 REST API 스펙을 JSON으로 작성하세요.

담당 엔드포인트:
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
- 위 지시문의 예시 문구("없으면 없음", "필드명" 등)를 값에 그대로 옮겨 쓰지 말 것 — 실제 내용만

응답 형식 (JSON만, method/path/description/authRequired는 위 값 그대로 유지):
{{
  "method": "{method}",
  "path": "{path}",
  "description": "{description}",
  "authRequired": {auth_required},
  "parameters": [
    {{"in": "query 또는 path", "name": "파라미터명", "type": "string/integer/boolean", "required": true 또는 false, "description": "설명"}}
  ],
  "requestBody": "본문이 없으면 없음",
  "successResponse": "반환 필드들",
  "errorCodes": "401 — 인증 실패, 404 — 리소스 없음"
}}

컨텍스트 (DB 스키마 포함):
{context}
"""

MANAGER_REVIEW_PROMPT = """아래는 sub-agent들이 작성한 API 엔드포인트 상세 스펙 전체입니다.

=== 엔드포인트 목록 ===
{endpoints_json}
=================

=== 검증 기준 ===
- 기능 목록의 모든 기능에 대응하는 엔드포인트가 존재하는가?
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
  "patches": {{
    "METHOD /api/v1/path": {{"method":"...","path":"...","description":"...","authRequired":true,"requestBody":"...","successResponse":"...","errorCodes":"..."}}
  }}
}}

문제되는 엔드포인트가 하나도 없으면 반드시 {{"patches": {{}}}}로 응답하세요.
"""

ENDPOINT_TEXT_SYSTEM = """API 엔드포인트 하나를 평문으로 작성합니다. JSON과 마크다운은 금지합니다.
METHOD, PATH, DESCRIPTION, AUTH_REQUIRED, REQUEST_BODY, SUCCESS_RESPONSE, ERROR_CODES 라벨을 각각 한 번 사용하세요.
담당 엔드포인트 밖의 기능을 섞지 말고 마지막 줄에 SELF_CHECK: PASS를 출력하세요."""


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
    return data


def _feature_resource_slug(feature: str, index: int) -> str:
    text = str(feature or "")
    fallback = re.sub(r"[^a-z0-9]+", "-", text.lower().strip())
    fallback = re.sub(r"-+", "-", fallback).strip("-")
    if not fallback:
        return f"feature-{index + 1}"
    return f"{fallback[:40]}-{index + 1}"


def _feature_methods(feature: str) -> tuple[str, ...]:
    return feature_methods(feature)


_ERROR_CODE_RE = re.compile(r"\b[45]\d{2}\b")


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
    # Deterministic canonical contracts (auth/me and already ERD-aligned repairs)
    # must survive normalization instead of falling back to payload: object.
    for field in ("requestBody", "successResponse", "errorCodes", "transactionRules"):
        if str(skeleton.get(field) or "").strip():
            result[field] = skeleton[field]
    if "{id}" in path:
        result["parameters"] = [{"in": "path", "name": "id", "type": "integer", "required": True, "description": "대상 ID"}]
    elif method == "GET":
        result["parameters"] = [
            {"in": "query", "name": "page", "type": "integer", "required": False, "description": "페이지 번호"},
            {"in": "query", "name": "size", "type": "integer", "required": False, "description": "페이지 크기"},
        ]
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


_STATE_FIELD_NAMES = {
    "quantity", "amount", "balance", "status", "state", "unit", "expiry_date",
    "expires_at", "scheduled_at", "position", "progress", "stock", "remaining_count",
}


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


def _align_endpoints_to_db(
    endpoints: list[dict], db_schema: str, prd_document: str = "", requirement_text: str = "",
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
        table = tables.get(slug.replace("-", "_"))
        feature_label = str(ep.get("featureName") or ep.get("description") or "").split(":", 1)[0].strip()
        if table and (_column_names(table) & {"credential_hash", "password_hash", "login_id"}):
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
        # If one unambiguous stateful aggregate references that catalog, make it the API resource.
        table_columns = _column_names(table) if table else set()
        is_stateful_resource = bool(table and "user_id" in table_columns and table_columns & _STATE_FIELD_NAMES)
        if table and not is_stateful_resource and "user_id" not in table_columns:
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
            if aggregate:
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
        writable = [
            c for c in business
            if c["name"] != "user_id" or association_contract
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
        result.append(dict(canonical.get(key, endpoint)))
        seen.add(key)
    result.extend(ep for key, ep in canonical.items() if key not in seen)
    return result


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


class _ApiGraphState(TypedDict):
    plan: list
    authentication: str
    endpoints: Annotated[list, operator.add]
    api_spec: str
    ctx: dict


class ApiAgent:

    def __init__(
        self,
        client: AsyncOpenAI,
        model: str = "gpt-4o-mini",
    ):
        self._client = client
        self._model = model
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

        context = (
            f"PRD: {(state.prd_document or '{}')[:8000]}\n"
            f"DB_SCHEMA: {(state.db_schema or '{}')[:8000]}\n"
            f"Phase 1 evidence: {(state.context_prompt or '')[:6000]}"
        )
        instruction = state.api_instruction or ""
        feature_str = "- " + "\n- ".join(state.feature_list) if state.feature_list else "(기능 목록 없음)"

        graph_input = {
            "plan": [],
            "authentication": "",
            "endpoints": [],
            "api_spec": "",
            "ctx": {
                "context": context,
                "instruction": instruction,
                "feature_str": feature_str,
                "feature_list": state.feature_list,
                "db_schema": state.db_schema or "{}",
                "prd_document": state.prd_document or "{}",
                "dump": dump,
            },
        }

        try:
            result = await self._graph.ainvoke(graph_input)
            api_spec = result["api_spec"]
        except Exception as e:
            logger.error("API 에이전트 실패: %s", e)
            return state.copy(
                api_spec="{}",
                status_message=f"API 에이전트 실패: {e}",
            )

        logger.info("API 에이전트 완료")
        return state.copy(
            api_spec=api_spec,
            prd_feedback_from_api="",
            status_message="API 에이전트 완료 — API 스펙 생성",
        )

    async def repair(self, state: PipelineState, issues: list[dict], dump=None) -> PipelineState:
        """Reconcile existing endpoints with ERD fields without rerunning the plan."""
        parsed = try_parse_json(state.api_spec or "{}") or {}
        endpoints = parsed.get("endpoints") if isinstance(parsed, dict) else None
        if not isinstance(endpoints, list):
            return state
        logger.info("API 선택 교정 — 결함 %d건, 기존 endpoint만 재조립", len(issues))
        auth_required = requires_auth(state.prd_document, state.feature_list, state.context_prompt)
        endpoints = _ensure_auth_endpoints(endpoints, auth_required)
        endpoints = _align_endpoints_to_db(
            endpoints, state.db_schema, state.prd_document, state.context_prompt,
        )
        endpoints = _ensure_feature_endpoint_groups(
            endpoints, state.db_schema, state.feature_list, state.prd_document, state.context_prompt
        )
        parsed["endpoints"] = _normalize_endpoints(_align_endpoints_to_db(
            endpoints, state.db_schema, state.prd_document, state.context_prompt,
        ))
        parsed["featureMappings"] = _api_feature_mappings(
            parsed["endpoints"], state.db_schema, state.feature_list,
        )
        return state.copy(api_spec=json.dumps(parsed, ensure_ascii=False))

    async def _plan_batch(
        self, ctx: dict, batch: list[str], batch_idx: int, with_auth: bool,
        existing_paths: list[str] | None = None,
    ) -> list[dict]:
        """기능 몇 개만 담당하는 plan 배치 하나를 생성한다.

        기능 21개에 엔드포인트 42개를 한 번에 요구하면 EXAONE이 12개쯤에서 멈춰
        (실측) 레시피·알림·장보기 같은 기능군이 통째로 빠진 채 통과됐다. 담당 범위를
        좁히면 요구 개수가 배치당 8~10개로 내려가 실제로 채워진다."""
        min_endpoints = len(batch) * _ENDPOINTS_PER_FEATURE + (_AUTH_ENDPOINT_COUNT if with_auth else 0)
        prompt = PLAN_PROMPT.format(
            instruction=ctx["instruction"],
            context=ctx["context"],
            feature_str="- " + "\n- ".join(batch),
            feature_count=len(batch),
            min_endpoints=min_endpoints,
            auth_rule=_AUTH_RULE if with_auth else "",
            existing_note=_existing_paths_note(existing_paths),
        )
        label = f"API_MANAGER_PLAN_{batch_idx}"

        best: list[dict] = []
        for attempt in range(3):
            raw = await self._call(
                prompt, max_tokens=8192, system=MANAGER_SYSTEM,
                enable_thinking=False, temperature=_PLAN_TEMPERATURE,
            )
            if ctx.get("dump"):
                ctx["dump"].log_raw(label, attempt + 1, raw)
            candidate = try_parse_json(raw)
            if not (candidate and isinstance(candidate.get("plan"), list)) or has_suspicious_script(candidate):
                logger.warning("%s 파싱 실패/오염 (attempt %d) — 재생성", label, attempt + 1)
                continue
            plan_list = [ep for ep in candidate["plan"] if isinstance(ep, dict)]
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
        auth_required = requires_auth(
            ctx.get("prd_document", ""), feature_list, ctx.get("context", "")
        )
        batches = [
            feature_list[index:index + _PLAN_BATCH_SIZE]
            for index in range(0, len(feature_list), _PLAN_BATCH_SIZE)
        ] or [[]]
        planned_batches: list[list[dict]] = []
        existing_paths: list[str] = []
        for index, batch in enumerate(batches):
            candidate = await self._plan_batch(
                ctx, batch, index + 1, auth_required and index == 0, existing_paths,
            )
            planned_batches.append(candidate)
            existing_paths.extend(
                f"{item.get('method', '')} {item.get('path', '')}"
                for item in candidate if isinstance(item, dict)
            )
        plan = self._merge_plans(planned_batches)

        missing = uncovered_features(
            feature_list, [str(item.get("description") or "") for item in plan]
        )
        if not plan or missing:
            missing_features = missing or feature_list
            slugs = await asyncio.gather(*[
                self._resource_slug_worker(feature, index, ctx.get("dump"))
                for index, feature in enumerate(missing_features)
            ])
            fallback = self._fallback_plan(
                missing_features, slugs, auth_required and not plan,
            )["plan"]
            plan = self._merge_plans([plan, fallback])
        plan = _sanitize_plan_paths(plan)
        logger.info("API 결정론적 plan 완료 — 엔드포인트 %d개 계획 (목표 %d개)", len(plan), min_endpoints)
        return {
            "plan": plan,
            "authentication": _AUTHENTICATION_DESC if auth_required else "인증 요구 없음",
        }

    async def _resource_slug_worker(self, feature: str, index: int, dump=None) -> str:
        prompt = (
            "아래 기능의 실제 의미를 영문 kebab-case 복수 명사 REST 리소스로 번역하세요. "
            "example-resources, feature-N 같은 예시·자리표시자와 JSON은 금지합니다.\n"
            f"번역할 기능: {feature}\n첫 줄은 RESOURCE: 로 시작하고 마지막 줄은 SELF_CHECK: PASS로 끝내세요."
        ) + contract_prompt("API Resource Worker", feature)
        for attempt in range(2):
            raw = await self._call(
                prompt, 120,
                "You translate only the supplied Korean feature into one accurate English REST resource noun. Output exactly RESOURCE and SELF_CHECK lines.",
                False, _PLAN_TEMPERATURE,
            )
            if dump:
                dump.log_raw(f"API_RESOURCE_{index + 1}", attempt + 1, raw)
            match = re.search(r"^\s*RESOURCE\s*:\s*([a-z][a-z0-9-]{1,50})\s*$", raw or "", re.I | re.M)
            slug = match.group(1).lower() if match else ""
            if slug and slug not in {"example-resources", "user-accounts"} and not slug.startswith("feature-") and self_check_passed(raw):
                return slug
            prompt += "\n이전 응답은 실제 기능 번역이 아니었습니다. 기능 의미가 드러나는 다른 이름으로 고치세요."
        return _feature_resource_slug(feature, index)

    def _fallback_plan(
        self, feature_list: list[str], slugs: list[str] | None = None,
        auth_required: bool = False,
    ) -> dict:
        plan = []
        if auth_required:
            plan.append({
                "method": "POST", "path": "/api/v1/auth/login",
                "description": "요구사항에 명시된 사용자 로그인",
                "authRequired": False, "featureName": "로그인",
            })
        for index, feature in enumerate(feature_list):
            slug = slugs[index] if slugs and index < len(slugs) else _feature_resource_slug(feature, index)
            collection = f"/api/v1/{slug}"
            for method in _feature_methods(feature):
                path = collection + "/{id}" if method in {"PATCH", "DELETE"} else collection
                action = {"GET": "목록 조회", "POST": "생성 또는 실행", "PATCH": "부분 수정", "DELETE": "삭제"}[method]
                plan.append({"method": method, "path": path, "description": f"{feature}: {action}",
                             "authRequired": auth_required, "featureName": feature})
        plan = _sanitize_plan_paths(plan)
        return {
            "plan": plan,
            "authentication": _AUTHENTICATION_DESC if auth_required else "인증 요구 없음",
        }

    def _dispatch(self, state: dict) -> list[Send]:
        ctx = state["ctx"]
        sends = []
        for skeleton in state["plan"]:
            cross_context = (
                f"{ctx['context']}\nDB_SCHEMA: {ctx.get('db_schema', '{}')}\n"
                f"PRD: {ctx.get('prd_document', '{}')}"
            )
            sends.append(Send("worker", {"skeleton": skeleton, "context": cross_context, "dump": ctx.get("dump")}))
        return sends

    async def _worker_node(self, state: dict) -> dict:
        skeleton = state["skeleton"]
        method = skeleton.get("method", "GET")
        path = skeleton.get("path", "/api/v1/unknown")

        # method/path/ERD가 이미 확정된 뒤의 계약은 생성 작업이 아니라 조립 작업이다.
        # LLM이 GET/DELETE 본문을 만들거나 필드명을 바꾸는 반복 실패를 제거하고,
        # manager의 ERD alignment가 최종 request/response 필드를 채우도록 한다.
        data = _deterministic_endpoint(skeleton)
        logger.debug("%s %s 계약 결정론적 조립", method, path)
        return {"endpoints": [data]}

    async def _manager_review_node(self, state: dict) -> dict:
        ctx = state["ctx"]
        endpoints = state["endpoints"]
        authentication = state["authentication"]
        dump = ctx.get("dump")

        bad_paths = _invalid_paths(endpoints)
        if bad_paths:
            logger.warning("API manager_review — 패치로 유입된 잘못된 경로 살균: %s", bad_paths)
            endpoints = _sanitize_plan_paths(endpoints)

        # 패치로 유입된 불완전 엔드포인트(method/path만 있는 경우 등)에 필수 필드 기본값 보강
        endpoints = _normalize_endpoints(endpoints)
        auth_required = requires_auth(
            ctx.get("prd_document", ""), ctx.get("feature_list") or [], ctx.get("context") or "",
        )
        endpoints = _ensure_auth_endpoints(endpoints, auth_required)
        endpoints = [
            _deterministic_endpoint(ep)
            if _endpoint_quality_issues(ep, "SELF_CHECK: PASS", ep, require_self_check=False)
            else ep
            for ep in endpoints
        ]
        endpoints = _align_endpoints_to_db(
            endpoints, ctx.get("db_schema") or "{}", ctx.get("prd_document") or "",
            ctx.get("context") or "",
        )

        api_spec = json.dumps({
            "endpoints": endpoints, "authentication": authentication,
            "featureMappings": _api_feature_mappings(
                endpoints, ctx.get("db_schema") or "{}", ctx.get("feature_list") or [],
            ),
        }, ensure_ascii=False)
        return {"api_spec": api_spec}

    async def _call(self, user_prompt: str, max_tokens: int, system: str, enable_thinking: bool, temperature: float = _TEMPERATURE) -> str:
        for attempt in range(2):
            try:
                async with llm_slot():
                    response = await self._client.chat.completions.create(
                        model=self._model, temperature=temperature, top_p=_TOP_P,
                        presence_penalty=_PRESENCE_PENALTY, max_tokens=max_tokens,
                        frequency_penalty=0.5,
                        messages=[{"role": "system", "content": system}, {"role": "user", "content": user_prompt}],
                        extra_body={"chat_template_kwargs": {"enable_thinking": enable_thinking}},
                    )
                return response.choices[0].message.content or ""
            except (InternalServerError, APITimeoutError, APIConnectionError) as e:
                logger.warning("API 호출 일시 오류 (attempt %d): %s — 재시도", attempt + 1, e)
                if attempt < 1:
                    await asyncio.sleep(1)
                else:
                    logger.error("API 호출 최종 실패")
                    return ""
        return ""


_VALID_PATH_RE = re.compile(r'^/[A-Za-z0-9/_\-{}]*$')
_PATH_STRIP_RE = re.compile(r'[^A-Za-z0-9\-{}/]+')
_PATH_DASH_COLLAPSE_RE = re.compile(r'-{2,}')


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


def _sanitize_plan_paths(plan: list) -> list:
    """유효하지 않은 path를 ASCII 슬러그로 강제 변환 (재생성 3회 실패 시 최후 수단)."""
    seen: set[tuple[str, str]] = set()
    for i, ep in enumerate(plan):
        if not isinstance(ep, dict):
            continue
        path = ep.get("path") or ""
        path = re.sub(r"^/api/v\d+/", "/api/v1/", path, flags=re.I)
        ep["path"] = path
        if not _VALID_PATH_RE.match(path):
            rest = path[len("/api/v1/"):] if path.startswith("/api/v1/") else path.lstrip("/")
            slug = _PATH_STRIP_RE.sub("-", rest)
            slug = _PATH_DASH_COLLAPSE_RE.sub("-", slug).strip("-").lower()
            path = f"/api/v1/{slug}" if slug else f"/api/v1/resource-{i}"
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
_SPEC_FIELD_MAX = 1200  # 여러 구체 필드를 나열한 정상 계약은 보존하되 장문 추론 유출은 차단


def _is_garbage_endpoint(ep: dict) -> bool:
    """구조는 유효하지만 내용이 EXAONE 추론·메타텍스트로 오염된 엔드포인트 탐지.
    (예: description에 '결정론적 커버리지 검사...누락...' 서술이 통째로 들어가고
    path가 /api/v1/api/v-{material}/{id} 처럼 중복·오염된 경우)"""
    path = str(ep.get("path") or "")
    if re.search(r'/api/v\d+/api/', path) or 'v-{' in path or path.count('{') > 2:
        return True
    for f in ("description", "requestBody", "successResponse", "errorCodes"):
        v = str(ep.get(f) or "")
        if len(v) > _SPEC_FIELD_MAX or '\n' in v:  # 여러 줄/과도 길이 = 스펙 아님
            return True
    desc = str(ep.get("description") or "")
    if sum(p in desc for p in _ENDPOINT_META_PHRASES) >= 2:
        return True
    return False


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
        if str(ep.get("method") or "").upper() == "PUT" and "{" in str(ep.get("path") or ""):
            ep["method"] = "PATCH"
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
    return out
