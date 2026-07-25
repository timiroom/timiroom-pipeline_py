import asyncio
import operator
import json
import logging
import re
from typing import Annotated, TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.types import Send
from openai import AsyncOpenAI, InternalServerError, APITimeoutError, APIConnectionError

from phase2.feature_coverage import uncovered_features, missing_features_note
from phase2.json_utils import try_parse_json, has_suspicious_script
from phase2.state import PipelineState

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

WORKER_SYSTEM = "JSON만 출력하세요. 설명·인사말·마크다운 코드블록 금지. { 로 시작해서 } 로 끝납니다."

MANAGER_SYSTEM = """당신은 시니어 백엔드 아키텍트 겸 API manager입니다.
JSON만 출력하세요. 설명·인사말·마크다운 코드블록 금지. { 로 시작해서 } 로 끝납니다."""

PLAN_PROMPT = """당신은 시니어 백엔드 아키텍트입니다.
아래 지시사항과 컨텍스트를 바탕으로 이 서비스에 필요한 전체 REST API 엔드포인트 목록(스켈레톤)과
인증 방식을 설계하세요. 상세 스펙(requestBody/successResponse/errorCodes)은 이후 단계에서 채울 것이므로
지금은 엔드포인트의 method/path/description/인증 필요 여부만 결정하세요.

설계 규칙:
- 회원가입, 로그인, 토큰 갱신, 로그아웃 등 인증 흐름에 필요한 엔드포인트를 반드시 포함
- RESTful 설계: 명사형 복수형 경로, 반드시 영문 소문자·숫자·하이픈(-)만 사용 (예: /api/v1/book-clubs)
- path에 한글, 공백, 괄호(), 콜론(:), 쉼표 등은 절대 사용 금지 — 기능명이 한글이어도
  의미를 압축한 영문 리소스명으로 직접 번역해서 사용 (예: "재료 등록(바코드 스캔)" 기능 → /api/v1/ingredients, 절대 /api/v1/재료-등록-(바코드-스캔) 처럼 쓰지 말 것)
- 아래 기능들 각각에 대해 실제로 필요한 조회·생성·수정·삭제 엔드포인트를 빠짐없이 설계
- 목록 조회 엔드포인트는 페이지네이션이 필요함을 description에 명시
- 최소 {min_endpoints}개 이상의 엔드포인트를 설계하세요

응답 형식 (JSON만):
{{
  "plan": [
    {{
      "method": "GET 또는 POST 또는 PUT 또는 DELETE 또는 PATCH",
      "path": "/api/v1/영문-리소스명 (한글 금지, 예: /api/v1/ingredients)",
      "description": "기능명: 이 API가 하는 일 한 줄 설명",
      "authRequired": true 또는 false
    }}
  ],
  "authentication": "JWT Bearer 토큰 방식. 로그인 후 accessToken을 Authorization: Bearer {{token}} 헤더로 전송."
}}

지시사항:
{instruction}

컨텍스트 (DB 스키마 포함):
{context}

기능 목록:
{feature_str}
"""

ENDPOINT_SPEC_PROMPT = """당신은 시니어 백엔드 개발자입니다.
아래 엔드포인트 스켈레톤 1개에 대한 상세 REST API 스펙을 JSON으로 작성하세요.

담당 엔드포인트:
- method: {method}
- path: {path}
- description: {description}
- authRequired: {auth_required}

설계 규칙:
- requestBody와 successResponse는 반드시 문자열로 작성 (객체 금지)
- 목록 조회 엔드포인트라면 requestBody에 page, size 쿼리 파라미터 포함
- errorCodes는 "코드 — 설명" 형식으로 2개 이상 작성

응답 형식 (JSON만, method/path/description/authRequired는 위 값 그대로 유지):
{{
  "method": "{method}",
  "path": "{path}",
  "description": "{description}",
  "authRequired": {auth_required},
  "requestBody": "field1: 타입 (설명) — 없으면 없음",
  "successResponse": "field1: 타입 — 반환 데이터 설명",
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

        context = state.context_prompt or ""
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

    async def _manager_plan_node(self, state: dict) -> dict:
        ctx = state["ctx"]
        min_endpoints = max(4, len(ctx["feature_list"]) * 2)
        prompt = PLAN_PROMPT.format(
            instruction=ctx["instruction"],
            context=ctx["context"],
            feature_str=ctx["feature_str"],
            min_endpoints=min_endpoints,
        )

        data = None
        best: dict | None = None
        best_uncovered = None
        for attempt in range(3):
            raw = await self._call(prompt, max_tokens=16384, system=MANAGER_SYSTEM, enable_thinking=False, temperature=_PLAN_TEMPERATURE)
            if ctx.get("dump"):
                ctx["dump"].log_raw("API_MANAGER_PLAN", attempt + 1, raw)
            candidate = try_parse_json(raw)
            bad_paths = _invalid_paths(candidate.get("plan")) if candidate and isinstance(candidate.get("plan"), list) else []
            plan_list = candidate.get("plan") if candidate and isinstance(candidate.get("plan"), list) else []
            missing = uncovered_features(ctx["feature_list"], [ep.get("description", "") for ep in plan_list if isinstance(ep, dict)])
            if best_uncovered is None or len(missing) < best_uncovered:
                best, best_uncovered = candidate, len(missing)
            if (
                candidate and isinstance(candidate.get("plan"), list)
                and len(candidate["plan"]) >= min_endpoints
                and not has_suspicious_script(candidate)
                and not bad_paths
                and not missing
            ):
                data = candidate
                break
            logger.warning(
                "API manager_plan 파싱 실패/개수 부족/경로 오류/기능 커버리지 미달 (attempt %d) — %d/%d개, "
                "잘못된 경로: %s, 미반영 기능: %s — 재생성",
                attempt + 1, len(plan_list), min_endpoints, bad_paths, missing,
            )

        if not data:
            data = best
            if best_uncovered:
                logger.warning("API manager_plan — 커버리지 기준 미달이지만 재생성 한도 도달, 최선의 결과로 진행")
        if not data:
            logger.error("API manager_plan 최종 실패 — feature_list 기반 fallback 스켈레톤 사용")
            data = self._fallback_plan(ctx["feature_list"])
        elif _invalid_paths(data.get("plan")):
            logger.warning("API manager_plan — 경로 형식 오류가 남아있어 강제 살균 적용")
            data["plan"] = _sanitize_plan_paths(data["plan"])

        plan = data.get("plan") or self._fallback_plan(ctx["feature_list"])["plan"]
        authentication = data.get("authentication") or "JWT Bearer 토큰"

        logger.info("API manager_plan 완료 — 엔드포인트 %d개 계획", len(plan))
        return {"plan": plan, "authentication": authentication}

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
            method=method,
            path=path,
            description=description,
            auth_required=str(auth_required).lower(),
            context=context,
        )

        label = f"API_ENDPOINT_{method}_{path}"
        data = None
        for attempt in range(3):
            raw = await self._call(prompt, max_tokens=1200, system=WORKER_SYSTEM, enable_thinking=False)
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
                "requestBody": "없음",
                "successResponse": "success: boolean",
                "errorCodes": "500 — 서버 오류",
            }
        else:
            data.setdefault("method", method)
            data.setdefault("path", path)
            data.setdefault("description", description)
            data.setdefault("authRequired", auth_required)
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
        dump = ctx.get("dump")

        missing = uncovered_features(ctx["feature_list"], [ep.get("description", "") for ep in endpoints if isinstance(ep, dict)])
        if missing:
            logger.warning("API manager_review — 커버리지 부족 감지: %s", missing)

        prompt = MANAGER_REVIEW_PROMPT.format(
            endpoints_json=json.dumps(endpoints, ensure_ascii=False),
            feature_str=ctx["feature_str"],
            context=ctx["context"],
            missing_note=missing_features_note(missing, "API 스펙"),
        )

        try:
            review = None
            for parse_attempt in range(2):
                raw = await self._call(prompt, max_tokens=16384, system=MANAGER_SYSTEM, enable_thinking=False)
                if dump:
                    dump.log_raw("API_MANAGER_REVIEW", parse_attempt + 1, raw)
                review = try_parse_json(raw)
                if review and isinstance(review, dict):
                    break
                logger.warning("API manager 리뷰 파싱 실패 (시도 %d) — 재시도", parse_attempt + 1)
            if review is None or not isinstance(review, dict):
                logger.warning("API manager 리뷰 파싱 최종 실패 — 원본 엔드포인트 유지")
            else:
                patches = review.get("patches")
                if isinstance(patches, dict) and patches:
                    valid_patches = {k: v for k, v in patches.items() if isinstance(v, dict)}
                    dropped = set(patches) - set(valid_patches)
                    if dropped:
                        logger.warning("API manager 리뷰 — 패치 값이 dict가 아니어서 무시: %s", list(dropped))
                    if valid_patches:
                        logger.info("API manager 리뷰 — %d개 엔드포인트 패치: %s", len(valid_patches), list(valid_patches.keys()))
                        by_key = {f"{ep.get('method')} {ep.get('path')}": ep for ep in endpoints}
                        by_key.update(valid_patches)
                        endpoints = list(by_key.values())
                else:
                    logger.info("API manager 리뷰 — 패치 없음, 원본 유지")
        except Exception as e:
            logger.warning("API manager 리뷰 실패 — 원본 엔드포인트 유지: %s", e)

        bad_paths = _invalid_paths(endpoints)
        if bad_paths:
            logger.warning("API manager_review — 패치로 유입된 잘못된 경로 살균: %s", bad_paths)
            endpoints = _sanitize_plan_paths(endpoints)

        # 패치로 유입된 불완전 엔드포인트(method/path만 있는 경우 등)에 필수 필드 기본값 보강
        endpoints = _normalize_endpoints(endpoints)

        api_spec = json.dumps({"endpoints": endpoints, "authentication": authentication}, ensure_ascii=False)
        return {"api_spec": api_spec}

    async def _call(self, user_prompt: str, max_tokens: int, system: str, enable_thinking: bool, temperature: float = _TEMPERATURE) -> str:
        for attempt in range(3):
            try:
                response = await self._client.chat.completions.create(
                    model=self._model,
                    temperature=temperature,
                    top_p=_TOP_P,
                    presence_penalty=_PRESENCE_PENALTY,
                    max_tokens=max_tokens,
                    frequency_penalty=0.5,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user_prompt},
                    ],
                    extra_body={"chat_template_kwargs": {"enable_thinking": enable_thinking}},
                )
                return response.choices[0].message.content or ""
            except (InternalServerError, APITimeoutError, APIConnectionError) as e:
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


def _invalid_paths(plan: list) -> list[str]:
    """한글/공백/괄호 등 REST 경로에 쓸 수 없는 문자가 섞인 path 목록 (EXAONE이 기능명을
    그대로 슬러그로 써버리는 경우 방어)."""
    bad = []
    for ep in plan:
        path = ep.get("path") if isinstance(ep, dict) else None
        if not isinstance(path, str) or not path or not _VALID_PATH_RE.match(path):
            bad.append(path)
    return bad


def _sanitize_plan_paths(plan: list) -> list:
    """유효하지 않은 path를 ASCII 슬러그로 강제 변환 (재생성 3회 실패 시 최후 수단)."""
    seen: set[tuple[str, str]] = set()
    for i, ep in enumerate(plan):
        if not isinstance(ep, dict):
            continue
        path = ep.get("path") or ""
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
_SPEC_FIELD_MAX = 400  # 정상 스펙 한 필드의 상한 (초과 시 서술/추론 유출로 간주)


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
        ep.setdefault("authRequired", True)
        if not str(ep.get("description") or "").strip():
            ep["description"] = f"{ep.get('method')} {ep.get('path')}"
        if not str(ep.get("requestBody") or "").strip():
            ep["requestBody"] = "없음"
        if not str(ep.get("successResponse") or "").strip():
            ep["successResponse"] = "success: boolean"
        if not str(ep.get("errorCodes") or "").strip():
            ep["errorCodes"] = "401 — 인증 실패, 500 — 서버 오류"
        out.append(ep)
    return out
