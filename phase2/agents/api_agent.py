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
    "JWT Bearer 토큰 방식. 로그인 시 발급받은 accessToken을 "
    "Authorization: Bearer <token> 헤더로 전송한다."
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

        # 배치들이 서로 모르는 채 같은 리소스를 설계해 병합 시 중복 제거로 개수가 깎인다.
        # 흔적이 없는 기능(uncovered)뿐 아니라 엔드포인트가 부족한 기능(undercovered)까지
        # 모아, 이미 설계된 경로를 알려주고 '없는 것만' 추가로 받아낸다. 최대 2라운드.
        for round_ in range(2):
            descriptions = [ep.get("description", "") for ep in plan]
            missing = uncovered_features(feature_list, descriptions)
            thin = undercovered_features(feature_list, descriptions, _ENDPOINTS_PER_FEATURE)
            need = missing + [f for f in thin if f not in missing]
            if not need or len(plan) >= min_endpoints:
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
            if len(merged) == len(plan):
                logger.warning("API manager_plan 보충 %d라운드 — 새 엔드포인트 없음, 중단", round_ + 1)
                break
            plan = merged

        if not plan:
            logger.error("API manager_plan 최종 실패 — feature_list 기반 fallback 스켈레톤 사용")
            plan = self._fallback_plan(feature_list)["plan"]
        elif _invalid_paths(plan):
            logger.warning("API manager_plan — 경로 형식 오류가 남아있어 강제 살균 적용")
            plan = _sanitize_plan_paths(plan)

        if len(plan) < min_endpoints:
            logger.warning("API manager_plan — 엔드포인트 %d/%d개로 목표 미달", len(plan), min_endpoints)

        logger.info("API manager_plan 완료 — 엔드포인트 %d개 계획 (목표 %d개)", len(plan), min_endpoints)
        return {"plan": plan, "authentication": _AUTHENTICATION_DESC}

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
