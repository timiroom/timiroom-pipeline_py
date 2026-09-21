import asyncio
import operator
import json
import logging
import re
from typing import Annotated, TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.types import Send
from openai import AsyncOpenAI, InternalServerError, APITimeoutError, APIConnectionError

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
_PLAN_BATCH_SIZE = 999
_ENDPOINTS_PER_FEATURE = 1
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
새 엔드포인트가 정말 필요한 경우에만 추가하세요.
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
  "errorCodes": "401 — 인증 실패, 404 — 리소스 없음"
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
        source_registry = normalize_feature_registry(
            state.feature_registry, state.feature_list, preserve_extra=True,
        )
        registry_features = [str(item.get("name")) for item in source_registry if item.get("name")]
        scoped_features = backend_features(registry_features or state.feature_list)
        feature_str = "- " + "\n- ".join(scoped_features) if scoped_features else "(API 대상 기능 없음)"
        registry = [item for item in source_registry if item.get("name") in scoped_features]

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
        if isinstance(parsed_spec, dict) and isinstance(parsed_spec.get("endpoints"), list):
            endpoints = _normalize_endpoints(parsed_spec["endpoints"])
            endpoints = _ensure_contract_endpoints(endpoints, registry)
            endpoints = _annotate_feature_ids(endpoints, registry)
            endpoints = [ep for ep in endpoints if isinstance(ep, dict) and str(ep.get("featureId") or "").strip()]
            parsed_spec["endpoints"] = _normalize_endpoints(endpoints)
            api_spec = json.dumps(parsed_spec, ensure_ascii=False)

        logger.info("API 에이전트 완료")
        return state.copy(
            api_spec=api_spec,
            prd_feedback_from_api=prd_issues,
            generation_blockers=(
                [*state.generation_blockers, blocker] if blocker else state.generation_blockers
            ),
            qa_api_blockers=(
                [*state.qa_api_blockers, blocker] if blocker else state.qa_api_blockers
            ),
            qa_approved=False if blocker else state.qa_approved,
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
                by_key[key] = {**by_key[key], **patch} if key in by_key else patch
                applied += 1

            if not applied:
                logger.warning("API targeted repair — 적용 가능한 endpoint 없음, 원본 유지")
                return state
            registry = normalize_feature_registry(
                state.feature_registry, state.feature_list, preserve_extra=True,
            )
            endpoints = _normalize_endpoints(list(by_key.values()))
            endpoints = _annotate_feature_ids(endpoints, registry)
            repaired = {**current, "endpoints": _normalize_endpoints(endpoints)}
            logger.info("API targeted repair — %d개 endpoint만 패치", applied)
            return state.copy(
                api_spec=json.dumps(repaired, ensure_ascii=False),
                prd_feedback_from_api="",
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
        for round_ in range(0):
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
        return {
            "api_spec": json.dumps(
                {"endpoints": endpoints, "authentication": authentication},
                ensure_ascii=False,
            ),
            "prd_issues": "",
        }

        missing = uncovered_features(ctx["feature_list"], [ep.get("description", "") for ep in endpoints if isinstance(ep, dict)])
        if missing:
            logger.warning("API manager_review — 커버리지 부족 감지: %s", missing)

        prompt = MANAGER_REVIEW_PROMPT.format(
            endpoints_json=json.dumps(endpoints, ensure_ascii=False),
            feature_str=ctx["feature_str"],
            context=ctx["context"],
            missing_note=missing_features_note(missing, "API 스펙"),
            feature_registry=ctx.get("feature_registry", "[]"),
        )

        try:
            review = None
            for parse_attempt in range(2):
                raw = await self._call(prompt, max_tokens=16384, system=MANAGER_SYSTEM)
                if dump:
                    dump.log_raw("API_MANAGER_REVIEW", parse_attempt + 1, raw)
                review = try_parse_json(raw)
                if review and isinstance(review, dict):
                    break
                logger.warning("API manager 리뷰 파싱 실패 (시도 %d) — 재시도", parse_attempt + 1)
            if review is None or not isinstance(review, dict):
                logger.warning("API manager 리뷰 파싱 최종 실패 — 원본 엔드포인트 유지")
            else:
                prd_issues = str(review.get("prdIssues", "") or "").strip()
                patches = review.get("patches")
                if isinstance(patches, dict) and patches:
                    valid_patches = {k: v for k, v in patches.items() if isinstance(v, dict)}
                    dropped = set(patches) - set(valid_patches)
                    if dropped:
                        logger.warning("API manager 리뷰 — 패치 값이 dict가 아니어서 무시: %s", list(dropped))
                    if valid_patches:
                        logger.info("API manager 리뷰 — %d개 엔드포인트 패치: %s", len(valid_patches), list(valid_patches.keys()))
                        by_key = {
                            (str(ep.get("method", "GET")).upper(), _canonical_api_path(ep.get("path"))): ep
                            for ep in endpoints if isinstance(ep, dict)
                        }
                        for patch_key, patch in valid_patches.items():
                            method, _, raw_path = str(patch_key).partition(" ")
                            method = str(patch.get("method") or method or "GET").upper()
                            path = _output_api_path(patch.get("path") or raw_path)
                            key = (method, _canonical_api_path(path))
                            existing = by_key.get(key, {})
                            by_key[key] = {**existing, **patch, "method": method, "path": path}
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
        endpoints = _annotate_feature_ids(endpoints, _registry_items(ctx.get("feature_registry", "[]")))
        endpoints = _normalize_endpoints(endpoints)
        endpoints = _prune_plan(
            endpoints,
            ctx["feature_list"],
            int(ctx.get("max_endpoints") or _endpoint_soft_cap(len(ctx["feature_list"]), include_auth=True)),
        )
        # Registry 계약 endpoint는 soft cap 이후에도 보존한다. cap은 파생 endpoint를
        # 줄이기 위한 것이며 PM이 명시한 최소 계약을 삭제하는 용도가 아니다.
        endpoints = _ensure_contract_endpoints(
            endpoints, _registry_items(ctx.get("feature_registry", "[]"))
        )
        # 계약 endpoint를 cap/patch 뒤에 다시 매핑한다. 새로 주입된 endpoint와
        # manager patch가 featureId를 잃어 QA에서 추적 불가능해지는 것을 방지한다.
        endpoints = _annotate_feature_ids(
            endpoints, _registry_items(ctx.get("feature_registry", "[]"))
        )
        endpoints = _normalize_endpoints(endpoints)

        api_spec = json.dumps({"endpoints": endpoints, "authentication": authentication}, ensure_ascii=False)
        return {"api_spec": api_spec, "prd_issues": prd_issues}

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
        if not isinstance(path, str) or not path or not _VALID_PATH_RE.match(path):
            bad.append(path)
    return bad


def _canonical_api_path(path: str) -> str:
    """Registry 계약(/tools)과 실제 API 경로(/api/v1/tools)를 같은 키로 비교한다."""
    value = "/" + str(path or "").strip().lstrip("/")
    value = re.sub(r"^/api/v1", "", value, flags=re.IGNORECASE) or "/"
    return value.rstrip("/") or "/"


def _output_api_path(path: str) -> str:
    """Return the single public path form used by generated API specs."""
    value = "/" + str(path or "").strip().lstrip("/")
    value = re.sub(r"^/api/v1/api(?:/|$)", "/api/v1/", value, flags=re.IGNORECASE)
    if not re.match(r"^/api/v1(?:/|$)", value, flags=re.IGNORECASE):
        value = "/api/v1/" + value.lstrip("/")
    return re.sub(r"/{2,}", "/", value).rstrip("/") or "/api/v1"


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
            method = str(contract.get("method") or "GET").upper()
            path = _canonical_api_path(contract.get("path"))
            if path:
                contracts[(method, path)] = feature_id
                route_prefixes.append((path.rsplit("/", 1)[0] or "/", feature_id))
    for ep in plan:
        if not isinstance(ep, dict):
            continue
        ep["path"] = _output_api_path(ep.get("path"))
        key = (str(ep.get("method") or "GET").upper(), _canonical_api_path(ep.get("path")))
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
        if not current and len(valid_ids) == 1:
            current = next(iter(valid_ids))
        if current:
            ep["featureId"] = current
        else:
            # Never assign an unrelated feature just to silence QA. An orphan
            # endpoint must remain visible as a mapping blocker for repair.
            ep.pop("featureId", None)
    return plan


def _registry_items(raw: str | list[dict] | None) -> list[dict]:
    if isinstance(raw, list):
        return [item for item in raw if isinstance(item, dict)]
    parsed = try_parse_json(raw or "[]")
    return parsed if isinstance(parsed, list) else []


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
        (str(ep.get("method", "GET")).upper(), _canonical_api_path(ep.get("path"))): ep
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
            method = str(contract.get("method", "GET")).upper()
            path = str(contract.get("path", "")).strip()
            if not path:
                continue
            path = _output_api_path(path)
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
        path = str(ep.get("path") or "").strip()
        path = re.sub(r"^/api/v1/api(?:/|$)", "/api/v1/", path, flags=re.I)
        if re.match(r"^/api/(?!v1/)", path, flags=re.I):
            path = "/api/v1/" + path[5:]
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
