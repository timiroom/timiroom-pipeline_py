import asyncio
import logging
import re

from openai import AsyncOpenAI, InternalServerError, APITimeoutError, APIConnectionError

from phase2.json_utils import try_parse_json, has_suspicious_script
from phase2.feature_registry import build_feature_registry, normalize_feature_registry
from phase2.llm_runtime import LlmRuntime
from phase2.state import PipelineState

logger = logging.getLogger(__name__)

TARGETED_REPAIR_PROMPT = """현재 PM 결과에서 검증 피드백이 지적한 기능 항목만 수정하세요.
전체 featureList를 다시 작성하지 마세요. patches에는 replace/remove/append 작업만 넣으세요.
index는 아래 현재 기능 목록의 0부터 시작하는 인덱스입니다.
문제없는 기능은 절대 변경하지 마세요.
JSON만 출력하세요.

[검증 피드백]
{feedback}

[현재 기능 목록]
{features}

출력 형식:
{{"patches":[{{"op":"replace","index":0,"value":"수정된 기능명"}}],"dbaInstruction":null,"apiInstruction":null}}
"""

# 길이는 손상 신호로는 약하다 — 아래 패턴 검사가 스크램블을 직접 잡으므로, 정상적인 긴
# 기능명(괄호 안 상세 설명 포함)이 잘려 DBA·API가 참고할 정보를 잃지 않도록 여유를 둔다.
# (실측: '보유 재료 기반 레시피 추천 (AI 기반 재료 조합 분석, 조리 시간, 난이도, ...)' 62자가
#  멀쩡한데도 부연이 통째로 잘렸다)
_MAX_FEATURE_NAME_LEN = 100
_MIN_FEATURE_COUNT = 5
_MAX_FEATURE_COUNT = 15
_MIN_MEANINGFUL_CHAR_RATIO = 0.5
_MAX_HYPHEN_SEGMENTS = 4
_MAX_HYPHEN_CHAIN = 3
_MIN_TRIMMED_NAME_LEN = 4

_HANGUL = "가-힣"
# EXAONE 토큰 손상 패턴 (전부 실측 사례)
# - '푸0일 전 푸시 알림'  : 한글 낱말 사이에 숫자가 끼어듦. '1일 전', '20-30대'처럼
#                          숫자가 낱말 앞·뒤에 오는 정상 표기는 걸리지 않는다.
_DIGIT_IN_HANGUL_RE = re.compile(rf"[{_HANGUL}]\d+[{_HANGUL}]")
# - '누Guest 재료 기반'   : 한글 바로 뒤에 영문이 붙음. 반대 방향(영문 뒤 한글 조사,
#                          예: 'API를', 'OCR로')은 정상 한국어 표기라 검사하지 않는다.
_HANGUL_THEN_LATIN_RE = re.compile(rf"[{_HANGUL}][A-Za-z]")
# - '_recipe_auto_suggest_via_ingredients_' : 내부 식별자가 기능명 자리로 유출
_IDENTIFIER_LIKE_RE = re.compile(r"^[A-Za-z0-9]*_[A-Za-z0-9_]*$")


def _corruption_reason(name: str) -> str | None:
    """손상으로 판단되는 이유. 정상이면 None."""
    if len(name) > _MAX_FEATURE_NAME_LEN:
        return "길이 초과"
    if has_suspicious_script(name):
        return "스크립트 오염"
    meaningful = sum(1 for ch in name if ch.isalnum())
    non_space = sum(1 for ch in name if not ch.isspace())
    if non_space and meaningful / non_space < _MIN_MEANINGFUL_CHAR_RATIO:
        return "기호 비율 과다"
    if name.count("-") > _MAX_HYPHEN_SEGMENTS:
        return "하이픈 과다"
    # 공백 없이 하이픈으로만 이어붙인 이름은 정상 한국어 기능명이 아니다
    # (실측: '기구-재물-현황-한다그-보'). 괄호 부연을 떼어낸 뒤에도 이 검사가 걸리므로
    # 통짜로 스크램블된 이름이 '복구됨'으로 살아남지 않는다.
    if name.count("-") >= _MAX_HYPHEN_CHAIN and " " not in name:
        return "하이픈 연결 나열"
    if _IDENTIFIER_LIKE_RE.match(name):
        return "식별자 유출"
    if _DIGIT_IN_HANGUL_RE.search(name):
        return "한글 사이 숫자 혼입"
    if _HANGUL_THEN_LATIN_RE.search(name):
        return "한글 뒤 영문 혼입"
    return None


def _is_plausible_feature(name) -> bool:
    """EXAONE이 토큰 손상으로 뱉는 스크램블된 기능명을 걸러내는 보수적 결정론적 필터.
    (예: '기구-재물-현황-한다그-보(목록-밀동-2 엘드폰)' 같은 실제 관찰된 손상 사례)"""
    if not isinstance(name, str) or not name.strip():
        return False
    return _corruption_reason(name.strip()) is None


def repair_feature_name(name) -> str | None:
    """손상된 기능명을 살릴 수 있으면 살리고, 못 살리면 None.

    손상은 주로 괄호 안 부연 설명에서 발생한다
    (실측: '유통기한 등록 및 임박 알림 (푸0일 전 푸시 알림, 위젯 노출)').
    이럴 때 기능 자체를 버리면 그 기능이 PRD·DB·API에서 통째로 사라지므로,
    괄호 앞부분이 멀쩡하면 부연만 떼어내고 기능은 유지한다."""
    if not isinstance(name, str) or not name.strip():
        return None
    stripped = name.strip()
    reason = _corruption_reason(stripped)
    if reason is None:
        return stripped

    head = stripped.split("(", 1)[0].strip(" ,-–—")
    if len(head) >= _MIN_TRIMMED_NAME_LEN and _corruption_reason(head) is None:
        logger.warning("PM 기능명 부연 손상(%s) — 괄호 이후 제거: %r → %r", reason, stripped, head)
        return head

    logger.warning("PM 기능명 손상(%s) — 제외: %r", reason, stripped)
    return None

# 생성 샘플링 파라미터
_TEMPERATURE = 1.0
_TOP_P = 0.95
_PRESENCE_PENALTY = 0.0

PM_PROMPT = """당신은 시니어 소프트웨어 아키텍트이자 PM입니다.
아래 요구사항을 분석하여 JSON 형식으로만 응답하세요.
응답 형식:
{{
  "projectPlan": {{"overview":"프로젝트 목적", "goals":[], "scope":{{"in":[],"out":[]}}, "targetUsers":[], "priorities":{{"must":[],"should":[],"could":[]}}}},
  "dbaInstruction": "PRD와 Feature Spec이 구현 계약을 만들 때 참고할 데이터 설계 원칙",
  "apiInstruction": "PRD와 Feature Spec이 구현 계약을 만들 때 참고할 API 설계 원칙"
}}

규칙:
- projectPlan: Phase 1 자료에 근거해 프로젝트 목표·범위·타겟 사용자·우선순위를 구조화하세요. 이후 PRD의 기준 문서입니다.
- projectPlan은 반드시 goals, scope.in, scope.out, targetUsers, priorities.must/should/could/wont를 모두 포함하세요. priorities 항목은 나중에 기능으로 추적할 수 있는 구체적인 사용자 여정 또는 지원 기능명으로 작성하세요.
- priorities.must에는 서비스가 성립하기 위한 핵심 여정만, should/could에는 핵심 여정을 보완하는 supporting 기능만 넣으세요. 요구사항에 실제로 필요한 경우에만 인증·권한·입력검증·상태이력·알림·감사로그를 supporting 후보로 기록하세요.
- priorities.wont 또는 scope.out에 넣은 항목은 PRD coreFeatures와 Feature Spec에서 생성하지 마세요. 같은 의미의 표현을 여러 우선순위 bucket에 중복해서 넣지 마세요.
- 각 goal은 최소 하나의 must 또는 should 항목으로 달성 가능해야 하며, targetUsers의 역할과 scope.in의 업무 흐름이 연결되어야 합니다.
- PM에서는 실행 가능한 featureList나 API/DB registry를 만들지 마세요. Phase 1의 기능 입력은 PRD가 핵심 기능을 도출할 때 참고하는 원자료로만 유지합니다.
- PRD Agent가 projectPlan, 목표, 범위, 사용자 여정, Phase 1 요구사항을 종합해 coreFeatures를 정의하고, Feature Spec Agent가 최종 기능 ID와 계약을 만듭니다.
- dbaInstruction/apiInstruction에는 기능 목록을 재생성하지 말고 범위, 권한 모델, 데이터 보존, API 설계 원칙처럼 후속 에이전트가 참고할 결정만 작성하세요.
- dbaInstruction: 필요한 테이블명, 주요 관계(1:N, N:M), 중요 컬럼이나 제약조건 명시 (2문장 이상)
- dbaInstruction에는 기능별 테이블·컬럼·PK/FK·제약조건 매핑을 포함하고 모든 *_id 참조의 실제 REFERENCES 대상을 명시하세요.
- apiInstruction에는 기능별 HTTP method/path 매핑을 포함하고, 서로 다른 행동을 한 endpoint 설명으로 합치지 마세요.
- 조직·동호회·팀 소속을 전제로 하면 단일 조직인지 다중 조직인지 결정하고, 다중 조직이면 조직과 membership 테이블을 설계 지시에 포함하세요.
- JSON 외 다른 텍스트는 절대 포함하지 마세요

요구사항:
{context}
{form_features_hint}"""


class PmAgent:

    def __init__(
        self,
        client: AsyncOpenAI,
        model: str = "gpt-5.4-mini",
        runtime: LlmRuntime | None = None,
    ):
        self._client = client
        self._model = model
        self._runtime = runtime

    async def execute(self, state: PipelineState, dump=None) -> PipelineState:
        logger.info("PM 에이전트 시작")

        # 폼에서 추출된 기능 목록을 프롬프트에 명시하여 PM 에이전트가 빠뜨리지 않도록 힌트 제공
        form_features = state.feature_list or []
        form_features_hint = (
            "\n\n[폼에서 입력된 기능 목록 — 반드시 포함하고 더 상세히 확장하세요]\n"
            + "\n".join(f"- {f}" for f in form_features)
        ) if form_features else ""

        prompt = PM_PROMPT.format(
            context=state.context_prompt,
            form_features_hint=form_features_hint,
        )

        data = None
        # PM은 기능 계약을 새로 확장하는 최초 계획만 담당한다. 파싱 실패 시 전체
        # prompt를 반복 호출하지 않고 기존 Phase1 feature_list를 보존한다.
        for attempt in range(1):
            try:
                async def request():
                    return await self._client.chat.completions.create(
                        model=self._model,
                        temperature=_TEMPERATURE,
                        top_p=_TOP_P,
                        presence_penalty=_PRESENCE_PENALTY,
                        frequency_penalty=0.3,
                        messages=[
                            {"role": "system", "content": "JSON만 출력하세요. 설명·인사말·마크다운 코드블록 금지. { 로 시작해서 } 로 끝납니다."},
                            {"role": "user", "content": prompt},
                        ],
                    )

                response = await self._runtime.call(request) if self._runtime else await request()
            except (InternalServerError, APITimeoutError, APIConnectionError, TimeoutError) as e:
                logger.warning("PM API 일시 오류 (attempt %d): %s — 재시도", attempt + 1, e)
                if attempt < 2:
                    await asyncio.sleep(5 * (attempt + 1))
                    continue
                break
            raw = response.choices[0].message.content or ""
            if dump:
                dump.log_raw("PM", attempt + 1, raw)
            data = try_parse_json(raw)
            if data and isinstance(data, dict) and data.get("projectPlan"):
                break
            logger.warning("PM 파싱 실패 (attempt %d) — raw:\n%s", attempt + 1, raw)

        _, dba_instruction, api_instruction, _ = self._extract(data)
        # PM은 기능 계약의 소유자가 아니다. Phase 1에서 들어온 기능은 PRD의 참고 입력으로만 유지한다.
        feature_list = list(state.feature_list or [])
        raw_registry = state.feature_registry
        project_plan = data.get("projectPlan", {}) if isinstance(data, dict) else {}
        project_plan = project_plan if isinstance(project_plan, dict) else {}

        if feature_list:
            # 손상된 이름은 먼저 복구를 시도하고(대개 괄호 안 부연만 깨져 있다),
            # 복구 불가능한 것만 제외한다 — 통째로 버리면 그 기능이 산출물 전체에서 사라진다
            repaired, dropped, seen = [], 0, set()
            for f in feature_list:
                name = repair_feature_name(f)
                if name is None:
                    dropped += 1
                    continue
                if name in seen:  # 부연 제거로 앞 항목과 같아질 수 있다
                    continue
                seen.add(name)
                repaired.append(name)
            if dropped or len(repaired) != len(feature_list):
                logger.warning(
                    "PM featureList 정리 — %d개 → %d개 (복구 불가 %d개 제외)",
                    len(feature_list), len(repaired), dropped,
                )
            feature_list = repaired

        # 파싱 완전 실패 또는 전량 필터링 시 기존 state의 feature_list 유지
        if not feature_list:
            feature_list = state.feature_list or []
            logger.warning("PM feature_list 도출 실패 — 기존 feature_list 유지 (%d개)", len(feature_list))

        if len(feature_list) > _MAX_FEATURE_COUNT:
            # 과도한 분해 결과를 그대로 PRD worker에 넘기면 배치가 비어 버리거나
            # 전체 coreFeatures가 누락된다. 사용자 폼 기능을 우선 보존한다.
            fallback = list(state.feature_list or [])
            feature_list = (fallback or feature_list)[:_MAX_FEATURE_COUNT]
            logger.warning(
                "PM featureList 상한 적용 — %d개 → %d개 (과도한 CRUD 분해 방지)",
                len(fallback or feature_list), len(feature_list),
            )

        feature_registry = normalize_feature_registry(raw_registry, feature_list)

        logger.info("PM 에이전트 완료 — %d 기능 도출", len(feature_list))
        return state.copy(
            feature_list=feature_list,
            feature_registry=feature_registry,
            project_plan=project_plan,
            dba_instruction=dba_instruction or "DB 설계 진행",
            api_instruction=api_instruction or "REST API 설계 진행",
            status_message="PM 에이전트 완료 — 기능 목록 도출",
        )

    async def repair(self, state: PipelineState, feedback: str, dump=None) -> PipelineState:
        """현재 기능 목록에 대한 delta 작업만 적용한다."""
        features = list(state.feature_list or [])
        prompt = TARGETED_REPAIR_PROMPT.format(
            feedback=str(feedback or "")[:8000],
            features="\n".join(f"{i}. {feature}" for i, feature in enumerate(features)),
        )
        try:
            async def request():
                return await self._client.chat.completions.create(
                    model=self._model,
                    temperature=_TEMPERATURE,
                    messages=[
                        {"role": "system", "content": "JSON만 출력하세요."},
                        {"role": "user", "content": prompt},
                    ],
                )

            response = await self._runtime.call(request) if self._runtime else await request()
            raw = response.choices[0].message.content or ""
            if dump:
                dump.log_raw("PM_TARGETED_REPAIR", 1, raw)
            data = try_parse_json(raw)
            patches = data.get("patches") if isinstance(data, dict) else None
            if not isinstance(patches, list):
                logger.warning("PM targeted repair — patches 없음, 원본 유지")
                return state

            for patch in patches:
                if not isinstance(patch, dict):
                    continue
                op = patch.get("op")
                index = patch.get("index")
                if op == "replace" and isinstance(index, int) and 0 <= index < len(features):
                    if isinstance(patch.get("value"), str) and patch["value"].strip():
                        features[index] = patch["value"].strip()
                elif op == "remove" and isinstance(index, int) and 0 <= index < len(features):
                    features.pop(index)
                elif op == "append" and isinstance(patch.get("value"), str) and patch["value"].strip():
                    features.append(patch["value"].strip())

            clean = []
            seen = set()
            for feature in features:
                name = repair_feature_name(feature)
                if name and name not in seen:
                    clean.append(name)
                    seen.add(name)
            kwargs = {
                "feature_list": clean,
                "feature_registry": normalize_feature_registry(state.feature_registry, clean),
                "status_message": "PM 에이전트 완료 — 지적 기능만 수정",
            }
            if isinstance(data.get("dbaInstruction"), str) and data["dbaInstruction"].strip():
                kwargs["dba_instruction"] = data["dbaInstruction"].strip()
            if isinstance(data.get("apiInstruction"), str) and data["apiInstruction"].strip():
                kwargs["api_instruction"] = data["apiInstruction"].strip()
            return state.copy(**kwargs)
        except Exception as e:
            logger.warning("PM targeted repair 실패 — 원본 유지: %s", e)
            return state

    def _extract(self, data) -> tuple[list[str], str, str, list[dict]]:
        if data and isinstance(data, dict):
            return (
                data.get("featureList", []),
                data.get("dbaInstruction", ""),
                data.get("apiInstruction", ""),
                data.get("featureRegistry", []),
            )
        return ([], "", "", [])
