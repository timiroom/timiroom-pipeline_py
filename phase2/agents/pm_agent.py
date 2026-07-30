import asyncio
import logging
import re

from openai import AsyncOpenAI, InternalServerError, APITimeoutError, APIConnectionError

from common.pm_skills import PmSkillsLoader
from phase2.json_utils import try_parse_json, has_suspicious_script
from phase2.state import PipelineState

logger = logging.getLogger(__name__)

# 길이는 손상 신호로는 약하다 — 아래 패턴 검사가 스크램블을 직접 잡으므로, 정상적인 긴
# 기능명(괄호 안 상세 설명 포함)이 잘려 DBA·API가 참고할 정보를 잃지 않도록 여유를 둔다.
# (실측: '보유 재료 기반 레시피 추천 (AI 기반 재료 조합 분석, 조리 시간, 난이도, ...)' 62자가
#  멀쩡한데도 부연이 통째로 잘렸다)
_MAX_FEATURE_NAME_LEN = 100
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

# EXAONE 모델 카드 권장 샘플링 파라미터
# https://huggingface.co/LGAI-EXAONE/K-EXAONE-236B-A23B
_TEMPERATURE = 1.0
_TOP_P = 0.95
_PRESENCE_PENALTY = 0.0

PM_PROMPT = """당신은 시니어 소프트웨어 아키텍트이자 PM입니다.
아래 요구사항을 분석하여 JSON 형식으로만 응답하세요.
{skills_section}
응답 형식:
{{
  "featureList": ["기능명 (구체적으로 — 예: 소셜 로그인(카카오/구글), 상품 목록 조회 및 필터링, 장바구니 추가/삭제)"],
  "dbaInstruction": "필요한 테이블 목록과 주요 관계를 명시 (예: users, products, orders, order_items 테이블 필요. users-orders 1:N, orders-products N:M. 사용자 인증에 refresh_token 컬럼 필요)",
  "apiInstruction": "필요한 API 그룹과 인증 방식 명시 (예: 인증 API(로그인/회원가입/토큰갱신), 상품 API(목록조회/상세/검색), 주문 API(생성/조회/취소). JWT Bearer 토큰 인증. 관리자 권한 체크 필요)"
}}

규칙:
- featureList: 서비스의 모든 핵심·부가 기능을 빠짐없이 열거 (최소 5개 이상, 기능명은 구체적으로)
- dbaInstruction: 필요한 테이블명, 주요 관계(1:N, N:M), 중요 컬럼이나 제약조건 명시 (2문장 이상)
- apiInstruction: 기능별 API 그룹, 인증 방식, 특별한 설계 요구사항 명시 (2문장 이상)
- JSON 외 다른 텍스트는 절대 포함하지 마세요

요구사항:
{context}
{form_features_hint}"""


class PmAgent:

    def __init__(self, client: AsyncOpenAI, skills_loader: PmSkillsLoader, model: str = "gpt-4o"):
        self._client = client
        self._skills = skills_loader
        self._model = model

    async def execute(self, state: PipelineState, dump=None) -> PipelineState:
        logger.info("PM 에이전트 시작")

        if self._skills.has_skills():
            skills_section = await self._skills.find_relevant_skills(state.context_prompt, 5)
            logger.info("PM 스킬 주입 완료")
        else:
            skills_section = ""
            logger.warning("PM 스킬 미적용")

        # 폼에서 추출된 기능 목록을 프롬프트에 명시하여 PM 에이전트가 빠뜨리지 않도록 힌트 제공
        form_features = state.feature_list or []
        form_features_hint = (
            "\n\n[폼에서 입력된 기능 목록 — 반드시 포함하고 더 상세히 확장하세요]\n"
            + "\n".join(f"- {f}" for f in form_features)
        ) if form_features else ""

        prompt = PM_PROMPT.format(
            skills_section=skills_section,
            context=state.context_prompt,
            form_features_hint=form_features_hint,
        )

        data = None
        for attempt in range(3):
            try:
                response = await self._client.chat.completions.create(
                    model=self._model,
                    temperature=_TEMPERATURE,
                    top_p=_TOP_P,
                    presence_penalty=_PRESENCE_PENALTY,
                    frequency_penalty=0.3,
                    messages=[
                        {"role": "system", "content": "JSON만 출력하세요. 설명·인사말·마크다운 코드블록 금지. { 로 시작해서 } 로 끝납니다."},
                        {"role": "user", "content": prompt},
                    ],
                    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                )
            except (InternalServerError, APITimeoutError, APIConnectionError) as e:
                logger.warning("PM API 일시 오류 (attempt %d): %s — 재시도", attempt + 1, e)
                if attempt < 2:
                    await asyncio.sleep(5 * (attempt + 1))
                    continue
                break
            raw = response.choices[0].message.content or ""
            if dump:
                dump.log_raw("PM", attempt + 1, raw)
            data = try_parse_json(raw)
            if data and isinstance(data, dict) and data.get("featureList"):
                break
            logger.warning("PM 파싱 실패 (attempt %d) — raw:\n%s", attempt + 1, raw)

        feature_list, dba_instruction, api_instruction = self._extract(data)

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

        logger.info("PM 에이전트 완료 — %d 기능 도출", len(feature_list))
        return state.copy(
            feature_list=feature_list,
            dba_instruction=dba_instruction or "DB 설계 진행",
            api_instruction=api_instruction or "REST API 설계 진행",
            status_message="PM 에이전트 완료 — 기능 목록 도출",
        )

    def _extract(self, data) -> tuple[list[str], str, str]:
        if data and isinstance(data, dict):
            return (
                data.get("featureList", []),
                data.get("dbaInstruction", ""),
                data.get("apiInstruction", ""),
            )
        return ([], "", "")
