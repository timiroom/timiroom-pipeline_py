import asyncio
import logging

from openai import AsyncOpenAI, InternalServerError, APITimeoutError, APIConnectionError

from common.pm_skills import PmSkillsLoader
from phase2.json_utils import try_parse_json, has_suspicious_script
from phase2.state import PipelineState

logger = logging.getLogger(__name__)

_MAX_FEATURE_NAME_LEN = 60
_MIN_MEANINGFUL_CHAR_RATIO = 0.5
_MAX_HYPHEN_SEGMENTS = 4


def _is_plausible_feature(name) -> bool:
    """EXAONE이 토큰 손상으로 뱉는 스크램블된 기능명을 걸러내는 보수적 결정론적 필터.
    (예: '기구-재물-현황-한다그-보(목록-밀동-2 엘드폰)' 같은 실제 관찰된 손상 사례)"""
    if not isinstance(name, str) or not name.strip():
        return False
    stripped = name.strip()
    if len(stripped) > _MAX_FEATURE_NAME_LEN:
        return False
    if has_suspicious_script(stripped):
        return False
    meaningful = sum(1 for ch in stripped if ch.isalnum())
    non_space = sum(1 for ch in stripped if not ch.isspace())
    if non_space and meaningful / non_space < _MIN_MEANINGFUL_CHAR_RATIO:
        return False
    if stripped.count("-") > _MAX_HYPHEN_SEGMENTS:
        return False
    return True

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
            kept = [f for f in feature_list if _is_plausible_feature(f)]
            dropped = [f for f in feature_list if f not in kept]
            if dropped:
                logger.warning("PM featureList — 손상 의심 항목 %d개 제외: %s", len(dropped), dropped)
            feature_list = kept

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
