import asyncio
import logging

from openai import AsyncOpenAI, InternalServerError, APITimeoutError, APIConnectionError

from common.pm_skills import PmSkillsLoader
from phase2.json_utils import try_parse_json
from phase2.state import PipelineState

logger = logging.getLogger(__name__)

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
"""


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

        prompt = PM_PROMPT.format(
            skills_section=skills_section,
            context=state.context_prompt,
        )

        data = None
        for attempt in range(3):
            try:
                response = await self._client.chat.completions.create(
                    model=self._model,
                    temperature=0.1,
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

        # 파싱 완전 실패 시 기존 state의 feature_list 유지
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
