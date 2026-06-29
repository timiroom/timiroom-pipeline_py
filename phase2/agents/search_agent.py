import asyncio
import logging

from openai import AsyncOpenAI

from phase2.state import PipelineState

logger = logging.getLogger(__name__)

LABELS = ["시장규모/경쟁사", "Pain Point", "법규", "기술트렌드", "사용자통계"]


class SearchAgent:

    def __init__(self, client: AsyncOpenAI, model: str):
        self._client = client
        self._model = model

    async def execute(self, state: PipelineState, dump=None) -> PipelineState:
        logger.info("Search 에이전트 시작 — 시장 데이터 수집")
        try:
            domain = await self._extract_domain(state.user_query)
            market_data = await self._collect_all(state.user_query, domain)
            logger.info("Search 에이전트 완료 — %d자 수집", len(market_data))
            result = state.copy(
                market_research=market_data,
                status_message="Search 에이전트 완료 — 시장 데이터 수집",
            )
            if dump:
                dump.log_raw("SEARCH", 1, market_data)
            return result
        except Exception as e:
            logger.error("Search 에이전트 실패: %s", e)
            return state.copy(
                market_research=f"시장 데이터 수집 실패: {e}",
                status_message="Search 에이전트 실패",
            )

    async def _collect_all(self, user_query: str, domain: str) -> str:
        queries = [
            f"""한국 {domain} 시장에 대해 알고 있는 정보를 바탕으로 아래 형식으로 정리하세요.

[시장규모] 수치 + 출처(기관명, 연도)
[성장률] 수치 + 출처
[경쟁사매출] 각 사별 실제 매출액 또는 사용자 수 + 출처
[경쟁사목록] 해당 도메인({domain})의 한국 실제 서비스명 5개 이상 나열
반드시 한국 서비스명만 사용. Shopify, Magento, WooCommerce 등 글로벌 플랫폼 금지.

서비스 요구사항: {user_query}""",

            f"""한국 {domain} 서비스 사용자들의 불편 데이터를 알고 있는 정보를 바탕으로 정리하세요.

[Pain Point 1~5] 내용 + 퍼센트 수치 + 출처
반드시 한국 기관 출처만 사용.

서비스 요구사항: {user_query}""",

            f"""한국 {domain} 서비스에 적용되는 법규를 알고 있는 정보를 바탕으로 정리하세요.

[개인정보보호법] 조항 번호 + 핵심 내용
[전자상거래법] 조항 번호 + 핵심 내용
[전자금융거래법] 조항 번호 + 핵심 내용

서비스 요구사항: {user_query}""",

            f"""한국 {domain} 서비스의 기술 트렌드를 알고 있는 정보를 바탕으로 정리하세요.

[권장기술스택] 각 레이어별 권장 기술 + 선택 이유
[성능벤치마크] 업계 평균 응답속도, 동시접속 수치 + 출처
[아키텍처] 업계 표준 아키텍처 패턴

서비스 요구사항: {user_query}""",

            f"""한국 {domain} 서비스의 사용자/비즈니스 데이터를 알고 있는 정보를 바탕으로 정리하세요.

[사용자통계] 연령별 이용률 + 출처
[전환율] 업계 평균 수치 + 출처
[재구매율] 수치 + 출처
[모바일비중] 수치 + 출처

서비스 요구사항: {user_query}""",
        ]

        tasks = [self._query(q) for q in queries]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        parts = []
        for i, result in enumerate(results):
            text = result if isinstance(result, str) else f"수집 실패: {result}"
            parts.append(f"=== {LABELS[i]} ===\n{text}")
        return "\n\n".join(parts)

    async def _query(self, prompt: str) -> str:
        resp = await self._client.chat.completions.create(
            model=self._model,
            max_tokens=2000,
            temperature=0.2,
            messages=[{"role": "user", "content": prompt}],
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        return (resp.choices[0].message.content or "").strip()

    async def _extract_domain(self, user_query: str) -> str:
        try:
            resp = await self._client.chat.completions.create(
                model=self._model,
                max_tokens=100,
                temperature=0,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "사용자의 서비스 설명을 읽고 해당 서비스의 업종/도메인을 한국어 2~4단어로만 답하세요. "
                            "예시) 이커머스 전자상거래, 음식 배달, 숙박 예약, 의료 헬스케어, "
                            "부동산 중개, 방탈출 예약, 피트니스 헬스, 반려동물 케어 "
                            "다른 설명 없이 도메인 단어만 출력하세요."
                        ),
                    },
                    {"role": "user", "content": user_query},
                ],
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
            domain = (resp.choices[0].message.content or "").strip()
            logger.info("도메인 추출: [%s]", domain)
            return domain
        except Exception as e:
            logger.warning("도메인 추출 실패: %s", e)
            return user_query
