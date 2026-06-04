import asyncio
import logging

import httpx

from phase2.state import PipelineState

logger = logging.getLogger(__name__)

LABELS = ["시장규모/경쟁사", "Pain Point", "법규", "기술트렌드", "사용자통계"]


class SearchAgent:

    def __init__(self, openai_api_key: str):
        self._api_key = openai_api_key

    async def execute(self, state: PipelineState) -> PipelineState:
        logger.info("Search 에이전트 시작 — 시장 데이터 수집")
        try:
            domain = await self._extract_domain(state.user_query)
            market_data = await self._collect_all(state.user_query, domain)
            logger.info("Search 에이전트 완료 — %d자 수집", len(market_data))
            return state.copy(
                market_research=market_data,
                status_message="Search 에이전트 완료 — 시장 데이터 수집",
            )
        except Exception as e:
            logger.error("Search 에이전트 실패: %s", e)
            return state.copy(
                market_research=f"웹 검색 실패: {e}",
                status_message="Search 에이전트 실패",
            )

    async def _collect_all(self, user_query: str, domain: str) -> str:
        queries = [
            f"""다음을 웹 검색하여 한국 시장 데이터를 수집하고 수치+출처를 정리하세요:
1. "국내 {domain} 시장 규모 2024 조원" 검색
2. "한국 {domain} 연간 성장률 CAGR 2024" 검색
3. "한국 {domain} 플랫폼 서비스 점유율 매출 순위 2023 2024" 검색
4. "{domain} 국내 주요 경쟁 서비스 앱 사용자 수 2024" 검색

형식:
[시장규모] 수치 + 출처(기관명, 날짜)
[성장률] 수치 + 출처
[경쟁사매출] 각 사별 실제 매출액 또는 사용자 수 + 출처
[경쟁사목록] 해당 도메인({domain})의 한국 실제 서비스명 5개 이상 나열
반드시 한국 서비스명만 사용. Shopify, Magento, WooCommerce 등 글로벌 플랫폼 금지.""",

            f"""다음을 웹 검색하여 한국 사용자 불편 데이터를 수집하세요:
1. "{domain} 사용자 불편사항 불만족 설문 통계 2024" 검색
2. "온라인쇼핑 소비자 불만 한국소비자원 KCA 2024" 검색
3. "{domain} 회원가입 로그인 UX 이탈률 문제점 리서치" 검색
4. "모바일 {domain} 사용자 이탈 원인 통계 2024" 검색

형식:
[Pain Point 1~5] 내용 + 퍼센트 수치 + 출처
반드시 한국 기관 출처만 사용.""",

            """다음을 웹 검색하여 최신 법규 정보를 수집하세요:
1. "개인정보보호법 2024 최신 개정 조항 이커머스 전자상거래" 검색
2. "전자상거래법 소비자보호법 2024 개정 주요 내용" 검색
3. "전자금융거래법 PG결제 관련 규정 2024" 검색

형식:
[개인정보보호법] 조항 번호 + 핵심 내용
[전자상거래법] 조항 번호 + 핵심 내용
[전자금융거래법] 조항 번호 + 핵심 내용""",

            f"""다음을 웹 검색하여 기술/성능 데이터를 수집하세요:
1. "{domain} 백엔드 기술 스택 트렌드 2024 Spring Django" 검색
2. "{domain} API 응답속도 성능 벤치마크 업계 평균 2024" 검색
3. "{domain} 동시접속 서버 스케일링 사례 2024" 검색

형식:
[권장기술스택] 각 레이어별 권장 기술 + 선택 이유
[성능벤치마크] 업계 평균 응답속도, 동시접속 수치 + 출처
[아키텍처] 업계 표준 아키텍처 패턴""",

            f"""다음을 웹 검색하여 한국 사용자/비즈니스 데이터를 수집하세요:
1. "한국 {domain} 연령별 이용률 통계 2024" 검색
2. "{domain} 구매 전환율 업계 평균 벤치마크 한국 2024" 검색
3. "{domain} 재구매율 고객 유지율 통계 한국 2024" 검색
4. "모바일 {domain} 비율 PC 대비 2024" 검색

형식:
[사용자통계] 연령별 이용률 + 출처
[전환율] 업계 평균 수치 + 출처
[재구매율] 수치 + 출처
[모바일비중] 수치 + 출처""",
        ]

        tasks = [self._search_web(q) for q in queries]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        parts = []
        for i, result in enumerate(results):
            text = result if isinstance(result, str) else f"검색 실패: {result}"
            parts.append(f"=== {LABELS[i]} ===\n{text}")
        return "\n\n".join(parts)

    async def _search_web(self, query: str) -> str:
        body = {
            "model": "gpt-4o",
            "tools": [{"type": "web_search_preview"}],
            "max_output_tokens": 2000,
            "input": [{"role": "user", "content": query}],
        }
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(
                "https://api.openai.com/v1/responses",
                headers={"Authorization": f"Bearer {self._api_key}"},
                json=body,
            )
            resp.raise_for_status()
            data = resp.json()

        parts = []
        for item in data.get("output", []):
            if item.get("type") == "message":
                for block in item.get("content", []):
                    if block.get("type") == "output_text":
                        parts.append(block.get("text", ""))
        return "".join(parts) or "검색 실패"

    async def _extract_domain(self, user_query: str) -> str:
        body = {
            "model": "gpt-4o-mini",
            "max_tokens": 50,
            "temperature": 0,
            "messages": [
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
        }
        try:
            async with httpx.AsyncClient(timeout=15) as client:
                resp = await client.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers={"Authorization": f"Bearer {self._api_key}"},
                    json=body,
                )
                data = resp.json()
            domain = data["choices"][0]["message"]["content"].strip()
            logger.info("도메인 추출: [%s]", domain)
            return domain
        except Exception as e:
            logger.warning("도메인 추출 실패: %s", e)
            return user_query
