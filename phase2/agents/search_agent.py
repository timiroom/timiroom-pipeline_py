import asyncio
import logging
from datetime import UTC, datetime

from openai import AsyncOpenAI

from phase2.llm_runtime import LlmRuntime
from phase2.state import PipelineState

logger = logging.getLogger(__name__)

LABELS = ["시장규모/경쟁사", "Pain Point", "법규", "기술트렌드", "사용자통계"]

# 생성 샘플링 파라미터
_TEMPERATURE = 1.0
_TOP_P = 0.95
_PRESENCE_PENALTY = 0.0


class SearchAgent:

    def __init__(
        self,
        client: AsyncOpenAI,
        model: str,
        runtime: LlmRuntime | None = None,
        web_search_enabled: bool = True,
    ):
        self._client = client
        self._model = model
        self._runtime = runtime
        self._web_search_enabled = web_search_enabled

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
            f"""한국 {domain} 시장을 웹에서 조사해 아래 형식으로 정리하세요.

[시장규모] 수치 + 출처(기관명, 연도)
[성장률] 수치 + 출처
[경쟁사매출] 각 사별 실제 매출액 또는 사용자 수 + 출처
[경쟁사목록] 해당 도메인({domain})의 한국 실제 서비스명 5개 이상 나열
반드시 한국 서비스명만 사용. Shopify, Magento, WooCommerce 등 글로벌 플랫폼 금지.

서비스 요구사항: {user_query}""",

            f"""한국 {domain} 서비스 사용자들의 불편 데이터를 웹에서 조사해 정리하세요.

[Pain Point 1~5] 내용 + 퍼센트 수치 + 출처
반드시 한국 기관 출처만 사용.

서비스 요구사항: {user_query}""",

            f"""한국 {domain} 서비스에 적용되는 최신 법규를 웹에서 조사해 정리하세요.

[개인정보보호법] 조항 번호 + 핵심 내용
[전자상거래법] 조항 번호 + 핵심 내용
[전자금융거래법] 조항 번호 + 핵심 내용

서비스 요구사항: {user_query}""",

            f"""한국 {domain} 서비스의 기술 트렌드를 웹에서 조사해 정리하세요.

[권장기술스택] 각 레이어별 권장 기술 + 선택 이유
[성능벤치마크] 업계 평균 응답속도, 동시접속 수치 + 출처
[아키텍처] 업계 표준 아키텍처 패턴

서비스 요구사항: {user_query}""",

            f"""한국 {domain} 서비스의 사용자/비즈니스 데이터를 웹에서 조사해 정리하세요.

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
        if self._web_search_enabled:
            async def web_request():
                return await self._client.responses.create(
                    model=self._model,
                    instructions=(
                        "웹 검색 결과에 근거해서만 답하세요. 수치·법률·시장 정보에는 반드시 "
                        "출처명, 문서 연도와 URL 인용을 붙이고 확인되지 않은 내용은 '확인 불가'로 표시하세요."
                    ),
                    input=f"기준일: {datetime.now(UTC).date().isoformat()}\n\n{prompt}",
                    tools=[{"type": "web_search"}],
                    tool_choice="auto",
                    include=["web_search_call.action.sources"],
                    max_output_tokens=2000,
                )

            response = await self._runtime.call(web_request) if self._runtime else await web_request()
            text = (response.output_text or "").strip()
            sources = self._extract_sources(response)
            if sources:
                source_lines = "\n".join(f"- {title}: {url}" for title, url in sources)
                text = f"{text}\n\n[검색 출처]\n{source_lines}"
            return text

        async def chat_request():
            return await self._client.chat.completions.create(
                model=self._model,
                max_completion_tokens=2000,
                temperature=_TEMPERATURE,
                top_p=_TOP_P,
                presence_penalty=_PRESENCE_PENALTY,
                messages=[{"role": "user", "content": prompt}],
            )

        response = await self._runtime.call(chat_request) if self._runtime else await chat_request()
        return (response.choices[0].message.content or "").strip()

    @staticmethod
    def _extract_sources(response) -> list[tuple[str, str]]:
        """Responses API의 web_search source metadata를 문자열 산출물에도 보존한다."""
        if not hasattr(response, "model_dump"):
            return []
        payload = response.model_dump()
        found: list[tuple[str, str]] = []
        seen: set[str] = set()

        def visit(node):
            if isinstance(node, dict):
                url = node.get("url")
                if isinstance(url, str) and url.startswith(("https://", "http://")) and url not in seen:
                    seen.add(url)
                    found.append((str(node.get("title") or node.get("name") or "출처"), url))
                for value in node.values():
                    visit(value)
            elif isinstance(node, list):
                for value in node:
                    visit(value)

        visit(payload.get("output", []))
        return found[:20]

    async def _extract_domain(self, user_query: str) -> str:
        try:
            async def request():
                return await self._client.chat.completions.create(
                    model=self._model,
                    max_completion_tokens=100,
                    temperature=_TEMPERATURE,
                    top_p=_TOP_P,
                    presence_penalty=_PRESENCE_PENALTY,
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
                )

            resp = await self._runtime.call(request) if self._runtime else await request()
            domain = (resp.choices[0].message.content or "").strip()
            logger.info("도메인 추출: [%s]", domain)
            return domain
        except Exception as e:
            logger.warning("도메인 추출 실패: %s", e)
            return user_query
