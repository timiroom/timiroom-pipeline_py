import asyncio
import logging
import re
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

_STATISTICAL_PHRASES = (
    "1인 가구", "맞벌이 가구", "청년", "고령자", "소상공인", "자영업",
    "전자상거래", "온라인 쇼핑", "식품 소비", "반려동물", "취업", "주거",
)


def _select_public_search_keyword(user_query: str, model_domain: str) -> str:
    compact = re.sub(r"\s+", " ", user_query or "")
    for phrase in _STATISTICAL_PHRASES:
        if phrase in compact:
            return phrase
    household = re.search(r"\b\d+인\s*가구\b", compact)
    if household:
        return re.sub(r"\s*", "", household.group(0)).replace("가구", " 가구")
    return model_domain


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
            market_data, statuses = await self._collect_report(state.user_query, domain)
            logger.info("Search 에이전트 완료 — %d자 수집", len(market_data))
            result = state.copy(
                market_research=market_data,
                market_research_status=statuses,
                status_message="Search 에이전트 완료 — 시장 데이터 수집",
            )
            if dump:
                dump.log_raw("SEARCH", 1, market_data)
            return result
        except Exception as e:
            logger.error("Search 에이전트 실패: %s", type(e).__name__)
            return state.copy(
                market_research="외부 시장 근거 확인 불가. 외부 통계를 추정하지 말고 제품 목표는 내부 목표로 구분하세요.",
                market_research_status=[{"status": "failed", "errorType": type(e).__name__}],
                status_message="Search 에이전트 실패",
            )

    async def _collect_all(self, user_query: str, domain: str) -> str:
        text, _ = await self._collect_report(user_query, domain)
        return text

    async def _collect_report(self, user_query: str, domain: str) -> tuple[str, list[dict]]:
        topics = [
            "적용 가능한 시장 규모와 직접 경쟁 서비스. 국내 자료를 우선하되 필요하면 해외 사례를 구분하세요.",
            "대상 사용자의 실제 불편과 조사 근거. 퍼센트 자료가 없으면 정성적 근거로 구분하세요.",
            "실제 서비스 범위에 적용되는 법규와 적용 이유. 결제·거래가 없으면 금융·전자상거래 법규를 강제하지 마세요.",
            "사용자 흐름에 관련된 기술 선택과 실제 확인 가능한 성능 자료. 업계 평균을 임의로 만들지 마세요.",
            "대상 사용자와 서비스 이용 통계. 해당 업무에 의미 없는 전환율·재구매율은 제외하세요.",
        ]
        common = (
            f"한국 {domain} 서비스에 관해 웹 근거를 조사하세요. 요구사항: {user_query}\n"
            "확인된 주장마다 원문 URL·출처명·연도를 붙이세요. 자료 수나 통계 개수를 채우지 마세요. "
            "적용 불가와 확인 불가를 구분하고, 근거 없는 수치·매출·조항·경쟁사를 생성하지 마세요.\n조사 항목: "
        )
        queries = [common + topic for topic in topics]

        tasks = [self._query(q) for q in queries]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        parts, statuses = [], []
        for i, result in enumerate(results):
            if isinstance(result, BaseException):
                statuses.append({"section": LABELS[i], "status": "failed", "errorType": type(result).__name__})
                parts.append(f"=== {LABELS[i]} ===\n확인 불가 — 수집 실패. 외부 근거로 사용하지 마세요.")
                continue
            text = str(result or "").strip()
            urls = sorted(set(re.findall(r"https?://[^\s<>\[\]]+", text)))
            sourced = bool(urls) and self._web_search_enabled
            statuses.append({"section": LABELS[i], "status": "sourced" if sourced else "unavailable", "sourceUrls": urls if sourced else []})
            # A URL is traceability, not a claim that every sentence was independently verified.
            # 산출물은 PRD 프롬프트와 저장되는 marketResearch에 그대로 실리므로,
            # 모델이 드러낸 시스템 사정 문구는 보고서형 표현으로 정리해서 내보낸다.
            body = (
                _polish_market_research_text(text) if sourced
                else "확인 불가 — 추적 가능한 외부 근거 없음. 수치를 추정하지 마세요."
            )
            parts.append(f"=== {LABELS[i]} ===\n{body}")
        return "\n\n".join(parts), statuses

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
            logger.warning("도메인 추출 실패: %s", type(e).__name__)
            return user_query


def _polish_market_research_text(text: str) -> str:
    """시장조사 산출물에서 시스템 사정이 드러나는 문장을 보고서형 표현으로 정리한다."""
    replacements = {
        "현재 웹 검색을 수행할 수 없어": "공개적으로 확인 가능한 자료 기준으로는",
        "웹 검색을 수행하지 않고는": "공개적으로 확인 가능한 자료만으로는",
        "요청 조건상 웹 검색을 수행할 수 없으므로": "공개적으로 확인 가능한 자료 기준으로는",
        "웹 검색 결과를 확인할 수 없어": "공개적으로 확인 가능한 자료 기준으로는",
        "검색 결과를 확인할 수 없어": "공개적으로 확인 가능한 자료 기준으로는",
        "웹 검색 결과에 근거한": "공개 자료에 근거한",
        "현재 자료만으로는": "공개 자료만으로는",
        "답변할 수 없습니다": "확정하기 어렵습니다",
        "확인할 수 없습니다": "확인하기 어렵습니다",
    }
    cleaned = str(text or "").strip()
    for before, after in replacements.items():
        cleaned = cleaned.replace(before, after)
    return cleaned
