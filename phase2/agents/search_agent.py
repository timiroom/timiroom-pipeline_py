import logging
import re

from openai import AsyncOpenAI

from config.settings import settings
from phase2.public_source_client import PublicSourceClient, format_collection_report
from phase2.state import PipelineState

logger = logging.getLogger(__name__)

_TEMPERATURE = 1.0
_TOP_P = 0.95
_PRESENCE_PENALTY = 0.0

_STATISTICAL_PHRASES = (
    "1인 가구", "맞벌이 가구", "청년", "고령자", "소상공인", "자영업",
    "전자상거래", "온라인 쇼핑", "식품 소비", "반려동물", "취업", "주거",
)


def _select_public_search_keyword(user_query: str, model_domain: str) -> str:
    compact = re.sub(r"\s+", " ", user_query)
    for phrase in _STATISTICAL_PHRASES:
        if phrase in compact:
            return phrase
    household = re.search(r"\b\d+인\s*가구\b", compact)
    if household:
        return re.sub(r"\s*", "", household.group(0)).replace("가구", " 가구")
    return model_domain


class SearchAgent:
    """EXAONE은 검색어 분류에만 사용하고 외부 자료는 비-AI 수집기로 확보한다."""

    def __init__(
        self,
        client: AsyncOpenAI,
        model: str,
        source_client: PublicSourceClient | None = None,
    ):
        self._client = client
        self._model = model
        self._sources = source_client or PublicSourceClient.from_settings(settings)

    async def execute(self, state: PipelineState, dump=None) -> PipelineState:
        logger.info("Search 에이전트 시작 — 비-AI 공식 자료 수집")
        try:
            domain = await self._extract_domain(state.user_query)
            public_keyword = _select_public_search_keyword(state.user_query, domain)
            logger.info("공공 통계 검색어: [%s]", public_keyword)
            report = await self._sources.collect(state.user_query, public_keyword)
            market_data = format_collection_report(report)
            logger.info(
                "Search 에이전트 완료 — 공식 출처 %d개, 상태 알림 %d개",
                len(report.sources),
                len(report.notices),
            )
            if dump:
                dump.log_raw("SEARCH_PUBLIC_SOURCES", 1, market_data)
            return state.copy(
                market_research=market_data,
                status_message="Search 에이전트 완료 — 비-AI 공식 자료 수집",
            )
        except Exception as exc:
            logger.error("Search 에이전트 실패: %s", exc, exc_info=True)
            return state.copy(
                market_research=(
                    "[수집 실패]\n공식 자료 수집에 실패했습니다. "
                    "검증되지 않은 수치·법률·출처는 생성하거나 저장하지 않습니다."
                ),
                status_message="Search 에이전트 실패",
            )

    async def _extract_domain(self, user_query: str) -> str:
        try:
            response = await self._client.chat.completions.create(
                model=self._model,
                max_tokens=100,
                temperature=_TEMPERATURE,
                top_p=_TOP_P,
                presence_penalty=_PRESENCE_PENALTY,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "사용자의 서비스 설명을 읽고 조사 키워드로 사용할 업종 또는 도메인을 "
                            "한국어 2~4단어로만 출력하세요. 다른 설명은 출력하지 마세요."
                        ),
                    },
                    {"role": "user", "content": user_query},
                ],
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
            domain = (response.choices[0].message.content or "").strip()
            logger.info("Search 조사 도메인: [%s]", domain)
            return domain[:100] or user_query[:100]
        except Exception as exc:
            logger.warning("Search 조사 도메인 추출 실패: %s", exc)
            return user_query[:100]
