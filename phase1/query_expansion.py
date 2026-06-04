import logging

from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

EXPANSION_PROMPT = """
당신은 검색 전문가입니다.
아래 쿼리를 6개의 서로 다른 검색 쿼리로 확장하세요.
각 쿼리는 줄바꿈으로 구분하고 다른 텍스트는 포함하지 마세요.

원본 쿼리: {query}
"""


class QueryExpansionService:

    def __init__(self, client: AsyncOpenAI):
        self._client = client

    async def expand(self, query: str) -> list[str]:
        try:
            response = await self._client.chat.completions.create(
                model="gpt-4o-mini",
                temperature=0.7,
                messages=[{"role": "user", "content": EXPANSION_PROMPT.format(query=query)}],
            )
            lines = response.choices[0].message.content.strip().splitlines()
            expanded = [l.strip() for l in lines if l.strip()]
            if not expanded:
                return [query]
            logger.debug("쿼리 확장: %d 개", len(expanded))
            return expanded
        except Exception as e:
            logger.warning("쿼리 확장 실패: %s", e)
            return [query]
