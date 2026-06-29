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

    def __init__(self, client: AsyncOpenAI, model: str = ""):
        self._client = client
        self._model = model

    async def expand(self, query: str) -> list[str]:
        try:
            response = await self._client.chat.completions.create(
                model=self._model,
                temperature=0.7,
                messages=[{"role": "user", "content": EXPANSION_PROMPT.format(query=query)}],
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
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
