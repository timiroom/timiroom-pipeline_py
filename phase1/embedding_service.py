import logging

from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

EMBED_DIM = 4096

UPSTAGE_BASE_URL = "https://api.upstage.ai/v1"


class EmbeddingService:
    """
    Upstage Solar Embedding API 기반 임베딩 서비스.
    query/passage 모델이 분리돼 있어 용도에 맞는 메서드를 사용해야 한다.
    """

    def __init__(
        self,
        api_key: str,
        query_model: str = "solar-embedding-2-query",
        passage_model: str = "solar-embedding-2-passage",
    ):
        self._client = AsyncOpenAI(api_key=api_key, base_url=UPSTAGE_BASE_URL)
        self._query_model = query_model
        self._passage_model = passage_model

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """문서/패시지 임베딩 (ingestion, chunking 등 저장 대상 텍스트)."""
        resp = await self._client.embeddings.create(model=self._passage_model, input=texts)
        return [d.embedding for d in resp.data]

    async def embed_query(self, text: str) -> list[float]:
        """단일 검색 쿼리 임베딩."""
        resp = await self._client.embeddings.create(model=self._query_model, input=text)
        return resp.data[0].embedding

    async def embed_queries(self, texts: list[str]) -> list[list[float]]:
        """복수 검색 쿼리 임베딩."""
        resp = await self._client.embeddings.create(model=self._query_model, input=texts)
        return [d.embedding for d in resp.data]
