import asyncio
import logging

from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

EMBED_DIM = 1024

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
        max_concurrency: int = 8,
        batch_size: int = 64,
    ):
        if max_concurrency <= 0:
            raise ValueError("max_concurrency는 1 이상이어야 합니다")
        if batch_size <= 0:
            raise ValueError("batch_size는 1 이상이어야 합니다")
        self._client = AsyncOpenAI(api_key=api_key, base_url=UPSTAGE_BASE_URL)
        self._query_model = query_model
        self._passage_model = passage_model
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._batch_size = batch_size

    async def _create(self, model: str, inputs: str | list[str]):
        async with self._semaphore:
            return await self._client.embeddings.create(model=model, input=inputs)

    async def close(self) -> None:
        await self._client.close()

    async def _embed_many(self, model: str, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self._batch_size):
            batch = texts[start:start + self._batch_size]
            response = await self._create(model, batch)
            ordered = sorted(response.data, key=lambda item: item.index)
            if len(ordered) != len(batch):
                raise RuntimeError(
                    f"임베딩 응답 개수 불일치: 요청 {len(batch)}개, 응답 {len(ordered)}개"
                )
            vectors.extend(item.embedding for item in ordered)
        return vectors

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """문서/패시지 임베딩 (ingestion, chunking 등 저장 대상 텍스트)."""
        return await self._embed_many(self._passage_model, texts)

    async def embed_query(self, text: str) -> list[float]:
        """단일 검색 쿼리 임베딩."""
        resp = await self._create(self._query_model, text)
        return resp.data[0].embedding

    async def embed_queries(self, texts: list[str]) -> list[list[float]]:
        """복수 검색 쿼리 임베딩."""
        return await self._embed_many(self._query_model, texts)
