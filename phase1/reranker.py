import asyncio
import logging
from dataclasses import dataclass

import httpx

from common.document_chunk import DocumentChunk

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RerankResult:
    chunks: list[DocumentChunk]
    applied: bool


class RerankerService:

    def __init__(
        self,
        top_k_final: int = 5,
        enabled: bool = True,
        api_key: str = "",
        model: str = "rerank-v4.0-pro",
        base_url: str = "https://api.cohere.com",
        client: httpx.AsyncClient | None = None,
        max_concurrency: int = 5,
    ):
        if max_concurrency <= 0:
            raise ValueError("max_concurrency는 1 이상이어야 합니다")
        self._top_k = top_k_final
        self._enabled = enabled
        self._api_key = api_key
        self._model = model
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            timeout=30.0,
        )
        self._semaphore = asyncio.Semaphore(max_concurrency)

    async def close(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def rerank(self, query: str, candidates: list[DocumentChunk]) -> RerankResult:
        if not self._enabled or not candidates:
            return RerankResult(candidates[: self._top_k], applied=False)

        if not self._api_key:
            logger.warning("COHERE_API_KEY가 없어 리랭킹 없이 원본 순서를 사용합니다")
            return RerankResult(candidates[: self._top_k], applied=False)

        try:
            async with self._semaphore:
                response = await self._client.post(
                    "/v2/rerank",
                    headers={
                        "Authorization": f"Bearer {self._api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": self._model,
                        "query": query,
                        "documents": [c.content for c in candidates],
                        "top_n": min(self._top_k, len(candidates)),
                    },
                )
            response.raise_for_status()
            ranked = response.json().get("results")
            if not isinstance(ranked, list) or not ranked:
                raise ValueError("Cohere 응답에 results가 없습니다")
            result = []
            for item in ranked:
                index = int(item["index"])
                if index < 0 or index >= len(candidates):
                    raise ValueError(f"Cohere 응답 index 범위 오류: {index}")
                c = candidates[index]
                c.relevance_score = float(item["relevance_score"])
                result.append(c)
            logger.debug("Cohere Rerank 완료 — %d docs 반환", len(result))
            return RerankResult(result, applied=True)
        except Exception as e:
            logger.warning("Cohere Rerank 실패 — 원본 순서 사용: %s", e)
            return RerankResult(candidates[: self._top_k], applied=False)
