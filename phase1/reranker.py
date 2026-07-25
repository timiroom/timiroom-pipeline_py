import asyncio
import logging

from common.document_chunk import DocumentChunk

logger = logging.getLogger(__name__)


class RerankerService:

    def __init__(
        self,
        top_k_final: int = 5,
        enabled: bool = True,
        ko_reranker_model: str = "Dongjin-kr/ko-reranker",
    ):
        self._top_k = top_k_final
        self._enabled = enabled
        self._local_reranker = None

        if ko_reranker_model:
            try:
                from sentence_transformers import CrossEncoder
                self._local_reranker = CrossEncoder(ko_reranker_model)
                logger.info("Ko-Reranker 로딩 완료: %s", ko_reranker_model)
            except Exception as e:
                logger.warning("Ko-Reranker 로딩 실패 — 리랭킹 없이 원본 순서 사용: %s", e)

    async def rerank(self, query: str, candidates: list[DocumentChunk]) -> list[DocumentChunk]:
        if not self._enabled or not candidates:
            return candidates[: self._top_k]

        if not self._local_reranker:
            return candidates[: self._top_k]

        try:
            pairs = [(query, c.content) for c in candidates]
            loop = asyncio.get_event_loop()
            scores = await loop.run_in_executor(
                None,
                lambda: self._local_reranker.predict(pairs, show_progress_bar=False),
            )
            ranked = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
            result = []
            for score, c in ranked[: self._top_k]:
                c.relevance_score = float(score)
                result.append(c)
            logger.debug("Ko-Reranker 완료 — %d docs 반환", len(result))
            return result
        except Exception as e:
            logger.warning("Ko-Reranker 추론 실패 — 원본 순서 사용: %s", e)
            return candidates[: self._top_k]
