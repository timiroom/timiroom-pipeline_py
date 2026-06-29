import asyncio
import logging

import httpx
from openai import AsyncOpenAI

from common.document_chunk import DocumentChunk

logger = logging.getLogger(__name__)

RERANK_PROMPT = """
당신은 검색 결과 관련도 평가 전문가입니다.
아래 쿼리와 문서 목록을 보고, 쿼리와 관련도가 높은 순서로 문서 번호를 반환하세요.

쿼리: {query}

문서 목록:
{docs}

규칙:
- 관련도 높은 순서로 상위 {top_k}개의 문서 번호만 반환하세요
- 숫자만 쉼표로 구분하여 반환하세요 (예: 3,1,5,2,4)
- 다른 텍스트는 절대 포함하지 마세요
"""


class RerankerService:

    def __init__(
        self,
        client: AsyncOpenAI,
        model: str = "",
        top_k_final: int = 5,
        enabled: bool = True,
        cohere_api_key: str = "",
        cohere_model: str = "rerank-multilingual-v3.0",
        ko_reranker_model: str = "Dongjin-kr/ko-reranker",
    ):
        self._client = client
        self._model = model
        self._top_k = top_k_final
        self._enabled = enabled
        self._cohere_key = cohere_api_key
        self._cohere_model = cohere_model
        self._local_reranker = None

        if ko_reranker_model:
            try:
                from sentence_transformers import CrossEncoder
                self._local_reranker = CrossEncoder(ko_reranker_model)
                logger.info("Ko-Reranker 로딩 완료: %s", ko_reranker_model)
            except Exception as e:
                logger.warning("Ko-Reranker 로딩 실패 — GPT fallback 사용: %s", e)

    async def rerank(self, query: str, candidates: list[DocumentChunk]) -> list[DocumentChunk]:
        if not self._enabled or not candidates:
            return candidates[: self._top_k]

        if self._local_reranker:
            return await self._rerank_local(query, candidates)
        if self._cohere_key:
            return await self._rerank_cohere(query, candidates)
        return await self._rerank_gpt(query, candidates)

    async def _rerank_local(self, query: str, candidates: list[DocumentChunk]) -> list[DocumentChunk]:
        try:
            pairs = [(query, c.content) for c in candidates]
            loop = asyncio.get_event_loop()
            scores = await loop.run_in_executor(
                None,
                lambda: self._local_reranker.predict(pairs, show_progress_bar=False),
            )
            ranked = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
            result = [c for _, c in ranked[: self._top_k]]
            logger.debug("Ko-Reranker 완료 — %d docs 반환", len(result))
            return result
        except Exception as e:
            logger.warning("Ko-Reranker 추론 실패: %s — GPT fallback", e)
            return await self._rerank_gpt(query, candidates)

    async def _rerank_gpt(self, query: str, candidates: list[DocumentChunk]) -> list[DocumentChunk]:
        try:
            doc_lines = []
            for i, chunk in enumerate(candidates):
                preview = chunk.content[:200] + ("..." if len(chunk.content) > 200 else "")
                doc_lines.append(f"{i + 1}. {preview}")

            response = await self._client.chat.completions.create(
                model=self._model,
                temperature=0.0,
                messages=[{
                    "role": "user",
                    "content": RERANK_PROMPT.format(
                        query=query,
                        docs="\n".join(doc_lines),
                        top_k=self._top_k,
                    ),
                }],
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
            text = response.choices[0].message.content.strip()
            indices = [int(p.strip()) - 1 for p in text.split(",") if p.strip().isdigit()]
            reranked = [candidates[i] for i in indices if 0 <= i < len(candidates)]
            if reranked:
                logger.debug("GPT Reranker 완료 — %d docs", len(reranked))
                return reranked
        except Exception as e:
            logger.warning("GPT Reranker 실패: %s", e)

        return candidates[: self._top_k]

    async def _rerank_cohere(self, query: str, candidates: list[DocumentChunk]) -> list[DocumentChunk]:
        try:
            documents = [c.content for c in candidates]
            async with httpx.AsyncClient(timeout=30) as http:
                resp = await http.post(
                    "https://api.cohere.ai/v1/rerank",
                    headers={"Authorization": f"Bearer {self._cohere_key}"},
                    json={
                        "model": self._cohere_model,
                        "query": query,
                        "documents": documents,
                        "top_n": self._top_k,
                    },
                )
                data = resp.json()

            results = sorted(data.get("results", []), key=lambda r: r["relevance_score"], reverse=True)
            return [candidates[r["index"]] for r in results]
        except Exception as e:
            logger.warning("Cohere Reranker 실패: %s — GPT fallback", e)
            return await self._rerank_gpt(query, candidates)
