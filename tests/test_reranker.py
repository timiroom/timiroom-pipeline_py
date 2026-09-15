import asyncio
import json
import uuid

import httpx

from common.document_chunk import DocumentChunk
from phase1.reranker import RerankerService


def test_cohere_reranker_maps_scores_and_indices() -> None:
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content))
        assert request.headers["Authorization"] == "Bearer test-key"
        return httpx.Response(
            200,
            json={
                "results": [
                    {"index": 1, "relevance_score": 0.91},
                    {"index": 0, "relevance_score": 0.42},
                ]
            },
        )

    async def run() -> None:
        client = httpx.AsyncClient(
            base_url="https://api.cohere.test",
            transport=httpx.MockTransport(handler),
        )
        service = RerankerService(
            top_k_final=2,
            api_key="test-key",
            model="rerank-v4.0-pro",
            client=client,
        )
        candidates = [
            DocumentChunk(id=uuid.uuid4(), content="첫 번째 문서"),
            DocumentChunk(id=uuid.uuid4(), content="두 번째 문서"),
        ]

        result = await service.rerank("검색어", candidates)

        assert result.applied is True
        assert result.chunks == [candidates[1], candidates[0]]
        assert result.chunks[0].relevance_score == 0.91
        assert captured == {
            "model": "rerank-v4.0-pro",
            "query": "검색어",
            "documents": ["첫 번째 문서", "두 번째 문서"],
            "top_n": 2,
        }
        await client.aclose()

    asyncio.run(run())


def test_cohere_failure_marks_rerank_as_not_applied() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={"message": "temporarily unavailable"})

    async def run() -> None:
        client = httpx.AsyncClient(
            base_url="https://api.cohere.test",
            transport=httpx.MockTransport(handler),
        )
        service = RerankerService(api_key="test-key", client=client)
        candidates = [DocumentChunk(id=uuid.uuid4(), content="원본")]

        result = await service.rerank("검색어", candidates)

        assert result.applied is False
        assert result.chunks == candidates
        await client.aclose()

    asyncio.run(run())
