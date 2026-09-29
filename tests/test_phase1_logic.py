import asyncio
import sys
import uuid
from types import SimpleNamespace

import pytest

from common.document_chunk import DocumentChunk
from phase1.document_ingestion import DocumentIngestionService
from phase1.hybrid_search import (
    _SEARCH_TYPE_SQL,
    RRF_K,
    SESSION_BOOST,
    HybridSearchService,
)
from phase1.recommendation.models import (
    PersonaRecommendationRequest,
    PersonaRecommendationResponse,
    RecommendedPersona,
)
from phase1.reranker import RerankerService
from phase1.semantic_chunking import SemanticChunkingService
from phase1.session_vector_store import SessionVectorStore
from routers import rag as rag_router
from routers import recommendation as recommendation_router


class DummyEmbedder:
    async def embed(self, texts):
        return [[1.0, 0.0] for _ in texts]


def make_chunk(content: str, score: float = 0.0) -> DocumentChunk:
    return DocumentChunk(id=uuid.uuid4(), content=content, relevance_score=score)


def make_search() -> HybridSearchService:
    return HybridSearchService(
        db_url="postgresql://unused",
        document_table="document_chunks",
        embedder=DummyEmbedder(),
        session_store=SessionVectorStore(),
    )


def test_long_single_sentence_chunking_terminates_and_respects_max_size():
    service = SemanticChunkingService(DummyEmbedder(), max_chunk_size=10, chunk_overlap=2)

    chunks = asyncio.run(service.chunk("가" * 25, {"source": "test"}))

    assert [len(chunk.content) for chunk in chunks] == [10, 10, 9]
    assert all(len(chunk.content) <= 10 for chunk in chunks)


def test_chunking_rejects_overlap_that_cannot_advance():
    with pytest.raises(ValueError, match="chunk_overlap"):
        SemanticChunkingService(DummyEmbedder(), max_chunk_size=10, chunk_overlap=10)


def test_source_documents_are_included_in_global_search_contract():
    assert "'source'" in _SEARCH_TYPE_SQL


def test_content_hash_keeps_document_types_separate_without_pipeline_id():
    text = "같은 본문"
    source_hash = DocumentIngestionService._content_hash(text, {"type": "source"})
    feature_hash = DocumentIngestionService._content_hash(text, {"type": "features"})

    assert source_hash != feature_hash


def test_ingestion_rolls_back_and_raises_on_chunk_storage_failure(monkeypatch):
    class Cursor:
        rowcount = 1

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def execute(self, *_args):
            raise RuntimeError("db write failed")

    class Connection:
        def __init__(self):
            self.rolled_back = False
            self.closed = False

        def cursor(self):
            return Cursor()

        def commit(self):
            raise AssertionError("실패한 트랜잭션은 commit하면 안 됩니다")

        def rollback(self):
            self.rolled_back = True

        def close(self):
            self.closed = True

    connection = Connection()
    service = DocumentIngestionService(
        "postgresql://unused", "document_chunks", DummyEmbedder()
    )
    monkeypatch.setattr(service, "_get_conn", lambda: connection)

    with pytest.raises(RuntimeError, match="db write failed"):
        service._store_to_db(["저장할 문서"], [[1.0, 0.0]], {"type": "source"})

    assert connection.rolled_back
    assert connection.closed


def test_rag_ingest_marks_source_searchable_and_reports_saved_count(monkeypatch):
    captured = {}

    class Ingestion:
        async def ingest(self, content, metadata):
            captured.update(content=content, metadata=metadata)
            return 3

    monkeypatch.setitem(
        sys.modules,
        "main",
        SimpleNamespace(document_ingestion_service=Ingestion()),
    )

    response = asyncio.run(rag_router.ingest(rag_router.IngestRequest(content="자료", source="manual")))

    assert captured["metadata"] == {"source": "manual", "type": "source"}
    assert response["data"]["savedChunks"] == 3


def test_search_raises_when_vector_and_keyword_channels_both_fail(monkeypatch):
    service = make_search()

    async def vector_failure(*_args):
        raise RuntimeError("vector down")

    def keyword_failure(*_args):
        raise RuntimeError("keyword down")

    monkeypatch.setattr(service, "_vector_search", vector_failure)
    monkeypatch.setattr(service, "_keyword_search", keyword_failure)

    with pytest.raises(RuntimeError, match="검색 모두 실패"):
        asyncio.run(service.search("query", top_k=5))


def test_search_uses_surviving_channel_when_only_one_fails(monkeypatch):
    service = make_search()
    expected = make_chunk("vector result", 0.9)

    async def vector_success(*_args):
        return [expected]

    def keyword_failure(*_args):
        raise RuntimeError("keyword down")

    monkeypatch.setattr(service, "_vector_search", vector_success)
    monkeypatch.setattr(service, "_keyword_search", keyword_failure)

    result = asyncio.run(service.search("query", top_k=5))

    assert [chunk.id for chunk in result] == [expected.id]


def test_session_rrf_weight_changes_final_order():
    service = make_search()
    global_chunk = make_chunk("global")
    session_chunk = make_chunk("uploaded pdf")

    result = service._rrf(
        [global_chunk],
        [session_chunk],
        top_k=2,
        weight_b=SESSION_BOOST,
    )

    assert result[0].id == session_chunk.id
    assert result[0].relevance_score == pytest.approx(SESSION_BOOST / (RRF_K + 1))


def test_multiquery_exposes_accumulated_rrf_score(monkeypatch):
    service = make_search()
    shared = make_chunk("shared", score=999.0)

    async def search(*_args, **_kwargs):
        return [shared]

    monkeypatch.setattr(service, "search", search)

    result = asyncio.run(service.search_multiple(["one", "two"], top_k=5))

    assert result[0].relevance_score == pytest.approx(2 / (RRF_K + 1))


def test_reranker_reports_whether_cross_encoder_scores_were_applied():
    candidates = [make_chunk("candidate", 0.2)]
    service = RerankerService(enabled=True, ko_reranker_model="")

    unchanged, applied = asyncio.run(service.rerank_with_status("query", candidates))
    assert unchanged == candidates
    assert not applied

    service._local_reranker = SimpleNamespace(
        predict=lambda *_args, **_kwargs: [0.9]
    )
    reranked, applied = asyncio.run(service.rerank_with_status("query", candidates))
    assert applied
    assert reranked[0].relevance_score == pytest.approx(0.9)


def test_recommendation_router_serializes_aliases_for_frontend(monkeypatch):
    class PersonaService:
        async def recommend(self, _req):
            return PersonaRecommendationResponse(personas=[RecommendedPersona(
                persona="팀원",
                usageEnvironment="메신저 협업 환경",
                biggestPainPoint="업무 누락",
            )])

    monkeypatch.setattr(
        recommendation_router,
        "_services",
        lambda: (None, PersonaService(), None),
    )
    request = PersonaRecommendationRequest.model_validate({
        "projectName": "팀플로우",
        "projectDescription": "업무 관리",
        "problemDefinition": {
            "currentPainPoint": "업무 누락",
            "currentSolution": "메모",
            "idealState": "자동 정리",
        },
    })

    response = asyncio.run(recommendation_router.persona(request))
    persona = response["data"]["personas"][0]

    assert persona["usageEnvironment"] == "메신저 협업 환경"
    assert persona["biggestPainPoint"] == "업무 누락"
    assert "usage_environment" not in persona
