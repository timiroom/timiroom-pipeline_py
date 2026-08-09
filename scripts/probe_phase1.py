"""Read-only live probe for Phase1 DB, embedding, retrieval, and reranking."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import psycopg2
from psycopg2 import sql


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.settings import settings
from phase1.embedding_service import EMBED_DIM, EmbeddingService
from phase1.form_to_query import FormToQueryService
from phase1.hybrid_search import HybridSearchService
from phase1.models import FormData
from phase1.pdf_parsing import PDFParsingService
from phase1.rag_pipeline import RagPipelineService
from phase1.reranker import RerankerService
from phase1.semantic_chunking import SemanticChunkingService
from phase1.session_vector_store import SessionVectorStore


FORM = FormData.model_validate({
    "projectName": "팀플로우",
    "projectDescription": "메신저에 흩어진 팀 업무와 마감일을 한곳에서 관리하는 서비스",
    "platform": "WEB_APP",
    "techStack": [],
    "problemDefinition": {
        "currentPainPoint": "담당자와 마감일이 여러 대화방에 흩어져 업무를 놓친다",
        "currentSolution": "캘린더와 메모 앱에 직접 기록한다",
        "idealState": "대화에서 할 일을 자동 추출해 담당자와 마감일을 확인한다",
    },
    "targetUsers": [{
        "persona": "메신저로 협업하는 직장인 팀원",
        "usageEnvironment": "웹과 모바일 협업 환경",
        "biggestPainPoint": "여러 프로젝트의 업무 누락",
    }],
    "featureDefinition": {
        "commonFeatures": [],
        "customFeatures": [
            {"featureName": "할 일 자동 추출", "priority": "MUST"},
            {"featureName": "담당자와 마감일 지정", "priority": "MUST"},
            {"featureName": "마감 전 알림", "priority": "MUST"},
        ],
    },
})


def document_distribution() -> list[dict]:
    table = sql.Identifier(settings.get_rag_document_table())
    query = sql.SQL("""
        SELECT COALESCE(metadata->>'type', '(none)') AS type,
               CASE WHEN COALESCE(metadata->>'pipeline_id', '') = ''
                    THEN 'global' ELSE 'pipeline' END AS scope,
               COUNT(*)
        FROM {}
        GROUP BY 1, 2
        ORDER BY 1, 2
    """).format(table)
    with psycopg2.connect(settings.db_url) as conn:
        with conn.cursor() as cur:
            cur.execute(query)
            return [
                {"type": row[0], "scope": row[1], "count": row[2]}
                for row in cur.fetchall()
            ]


async def run() -> None:
    embedder = EmbeddingService(
        settings.upstage_api_key,
        settings.solar_embedding_query_model,
        settings.solar_embedding_passage_model,
    )
    query_vector = await embedder.embed_query("팀 업무와 마감일 관리")
    if len(query_vector) != EMBED_DIM:
        raise RuntimeError(f"임베딩 차원 불일치: {len(query_vector)} != {EMBED_DIM}")

    session_store = SessionVectorStore()
    chunker = SemanticChunkingService(
        embedder,
        max_chunk_size=settings.rag_chunk_size,
        chunk_overlap=settings.rag_chunk_overlap,
    )
    hybrid = HybridSearchService(
        db_url=settings.db_url,
        document_table=settings.get_rag_document_table(),
        embedder=embedder,
        session_store=session_store,
        top_k_vector=settings.rag_top_k_vector,
        top_k_keyword=settings.rag_top_k_keyword,
        similarity_threshold=settings.rag_similarity_threshold,
        min_threshold=settings.rag_min_threshold,
        min_results=settings.rag_min_results,
        threshold_step=settings.rag_threshold_step,
        rl_service=None,
    )
    reranker = RerankerService(
        top_k_final=settings.rag_top_k_final,
        enabled=settings.rag_reranker_enabled,
        ko_reranker_model=settings.ko_reranker_model,
    )
    pipeline = RagPipelineService(
        hybrid_search=hybrid,
        reranker=reranker,
        form_to_query=FormToQueryService(),
        pdf_parsing=PDFParsingService(session_store, embedder, chunker),
        session_store=session_store,
        rl_service=None,
    )
    state = await pipeline.build_from_form(FORM)
    knowledge_count = state.context_prompt.count("\n---")
    print(json.dumps({
        "embeddingDimension": len(query_vector),
        "documentDistribution": document_distribution(),
        "knowledgeChunks": knowledge_count,
        "contextCharacters": len(state.context_prompt),
        "mustFeatures": state.must_features,
        "status": state.status_message,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(run())
