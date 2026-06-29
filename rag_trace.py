import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import asyncio
import psycopg2
import psycopg2.extras
import numpy as np
from pgvector.psycopg2 import register_vector

from config.settings import settings
from phase1.embedding_service import EmbeddingService
from phase1.hybrid_search import HybridSearchService
from phase1.session_vector_store import SessionVectorStore
from phase1.query_expansion import QueryExpansionService
from phase1.reranker import RerankerService
from openai import AsyncOpenAI

SEP = "=" * 60

async def main():
    embedder = EmbeddingService(settings.embedding_model)
    session_store = SessionVectorStore()
    exaone = AsyncOpenAI(
        api_key=settings.exaone_api_key,
        base_url="https://api.friendli.ai/dedicated/v1",
    )

    test_query = "냉장고 재료 레시피 추천 모바일 앱"

    # ── 1. DB에 저장된 임베딩 샘플 확인 ─────────────────────
    print(SEP)
    print("1. DB 임베딩 저장 상태 확인")
    conn = psycopg2.connect(settings.db_url)
    register_vector(conn)
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("""
            SELECT id, content, metadata,
                   embedding IS NOT NULL AS has_emb,
                   array_length(embedding::real[], 1) AS emb_dim
            FROM document_chunks LIMIT 3
        """)
        rows = cur.fetchall()
    conn.close()
    for r in rows:
        print(f"  id={str(r['id'])[:8]}... | dim={r['emb_dim']} | topic={r['metadata'].get('topic','?')}")
        print(f"  내용: {r['content'][:60]}...")

    # ── 2. 쿼리 임베딩 생성 ──────────────────────────────────
    print(SEP)
    print(f"2. 쿼리 임베딩 생성: '{test_query}'")
    query_vec = await embedder.embed_one(test_query)
    print(f"  쿼리 벡터 dim={len(query_vec)} | 첫 5값={[round(v,4) for v in query_vec[:5]]}")

    # ── 3. 벡터 검색 (threshold 없이 전체) ───────────────────
    print(SEP)
    print("3. 벡터 검색 전체 결과 (threshold 없이)")
    conn = psycopg2.connect(settings.db_url)
    register_vector(conn)
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("""
            SELECT content, metadata,
                   1 - (embedding <=> %s::vector) AS score
            FROM document_chunks
            ORDER BY score DESC
            LIMIT 10
        """, (query_vec,))
        all_rows = cur.fetchall()
    conn.close()
    for i, r in enumerate(all_rows):
        marker = " ← threshold(0.5) 통과" if r["score"] >= 0.5 else ""
        print(f"  [{i+1}] score={r['score']:.4f} topic={r['metadata'].get('topic','?')}{marker}")
        print(f"       {r['content'][:70]}...")

    # ── 4. 벡터 일관성 검증 (같은 텍스트 재임베딩 후 비교) ──
    print(SEP)
    print("4. 임베딩 일관성 검증")
    sample_text = all_rows[0]["content"]
    conn = psycopg2.connect(settings.db_url)
    register_vector(conn)
    with conn.cursor() as cur:
        cur.execute("""
            SELECT embedding FROM document_chunks
            WHERE content = %s LIMIT 1
        """, (sample_text,))
        db_vec = cur.fetchone()[0]
    conn.close()

    fresh_vec = await embedder.embed_one(sample_text)
    db_arr = np.array(db_vec, dtype=np.float32)
    fresh_arr = np.array(fresh_vec, dtype=np.float32)
    cosine = float(np.dot(db_arr, fresh_arr) / (np.linalg.norm(db_arr) * np.linalg.norm(fresh_arr)))
    print(f"  DB 저장 벡터 vs 재생성 벡터 코사인 유사도: {cosine:.6f}")
    print(f"  {'완전 일치 (동일 모델 사용 확인)' if cosine > 0.9999 else '불일치 - 모델이 다를 수 있음'}")

    # ── 5. HybridSearch 전체 파이프라인 ─────────────────────
    print(SEP)
    print("5. HybridSearch 실제 실행")
    hybrid = HybridSearchService(
        db_url=settings.db_url,
        embedder=embedder,
        session_store=session_store,
        top_k_vector=settings.rag_top_k_vector,
        top_k_keyword=settings.rag_top_k_keyword,
    )
    results = await hybrid.search(test_query, top_k=10)
    print(f"  HybridSearch 반환 {len(results)}건")
    for i, c in enumerate(results):
        print(f"  [{i+1}] rrf_score={c.relevance_score:.5f} topic={c.metadata.get('topic','?')}")
        print(f"       {c.content[:70]}...")

    # ── 6. Reranker 결과 ─────────────────────────────────────
    print(SEP)
    print("6. Reranker 후 최종 순위")
    reranker = RerankerService(
        client=exaone,
        model=settings.exaone_endpoint_id,
        top_k_final=settings.rag_top_k_final,
        enabled=settings.rag_reranker_enabled,
        cohere_api_key=settings.cohere_api_key,
        ko_reranker_model=settings.ko_reranker_model,
    )
    reranked = await reranker.rerank(test_query, results)
    print(f"  Reranker 반환 {len(reranked)}건")
    for i, c in enumerate(reranked):
        print(f"  [{i+1}] topic={c.metadata.get('topic','?')}")
        print(f"       {c.content[:80]}...")

    print(SEP)
    print("추적 완료")

if __name__ == "__main__":
    asyncio.run(main())
