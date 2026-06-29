import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import asyncio
import psycopg2
import psycopg2.extras
from pgvector.psycopg2 import register_vector
from kiwipiepy import Kiwi
from config.settings import settings
from phase1.embedding_service import EmbeddingService

SEP = "=" * 55
SEARCH_TAGS = {"NNG", "NNP", "NNB", "SL", "SH"}


async def main():
    embedder = EmbeddingService(settings.embedding_model)
    kiwi = Kiwi()
    test_query = "냉장고 재료 레시피 추천 모바일 앱"

    # 1. DB 청크 현황
    print(SEP)
    print("1. DB 청크 현황")
    conn = psycopg2.connect(settings.db_url)
    register_vector(conn)
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM document_chunks")
        total = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM document_chunks WHERE tokens IS NOT NULL")
        with_tokens = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM document_chunks WHERE embedding IS NOT NULL")
        with_emb = cur.fetchone()[0]
    conn.close()
    print(f"  전체 청크: {total}")
    print(f"  tokens 있음: {with_tokens}")
    print(f"  embedding 있음: {with_emb}")

    # 2. 벡터 검색 threshold 분포
    print(SEP)
    print(f"2. 코사인 유사도 분포 (쿼리: '{test_query}')")
    query_vec = await embedder.embed_one(test_query)
    conn = psycopg2.connect(settings.db_url)
    register_vector(conn)
    with conn.cursor() as cur:
        cur.execute("""
            SELECT
                COUNT(*) FILTER (WHERE 1-(embedding <=> %s::vector) >= 0.7) as ge07,
                COUNT(*) FILTER (WHERE 1-(embedding <=> %s::vector) >= 0.5) as ge05,
                COUNT(*) FILTER (WHERE 1-(embedding <=> %s::vector) >= 0.3) as ge03
            FROM document_chunks
        """, (query_vec, query_vec, query_vec))
        row = cur.fetchone()
    print(f"  score >= 0.7: {row[0]}개")
    print(f"  score >= 0.5: {row[1]}개  <-- 현재 threshold")
    print(f"  score >= 0.3: {row[2]}개")

    # 3. 벡터 검색 상위 5개
    print(SEP)
    print("3. 벡터 검색 상위 5개 (threshold 없이)")
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("""
            SELECT content, metadata, 1-(embedding <=> %s::vector) AS score
            FROM document_chunks ORDER BY score DESC LIMIT 5
        """, (query_vec,))
        rows = cur.fetchall()
    for i, r in enumerate(rows):
        topic = r["metadata"].get("topic", "?")
        print(f"  [{i+1}] score={r['score']:.4f} topic={topic}")
        print(f"       {r['content'][:80]}...")

    # 4. 키워드 검색 AND vs OR 비교
    print(SEP)
    print(f"4. 키워드 검색 AND vs OR")
    tokens = [t.form for t in kiwi.tokenize(test_query) if t.tag in SEARCH_TAGS]
    ts_and = " & ".join(tokens)
    ts_or = " | ".join(tokens)
    print(f"  추출 토큰: {tokens}")
    with conn.cursor() as cur:
        cur.execute(
            "SELECT COUNT(*) FROM document_chunks WHERE tokens IS NOT NULL AND tokens @@ to_tsquery('simple', %s)",
            (ts_and,)
        )
        and_cnt = cur.fetchone()[0]
        cur.execute(
            "SELECT COUNT(*) FROM document_chunks WHERE tokens IS NOT NULL AND tokens @@ to_tsquery('simple', %s)",
            (ts_or,)
        )
        or_cnt = cur.fetchone()[0]
    conn.close()
    print(f"  AND 매칭: {and_cnt}건  <-- 현재 방식 (모든 토큰 일치 필요)")
    print(f"  OR  매칭: {or_cnt}건  (하나라도 일치)")
    print(SEP)
    print("진단 완료")


if __name__ == "__main__":
    asyncio.run(main())
