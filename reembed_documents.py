"""RAG_DOCUMENT_TABLE의 빈 임베딩을 Solar passage 모델로 재수집한다."""
import asyncio

import psycopg2
import psycopg2.extras
from pgvector.psycopg2 import register_vector

from config.settings import settings
from phase1.embedding_service import EmbeddingService

BATCH_SIZE = 50


async def main() -> None:
    table_name = settings.get_rag_document_table()
    embedder = EmbeddingService(
        settings.upstage_api_key,
        settings.solar_embedding_query_model,
        settings.solar_embedding_passage_model,
    )

    conn = psycopg2.connect(settings.db_url)
    register_vector(conn)
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(f"SELECT id, content FROM {table_name} WHERE embedding IS NULL")
            rows = cur.fetchall()

        print(f"재임베딩 대상: {len(rows)}건")

        for i in range(0, len(rows), BATCH_SIZE):
            batch = rows[i : i + BATCH_SIZE]
            texts = [r["content"] for r in batch]
            vectors = await embedder.embed(texts)

            with conn.cursor() as cur:
                for row, vec in zip(batch, vectors):
                    cur.execute(
                        f"UPDATE {table_name} SET embedding = %s::vector WHERE id = %s",
                        (vec, row["id"]),
                    )
            conn.commit()
            print(f"  [{min(i + BATCH_SIZE, len(rows))}/{len(rows)}] 저장 완료")

        print("재임베딩 완료")
    finally:
        conn.close()


if __name__ == "__main__":
    asyncio.run(main())
