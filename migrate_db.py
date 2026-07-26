"""Solar 4096차원 RAG 테이블과 RL 테이블을 멱등 생성한다.

기존 Spring 파이프라인의 ``document_chunks``(vector(1024))는 건드리지 않는다.
NAS 배포에서는 RAG_DOCUMENT_TABLE=document_chunks_ko를 사용한다.
"""

import psycopg2
from psycopg2 import sql

from config.settings import settings


def main() -> None:
    table_name = settings.get_rag_document_table()
    table = sql.Identifier(table_name)

    with psycopg2.connect(settings.db_url) as conn:
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            cur.execute(
                sql.SQL(
                    """
                    CREATE TABLE IF NOT EXISTS {} (
                        id           UUID PRIMARY KEY,
                        content      TEXT NOT NULL,
                        content_hash TEXT UNIQUE,
                        metadata     JSONB DEFAULT '{{}}'::jsonb,
                        embedding    vector(4096),
                        tokens       TSVECTOR
                    )
                    """
                ).format(table)
            )

            cur.execute(
                """
                SELECT a.atttypmod
                FROM pg_attribute a
                JOIN pg_class c ON c.oid = a.attrelid
                WHERE c.relname = %s AND a.attname = 'embedding'
                """,
                (table_name,),
            )
            row = cur.fetchone()
            if not row or row[0] != 4096:
                raise RuntimeError(
                    f"{table_name}.embedding은 vector(4096)이어야 합니다. "
                    "기존 vector(1024) 테이블을 변경하지 말고 "
                    "RAG_DOCUMENT_TABLE=document_chunks_ko를 사용하세요."
                )

            cur.execute(
                sql.SQL("CREATE INDEX IF NOT EXISTS {} ON {} USING GIN (tokens)").format(
                    sql.Identifier(f"idx_{table_name}_tokens"), table
                )
            )

            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS rl_params (
                    id                   INT PRIMARY KEY,
                    similarity_threshold DOUBLE PRECISION NOT NULL DEFAULT 0.3,
                    total_runs           BIGINT NOT NULL DEFAULT 0,
                    updated_at           TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
                """
            )
            cur.execute(
                """
                INSERT INTO rl_params (id, similarity_threshold)
                VALUES (1, 0.3)
                ON CONFLICT (id) DO NOTHING
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS rl_search_log (
                    pipeline_id          TEXT PRIMARY KEY,
                    vector_weight        DOUBLE PRECISION NOT NULL,
                    keyword_weight       DOUBLE PRECISION NOT NULL,
                    similarity_threshold DOUBLE PRECISION NOT NULL,
                    chunk_count          INT NOT NULL,
                    avg_cohere_score     DOUBLE PRECISION,
                    created_at           TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
                """
            )

    print(f"DB migration complete: {table_name} (vector(4096))")


if __name__ == "__main__":
    main()
