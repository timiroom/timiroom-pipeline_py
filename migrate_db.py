"""Solar Embedding 2용 1024차원 RAG 테이블과 RL 테이블을 멱등 생성한다.

기존 테이블이 같은 1024차원이라면 원문과 벡터를 보존하면서 필요한 컬럼만 보강한다.
"""

import psycopg2
from psycopg2 import sql

from config.settings import settings


def remove_global_content_hash_uniqueness(cur, table_name: str) -> None:
    """Replace legacy hash-only uniqueness, including custom and partial indexes."""
    cur.execute(
        """
        SELECT tn.nspname, t.relname, ixn.nspname, ix.relname, con.conname
        FROM pg_index i
        JOIN pg_class t ON t.oid = i.indrelid
        JOIN pg_namespace tn ON tn.oid = t.relnamespace
        JOIN pg_class ix ON ix.oid = i.indexrelid
        JOIN pg_namespace ixn ON ixn.oid = ix.relnamespace
        JOIN pg_attribute a ON a.attrelid = t.oid AND a.attnum = i.indkey[0]
        LEFT JOIN pg_constraint con ON con.conindid = i.indexrelid AND con.contype = 'u'
        WHERE i.indrelid = %s::regclass AND i.indisunique AND NOT i.indisprimary
          AND i.indnkeyatts = 1 AND i.indexprs IS NULL AND a.attname = 'content_hash'
        """,
        (table_name,),
    )
    for table_schema, resolved_table, index_schema, index_name, constraint in cur.fetchall():
        if constraint:
            cur.execute(sql.SQL("ALTER TABLE {} DROP CONSTRAINT {}").format(
                sql.Identifier(table_schema, resolved_table), sql.Identifier(constraint),
            ))
        else:
            cur.execute(sql.SQL("DROP INDEX {}").format(sql.Identifier(index_schema, index_name)))


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
                        content_hash TEXT NOT NULL,
                        source_key   TEXT UNIQUE NOT NULL,
                        metadata     JSONB DEFAULT '{{}}'::jsonb,
                        embedding    vector(1024),
                        tokens       TSVECTOR
                    )
                    """
                ).format(table)
            )

            # 기존 전역 content_hash UNIQUE는 프로젝트가 다른 동일 문서를 누락시키므로 제거하고,
            # pipeline/type/chunk 범위의 source_key를 중복 방지 키로 사용한다.
            cur.execute(
                sql.SQL("ALTER TABLE {} ADD COLUMN IF NOT EXISTS source_key TEXT").format(table)
            )
            cur.execute(
                sql.SQL(
                    "UPDATE {} SET source_key = md5(COALESCE(metadata->>'pipeline_id', 'legacy') || id::text) "
                    "WHERE source_key IS NULL"
                ).format(table)
            )
            cur.execute(sql.SQL("ALTER TABLE {} ALTER COLUMN source_key SET NOT NULL").format(table))
            cur.execute(
                sql.SQL("CREATE UNIQUE INDEX IF NOT EXISTS {} ON {} (source_key)").format(
                    sql.Identifier(f"idx_{table_name}_source_key"), table
                )
            )
            remove_global_content_hash_uniqueness(cur, table_name)

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
            if not row or row[0] != 1024:
                raise RuntimeError(
                    f"{table_name}.embedding은 Solar Embedding 2와 호환되는 "
                    "vector(1024)여야 합니다."
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
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS pipeline_outbox (
                    pipeline_id TEXT PRIMARY KEY,
                    topic       TEXT NOT NULL,
                    payload     JSONB NOT NULL,
                    attempts    INT NOT NULL DEFAULT 0,
                    last_error  TEXT,
                    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
                """
            )
            cur.execute("""CREATE TABLE IF NOT EXISTS pipeline_storage_receipts (
                pipeline_id TEXT PRIMARY KEY, source_keys JSONB NOT NULL,
                published BOOLEAN NOT NULL DEFAULT FALSE,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )""")

    print(f"DB migration complete: {table_name} (vector(1024))")


if __name__ == "__main__":
    main()
