import psycopg2
from pgvector.psycopg2 import register_vector
from config.settings import settings

conn = psycopg2.connect(settings.db_url)
register_vector(conn)
cur = conn.cursor()

print("=== DB 마이그레이션 ===")

# 1. tokens 컬럼 추가 (이미 완료됐으면 skip)
cur.execute("""
    ALTER TABLE document_chunks
    ADD COLUMN IF NOT EXISTS tokens TSVECTOR
""")
print("  [OK] tokens TSVECTOR 컬럼")

# 2. metadata json -> jsonb 변환
cur.execute("""
    SELECT data_type FROM information_schema.columns
    WHERE table_name = 'document_chunks' AND column_name = 'metadata'
""")
meta_type = cur.fetchone()[0]
if meta_type != "jsonb":
    cur.execute("ALTER TABLE document_chunks ALTER COLUMN metadata TYPE JSONB USING metadata::JSONB")
    print("  [OK] metadata -> jsonb 변환")
else:
    print("  [SKIP] metadata 이미 jsonb")

# 3. embedding 컬럼 차원 확인
cur.execute("""
    SELECT atttypmod FROM pg_attribute
    JOIN pg_class ON pg_class.oid = pg_attribute.attrelid
    WHERE pg_class.relname = 'document_chunks' AND pg_attribute.attname = 'embedding'
""")
row = cur.fetchone()
atttypmod = row[0] if row else -1
print(f"  embedding atttypmod: {atttypmod} (4096이면 정상, -1이면 차원 미지정)")

if atttypmod != 4096:
    print("  embedding 컬럼을 vector(4096)로 재생성 (Solar 임베딩 전환) — 기존 임베딩 데이터는 소실되므로 재수집 필요")
    cur.execute("ALTER TABLE document_chunks DROP COLUMN embedding")
    cur.execute("ALTER TABLE document_chunks ADD COLUMN embedding vector(4096)")
    print("  [OK] embedding vector(4096) 재생성 — reembed_documents.py로 재수집 필요")

# 4. GIN 인덱스
cur.execute("""
    CREATE INDEX IF NOT EXISTS idx_chunks_tokens
    ON document_chunks USING GIN (tokens)
""")
print("  [OK] idx_chunks_tokens GIN 인덱스")

# 5. content_hash 중복 방지 컬럼
cur.execute("""
    ALTER TABLE document_chunks
    ADD COLUMN IF NOT EXISTS content_hash TEXT
""")
cur.execute("""
    DO $$
    BEGIN
        IF NOT EXISTS (
            SELECT 1 FROM pg_constraint
            WHERE conname = 'uq_chunks_content_hash'
        ) THEN
            ALTER TABLE document_chunks
            ADD CONSTRAINT uq_chunks_content_hash UNIQUE (content_hash);
        END IF;
    END$$
""")
print("  [OK] content_hash UNIQUE 컬럼 + 제약조건")

# 6. 벡터 인덱스는 생성하지 않음 — pgvector HNSW/IVFFlat은 vector 타입 기준 2000차원,
#    halfvec으로도 4000차원이 한계라 4096차원 Solar 임베딩은 인덱싱 불가. 순차 스캔 사용.
cur.execute("DROP INDEX IF EXISTS idx_chunks_embedding")
print("  [OK] 벡터 인덱스 없이 순차 스캔 사용 (구 HNSW 인덱스 있었다면 제거)")

# 7. Phase1 SearchRLService (phase1/search_rl_service.py) 테이블
cur.execute("""
    CREATE TABLE IF NOT EXISTS rl_params (
        id                   INT PRIMARY KEY,
        similarity_threshold DOUBLE PRECISION NOT NULL DEFAULT 0.3,
        total_runs           BIGINT NOT NULL DEFAULT 0,
        updated_at           TIMESTAMPTZ NOT NULL DEFAULT NOW()
    )
""")
cur.execute("INSERT INTO rl_params (id, similarity_threshold) VALUES (1, 0.3) ON CONFLICT (id) DO NOTHING")
cur.execute("""
    CREATE TABLE IF NOT EXISTS rl_search_log (
        pipeline_id          TEXT PRIMARY KEY,
        vector_weight        DOUBLE PRECISION NOT NULL,
        keyword_weight       DOUBLE PRECISION NOT NULL,
        similarity_threshold DOUBLE PRECISION NOT NULL,
        chunk_count          INT NOT NULL,
        avg_cohere_score     DOUBLE PRECISION,
        created_at           TIMESTAMPTZ NOT NULL DEFAULT NOW()
    )
""")
print("  [OK] rl_params / rl_search_log 테이블 (SearchRLService)")

conn.commit()
conn.close()

print("\n최종 스키마:")
conn2 = psycopg2.connect(settings.db_url)
cur2 = conn2.cursor()
cur2.execute(
    "SELECT column_name, data_type FROM information_schema.columns "
    "WHERE table_name = 'document_chunks' ORDER BY ordinal_position"
)
for r in cur2.fetchall():
    print(f"  {r[0]}: {r[1]}")

cur2.execute("SELECT indexname FROM pg_indexes WHERE tablename = 'document_chunks'")
print("\n인덱스:")
for r in cur2.fetchall():
    print(f"  {r[0]}")

conn2.close()
print("\n마이그레이션 완료")
