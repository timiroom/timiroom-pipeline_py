CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS document_chunks (
    id           UUID PRIMARY KEY,
    content      TEXT    NOT NULL,
    content_hash TEXT    UNIQUE,
    metadata     JSONB   DEFAULT '{}',
    embedding    vector(1024),
    tokens       TSVECTOR
);

CREATE INDEX IF NOT EXISTS idx_chunks_tokens ON document_chunks USING GIN (tokens);

-- Phase1 SearchRLService (phase1/search_rl_service.py) — similarity_threshold 자동 튜닝
CREATE TABLE IF NOT EXISTS rl_params (
    id                   INT PRIMARY KEY,
    similarity_threshold DOUBLE PRECISION NOT NULL DEFAULT 0.3,
    total_runs           BIGINT NOT NULL DEFAULT 0,
    updated_at           TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
INSERT INTO rl_params (id, similarity_threshold) VALUES (1, 0.3)
ON CONFLICT (id) DO NOTHING;

CREATE TABLE IF NOT EXISTS rl_search_log (
    pipeline_id          TEXT PRIMARY KEY,
    vector_weight        DOUBLE PRECISION NOT NULL,
    keyword_weight       DOUBLE PRECISION NOT NULL,
    similarity_threshold DOUBLE PRECISION NOT NULL,
    chunk_count          INT NOT NULL,
    avg_cohere_score     DOUBLE PRECISION,
    created_at           TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
