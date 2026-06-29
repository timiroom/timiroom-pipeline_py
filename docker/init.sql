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
