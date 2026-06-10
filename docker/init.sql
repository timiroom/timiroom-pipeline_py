CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS document_chunks (
    id        UUID PRIMARY KEY,
    content   TEXT    NOT NULL,
    metadata  JSONB   DEFAULT '{}',
    embedding vector(3072),
    tokens    TSVECTOR
);

CREATE INDEX IF NOT EXISTS idx_chunks_tokens ON document_chunks USING GIN (tokens);
