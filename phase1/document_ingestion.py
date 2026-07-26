import asyncio
import hashlib
import json
import logging
import uuid

import psycopg2
import psycopg2.extras
from kiwipiepy import Kiwi
from pgvector.psycopg2 import register_vector

from .embedding_service import EmbeddingService
from .semantic_chunking import SemanticChunkingService

logger = logging.getLogger(__name__)

_kiwi = Kiwi()

KAFKA_CHUNK_SIZE = 800


class DocumentIngestionService:

    def __init__(
        self,
        db_url: str,
        document_table: str,
        embedder: EmbeddingService,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
    ):
        self._db_url = db_url
        self._document_table = document_table
        self._embedder = embedder
        self._semantic_chunker = SemanticChunkingService(
            embedder, max_chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )

    def _get_conn(self):
        conn = psycopg2.connect(self._db_url)
        register_vector(conn)
        return conn

    async def ingest(self, content: str, metadata: dict) -> int:
        logger.info("문서 수집 시작 — source: %s", metadata.get("source", "unknown"))
        chunks = await self._semantic_chunker.chunk(content, metadata)
        texts = [c.content for c in chunks]
        saved = await self._embed_and_store(texts, metadata)
        logger.info("문서 수집 완료 — %d chunks 저장됨", saved)
        return saved

    async def ingest_fixed(self, content: str, metadata: dict) -> int:
        """고정 크기 청크 → 임베딩 → pgvector 저장. Kafka Consumer path 전용."""
        texts = self._split_fixed(content, KAFKA_CHUNK_SIZE)
        return await self._embed_and_store(texts, metadata)

    async def ingest_all(self, contents: list[str], shared_metadata: dict) -> int:
        total = 0
        for i, content in enumerate(contents):
            meta = {**shared_metadata, "doc_index": i}
            total += await self.ingest(content, meta)
        return total

    async def _embed_and_store(self, texts: list[str], metadata: dict) -> int:
        if not texts:
            return 0
        embeddings = await self._embedder.embed(texts)
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._store_to_db, texts, embeddings, metadata)

    def _store_to_db(self, texts: list[str], embeddings: list[list[float]], metadata: dict) -> int:
        conn = self._get_conn()
        saved = 0
        try:
            with conn.cursor() as cur:
                for i, (text, vec) in enumerate(zip(texts, embeddings)):
                    try:
                        meta = {**metadata, "chunk_index": i}
                        tokens_text = " ".join(t.form for t in _kiwi.tokenize(text))
                        content_hash = hashlib.md5(text.encode()).hexdigest()
                        cur.execute(
                            f"""
                            INSERT INTO {self._document_table} (id, content, content_hash, metadata, embedding, tokens)
                            VALUES (%s, %s, %s, %s::jsonb, %s::vector, to_tsvector('simple', %s))
                            ON CONFLICT (content_hash) DO NOTHING
                            """,
                            (
                                str(uuid.uuid4()),
                                text,
                                content_hash,
                                json.dumps(meta, ensure_ascii=False),
                                str(vec),
                                tokens_text,
                            ),
                        )
                        if cur.rowcount > 0:
                            saved += 1
                        conn.commit()
                    except Exception as e:
                        conn.rollback()
                        logger.warning("청크 저장 실패 (건너뜀) — index %d: %s", i, e)
        finally:
            conn.close()
        return saved

    @staticmethod
    def _split_fixed(text: str, size: int) -> list[str]:
        return [text[i:i + size].strip() for i in range(0, len(text), size) if text[i:i + size].strip()]
