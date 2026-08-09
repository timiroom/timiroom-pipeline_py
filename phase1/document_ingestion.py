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
        if len(embeddings) != len(texts):
            raise RuntimeError(
                f"임베딩 개수 불일치: texts={len(texts)}, embeddings={len(embeddings)}"
            )
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._store_to_db, texts, embeddings, metadata)

    def _store_to_db(self, texts: list[str], embeddings: list[list[float]], metadata: dict) -> int:
        if len(texts) != len(embeddings):
            raise ValueError("texts와 embeddings 개수가 다릅니다")
        conn = self._get_conn()
        saved = 0
        try:
            with conn.cursor() as cur:
                for i, (text, vec) in enumerate(zip(texts, embeddings)):
                    meta = {**metadata, "chunk_index": i}
                    tokens_text = " ".join(t.form for t in _kiwi.tokenize(text))
                    content_hash = self._content_hash(text, metadata)
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
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
        return saved

    @staticmethod
    def _content_hash(text: str, metadata: dict) -> str:
        """Deduplicate source documents globally, but pipeline outputs per run.

        Kafka-generated artifacts must retain their own pipeline metadata even when
        two runs produce identical text. Source ingestion without a pipeline_id
        keeps the original global content deduplication behavior.
        """
        pipeline_id = str((metadata or {}).get("pipeline_id") or "").strip()
        doc_type = str((metadata or {}).get("type") or "").strip()
        # 문서 유형은 검색 가능성에 직접 영향을 주므로 같은 본문이라도 유형이
        # 다르면 별도 문서로 보존한다. 파이프라인 산출물은 실행 단위도 격리한다.
        scope = f"{pipeline_id}\0{doc_type}\0"
        return hashlib.md5(f"{scope}{text}".encode()).hexdigest()

    @staticmethod
    def _split_fixed(text: str, size: int) -> list[str]:
        return [text[i:i + size].strip() for i in range(0, len(text), size) if text[i:i + size].strip()]
