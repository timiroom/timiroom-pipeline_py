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
        db_semaphore: asyncio.Semaphore | None = None,
    ):
        self._db_url = db_url
        self._document_table = document_table
        self._embedder = embedder
        self._db_semaphore = db_semaphore or asyncio.Semaphore(10)
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

    async def ingest_event(self, documents: list[tuple[str, dict]]) -> int:
        """Kafka 이벤트의 모든 문서를 한 트랜잭션으로 저장한다."""
        records: list[tuple[str, dict]] = []
        for content, metadata in documents:
            texts = self._split_fixed(content, KAFKA_CHUNK_SIZE)
            records.extend(
                (text, {**metadata, "chunk_index": index})
                for index, text in enumerate(texts)
            )
        if not records:
            return 0
        embeddings = await self._embedder.embed([text for text, _ in records])
        if len(embeddings) != len(records):
            raise RuntimeError("이벤트 임베딩 개수가 청크 개수와 일치하지 않습니다")
        loop = asyncio.get_running_loop()
        async with self._db_semaphore:
            return await loop.run_in_executor(None, self._store_records_to_db, records, embeddings)

    async def _embed_and_store(self, texts: list[str], metadata: dict) -> int:
        if not texts:
            return 0
        embeddings = await self._embedder.embed(texts)
        loop = asyncio.get_running_loop()
        async with self._db_semaphore:
            return await loop.run_in_executor(None, self._store_to_db, texts, embeddings, metadata)

    def _store_to_db(self, texts: list[str], embeddings: list[list[float]], metadata: dict) -> int:
        records = [(text, {**metadata, "chunk_index": index}) for index, text in enumerate(texts)]
        return self._store_records_to_db(records, embeddings)

    def _store_records_to_db(
        self,
        records: list[tuple[str, dict]],
        embeddings: list[list[float]],
    ) -> int:
        conn = self._get_conn()
        try:
            with conn.cursor() as cur:
                for (text, metadata), vec in zip(records, embeddings, strict=True):
                    tokens_text = " ".join(token.form for token in _kiwi.tokenize(text))
                    content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
                    source_key = self._source_key(text, metadata)
                    cur.execute(
                        f"""
                        INSERT INTO {self._document_table}
                            (id, content, content_hash, source_key, metadata, embedding, tokens)
                        VALUES (%s, %s, %s, %s, %s::jsonb, %s::vector, to_tsvector('simple', %s))
                        ON CONFLICT (source_key) DO NOTHING
                        """,
                        (
                            str(uuid.uuid4()),
                            text,
                            content_hash,
                            source_key,
                            json.dumps(metadata, ensure_ascii=False),
                            str(vec),
                            tokens_text,
                        ),
                    )
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
        # 충돌은 동일 이벤트의 재처리이므로 논리적으로 저장 완료로 본다.
        return len(records)

    @staticmethod
    def _source_key(text: str, metadata: dict) -> str:
        identity = "\x1f".join([
            str(metadata.get("pipeline_id", "")),
            str(metadata.get("type", "")),
            str(metadata.get("chunk_index", "")),
            text,
        ])
        return hashlib.sha256(identity.encode("utf-8")).hexdigest()

    @staticmethod
    def _split_fixed(text: str, size: int) -> list[str]:
        return [text[i:i + size].strip() for i in range(0, len(text), size) if text[i:i + size].strip()]
