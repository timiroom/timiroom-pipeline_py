import json
import logging
import uuid

import psycopg2
import psycopg2.extras
from kiwipiepy import Kiwi
from openai import AsyncOpenAI
from pgvector.psycopg2 import register_vector

from .semantic_chunking import SemanticChunkingService

logger = logging.getLogger(__name__)

_kiwi = Kiwi()

EMBED_MODEL = "text-embedding-3-large"
# Kafka consumer path용 고정 크기 청크 (SemanticChunking 미사용)
KAFKA_CHUNK_SIZE = 800


class DocumentIngestionService:

    def __init__(
        self,
        db_url: str,
        client: AsyncOpenAI,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        embed_model: str = "text-embedding-3-large",
    ):
        self._db_url = db_url
        self._client = client
        self._embed_model = embed_model
        self._semantic_chunker = SemanticChunkingService(
            client, max_chunk_size=chunk_size, chunk_overlap=chunk_overlap, embed_model=embed_model
        )

    def _get_conn(self):
        conn = psycopg2.connect(self._db_url)
        register_vector(conn)
        return conn

    async def ingest(self, content: str, metadata: dict) -> int:
        """
        Semantic Chunking → 임베딩 → pgvector 저장.
        /api/v1/rag/ingest 엔드포인트 전용.
        """
        logger.info("문서 수집 시작 — source: %s", metadata.get("source", "unknown"))
        chunks = await self._semantic_chunker.chunk(content, metadata)
        texts = [c.content for c in chunks]
        saved = await self._embed_and_store(texts, metadata)
        logger.info("문서 수집 완료 — %d chunks 저장됨", saved)
        return saved

    async def ingest_fixed(self, content: str, metadata: dict) -> int:
        """
        고정 크기 청크 → 임베딩 → pgvector 저장.
        Kafka Consumer path 전용 (Java KafkaConsumerService와 동일).
        """
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

        resp = await self._client.embeddings.create(model=self._embed_model, input=texts)
        embeddings = [item.embedding for item in resp.data]

        conn = self._get_conn()
        saved = 0
        try:
            with conn.cursor() as cur:
                for i, (text, vec) in enumerate(zip(texts, embeddings)):
                    meta = {**metadata, "chunk_index": i}
                    tokens_text = " ".join(t.form for t in _kiwi.tokenize(text))
                    cur.execute(
                        """
                        INSERT INTO document_chunks (id, content, metadata, embedding, tokens)
                        VALUES (%s, %s, %s::jsonb, %s::vector, to_tsvector('simple', %s))
                        """,
                        (
                            str(uuid.uuid4()),
                            text,
                            json.dumps(meta, ensure_ascii=False),
                            str(vec),
                            tokens_text,
                        ),
                    )
                    saved += 1
            conn.commit()
        except Exception as e:
            conn.rollback()
            logger.error("청크 저장 실패: %s", e)
            raise
        finally:
            conn.close()

        return saved

    @staticmethod
    def _split_fixed(text: str, size: int) -> list[str]:
        return [text[i:i + size].strip() for i in range(0, len(text), size) if text[i:i + size].strip()]
