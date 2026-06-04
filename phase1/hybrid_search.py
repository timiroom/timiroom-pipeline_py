import logging
import uuid
from collections import defaultdict

import psycopg2
import psycopg2.extras
from openai import AsyncOpenAI
from pgvector.psycopg2 import register_vector

from common.document_chunk import DocumentChunk
from .session_vector_store import SessionVectorStore

logger = logging.getLogger(__name__)

RRF_K = 60
SESSION_BOOST = 1.5


class HybridSearchService:

    def __init__(
        self,
        db_url: str,
        client: AsyncOpenAI,
        session_store: SessionVectorStore,
        top_k_vector: int = 20,
        top_k_keyword: int = 20,
        embed_model: str = "text-embedding-3-large",
    ):
        self._db_url = db_url
        self._client = client
        self._session_store = session_store
        self._top_k_vector = top_k_vector
        self._top_k_keyword = top_k_keyword
        self._embed_model = embed_model

    def _get_conn(self):
        conn = psycopg2.connect(self._db_url)
        register_vector(conn)
        return conn

    async def search(self, query: str, top_k: int) -> list[DocumentChunk]:
        vector_results = await self._vector_search(query)
        keyword_results = self._keyword_search(query)
        return self._rrf(vector_results, keyword_results, top_k)

    async def search_multiple(self, queries: list[str], top_k: int) -> list[DocumentChunk]:
        seen: dict[str, DocumentChunk] = {}
        for query in queries:
            for chunk in await self.search(query, top_k):
                seen.setdefault(str(chunk.id), chunk)
        all_chunks = list(seen.values())
        return all_chunks[:top_k]

    async def search_with_session(
        self, queries: list[str], session_id: str
    ) -> list[DocumentChunk]:
        global_results = await self.search_multiple(queries, self._top_k_vector)

        if self._session_store.has_session(session_id):
            session_chunks = self._session_store.get(session_id)
            session_results = self._similarity_search(queries, session_chunks, SESSION_BOOST)
            return self._rrf(global_results, session_results, self._top_k_vector)

        return global_results

    async def _vector_search(self, query: str) -> list[DocumentChunk]:
        try:
            resp = await self._client.embeddings.create(
                model=self._embed_model, input=query
            )
            query_vec = resp.data[0].embedding

            conn = self._get_conn()
            try:
                with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                    cur.execute(
                        """
                        SELECT id, content, metadata,
                               1 - (embedding <=> %s::vector) AS score
                        FROM document_chunks
                        WHERE 1 - (embedding <=> %s::vector) >= 0.5
                        ORDER BY score DESC
                        LIMIT %s
                        """,
                        (query_vec, query_vec, self._top_k_vector),
                    )
                    rows = cur.fetchall()
            finally:
                conn.close()

            return [
                DocumentChunk(
                    id=uuid.UUID(str(row["id"])),
                    content=row["content"],
                    metadata=row["metadata"] or {},
                    relevance_score=float(row["score"]),
                )
                for row in rows
            ]
        except Exception as e:
            logger.warning("벡터 검색 실패: %s", e)
            return []

    def _keyword_search(self, query: str) -> list[DocumentChunk]:
        cleaned = "".join(
            c if c.isalnum() or c.isspace() else " " for c in query
        ).strip()
        if any("가" <= c <= "힣" for c in cleaned):
            return []

        ts_query = " & ".join(w for w in cleaned.split() if w)
        if not ts_query:
            return []

        try:
            conn = self._get_conn()
            try:
                with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                    cur.execute(
                        """
                        SELECT id, content, metadata,
                               ts_rank(to_tsvector('english', content),
                                       to_tsquery('english', %s)) AS rank
                        FROM document_chunks
                        WHERE to_tsvector('english', content) @@ to_tsquery('english', %s)
                        ORDER BY rank DESC
                        LIMIT %s
                        """,
                        (ts_query, ts_query, self._top_k_keyword),
                    )
                    rows = cur.fetchall()
            finally:
                conn.close()

            return [
                DocumentChunk(
                    id=uuid.UUID(str(row["id"])),
                    content=row["content"],
                    metadata=row["metadata"] or {},
                    relevance_score=float(row["rank"]),
                )
                for row in rows
            ]
        except Exception as e:
            logger.warning("키워드 검색 실패: %s", e)
            return []

    def _similarity_search(
        self,
        queries: list[str],
        chunks: list[DocumentChunk],
        boost: float,
    ) -> list[DocumentChunk]:
        if not chunks:
            return []
        terms = {
            t.lower()
            for q in queries
            for t in q.split()
            if len(t) > 1
        }
        if not terms:
            return chunks

        results = []
        for chunk in chunks:
            lower = chunk.content.lower()
            hits = sum(1 for t in terms if t in lower)
            score = (hits / len(terms)) * boost
            if score > 0:
                results.append(DocumentChunk(
                    id=chunk.id,
                    content=chunk.content,
                    metadata=chunk.metadata,
                    relevance_score=score,
                ))
        return sorted(results, key=lambda c: c.relevance_score or 0, reverse=True)

    def _rrf(
        self,
        list_a: list[DocumentChunk],
        list_b: list[DocumentChunk],
        top_k: int,
    ) -> list[DocumentChunk]:
        scores: dict[str, float] = defaultdict(float)
        chunks: dict[str, DocumentChunk] = {}

        for i, chunk in enumerate(list_a):
            key = str(chunk.id)
            scores[key] += 1.0 / (RRF_K + i + 1)
            chunks.setdefault(key, chunk)

        for i, chunk in enumerate(list_b):
            key = str(chunk.id)
            scores[key] += 1.0 / (RRF_K + i + 1)
            chunks.setdefault(key, chunk)

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return [
            DocumentChunk(
                id=chunks[k].id,
                content=chunks[k].content,
                metadata=chunks[k].metadata,
                relevance_score=v,
            )
            for k, v in ranked
        ]
