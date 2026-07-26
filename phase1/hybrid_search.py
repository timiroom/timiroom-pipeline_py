import asyncio
import logging
import uuid
from collections import defaultdict

import numpy as np
import psycopg2
import psycopg2.extras
from kiwipiepy import Kiwi
from pgvector.psycopg2 import register_vector

from common.document_chunk import DocumentChunk
from .embedding_service import EmbeddingService
from .search_rl_service import SearchParams, SearchRLService
from .session_vector_store import SessionVectorStore

logger = logging.getLogger(__name__)

RRF_K = 60
SESSION_BOOST = 1.5

# Phase1 검색 대상 타입 — ERD·API는 JSON 구조라 의미 벡터 품질이 낮고 노이즈가 됨 (rag-pipeline과 동일)
_SEARCH_TYPES = ("prd", "market_research", "features")
_SEARCH_TYPE_SQL = "AND metadata->>'type' IN ('prd', 'market_research', 'features')"

_SEARCH_TAGS = {"NNG", "NNP", "NNB", "SL", "SH"}
_kiwi = Kiwi()


class HybridSearchService:

    def __init__(
        self,
        db_url: str,
        document_table: str,
        embedder: EmbeddingService,
        session_store: SessionVectorStore,
        top_k_vector: int = 20,
        top_k_keyword: int = 20,
        similarity_threshold: float = 0.3,
        min_threshold: float = 0.1,
        min_results: int = 5,
        threshold_step: float = 0.1,
        rl_service: SearchRLService | None = None,
    ):
        self._db_url = db_url
        self._document_table = document_table
        self._embedder = embedder
        self._session_store = session_store
        self._top_k_vector = top_k_vector
        self._top_k_keyword = top_k_keyword
        self._similarity_threshold = similarity_threshold
        self._min_threshold = min_threshold
        self._min_results = min_results
        self._threshold_step = threshold_step
        self._rl_service = rl_service

    def _get_conn(self):
        conn = psycopg2.connect(self._db_url)
        register_vector(conn)
        return conn

    async def search(self, query: str, top_k: int, threshold: float | None = None) -> list[DocumentChunk]:
        """벡터·키워드 검색을 병렬 실행 후 RRF 합산."""
        loop = asyncio.get_running_loop()
        vector_results, keyword_results = await asyncio.gather(
            self._vector_search(query, threshold),
            loop.run_in_executor(None, self._keyword_search, query),
        )
        rrf_results = self._rrf(vector_results, keyword_results, top_k)
        logger.info(
            "[HybridSearch] query=%r | vector=%d | keyword=%d | rrf=%d",
            query, len(vector_results), len(keyword_results), len(rrf_results),
        )
        return rrf_results

    async def search_multiple(
        self, queries: list[str], top_k: int, threshold: float | None = None
    ) -> list[DocumentChunk]:
        """쿼리 목록을 병렬 실행하고 RRF로 크로스쿼리 합산."""
        all_results = await asyncio.gather(
            *[self.search(query, top_k, threshold) for query in queries]
        )

        scores: dict[str, float] = defaultdict(float)
        chunks: dict[str, DocumentChunk] = {}
        for ranked_list in all_results:
            for rank, chunk in enumerate(ranked_list):
                key = str(chunk.id)
                scores[key] += 1.0 / (RRF_K + rank + 1)
                chunks.setdefault(key, chunk)

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        result = [chunks[k] for k, _ in ranked]
        logger.info("[MultiQuery] 쿼리 %d개 병렬 실행 → RRF 합산 %d건", len(queries), len(result))
        return result

    async def search_with_session(
        self, queries: list[str], session_id: str
    ) -> list[DocumentChunk]:
        """RL 연동 메인 진입점 — RagPipelineService에서 호출.

        1. SearchRLService.get_params()로 epsilon-greedy 파라미터 선택
        2. 파라미터(임계값)로 Hybrid Search 수행
        3. 세션 PDF 청크가 있으면 RRF로 병합
        4. 결과를 RL 로그에 기록 → 리랭커 피드백 대기
        """
        if self._rl_service is not None:
            params = self._rl_service.get_params()
            logger.debug(
                "RL 파라미터 선택 — vw:%.2f kw:%.2f st:%.4f",
                params.vector_weight, params.keyword_weight, params.similarity_threshold,
            )
            threshold = params.similarity_threshold
        else:
            params = SearchParams(1.0, 1.0, self._similarity_threshold)
            threshold = self._similarity_threshold

        global_results = await self.search_multiple(queries, self._top_k_vector, threshold)

        if self._session_store.has_session(session_id):
            session_chunks = self._session_store.get(session_id)
            session_results = await self._session_similarity_search(queries, session_chunks)
            global_results = self._rrf(global_results, session_results, self._top_k_vector)

        if self._rl_service is not None:
            self._rl_service.log_search(session_id, params, len(global_results))

        return global_results

    async def _vector_search(self, query: str, threshold: float | None = None) -> list[DocumentChunk]:
        try:
            query_vec = await self._embedder.embed_query(query)
            loop = asyncio.get_running_loop()
            chunks = await loop.run_in_executor(None, self._vector_search_sync, query_vec, threshold)
            if chunks:
                logger.info(
                    "[Vector] %d 건 | 최고점수=%.4f | 최저점수=%.4f",
                    len(chunks), chunks[0].relevance_score, chunks[-1].relevance_score,
                )
            else:
                logger.info("[Vector] 결과 없음")
            return chunks
        except Exception as e:
            logger.warning("벡터 검색 실패: %s", e)
            return []

    def _vector_search_sync(
        self, query_vec: list[float], threshold_override: float | None = None
    ) -> list[DocumentChunk]:
        conn = self._get_conn()
        try:
            threshold = threshold_override if threshold_override is not None else self._similarity_threshold
            rows = []
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                while True:
                    cur.execute(
                        f"""
                        SELECT id, content, metadata,
                               1 - (embedding <=> %s::vector) AS score
                        FROM {self._document_table}
                        WHERE 1 - (embedding <=> %s::vector) >= %s
                          {_SEARCH_TYPE_SQL}
                        ORDER BY score DESC
                        LIMIT %s
                        """,
                        (query_vec, query_vec, threshold, self._top_k_vector),
                    )
                    rows = cur.fetchall()

                    if len(rows) >= self._min_results or threshold - self._threshold_step < self._min_threshold:
                        break
                    threshold -= self._threshold_step
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

    def _keyword_search(self, query: str) -> list[DocumentChunk]:
        tokens = [t.form for t in _kiwi.tokenize(query) if t.tag in _SEARCH_TAGS]
        if not tokens:
            tokens = [w for w in query.split() if len(w) > 1]
        ts_query = " | ".join(tokens)
        if not ts_query:
            return []

        try:
            conn = self._get_conn()
            try:
                with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                    cur.execute(
                        f"""
                        SELECT id, content, metadata,
                               ts_rank(tokens, to_tsquery('simple', %s)) AS rank
                        FROM {self._document_table}
                        WHERE tokens IS NOT NULL
                          AND tokens @@ to_tsquery('simple', %s)
                          {_SEARCH_TYPE_SQL}
                        ORDER BY rank DESC
                        LIMIT %s
                        """,
                        (ts_query, ts_query, self._top_k_keyword),
                    )
                    rows = cur.fetchall()
            finally:
                conn.close()

            chunks = [
                DocumentChunk(
                    id=uuid.UUID(str(row["id"])),
                    content=row["content"],
                    metadata=row["metadata"] or {},
                    relevance_score=float(row["rank"]),
                )
                for row in rows
            ]
            if chunks:
                logger.info(
                    "[Keyword] ts_query=%r | %d 건 | 최고rank=%.4f",
                    ts_query, len(chunks), chunks[0].relevance_score,
                )
            else:
                logger.info("[Keyword] ts_query=%r | 결과 없음", ts_query)
            return chunks
        except Exception as e:
            logger.warning("키워드 검색 실패: %s", e)
            return []

    async def _session_similarity_search(
        self,
        queries: list[str],
        chunks: list[DocumentChunk],
    ) -> list[DocumentChunk]:
        """PDF 세션 청크를 벡터 유사도로 검색. 임베딩 없으면 Kiwi 형태소 fallback."""
        if not chunks:
            return []

        if chunks[0].embedding is not None:
            try:
                query_vecs = await self._embedder.embed_queries(queries)
                avg_vec = np.mean(
                    [np.array(v, dtype=np.float32) for v in query_vecs], axis=0
                )
                results = []
                for chunk in chunks:
                    chunk_vec = np.array(chunk.embedding, dtype=np.float32)
                    denom = np.linalg.norm(avg_vec) * np.linalg.norm(chunk_vec)
                    score = float(np.dot(avg_vec, chunk_vec) / denom) * SESSION_BOOST if denom else 0.0
                    if score > 0:
                        results.append(DocumentChunk(
                            id=chunk.id,
                            content=chunk.content,
                            metadata=chunk.metadata,
                            relevance_score=score,
                        ))
                return sorted(results, key=lambda c: c.relevance_score or 0, reverse=True)
            except Exception as e:
                logger.warning("세션 벡터 유사도 실패, Kiwi fallback: %s", e)

        return self._session_similarity_kiwi(queries, chunks)

    def _session_similarity_kiwi(
        self,
        queries: list[str],
        chunks: list[DocumentChunk],
    ) -> list[DocumentChunk]:
        terms = {
            t.form
            for q in queries
            for t in _kiwi.tokenize(q)
            if t.tag in _SEARCH_TAGS
        }
        if not terms:
            return chunks

        results = []
        for chunk in chunks:
            chunk_tokens = {
                t.form for t in _kiwi.tokenize(chunk.content) if t.tag in _SEARCH_TAGS
            }
            hits = sum(1 for t in terms if t in chunk_tokens)
            score = (hits / len(terms)) * SESSION_BOOST
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
        results = [
            DocumentChunk(
                id=chunks[k].id,
                content=chunks[k].content,
                metadata=chunks[k].metadata,
                relevance_score=v,
            )
            for k, v in ranked
        ]
        if results:
            logger.info(
                "[RRF] 합산 후 %d 건 | 1위=%.5f | 꼴찌=%.5f",
                len(results), results[0].relevance_score, results[-1].relevance_score,
            )
        return results
