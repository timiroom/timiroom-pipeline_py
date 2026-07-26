import logging
import re
import uuid

import numpy as np

from common.document_chunk import DocumentChunk
from .embedding_service import EmbeddingService

logger = logging.getLogger(__name__)

SIMILARITY_THRESHOLD = 0.75


class SemanticChunkingService:
    """
    문장 간 cosine similarity가 떨어지는 의미 경계에서 분할.
    1. 문서 → 문장 단위 분리
    2. 인접 문장 쌍의 cosine similarity 계산
    3. threshold 이하인 지점을 경계로 청크 조립
    4. 최대 크기 초과 시 강제 분할
    """

    def __init__(self, embedder: EmbeddingService, max_chunk_size: int = 512, chunk_overlap: int = 64):
        self._embedder = embedder
        self._max_size = max_chunk_size
        self._overlap = chunk_overlap

    async def chunk(self, text: str, metadata: dict) -> list[DocumentChunk]:
        logger.debug("Semantic Chunking 시작 — 문서 길이: %d chars", len(text))

        sentences = self._split_sentences(text)
        if len(sentences) <= 1:
            return [self._build_chunk(text, metadata)]

        embeddings = await self._embed_sentences(sentences)
        boundaries = self._detect_boundaries(embeddings)
        chunks = self._assemble_chunks(sentences, boundaries, metadata)

        logger.debug("Semantic Chunking 완료 — %d chunks 생성", len(chunks))
        return chunks

    def _split_sentences(self, text: str) -> list[str]:
        parts = re.split(r"(?<=[.!?])\s+|\n{2,}", text)
        return [s.strip() for s in parts if s.strip()]

    async def _embed_sentences(self, sentences: list[str]) -> list[np.ndarray]:
        vecs = await self._embedder.embed(sentences)
        return [np.array(v, dtype=np.float32) for v in vecs]

    def _detect_boundaries(self, embeddings: list[np.ndarray]) -> list[int]:
        boundaries = [0]
        for i in range(len(embeddings) - 1):
            sim = self._cosine(embeddings[i], embeddings[i + 1])
            if sim < SIMILARITY_THRESHOLD:
                boundaries.append(i + 1)
                logger.debug("경계 감지 — 문장 %d (similarity: %.3f)", i + 1, sim)
        boundaries.append(len(embeddings))
        return boundaries

    def _assemble_chunks(
        self,
        sentences: list[str],
        boundaries: list[int],
        metadata: dict,
    ) -> list[DocumentChunk]:
        chunks: list[DocumentChunk] = []
        for i in range(len(boundaries) - 1):
            segment = " ".join(sentences[boundaries[i]:boundaries[i + 1]])
            if len(segment) > self._max_size:
                chunks.extend(self._split_by_size(segment, metadata))
            else:
                chunks.append(self._build_chunk(segment, metadata))
        return chunks

    def _split_by_size(self, text: str, metadata: dict) -> list[DocumentChunk]:
        result = []
        start = 0
        while start < len(text):
            end = min(start + self._max_size, len(text))
            result.append(self._build_chunk(text[start:end], metadata))
            start = end - self._overlap
            if start < 0:
                start = 0
        return result

    def _build_chunk(self, content: str, metadata: dict) -> DocumentChunk:
        return DocumentChunk(id=uuid.uuid4(), content=content, metadata=dict(metadata))

    @staticmethod
    def _cosine(a: np.ndarray, b: np.ndarray) -> float:
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        return float(np.dot(a, b) / denom) if denom else 0.0
