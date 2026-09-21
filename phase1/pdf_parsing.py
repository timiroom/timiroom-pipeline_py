import asyncio
import io
import logging
from dataclasses import dataclass

import pdfplumber

from common.document_chunk import DocumentChunk
from .embedding_service import EmbeddingService
from .semantic_chunking import SemanticChunkingService
from .session_vector_store import SessionVectorStore

logger = logging.getLogger(__name__)

MIN_TEXT_LENGTH = 100
MAX_FILES = 5


@dataclass(frozen=True)
class PdfParseResult:
    total_files: int
    processed_files: int
    failed_files: int
    chunk_count: int


class PDFParsingService:

    def __init__(
        self,
        session_store: SessionVectorStore,
        embedder: EmbeddingService,
        chunker: SemanticChunkingService,
    ):
        self._session_store = session_store
        self._embedder = embedder
        self._chunker = chunker

    async def parse_and_store_all(self, pdf_bytes_list: list[tuple[str, bytes]], session_id: str) -> PdfParseResult:
        if not pdf_bytes_list:
            return PdfParseResult(0, 0, 0, 0)

        if len(pdf_bytes_list) > MAX_FILES:
            logger.warning(
                "업로드된 PDF %d개가 최대 %d개를 초과 — 앞 %d개만 처리",
                len(pdf_bytes_list), MAX_FILES, MAX_FILES,
            )
            pdf_bytes_list = pdf_bytes_list[:MAX_FILES]

        loop = asyncio.get_running_loop()
        chunks: list[DocumentChunk] = []
        processed_files = 0
        failed_files = 0
        for filename, data in pdf_bytes_list:
            try:
                text = await loop.run_in_executor(None, self._extract_text, filename, data)
                if text is None:
                    continue
                parsed = await self._chunker.chunk(text, {"source": filename})
                chunks.extend(parsed)
                processed_files += 1
                logger.info("PDF 파싱 완료: %s — %d 청크", filename, len(parsed))
            except Exception as e:
                failed_files += 1
                logger.warning("PDF 파싱 실패: %s — %s", filename, e)

        if not chunks:
            return PdfParseResult(len(pdf_bytes_list), processed_files, failed_files, 0)

        try:
            texts = [c.content for c in chunks]
            embeddings = await self._embedder.embed(texts)
            for chunk, emb in zip(chunks, embeddings):
                chunk.embedding = emb
            logger.info("PDF 임베딩 완료 — %d 청크", len(chunks))
        except Exception as e:
            logger.warning("PDF 임베딩 실패, 임베딩 없이 저장: %s", e)

        self._session_store.put(session_id, chunks)
        logger.info("세션 벡터스토어 저장 완료 — %d 청크", len(chunks))
        return PdfParseResult(len(pdf_bytes_list), processed_files, failed_files, len(chunks))

    def _extract_text(self, filename: str, data: bytes) -> str | None:
        with pdfplumber.open(io.BytesIO(data)) as pdf:
            full_text = "\n".join(
                page.extract_text() or "" for page in pdf.pages
            ).strip()

        if len(full_text) < MIN_TEXT_LENGTH:
            logger.warning(
                "PDF 텍스트가 너무 짧음 (스캔본 추정) — %s — %d chars, 건너뜀",
                filename, len(full_text),
            )
            return None

        return full_text
