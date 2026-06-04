import io
import logging
import uuid

import pdfplumber

from common.document_chunk import DocumentChunk
from .session_vector_store import SessionVectorStore

logger = logging.getLogger(__name__)

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


class PDFParsingService:

    def __init__(self, session_store: SessionVectorStore):
        self._session_store = session_store

    def parse_and_store_all(self, pdf_bytes_list: list[tuple[str, bytes]], session_id: str) -> None:
        if not pdf_bytes_list:
            return

        chunks: list[DocumentChunk] = []
        for filename, data in pdf_bytes_list:
            try:
                parsed = self._parse(filename, data)
                chunks.extend(parsed)
                logger.info("PDF 파싱 완료: %s — %d 청크", filename, len(parsed))
            except Exception as e:
                logger.warning("PDF 파싱 실패: %s — %s", filename, e)

        if chunks:
            self._session_store.put(session_id, chunks)
            logger.info("세션 벡터스토어 저장 완료 — %d 청크", len(chunks))

    def _parse(self, filename: str, data: bytes) -> list[DocumentChunk]:
        chunks = []
        with pdfplumber.open(io.BytesIO(data)) as pdf:
            full_text = "\n".join(
                page.extract_text() or "" for page in pdf.pages
            ).strip()

        if not full_text:
            return chunks

        start = 0
        while start < len(full_text):
            end = min(start + CHUNK_SIZE, len(full_text))
            text = full_text[start:end].strip()
            if text:
                chunks.append(DocumentChunk(
                    id=uuid.uuid4(),
                    content=text,
                    metadata={"source": filename},
                ))
            start += CHUNK_SIZE - CHUNK_OVERLAP

        return chunks
