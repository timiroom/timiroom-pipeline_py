from fastapi import APIRouter
from pydantic import BaseModel

from common.api_response import ok

router = APIRouter(prefix="/api/v1/rag", tags=["rag"])


class IngestRequest(BaseModel):
    content: str
    source: str | None = None


@router.post("/ingest")
async def ingest(req: IngestRequest) -> dict:
    from main import document_ingestion_service
    source = req.source or "unknown"
    saved = await document_ingestion_service.ingest(
        req.content,
        {"source": source, "type": "source"},
    )
    return ok({
        "source": source,
        "savedChunks": saved,
        "message": "문서가 성공적으로 저장되었습니다",
    })
