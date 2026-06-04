from fastapi import APIRouter
from pydantic import BaseModel

from common.api_response import ok

router = APIRouter(prefix="/api/v1/rag", tags=["rag"])


class IngestRequest(BaseModel):
    content: str
    source: str | None = None


class ContextRequest(BaseModel):
    query: str


@router.post("/ingest")
async def ingest(req: IngestRequest) -> dict:
    from main import document_ingestion_service
    source = req.source or "unknown"
    await document_ingestion_service.ingest(req.content, {"source": source})
    return ok({"source": source, "message": "문서가 성공적으로 저장되었습니다"})


@router.post("/context")
async def build_context(req: ContextRequest) -> dict:
    from main import rag_pipeline_service
    prompt = await rag_pipeline_service.build_context(req.query)
    return ok({"query": req.query, "prompt": prompt})
