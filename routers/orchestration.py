import asyncio
import json
import logging
import uuid
from typing import Annotated

from fastapi import APIRouter, Form, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse

from common.api_response import ok
from phase1.models import FormData
from phase1.rag_pipeline import RagPipelineService
from phase2.orchestration_graph import OrchestrationGraph
from phase2.sse_service import PipelineProgressService
from phase3.retry_service import RetryService
from phase3.validation_service import ValidationService
from phase4.kafka_producer import KafkaProducerService

logger = logging.getLogger(__name__)

MAX_FILE_SIZE = 20 * 1024 * 1024  # 20 MB

router = APIRouter(prefix="/api/v1/orchestration", tags=["orchestration"])


def _services():
    from main import (
        rag_pipeline_service, orchestration_graph, validation_service,
        retry_service, kafka_producer_service, progress_service,
    )
    return (
        rag_pipeline_service, orchestration_graph, validation_service,
        retry_service, kafka_producer_service, progress_service,
    )


@router.post("/generate")
async def generate(
    request: Annotated[str, Form()],
    files: list[UploadFile] = File(default=[]),
):
    """
    파이프라인 시작 — 즉시 pipelineId 반환, 파이프라인은 백그라운드 실행.
    클라이언트는 반환된 pipelineId로 GET /progress/{pipelineId} 구독.
    """
    try:
        form_data = FormData.model_validate_json(request)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"요청 형식 오류: {e}")

    _validate_form(form_data)

    # 파일 읽기 (요청 스레드에서 처리)
    pdf_bytes: list[tuple[str, bytes]] = []
    for file in files:
        data = await file.read()
        if len(data) > MAX_FILE_SIZE:
            raise HTTPException(400, f"PDF 파일 크기 초과 (최대 20MB): {file.filename}")
        pdf_bytes.append((file.filename or "unknown.pdf", data))

    pipeline_id = str(uuid.uuid4())
    logger.info("파이프라인 시작 | pipelineId: %s, project: %s",
                pipeline_id[:8], form_data.project_name)

    (rag_svc, orch, val_svc, retry_svc, kafka_svc, progress_svc) = _services()

    asyncio.create_task(
        _run_pipeline(pipeline_id, form_data, pdf_bytes,
                      rag_svc, orch, val_svc, retry_svc, kafka_svc, progress_svc)
    )

    return ok({"pipelineId": pipeline_id})


@router.get("/progress/{pipeline_id}")
async def progress(pipeline_id: str):
    """SSE 구독 — 파이프라인 진행 상황 실시간 수신."""
    _, _, _, _, _, progress_svc = _services()

    async def event_generator():
        try:
            async for chunk in progress_svc.subscribe(pipeline_id):
                yield chunk
        except asyncio.TimeoutError:
            yield "event: error\ndata: {\"message\": \"타임아웃\"}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


async def _run_pipeline(
    pipeline_id: str,
    form_data: FormData,
    pdf_bytes: list[tuple[str, bytes]],
    rag_svc: RagPipelineService,
    orch: OrchestrationGraph,
    val_svc: ValidationService,
    retry_svc: RetryService,
    kafka_svc: KafkaProducerService,
    progress_svc: PipelineProgressService,
) -> None:
    try:
        # Phase 1
        progress_svc.send(pipeline_id, "PHASE1_START", "PDF 파싱 및 검색 준비 중...", 5)
        state = await rag_svc.build_from_form(form_data, pdf_bytes)
        progress_svc.send(pipeline_id, "PHASE1_DONE", "검색 컨텍스트 구성 완료", 25)

        # Phase 2
        phase2_state = await orch.run(state, pipeline_id)
        logger.debug("Phase 2 완료 | retryCount: %d", phase2_state.retry_count)

        # Phase 3
        progress_svc.send(pipeline_id, "PHASE3", "결과 검증 중...", 85)
        validated = val_svc.validate(phase2_state)
        if not validated.validated:
            logger.warning("Phase 3 검증 실패 — RetryService 진입")
            validated = await retry_svc.retry_with(validated, orch, val_svc)

        if not validated.validated:
            logger.warning("HITL 요청 | 자동 검증 실패")
            progress_svc.error(pipeline_id, "자동 검증 실패 — 관리자 검토가 필요합니다")
            return

        # Phase 4
        progress_svc.send(pipeline_id, "PHASE4", "결과 저장 중...", 95)
        await kafka_svc.publish(validated)
        logger.info("파이프라인 완료 | pipelineId: %s", pipeline_id)

        result = {
            "projectName": validated.project_name,
            "featureList": validated.feature_list,
            "prdDocument": _parse_json(validated.prd_document),
            "dbSchema": _parse_json(validated.db_schema),
            "apiSpec": _parse_json(validated.api_spec),
            "marketResearch": validated.market_research,
            "status": validated.status_message,
            "retryCount": validated.retry_count,
        }
        progress_svc.complete(pipeline_id, result)

    except Exception as e:
        logger.error("파이프라인 실패 | %s", e, exc_info=True)
        progress_svc.error(pipeline_id, str(e) or "알 수 없는 오류")


def _parse_json(value: str | None) -> dict | list:
    """JSON 문자열 → dict/list. 파싱 실패 시 빈 dict 반환."""
    if not value or not value.strip():
        return {}
    try:
        return json.loads(value)
    except Exception:
        return {}


def _validate_form(form: FormData) -> None:
    if not form.project_name or not form.project_name.strip():
        raise HTTPException(400, "프로젝트 이름은 필수입니다")
    if not form.problem_definition:
        raise HTTPException(400, "문제정의는 필수입니다")
    if not form.target_users:
        raise HTTPException(400, "타겟유저는 최소 1명 필요합니다")
    if not form.feature_definition:
        raise HTTPException(400, "기능정의는 필수입니다")
