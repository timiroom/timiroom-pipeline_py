import asyncio
import json
import logging
import uuid
from typing import Annotated

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

from common.ai_errors import describe
from common.api_response import ok
from common.logging_middleware import pipeline_id_var
from phase1.form_to_query import FormToQueryService
from phase1.models import FormData
from phase1.pdf_parsing import MAX_FILES
from phase1.rag_pipeline import RagPipelineService
from phase2.execution_error import state_from_error
from phase2.orchestration_graph import OrchestrationGraph
from phase2.sse_service import PipelineProgressService
from phase2.state import PipelineState
from phase3.retry_service import RetryService
from phase3.validation_service import ValidationService
from phase4.kafka_producer import KafkaProducerService

logger = logging.getLogger(__name__)

MAX_FILE_SIZE = 20 * 1024 * 1024  # 20 MB
MAX_CONCURRENT_PIPELINES = 10  # Java AsyncConfig.pipelineExecutor(core=4, max=10)에 대응

router = APIRouter(prefix="/api/v1/orchestration", tags=["orchestration"])
_pipeline_semaphore = asyncio.Semaphore(MAX_CONCURRENT_PIPELINES)
_pipeline_tasks: set[asyncio.Task] = set()


def _services():
    from main import (
        form_to_query,
        kafka_producer_service,
        orchestration_graph,
        progress_service,
        rag_pipeline_service,
        retry_service,
        validation_service,
    )
    return (
        rag_pipeline_service, orchestration_graph, validation_service,
        retry_service, kafka_producer_service, progress_service, form_to_query,
    )


@router.post("/generate")
async def generate(
    request: Annotated[str, Form()],
    files: list[UploadFile] = File(default=[]),
    skip_phase1: Annotated[bool, Form()] = False,
):
    """
    파이프라인 시작 — 즉시 pipelineId 반환, 파이프라인은 백그라운드 실행.
    클라이언트는 반환된 pipelineId로 GET /progress/{pipelineId} 구독.
    """
    try:
        form_data = FormData.model_validate_json(request)
    except Exception as e:
        logger.error(
            "FormData 검증 실패: %s | 수신 JSON 앞 600자: %.600s",
            e, request,
        )
        raise HTTPException(status_code=400, detail=f"요청 형식 오류: {e}")

    _validate_form(form_data)

    # 파일 읽기 (요청 스레드에서 처리)
    _validate_file_count(len(files))

    pdf_bytes: list[tuple[str, bytes]] = []
    for file in files:
        data = await file.read()
        if len(data) > MAX_FILE_SIZE:
            raise HTTPException(400, f"PDF 파일 크기 초과 (최대 20MB): {file.filename}")
        pdf_bytes.append((file.filename or "unknown.pdf", data))

    pipeline_id = str(uuid.uuid4())
    logger.info("파이프라인 시작 | pipelineId: %s, project: %s",
                pipeline_id[:8], form_data.project_name)

    (rag_svc, orch, val_svc, retry_svc, kafka_svc, progress_svc, form_to_query_svc) = _services()

    progress_svc.send(pipeline_id, "QUEUED", "파이프라인 실행 대기 중...", 0)
    task = asyncio.create_task(
        _run_pipeline(pipeline_id, form_data, pdf_bytes,
                      rag_svc, orch, val_svc, retry_svc, kafka_svc, progress_svc,
                      form_to_query_svc, skip_phase1=skip_phase1)
    )
    _pipeline_tasks.add(task)
    task.add_done_callback(_pipeline_tasks.discard)

    return ok({"pipelineId": pipeline_id})


@router.get("/progress/{pipeline_id}")
async def progress(pipeline_id: str):
    """SSE 구독 — 파이프라인 진행 상황 실시간 수신."""
    _, _, _, _, _, progress_svc, _ = _services()

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


@router.get("/storage/{pipeline_id}")
async def storage_status(pipeline_id: uuid.UUID):
    _, _, _, _, kafka_svc, _, _ = _services()
    return ok(await kafka_svc.storage_status(str(pipeline_id)))


@router.get("/state/{pipeline_id}")
async def execution_state(pipeline_id: uuid.UUID):
    _, _, _, _, _, progress_svc, _ = _services()
    return ok(progress_svc.snapshot(str(pipeline_id)))


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
    form_to_query_svc: FormToQueryService,
    skip_phase1: bool = False,
) -> None:
    token = pipeline_id_var.set(pipeline_id[:8])
    try:
        async with _pipeline_semaphore:
            await _run_pipeline_inner(
                pipeline_id, form_data, pdf_bytes,
                rag_svc, orch, val_svc, retry_svc, kafka_svc, progress_svc,
                form_to_query_svc, skip_phase1,
            )
    finally:
        pipeline_id_var.reset(token)


async def _run_pipeline_inner(
    pipeline_id: str,
    form_data: FormData,
    pdf_bytes: list[tuple[str, bytes]],
    rag_svc: RagPipelineService,
    orch: OrchestrationGraph,
    val_svc: ValidationService,
    retry_svc: RetryService,
    kafka_svc: KafkaProducerService,
    progress_svc: PipelineProgressService,
    form_to_query_svc: FormToQueryService,
    skip_phase1: bool = False,
) -> None:
    latest_state: PipelineState | None = None
    try:
        # Phase 1
        if skip_phase1:
            progress_svc.send(pipeline_id, "PHASE1_SKIP", "Phase 1 건너뜀 (테스트 모드)", 25)
            user_query = form_to_query_svc.synthesize(form_data)
            state = PipelineState(
                project_name=form_data.project_name,
                platform=form_data.platform,
                tech_stack=form_data.tech_stack or [],
                problem_definition=form_data.problem_definition,
                target_users=form_data.target_users,
                must_features=form_to_query_svc.extract_must_features(form_data),
                supporting_feature_names=form_to_query_svc.extract_supporting_features(form_data),
                should_features=form_to_query_svc.extract_should_features(form_data),
                could_features=form_to_query_svc.extract_could_features(form_data),
                excluded_features=form_to_query_svc.extract_excluded_features(form_data),
                feature_list=form_to_query_svc.extract_all_included_features(form_data),
                user_query=user_query,
                context_prompt=user_query,
                status_message="Phase 1 건너뜀",
            )
        else:
            progress_svc.send(pipeline_id, "PHASE1_START", "PDF 파싱 및 검색 준비 중...", 5)
            state = await rag_svc.build_from_form(form_data, pdf_bytes)
            progress_svc.send(pipeline_id, "PHASE1_DONE", "검색 컨텍스트 구성 완료", 25)
        latest_state = state

        # Phase 2
        phase2_state = await orch.run(state, pipeline_id)
        latest_state = phase2_state
        logger.debug("Phase 2 완료 | retryCount: %d", phase2_state.retry_count)

        # Phase 3
        if not phase2_state.failed_stage:
            progress_svc.send(pipeline_id, "PHASE3", "결과 검증 중...", 85)
        validated = val_svc.validate(phase2_state)
        latest_state = validated
        if not validated.validated:
            logger.warning("Phase 3 검증 실패 — RetryService 진입")
            validated = await retry_svc.retry_with(validated, orch, val_svc, pipeline_id=pipeline_id)
            latest_state = validated

        if not validated.validated:
            logger.warning("HITL 요청 | 자동 검증 실패")
            progress_svc.error(
                pipeline_id,
                "자동 검증 실패 — 관리자 검토가 필요합니다",
                details={
                    "validationErrors": validated.validation_error_codes,
                    "validationMessage": validated.last_validation_error,
                    "validationBlockers": validated.validation_unresolved_blockers,
                    "resolvedBlockers": validated.validation_resolved_blockers,
                    "validationBlockerComparison": {
                        "resolved": validated.validation_resolved_blockers,
                        "unresolved": validated.validation_unresolved_blockers,
                    },
                    "blockerDetails": validated.qa_blocker_details,
                    "qa": {
                        "approved": validated.qa_approved,
                        "blockingIssues": {
                            "db": validated.qa_db_blockers,
                            "api": validated.qa_api_blockers,
                            "prd": validated.qa_prd_blockers,
                        },
                        "warnings": {
                            "db": validated.qa_db_warnings,
                            "api": validated.qa_api_warnings,
                            "prd": validated.qa_prd_warnings,
                        },
                    },
                    "phaseReached": validated.failed_stage or "PHASE3",
                    "validationStage": "PHASE3",
                    **_stage_details(validated),
                    "failedStage": validated.failed_stage or "PHASE3",
                    "partialResult": _artifact_snapshot(validated),
                    "usage": _usage_summary(),
                },
            )
            return

        # Phase 4
        progress_svc.send(pipeline_id, "PHASE4", "결과 저장 중...", 95)
        from config.settings import settings
        if not settings.kafka_enabled:
            progress_svc.error(pipeline_id, "생성은 완료됐지만 Kafka가 비활성화되어 결과를 저장하지 않았습니다.", details={
                "failedStage": "PHASE4", "lastCompletedStage": "PHASE3", "skippedStages": [],
                "storage": {"status": "skipped", "kafkaPublished": False, "reason": "KAFKA_DISABLED"},
                "retrySafe": True, "partialResult": _artifact_snapshot(validated),
            })
            return
        published_pipeline_id = await kafka_svc.publish(validated, pipeline_id)
        if published_pipeline_id != pipeline_id:
            raise RuntimeError("Kafka 발행 pipelineId가 요청 pipelineId와 일치하지 않습니다")
        progress_svc.send(pipeline_id, "STORAGE_CONFIRMATION", "Kafka 발행 완료 · DB 저장 확인 중...", 97)
        try:
            storage = await kafka_svc.confirm_storage(validated, pipeline_id)
        except Exception as exc:
            logger.warning("DB 저장 확인 실패 — 발행은 완료됨: %s", type(exc).__name__)
            storage = {"status": "pending", "kafkaPublished": True, "reason": "STORAGE_VERIFICATION_UNAVAILABLE"}
        if storage.get("status") != "stored":
            progress_svc.error(pipeline_id, "Kafka 발행은 완료됐지만 DB 저장 확인이 대기 중입니다. 재생성하지 마세요.", details={
                "failedStage": "STORAGE_CONFIRMATION", "lastCompletedStage": "KAFKA_PUBLISHED",
                "skippedStages": [], "storage": storage, "retrySafe": False,
                "partialResult": _artifact_snapshot(validated),
            })
            return
        logger.info("파이프라인 완료 | pipelineId: %s", pipeline_id)

        feature_spec = _parse_json(validated.feature_spec_document)
        summary_features = [item for item in validated.feature_registry if isinstance(item, dict)]
        core_count = sum(1 for item in summary_features if str(item.get("priority") or "").upper() == "P0")
        supporting_count = len(summary_features) - core_count
        result = {
            "projectName": validated.project_name,
            "storage": storage,
            "completionStatus": "STORED",
            "artifactQuality": validated.artifact_quality,
            "generationDiagnostics": validated.generation_diagnostics,
            # 내부 Phase/ Kafka 계약은 문자열 feature_list를 유지하지만,
            # 저장되는 FEATURE_LIST 아티팩트는 Feature Spec 상세 객체를
            # 우선 사용해야 프론트가 설명·요구사항·계약을 잃지 않는다.
            "featureNames": validated.feature_list,
            "featureList": validated.feature_registry,
            "projectPlan": validated.project_plan,
            "featureSpecDocument": feature_spec,
            "featureRegistry": validated.feature_registry,
            "allFeatures": [
                item.get("name") for item in validated.feature_registry
                if isinstance(item, dict) and item.get("name")
            ],
            "featureSummary": {
                "coreCount": core_count,
                "supportingCount": supporting_count,
                "totalCount": len(summary_features),
            },
            "prdDocument": _parse_json(validated.prd_document),
            "dbSchema": _parse_json(validated.db_schema),
            "apiSpec": _parse_json(validated.api_spec),
            "openApiSpec": _parse_json(validated.openapi_spec),
            "marketResearch": validated.market_research,
            "status": validated.status_message,
            "qa": {
                "approved": validated.qa_approved,
                "blockingIssues": {
                    "db": validated.qa_db_blockers,
                    "api": validated.qa_api_blockers,
                    "prd": validated.qa_prd_blockers,
                },
                "warnings": {
                    "db": validated.qa_db_warnings,
                    "api": validated.qa_api_warnings,
                    "prd": validated.qa_prd_warnings,
                },
                "blockerDetails": validated.qa_blocker_details,
            },
            "retryCount": validated.retry_count,
            "validationBlockers": validated.validation_unresolved_blockers,
            "resolvedBlockers": validated.validation_resolved_blockers,
            "validationBlockerComparison": {
                "resolved": validated.validation_resolved_blockers,
                "unresolved": validated.validation_unresolved_blockers,
            },
            "pdfFilesTotal": validated.pdf_files_total,
            "pdfFilesProcessed": validated.pdf_files_processed,
            "pdfFilesFailed": validated.pdf_files_failed,
            "rag": {
                "chunksRetrieved": validated.rag_chunks_retrieved,
                "usedFormDataOnly": validated.rag_context_fallback,
            },
        }
        _attach_usage(result)
        progress_svc.complete(pipeline_id, result)

    except Exception as e:
        logger.error("파이프라인 실패 | %s", e, exc_info=True)
        # A stage that raised carries the artifacts finished before it. Without
        # this the snapshot falls back to the pre-Phase2 state and a completed
        # PRD, project plan or DB schema disappears from the failure report.
        latest_state = state_from_error(e, latest_state)
        error_type = type(e).__name__
        ai = describe(e)
        # AI 제공사 오류는 원문(영어, 키 일부가 섞일 수 있음) 대신 한국어 안내와 코드를 보낸다
        error_message = ai.message if ai else (str(e).strip() or error_type)
        progress_svc.error(
            pipeline_id,
            error_message,
            details={
                "errorType": error_type,
                "errorCode": ai.code if ai else None,
                "errorDetail": None if ai else repr(e),
                **_stage_details(latest_state),
                "partialResult": _artifact_snapshot(latest_state),
                "usage": _usage_summary(),
            },
        )


def _usage_summary() -> list[dict]:
    """이 파이프라인에서 쓴 AI 사용량. 백엔드가 사용량 추적을 요청하지 않았으면 빈 목록."""
    from common.usage import current_collector
    collector = current_collector.get()
    return collector.summary() if collector is not None else []


def _attach_usage(result: dict) -> None:
    result["usage"] = _usage_summary()


def _parse_json(value: str | None) -> dict | list:
    """JSON 문자열 → dict/list. 파싱 실패 시 빈 dict 반환."""
    if not value or not value.strip():
        return {}
    try:
        return json.loads(value)
    except Exception:
        return {}


def _artifact_snapshot(state: PipelineState | None) -> dict:
    """Return inspectable, non-secret artifacts even when Phase4 is not reached."""
    if state is None:
        return {}
    return {
        "projectName": state.project_name,
        "projectPlan": state.project_plan,
        "featureNames": state.feature_list,
        "featureList": state.feature_registry,
        "retryCount": state.retry_count,
        "featureRegistry": state.feature_registry,
        "featureRegistryHash": state.feature_registry_hash,
        "featureSpecDocument": _parse_json(state.feature_spec_document),
        "prdDocument": _parse_json(state.prd_document),
        "dbSchema": _parse_json(state.db_schema),
        "apiSpec": _parse_json(state.api_spec),
        "openApiSpec": _parse_json(state.openapi_spec),
        "marketResearch": state.market_research,
        "artifactQuality": state.artifact_quality,
        "generationDiagnostics": state.generation_diagnostics,
        "rag": {
            "chunksRetrieved": state.rag_chunks_retrieved,
            "usedFormDataOnly": state.rag_context_fallback,
        },
    }


def _stage_details(state: PipelineState | None) -> dict:
    if state is None:
        return {}
    return {**({"failedStage": state.failed_stage} if state.failed_stage else {}),
            "lastCompletedStage": state.last_completed_stage,
            "skippedStages": state.skipped_stages,
            "generationDiagnostics": state.generation_diagnostics}


def _validate_form(form: FormData) -> None:
    if not form.project_name or not form.project_name.strip():
        raise HTTPException(400, "프로젝트 이름은 필수입니다")
    if not form.problem_definition:
        raise HTTPException(400, "문제정의는 필수입니다")
    if not form.target_users:
        raise HTTPException(400, "타겟유저는 최소 1명 필요합니다")
    if not form.feature_definition:
        raise HTTPException(400, "기능정의는 필수입니다")


def _validate_file_count(file_count: int) -> None:
    if file_count > MAX_FILES:
        raise HTTPException(400, f"PDF 파일은 최대 {MAX_FILES}개까지 업로드할 수 있습니다")
