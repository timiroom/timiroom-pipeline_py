import asyncio
import io
import logging
import logging.config
import sys

# Windows cp949 터미널에서 UTF-8 출력 강제
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "buffer"):
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from openai import AsyncOpenAI

from common.exception_handler import register_exception_handlers
from common.logging_middleware import RequestIdFilter, RequestIdMiddleware
from common.pm_skills import PmSkillsLoader
from config.settings import settings
from phase1.document_ingestion import DocumentIngestionService
from phase1.embedding_service import EmbeddingService
from phase1.form_to_query import FormToQueryService
from phase1.hybrid_search import HybridSearchService
from phase1.pdf_parsing import PDFParsingService
from phase1.rag_pipeline import RagPipelineService
from phase1.recommendation.services import (
    FeatureRecommendationService,
    PersonaRecommendationService,
    TechStackRecommendationService,
)
from phase1.reranker import RerankerService
from phase1.search_rl_service import SearchRLService
from phase1.semantic_chunking import SemanticChunkingService
from phase1.session_vector_store import SessionVectorStore
from phase2.agents.api_agent import ApiAgent
from phase2.agents.dba_agent import DbaAgent
from phase2.agents.pm_agent import PmAgent
from phase2.agents.prd_agent import PrdAgent
from phase2.agents.qa_agent import QaAgent
from phase2.agents.search_agent import SearchAgent
from phase2.llm_runtime import LlmRuntime
from phase2.orchestration_graph import OrchestrationGraph
from phase2.sse_service import PipelineProgressService
from phase3.retry_service import RetryService
from phase3.schema_validator import SchemaValidator
from phase3.validation_service import ValidationService
from phase4.kafka_consumer import KafkaConsumerService
from phase4.kafka_producer import KafkaProducerService

# ── 로깅 설정 (requestId 컨텍스트 포함) ──────────────────────────

_root_logger = logging.getLogger()
_root_logger.setLevel(logging.DEBUG)

_handler = logging.StreamHandler(sys.stdout)
_handler.setFormatter(logging.Formatter(
    "%(asctime)s.%(msecs)03d [%(pipeline_id)s] [%(request_id)s] %(levelname)-5s %(name)s - %(message)s",
    datefmt="%H:%M:%S",
))
_handler.addFilter(RequestIdFilter())
_root_logger.handlers = [_handler]

# 외부 라이브러리 로그 레벨 억제
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("aiokafka").setLevel(logging.INFO)

# ── API 클라이언트 ────────────────────────────────────────────────

openai_client = AsyncOpenAI(
    api_key=settings.openai_api_key,
    base_url=settings.openai_base_url,
    timeout=settings.openai_request_timeout_seconds,
    max_retries=0,
)

# ── Phase 1 ───────────────────────────────────────────────────────

session_store = SessionVectorStore()
db_io_semaphore = asyncio.Semaphore(settings.rag_db_max_concurrency)
embedding_service = EmbeddingService(
    settings.upstage_api_key,
    settings.solar_embedding_query_model,
    settings.solar_embedding_passage_model,
    max_concurrency=settings.embedding_max_concurrency,
    batch_size=settings.embedding_batch_size,
)
pm_skills = PmSkillsLoader(embedding_service)

search_rl_service = SearchRLService(
    db_url=settings.db_url,
    db_semaphore=db_io_semaphore,
)
hybrid_search = HybridSearchService(
    db_url=settings.db_url,
    document_table=settings.get_rag_document_table(),
    embedder=embedding_service,
    session_store=session_store,
    top_k_vector=settings.rag_top_k_vector,
    top_k_keyword=settings.rag_top_k_keyword,
    similarity_threshold=settings.rag_similarity_threshold,
    min_threshold=settings.rag_min_threshold,
    min_results=settings.rag_min_results,
    threshold_step=settings.rag_threshold_step,
    rl_service=search_rl_service,
    db_semaphore=db_io_semaphore,
)
reranker = RerankerService(
    top_k_final=settings.rag_top_k_final,
    enabled=settings.rag_reranker_enabled,
    api_key=settings.cohere_api_key,
    model=settings.cohere_rerank_model,
    base_url=settings.cohere_base_url,
    max_concurrency=settings.cohere_max_concurrency,
)
form_to_query = FormToQueryService()
pdf_chunker = SemanticChunkingService(
    embedding_service,
    max_chunk_size=settings.rag_chunk_size,
    chunk_overlap=settings.rag_chunk_overlap,
)
pdf_parsing = PDFParsingService(session_store, embedding_service, pdf_chunker)
document_ingestion_service = DocumentIngestionService(
    db_url=settings.db_url,
    document_table=settings.get_rag_document_table(),
    embedder=embedding_service,
    chunk_size=settings.rag_chunk_size,
    chunk_overlap=settings.rag_chunk_overlap,
    db_semaphore=db_io_semaphore,
)

rag_pipeline_service = RagPipelineService(
    hybrid_search=hybrid_search,
    reranker=reranker,
    form_to_query=form_to_query,
    pdf_parsing=pdf_parsing,
    session_store=session_store,
    rl_service=search_rl_service,
)

# ── Phase 1 추천 서비스 ───────────────────────────────────────────

tech_stack_service = TechStackRecommendationService(openai_client, settings.openai_chat_model)
persona_service = PersonaRecommendationService(openai_client, settings.openai_chat_model)
feature_service = FeatureRecommendationService(openai_client, settings.openai_chat_model)

# ── Phase 2 ───────────────────────────────────────────────────────

progress_service = PipelineProgressService()
phase2_llm_runtime = LlmRuntime(
    settings.phase2_llm_max_concurrency,
    settings.openai_request_timeout_seconds,
)

search_agent = SearchAgent(
    openai_client,
    settings.openai_chat_model,
    runtime=phase2_llm_runtime,
    web_search_enabled=settings.phase2_web_search_enabled,
)
pm_agent = PmAgent(openai_client, pm_skills, settings.openai_chat_model, runtime=phase2_llm_runtime)
prd_agent = PrdAgent(openai_client, settings.openai_chat_model, runtime=phase2_llm_runtime)
dba_agent = DbaAgent(openai_client, settings.openai_chat_model, runtime=phase2_llm_runtime)
api_agent = ApiAgent(openai_client, settings.openai_chat_model, runtime=phase2_llm_runtime)
qa_agent = QaAgent(openai_client, settings.openai_chat_model, runtime=phase2_llm_runtime)

orchestration_graph = OrchestrationGraph(
    search_agent=search_agent,
    pm_agent=pm_agent,
    prd_agent=prd_agent,
    dba_agent=dba_agent,
    api_agent=api_agent,
    qa_agent=qa_agent,
    progress_service=progress_service,
    timeout_seconds=settings.phase2_timeout_seconds,
    repair_timeout_seconds=settings.phase3_repair_timeout_seconds,
)

# ── Phase 3 ───────────────────────────────────────────────────────

schema_validator = SchemaValidator()
validation_service = ValidationService(schema_validator)
retry_service = RetryService(max_retry=settings.validation_max_retry)

# ── Phase 4 ───────────────────────────────────────────────────────

kafka_producer_service = KafkaProducerService(
    bootstrap_servers=settings.kafka_bootstrap_servers,
    topic=settings.kafka_topic_pipeline_result,
    db_url=settings.db_url,
    max_publish_retry=settings.kafka_publish_max_retry,
)
kafka_consumer_service = KafkaConsumerService(
    bootstrap_servers=settings.kafka_bootstrap_servers,
    topic=settings.kafka_topic_pipeline_result,
    group_id=settings.kafka_consumer_group_id,
    ingestion_service=document_ingestion_service,
    dead_letter_topic=settings.kafka_topic_dead_letter,
    max_poll_interval_ms=settings.kafka_max_poll_interval_ms,
)


# ── 앱 생명주기 ────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    await pm_skills.load()

    # Kafka는 백그라운드 비동기 연결 — 앱 기동 차단 없음 (Java fail-fast: false 동일)
    kafka_producer_service.start()
    kafka_consumer_service.start()

    yield

    await kafka_consumer_service.stop()
    await kafka_producer_service.stop()
    await reranker.close()
    await embedding_service.close()
    await openai_client.close()


# ── FastAPI 앱 ────────────────────────────────────────────────────

app = FastAPI(
    title="RAG Pipeline API",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(RequestIdMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.get_allowed_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Content-Type", "Cache-Control", "Transfer-Encoding", "X-Accel-Buffering"],
    max_age=3600,
)

from routers.agent import router as agent_router
from routers.chat import router as chat_router
from routers.document import router as document_router
from routers.orchestration import router as orchestration_router
from routers.rag import router as rag_router
from routers.recommendation import router as recommendation_router

app.include_router(orchestration_router)
app.include_router(recommendation_router)
app.include_router(agent_router)
app.include_router(chat_router)
app.include_router(document_router)
app.include_router(rag_router)

register_exception_handlers(app)


@app.get("/actuator/health")
def health():
    return {"status": "UP"}


@app.get("/actuator/health/liveness")
def liveness():
    return {"status": "UP"}


@app.get("/actuator/health/readiness")
def readiness():
    return {"status": "UP"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8081, reload=False)
