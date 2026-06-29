"""Phase 1 → 2 → 3 → 4 전체 파이프라인 통합 테스트"""
import asyncio
import logging
import time
import uuid

from openai import AsyncOpenAI

from config.settings import settings
from common.pm_skills import PmSkillsLoader
from phase1.document_ingestion import DocumentIngestionService
from phase1.embedding_service import EmbeddingService
from phase1.form_to_query import FormToQueryService
from phase1.hybrid_search import HybridSearchService
from phase1.models import (
    FormData, PlatformType, ProblemDefinition,
    TargetUser, FeatureDefinition, CustomFeature, MoSCoW,
)
from phase1.pdf_parsing import PDFParsingService
from phase1.query_expansion import QueryExpansionService
from phase1.rag_pipeline import RagPipelineService
from phase1.reranker import RerankerService
from phase1.session_vector_store import SessionVectorStore
from phase2.agents.api_agent import ApiAgent
from phase2.agents.dba_agent import DbaAgent
from phase2.agents.pm_agent import PmAgent
from phase2.agents.prd_agent import PrdAgent
from phase2.agents.qa_agent import QaAgent
from phase2.agents.search_agent import SearchAgent
from phase2.orchestration_graph import OrchestrationGraph
from phase2.sse_service import PipelineProgressService
from phase3.retry_service import RetryService
from phase3.schema_validator import SchemaValidator
from phase3.validation_service import ValidationService
from phase4.kafka_producer import KafkaProducerService

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logging.getLogger("phase1").setLevel(logging.INFO)
logging.getLogger("phase1.hybrid_search").setLevel(logging.INFO)
logging.getLogger("phase2").setLevel(logging.INFO)
logging.getLogger("phase3").setLevel(logging.INFO)
logging.getLogger("phase4").setLevel(logging.INFO)

SEP = "=" * 60

FORM = FormData(
    projectName="냉장고 레시피 앱",
    projectDescription="냉장고 재료로 레시피를 추천해주는 모바일 앱",
    platform=PlatformType.APP,
    techStack=["React Native", "FastAPI", "PostgreSQL"],
    problemDefinition=ProblemDefinition(
        currentPainPoint="냉장고에 뭐가 있는지 몰라서 재료를 낭비하고 요리 결정을 못함",
        currentSolution="메모장에 직접 적거나 냉장고 문에 붙여둔 메모지로 관리",
        idealState="앱을 열면 바로 가진 재료로 만들 수 있는 레시피가 자동 추천",
        businessImpact="식재료 낭비 30% 감소, 요리 결정 시간 단축",
    ),
    targetUsers=[
        TargetUser(
            persona="20대 자취생",
            usageEnvironment="모바일, 집에서 혼자 요리",
            biggestPainPoint="뭘 해먹어야 할지 매번 고민, 재료 낭비",
        ),
        TargetUser(
            persona="바쁜 30대 직장인",
            usageEnvironment="퇴근 후 빠르게 요리",
            biggestPainPoint="장 보고 남은 재료 처리 못하고 버림",
        ),
    ],
    featureDefinition=FeatureDefinition(
        customFeatures=[
            CustomFeature(featureName="재료 등록 및 관리", priority=MoSCoW.MUST),
            CustomFeature(featureName="레시피 자동 추천", priority=MoSCoW.MUST),
            CustomFeature(featureName="회원가입/로그인", priority=MoSCoW.MUST),
            CustomFeature(featureName="유통기한 알림", priority=MoSCoW.SHOULD),
            CustomFeature(featureName="장보기 목록 자동 생성", priority=MoSCoW.COULD),
            CustomFeature(featureName="SNS 공유", priority=MoSCoW.WONT),
        ]
    ),
)


def print_phase(n: int, name: str, elapsed: float | None = None):
    suffix = f" ({elapsed:.1f}s)" if elapsed is not None else ""
    print(f"\n{SEP}")
    print(f"  Phase {n}: {name}{suffix}")
    print(SEP)


async def main():
    t_start = time.time()
    pipeline_id = str(uuid.uuid4())
    print(f"Pipeline ID: {pipeline_id[:8]}")

    # ── 서비스 초기화 ──────────────────────────────────────────────
    print("\n서비스 초기화 중...")
    exaone = AsyncOpenAI(
        api_key=settings.exaone_api_key,
        base_url="https://api.friendli.ai/dedicated/v1",
    )
    embedder = EmbeddingService(settings.embedding_model)
    session_store = SessionVectorStore()
    pm_skills = PmSkillsLoader(exaone)

    # Phase 1
    rag_svc = RagPipelineService(
        query_expansion=QueryExpansionService(exaone, settings.exaone_endpoint_id),
        hybrid_search=HybridSearchService(
            db_url=settings.db_url,
            embedder=embedder,
            session_store=session_store,
            top_k_vector=settings.rag_top_k_vector,
            top_k_keyword=settings.rag_top_k_keyword,
        ),
        reranker=RerankerService(
            client=exaone,
            model=settings.exaone_endpoint_id,
            top_k_final=settings.rag_top_k_final,
            enabled=settings.rag_reranker_enabled,
            cohere_api_key=settings.cohere_api_key,
        ),
        form_to_query=FormToQueryService(),
        pdf_parsing=PDFParsingService(session_store),
        session_store=session_store,
        top_k_hybrid=settings.rag_top_k_vector,
    )

    # Phase 2
    progress_svc = PipelineProgressService()
    await pm_skills.load()
    orch = OrchestrationGraph(
        search_agent=SearchAgent(exaone, settings.exaone_endpoint_id),
        pm_agent=PmAgent(exaone, pm_skills, settings.exaone_endpoint_id),
        prd_agent=PrdAgent(exaone, settings.exaone_endpoint_id),
        dba_agent=DbaAgent(exaone, settings.exaone_endpoint_id),
        api_agent=ApiAgent(exaone, settings.exaone_endpoint_id),
        qa_agent=QaAgent(exaone, settings.exaone_endpoint_id),
        progress_service=progress_svc,
    )

    # Phase 3
    val_svc = ValidationService(SchemaValidator())
    retry_svc = RetryService(max_retry=settings.validation_max_retry)

    # Phase 4
    kafka_svc = KafkaProducerService(
        bootstrap_servers=settings.kafka_bootstrap_servers,
        topic=settings.kafka_topic_pipeline_result,
    )
    kafka_svc.start()

    print("  완료")

    # ── Phase 1 ────────────────────────────────────────────────────
    print_phase(1, "RAG 컨텍스트 구성")
    t1 = time.time()
    state = await rag_svc.build_from_form(FORM)
    elapsed1 = time.time() - t1

    print(f"  context_prompt: {len(state.context_prompt)}자")
    print(f"  feature_list  : {state.feature_list}")
    print(f"  must_features : {state.must_features}")
    print(f"  소요: {elapsed1:.1f}s")

    # Phase 1 검색 단계 별도 검증
    print("\n  [Phase 1 검색 단계 직접 검증]")
    hybrid_svc = rag_svc._hybrid_search  # type: ignore[attr-defined]
    test_query = "레시피 추천 앱 재료 관리"
    v_hits = await hybrid_svc._vector_search(test_query)
    k_hits = hybrid_svc._keyword_search(test_query)
    rrf_hits = hybrid_svc._rrf(v_hits, k_hits, 5)
    print(f"  쿼리: '{test_query}'")
    print(f"  벡터 검색: {len(v_hits)}건")
    print(f"  키워드 검색: {len(k_hits)}건")
    print(f"  RRF 병합: {len(rrf_hits)}건")
    for i, c in enumerate(rrf_hits[:3], 1):
        print(f"    #{i} score={c.relevance_score:.5f} | {c.content[:60]}...")

    # ── Phase 2 ────────────────────────────────────────────────────
    print_phase(2, "멀티 에이전트 오케스트레이션")
    print("  Search → PM → PRD → DBA/API → QA 순으로 실행...")
    t2 = time.time()
    phase2_state = await orch.run(state, pipeline_id)
    elapsed2 = time.time() - t2

    print(f"  market_research : {len(phase2_state.market_research or '')}자")
    print(f"  prd_document    : {len(phase2_state.prd_document or '')}자")
    print(f"  db_schema       : {len(phase2_state.db_schema or '')}자")
    print(f"  api_spec        : {len(phase2_state.api_spec or '')}자")
    print(f"  retry_count     : {phase2_state.retry_count}")
    print(f"  소요: {elapsed2:.1f}s")

    # ── Phase 3 ────────────────────────────────────────────────────
    print_phase(3, "결과 검증")
    t3 = time.time()
    validated = val_svc.validate(phase2_state)
    if not validated.validated:
        print(f"  검증 실패 — RetryService 진입: {validated.last_validation_error}")
        validated = await retry_svc.retry_with(validated, orch, val_svc)
    elapsed3 = time.time() - t3

    print(f"  validated : {validated.validated}")
    print(f"  status    : {validated.status_message}")
    if validated.last_validation_error:
        print(f"  오류      : {validated.last_validation_error}")
    print(f"  소요: {elapsed3:.1f}s")

    if not validated.validated:
        print("\n  HITL 필요 — 관리자 검토 요청")
        return

    # ── Phase 4 ────────────────────────────────────────────────────
    print_phase(4, "Kafka 발행 (지식베이스 저장)")
    t4 = time.time()
    pid = await kafka_svc.publish(validated)
    elapsed4 = time.time() - t4

    if pid:
        print(f"  Kafka 발행 완료 — pipelineId: {pid}")
    else:
        print("  Kafka 미연결 — 저장 건너뜀 (운영 환경에서는 자동 재연결)")
    print(f"  소요: {elapsed4:.1f}s")

    # ── 최종 결과 ──────────────────────────────────────────────────
    total = time.time() - t_start
    print(f"\n{SEP}")
    print(f"  전체 완료 — {total:.1f}s")
    print(SEP)
    print(f"  Phase 1: {elapsed1:.1f}s")
    print(f"  Phase 2: {elapsed2:.1f}s")
    print(f"  Phase 3: {elapsed3:.1f}s")
    print(f"  Phase 4: {elapsed4:.1f}s")
    print()
    print("  [산출물 미리보기]")
    if phase2_state.prd_document:
        print(f"  PRD    : {phase2_state.prd_document[:200]}...")
    if phase2_state.db_schema:
        print(f"  DB     : {phase2_state.db_schema[:200]}...")
    if phase2_state.api_spec:
        print(f"  API    : {phase2_state.api_spec[:200]}...")

    # dump 파일 위치 안내
    from pathlib import Path
    dumps = sorted(Path("pipeline_dumps").glob("*.txt"), key=lambda p: p.stat().st_mtime)
    if dumps:
        print(f"\n  전체 덤프: {dumps[-1]}")


if __name__ == "__main__":
    asyncio.run(main())
