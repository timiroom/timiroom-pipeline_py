"""Phase 1 전체 흐름 통합 테스트: ingest → build_from_form → PipelineState"""
import asyncio
import logging
import sys
import io
import time

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d  %(levelname)-5s  %(name)s - %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)

from config.settings import settings
from openai import AsyncOpenAI

from phase1.embedding_service import EmbeddingService
from phase1.session_vector_store import SessionVectorStore
from phase1.document_ingestion import DocumentIngestionService
from phase1.form_to_query import FormToQueryService
from phase1.hybrid_search import HybridSearchService
from phase1.pdf_parsing import PDFParsingService
from phase1.query_expansion import QueryExpansionService
from phase1.rag_pipeline import RagPipelineService
from phase1.reranker import RerankerService
from phase1.models import (
    FormData, PlatformType, ProblemDefinition,
    TargetUser, FeatureDefinition, CustomFeature, MoSCoW,
)

SEP = "=" * 55

TEST_DOCS = [
    {
        "content": (
            "React Native는 Facebook이 개발한 크로스 플랫폼 모바일 앱 개발 프레임워크입니다. "
            "하나의 코드베이스로 iOS와 Android 앱을 동시에 개발할 수 있습니다. "
            "JavaScript와 React를 기반으로 하며, 네이티브 컴포넌트를 직접 사용합니다. "
            "핫 리로딩 기능으로 개발 생산성이 높고, 대규모 커뮤니티와 풍부한 라이브러리를 보유하고 있습니다."
        ),
        "metadata": {"source": "tech_stack_guide", "topic": "mobile"},
    },
    {
        "content": (
            "FastAPI는 Python 기반의 고성능 웹 프레임워크로, 자동 문서화와 타입 힌트를 지원합니다. "
            "Starlette와 Pydantic을 기반으로 하며, 비동기 처리를 기본으로 지원합니다. "
            "OpenAPI와 JSON Schema를 자동 생성하여 API 문서화를 간편하게 합니다. "
            "Django나 Flask 대비 최대 300% 빠른 성능을 보여줍니다."
        ),
        "metadata": {"source": "tech_stack_guide", "topic": "backend"},
    },
    {
        "content": (
            "레시피 추천 시스템은 사용자가 보유한 식재료를 기반으로 만들 수 있는 요리를 추천합니다. "
            "재료 매칭 알고리즘과 사용자 선호도 학습을 결합하여 개인화된 추천을 제공합니다. "
            "냉장고 재료 관리, 유통기한 추적, 장보기 목록 자동 생성 기능을 포함합니다. "
            "음식 낭비를 줄이고 요리 결정 피로를 해소하는 것이 핵심 가치입니다."
        ),
        "metadata": {"source": "domain_knowledge", "topic": "recipe_app"},
    },
    {
        "content": (
            "모바일 앱 사용자 인증은 JWT 토큰 기반으로 구현하는 것이 일반적입니다. "
            "소셜 로그인(Google, Apple, Kakao)을 지원하면 사용자 진입 장벽을 낮출 수 있습니다. "
            "Refresh Token 전략으로 보안과 사용자 편의성을 동시에 확보합니다. "
            "회원가입 시 이메일 인증 또는 휴대폰 번호 인증을 추가하면 스팸 방지에 효과적입니다."
        ),
        "metadata": {"source": "best_practices", "topic": "auth"},
    },
    {
        "content": (
            "PostgreSQL과 pgvector 조합은 벡터 유사도 검색과 관계형 데이터를 함께 처리할 수 있어 "
            "RAG 시스템 구현에 적합합니다. "
            "HNSW 인덱스를 사용하면 대규모 벡터 검색에서 높은 성능을 발휘합니다. "
            "음식 레시피 데이터를 임베딩으로 저장하면 의미 기반 레시피 검색이 가능합니다."
        ),
        "metadata": {"source": "tech_stack_guide", "topic": "database"},
    },
]

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


async def main():
    t_total = time.time()

    # ── 서비스 초기화 ──────────────────────────────────────
    print(SEP)
    print("서비스 초기화")
    embedder = EmbeddingService(settings.embedding_model)
    session_store = SessionVectorStore()
    exaone = AsyncOpenAI(
        api_key=settings.exaone_api_key,
        base_url="https://api.friendli.ai/dedicated/v1",
    )

    ingestion = DocumentIngestionService(
        db_url=settings.db_url,
        embedder=embedder,
        chunk_size=settings.rag_chunk_size,
        chunk_overlap=settings.rag_chunk_overlap,
    )
    rag = RagPipelineService(
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
            ko_reranker_model=settings.ko_reranker_model,
        ),
        form_to_query=FormToQueryService(),
        pdf_parsing=PDFParsingService(session_store, embedder),
        session_store=session_store,
        top_k_hybrid=settings.rag_top_k_vector,
    )
    print("  완료")

    # ── Step 1: 문서 Ingest ──────────────────────────────
    print(SEP)
    print("Step 1: 테스트 문서 Ingest")
    t0 = time.time()
    total_chunks = 0
    for doc in TEST_DOCS:
        n = await ingestion.ingest(doc["content"], doc["metadata"])
        total_chunks += n
        print(f"  [{doc['metadata']['topic']}] {n}청크 저장")
    print(f"  합계: {total_chunks}청크 ({time.time()-t0:.1f}s)")

    # ── Step 2~6: build_from_form ────────────────────────
    print(SEP)
    print("Step 2~6: RagPipelineService.build_from_form()")
    t1 = time.time()
    state = await rag.build_from_form(FORM)
    elapsed = time.time() - t1

    # ── 결과 출력 ─────────────────────────────────────────
    print(SEP)
    print(f"Phase 1 완료 ({elapsed:.1f}s)")
    print(f"  session_id   : {state.session_id[:8]}...")
    print(f"  project_name : {state.project_name}")
    print(f"  platform     : {state.platform}")
    print(f"  tech_stack   : {state.tech_stack}")
    print(f"  must_features: {state.must_features}")
    print(f"  excl_features: {state.excluded_features}")
    print(f"  feature_list : {state.feature_list}")
    print(f"  user_query   : {state.user_query[:80]}...")
    print()
    print("context_prompt 미리보기 (300자):")
    print(state.context_prompt[:300])
    print("...")
    print()
    print(f"전체 소요: {time.time()-t_total:.1f}s")
    print(SEP)
    print("PASS")


if __name__ == "__main__":
    asyncio.run(main())
