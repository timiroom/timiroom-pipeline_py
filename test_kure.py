import asyncio
import time


async def test():
    print("=== Step 1: KURE-v1 모델 로딩 ===")
    t0 = time.time()
    from phase1.embedding_service import EmbeddingService
    svc = EmbeddingService("nlpai-lab/KURE-v1")
    print(f"  로딩 완료 ({time.time()-t0:.1f}s)")

    print()
    print("=== Step 2: 임베딩 생성 테스트 ===")
    texts = ["한국어 텍스트 임베딩 테스트", "회원가입 로그인 기능", "RAG 파이프라인 검색"]
    t1 = time.time()
    vecs = await svc.embed(texts)
    print(f"  texts: {len(texts)}개")
    print(f"  dim: {len(vecs[0])}")
    print(f"  elapsed: {time.time()-t1:.2f}s")
    for i, (t, v) in enumerate(zip(texts, vecs)):
        norm = sum(x**2 for x in v) ** 0.5
        print(f"  [{i}] \"{t}\" -> norm={norm:.4f}")

    print()
    print("=== Step 3: 단일 임베딩 (embed_one) ===")
    vec = await svc.embed_one("프로젝트 기획서 요구사항 분석")
    norm = sum(x**2 for x in vec) ** 0.5
    print(f"  dim: {len(vec)}, norm={norm:.4f}")

    print()
    print("=== Step 4: FormToQueryService ===")
    from phase1.form_to_query import FormToQueryService
    from phase1.models import (
        FormData, PlatformType, ProblemDefinition,
        TargetUser, FeatureDefinition, CustomFeature, MoSCoW, CommonFeature
    )
    form = FormData(
        projectName="냉장고 레시피 앱",
        projectDescription="냉장고 재료로 레시피를 추천해주는 앱",
        platform=PlatformType.APP,
        techStack=["React Native", "FastAPI", "PostgreSQL"],
        problemDefinition=ProblemDefinition(
            currentPainPoint="냉장고 재료를 몰라서 낭비",
            currentSolution="메모장에 직접 기록",
            idealState="앱 열면 바로 레시피 추천",
            businessImpact="식재료 낭비 30% 감소",
        ),
        targetUsers=[
            TargetUser(
                persona="20대 자취생",
                usageEnvironment="모바일, 집에서",
                biggestPainPoint="요리 결정 어려움",
            )
        ],
        featureDefinition=FeatureDefinition(
            customFeatures=[
                CustomFeature(featureName="재료 등록/관리", priority=MoSCoW.MUST),
                CustomFeature(featureName="레시피 자동 추천", priority=MoSCoW.MUST),
                CustomFeature(featureName="유통기한 알림", priority=MoSCoW.SHOULD),
                CustomFeature(featureName="소셜 공유", priority=MoSCoW.WONT),
            ]
        ),
    )
    synthesized = FormToQueryService().synthesize(form)
    print(f"  synthesized ({len(synthesized)}자):")
    print("  " + synthesized[:300].replace("\n", "\n  "))

    print()
    print("=== Step 5: QueryExpansionService ===")
    from openai import AsyncOpenAI
    from config.settings import settings
    from phase1.query_expansion import QueryExpansionService
    client = AsyncOpenAI(
        api_key=settings.exaone_api_key,
        base_url="https://api.friendli.ai/dedicated/v1",
    )
    t2 = time.time()
    expanded = await QueryExpansionService(client, settings.exaone_endpoint_id).expand(synthesized)
    print(f"  확장 쿼리 {len(expanded)}개 ({time.time()-t2:.1f}s)")
    for i, q in enumerate(expanded):
        print(f"  [{i+1}] {q[:80]}")

    print()
    print("=== Step 6: HybridSearch (DB 검색) ===")
    from phase1.hybrid_search import HybridSearchService
    from phase1.session_vector_store import SessionVectorStore
    session_store = SessionVectorStore()
    hybrid = HybridSearchService(
        db_url=settings.db_url,
        embedder=svc,
        session_store=session_store,
    )
    t3 = time.time()
    results = await hybrid.search_multiple(expanded, top_k=10)
    print(f"  검색 결과: {len(results)}개 ({time.time()-t3:.1f}s)")
    for r in results[:3]:
        print(f"  score={r.relevance_score:.4f} | {r.content[:60]}...")

    print()
    print("=== Step 7: Reranker ===")
    from phase1.reranker import RerankerService
    reranker = RerankerService(
        client=client,
        model=settings.exaone_endpoint_id,
        top_k_final=5,
        enabled=settings.rag_reranker_enabled,
    )
    t4 = time.time()
    reranked = await reranker.rerank(synthesized, results)
    print(f"  리랭킹 완료: {len(reranked)}개 ({time.time()-t4:.1f}s)")

    print()
    print("=" * 50)
    print("전체 테스트 PASS")


if __name__ == "__main__":
    asyncio.run(test())
