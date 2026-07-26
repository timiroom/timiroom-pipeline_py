import logging
import uuid

from .form_to_query import FormToQueryService
from .hybrid_search import HybridSearchService
from .models import FormData
from .pdf_parsing import PDFParsingService
from .reranker import RerankerService
from .search_rl_service import SearchRLService
from .session_vector_store import SessionVectorStore

logger = logging.getLogger(__name__)


class RagPipelineService:

    def __init__(
        self,
        hybrid_search: HybridSearchService,
        reranker: RerankerService,
        form_to_query: FormToQueryService,
        pdf_parsing: PDFParsingService,
        session_store: SessionVectorStore,
        rl_service: SearchRLService | None = None,
    ):
        self._hybrid_search = hybrid_search
        self._reranker = reranker
        self._form_to_query = form_to_query
        self._pdf_parsing = pdf_parsing
        self._session_store = session_store
        self._rl_service = rl_service

    async def build_from_form(
        self,
        form: FormData,
        pdf_files: list[tuple[str, bytes]] | None = None,
    ) -> dict:
        session_id = str(uuid.uuid4())
        logger.info("[%s] Phase 1 시작 — 프로젝트: %s", session_id[:8], form.project_name)

        try:
            # Step 1: PDF 파싱
            if pdf_files:
                logger.info("[%s] ▶ Step 1: PDF 파싱 시작 (%d개)", session_id[:8], len(pdf_files))
                await self._pdf_parsing.parse_and_store_all(pdf_files, session_id)
                logger.info("[%s] ✔ Step 1: PDF 파싱 완료", session_id[:8])

            # Step 2: 폼 → 쿼리 합성
            logger.info("[%s] ▶ Step 2: 폼 → 쿼리 합성", session_id[:8])
            synthesized = self._form_to_query.synthesize(form)
            logger.info("[%s] ✔ Step 2: 합성 쿼리\n%s", session_id[:8], synthesized[:200])

            # Step 3: 폼 데이터 섹션에서 검색 쿼리 직접 추출 (LLM 호출 없음)
            must_features = self._form_to_query.extract_must_features(form)
            queries = self._build_queries_from_form(form, synthesized, must_features)
            logger.info("[%s] ✔ Step 3: %d개 쿼리 추출 (폼 데이터 직접)", session_id[:8], len(queries))
            for i, q in enumerate(queries, 1):
                logger.info("[%s]   [%d] %s", session_id[:8], i, q)

            # Step 4: Hybrid Search
            logger.info("[%s] ▶ Step 4: Hybrid Search (벡터 + 키워드)", session_id[:8])
            retrieved = await self._hybrid_search.search_with_session(queries, session_id)
            logger.info("[%s] ✔ Step 4: %d개 청크 검색됨", session_id[:8], len(retrieved))
            for i, c in enumerate(retrieved[:5], 1):
                logger.info(
                    "[%s]   [%d] rrf=%.5f topic=%s | %s",
                    session_id[:8], i, c.relevance_score or 0,
                    c.metadata.get("topic", "?"), c.content[:60],
                )

            # Step 5: Reranking
            logger.info("[%s] ▶ Step 5: Reranking (Ko-Reranker)", session_id[:8])
            reranked = await self._reranker.rerank(synthesized, retrieved)
            logger.info("[%s] ✔ Step 5: %d개로 압축", session_id[:8], len(reranked))
            for i, c in enumerate(reranked, 1):
                logger.info(
                    "[%s]   [%d] topic=%s | %s",
                    session_id[:8], i,
                    c.metadata.get("topic", "?"), c.content[:60],
                )

            # Step 5-1: 리랭커 평균 점수 → Phase1 RL 피드백
            if self._rl_service is not None and reranked:
                avg_score = sum(c.relevance_score or 0.0 for c in reranked) / len(reranked)
                self._rl_service.apply_rerank_score(session_id, avg_score)
                logger.info("[%s] Phase1 RL 피드백 적용 — avgScore:%.3f", session_id[:8], avg_score)

            # Step 6: PipelineState 조립
            logger.info("[%s] ▶ Step 6: Context 조립", session_id[:8])
            context_prompt = self._assemble_context(reranked, synthesized)
            logger.info("[%s] ✔ Step 6: context_prompt %d자 생성", session_id[:8], len(context_prompt))

            from phase2.state import PipelineState
            return PipelineState(
                session_id=session_id,
                user_query=synthesized,
                project_name=form.project_name,
                platform=form.platform,
                tech_stack=form.tech_stack or [],
                problem_definition=form.problem_definition,
                target_users=form.target_users,
                must_features=must_features,
                excluded_features=self._form_to_query.extract_excluded_features(form),
                feature_list=self._form_to_query.extract_all_included_features(form),
                context_prompt=context_prompt,
                status_message="Phase 1 완료",
            )
        finally:
            self._session_store.clear(session_id)

    def _build_queries_from_form(
        self,
        form: FormData,
        synthesized_query: str,
        must_features: list[str],
    ) -> list[str]:
        """폼 데이터 섹션을 검색 쿼리 목록으로 변환 — LLM 호출 없음.

        1. 합성 쿼리  — 전체 맥락
        2. 서비스 개요 — 프로젝트명 + 설명
        3. 문제 정의  — 핵심 페인포인트 + 이상적 상태
        4. 타겟 유저  — 페르소나 + 불편함
        5. 기능 목록  — Must 기능 키워드
        """
        queries = [synthesized_query]

        queries.append(f"{form.project_name} {form.project_description}")

        pd = form.problem_definition
        problem_query = f"{pd.current_pain_point} {pd.ideal_state}"
        if pd.competitor_gap:
            problem_query += f" {pd.competitor_gap}"
        queries.append(problem_query)

        if form.target_users:
            target_query = " ".join(
                f"{u.persona} {u.biggest_pain_point}" for u in form.target_users
            )
            queries.append(target_query)

        valid_features = [f for f in must_features if f]
        if valid_features:
            queries.append(" ".join(valid_features))

        return queries

    def _assemble_context(self, chunks, query: str) -> str:
        parts = ["[프로젝트 컨텍스트]", query, ""]
        if chunks:
            parts.append(f"[관련 지식베이스 — 상위 {len(chunks)}개]")
            for c in chunks:
                parts.append(c.content)
                parts.append("---")
        return "\n".join(parts)
