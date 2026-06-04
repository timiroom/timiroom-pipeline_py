import logging
import uuid

from .form_to_query import FormToQueryService
from .hybrid_search import HybridSearchService
from .models import FormData
from .pdf_parsing import PDFParsingService
from .query_expansion import QueryExpansionService
from .reranker import RerankerService
from .session_vector_store import SessionVectorStore

logger = logging.getLogger(__name__)

CONTEXT_TEMPLATE = """당신은 소프트웨어 아키텍트입니다.
아래 참조 문서와 사용자 요청을 바탕으로 분석하세요.

[참조 문서]
{context}

[사용자 요청]
{query}

참조 문서를 최대한 활용하여 요청을 분석하고
필요한 기능과 설계 방향을 도출하세요."""


class RagPipelineService:

    def __init__(
        self,
        query_expansion: QueryExpansionService,
        hybrid_search: HybridSearchService,
        reranker: RerankerService,
        form_to_query: FormToQueryService,
        pdf_parsing: PDFParsingService,
        session_store: SessionVectorStore,
        top_k_hybrid: int = 20,
    ):
        self._query_expansion = query_expansion
        self._hybrid_search = hybrid_search
        self._reranker = reranker
        self._form_to_query = form_to_query
        self._pdf_parsing = pdf_parsing
        self._session_store = session_store
        self._top_k_hybrid = top_k_hybrid

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
                self._pdf_parsing.parse_and_store_all(pdf_files, session_id)

            # Step 2: 폼 → 쿼리 합성
            synthesized = self._form_to_query.synthesize(form)

            # Step 3: 쿼리 확장
            expanded = await self._query_expansion.expand(synthesized)

            # Step 4: Hybrid Search
            retrieved = await self._hybrid_search.search_with_session(expanded, session_id)

            # Step 5: Reranking
            reranked = await self._reranker.rerank(synthesized, retrieved)

            # Step 6: PipelineState 조립
            context_prompt = self._assemble_context(reranked, synthesized)

            from phase2.state import PipelineState
            return PipelineState(
                session_id=session_id,
                user_query=synthesized,
                project_name=form.project_name,
                platform=form.platform,
                tech_stack=form.tech_stack or [],
                problem_definition=form.problem_definition,
                target_users=form.target_users,
                must_features=self._form_to_query.extract_must_features(form),
                excluded_features=self._form_to_query.extract_excluded_features(form),
                feature_list=self._form_to_query.extract_all_included_features(form),
                context_prompt=context_prompt,
                status_message="Phase 1 완료",
            )
        finally:
            self._session_store.clear(session_id)

    async def build_context(self, user_query: str) -> str:
        """단일 쿼리 → 컨텍스트 프롬프트 (RAG ingest 엔드포인트용)."""
        expanded = await self._query_expansion.expand(user_query)
        candidates = await self._hybrid_search.search_multiple(expanded, self._top_k_hybrid)
        reranked = await self._reranker.rerank(user_query, candidates)
        context = self._assemble_context(reranked, user_query)
        return CONTEXT_TEMPLATE.format(context=context, query=user_query)

    def _assemble_context(self, chunks, query: str) -> str:
        parts = ["[프로젝트 컨텍스트]", query, ""]
        if chunks:
            parts.append(f"[관련 지식베이스 — 상위 {len(chunks)}개]")
            for c in chunks:
                parts.append(c.content)
                parts.append("---")
        return "\n".join(parts)
