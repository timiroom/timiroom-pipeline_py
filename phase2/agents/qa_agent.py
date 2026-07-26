import asyncio
import json
import logging
from typing import TypedDict

from langgraph.graph import StateGraph, START, END
from openai import AsyncOpenAI, InternalServerError, APITimeoutError, APIConnectionError

from phase2.agents.dba_agent import _to_array_schema, min_table_count, sanitize_tables
from phase2.agents.api_agent import _normalize_endpoints
from phase2.feature_coverage import uncovered_features
from phase2.json_utils import try_parse_json
from phase2.state import PipelineState

logger = logging.getLogger(__name__)

_MAX_SCHEMA_CHARS = 8000  # reviewer 컨텍스트에 전달할 db_schema/api_spec 최대 길이
_MAX_ROUNDS = 2  # 도메인당 reviewer↔manager_check 최대 라운드 (2회째는 강제 승인)

# EXAONE 모델 카드 권장 샘플링 파라미터
# https://huggingface.co/LGAI-EXAONE/K-EXAONE-236B-A23B
_TEMPERATURE = 1.0
_TOP_P = 0.95
_PRESENCE_PENALTY = 0.0

REVIEW_SYSTEM = "JSON만 출력하세요. 설명·인사말·마크다운 코드블록 금지. { 로 시작해서 } 로 끝납니다."

_LENIENCY_NOTE = """
결함 판단 기준 (엄하게 적용하지 말 것):
- 사소한 필드명 차이 (camelCase vs snake_case)는 결함이 아닙니다
- 일부 필드가 빠져도 핵심 기능이 동작하면 통과입니다
- 전체적인 설계 방향이 올바르면 세부 불일치는 무시하세요
- 연체료 계산, 예약 우선순위 등 비즈니스 로직은 서버 내부에서 처리되므로
  별도 API 엔드포인트가 없어도 정상입니다"""

DB_REVIEW_PROMPT = """당신은 시니어 DBA입니다.
아래 DB 스키마가 API 스펙·기능 목록과 정합성이 있는지 정확히 검증하고,
문제가 있으면 그 자리에서 직접 수정하세요.

[현재 DB 스키마]
{draft}

[API 스펙 (참고용 — 수정 금지)]
{sibling}

[기능 목록]
{feature_str}
{feedback_section}
{completeness_hint}

검수 기준:
1. DB 스키마에 핵심 테이블과 컬럼이 존재하는가?
2. API 스펙의 각 엔드포인트가 필요로 하는 DB 테이블과 컬럼이 존재하는가?
3. DB 테이블 간 외래키 관계가 올바르게 설계되어 있는가?
4. API 필드명과 DB 컬럼명이 일치하는가?
5. 컬럼 제약조건이 기능과 모순되지 않는가? — 사용자가 수동/선택 입력할 수 있어야 하는 필드에
   NOT_NULL·UNIQUE가 걸려 입력이 막히면 결함이다. (예: "바코드 없는 재료도 수동 등록" 기능이 있으면
   barcode 컬럼은 NOT_NULL·UNIQUE이면 안 됨 → NULL 허용으로 수정)
6. MVP에서 제외(excluded)된 기능만을 위해 존재하는 테이블이 포함돼 있지 않은가? (있으면 과설계 — 지적)
{leniency}

문제가 있으면 DB 스키마 전체를 수정해 fixed에 담고, 문제가 없으면 원본 그대로 fixed에 담으세요.
fixed의 구조는 원본과 동일하게 유지하세요 (tables는 배열, 각 원소는 {{"name":..., "description":..., "columns":[...], "indexes":[...]}}, 각 컬럼은 "컬럼명 타입 제약조건" 문자열).

JSON:
{{"issues": ["결함 설명"], "fixed": {{"tables": {{...}}, "relationships": [...]}}}}
"""

API_REVIEW_PROMPT = """당신은 시니어 백엔드 아키텍트입니다.
아래 API 스펙이 기능 목록·PRD·DB 스키마와 정합성이 있는지 정확히 검증하고,
문제가 있으면 그 자리에서 직접 수정하세요.

[현재 API 스펙]
{draft}

[DB 스키마 (참고용 — 수정 금지)]
{sibling}

[PRD coreFeatures 요약 (참고용)]
{prd_summary}

[기능 목록]
{feature_str}
{feedback_section}
{completeness_hint}

검수 기준:
1. 기능 목록의 모든 기능에 해당하는 API 엔드포인트가 존재하는가?
2. PRD의 coreFeatures 각 항목에 대응하는 API 엔드포인트가 존재하는가?
3. DB 스키마와 필드명/테이블명이 어긋나지 않는가?
{leniency}

문제가 있으면 API 스펙 전체를 수정해 fixed에 담고, 문제가 없으면 원본 그대로 fixed에 담으세요.
fixed의 구조는 원본과 동일하게 유지하세요 ({{"endpoints": [...], "authentication": "..."}}).

JSON:
{{"issues": ["결함 설명"], "fixed": {{"endpoints": [...], "authentication": "..."}}}}
"""

PRD_REVIEW_PROMPT = """당신은 10년 경력의 시니어 PM입니다.
아래 PRD 문서가 기능 목록·DB 스키마·API 스펙과 **정합성**이 있는지만 검증하세요
(글자수·항목 개수 등 PRD 자체의 완성도 기준은 이미 검증되었으므로 다루지 않습니다).
정합성 문제가 있으면 그 자리에서 직접 수정하세요.

[현재 PRD 문서]
{draft}

[DB 스키마 (참고용 — 수정 금지)]
{db_sibling}

[API 스펙 (참고용 — 수정 금지)]
{api_sibling}

[기능 목록]
{feature_str}
{feedback_section}
{completeness_hint}

검수 기준:
1. PRD의 coreFeatures가 기능 목록과 일치하는가? (누락·불일치 항목)
2. PRD의 coreFeatures 각 항목에 대응하는 DB 테이블이 존재하는가?
3. PRD의 goals/kpi가 실제 설계된 기능과 현실적으로 연결되는가?
{leniency}

정합성 문제가 있으면 PRD 문서 전체를 수정해 fixed에 담고, 문제가 없으면 원본 그대로 fixed에 담으세요.
fixed의 구조(키 목록)는 원본과 동일하게 유지하세요.

JSON:
{{"issues": ["결함 설명"], "fixed": {{...PRD 문서 전체...}}}}
"""

_REVIEW_PROMPTS = {"db": DB_REVIEW_PROMPT, "api": API_REVIEW_PROMPT, "prd": PRD_REVIEW_PROMPT}

_DOMAIN_LABEL = {"db": "DB 스키마", "api": "API 스펙", "prd": "PRD 문서"}

_DOMAIN_CRITERIA = {
    "db": "핵심 테이블·컬럼 존재, API와 필드명 일치, 외래키 관계 정확성",
    "api": "기능 목록·PRD coreFeatures 전체 커버리지, DB 스키마와의 필드 정합성",
    "prd": "기능 목록·DB 스키마·API 스펙과의 정합성 (PRD 자체 완성도는 검사 대상 아님)",
}

MANAGER_CHECK_PROMPT = """당신은 시니어 소프트웨어 아키텍트 겸 QA manager입니다.
아래는 {domain_label} reviewer가 라운드 {round}에 제출한 수정본입니다.
검수 기준을 충분히 충족했는지 판정하세요.

[수정본]
{draft}

[reviewer가 보고한 결함]
{issues}

검수 기준: {criteria}

기준을 충분히 충족했으면 approved=true, 아직 부족하면 approved=false와 함께
다음 라운드에서 구체적으로 무엇을 더 고쳐야 하는지 feedback에 작성하세요.

JSON:
{{"approved": true 또는 false, "feedback": "다음 라운드 반영 지시 (approved=true면 빈 문자열)"}}
"""


def _count_items(domain: str, data) -> int | None:
    """도메인별 '핵심 항목 수'를 센다 — reviewer의 fixed가 JSON 잘림으로 원본보다
    비정상적으로 축소되었는지 판단하기 위한 최소한의 결정론적 크기 검증."""
    if not isinstance(data, dict):
        return None
    if domain == "db":
        tables = data.get("tables")
        if isinstance(tables, (list, dict)):
            return len(tables)
        return 0
    if domain == "api":
        endpoints = data.get("endpoints")
        return len(endpoints) if isinstance(endpoints, list) else 0
    if domain == "prd":
        # 최상위 키 개수(len(data))는 coreFeatures 리스트가 통째로 비워져도 변하지 않아
        # 축소를 못 잡는다. 개수에 민감한 실제 콘텐츠 리스트 섹션의 항목 총합을 센다.
        total = 0
        for field in ("coreFeatures", "kpi", "userPersonas", "releaseSchedule", "goals"):
            val = data.get(field)
            if isinstance(val, list):
                total += len(val)
        return total
    return None


class _QaGraphState(TypedDict, total=False):
    ctx: dict
    db_draft: str
    db_round: int
    db_approved: bool
    db_issues: list
    db_feedback: str
    api_draft: str
    api_round: int
    api_approved: bool
    api_issues: list
    api_feedback: str
    prd_draft: str
    prd_round: int
    prd_approved: bool
    prd_issues: list
    prd_feedback: str


class QaAgent:

    def __init__(self, client: AsyncOpenAI, model: str = "gpt-4o"):
        self._client = client
        self._model = model
        self._graph = self._build_graph()

    def _build_graph(self):
        graph = StateGraph(_QaGraphState)
        for domain in ("db", "api", "prd"):
            graph.add_node(f"{domain}_reviewer", self._make_reviewer_node(domain))
            graph.add_node(f"{domain}_check", self._make_check_node(domain))
            graph.add_edge(START, f"{domain}_reviewer")
            graph.add_edge(f"{domain}_reviewer", f"{domain}_check")
            graph.add_conditional_edges(
                f"{domain}_check", self._make_router(domain), [f"{domain}_reviewer", END]
            )
        return graph.compile()

    async def execute(self, state: PipelineState, dump=None) -> PipelineState:
        logger.info("QA 에이전트 시작 (reviewer↔manager 도메인별 독립 루프)")

        api_completeness_hint = self._api_completeness_hint(
            try_parse_json(state.api_spec) if state.api_spec else None,
            state.feature_list,
        )
        db_completeness_hint = self._db_completeness_hint(
            try_parse_json(state.db_schema) if state.db_schema else None,
            state.feature_list,
        )
        prd_completeness_hint = self._prd_completeness_hint(
            try_parse_json(state.prd_document) if state.prd_document else None,
            len(state.feature_list),
        )

        graph_input: _QaGraphState = {
            "ctx": {
                "feature_str": "\n".join(state.feature_list),
                "orig_db_schema": self._safe_truncate(state.db_schema or "", _MAX_SCHEMA_CHARS, "db_schema"),
                "orig_api_spec": self._safe_truncate(state.api_spec or "", _MAX_SCHEMA_CHARS, "api_spec"),
                "orig_prd_document": (state.prd_document or "")[:3000] or "(PRD 없음)",
                "api_completeness_hint": api_completeness_hint,
                "db_completeness_hint": db_completeness_hint,
                "prd_completeness_hint": prd_completeness_hint,
                "dump": dump,
            },
            "db_draft": state.db_schema or "{}",
            "db_round": 0,
            "db_approved": False,
            "db_issues": [],
            "db_feedback": "",
            "api_draft": state.api_spec or "{}",
            "api_round": 0,
            "api_approved": False,
            "api_issues": [],
            "api_feedback": "",
            "prd_draft": state.prd_document or "{}",
            "prd_round": 0,
            "prd_approved": False,
            "prd_issues": [],
            "prd_feedback": "",
        }

        result = await self._graph.ainvoke(graph_input)

        # DBA와 동일한 결정론적 백스톱을 QA 최종 db에도 적용 — 리뷰 재작성이 문장형 테이블명이나
        # 컬럼 0개 테이블을 재발시킬 수 있으므로 마지막 출력 직전에 한 번 더 정규화한다.
        db_draft = result.get("db_draft", "") or "{}"
        db_parsed = try_parse_json(db_draft)
        if isinstance(db_parsed, dict) and isinstance(db_parsed.get("tables"), list):
            db_parsed["tables"] = sanitize_tables(db_parsed["tables"])
            db_draft = json.dumps(db_parsed, ensure_ascii=False)

        # API도 동일 — 리뷰 재작성이 필드 없는 불완전 엔드포인트를 남길 수 있어 최종 보강
        api_draft = result.get("api_draft", "") or "{}"
        api_parsed = try_parse_json(api_draft)
        if isinstance(api_parsed, dict) and isinstance(api_parsed.get("endpoints"), list):
            api_parsed["endpoints"] = _normalize_endpoints(api_parsed["endpoints"])
            api_draft = json.dumps(api_parsed, ensure_ascii=False)

        # LLM 리뷰가 놓쳤을 수 있는 결함을 결정론적 체크로 최종본에 대해 한 번 더 검증 —
        # LLM 판정만으로는 매 실행마다 탐지율 편차가 크므로 이를 보완한다.
        db_issues = list(result.get("db_issues", []))
        api_issues = list(result.get("api_issues", []))
        prd_issues = list(result.get("prd_issues", []))
        db_issues += [
            i for i in self._check_db_completeness(
                try_parse_json(db_draft), state.feature_list,
            )
            if i not in db_issues
        ]
        api_issues += [
            i for i in self._check_api_completeness(
                try_parse_json(api_draft), state.feature_list,
            )
            if i not in api_issues
        ]
        prd_issues += [
            i for i in self._check_prd_completeness(
                try_parse_json(result.get("prd_draft", "")), len(state.feature_list),
            )
            if i not in prd_issues
        ]
        total_issues = len(db_issues) + len(api_issues) + len(prd_issues)
        quality_score = self._compute_quality_score(True, total_issues)

        forced = [
            d for d in ("db", "api", "prd")
            if result.get(f"{d}_round", 0) >= _MAX_ROUNDS and result.get(f"{d}_issues")
        ]
        status = "QA 완료 — 전 항목 승인"
        if forced:
            status += f" (강제 승인 도메인: {', '.join(forced)}, 잔존 결함 존재)"
        logger.info(
            "QA 완료 — DB:%d회/%d결함, API:%d회/%d결함, PRD:%d회/%d결함 (qualityScore=%.2f)",
            result.get("db_round", 0), len(db_issues),
            result.get("api_round", 0), len(api_issues),
            result.get("prd_round", 0), len(prd_issues),
            quality_score,
        )

        return state.copy(
            db_schema=db_draft,
            api_spec=api_draft,
            prd_document=result["prd_draft"],
            qa_quality_score=quality_score,
            qa_db_issues=db_issues,
            qa_api_issues=api_issues,
            qa_prd_issues=prd_issues,
            last_validation_error="",
            status_message=status,
        )

    def _make_reviewer_node(self, domain: str):
        async def reviewer(state: dict) -> dict:
            ctx = state["ctx"]
            draft = state.get(f"{domain}_draft") or "{}"
            feedback = state.get(f"{domain}_feedback", "")
            feedback_section = (
                f"\n[이전 manager 피드백 — 반드시 반영]\n{feedback}" if feedback else ""
            )

            if domain == "db":
                prompt = DB_REVIEW_PROMPT.format(
                    draft=draft, sibling=ctx["orig_api_spec"], feature_str=ctx["feature_str"],
                    feedback_section=feedback_section, leniency=_LENIENCY_NOTE,
                    completeness_hint=ctx["db_completeness_hint"],
                )
            elif domain == "api":
                prompt = API_REVIEW_PROMPT.format(
                    draft=draft, sibling=ctx["orig_db_schema"], prd_summary=ctx["orig_prd_document"],
                    feature_str=ctx["feature_str"], feedback_section=feedback_section,
                    completeness_hint=ctx["api_completeness_hint"], leniency=_LENIENCY_NOTE,
                )
            else:
                prompt = PRD_REVIEW_PROMPT.format(
                    draft=draft, db_sibling=ctx["orig_db_schema"], api_sibling=ctx["orig_api_spec"],
                    feature_str=ctx["feature_str"], feedback_section=feedback_section, leniency=_LENIENCY_NOTE,
                    completeness_hint=ctx["prd_completeness_hint"],
                )

            round_ = state.get(f"{domain}_round", 0)
            data = None
            for parse_attempt in range(2):
                raw = await self._call(prompt, max_tokens=16384, enable_thinking=False)
                if ctx.get("dump"):
                    ctx["dump"].log_raw(f"QA_{domain.upper()}_REVIEWER", round_ + 1, raw)
                data = try_parse_json(raw)
                if data and isinstance(data, dict):
                    break
                logger.warning(
                    "%s reviewer 파싱 실패 (round %d, 시도 %d) — 재시도",
                    domain, round_ + 1, parse_attempt + 1,
                )

            issues: list = []
            fixed = draft
            if data and isinstance(data, dict):
                issues = [str(i) for i in data.get("issues", [])]
                fixed_data = data.get("fixed")
                if isinstance(fixed_data, dict):
                    if domain == "db":
                        fixed_data = _to_array_schema(fixed_data)
                    orig_count = _count_items(domain, try_parse_json(draft))
                    new_count = _count_items(domain, fixed_data)
                    # QA는 정합성 교정 단계이지 가지치기 단계가 아니다 — 항목 수가 원본보다
                    # 줄어드는 것은 (EXAONE 재생성 시 테이블/엔드포인트/기능 누락) 항상 결함으로
                    # 취급하고 원본을 유지한다. 원본과 같거나 많을 때만 수정본을 채택한다.
                    if orig_count and new_count is not None and new_count < orig_count:
                        logger.warning(
                            "%s reviewer의 fixed가 원본보다 항목이 줄어듦(%d → %d, EXAONE 누락 추정) — 원본 유지",
                            domain, orig_count, new_count,
                        )
                    else:
                        fixed = json.dumps(fixed_data, ensure_ascii=False)
            else:
                logger.warning("%s reviewer 파싱 실패 (round %d) — 이전 산출물 유지", domain, round_ + 1)

            return {f"{domain}_draft": fixed, f"{domain}_issues": issues}

        return reviewer

    def _make_check_node(self, domain: str):
        async def check(state: dict) -> dict:
            ctx = state["ctx"]
            round_ = state.get(f"{domain}_round", 0) + 1

            if round_ >= _MAX_ROUNDS:
                logger.warning("%s manager_check — 최대 라운드(%d) 도달, 강제 승인", domain, _MAX_ROUNDS)
                return {f"{domain}_round": round_, f"{domain}_approved": True, f"{domain}_feedback": ""}

            draft = state.get(f"{domain}_draft", "")
            issues = state.get(f"{domain}_issues", [])
            prompt = MANAGER_CHECK_PROMPT.format(
                domain_label=_DOMAIN_LABEL[domain],
                round=round_,
                draft=self._safe_truncate(draft, _MAX_SCHEMA_CHARS, f"{domain}_draft"),
                issues="\n".join(f"- {i}" for i in issues) or "(reviewer가 보고한 결함 없음)",
                criteria=_DOMAIN_CRITERIA[domain],
            )
            raw = await self._call(prompt, max_tokens=16384, enable_thinking=False)
            if ctx.get("dump"):
                ctx["dump"].log_raw(f"QA_{domain.upper()}_CHECK", round_, raw)

            data = try_parse_json(raw)
            if data and isinstance(data, dict):
                approved = bool(data.get("approved", True))
                feedback = str(data.get("feedback", "")) if not approved else ""
            else:
                # 파싱 실패를 승인으로 취급하지 않는다 — 다음 라운드에서 재작성을 강제하고,
                # 그래도 안 되면 위 _MAX_ROUNDS 도달 시 강제 승인 경로로 안전하게 빠진다.
                logger.warning("%s manager_check 파싱 실패 (round %d) — 미승인 처리, 재작성 유도", domain, round_)
                approved = False
                feedback = "이전 응답이 유효한 JSON으로 파싱되지 않았습니다. 반드시 JSON만 출력하세요."

            return {f"{domain}_round": round_, f"{domain}_approved": approved, f"{domain}_feedback": feedback}

        return check

    def _make_router(self, domain: str):
        def router(state: dict) -> str:
            return END if state.get(f"{domain}_approved") else f"{domain}_reviewer"

        return router

    async def _call(self, user_prompt: str, max_tokens: int, enable_thinking: bool) -> str:
        for attempt in range(3):
            try:
                response = await self._client.chat.completions.create(
                    model=self._model,
                    temperature=_TEMPERATURE,
                    top_p=_TOP_P,
                    presence_penalty=_PRESENCE_PENALTY,
                    max_tokens=max_tokens,
                    frequency_penalty=0.3,
                    messages=[
                        {"role": "system", "content": REVIEW_SYSTEM},
                        {"role": "user", "content": user_prompt},
                    ],
                    extra_body={"chat_template_kwargs": {"enable_thinking": enable_thinking}},
                )
                return response.choices[0].message.content or ""
            except (InternalServerError, APITimeoutError, APIConnectionError) as e:
                logger.warning("QA API 일시 오류 (attempt %d): %s — 재시도", attempt + 1, e)
                if attempt < 2:
                    await asyncio.sleep(10 * (attempt + 1))
                else:
                    logger.error("QA API 최종 실패")
                    return ""
        return ""

    # PRD 프롬프트(prd_agent.SECTION_PROMPTS)가 스스로 요구하는 최소 개수 기준
    _PRD_MIN_COUNTS = (
        ("userPersonas", 3),
        ("releaseSchedule", 6),
        ("kpi", 7),
    )

    def _check_prd_completeness(self, prd_doc, feature_count: int) -> list[str]:
        """PRD 필수 섹션의 최소 분량을 결정론적으로 검증 (LLM 판정에만 의존하지 않기 위한 보조 체크)."""
        if not isinstance(prd_doc, dict) or not prd_doc:
            return ["PRD 문서가 비어 있거나 파싱되지 않았습니다"]

        def count(field: str) -> int:
            val = prd_doc.get(field)
            return len(val) if isinstance(val, list) else 0

        issues: list[str] = []

        core_count = count("coreFeatures")
        if feature_count and core_count < feature_count:
            issues.append(f"coreFeatures가 {core_count}개뿐 — 기능 목록({feature_count}개)만큼 채워지지 않음")

        for field, minimum in self._PRD_MIN_COUNTS:
            n = count(field)
            if n < minimum:
                issues.append(f"{field}가 {n}개뿐 — 최소 {minimum}개 필요")

        return issues

    def _check_api_completeness(self, api_spec, feature_list: list[str]) -> list[str]:
        """API 스펙의 엔드포인트 수/인증 정보/기능별 커버리지가 충분한지 결정론적으로 검증."""
        if not isinstance(api_spec, dict) or not api_spec:
            return ["API 스펙이 비어 있거나 파싱되지 않았습니다"]

        feature_count = len(feature_list)
        issues: list[str] = []
        endpoints = api_spec.get("endpoints")
        n = len(endpoints) if isinstance(endpoints, list) else 0
        minimum = max(4, feature_count)
        if n < minimum:
            issues.append(f"endpoints가 {n}개뿐 — 기능 목록({feature_count}개) 기준 최소 {minimum}개 필요")
        if not (api_spec.get("authentication") or "").strip():
            issues.append("authentication 필드가 비어 있음")
        if isinstance(endpoints, list):
            missing = uncovered_features(feature_list, [ep.get("description", "") for ep in endpoints if isinstance(ep, dict)])
            if missing:
                issues.append(f"다음 기능에 대응하는 엔드포인트가 없어 보임: {missing}")

        return issues

    def _api_completeness_hint(self, api_spec, feature_list: list[str]) -> str:
        """api_reviewer 컨텍스트에 주입할 결정론적 결함 힌트."""
        issues = self._check_api_completeness(api_spec, feature_list)
        if not issues:
            return ""
        return "\n[결정론적 완전성 검사 결과 — 반드시 확인]\n" + "\n".join(f"- {i}" for i in issues)

    def _prd_completeness_hint(self, prd_doc, feature_count: int) -> str:
        """prd_reviewer 컨텍스트에 주입할 결정론적 결함 힌트."""
        issues = self._check_prd_completeness(prd_doc, feature_count)
        if not issues:
            return ""
        return "\n[결정론적 완전성 검사 결과 — 반드시 확인]\n" + "\n".join(f"- {i}" for i in issues)

    def _check_db_completeness(self, db_schema, feature_list: list[str]) -> list[str]:
        """DB 스키마의 테이블/컬럼/기능별 커버리지가 실질적으로 채워져 있는지 결정론적으로 검증."""
        if not isinstance(db_schema, dict) or not db_schema:
            return ["DB 스키마가 비어 있거나 파싱되지 않았습니다"]

        tables = db_schema.get("tables")
        if not isinstance(tables, list) or not tables:
            return ["tables가 비어 있음"]

        feature_count = len(feature_list)
        issues: list[str] = []
        empty = [t.get("name", "?") for t in tables if isinstance(t, dict) and not t.get("columns")]
        if empty:
            issues.append(f"컬럼이 0개인 테이블 존재: {empty}")
        if not db_schema.get("relationships"):
            issues.append("relationships가 비어 있음 — 테이블 간 관계가 정의되지 않음")
        minimum = min_table_count(feature_count)
        if feature_count and len(tables) < minimum:
            issues.append(f"테이블이 {len(tables)}개뿐 — 기능 목록({feature_count}개) 기준 최소 {minimum}개 필요")
        haystack = []
        for t in tables:
            if isinstance(t, dict):
                haystack.append(str(t.get("name", "")))
                haystack.append(str(t.get("description", "")))
        missing = uncovered_features(feature_list, haystack)
        if missing:
            issues.append(f"다음 기능을 저장할 테이블이 없어 보임: {missing}")

        return issues

    def _db_completeness_hint(self, db_schema, feature_list: list[str]) -> str:
        """db_reviewer 컨텍스트에 주입할 결정론적 결함 힌트."""
        issues = self._check_db_completeness(db_schema, feature_list)
        if not issues:
            return ""
        return "\n[결정론적 완전성 검사 결과 — 반드시 확인]\n" + "\n".join(f"- {i}" for i in issues)

    def _safe_truncate(self, text: str, max_chars: int, label: str) -> str:
        if len(text) <= max_chars:
            return text
        # 마지막 }, ] 경계에서 잘라서 JSON 파서가 일부라도 읽을 수 있게 함
        truncated = text[:max_chars]
        last_boundary = max(truncated.rfind("},"), truncated.rfind("},\n"), truncated.rfind("]"))
        if last_boundary > max_chars // 2:
            truncated = truncated[:last_boundary + 1]
        logger.warning("QA: %s 잘림 (%d → %d chars)", label, len(text), len(truncated))
        return truncated + "\n... (이하 생략, 위 내용으로만 검수)"

    def _compute_quality_score(self, passed: bool, issue_count: int) -> float:
        """QA 품질 점수 (0.0~1.0). rag-pipeline QaAgent.computeQualityScore와 동일한 룩업 테이블."""
        if passed:
            if issue_count == 0:
                return 1.00
            if issue_count <= 2:
                return 0.80
            return 0.60
        else:
            if issue_count <= 2:
                return 0.40
            return 0.20
