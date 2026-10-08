import asyncio
import json
import logging
import re
from typing import TypedDict

from langgraph.graph import StateGraph, START, END
from openai import AsyncOpenAI, InternalServerError, APITimeoutError, APIConnectionError

from phase2.agents.dba_agent import (
    _ensure_fk_references, _to_array_schema, dedupe_meta_tables, ensure_primary_keys, min_table_count,
    reconcile_fk_types, sanitize_tables, _normalize_special_references, _table_name_variants,
)
from phase2.agents.api_agent import _canonical_api_path, _normalize_endpoints
from phase2.agents.prd_agent import reconcile_core_features
from phase2.feature_coverage import strictly_uncovered_features, uncovered_features
from phase2.feature_registry import missing_db_contract_features
from phase2.feature_scope import backend_features
from phase2.feature_registry import normalize_feature_registry, registry_text
from phase2.json_utils import try_parse_json
from phase2.llm_runtime import LlmRuntime
from phase2.state import PipelineState

from phase2.agents.dba_agent import _build_name_lookup
from phase2.agents.api_agent import _endpoint_quality_issues, _feature_methods, _feature_resource_slug
from phase2.agent_contract import IssueSeverity, classify_issue, feature_relation_kind, requires_auth
from phase2.quality_rules import contamination_reasons, has_placeholder, kpi_basis_issues, near_duplicate, relevance_score, required_field_concepts, scoped_unique_columns

logger = logging.getLogger(__name__)

def _english_stems(value: str) -> set[str]:
    stems = set()
    for token in re.findall(r"[a-z]+", str(value or "").lower().replace("-", "_")):
        for suffix in ("ments", "ment", "ees", "ee", "ed", "ing", "ies", "s"):
            if token.endswith(suffix) and len(token) > len(suffix) + 2:
                token = token[:-len(suffix)]
                break
        stems.add(token)
    return stems

def _request_field_names(value) -> set[str]:
    if isinstance(value, dict):
        return {str(name).lower() for name in value}
    return {
        match.lower()
        for match in re.findall(
            r"(?:^|[,;{]\s*)['\"]?([a-z][a-z0-9_]*)['\"]?\s*:",
            str(value or ""), re.I,
        )
    }

def _endpoint_table(endpoint: dict, tables: dict[str, dict], mappings: list | None = None) -> tuple[str, dict] | tuple[None, None]:
    feature_id = str(endpoint.get("featureId") or "")
    feature_name = str(endpoint.get("featureName") or "")
    declared = {
        str(item.get("table") or "") for item in mappings or [] if isinstance(item, dict)
        and ((feature_id and item.get("featureId") == feature_id)
             or (not feature_id and feature_name and item.get("featureName") == feature_name))
    }
    if len(declared) == 1:
        name = next(iter(declared))
        if name in tables:
            return name, tables[name]
    path = str(endpoint.get("path") or "")
    slug = path.removeprefix("/api/v1/").split("/", 1)[0].replace("-", "_")
    if slug in tables:
        return slug, tables[slug]
    slug_stems = _english_stems(slug)
    lexical = [
        (len(slug_stems & _english_stems(name)), name, table)
        for name, table in tables.items()
    ]
    lexical_best = max((score for score, _name, _table in lexical), default=0)
    lexical_winners = [(name, table) for score, name, table in lexical if score == lexical_best and score > 0]
    if len(lexical_winners) == 1:
        return lexical_winners[0]
    meaning = str(endpoint.get("featureName") or endpoint.get("description") or "")
    scored = [
        (max(
            relevance_score(meaning, f"{name} {table.get('description', '')}"),
            relevance_score(str(table.get("description") or ""), meaning),
        ), name, table)
        for name, table in tables.items()
    ]
    best = max((score for score, _name, _table in scored), default=0.0)
    winners = [(name, table) for score, name, table in scored if score == best and score >= 0.25]
    return winners[0] if len(winners) == 1 else (None, None)



def _replace_named_items(base: list, patches: list, key_fn) -> list:
    """식별 키가 같은 항목은 교체하고 새 항목은 추가한다. 원본 항목은 절대 삭제하지 않는다."""
    result = [item for item in (base or []) if isinstance(item, dict)]
    positions = {key_fn(item): i for i, item in enumerate(result) if key_fn(item)}
    for patch in patches or []:
        if not isinstance(patch, dict):
            continue
        key = key_fn(patch)
        if not key:
            continue
        if key in positions:
            result[positions[key]] = patch
        else:
            positions[key] = len(result)
            result.append(patch)
    return result

def _apply_review_patches(domain: str, draft_data: dict, patches: dict) -> dict:
    """QA reviewer의 부분 patch를 결정론적으로 병합한다."""
    if not isinstance(draft_data, dict) or not isinstance(patches, dict):
        return draft_data
    result = dict(draft_data)

    if domain == "db":
        table_patches = patches.get("tables")
        if isinstance(table_patches, dict):
            table_patches = [dict(value, name=name) for name, value in table_patches.items() if isinstance(value, dict)]
        if isinstance(table_patches, list):
            normalized = _to_array_schema({"tables": table_patches}).get("tables", [])
            result["tables"] = _replace_named_items(
                result.get("tables") or [], normalized, lambda item: str(item.get("name") or "").strip(),
            )
        relationships = patches.get("relationships")
        if isinstance(relationships, list):
            existing = list(result.get("relationships") or [])
            for relationship in relationships:
                if relationship not in existing:
                    existing.append(relationship)
            result["relationships"] = existing
        return result

    if domain == "api":
        endpoint_patches = patches.get("endpoints")
        if isinstance(endpoint_patches, list):
            # Merge by the exact artifact identity. Canonicalizing only the
            # patch path here can append a duplicate instead of replacing it;
            # the owning API agent normalizes the completed document.
            result["endpoints"] = _replace_named_items(
                result.get("endpoints") or [], endpoint_patches,
                lambda item: f"{str(item.get('method') or '').upper()} {str(item.get('path') or '').strip()}",
            )
        if patches.get("authentication"):
            result["authentication"] = patches["authentication"]
        return result

    if domain == "prd":
        list_fields = {"coreFeatures", "kpi", "userPersonas", "releaseSchedule", "goals"}
        label_fields = {
            "coreFeatures": "name", "kpi": "metric", "userPersonas": "name",
            "releaseSchedule": "milestone",
        }
        for field, value in patches.items():
            if field in list_fields and isinstance(value, list):
                current = result.get(field) if isinstance(result.get(field), list) else []
                if field == "goals":
                    result[field] = current + [item for item in value if item not in current]
                else:
                    label = label_fields[field]
                    result[field] = _replace_named_items(
                        current, value, lambda item, label=label: str(item.get(label) or "").strip(),
                    )
            elif value is not None:
                result[field] = value
        return result

    return result


def _derive_patches_from_full_output(domain: str, draft_data: dict, fixed: dict) -> dict:
    """모델이 금지된 fixed 전체 문서를 반환해도 삭제 없는 최소 patch로 축약한다."""
    if not isinstance(fixed, dict):
        return {}
    if domain == "db":
        normalized = _to_array_schema(fixed)
        return {key: normalized[key] for key in ("tables", "relationships") if isinstance(normalized.get(key), list)}
    if domain == "api":
        return {key: fixed[key] for key in ("endpoints", "authentication") if key in fixed}
    if domain == "prd":
        patches = {}
        for key, value in fixed.items():
            if draft_data.get(key) != value:
                patches[key] = value
        return patches
    return {}


def _same_table_name(left: str, right: str) -> bool:
    """Compare canonical table names with regular plural variants (classes/class)."""
    return bool(left and right) and (
        left in _table_name_variants(right) or right in _table_name_variants(left)
    )

_MAX_SCHEMA_CHARS = 60000  # GPT-5.4 mini 컨텍스트 안에서 전체 DB/API 구조를 보존
_MAX_ROUNDS = 2  # 도메인당 reviewer↔manager_check 최대 라운드
_LIGHTWEIGHT_GATE_ENABLED = True  # 엄격한 최종 검증은 Phase3 SchemaValidator가 담당

# 생성 샘플링 파라미터
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


_PRD_COUNTED_FIELDS = ("coreFeatures", "kpi", "userPersonas", "releaseSchedule", "goals")


def _count_items_per_field(domain: str, data) -> dict[str, int]:
    """축소 판정을 섹션별로 하기 위한 필드별 항목 수.
    PRD를 합계로만 보면 coreFeatures가 2개 줄어도 kpi가 2개 늘어 상쇄되어 통과한다
    (실측: coreFeatures 9 → 7). 필드 하나라도 줄면 축소로 본다."""
    if not isinstance(data, dict):
        return {}
    if domain == "db":
        tables = data.get("tables")
        return {"tables": len(tables) if isinstance(tables, (list, dict)) else 0}
    if domain == "api":
        endpoints = data.get("endpoints")
        return {"endpoints": len(endpoints) if isinstance(endpoints, list) else 0}
    if domain == "prd":
        return {f: len(v) for f in _PRD_COUNTED_FIELDS if isinstance(v := data.get(f), list)}
    return {}


def _shrunk_fields(orig: dict[str, int], new: dict[str, int]) -> dict[str, tuple[int, int]]:
    """원본에 있던 필드 중 항목 수가 줄어든 것만 추린다. 원본에 없던 필드는 판단하지 않는다."""
    return {
        field: (count, new[field])
        for field, count in orig.items()
        if count and field in new and new[field] < count
    }


class _QaGraphState(TypedDict, total=False):
    ctx: dict
    db_draft: str
    db_round: int
    db_approved: bool
    db_forced: bool
    db_issues: list
    db_feedback: str
    api_draft: str
    api_round: int
    api_approved: bool
    api_forced: bool
    api_issues: list
    api_feedback: str
    prd_draft: str
    prd_round: int
    prd_approved: bool
    prd_forced: bool
    prd_issues: list
    prd_feedback: str


class QaAgent:

    def __init__(
        self,
        client: AsyncOpenAI,
        model: str = "gpt-5.4-mini",
        runtime: LlmRuntime | None = None,
    ):
        self._client = client
        self._model = model
        self._runtime = runtime
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
        if _LIGHTWEIGHT_GATE_ENABLED:
            logger.info("QA 에이전트 시작 (경량 게이트 — Phase3 검증 이관)")
            return self._execute_lightweight_gate(state)

        logger.info("QA 에이전트 시작 (reviewer↔manager 도메인별 독립 루프)")

        api_completeness_hint = self._api_completeness_hint(
            try_parse_json(state.api_spec) if state.api_spec else None,
            state.feature_list,
        )
        db_completeness_hint = self._db_completeness_hint(
            try_parse_json(state.db_schema) if state.db_schema else None,
            state.feature_list,
            state.feature_registry,
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
                "orig_prd_document": self._safe_truncate(state.prd_document or "", _MAX_SCHEMA_CHARS, "prd_document") or "(PRD 없음)",
                "api_completeness_hint": api_completeness_hint,
                "db_completeness_hint": db_completeness_hint,
                "prd_completeness_hint": prd_completeness_hint,
                "dump": dump,
            },
            "db_draft": state.db_schema or "{}",
            "db_round": 0,
            "db_approved": False,
            "db_forced": False,
            "db_issues": [],
            "db_feedback": "",
            "api_draft": state.api_spec or "{}",
            "api_round": 0,
            "api_approved": False,
            "api_forced": False,
            "api_issues": [],
            "api_feedback": "",
            "prd_draft": state.prd_document or "{}",
            "prd_round": 0,
            "prd_approved": False,
            "prd_forced": False,
            "prd_issues": [],
            "prd_feedback": "",
        }

        result = await self._graph.ainvoke(graph_input)

        # DBA와 동일한 결정론적 백스톱을 QA 최종 db에도 적용 — 리뷰 재작성이 문장형 테이블명이나
        # 컬럼 0개 테이블, FK 타입 불일치를 재발시킬 수 있으므로 마지막 출력 직전에 한 번 더 정규화한다.
        db_draft = result.get("db_draft", "") or "{}"
        db_parsed = try_parse_json(db_draft)
        if isinstance(db_parsed, dict) and isinstance(db_parsed.get("tables"), list):
            db_parsed["tables"] = ensure_primary_keys(
                dedupe_meta_tables(sanitize_tables(db_parsed["tables"]))
            )
            _normalize_special_references(db_parsed["tables"])
            _ensure_fk_references(
                db_parsed["tables"],
                normalize_feature_registry(state.feature_registry, state.feature_list, preserve_extra=True),
            )
            db_parsed["tables"] = reconcile_fk_types(db_parsed["tables"])
            db_draft = json.dumps(db_parsed, ensure_ascii=False)

        # API도 동일 — 리뷰 재작성이 필드 없는 불완전 엔드포인트를 남길 수 있어 최종 보강
        api_draft = result.get("api_draft", "") or "{}"
        api_parsed = try_parse_json(api_draft)
        if isinstance(api_parsed, dict) and isinstance(api_parsed.get("endpoints"), list):
            api_parsed["endpoints"] = _normalize_endpoints(api_parsed["endpoints"])
            api_draft = json.dumps(api_parsed, ensure_ascii=False)

        # PRD도 동일 — 리뷰 재작성이 우선순위를 전부 P0으로 되돌리거나 비워둘 수 있어
        # MVP 범위 기준 재배정을 최종 출력 직전에 한 번 더 적용한다
        prd_draft = result.get("prd_draft", "") or "{}"
        prd_parsed = try_parse_json(prd_draft)
        if isinstance(prd_parsed, dict) and isinstance(prd_parsed.get("coreFeatures"), list):
            prd_parsed["coreFeatures"] = reconcile_core_features(
                prd_parsed["coreFeatures"], prd_parsed.get("mvpScope"), state.feature_list,
            )
            prd_draft = json.dumps(prd_parsed, ensure_ascii=False)

        # LLM 리뷰가 놓쳤을 수 있는 결함을 결정론적 체크로 최종본에 대해 한 번 더 검증 —
        # LLM 판정만으로는 매 실행마다 탐지율 편차가 크므로 이를 보완한다.
        db_issues = list(result.get("db_issues", []))
        api_issues = list(result.get("api_issues", []))
        prd_issues = list(result.get("prd_issues", []))
        deterministic_db = self._check_db_completeness(
            try_parse_json(db_draft), state.feature_list, state.feature_registry,
        )
        deterministic_api = self._check_api_completeness(try_parse_json(api_draft), state.feature_list)
        deterministic_prd = self._check_prd_completeness(try_parse_json(prd_draft), len(state.feature_list), state.market_research or "")
        db_issues += [i for i in deterministic_db if i not in db_issues]
        api_issues += [i for i in deterministic_api if i not in api_issues]
        prd_issues += [i for i in deterministic_prd if i not in prd_issues]
        cross_db, cross_api, cross_prd = self._check_cross_artifacts(
            try_parse_json(db_draft), try_parse_json(api_draft), try_parse_json(prd_draft),
        )
        semantic_db, semantic_api, semantic_prd = self._check_cross_document_semantics(
            try_parse_json(prd_draft), try_parse_json(db_draft), try_parse_json(api_draft),
            state.feature_list, state.context_prompt or "",
        )
        cross_db.extend(semantic_db)
        cross_api.extend(semantic_api)
        cross_prd.extend(semantic_prd)
        db_issues.extend(x for x in cross_db if x not in db_issues)
        api_issues.extend(x for x in cross_api if x not in api_issues)
        prd_issues.extend(x for x in cross_prd if x not in prd_issues)
        deterministic_db.extend(x for x in cross_db if x not in deterministic_db)
        deterministic_api.extend(x for x in cross_api if x not in deterministic_api)
        deterministic_prd.extend(x for x in cross_prd if x not in deterministic_prd)
        forced = [
            d for d in ("db", "api", "prd")
            if result.get(f"{d}_forced", False)
        ]
        total_issues = len(db_issues) + len(api_issues) + len(prd_issues)
        classified = self._classify_issues(
            db_issues, api_issues, prd_issues,
            deterministic=set(deterministic_db + deterministic_api + deterministic_prd),
        )
        generation_db = [item for item in state.generation_blockers if str(item).startswith("DBA_")]
        generation_api = [item for item in state.generation_blockers if str(item).startswith("API_")]
        generation_other = [item for item in state.generation_blockers if item not in generation_db + generation_api]
        classified["db_blockers"] = list(dict.fromkeys(generation_db + classified["db_blockers"]))
        classified["api_blockers"] = list(dict.fromkeys(generation_api + classified["api_blockers"]))
        classified["prd_blockers"] = list(dict.fromkeys(generation_other + classified["prd_blockers"]))
        blockers = classified["db_blockers"] + classified["api_blockers"] + classified["prd_blockers"]
        passed = not forced and not blockers and not state.generation_blockers
        quality_score = self._compute_quality_score(passed, total_issues)

        status = "QA 완료 — 전 항목 승인" if passed else "QA 완료 — 미승인 항목 있음"
        if forced:
            status += f" (최대 검수 횟수 도달: {', '.join(forced)})"
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
            prd_document=prd_draft,
            qa_quality_score=quality_score,
            qa_approved=passed,
            qa_db_issues=db_issues,
            qa_api_issues=api_issues,
            qa_prd_issues=prd_issues,
            qa_db_blockers=classified["db_blockers"],
            qa_api_blockers=classified["api_blockers"],
            qa_prd_blockers=classified["prd_blockers"],
            qa_db_warnings=classified["db_warnings"],
            qa_api_warnings=classified["api_warnings"],
            qa_prd_warnings=classified["prd_warnings"],
            qa_blocker_details=self._build_blocker_details(classified),
            qa_issue_details=self._structured_issues(db_issues, api_issues, prd_issues),
            qa_repair_issues=[
                issue for issue in self._structured_issues(db_issues, api_issues, prd_issues)
                if issue["severity"] in {IssueSeverity.ERROR.value, IssueSeverity.BLOCKER.value}
            ],
            last_validation_error="",
            status_message=status,
        )

    def _execute_lightweight_gate(self, state: PipelineState) -> PipelineState:
        """Phase2 QA는 산출물 생성 완료 여부만 확인하고, 엄격한 판단은 Phase3로 넘긴다."""
        db_draft, api_draft, prd_draft = state.db_schema, state.api_spec, state.prd_document
        db_parsed = try_parse_json(db_draft)
        api_parsed = try_parse_json(api_draft)
        prd_parsed = try_parse_json(prd_draft)

        contract_db, contract_api, contract_prd = self._check_registry_contracts(
            normalize_feature_registry(
                state.feature_registry, state.feature_list, preserve_extra=True,
            ),
            try_parse_json(db_draft), try_parse_json(api_draft), try_parse_json(prd_draft),
        )

        db_issues = self._check_db_completeness(
            try_parse_json(db_draft), state.feature_list, state.feature_registry,
        )
        api_issues = self._check_api_completeness(try_parse_json(api_draft), state.feature_list)
        prd_issues = self._check_prd_completeness(try_parse_json(prd_draft), len(state.feature_list), state.market_research or "")
        cross_db, cross_api, cross_prd = self._check_cross_artifacts(
            try_parse_json(db_draft), try_parse_json(api_draft), try_parse_json(prd_draft),
        )
        semantic_db, semantic_api, semantic_prd = self._check_cross_document_semantics(
            try_parse_json(prd_draft), try_parse_json(db_draft), try_parse_json(api_draft),
            state.feature_list, state.context_prompt or "",
        )
        cross_db.extend(semantic_db)
        cross_api.extend(semantic_api)
        cross_prd.extend(semantic_prd)
        db_issues.extend(x for x in cross_db if x not in db_issues)
        api_issues.extend(x for x in cross_api if x not in api_issues)
        prd_issues.extend(x for x in cross_prd if x not in prd_issues)
        db_issues.extend(x for x in contract_db if x not in db_issues)
        api_issues.extend(x for x in contract_api if x not in api_issues)
        prd_issues.extend(x for x in contract_prd if x not in prd_issues)
        db_issues.extend(x for x in self._check_domain_scope(db_parsed, prd_parsed) if x not in db_issues)
        db_issues.extend(x for x in self._check_duplicate_domains(db_parsed) if x not in db_issues)
        total_issues = len(db_issues) + len(api_issues) + len(prd_issues)
        score = max(0.0, 1.0 - min(total_issues, 4) * 0.1)
        # Lightweight gate의 모든 이슈는 결정론적 검사에서 나온 계약 결함이다.
        classified = self._classify_issues(
            db_issues, api_issues, prd_issues,
            deterministic=set(db_issues + api_issues + prd_issues),
        )
        generation_db = [item for item in state.generation_blockers if str(item).startswith("DBA_")]
        generation_api = [item for item in state.generation_blockers if str(item).startswith("API_")]
        generation_other = [item for item in state.generation_blockers if item not in generation_db + generation_api]
        classified["db_blockers"] = list(dict.fromkeys(generation_db + classified["db_blockers"]))
        classified["api_blockers"] = list(dict.fromkeys(generation_api + classified["api_blockers"]))
        classified["prd_blockers"] = list(dict.fromkeys(generation_other + classified["prd_blockers"]))
        blockers = classified["db_blockers"] + classified["api_blockers"] + classified["prd_blockers"]
        approved = not blockers and not state.generation_blockers

        logger.info(
            "QA 경량 게이트 완료 — DB:%d결함, API:%d결함, PRD:%d결함 (qualityScore=%.2f, Phase3 이관)",
            len(db_issues),
            len(api_issues),
            len(prd_issues),
            score,
        )

        return state.copy(
            db_schema=db_draft,
            api_spec=api_draft,
            prd_document=prd_draft,
            qa_quality_score=score,
            qa_approved=approved,
            qa_db_issues=db_issues,
            qa_api_issues=api_issues,
            qa_prd_issues=prd_issues,
            qa_db_blockers=classified["db_blockers"],
            qa_api_blockers=classified["api_blockers"],
            qa_prd_blockers=classified["prd_blockers"],
            qa_db_warnings=classified["db_warnings"],
            qa_api_warnings=classified["api_warnings"],
            qa_prd_warnings=classified["prd_warnings"],
            qa_blocker_details=self._build_blocker_details(classified),
            qa_issue_details=self._structured_issues(db_issues, api_issues, prd_issues),
            qa_repair_issues=[
                issue for issue in self._structured_issues(db_issues, api_issues, prd_issues)
                if issue["severity"] in {IssueSeverity.ERROR.value, IssueSeverity.BLOCKER.value}
            ],
            last_validation_error="",
            status_message="QA 경량 게이트 완료 — Phase 3 구조 검증으로 이관",
        )

    @staticmethod
    def _build_blocker_details(classified: dict[str, list[str]]) -> list[dict]:
        details = []
        for category in ("db", "api", "prd"):
            for problem in classified.get(f"{category}_blockers", []):
                text = str(problem)
                detail = {
                    "category": category,
                    "problem": text,
                    "repairTarget": category,
                }
                endpoint = re.search(r"\b(GET|POST|PUT|PATCH|DELETE)\s+(/api/[^\s:]+)", text)
                if endpoint:
                    detail["artifactKey"] = f"{endpoint.group(1)} {endpoint.group(2)}"
                if category == "db":
                    table = re.search(r"(?:DB 스키마:\s*)?([a-z][a-z0-9_]*)", text)
                    if table:
                        detail["artifactKey"] = table.group(1)
                feature = re.match(r"([\w.-]+):", text)
                if feature:
                    detail["featureId"] = feature.group(1)
                details.append(detail)
        return details

    @staticmethod
    def _classify_issues(
        db_issues: list[str],
        api_issues: list[str],
        prd_issues: list[str],
        deterministic: set[str] | None = None,
    ) -> dict[str, list[str]]:
        """Separate contract-breaking defects from quality warnings.

        The full issue lists remain available for review, but only blockers stop
        Phase3/4. External and polymorphic references are warnings when they are
        explicitly identified as such; unresolved internal references remain blockers.
        """
        blocker_tokens = (
            "파싱", "비어", "누락", "중복", "불일치", "잘못된", "없어", "찾을 수",
            "참조 대상", "대응하는", "featureid", "feature id", "relationship",
            "기능 목록", "foreign_key", " fk", "엔드포인트가 없어",
        )
        warning_tokens = (
            "설명", "kpi", "최소", "supporting", "선택", "권장", "표현", "문구",
            "외부", "external", "polymorphic", "다형성", "보강 필요",
        )

        def split(items: list[str]) -> tuple[list[str], list[str]]:
            blockers, warnings = [], []
            for item in items:
                text = str(item)
                lower = text.lower()
                if deterministic is not None and text in deterministic:
                    if classify_issue(text) == IssueSeverity.WARNING:
                        warnings.append(text)
                    else:
                        blockers.append(text)
                    continue
                if any(token in lower for token in ("외부 id", "external id", "polymorphic", "다형성")):
                    warnings.append(text)
                elif any(token in lower for token in blocker_tokens):
                    blockers.append(text)
                elif any(token in lower for token in warning_tokens):
                    warnings.append(text)
                else:
                    warnings.append(text)
            return blockers, warnings

        result = {}
        for name, items in (("db", db_issues), ("api", api_issues), ("prd", prd_issues)):
            blockers, warnings = split(items)
            result[f"{name}_blockers"] = blockers
            result[f"{name}_warnings"] = warnings
        return result

    @staticmethod
    def _check_duplicate_domains(db: dict | None) -> list[str]:
        names = {str(item.get("name") or "").lower() for item in (db or {}).get("tables", []) if isinstance(item, dict)}
        issues = []
        for left, right, reason in (
            ("transactions", "orders", "주문과 거래 원장 책임이 중복될 수 있음"),
            ("appointments", "trade_appointments", "거래 약속 테이블이 중복됨"),
        ):
            if left in names and right in names:
                issues.append(f"DB 도메인 중복: {left}/{right} — {reason}")
        return issues

    @staticmethod
    def _check_domain_scope(db: dict | None, prd: dict | None) -> list[str]:
        excluded = (prd or {}).get("mvpScope", {}).get("excluded", []) if isinstance((prd or {}).get("mvpScope"), dict) else []
        excluded_text = " ".join(str(item).casefold() for item in excluded)
        if not excluded_text:
            return []
        issues = []
        for table in (db or {}).get("tables", []):
            if not isinstance(table, dict):
                continue
            haystack = f"{table.get('name', '')} {table.get('description', '')}".casefold()
            if any(token in haystack for token in ("review", "rating", "평가", "후기", "리뷰") if token in excluded_text):
                issues.append(f"DB 범위 위반: MVP 제외 기능을 위한 테이블 {table.get('name')}가 포함됨")
        return issues

    @staticmethod
    def _check_registry_contracts(registry, db, api, prd):
        """PM 계약의 명시된 부분만 최종 산출물과 결정론적으로 대조한다."""
        db_issues, api_issues, prd_issues = [], [], []
        endpoints = [e for e in (api or {}).get("endpoints", []) if isinstance(e, dict)]
        endpoint_keys = {
            (str(e.get("method", "GET")).upper(), _canonical_api_path(e.get("path")))
            for e in endpoints
        }
        registry_ids = {str(item.get("featureId") or item.get("id") or "").strip() for item in registry or []}
        for endpoint in endpoints:
            feature_id = str(endpoint.get("featureId") or "").strip()
            if not feature_id or feature_id not in registry_ids:
                api_issues.append(
                    f"API endpoint {endpoint.get('method', 'GET')} {endpoint.get('path', '')}: Registry featureId 매핑 누락"
                )
        tables = {str(t.get("name")) for t in (db or {}).get("tables", []) if isinstance(t, dict) and t.get("name")}
        tables_by_name = {
            str(t.get("name")): t for t in (db or {}).get("tables", [])
            if isinstance(t, dict) and t.get("name")
        }
        table_text = " ".join(json.dumps(t, ensure_ascii=False) for t in (db or {}).get("tables", []))
        core = [f for f in (prd or {}).get("coreFeatures", []) if isinstance(f, dict)]
        core_ids = {str(f.get("featureId") or "").strip() for f in core}
        for item in registry or []:
            fid = str(item.get("featureId") or item.get("id") or "").strip()
            name = str(item.get("name") or fid)
            for contract in item.get("apiContract") or []:
                if isinstance(contract, dict):
                    key = (
                        str(contract.get("method", "GET")).upper(),
                        _canonical_api_path(contract.get("path")),
                    )
                    if key[1] and key not in endpoint_keys:
                        api_issues.append(f"{fid}: apiContract {key[0]} {key[1]} 누락")
            db_contract = item.get("dbContract") or {}
            for table in db_contract.get("tables", []) if isinstance(db_contract, dict) else []:
                table_name = table.get("name") or table.get("table") if isinstance(table, dict) else table
                table_name = str(table_name or "").strip()
                if table_name and table_name not in tables:
                    db_issues.append(f"{fid}: dbContract 테이블 {table_name} 누락")
                elif table_name:
                    mapped_ids = tables_by_name.get(table_name, {}).get("featureIds", [])
                    if fid not in mapped_ids:
                        db_issues.append(f"{fid}: DB 테이블 {table_name} featureId 매핑 불일치")
            for fk in db_contract.get("foreignKeys", []) if isinstance(db_contract, dict) else []:
                text = json.dumps(fk, ensure_ascii=False) if isinstance(fk, dict) else str(fk)
                if isinstance(fk, dict):
                    references = fk.get("references") if isinstance(fk.get("references"), dict) else {}
                    target = str(references.get("table") or "").strip()
                    if target and target not in tables and not any(_same_table_name(target, name) for name in tables):
                        db_issues.append(f"{fid}: dbContract FK 대상 {target} 누락")
                if "->" in text:
                    target = text.rsplit("->", 1)[-1].strip().split(".", 1)[0]
                    if target and target not in table_text and not any(
                        _same_table_name(target, name) for name in tables
                    ):
                        db_issues.append(f"{fid}: dbContract FK 대상 {target} 누락")
            source = str(item.get("source") or "").strip().casefold()
            priority = str(item.get("priority") or "").strip().upper()
            # Supporting 기능은 Feature Spec의 책임이다. PRD coreFeatures에
            # 없다는 이유만으로 blocker로 만들면 supporting 설계가 실패한다.
            requires_prd_core = source in {"prd_core", "core"} or priority == "P0"
            if fid and core and requires_prd_core and fid not in core_ids and name not in json.dumps(core, ensure_ascii=False):
                prd_issues.append(f"{fid}: PRD coreFeatures에 featureId/name 누락")
        return db_issues, api_issues, prd_issues

    @staticmethod
    def _check_cross_artifacts(db, api, prd) -> tuple[list[str], list[str], list[str]]:
        """PRD·DB·API 사이의 명칭/관계 단절을 결정론적으로 찾는다."""
        db_issues: list[str] = []
        api_issues: list[str] = []
        prd_issues: list[str] = []
        tables = [t for t in (db or {}).get("tables", []) if isinstance(t, dict)]
        table_names = {str(t.get("name")) for t in tables if t.get("name")}

        relationships = (db or {}).get("relationships")
        if isinstance(relationships, list):
            normalized = [re.sub(r"\s+", " ", str(r)).strip().casefold() for r in relationships]
            if len(normalized) != len(set(normalized)):
                db_issues.append("중복 relationship이 존재합니다")

        for table in tables:
            for column in table.get("columns", []):
                if not isinstance(column, dict) or not str(column.get("name", "")).endswith("_id"):
                    continue
                column_name = str(column.get("name", ""))
                # These identifiers are intentionally external or polymorphic;
                # requiring an internal table would create a false blocker.
                if (
                    column_name in {"target_id", "feature_id", "provider_message_id"}
                    or column_name.startswith("target_")
                ):
                    continue
                constraints = str(column.get("constraints", "")).upper()
                if "FOREIGN_KEY" not in constraints:
                    continue
                reference = re.search(r"\bREFERENCES\s+([a-z][a-z0-9_]*)\s*\(", constraints, re.IGNORECASE)
                if reference:
                    candidates = {reference.group(1)}
                else:
                    candidates = set()
                # LLMs often emit singular aliases (user/reservation) in a
                # REFERENCES clause. If that alias is not an actual table,
                # fall back to the same deterministic column-name inference
                # used by DBA normalization.
                if not candidates & table_names:
                    stem = column_name[:-3]
                    candidates.update({
                        stem,
                        f"{stem}s",
                        stem.removesuffix("y") + "ies",
                    })
                    if stem in {
                        "owner", "renter", "borrower", "requester", "provider", "buyer", "seller",
                        "actor", "creator", "uploader", "assignee", "reviewer",
                    } or stem.endswith("_user") or stem.endswith("_by"):
                        candidates.update({"user", "users"})
                    # Prefixes such as target_equipment_id still refer to the
                    # canonical equipment table.
                    for table_name in table_names:
                        table_tail = table_name.rsplit("_", 1)[-1]
                        table_variants = _table_name_variants(table_tail)
                        if (
                            stem.endswith("_" + table_name)
                            or stem.endswith("_" + table_tail)
                            or stem in table_variants
                        ):
                            candidates.add(table_name)
                    if not candidates & table_names:
                        for table_name in table_names:
                            table_tail = table_name.rsplit("_", 1)[-1]
                            if stem in _table_name_variants(table_tail):
                                candidates.add(table_name)
                if not candidates & table_names:
                    db_issues.append(f"{table.get('name')}.{column['name']}의 FK 참조 대상을 찾을 수 없습니다")

        declared = set(re.findall(
            r"\b([a-z][a-z0-9_]*)\s*\(id\s+PK\b",
            json.dumps(prd or {}, ensure_ascii=False),
        ))
        missing = sorted(declared - table_names)
        if missing:
            prd_issues.append(f"PRD에 정의된 테이블이 DBA 스키마에 없습니다: {missing}")

        return db_issues, api_issues, prd_issues

    @staticmethod
    def _structured_issues(db_issues: list[str], api_issues: list[str], prd_issues: list[str]) -> list[dict]:
        """Convert QA findings into routing metadata without inventing a fix."""
        result: list[dict] = []
        patterns = {
            "DBA": re.compile(r"(?:누락:|FK 인덱스 누락:|참조 대상 없는 FK 컬럼:)\s*([^\s,]+)"),
            "API": re.compile(r"(?:누락:|API[^:]*:)\s*(?:(GET|POST|PUT|PATCH|DELETE)\s+)?([^\s,]+)"),
            "PRD": re.compile(r"(?:이탈:|충돌:|미완결:)\s*(.+)$"),
        }
        for agent, issues in (("DBA", db_issues), ("API", api_issues), ("PRD", prd_issues)):
            for message in issues:
                match = patterns[agent].search(message)
                target = "document"
                if match:
                    target = " ".join(part for part in match.groups() if part).strip() or "document"
                result.append({
                    "agent": agent,
                    "target": target,
                    "reason": message,
                    "severity": classify_issue(message).value,
                    "action": "patch",
                })
        return result

    def _check_cross_document_semantics(
        self, prd, db, api, feature_list: list[str], requirement_text: str = "",
    ):
        """Validate ownership, references and behavior across PRD, ERD and API."""
        db_issues: list[str] = []
        api_issues: list[str] = []
        prd_issues: list[str] = []
        if not isinstance(db, dict) or not isinstance(api, dict) or not isinstance(prd, dict):
            return db_issues, api_issues, prd_issues

        state_fields = {
            "quantity", "amount", "balance", "status", "state", "unit", "expiry_date",
            "scheduled_at", "progress", "stock", "remaining_count",
        }
        tables = {str(t.get("name") or ""): t for t in db.get("tables") or [] if isinstance(t, dict)}
        columns = {
            name: {str(c.get("name")) for c in table.get("columns") or [] if isinstance(c, dict)}
            for name, table in tables.items()
        }
        # Goals/KPIs and excluded capabilities are not storage requirements.
        # Inspect business requirements without allowing metadata from an
        # unrelated domain to introduce state or notification tables.
        scope = prd.get("mvpScope") if isinstance(prd.get("mvpScope"), dict) else {}
        excluded = {str(item) for item in scope.get("excluded") or []}
        core_features = [
            feature for feature in prd.get("coreFeatures") or []
            if isinstance(feature, dict) and str(feature.get("name") or "") not in excluded
        ]
        combined_prd = json.dumps({
            "projectOverview": prd.get("projectOverview"),
            "background": prd.get("background"),
            "coreFeatures": core_features,
        }, ensure_ascii=False).lower()
        refs: dict[str, set[str]] = {}
        for name, table in tables.items():
            refs[name] = set()
            for column in table.get("columns") or []:
                if not isinstance(column, dict):
                    continue
                match = re.search(r"REFERENCES\s+([a-z][a-z0-9_]*)", str(column.get("constraints") or ""), re.I)
                if match:
                    refs[name].add(match.group(1).lower())

        seen_cycles: set[frozenset[str]] = set()
        for source, targets in refs.items():
            for target in targets:
                pair = frozenset((source, target))
                if source != target and source in refs.get(target, set()) and pair not in seen_cycles:
                    seen_cycles.add(pair)
                    db_issues.append(f"상호 참조 순환 FK: {source} ↔ {target}")

        # Declared ERD relationships are documentation of the physical FK graph,
        # not an independent guess. Reject a direction or pair that no FK supports.
        fk_edges = {(target, source) for source, targets in refs.items() for target in targets}
        for relationship in db.get("relationships") or []:
            match = re.match(
                r"^\s*([a-z][a-z0-9_]*)\s*\([^)]*\)\s*([a-z][a-z0-9_]*)\s*$",
                str(relationship), re.I,
            )
            if match and (match.group(1).lower(), match.group(2).lower()) not in fk_edges:
                db_issues.append(f"FK와 역할·방향이 다른 relationship: {relationship}")

        name_lookup = _build_name_lookup(list(tables.values()))
        for table_name, table in tables.items():
            column_names = [
                str(column.get("name") or "")
                for column in table.get("columns") or [] if isinstance(column, dict)
            ]
            duplicates = sorted({
                name for name in column_names if name and column_names.count(name) > 1
            })
            if duplicates:
                db_issues.append(f"테이블 내부 중복 컬럼: {table_name} {duplicates}")
            invalid_names = sorted({
                name for name in column_names
                if name and not re.match(r'^[a-z][a-z0-9_]*$', name)
            })
            if invalid_names:
                db_issues.append(f"PostgreSQL 비정상 컬럼명: {table_name} {invalid_names}")
            for column in table.get("columns") or []:
                if not isinstance(column, dict):
                    continue
                column_name = str(column.get("name") or "")
                constraints = str(column.get("constraints") or "")
                match = re.search(r"REFERENCES\s+([a-z][a-z0-9_]*)", constraints, re.I)
                target = match.group(1).lower() if match else ""
                column_type = str(column.get("type") or "").upper()
                if target and target not in tables:
                    db_issues.append(f"존재하지 않는 FK 대상: {table_name}.{column_name} → {target}")
                actor_identifier = column_type.startswith(("BIGINT", "INTEGER", "UUID"))
                if column_name.endswith("_by") and actor_identifier and not target:
                    db_issues.append(f"행위자 역할 컬럼 FK 누락: {table_name}.{column_name}")
                if not target or not column_name.endswith("_id"):
                    continue
                role = column_name[:-3]
                exact_role_target = name_lookup.get(role)
                if exact_role_target and exact_role_target != target:
                    db_issues.append(
                        f"FK 역할 불일치: {table_name}.{column_name} → {target} "
                        f"(역할 대상은 {exact_role_target})"
                    )
        for feature in prd.get("coreFeatures") or []:
            if not isinstance(feature, dict):
                continue
            feature_name = str(feature.get("name") or "")
            content = f"{feature.get('description', '')} {' '.join(feature.get('requirements') or [])}"
            concepts = required_field_concepts(content)
            scored = [
                (relevance_score(feature_name, f"{name} {tables[name].get('description', '')}"), name)
                for name in tables
                if not (columns.get(name, set()) & {"password_hash", "credential_hash", "login_id", "email"})
            ]
            best = max((score for score, _ in scored), default=0.0)
            winners = [name for score, name in scored if score == best and score >= 0.2]
            if len(winners) == 1:
                table_name = winners[0]
                for concept, aliases in concepts.items():
                    represented = bool(columns.get(table_name, set()) & aliases)
                    if not represented and concept in {"status", "state", "due_date", "priority"} and any(
                        token in table_name for token in (
                            "assigned", "assignment", "membership", "link", "mapping",
                            "record", "history", "log", "event",
                        )
                    ):
                        represented = any(
                            bool(columns.get(parent, set()) & aliases)
                            for parent in refs.get(table_name, set())
                            if parent in tables
                        )
                    if not represented and concept in {"quantity", "amount", "balance", "unit"}:
                        represented = any(
                            table_name in refs.get(source, set()) and bool(columns.get(source, set()) & aliases)
                            for source in tables
                        )
                    if not represented:
                        db_issues.append(f"PRD 요구 필드 ERD 누락: {table_name}.{concept}")

        def matched_table(feature_name: str) -> str | None:
            scored = [
                (relevance_score(feature_name, f"{name} {table.get('description', '')}"), name)
                for name, table in tables.items()
            ]
            best = max((score for score, _name in scored), default=0.0)
            winners = [name for score, name in scored if score == best and score >= 0.2]
            return winners[0] if len(winners) == 1 else None

        feature_tables = [
            (feature, matched_table(str(feature))) for feature in feature_list or []
        ]
        root_table = next(
            (table_name for feature, table_name in feature_tables
             if table_name and feature_relation_kind(str(feature)) == "aggregate"),
            None,
        )
        auth_contract = requires_auth(prd, feature_list, requirement_text)
        credential_fields = {"password_hash", "credential_hash", "login_id", "email"}
        principal_tables = {
            name for name, names in columns.items() if names & credential_fields
        }
        if auth_contract and not principal_tables:
            db_issues.append("PRD 로그인 요구가 있지만 인증 주체 저장 구조가 없음")
        for feature, table_name in feature_tables:
            if not table_name or not root_table or table_name == root_table:
                continue
            kind = feature_relation_kind(str(feature))
            # Event/history resources may legitimately reference a state entity
            # rather than the first aggregate feature's table. Only explicit
            # association features require this topology check here.
            if kind == "association" and root_table not in refs.get(table_name, set()):
                db_issues.append(f"기능 관계 FK 누락: {table_name} → {root_table} ({feature})")
            if kind == "association" and auth_contract and principal_tables:
                if not (refs.get(table_name, set()) & principal_tables):
                    db_issues.append(f"배정 주체 FK 누락: {table_name} ({feature})")

        for table_name, table in tables.items():
            required_keys = scoped_unique_columns(table, list(tables.values()), core_features)
            if not required_keys:
                continue
            # The declared key must be covered by one composite unique index;
            # separate ordinary indexes cannot complete an unrelated UNIQUE.
            unique_keys = []
            for index in table.get("indexes") or []:
                match = re.search(r"\bUNIQUE\b[^()]*\(([^()]*)\)", str(index), re.IGNORECASE)
                if match:
                    unique_keys.append({part.strip().strip('"').lower() for part in match.group(1).split(",")})
            if set(required_keys) not in unique_keys:
                db_issues.append(f"중복 방지 UNIQUE 제약 누락: {table_name}")
        occurrence_fields = {
            "occurred_at", "recorded_at", "event_at", "completed_at", "processed_at",
            "started_at", "ended_at", "effective_at",
        }
        # A table with its own lifecycle state that other tables reference is the parent
        # entity (tasks.completed_at), not a history log of another aggregate.
        referenced_tables = {target for targets in refs.values() for target in targets}
        event_tables = {
            name for name, names in columns.items()
            if refs.get(name) and bool(names & occurrence_fields)
            and not any(token in name for token in ("assignment", "membership", "link", "mapping"))
            and not (names & {"status", "state"} and name in referenced_tables)
        }
        aggregates = {
            name for name, names in columns.items()
            if names & state_fields and name not in event_tables
        }
        mutating_contract = any(token in combined_prd for token in (
            "차감", "증가", "감소", "잔여", "수량을 갱신", "상태를 갱신", "balance", "quantity",
        ))
        if mutating_contract and not aggregates:
            db_issues.append("PRD가 상태 변경을 요구하지만 ERD에 변경 대상 상태 컬럼이 없음")
        for aggregate in aggregates:
            for catalog in refs.get(aggregate, set()):
                overlap = (columns.get(catalog, set()) & columns[aggregate] & (state_fields - {"status", "state"}))
                if overlap:
                    db_issues.append(f"기준 엔티티 {catalog}와 상태 엔티티 {aggregate}의 책임 중복: {sorted(overlap)}")

        for name in event_tables:
            aggregate_refs = refs.get(name, set()) & aggregates
            catalog_refs = refs.get(name, set()) - aggregates
            related_aggregate = {
                aggregate for aggregate in aggregates if refs.get(aggregate, set()) & catalog_refs
            }
            if related_aggregate and not aggregate_refs:
                db_issues.append(f"이력 엔티티 {name}가 실제 상태 엔티티를 참조하지 않음: {sorted(related_aggregate)}")

        endpoint_list = [ep for ep in api.get("endpoints") or [] if isinstance(ep, dict)]
        auth_required_by_prd = requires_auth(prd, feature_list, requirement_text)
        if auth_required_by_prd:
            for endpoint in endpoint_list:
                if str(endpoint.get("path") or "").startswith("/api/v1/auth/"):
                    continue
                if not bool(endpoint.get("authRequired")):
                    api_issues.append(f"PRD 인증 요구와 authRequired 불일치: {endpoint.get('method')} {endpoint.get('path')}")
        auth_paths = " ".join(str(ep.get("path") or "") for ep in endpoint_list)
        has_refresh_store = any(
            ({"token_hash", "expires_at"} <= names or {"refresh_token", "expires_at"} <= names)
            for names in columns.values()
        )
        if ("/auth/refresh" in auth_paths or "/auth/logout" in auth_paths) and not has_refresh_store:
            db_issues.append("Refresh Token API가 있지만 만료·회수 가능한 토큰 저장 구조가 없음")
        # Delivery infrastructure is deployment-specific. A product may use web
        # push, e-mail, SMS, or an external provider without persisting a device
        # token in this service, so QA must not force a push-device schema.
        preference_required = any(token in combined_prd for token in ("notification setting", "alert setting", "알림 설정", "수신 설정"))
        def _is_preference_column(name: str) -> bool:
            return bool(re.search(
                r"(?:^|_)(?:is_)?enabled?(?:_|$)|(?:^|_)opt(?:ed)?_(?:in|out)(?:_|$)"
                r"|(?:^|_)(?:subscribed|channel)(?:_|$)",
                name,
            ))

        has_preferences = any(
            "user_id" in names and any(_is_preference_column(name) for name in names)
            for names in columns.values()
        )
        if preference_required and not has_preferences:
            db_issues.append("알림 설정 요구사항이 있지만 사용자별 수신 설정 구조가 없음")

        for table_name, table in tables.items():
            index_text = " ".join(str(value).replace(" ", "").lower() for value in table.get("indexes") or [])
            for column in table.get("columns") or []:
                if not isinstance(column, dict):
                    continue
                column_name = str(column.get("name") or "")
                if re.search(r"REFERENCES\s+", str(column.get("constraints") or ""), re.I):
                    if f"({column_name.lower()})" not in index_text:
                        db_issues.append(f"FK 인덱스 누락: {table_name}.{column_name}")
            if {"status", "scheduled_at"}.issubset(columns.get(table_name, set())):
                if "(status,scheduled_at)" not in index_text:
                    db_issues.append(f"스케줄러 인덱스 누락: {table_name}(status, scheduled_at)")

        for endpoint in endpoint_list:
            method = str(endpoint.get("method") or "").upper()
            path = str(endpoint.get("path") or "")
            if path.startswith("/api/v1/auth/"):
                fields = _request_field_names(endpoint.get("requestBody"))
                if path.endswith("/login") and not (
                    fields & {"loginid", "login_id", "email", "username"}
                    and fields & {"credential", "password", "secret"}
                ):
                    api_issues.append(f"로그인 API 식별자·자격증명 계약 누락: {path}")
                continue
            if method == "GET" and "{" not in path and not path.endswith("/me"):
                if "404" in str(endpoint.get("errorCodes") or ""):
                    api_issues.append(f"목록 API가 빈 결과에 404를 명시: {path}")
                if "[]" not in str(endpoint.get("successResponse") or ""):
                    api_issues.append(f"목록 API의 빈 배열 계약 누락: {path}")
            slug = path.removeprefix("/api/v1/").split("/", 1)[0].replace("-", "_")
            mapped_name, mapped_table = _endpoint_table(endpoint, tables, db.get("featureMappings"))
            slug = mapped_name or slug
            if method not in {"GET", "DELETE"}:
                request_body = endpoint.get("requestBody")
                request_text = json.dumps(request_body, ensure_ascii=False) if isinstance(request_body, dict) else str(request_body or "")
                if re.search(r"\bpayload\s*:\s*object\b|기능(?:별)?\s*입력", request_text, re.I):
                    api_issues.append(f"API 요청 계약이 구체 필드 없이 payload로만 정의됨: {method} {path}")
                if mapped_table:
                    request_fields = _request_field_names(request_body)
                    endpoint_feature = str(
                        endpoint.get("featureName") or endpoint.get("description") or ""
                    ).split(":", 1)[0]
                    association_contract = feature_relation_kind(endpoint_feature) == "association"
                    server_managed = {"id", "created_at", "updated_at"}
                    if not association_contract:
                        server_managed.add("user_id")
                    db_fields = columns.get(mapped_name, set()) - server_managed
                    unknown = request_fields - db_fields
                    if unknown:
                        api_issues.append(
                            f"API 요청 필드가 ERD와 불일치: {method} {path} {sorted(unknown)}"
                        )
                    if method == "POST":
                        required = {
                            str(column.get("name"))
                            for column in mapped_table.get("columns") or []
                            if isinstance(column, dict)
                            and str(column.get("name")) not in server_managed
                            and "NOT_NULL" in str(column.get("constraints") or "").upper()
                            and "DEFAULT" not in str(column.get("constraints") or "").upper()
                        }
                        missing = required - request_fields
                        if missing:
                            api_issues.append(
                                f"API POST 필수 필드가 ERD보다 부족함: {path} {sorted(missing)}"
                            )
            if method in {"POST", "PUT", "PATCH"} and slug in tables:
                # A table referenced by a stateful table is not necessarily a
                # read-only catalog. Schedules, orders and projects are valid
                # aggregate roots even when child state tables reference them.
                # Report a misplaced state mutation only when the request
                # actually writes state fields that belong exclusively to the
                # referencing aggregate.
                aggregate_candidates = [
                    aggregate for aggregate in aggregates if slug in refs.get(aggregate, set())
                ]
                request_body = endpoint.get("requestBody")
                if isinstance(request_body, dict):
                    request_fields = {str(name).lower() for name in request_body}
                else:
                    request_fields = {
                        match.lower()
                        for match in re.findall(
                            r"(?:^|[,;{]\s*)['\"]?([a-z][a-z0-9_]*)['\"]?\s*:",
                            str(request_body or ""),
                            re.I,
                        )
                    }
                misplaced_fields: set[str] = set()
                for aggregate in aggregate_candidates:
                    misplaced_fields.update(
                        request_fields
                        & state_fields
                        & (columns.get(aggregate, set()) - columns.get(slug, set()))
                    )
                if misplaced_fields:
                    api_issues.append(
                        f"상태 변경 API가 {slug}에 없는 상태 필드를 직접 변경함: "
                        f"{path} {sorted(misplaced_fields)}"
                    )
            if method in {"POST", "PATCH", "DELETE"} and slug in event_tables:
                if refs.get(slug, set()) & aggregates and not endpoint.get("transactionRules"):
                    api_issues.append(f"상태 이력 API의 원자적 보정 규칙 누락: {method} {path}")
            if method in {"POST", "PATCH", "DELETE"} and slug in tables:
                has_unique_contract = any(
                    re.search(r"CREATE\s+UNIQUE\s+INDEX", str(index), re.I)
                    for index in tables[slug].get("indexes") or []
                )
                has_state_parent = any(
                    columns.get(target, set()) & state_fields
                    for target in refs.get(slug, set())
                )
                if (has_unique_contract or has_state_parent) and not endpoint.get("transactionRules"):
                    api_issues.append(f"중복·선행 업무 트랜잭션 규칙 누락: {method} {path}")

        incomplete_tail = re.compile(
            r"(?:하는|되는|위한|통한|해소하는|구현하는|제공하는|기반 마련|문제를 해결|"
            r"태스크\s*관리|기능\s*구현|서비스\s*설계)\s*[.]?$"
        )
        for item in prd.get("releaseSchedule") or []:
            if isinstance(item, dict):
                description = str(item.get("description") or "").strip()
                if incomplete_tail.search(description):
                    prd_issues.append(f"일정 문장 잘림 또는 미완결: {description}")
        return list(dict.fromkeys(db_issues)), list(dict.fromkeys(api_issues)), list(dict.fromkeys(prd_issues))

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
                raw = await self._call(prompt, max_tokens=16384)
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
                draft_data = try_parse_json(draft) or {}
                patches = data.get("patches")
                if not isinstance(patches, dict):
                    patches = _derive_patches_from_full_output(domain, draft_data, data.get("fixed"))
                fixed_data = _apply_review_patches(domain, draft_data, patches)
                if isinstance(fixed_data, dict):
                    if domain == "db":
                        fixed_data = _to_array_schema(fixed_data)
                    # QA는 정합성 교정 단계이지 가지치기 단계가 아니다 — 항목 수가 원본보다
                    # 줄어드는 것은 (EXAONE 재생성 시 테이블/엔드포인트/기능 누락) 항상 결함으로
                    # 취급하고 원본을 유지한다. 원본과 같거나 많을 때만 수정본을 채택한다.
                    # 합계가 아니라 섹션별로 비교해야 한 섹션의 누락이 다른 섹션 증가에 가려지지 않는다.
                    shrunk = _shrunk_fields(
                        _count_items_per_field(domain, try_parse_json(draft)),
                        _count_items_per_field(domain, fixed_data),
                    )
                    if shrunk:
                        logger.warning(
                            "%s reviewer의 fixed가 원본보다 항목이 줄어듦(%s, EXAONE 누락 추정) — 원본 유지",
                            domain,
                            ", ".join(f"{f} {o}→{n}" for f, (o, n) in shrunk.items()),
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
                logger.warning("%s manager_check — 최대 라운드(%d) 도달, 미승인 상태로 종료", domain, _MAX_ROUNDS)
                return {
                    f"{domain}_round": round_,
                    f"{domain}_approved": False,
                    f"{domain}_forced": True,
                    f"{domain}_feedback": "",
                }

            draft = state.get(f"{domain}_draft", "")
            issues = state.get(f"{domain}_issues", [])
            prompt = MANAGER_CHECK_PROMPT.format(
                domain_label=_DOMAIN_LABEL[domain],
                round=round_,
                draft=self._safe_truncate(draft, _MAX_SCHEMA_CHARS, f"{domain}_draft"),
                issues="\n".join(f"- {i}" for i in issues) or "(reviewer가 보고한 결함 없음)",
                criteria=_DOMAIN_CRITERIA[domain],
            )
            raw = await self._call(prompt, max_tokens=16384)
            if ctx.get("dump"):
                ctx["dump"].log_raw(f"QA_{domain.upper()}_CHECK", round_, raw)

            data = try_parse_json(raw)
            if data and isinstance(data, dict):
                approved = data.get("approved") is True
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
            finished = state.get(f"{domain}_approved") or state.get(f"{domain}_forced")
            return END if finished else f"{domain}_reviewer"

        return router

    async def _call(self, user_prompt: str, max_tokens: int) -> str:
        for attempt in range(3):
            try:
                async def request():
                    return await self._client.chat.completions.create(
                        model=self._model,
                        temperature=_TEMPERATURE,
                        top_p=_TOP_P,
                        presence_penalty=_PRESENCE_PENALTY,
                        max_completion_tokens=max_tokens,
                        frequency_penalty=0.3,
                        messages=[
                            {"role": "system", "content": REVIEW_SYSTEM},
                            {"role": "user", "content": user_prompt},
                        ],
                    )

                response = await self._runtime.call(request) if self._runtime else await request()
                return response.choices[0].message.content or ""
            except (InternalServerError, APITimeoutError, APIConnectionError, TimeoutError) as e:
                logger.warning("QA API 일시 오류 (attempt %d): %s — 재시도", attempt + 1, e)
                if attempt < 2:
                    # QA는 Phase3 복구 중 호출된다. 긴 backoff가 도메인별 루프와 곱해지지 않도록 짧게 제한한다.
                    await asyncio.sleep(2 * (attempt + 1))
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

    def _check_prd_completeness(
        self, prd_doc, feature_count: int, market_data: str = ""
    ) -> list[str]:
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

        if contamination_reasons(prd_doc):
            issues.append(f"PRD 오염 문자열 감지: {contamination_reasons(prd_doc)}")
        if has_placeholder(prd_doc):
            issues.append("PRD에 placeholder·미정·수동 보완 값이 남아 있음")

        goals = [str(x) for x in prd_doc.get("goals") or []]
        if any(near_duplicate(goals[i], goals[j], 0.6) for i in range(len(goals)) for j in range(i)):
            issues.append("goals에 의미상 중복 목표가 있음")
        metrics = [str(x.get("metric") or "") for x in prd_doc.get("kpi") or [] if isinstance(x, dict)]
        if any(near_duplicate(metrics[i], metrics[j], 0.6) for i in range(len(metrics)) for j in range(i)):
            issues.append("KPI 지표가 의미상 중복됨")
        for item in prd_doc.get("kpi") or []:
            if isinstance(item, dict) and "→" not in str(item.get("target") or ""):
                issues.append(f"KPI target 형식 오류: {item.get('metric')}")
            elif isinstance(item, dict):
                for reason in kpi_basis_issues(item, market_data):
                    issues.append(f"KPI 근거 검증 실패: {item.get('metric')} — {reason}")

        scope = prd_doc.get("mvpScope") if isinstance(prd_doc.get("mvpScope"), dict) else {}
        excluded = set(scope.get("excluded") or [])
        p0_excluded = [item.get("name") for item in prd_doc.get("coreFeatures") or []
                       if isinstance(item, dict) and item.get("priority") == "P0" and item.get("name") in excluded]
        if p0_excluded:
            issues.append(f"P0 기능이 MVP excluded와 충돌: {p0_excluded}")
        for item in prd_doc.get("coreFeatures") or []:
            if not isinstance(item, dict):
                continue
            feature = str(item.get("name") or "")
            content = f"{item.get('description', '')} {' '.join(item.get('requirements') or [])}"
            if feature and relevance_score(feature, content) < 0.2:
                issues.append(f"기능 설명·요구사항이 담당 기능에서 이탈: {feature}")
            if str(item.get("origin") or "").upper() == "DERIVED":
                if not str(item.get("parentFeature") or "").strip():
                    issues.append(f"파생 기능의 상위 기능 누락: {feature}")
                if not item.get("source") or len(str(item.get("rationale") or "").strip()) < 12:
                    issues.append(f"파생 기능의 요구사항 근거 부족: {feature}")
                if not item.get("acceptanceCriteria"):
                    issues.append(f"파생 기능의 수용 기준 누락: {feature}")
            if item.get("origin") and not item.get("actions"):
                issues.append(f"상세 기능 행위 명세 누락: {feature}")
            if item.get("origin") and not item.get("dataRequirements"):
                issues.append(f"상세 기능 데이터 명세 누락: {feature}")

        signatures = []
        for item in prd_doc.get("userPersonas") or []:
            if isinstance(item, dict):
                sig = f"{item.get('age', '')} {item.get('job', '')}"
                if any(near_duplicate(sig, prior, 0.75) for prior in signatures):
                    issues.append("Persona 연령·직업 조합이 중복되어 세그먼트 구분이 약함")
                    break
                signatures.append(sig)

        return list(dict.fromkeys(issues))

    def _check_api_completeness(self, api_spec, feature_list: list[str]) -> list[str]:
        """API 스펙의 엔드포인트 수/인증 정보/기능별 커버리지가 충분한지 결정론적으로 검증."""
        if not isinstance(api_spec, dict) or not api_spec:
            return ["API 스펙이 비어 있거나 파싱되지 않았습니다"]

        feature_list = backend_features(feature_list)
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
            endpoint_texts = [
                " ".join(
                    str(ep.get(key, ""))
                    for key in ("method", "path", "description", "featureId", "featureName", "action")
                )
                for ep in endpoints if isinstance(ep, dict)
            ]
            missing = strictly_uncovered_features(feature_list, endpoint_texts)
            # The combined feature is commonly represented by separate
            # signup/login contracts. Treat the pair as deterministic coverage
            # even when the LLM descriptions do not repeat the Korean label.
            auth_paths = " ".join(endpoint_texts).casefold()
            if any("회원가입" in feature and "로그인" in feature for feature in missing):
                if "/auth/signup" in auth_paths and "/auth/login" in auth_paths:
                    missing = [feature for feature in missing if not ("회원가입" in feature and "로그인" in feature)]
            if missing:
                issues.append(f"다음 기능에 대응하는 엔드포인트가 없어 보임: {missing}")

            seen = set()
            for ep in endpoints:
                if not isinstance(ep, dict):
                    issues.append("endpoint가 객체 형식이 아님")
                    continue
                key = (str(ep.get("method") or "").upper(), str(ep.get("path") or ""))
                if key in seen:
                    issues.append(f"중복 endpoint: {key[0]} {key[1]}")
                seen.add(key)
                path = key[1]
                if re.search(r"/(?:resource|qa-feature)-\d+", path) or "/domain-features-" in path:
                    issues.append(f"의미 없는 리소스 경로: {path}")
                for issue in _endpoint_quality_issues(ep, "SELF_CHECK: PASS", ep, require_self_check=False):
                    issues.append(f"{key[0]} {path}: {issue}")

            for index, feature in enumerate(feature_list):
                slug = _feature_resource_slug(feature, index)
                actual = {str(ep.get("method") or "").upper() for ep in endpoints
                          if isinstance(ep, dict) and (
                              f"/api/v1/{slug}" in str(ep.get("path") or "")
                              or relevance_score(
                                  feature,
                                  f"{ep.get('featureName', '')} {ep.get('description', '')}",
                              ) >= 0.2
                          )}
                mapping = next((
                    item for item in api_spec.get("featureMappings") or []
                    if isinstance(item, dict) and str(item.get("featureName") or "") == str(feature)
                ), None)
                if mapping:
                    actual.update(
                        str(operation.get("method") or "").upper()
                        for operation in mapping.get("operations") or [] if isinstance(operation, dict)
                    )
                required = set(_feature_methods(feature))
                missing_methods = required - actual
                if missing_methods:
                    issues.append(f"{feature} CRUD/행위 계약 누락: {sorted(missing_methods)}")

        return list(dict.fromkeys(issues))

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

    def _check_db_completeness(self, db_schema, feature_list: list[str], feature_registry=None) -> list[str]:
        """DB 스키마의 테이블/컬럼/기능별 커버리지가 실질적으로 채워져 있는지 결정론적으로 검증."""
        if not isinstance(db_schema, dict) or not db_schema:
            return ["DB 스키마가 비어 있거나 파싱되지 않았습니다"]

        tables = db_schema.get("tables")
        if not isinstance(tables, list) or not tables:
            return ["tables가 비어 있음"]

        feature_list = backend_features(feature_list)
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
        if feature_registry:
            missing = missing_db_contract_features(feature_registry, tables)
        else:
            haystack = []
            for t in tables:
                if isinstance(t, dict):
                    haystack.append(str(t.get("name", "")))
                    haystack.append(str(t.get("description", "")))
            missing = strictly_uncovered_features(feature_list, haystack)
        if missing:
            issues.append(f"다음 기능을 저장할 테이블이 없어 보임: {missing}")

        db_contamination = [
            reason for reason in contamination_reasons(db_schema)
            if reason != "foreign-language sentence leak"
        ]
        if db_contamination:
            issues.append(f"DB 스키마 오염 문자열 감지: {db_contamination}")
        if has_placeholder(db_schema):
            issues.append("DB 스키마에 placeholder가 남아 있음")

        for table in tables:
            if not isinstance(table, dict):
                continue
            col_names = {str(c.get("name") or "") for c in table.get("columns") or [] if isinstance(c, dict)}
            for col in table.get("columns") or []:
                if not isinstance(col, dict):
                    continue
                name = str(col.get("name") or "")
                col_type = str(col.get("type") or "").upper()
                constraints = str(col.get("constraints") or "").upper()
                if col_type == "DATETIME" or "AUTO_INCREMENT" in constraints or "ON UPDATE" in constraints:
                    issues.append(f"PostgreSQL 비호환 정의: {table.get('name')}.{name}")
                if col_type == "TIMESTAMPTZ" and "WITHOUT TIME ZONE" in constraints:
                    issues.append(f"PostgreSQL 타입·제약 모순: {table.get('name')}.{name}")
                if col_type == "TIMESTAMP" and "WITH TIME ZONE" in constraints:
                    issues.append(f"PostgreSQL 타입 분리 오류: {table.get('name')}.{name}")
                if re.search(r"GENERATED\s+(?:BY\s+DEFAULT|ALWAYS)(?!\s+AS\s+IDENTITY)", constraints):
                    issues.append(f"PostgreSQL IDENTITY 제약 불완전: {table.get('name')}.{name}")
                if col_type not in {
                    "BIGINT", "INT", "INTEGER", "SMALLINT", "VARCHAR", "CHAR", "TEXT", "BOOLEAN", "DATE",
                    "TIMESTAMP", "TIMESTAMPTZ", "REAL", "DOUBLE PRECISION", "JSONB", "UUID",
                } and not re.match(r"^(?:VARCHAR|CHAR)\(\d+\)$|^(?:DECIMAL|NUMERIC)\(\d+,\d+\)$", col_type):
                    issues.append(f"PostgreSQL 허용 타입 아님: {table.get('name')}.{name}={col_type}")
                explicit_reference = re.search(
                    r"REFERENCES\s+([a-z][a-z0-9_]*)", constraints, re.I
                )
                explicit_target = explicit_reference.group(1).lower() if explicit_reference else ""
                has_valid_reference = explicit_target in {
                    str(candidate.get("name") or "").lower()
                    for candidate in tables if isinstance(candidate, dict)
                }
                # `_id`는 로그인 ID·외부 시스템 ID 같은 일반 식별자일 수도 있다.
                # FK라고 명시한 컬럼만 실제 대상 테이블 존재 여부를 검사한다.
                if explicit_reference and not has_valid_reference:
                    issues.append(f"참조 대상 없는 FK 컬럼: {table.get('name')}.{name}")
            for index_sql in table.get("indexes") or []:
                match = re.search(r"\(([^)]*)\)", str(index_sql))
                if match:
                    unknown = [x.strip() for x in match.group(1).split(",") if x.strip() not in col_names]
                    if unknown:
                        issues.append(f"없는 컬럼을 참조하는 인덱스: {table.get('name')} {unknown}")

        return list(dict.fromkeys(issues))

    def _db_completeness_hint(self, db_schema, feature_list: list[str], feature_registry=None) -> str:
        """db_reviewer 컨텍스트에 주입할 결정론적 결함 힌트."""
        issues = self._check_db_completeness(db_schema, feature_list, feature_registry)
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
