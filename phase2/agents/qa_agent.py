import asyncio
import json
import logging
import re
from openai import AsyncOpenAI, InternalServerError, APITimeoutError, APIConnectionError

from phase2.agents.dba_agent import (
    _to_array_schema, _parse_table_text, _build_name_lookup, _fk_target, dedupe_meta_tables,
    enforce_schema_contracts, ensure_domain_columns, min_table_count, reconcile_fk_types, sanitize_tables,
)
from phase2.agents.api_agent import (
    _align_endpoints_to_db, _endpoint_quality_issues, _feature_methods, _feature_resource_slug,
    _normalize_endpoints, _parse_endpoint_text,
)
from phase2.agents.prd_agent import PrdAgent, reconcile_core_features
from phase2.feature_coverage import uncovered_features
from phase2.json_utils import try_parse_json
from phase2.state import PipelineState
from phase2.agent_contract import (
    IssueSeverity, classify_issue, feature_relation_kind, requires_auth,
)
from phase2.quality_rules import (
    contamination_reasons, has_placeholder, kpi_basis_issues, near_duplicate, relevance_score,
    required_field_concepts, self_check_passed,
)

logger = logging.getLogger(__name__)


def _singular_actor_name(name: str) -> str:
    if name.endswith("ies"):
        return name[:-3] + "y"
    return name[:-1] if name.endswith("s") else name


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


def _endpoint_table(endpoint: dict, tables: dict[str, dict]) -> tuple[str, dict] | tuple[None, None]:
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

_MAX_SCHEMA_CHARS = 8000  # reviewer 컨텍스트에 전달할 db_schema/api_spec 최대 길이
_MAX_ROUNDS = 2  # 도메인당 reviewer↔manager_check 최대 라운드 (2회째는 강제 승인)

# EXAONE 모델 카드 권장 샘플링 파라미터
# https://huggingface.co/LGAI-EXAONE/K-EXAONE-236B-A23B
_TEMPERATURE = 1.0
_TOP_P = 0.95
_PRESENCE_PENALTY = 0.0

REVIEW_SYSTEM = """JSON만 출력하세요. 설명·인사말·마크다운 코드블록 금지.
reviewer 응답의 최상위 키는 issues와 patches만 허용합니다. fixed 키와 전체 문서 재출력은 절대 금지합니다.
manager_check 응답의 최상위 키는 approved와 feedback만 허용합니다."""

_LENIENCY_NOTE = """
결함 판단 기준:
- 문서 간 필드명·기능 의미·우선순위·경로·FK가 다르면 구체적인 결함으로 보고합니다.
- 구현할 수 없는 평문 계약, placeholder, 오염 문자열, PostgreSQL 비호환 문법은 통과시키지 않습니다.
- 전체 문서를 다시 쓰지 말고 문제 목록과 변경할 필드 patch만 반환합니다."""

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

DB_PATCH_REVIEW_PROMPT = """당신은 시니어 DBA reviewer입니다.
현재 DB 스키마와 API/기능 목록의 정합성 문제만 찾으세요.

[현재 DB 스키마]
{draft}
[API 참고]
{sibling}
[기능 목록]
{feature_str}
{feedback_section}
{completeness_hint}

전체 스키마를 다시 출력하지 마세요. 수정이 필요한 테이블만 완전한 단일 테이블 객체로 patches.tables에 넣고,
추가할 relationship만 patches.relationships에 넣으세요. 삭제는 하지 않습니다.
JSON: {{"issues":["문제"],"patches":{{"tables":[{{"name":"...","description":"...","columns":[],"indexes":[]}}],"relationships":[]}}}}
문제가 없으면 {{"issues":[],"patches":{{}}}}만 출력하세요. JSON 외 텍스트 금지."""

API_PATCH_REVIEW_PROMPT = """당신은 시니어 백엔드 아키텍트 reviewer입니다.
현재 API 스펙이 DB/PRD/기능 목록을 커버하는지만 검토하세요.

[현재 API 스펙]
{draft}
[DB 참고]
{sibling}
[PRD 요약]
{prd_summary}
[기능 목록]
{feature_str}
{feedback_section}
{completeness_hint}

전체 API 스펙을 다시 출력하지 마세요. 추가·교체할 엔드포인트만 완전한 단일 endpoint 객체로
patches.endpoints에 넣으세요. endpoint 식별자는 method+path입니다.
JSON: {{"issues":["문제"],"patches":{{"endpoints":[{{"method":"GET","path":"/api/v1/..."}}]}}}}
문제가 없으면 {{"issues":[],"patches":{{}}}}만 출력하세요. JSON 외 텍스트 금지."""

PRD_PATCH_REVIEW_PROMPT = """당신은 시니어 PM reviewer입니다.
현재 PRD와 DB/API/기능 목록 사이의 정합성 문제만 검토하세요.

[현재 PRD]
{draft}
[DB 참고]
{db_sibling}
[API 참고]
{api_sibling}
[기능 목록]
{feature_str}
{feedback_section}
{completeness_hint}

전체 PRD를 다시 출력하지 마세요. 변경이 필요한 최상위 필드만 patches에 넣으세요.
목록 필드를 수정할 경우 원본 항목을 삭제하지 말고 추가·교체할 항목만 해당 목록에 넣으세요.
JSON: {{"issues":["문제"],"patches":{{"coreFeatures":[{{"name":"..."}}],"techStack":{{...}}}}}}
문제가 없으면 {{"issues":[],"patches":{{}}}}만 출력하세요. JSON 외 텍스트 금지."""

_REVIEW_PROMPTS = {
    "db": DB_PATCH_REVIEW_PROMPT,
    "api": API_PATCH_REVIEW_PROMPT,
    "prd": PRD_PATCH_REVIEW_PROMPT,
}

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
            normalized = _normalize_endpoints(endpoint_patches)
            result["endpoints"] = _replace_named_items(
                result.get("endpoints") or [], normalized,
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


class QaAgent:

    def __init__(self, client: AsyncOpenAI, model: str = "gpt-4o"):
        self._client = client
        self._model = model

    async def execute(self, state: PipelineState, dump=None) -> PipelineState:
        logger.info("QA 에이전트 시작 — 산출물 간 연관성 검증 전용")
        db = try_parse_json(state.db_schema or "{}") or {}
        api = try_parse_json(state.api_spec or "{}") or {}
        prd = try_parse_json(state.prd_document or "{}") or {}

        # QA never writes documents. It reports defects against the exact drafts
        # produced by PM/PRD/DBA/API, so resolved and stale issues cannot be mixed.
        db_issues = self._check_db_completeness(db, state.feature_list)
        api_issues = self._check_api_completeness(api, state.feature_list)
        prd_issues = self._check_prd_completeness(
            prd, len(state.feature_list), state.market_research or ""
        )
        cross_db, cross_api, cross_prd = self._check_cross_document_semantics(
            prd, db, api, state.feature_list, state.context_prompt or "",
        )
        db_issues = list(dict.fromkeys(db_issues + cross_db))
        api_issues = list(dict.fromkeys(api_issues + cross_api))
        prd_issues = list(dict.fromkeys(prd_issues + cross_prd))
        total_issues = len(db_issues) + len(api_issues) + len(prd_issues)
        quality_score = self._compute_quality_score(total_issues == 0, total_issues)
        status = "QA 완료 — 전 항목 승인" if total_issues == 0 else f"QA 완료 — 잔존 결함 {total_issues}건 (승인 보류)"
        issue_details = self._structured_issues(db_issues, api_issues, prd_issues)
        repair_issues = [
            issue for issue in issue_details
            if issue["severity"] in {IssueSeverity.BLOCKER.value, IssueSeverity.ERROR.value}
        ]
        logger.info(
            "QA 연관성 검사 완료 — DB:%d/API:%d/PRD:%d 결함 (qualityScore=%.2f)",
            len(db_issues), len(api_issues), len(prd_issues), quality_score,
        )
        if total_issues:
            logger.warning("QA 결함 상세 — DB=%s | API=%s | PRD=%s", db_issues, api_issues, prd_issues)

        return state.copy(
            qa_quality_score=quality_score,
            qa_db_issues=db_issues,
            qa_api_issues=api_issues,
            qa_prd_issues=prd_issues,
            qa_issue_details=issue_details,
            qa_repair_issues=repair_issues,
            last_validation_error="",
            status_message=status,
        )

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

    async def _run_deterministic_review(self, graph_input: dict, feature_list: list[str], dump=None) -> dict:
        """전체 문서 JSON reviewer 대신 코드 검사 + 누락 기능별 평문 patch worker를 실행한다."""
        db = try_parse_json(graph_input.get("db_draft") or "{}") or {}
        api = try_parse_json(graph_input.get("api_draft") or "{}") or {}
        prd = try_parse_json(graph_input.get("prd_draft") or "{}") or {}

        db_missing = uncovered_features(feature_list, json.dumps(db.get("tables") or [], ensure_ascii=False))
        api_missing = uncovered_features(feature_list, json.dumps(api.get("endpoints") or [], ensure_ascii=False))
        prd_missing = uncovered_features(feature_list, json.dumps(prd.get("coreFeatures") or [], ensure_ascii=False))

        jobs = []
        for domain, missing in (("db", db_missing), ("api", api_missing), ("prd", prd_missing)):
            for index, feature in enumerate(missing):
                jobs.append(self._repair_feature_item(domain, feature, index, dump))
        patches = await asyncio.gather(*jobs) if jobs else []

        for domain, patch in patches:
            if not patch:
                continue
            if domain == "db":
                db = _apply_review_patches("db", db, {"tables": [patch]})
            elif domain == "api":
                api = _apply_review_patches("api", api, {"endpoints": [patch]})
            else:
                prd = _apply_review_patches("prd", prd, {"coreFeatures": [patch]})

        db_issues = self._check_db_completeness(db, feature_list)
        api_issues = self._check_api_completeness(api, feature_list)
        prd_issues = self._check_prd_completeness(prd, len(feature_list))
        cross_db, cross_api, cross_prd = self._check_cross_document_semantics(prd, db, api, feature_list)
        db_issues.extend(issue for issue in cross_db if issue not in db_issues)
        api_issues.extend(issue for issue in cross_api if issue not in api_issues)
        prd_issues.extend(issue for issue in cross_prd if issue not in prd_issues)
        logger.info(
            "QA 결정론적 검사 완료 — patch worker %d개, DB:%d/API:%d/PRD:%d 결함",
            len(jobs), len(db_issues), len(api_issues), len(prd_issues),
        )
        return {
            "db_draft": json.dumps(db, ensure_ascii=False),
            "api_draft": json.dumps(api, ensure_ascii=False),
            "prd_draft": json.dumps(prd, ensure_ascii=False),
            "db_round": 1 if db_missing else 0, "api_round": 1 if api_missing else 0,
            "prd_round": 1 if prd_missing else 0,
            "db_approved": not db_issues, "api_approved": not api_issues, "prd_approved": not prd_issues,
            "db_issues": db_issues, "api_issues": api_issues, "prd_issues": prd_issues,
        }

    async def _repair_feature_item(self, domain: str, feature: str, index: int, dump=None):
        label = f"QA_{domain.upper()}_PATCH_{index + 1}"
        if domain == "db":
            name = f"qa_feature_{index + 1}_records"
            prompt = (
                f"누락 기능: {feature}\nNAME: {name}\nDESCRIPTION: 기능 목적\n"
                "COLUMN: id BIGINT PRIMARY_KEY GENERATED_IDENTITY\nCOLUMN: created_at TIMESTAMPTZ NOT_NULL DEFAULT_NOW\n"
                "COLUMN: updated_at TIMESTAMPTZ NOT_NULL DEFAULT_NOW\nCOLUMN: 기능 데이터 컬럼\nINDEX: 없음\nSELF_CHECK: PASS"
            )
            system = "DB 테이블 하나를 PostgreSQL 전용 NAME/DESCRIPTION/COLUMN/INDEX 라벨 평문으로 작성하고 SELF_CHECK: PASS로 끝내세요. JSON 금지."
        elif domain == "api":
            skeleton = {"method": "POST", "path": f"/api/v1/{_feature_resource_slug(feature, index)}",
                        "description": f"{feature}: 생성 또는 실행", "authRequired": True, "featureName": feature}
            prompt = (
                f"누락 기능: {feature}\nMETHOD: POST\nPATH: {skeleton['path']}\nDESCRIPTION:\n"
                "AUTH_REQUIRED: true\nREQUEST_BODY:\nSUCCESS_RESPONSE:\nERROR_CODES:\nSELF_CHECK: PASS"
            )
            system = "API 하나를 METHOD/PATH/DESCRIPTION/AUTH_REQUIRED/REQUEST_BODY/SUCCESS_RESPONSE/ERROR_CODES 평문으로 작성하고 SELF_CHECK: PASS로 끝내세요. JSON 금지."
        else:
            prompt = (
                f"누락 기능: {feature}\nNAME: {feature}\nDESCRIPTION:\nPRIORITY: P1\nREQUIREMENTS:\nSELF_CHECK: PASS"
            )
            system = "PRD 기능 하나를 NAME/DESCRIPTION/PRIORITY/REQUIREMENTS 라벨 평문으로 작성하고 SELF_CHECK: PASS로 끝내세요. JSON 금지."

        for attempt in range(3):
            raw = await self._call_text(prompt, 1200, system)
            if dump:
                dump.log_raw(label, attempt + 1, raw)
            if domain == "db":
                item = _parse_table_text(raw, name, feature)
            elif domain == "api":
                item = _parse_endpoint_text(raw, skeleton)
            else:
                item = PrdAgent._parse_item_text("core_feature", raw, feature, index)
                if not PrdAgent._valid_item("core_feature", item):
                    item = None
            if item and not self_check_passed(raw):
                item = None
            if item:
                return domain, item
            logger.warning("%s 평문 patch 파싱 실패 (attempt %d) — 해당 문제만 재시도", label, attempt + 1)
        return domain, None

    async def _call_text(self, user_prompt: str, max_tokens: int, system: str) -> str:
        for attempt in range(3):
            try:
                response = await self._client.chat.completions.create(
                    model=self._model, temperature=_TEMPERATURE, top_p=_TOP_P,
                    presence_penalty=_PRESENCE_PENALTY, max_tokens=max_tokens, frequency_penalty=0.3,
                    messages=[{"role": "system", "content": system}, {"role": "user", "content": user_prompt}],
                    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                )
                return response.choices[0].message.content or ""
            except (InternalServerError, APITimeoutError, APIConnectionError) as e:
                logger.warning("QA 평문 patch API 오류 (attempt %d): %s", attempt + 1, e)
                if attempt < 2:
                    await asyncio.sleep(5 * (attempt + 1))
        return ""

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
                raw = await self._call(prompt, max_tokens=6000, enable_thinking=False)
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
                draft_data = try_parse_json(draft)
                patches = data.get("patches")
                if not isinstance(patches, dict) and isinstance(data.get("fixed"), dict) and isinstance(draft_data, dict):
                    logger.warning("%s reviewer가 금지된 fixed 전체 문서를 반환 — 최소 patch로 축약", domain)
                    patches = _derive_patches_from_full_output(domain, draft_data, data["fixed"])
                if isinstance(patches, dict) and isinstance(draft_data, dict):
                    patched = _apply_review_patches(domain, draft_data, patches)
                    fixed = json.dumps(patched, ensure_ascii=False)
                    patch_count = sum(len(v) if isinstance(v, list) else 1 for v in patches.values())
                    logger.info("%s reviewer patch-only 적용 — %d개 변경", domain, patch_count)
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
            raw = await self._call(prompt, max_tokens=2000, enable_thinking=False)
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
                    response_format={"type": "json_object"},
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

    @staticmethod
    def _relationships_from_fk(tables: list[dict]) -> list[str]:
        relationships = []
        for table in tables:
            for column in table.get("columns") or []:
                if not isinstance(column, dict):
                    continue
                match = re.search(
                    r"REFERENCES\s+([a-z][a-z0-9_]*)",
                    str(column.get("constraints") or ""), re.I,
                )
                target = match.group(1).lower() if match else ""
                relation = f"{target} (1:N) {table.get('name')}" if target else ""
                if relation and relation not in relationships and target != table.get("name"):
                    relationships.append(relation)
        return relationships

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
        combined_prd = json.dumps(prd, ensure_ascii=False).lower()
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

        duplicate_required = bool(re.search(
            r"(?:중복.{0,20}(?:방지|차단|금지)|중복\s*예약|동일.{0,20}(?:한\s*번|1회)|하나만)", combined_prd,
        ))
        if duplicate_required:
            for table_name, names in columns.items():
                if any(token in table_name for token in ("record", "history", "log", "event")):
                    continue
                fk_ids = []
                for column in tables[table_name].get("columns") or []:
                    if isinstance(column, dict) and re.search(
                        r"REFERENCES\s+", str(column.get("constraints") or ""), re.I
                    ):
                        fk_ids.append(str(column.get("name") or ""))
                if len(fk_ids) < 2:
                    continue
                index_text = " ".join(str(index).lower() for index in tables[table_name].get("indexes") or [])
                if "unique" not in index_text or not all(column in index_text for column in fk_ids[:2]):
                    db_issues.append(f"중복 방지 UNIQUE 제약 누락: {table_name}")
        occurrence_fields = {
            "occurred_at", "recorded_at", "event_at", "completed_at", "processed_at",
            "started_at", "ended_at", "effective_at",
        }
        event_tables = {
            name for name, names in columns.items()
            if refs.get(name) and bool(names & occurrence_fields)
            and not any(token in name for token in ("assignment", "membership", "link", "mapping"))
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
            mapped_name, mapped_table = _endpoint_table(endpoint, tables)
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
            mapping_names = [
                str(item.get("featureName") or "")
                for item in api_spec.get("featureMappings") or [] if isinstance(item, dict)
                and item.get("operations")
            ]
            missing = uncovered_features(feature_list, [
                f"{ep.get('featureName', '')} {ep.get('description', '')}"
                for ep in endpoints if isinstance(ep, dict)
            ] + mapping_names)
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
