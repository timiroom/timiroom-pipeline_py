import asyncio
import difflib
import json
import logging
import math
import operator
import re
from typing import Annotated, TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.types import Send
from openai import AsyncOpenAI, InternalServerError, APITimeoutError, APIConnectionError

from phase2.feature_coverage import uncovered_features, missing_features_note
from phase2.json_utils import try_parse_json, has_suspicious_script
from phase2.state import PipelineState

logger = logging.getLogger(__name__)

# EXAONE 모델 카드 권장 샘플링 파라미터
# https://huggingface.co/LGAI-EXAONE/K-EXAONE-236B-A23B
_TEMPERATURE = 1.0
_TOP_P = 0.95
_PRESENCE_PENALTY = 0.0
# 테이블 목록(plan) 생성은 개수·커버리지 일관성이 중요한 구조 생성 단계 —
# temp=1.0은 실행마다 편차를 키우므로 이 단계만 낮춰 완성도/재현성을 높인다.
_PLAN_TEMPERATURE = 0.4

WORKER_SYSTEM = "JSON만 출력하세요. 설명·인사말·마크다운 코드블록 금지. { 로 시작해서 } 로 끝납니다."

MANAGER_SYSTEM = """당신은 시니어 DBA 겸 DB manager입니다.
JSON만 출력하세요. 설명·인사말·마크다운 코드블록 금지. { 로 시작해서 } 로 끝납니다."""


def min_table_count(feature_count: int) -> int:
    """기능 개수 대비 DB 테이블 최소 개수 — DBA/QA/Phase3 검증이 동일 기준을 쓰도록 공유."""
    return max(3, math.ceil(feature_count / 3))


def _parse_column_strings(raw_columns) -> list[dict]:
    """"컬럼명 타입 제약조건들" 문자열 리스트 → {name,type,constraints} 구조로 변환.
    이미 dict 리스트면 그대로 두고, {컬럼명: 타입} 형태로 오면 방어적으로 복구."""
    if isinstance(raw_columns, dict):
        raw_columns = [f"{k} {v}" if isinstance(v, str) else str(k) for k, v in raw_columns.items()]
    elif not isinstance(raw_columns, list):
        raw_columns = []

    columns = []
    for col in raw_columns:
        if isinstance(col, str):
            parts = col.split()
            if not parts:
                continue
            columns.append({
                "name": parts[0],
                "type": parts[1] if len(parts) > 1 else "",
                "constraints": " ".join(parts[2:]),
            })
        elif isinstance(col, dict):
            columns.append(col)
    return columns


# EXAONE이 간헐적으로 붙이는 오염된 접두 키 → 표준 키.
# 실측: 한 테이블이 실제 컬럼을 c_columns에 담아 보내 columns가 비었고, 그 결과 최소 컬럼 3개만
# 주입된 채(id/created_at/updated_at) ERD에 표시됐다. 표준 키가 비어 있을 때만 값을 옮긴다.
_TABLE_KEY_ALIASES = {
    "c_columns": "columns",
    "c_indexes": "indexes",
    "c_constraints": "constraints",
}


def _absorb_alias_keys(tbl: dict) -> None:
    """오염된 별칭 키의 값을 표준 키가 비어 있을 때만 옮긴다 — 원본 제자리 수정."""
    if not isinstance(tbl, dict):
        return
    for alias, canonical in _TABLE_KEY_ALIASES.items():
        if alias not in tbl:
            continue
        value = tbl.pop(alias)
        if value and not tbl.get(canonical):
            logger.warning(
                "DBA 테이블 '%s' — 오염 키 %s 의 값을 %s 로 복구", tbl.get("name", "?"), alias, canonical,
            )
            tbl[canonical] = value


def _to_array_schema(data: dict) -> dict:
    """object-keyed(테이블명을 키로 하는 오브젝트) 또는 array(문자열 컬럼) 포맷의 tables를
    표준 array 포맷({name,description,columns:[{name,type,constraints}],indexes})으로 정규화.
    QA reviewer 등 다른 단계에서 여전히 두 포맷 중 하나로 응답할 수 있어 방어적으로 둔다.
    이미 완전히 정규화된 포맷이면 멱등."""
    tables = data.get("tables")
    if isinstance(tables, dict):
        array_tables = []
        for name, tbl in tables.items():
            if not isinstance(tbl, dict):
                logger.warning(
                    "_to_array_schema: 테이블 '%s' 값이 dict가 아님(%s) — 건너뜀: %r",
                    name, type(tbl).__name__, tbl,
                )
                continue
            _absorb_alias_keys(tbl)
            indexes = tbl.get("indexes", [])
            if not isinstance(indexes, list):
                indexes = [str(indexes)] if indexes else []
            array_tables.append({
                "name": name,
                "description": tbl.get("description", ""),
                "columns": _parse_column_strings(tbl.get("columns")),
                "indexes": indexes,
            })
        if not array_tables:
            logger.warning("_to_array_schema: 변환 후 테이블이 0개 — 원본 tables keys: %s", list(tables.keys()))
        data["tables"] = array_tables
        return data
    if isinstance(tables, list):
        for tbl in tables:
            if isinstance(tbl, dict):
                _absorb_alias_keys(tbl)
                tbl["columns"] = _parse_column_strings(tbl.get("columns"))
                if not isinstance(tbl.get("indexes"), list):
                    tbl["indexes"] = [str(tbl["indexes"])] if tbl.get("indexes") else []
        return data
    return data


def _table_name_variants(name: str) -> set[str]:
    """단/복수 변형을 포함한 테이블명 매칭 후보 (예: notification/notifications, category/categories)."""
    variants = {name}
    if name.endswith("ies"):
        variants.add(name[:-3] + "y")
    elif name.endswith("s"):
        variants.add(name[:-1])
    else:
        variants.add(name + "s")
    return variants


def _build_name_lookup(tables: list[dict]) -> dict[str, str]:
    """'테이블명 변형 → 실제 테이블명' 매핑 — FK 컬럼명 해석용.
    관계 합성과 FK 타입 정합 백스톱이 동일한 기준으로 참조 대상을 찾도록 공유한다."""
    lookup: dict[str, str] = {}
    for tbl in tables:
        if not isinstance(tbl, dict) or not tbl.get("name"):
            continue
        for variant in _table_name_variants(tbl["name"]):
            lookup.setdefault(variant, tbl["name"])
    return lookup


def _fk_target(col_name, name_lookup: dict[str, str]) -> str | None:
    """FK 패턴 컬럼명(`<참조테이블>_id`)에서 참조 대상 테이블명을 해석. FK가 아니면 None."""
    if not isinstance(col_name, str) or col_name == "id" or not col_name.endswith("_id"):
        return None
    return name_lookup.get(col_name[:-3])


def _synthesize_relationships(tables: list[dict]) -> list[str]:
    """LLM이 relationships를 비워둔 경우, FK 패턴 컬럼(`<prefix>_id`)으로부터
    'A (1:N) B' 관계를 결정론적으로 역추론한다. LLM 지시 불이행에 대한 안전망."""
    if not isinstance(tables, list) or not tables:
        return []

    name_lookup = _build_name_lookup(tables)

    relationships: list[str] = []
    seen: set[tuple[str, str]] = set()
    for tbl in tables:
        if not isinstance(tbl, dict) or not tbl.get("name"):
            continue
        table_name = tbl["name"]
        for col in tbl.get("columns") or []:
            ref_table = _fk_target(col.get("name") if isinstance(col, dict) else None, name_lookup)
            if not ref_table or ref_table == table_name:
                continue
            key = (ref_table, table_name)
            if key in seen:
                continue
            seen.add(key)
            relationships.append(f"{ref_table} (1:N) {table_name}")

    return relationships


def _pk_column(table: dict) -> tuple[str, str] | None:
    """테이블의 PK (컬럼명, 타입) — 'id' 컬럼 우선, 없으면 PRIMARY_KEY 제약이 붙은 첫 컬럼.
    둘 다 없으면 참조 기준을 특정할 수 없으므로 None."""
    fallback = None
    for col in table.get("columns") or []:
        if not isinstance(col, dict):
            continue
        col_name, col_type = str(col.get("name") or ""), str(col.get("type") or "").strip()
        if not col_type:
            continue
        if col_name.lower() == "id":
            return col_name, col_type
        if fallback is None and "PRIMARY_KEY" in str(col.get("constraints") or "").upper():
            fallback = (col_name, col_type)
    return fallback


def reconcile_fk_types(tables: list) -> list:
    """FK 컬럼(`<참조테이블>_id`)의 타입을 참조 대상 테이블의 PK 타입에 맞춘다 — 원본을 제자리 수정.

    worker 서브에이전트는 자기 테이블 하나만 보고 설계하므로 다른 테이블의 PK 타입을 알 수 없다.
    그래서 `users.id BIGINT`인데 `orders.user_id VARCHAR(255)`처럼 조인 불가능한 스키마가
    구조적으로 발생한다. 지금까지 이걸 잡는 건 manager_review(LLM)뿐이었고 리뷰 파싱이
    실패하면 그대로 통과했으므로, PK 타입을 단일 기준으로 삼아 결정론적으로 교정한다.
    (참조 대상이 없는 dangling FK는 컬럼 삭제/개명이 API 스펙과 어긋날 수 있어 건드리지 않고
    경고만 남긴다 — 판단은 LLM 리뷰에 맡긴다.)"""
    if not isinstance(tables, list) or not tables:
        return tables

    name_lookup = _build_name_lookup(tables)
    pk_columns = {
        tbl["name"]: pk
        for tbl in tables
        if isinstance(tbl, dict) and tbl.get("name") and (pk := _pk_column(tbl))
    }

    fixed, dangling = 0, []
    for tbl in tables:
        if not isinstance(tbl, dict) or not tbl.get("name"):
            continue
        for col in tbl.get("columns") or []:
            if not isinstance(col, dict):
                continue
            col_name = col.get("name")
            ref_table = _fk_target(col_name, name_lookup)
            if not ref_table:
                if isinstance(col_name, str) and col_name != "id" and col_name.endswith("_id"):
                    dangling.append(f"{tbl['name']}.{col_name}")
                continue
            ref_pk = pk_columns.get(ref_table)
            if not ref_pk:
                continue
            ref_pk_name, ref_pk_type = ref_pk
            cur = str(col.get("type") or "").strip()
            if cur.upper() == ref_pk_type.upper():
                continue
            logger.warning(
                "DBA FK 타입 불일치 — %s.%s: %s → %s (%s.%s 기준)",
                tbl["name"], col_name, cur or "(없음)", ref_pk_type, ref_table, ref_pk_name,
            )
            col["type"] = ref_pk_type
            fixed += 1

    if fixed:
        logger.info("DBA FK 타입 %d개 교정 — 참조 PK 타입에 정합", fixed)
    if dangling:
        logger.warning("DBA 참조 대상 없는 FK 컬럼 %d개 (교정 안 함): %s", len(dangling), dangling)
    return tables


def _existing_relationship_pairs(relationships: list, table_names: set[str]) -> set[frozenset[str]]:
    """각 relationship 문자열에 실제 테이블명이 몇 개 언급됐는지 확인해 이미 다뤄진
    테이블 쌍을 추출한다 — LLM이 일부 관계만 채운 경우 FK 합성이 이미 다룬 쌍까지
    중복 추가하지 않도록 방지."""
    pairs: set[frozenset[str]] = set()
    for rel in relationships:
        if not isinstance(rel, str):
            continue
        mentioned = [name for name in table_names if re.search(rf'\b{re.escape(name)}\b', rel)]
        for i in range(len(mentioned)):
            for j in range(i + 1, len(mentioned)):
                pairs.add(frozenset((mentioned[i], mentioned[j])))
    return pairs


def _tables_haystack(tables: list[dict]) -> list[str]:
    """array-format tables에서 커버리지 매칭용 한국어 텍스트 추출."""
    texts = []
    for tbl in tables or []:
        if isinstance(tbl, dict):
            texts.append(str(tbl.get("name", "")))
            texts.append(str(tbl.get("description", "")))
    return texts


# EXAONE이 테이블명 대신 설명 문장을 뱉거나(예: barcodes_can_be_used_to_...) 컬럼을 통째로
# 비워내는 것을 결정론적으로 교정하는 백스톱. api_agent._sanitize_plan_paths의 DBA판.
_VALID_TABLE_NAME_RE = re.compile(r'^[a-z][a-z0-9_]*$')
_MAX_TABLE_NAME_LEN = 40
_MIN_COLUMNS = [
    {"name": "id", "type": "BIGINT", "constraints": "PRIMARY_KEY AUTO_INCREMENT"},
    {"name": "created_at", "type": "DATETIME", "constraints": "NOT_NULL"},
    {"name": "updated_at", "type": "DATETIME", "constraints": "NOT_NULL"},
]


_VALID_TYPE_RE = re.compile(
    r'^(BIGINT|INT|INTEGER|SMALLINT|VARCHAR\(\d+\)|CHAR\(\d+\)|TEXT|BOOLEAN|DATETIME|DATE|TIMESTAMP|DECIMAL\(\d+,\d+\)|FLOAT|DOUBLE)$',
    re.I,
)


# 공백으로 끊긴 제약조건 키워드 → 표준 언더스코어 형태 (예: 'NOT NULL' → 'NOT_NULL')
_CONSTRAINT_PHRASE_FIXES = [
    (re.compile(r'\bPRIMARY\s+KEY\b', re.I), 'PRIMARY_KEY'),
    (re.compile(r'\bAUTO\s+INCREMENT\b', re.I), 'AUTO_INCREMENT'),
    (re.compile(r'\bFOREIGN\s+KEY\b', re.I), 'FOREIGN_KEY'),
    (re.compile(r'\bNOT\s+NULL\b', re.I), 'NOT_NULL'),
    (re.compile(r'\bDEFAULT\s+FALSE\b', re.I), 'DEFAULT_FALSE'),
    (re.compile(r'\bDEFAULT\s+TRUE\b', re.I), 'DEFAULT_TRUE'),
]
# 개별 토큰 오타 → 표준형 (빈 문자열이면 제거). EXAONE 실측 오타: NOTULL/NOTNULL/NOT_EMPTY 등
_CONSTRAINT_TOKEN_FIXES = {
    'NOTULL': 'NOT_NULL', 'NOTNULL': 'NOT_NULL', 'NOTNUL': 'NOT_NULL', 'NOT_NUL': 'NOT_NULL',
    'NOT_EMPTY': 'NOT_NULL', 'NOT_HIDDEN': '',
    'PRIMARYKEY': 'PRIMARY_KEY', 'AUTOINCREMENT': 'AUTO_INCREMENT', 'FOREIGNKEY': 'FOREIGN_KEY',
    'DEFAULTFALSE': 'DEFAULT_FALSE', 'DEFAULTTRUE': 'DEFAULT_TRUE',
}


def _clean_constraint(s) -> str:
    """제약조건 문자열을 정리한다: (1) 'NOT NULL' 같은 공백 변형과 NOTULL 같은 오타를 표준형으로
    교정, (2) 반복 토큰 폭주(예: 'NOT_NOT_HIDDEN_DEFAULT_NOT_HIDDEN...')·중복 제거, (3) 길이 제한.
    DEFAULT 값·REFERENCES 등 인식 못하는 토큰은 건드리지 않고 그대로 둔다."""
    if not isinstance(s, str):
        return ""
    for rx, rep in _CONSTRAINT_PHRASE_FIXES:
        s = rx.sub(rep, s)
    out, seen = [], set()
    for t in s.split():
        if len(t) > 24:  # 반복 폭주 토큰
            continue
        fixed = _CONSTRAINT_TOKEN_FIXES.get(t.upper())
        if fixed is not None:
            t = fixed
        if not t or t in seen:
            continue
        seen.add(t)
        out.append(t)
    return " ".join(out)[:80]


def _infer_column_type(name: str) -> tuple[str, str]:
    """컬럼명으로 타입·제약조건을 결정론적으로 추론 — EXAONE이 타입 없이 컬럼명만 낸 경우 보강."""
    n = (name or "").lower()
    if n == "id":
        return "BIGINT", "PRIMARY_KEY AUTO_INCREMENT"
    if n.endswith("_id"):
        return "BIGINT", "NOT_NULL"
    if n in ("created_at", "updated_at"):
        return "DATETIME", "NOT_NULL"
    if n.endswith("_at") or n.endswith("_date") or "date" in n or "time" in n:
        return "DATETIME", "NULL"
    if n.startswith("is_") or n.startswith("has_") or n.endswith("_flag"):
        return "BOOLEAN", "DEFAULT_FALSE"
    if any(k in n for k in ("quantity", "count", "amount", "stock", "qty", "price", "cost")):
        return "DECIMAL(10,2)", "NOT_NULL"
    if "email" in n:
        return "VARCHAR(255)", "UNIQUE"
    if any(k in n for k in ("description", "content", "memo", "body", "text")):
        return "TEXT", "NULL"
    return "VARCHAR(255)", "NULL"


# 숫자 제거 결과로 남으면 안 되는 이름 — SQL 예약어라 DDL이 깨지거나, 원본보다 의미가 없어지는 경우.
# 실측: 'table_5' → 'table' 이 되어 예약어 테이블명이 됐고, 인덱스는 table_5를 가리킨 채 남아
# 이름과 인덱스가 서로 어긋났다. 이런 경우엔 숫자를 남긴 원본이 차라리 낫다.
_RESERVED_TABLE_NAMES = {
    "table", "tables", "column", "columns", "index", "indexes", "order", "group",
    "select", "from", "where", "key", "primary", "foreign", "user", "data", "entity", "temp",
}

# "INDEX idx_x ON tbl(col_a, col_b)" 의 괄호 안 컬럼 목록
_INDEX_COLUMNS_RE = re.compile(r'\(([^)]*)\)')
# 오타난 컬럼명을 실제 컬럼으로 되돌릴 때의 최소 유사도 (is_exped → is_expired)
_INDEX_COLUMN_MATCH_CUTOFF = 0.75


# 컬럼명이 한글로 나온 경우의 결정론적 영문화 사전. 실측: consumption_logs에
# 레시피_id / 소모량 / 기록일시 / 재료_id 가 그대로 들어가 SQL로 쓸 수 없고 FK 연결도 끊겼다.
# 프롬프트·재생성으로 대부분 걸러지고, 여기는 마지막 안전망이다.
_KO_COLUMN_TERMS = {
    "아이디": "id", "이름": "name", "제목": "title", "내용": "content", "설명": "description",
    "수량": "quantity", "소모량": "quantity", "사용량": "quantity", "개수": "count",
    "가격": "price", "금액": "amount", "단위": "unit", "상태": "status", "종류": "type",
    "분류": "category", "카테고리": "category", "메모": "memo", "비고": "note",
    "날짜": "date", "일자": "date", "시간": "time", "일시": "datetime",
    "기록일시": "recorded_at", "등록일": "created_at", "생성일": "created_at",
    "수정일": "updated_at", "삭제일": "deleted_at", "만료일": "expires_at",
    "유통기한": "expiry_date", "사용자": "user", "회원": "member", "재료": "ingredient",
    "레시피": "recipe", "알림": "notification", "이메일": "email", "비밀번호": "password",
    "전화번호": "phone_number", "주소": "address", "이미지": "image", "순서": "sort_order",
    "여부": "flag", "활성화": "is_active", "읽음": "is_read",
}
_NON_ASCII_RE = re.compile(r'[^\x00-\x7F]')


# EXAONE이 name 대신 쓰는 키들 — 이름이 '없는' 게 아니라 다른 키에 담긴 경우를 먼저 회수한다
_PLAN_NAME_ALIASES = ("name", "table", "tableName", "table_name", "테이블명", "이름")


def _plan_table_name(item: dict) -> str:
    """plan 항목에서 쓸 만한 테이블명을 뽑는다. 없으면 빈 문자열."""
    for key in _PLAN_NAME_ALIASES:
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _korean_column_names(raw_columns) -> list[str]:
    """컬럼 정의(문자열 또는 dict)에서 비-ASCII 컬럼명만 추린다 — worker 재생성 판정용."""
    found = []
    for col in _parse_column_strings(raw_columns):
        name = str(col.get("name") or "")
        if name and _NON_ASCII_RE.search(name):
            found.append(name)
    return found


def _englishize_column(col_name: str, idx: int, name_lookup: dict[str, str]) -> str:
    """한글 컬럼명을 영문 식별자로 변환.
    '<한글>_id'는 먼저 실제 테이블 설명/이름에서 참조 대상을 찾아 '<테이블>_id'로 되돌린다."""
    stem = col_name[:-3] if col_name.endswith("_id") else None
    if stem:
        mapped = _KO_COLUMN_TERMS.get(stem)
        # 사전에 있으면 그 영문 어간의 테이블을 우선 찾고, 없으면 어간 자체를 쓴다
        for candidate in filter(None, (mapped, stem)):
            target = name_lookup.get(candidate)
            if target:
                return f"{target.rstrip('s')}_id" if not target.endswith("_id") else target
        if mapped:
            return f"{mapped}_id"
    direct = _KO_COLUMN_TERMS.get(col_name)
    if direct:
        return direct
    parts = [_KO_COLUMN_TERMS.get(tok, "") for tok in col_name.split("_")]
    if all(parts):
        return "_".join(parts)
    return f"col_{idx}"


def _normalize_column_names(table: dict, name_lookup: dict[str, str]) -> None:
    """비-ASCII 컬럼명을 영문으로 바꾸고, 같은 이름을 참조하던 인덱스 문자열도 함께 고친다."""
    columns = table.get("columns")
    if not isinstance(columns, list):
        return
    existing = {str(c.get("name")) for c in columns if isinstance(c, dict) and c.get("name")}
    renames: list[tuple[str, str]] = []
    for i, col in enumerate(columns):
        if not isinstance(col, dict):
            continue
        old = str(col.get("name") or "")
        if not old or not _NON_ASCII_RE.search(old):
            continue
        new = _englishize_column(old, i, name_lookup)
        while new in existing:
            new = f"{new}_2"
        existing.add(new)
        col["name"] = new
        renames.append((old, new))

    if not renames:
        return
    logger.warning("DBA 테이블 '%s' — 한글 컬럼명 %d개 영문화: %s", table.get("name"), len(renames), renames)
    indexes = table.get("indexes")
    if isinstance(indexes, list):
        table["indexes"] = [
            _rename_in_text(str(entry), renames) if isinstance(entry, (str, int)) else entry
            for entry in indexes
        ]


def _rename_in_text(text: str, renames: list[tuple[str, str]]) -> str:
    for old, new in renames:
        text = text.replace(old, new)
    return text


# 'users_final_fix'처럼 리뷰 단계가 원본의 수정본을 별도 테이블로 덧붙일 때 붙는 접미사.
# (실측: manager 리뷰가 users의 수정본을 users_final_fix 라는 새 테이블로 추가해 중복 생성)
_TABLE_META_SUFFIX_RE = re.compile(r'_(final|fix|fixed|update[d]?|new|copy|revised|modified|v\d+)$')


def _base_table_name(name: str) -> str:
    """메타 접미사를 반복 제거한 기준 이름 (users_final_fix → users)."""
    prev = None
    while prev != name:
        prev = name
        name = _TABLE_META_SUFFIX_RE.sub("", name)
    return name


def dedupe_meta_tables(tables: list) -> list:
    """리뷰가 만든 '원본의 수정본' 중복 테이블을 정리한다.
    기준 테이블이 실제로 존재할 때만 병합하며, 컬럼이 더 많은 쪽을 기준 이름으로 남긴다."""
    if not isinstance(tables, list) or len(tables) < 2:
        return tables
    by_name = {t["name"]: t for t in tables if isinstance(t, dict) and t.get("name")}

    dropped = []
    out = []
    for t in tables:
        if not isinstance(t, dict) or not t.get("name"):
            out.append(t)
            continue
        name = t["name"]
        base = _base_table_name(name)
        if base == name or base not in by_name:
            out.append(t)
            continue
        original = by_name[base]
        if len(t.get("columns") or []) > len(original.get("columns") or []):
            # 수정본이 더 충실하면 내용을 기준 테이블로 옮긴다 (이름은 기준 이름 유지)
            original["columns"] = t.get("columns")
            original["indexes"] = t.get("indexes") or original.get("indexes")
            if t.get("description"):
                original["description"] = t["description"]
        dropped.append(f"{name} → {base}")

    if dropped:
        logger.warning("DBA 리뷰 중복 테이블 %d개 병합: %s", len(dropped), dropped)
    return out


def _repair_indexes(table: dict) -> None:
    """인덱스가 참조하는 컬럼명을 실제 컬럼과 대조해 오타는 교정하고, 대응 컬럼이 없으면 제거한다.
    실측: 'INDEX idx_ingredients_is_expired ON ingredients(is_exped)' — 존재하지 않는 컬럼 참조.
    괄호 안 컬럼 목록을 읽을 수 없는 형식은 판단하지 않고 그대로 둔다."""
    indexes = table.get("indexes")
    columns = table.get("columns")
    if not isinstance(indexes, list) or not indexes or not isinstance(columns, list):
        return
    col_names = [str(c["name"]) for c in columns if isinstance(c, dict) and c.get("name")]
    if not col_names:
        return

    kept, fixed, dropped = [], [], []
    for entry in indexes:
        text = str(entry)
        match = _INDEX_COLUMNS_RE.search(text)
        if not match:
            kept.append(entry)
            continue
        referenced = [c.strip() for c in match.group(1).split(",") if c.strip()]
        unknown = [c for c in referenced if c not in col_names]
        if not unknown:
            kept.append(entry)
            continue
        repaired = text
        for bad in unknown:
            close = difflib.get_close_matches(bad, col_names, n=1, cutoff=_INDEX_COLUMN_MATCH_CUTOFF)
            if not close:
                repaired = None
                break
            repaired = re.sub(rf'\b{re.escape(bad)}\b', close[0], repaired)
        if repaired is None:
            dropped.append(text)
        else:
            fixed.append((text, repaired))
            kept.append(repaired)

    if fixed:
        logger.warning("DBA 테이블 '%s' — 인덱스 컬럼 오타 %d개 교정: %s", table.get("name"), len(fixed), fixed)
    if dropped:
        logger.warning(
            "DBA 테이블 '%s' — 없는 컬럼을 참조하는 인덱스 %d개 제거: %s",
            table.get("name"), len(dropped), dropped,
        )
    if fixed or dropped:
        table["indexes"] = kept


def _slugify_table_name(name, idx: int) -> str:
    """비정상 테이블명을 짧은 snake_case 식별자로 강제 변환 (문장형이면 앞 3토큰만)."""
    s = re.sub(r'[^a-z0-9]+', '_', str(name).lower()).strip('_')
    tokens = [t for t in s.split('_') if t]
    s = '_'.join(tokens[:3])[:_MAX_TABLE_NAME_LEN].strip('_')
    if not s or not s[0].isalpha():
        s = f"table_{idx}"
    return s


def sanitize_tables(tables: list) -> list:
    """최종 테이블 목록의 (1) 문장형/비식별자 테이블명 슬러그화, (2) 컬럼 0개 테이블에
    최소 컬럼 주입을 수행한다. EXAONE 지시 불이행에 대한 결정론적 안전망 — 원본을 제자리 수정."""
    if not isinstance(tables, list):
        return tables
    # 한글 '<대상>_id' 컬럼을 실제 테이블로 되돌리기 위한 조회표 (컬럼 영문화보다 먼저 만든다)
    name_lookup = _build_name_lookup([t for t in tables if isinstance(t, dict)])
    seen: set[str] = set()
    out = []
    for i, t in enumerate(tables):
        if not isinstance(t, dict):
            continue
        name = t.get("name") or ""
        if not isinstance(name, str) or not _VALID_TABLE_NAME_RE.match(name) or len(name) > _MAX_TABLE_NAME_LEN:
            new = _slugify_table_name(name, i)
            logger.warning("DBA 테이블명 비정상 — 슬러그화: %r → %s", str(name)[:60], new)
            name = new
        elif re.search(r'\d', name):
            # 유효하지만 숫자가 낀 이름(예: refriger60_ownerships)은 EXAONE 손상일 가능성이 높으므로
            # 숫자를 제거한다. (중복 방지 접미사 _2/_3는 아래 dedup 단계에서 숫자 제거 후 붙으므로 안전)
            stripped = re.sub(r'_+', '_', re.sub(r'\d+', '', name)).strip('_')
            if stripped in _RESERVED_TABLE_NAMES:
                # 'table_5' → 'table' 처럼 예약어/무의미한 이름이 되면 원본을 유지한다 —
                # 인덱스 등 다른 곳이 원본 이름을 참조하고 있어 바꾸면 오히려 어긋난다
                logger.warning("DBA 테이블명 숫자 제거 시 예약어가 됨(%s → %s) — 원본 유지", name, stripped)
            elif stripped and _VALID_TABLE_NAME_RE.match(stripped):
                logger.warning("DBA 테이블명 숫자 혼입 제거: %s → %s", name, stripped)
                name = stripped
        base, n = name, 2
        while name in seen:
            name = f"{base}_{n}"
            n += 1
        seen.add(name)
        t["name"] = name
        cols = t.get("columns")
        if not isinstance(cols, list) or not cols:
            logger.warning("DBA 테이블 '%s' 컬럼 0개 — 최소 컬럼 주입", name)
            t["columns"] = [dict(c) for c in _MIN_COLUMNS]
        else:
            # 타입이 없거나(이름만) 손상된(예: DAT4ETIME) 컬럼은 이름 기반 추론으로 보강,
            # 제약조건의 반복 토큰 폭주·중복은 정리
            filled = 0
            for c in cols:
                if not isinstance(c, dict) or not c.get("name"):
                    continue
                ty = str(c.get("type") or "").strip()
                if not ty or not _VALID_TYPE_RE.match(ty):
                    inf_ty, inf_con = _infer_column_type(c.get("name"))
                    c["type"] = inf_ty
                    if not str(c.get("constraints") or "").strip():
                        c["constraints"] = inf_con
                    filled += 1
                c["constraints"] = _clean_constraint(c.get("constraints"))
            if filled:
                logger.warning("DBA 테이블 '%s' — 타입 없음/손상 컬럼 %d개 추론 보강", name, filled)
        # 인덱스 대조 전에 컬럼명을 영문화해야 인덱스가 '없는 컬럼 참조'로 오판돼 삭제되지 않는다
        _normalize_column_names(t, name_lookup)
        # 컬럼이 확정된 뒤에 인덱스를 대조해야 교정 기준이 최종 컬럼 목록을 반영한다
        _repair_indexes(t)
        out.append(t)
    return out


PLAN_PROMPT = """당신은 시니어 DBA입니다.
아래 지시사항과 기능 목록을 바탕으로 이 서비스에 필요한 전체 DB 테이블 목록(스켈레톤)을 설계하세요.
상세 컬럼/인덱스 설계는 이후 단계에서 채울 것이므로 지금은 테이블명과 목적(purpose)만 결정하세요.

설계 규칙:
- 기능 목록의 각 기능 영역(등록/조회/알림/추천/통계 등)마다 저장이 필요한 도메인 엔티티를 빠짐없이 테이블로 설계하세요.
  "users"와 범용 카탈로그 테이블 한두 개만 만들고 끝내는 뭉뚱그린 설계는 금지 — 기능 목록에 명시된
  구체적인 대상(예: 재고 항목, 알림 설정, 추천 이력 등)마다 별도 테이블이 필요한지 검토하세요.
- 인증/회원 기능이 있으면 users 테이블 필수
- 테이블명은 영문 소문자 snake_case, 복수형 (예: users, ingredients, notifications)
- 최소 {min_tables}개 이상의 테이블을 설계하세요 (개수 상한 없음, 과도한 중복/분할은 지양)

응답 형식 (JSON만):
{{
  "plan": [
    {{"name": "테이블명", "purpose": "이 테이블이 어떤 기능을 지원하는지 한 줄 설명 (한글)"}}
  ]
}}

지시사항:
{instruction}

기능 목록:
{feature_str}

컨텍스트:
{context}
"""

TABLE_NAMING_PROMPT = """아래는 DB 테이블의 용도 설명 {count}개입니다.
각 용도에 맞는 영문 테이블명을 순서대로 지어주세요.

규칙:
- 영문 소문자 snake_case, 복수형 (예: notification_settings, recommendation_logs)
- 기존 테이블명과 중복 금지: {existing}
- table_1, entity_2 같은 자리표시자 금지 — 반드시 용도를 반영한 의미 있는 이름
- 정확히 {count}개를 순서대로

용도 목록:
{items}

응답 형식 (JSON만): {{"names": ["첫번째_용도의_테이블명", "..."]}}
"""

TABLE_SPEC_PROMPT = """당신은 시니어 DBA입니다.
아래 테이블 1개에 대한 상세 컬럼/인덱스 설계를 JSON으로 작성하세요.

담당 테이블:
- name: {name}
- purpose: {purpose}

전체 테이블 목록 (외래키를 참조해야 한다면 반드시 이 목록 안에서만 참조하세요): {all_table_names}

설계 규칙:
- 각 컬럼은 문자열 한 줄: "컬럼명 타입 제약조건들"
- ⚠️ 컬럼명은 반드시 영문 소문자 snake_case. 한글 컬럼명 절대 금지
  (예: '소모량' ❌ → quantity ⭕, '기록일시' ❌ → recorded_at ⭕, '재료_id' ❌ → ingredient_id ⭕)
- 인덱스 정의에 쓰는 컬럼명도 위에서 정의한 영문 컬럼명과 정확히 일치해야 함
- 허용 타입: BIGINT, VARCHAR(255), VARCHAR(100), VARCHAR(50), TEXT, BOOLEAN, DATETIME, DECIMAL(10,2)
- 제약조건 키워드(공백 구분): PRIMARY_KEY, AUTO_INCREMENT, NOT_NULL, NULL, UNIQUE, FOREIGN_KEY, DEFAULT_FALSE, DEFAULT_TRUE
- "id BIGINT PRIMARY_KEY AUTO_INCREMENT", "created_at DATETIME NOT_NULL", "updated_at DATETIME NOT_NULL" 필수
- 다른 테이블을 참조하는 외래키 컬럼명은 참조테이블_id 형식 (예: user_id)

응답 형식 (JSON만, name/description은 위 값 그대로 유지):
{{
  "name": "{name}",
  "description": "{purpose}",
  "columns": ["id BIGINT PRIMARY_KEY AUTO_INCREMENT", "..."],
  "indexes": ["INDEX idx_{name}_x ON {name}(x)"]
}}

컨텍스트:
{context}
"""

MANAGER_REVIEW_PROMPT = """당신은 시니어 DBA 겸 DB manager입니다.
아래는 sub-agent들이 작성한 DB 테이블 상세 설계 전체입니다. 정확히 검증하고, 문제가 있는 테이블만 직접 수정하세요.

=== 테이블 목록 ===
{tables_json}
=================

기능 목록: {feature_str}
{missing_note}

검증 기준:
- 모든 FK 컬럼(예: user_id)이 실제 참조 대상 테이블의 PK 타입과 일치하는가? 존재하지 않는 테이블을 참조하지 않는가?
- 각 테이블에 id/created_at/updated_at 필수 컬럼이 있는가?
- relationships가 실제 FK 구조와 일치하는가? (테이블이 2개 이상인데 비어있으면 결함)
- [결정론적 커버리지 검사]에 나열된 기능이 있다면 반드시 그 기능을 지원하는 테이블을 patches로 새로 추가하세요
  (기존에 없던 테이블명도 patches에 키로 추가 가능 — 문제 있는/누락된 테이블만 담고, 나머지는 그대로 두세요)

JSON:
{{
  "patches": {{"테이블명": {{"description":"...", "columns":["..."], "indexes":["..."]}}}},
  "relationships": ["테이블A (1:N) 테이블B", "..."],
  "prdIssues": "PRD에서 기능 목록과 어긋나 보이는 부분이 있으면 한 줄, 없으면 빈 문자열"
}}

문제되는 테이블이 하나도 없으면 patches는 {{}}로, relationships는 검증 결과를 반영한 최종 목록으로 응답하세요.
"""


class _DbaGraphState(TypedDict):
    plan: list
    tables: Annotated[list, operator.add]  # worker fan-out 누적 전용 — manager_review는 여기 쓰지 않음
    final_tables: list
    relationships: list
    prd_issues: str
    ctx: dict


class DbaAgent:

    def __init__(
        self,
        client: AsyncOpenAI,
        model: str = "gpt-4o-mini",
    ):
        self._client = client
        self._model = model
        self._graph = self._build_graph()

    def _build_graph(self):
        graph = StateGraph(_DbaGraphState)
        graph.add_node("manager_plan", self._manager_plan_node)
        graph.add_node("worker", self._worker_node)
        graph.add_node("manager_review", self._manager_review_node)
        graph.add_edge(START, "manager_plan")
        graph.add_conditional_edges("manager_plan", self._dispatch, ["worker"])
        graph.add_edge("worker", "manager_review")
        graph.add_edge("manager_review", END)
        return graph.compile()

    async def execute(self, state: PipelineState, dump=None) -> PipelineState:
        logger.info("DBA 에이전트 시작 (manager plan + 동적 fan-out 서브그래프)")

        feature_str = "- " + "\n- ".join(state.feature_list) if state.feature_list else "(기능 목록 없음)"
        graph_input: _DbaGraphState = {
            "plan": [],
            "tables": [],
            "final_tables": [],
            "relationships": [],
            "prd_issues": "",
            "ctx": {
                "instruction": state.dba_instruction or "",
                "feature_str": feature_str,
                "feature_list": state.feature_list,
                "context": state.context_prompt or "",
                "dump": dump,
            },
        }

        result = await self._graph.ainvoke(graph_input)

        tables = result.get("final_tables") or result.get("tables") or []
        relationships = result.get("relationships") or []

        # 결정론적 커버리지 체크 — manager_plan 단계에서 이미 대부분 걸러지지만,
        # worker/manager_review 이후에도 기능이 반영 안 됐으면 targeted hint와 함께
        # manager_review를 최대 2회 더 재실행 (그래프 밖에서 직접 노드 재호출).
        # Phase2 전체 재실행(Phase3 재시도)은 feature_list 자체가 달라져 수렴하지 않으므로
        # 이 in-run 루프가 커버리지를 실질적으로 채우는 주된 수단이다.
        _MAX_COVERAGE_ROUNDS = 3
        for round_ in range(_MAX_COVERAGE_ROUNDS):
            missing = uncovered_features(state.feature_list, _tables_haystack(tables))
            if not missing:
                break
            logger.warning("DBA 커버리지 부족 (round %d): 미반영 기능 %d개 — %s", round_ + 1, len(missing), missing)
            if round_ >= _MAX_COVERAGE_ROUNDS - 1:
                logger.warning("DBA 커버리지 보정 한도 도달 — 미반영 상태로 진행")
                break
            review_result = await self._manager_review_node({
                "ctx": {**graph_input["ctx"], "missing_note": missing_features_note(missing, "DB 스키마")},
                "tables": tables,
                "relationships": relationships,
            })
            tables = review_result.get("final_tables", tables)
            relationships = review_result.get("relationships", relationships)

        # 결정론적 백스톱: 문장형 테이블명 슬러그화 + 컬럼 0개 테이블 최소 컬럼 주입
        # (관계 합성이 테이블명을 참조하므로 그 전에 정규화)
        tables = sanitize_tables(tables)
        # 이름이 확정된 뒤에 리뷰가 덧붙인 '수정본' 중복 테이블을 병합
        tables = dedupe_meta_tables(tables)
        # FK 컬럼 타입을 참조 대상 PK 타입에 정합 — sanitize가 타입 추론/이름 슬러그화를
        # 끝낸 뒤에 실행해야 참조 해석과 비교 기준이 최종 상태를 반영한다.
        tables = reconcile_fk_types(tables)

        table_names = {t["name"] for t in tables if isinstance(t, dict) and t.get("name")}
        covered = _existing_relationship_pairs(relationships, table_names)
        synth_missing = [
            rel for rel in _synthesize_relationships(tables)
            if frozenset(rel.split(" (1:N) ")) not in covered
        ]
        if synth_missing:
            logger.warning(
                "DBA relationships 일부 누락 — FK 컬럼 기반 자동 보강 %d개: %s",
                len(synth_missing), synth_missing,
            )
            relationships = relationships + synth_missing

        clean = json.dumps({"tables": tables, "relationships": relationships}, ensure_ascii=False)
        prd_issues = result.get("prd_issues", "")

        if prd_issues:
            logger.warning("DBA → PRD 피드백: %s", prd_issues)
        logger.info("DBA 에이전트 완료 — 테이블 %d개", len(tables))
        return state.copy(
            db_schema=clean,
            prd_feedback_from_dba=prd_issues,
            status_message="DBA 에이전트 완료 — DB 스키마 생성",
        )

    async def _name_unnamed_tables(self, unnamed: list[dict], seen: set[str], ctx: dict) -> None:
        """이름이 빠진 plan 항목에 purpose를 근거로 영문 테이블명을 붙인다 — 원본 제자리 수정.
        purpose가 한글이라 결정론적 슬러그화가 불가능해 LLM에 명명만 따로 요청하고,
        그마저 실패하면 마지막에야 table_{n} 형태로 떨어진다."""
        purposes = [str(p.get("purpose") or "(설명 없음)") for p in unnamed]
        prompt = TABLE_NAMING_PROMPT.format(
            existing=", ".join(sorted(seen)) or "(없음)",
            items="\n".join(f"{i + 1}. {p}" for i, p in enumerate(purposes)),
            count=len(unnamed),
        )
        names: list = []
        try:
            raw = await self._call(prompt, max_tokens=800, system=MANAGER_SYSTEM,
                                   enable_thinking=False, temperature=_PLAN_TEMPERATURE)
            parsed = try_parse_json(raw)
            if isinstance(parsed, dict) and isinstance(parsed.get("names"), list):
                names = parsed["names"]
        except Exception as e:
            logger.warning("DBA 테이블 명명 요청 실패: %s", e)

        for i, item in enumerate(unnamed):
            candidate = names[i] if i < len(names) else None
            name = candidate.strip() if isinstance(candidate, str) else ""
            if not name or not _VALID_TABLE_NAME_RE.match(name) or name in _RESERVED_TABLE_NAMES:
                name = f"table_{len(seen) + 1}"
                logger.warning("DBA 테이블 명명 실패 — purpose=%r → %s", purposes[i][:60], name)
            while name in seen:
                name = f"{name}_2"
            item["name"] = name
            seen.add(name)

    async def _manager_plan_node(self, state: dict) -> dict:
        ctx = state["ctx"]
        min_tables = min_table_count(len(ctx["feature_list"]))
        prompt = PLAN_PROMPT.format(
            instruction=ctx["instruction"],
            feature_str=ctx["feature_str"],
            context=ctx["context"],
            min_tables=min_tables,
        )

        data = None
        best: dict | None = None
        best_uncovered = None
        for attempt in range(3):
            raw = await self._call(prompt, max_tokens=4000, system=MANAGER_SYSTEM, enable_thinking=False, temperature=_PLAN_TEMPERATURE)
            if ctx.get("dump"):
                ctx["dump"].log_raw("DBA_MANAGER_PLAN", attempt + 1, raw)
            candidate = try_parse_json(raw)
            plan_list = candidate.get("plan") if candidate and isinstance(candidate.get("plan"), list) else []
            missing = uncovered_features(ctx["feature_list"], [p.get("purpose", "") for p in plan_list if isinstance(p, dict)])
            if best_uncovered is None or len(missing) < best_uncovered:
                best, best_uncovered = candidate, len(missing)
            if (
                candidate and isinstance(candidate.get("plan"), list)
                and len(candidate["plan"]) >= min_tables
                and not has_suspicious_script(candidate)
                and not missing
            ):
                data = candidate
                break
            logger.warning(
                "DBA manager_plan 파싱 실패/개수 부족/기능 커버리지 미달 (attempt %d) — %d/%d개, 미반영 기능: %s — 재생성",
                attempt + 1, len(plan_list), min_tables, missing,
            )

        if not data:
            data = best
            if best_uncovered:
                logger.warning("DBA manager_plan — 커버리지 기준 미달이지만 재생성 한도 도달, 최선의 결과로 진행")
        if not data or not isinstance(data.get("plan"), list) or not data["plan"]:
            logger.error("DBA manager_plan 최종 실패 — fallback 스켈레톤 사용")
            data = self._fallback_plan(ctx["feature_list"])

        plan = data.get("plan") or self._fallback_plan(ctx["feature_list"])["plan"]
        # 이름 정규화 + 중복 방지 — EXAONE이 name 키를 누락/비문자열로 내는 경우도 방어
        # (name 없는 항목 2개가 있으면 기존 코드가 p["name"]에서 KeyError로 파이프라인을 중단시켰음)
        seen_names: set[str] = set()
        unnamed: list[dict] = []
        for i, p in enumerate(plan):
            if not isinstance(p, dict):
                continue
            name = _plan_table_name(p)
            if not name:
                # 여기서 table_{i} 로 때우면 의미 없는 테이블명이 최종 ERD까지 그대로 나간다
                # (실측: table_5, table_6). 아래에서 purpose 기반으로 이름을 받아온다.
                unnamed.append(p)
                continue
            while name in seen_names:
                name = f"{name}_2"
            p["name"] = name
            seen_names.add(name)

        if unnamed:
            logger.warning("DBA manager_plan — 이름 없는 테이블 %d개, purpose 기반 명명 요청", len(unnamed))
            await self._name_unnamed_tables(unnamed, seen_names, ctx)

        logger.info("DBA manager_plan 완료 — 테이블 %d개 계획: %s", len(plan), [p.get("name") for p in plan if isinstance(p, dict)])
        return {"plan": plan}

    def _fallback_plan(self, feature_list: list[str]) -> dict:
        plan = [{"name": "users", "purpose": "사용자 계정 정보"}]
        for i, f in enumerate(feature_list):
            plan.append({"name": f"feature_{i}_records", "purpose": f})
        return {"plan": plan}

    def _dispatch(self, state: dict) -> list[Send]:
        ctx = state["ctx"]
        all_table_names = [p.get("name") for p in state["plan"] if isinstance(p, dict)]
        sends = []
        for skeleton in state["plan"]:
            sends.append(Send("worker", {
                "skeleton": skeleton,
                "all_table_names": all_table_names,
                "context": ctx["context"],
                "dump": ctx.get("dump"),
            }))
        return sends

    async def _worker_node(self, state: dict) -> dict:
        skeleton = state["skeleton"]
        context = state["context"]
        dump = state.get("dump")

        name = skeleton.get("name", "unknown_table")
        purpose = skeleton.get("purpose", "")

        prompt = TABLE_SPEC_PROMPT.format(
            name=name,
            purpose=purpose,
            all_table_names=", ".join(state.get("all_table_names") or []),
            context=context,
        )

        label = f"DBA_TABLE_{name}"
        data = None
        best: dict | None = None
        for attempt in range(3):
            raw = await self._call(prompt, max_tokens=1500, system=WORKER_SYSTEM, enable_thinking=False)
            if dump:
                dump.log_raw(label, attempt + 1, raw)
            candidate = try_parse_json(raw)
            if candidate and isinstance(candidate, dict) and candidate.get("columns"):
                if has_suspicious_script(candidate):
                    logger.warning("%s 스크립트 오염 감지 (attempt %d) — 재생성", label, attempt + 1)
                    continue
                # 한글 컬럼명은 SQL에서 쓸 수 없고 FK 연결도 끊기므로 재생성 — 3회 모두
                # 실패하면 sanitize_tables가 결정론적으로 영문화한다(그래도 최선의 결과는 보관)
                korean = _korean_column_names(candidate.get("columns"))
                if korean:
                    best = best or candidate
                    logger.warning(
                        "%s 한글 컬럼명 %s (attempt %d) — 재생성", label, korean, attempt + 1,
                    )
                    continue
                data = candidate
                break
            logger.warning("%s 파싱 실패 또는 컬럼 없음 (attempt %d) — 재생성", label, attempt + 1)
        data = data or best

        if not data:
            logger.error("%s 최종 파싱 실패 — 최소 스펙으로 대체", label)
            data = {
                "name": name,
                "description": purpose,
                "columns": [
                    "id BIGINT PRIMARY_KEY AUTO_INCREMENT",
                    "created_at DATETIME NOT_NULL",
                    "updated_at DATETIME NOT_NULL",
                ],
                "indexes": [],
            }
        else:
            data["name"] = name
            data.setdefault("description", purpose)

        data["columns"] = _parse_column_strings(data.get("columns"))
        if not isinstance(data.get("indexes"), list):
            data["indexes"] = [str(data["indexes"])] if data.get("indexes") else []

        return {"tables": [data]}

    async def _manager_review_node(self, state: dict) -> dict:
        ctx = state["ctx"]
        tables = state.get("tables", [])
        relationships = state.get("relationships", [])

        if not tables:
            logger.warning("DBA manager_review — 생성된 테이블 없음, 검토 생략")
            return {}

        by_name = {t["name"]: t for t in tables if isinstance(t, dict) and t.get("name")}
        missing_note = ctx.get("missing_note", "")
        if not missing_note:
            missing = uncovered_features(ctx["feature_list"], _tables_haystack(tables))
            missing_note = missing_features_note(missing, "DB 스키마")

        prompt = MANAGER_REVIEW_PROMPT.format(
            tables_json=json.dumps(tables, ensure_ascii=False),
            feature_str=ctx["feature_str"],
            missing_note=missing_note,
        )

        prd_issues = ""
        try:
            review = None
            for parse_attempt in range(2):
                raw = await self._call(prompt, max_tokens=16384, system=MANAGER_SYSTEM, enable_thinking=False)
                if ctx.get("dump"):
                    ctx["dump"].log_raw("DBA_MANAGER_REVIEW", parse_attempt + 1, raw)
                review = try_parse_json(raw)
                if review and isinstance(review, dict):
                    break
                logger.warning("DBA manager 리뷰 파싱 실패 (시도 %d) — 재시도", parse_attempt + 1)
            if review and isinstance(review, dict):
                prd_issues = review.get("prdIssues", "") or ""
                patches = review.get("patches")
                if isinstance(patches, dict) and patches:
                    valid_patches = {}
                    for k, v in patches.items():
                        if not isinstance(v, dict):
                            continue
                        v.setdefault("name", k)
                        v["columns"] = _parse_column_strings(v.get("columns"))
                        if not isinstance(v.get("indexes"), list):
                            v["indexes"] = [str(v["indexes"])] if v.get("indexes") else []
                        valid_patches[k] = v
                    if valid_patches:
                        logger.info("DBA manager 리뷰 — %d개 테이블 패치: %s", len(valid_patches), list(valid_patches.keys()))
                        by_name.update(valid_patches)
                else:
                    logger.info("DBA manager 리뷰 — 패치 없음, 원본 유지")
                new_relationships = review.get("relationships")
                if isinstance(new_relationships, list) and new_relationships:
                    relationships = new_relationships
            else:
                logger.warning("DBA manager 리뷰 파싱 최종 실패 — 원본 유지")
        except Exception as e:
            logger.warning("DBA manager 리뷰 실패 — 원본 유지: %s", e)

        return {"final_tables": list(by_name.values()), "relationships": relationships, "prd_issues": prd_issues}

    async def _call(self, user_prompt: str, max_tokens: int, system: str, enable_thinking: bool, temperature: float = _TEMPERATURE) -> str:
        for attempt in range(3):
            try:
                response = await self._client.chat.completions.create(
                    model=self._model,
                    temperature=temperature,
                    top_p=_TOP_P,
                    presence_penalty=_PRESENCE_PENALTY,
                    max_tokens=max_tokens,
                    frequency_penalty=0.3,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user_prompt},
                    ],
                    extra_body={"chat_template_kwargs": {"enable_thinking": enable_thinking}},
                )
                return response.choices[0].message.content or ""
            except (InternalServerError, APITimeoutError, APIConnectionError) as e:
                logger.warning("DBA API 일시 오류 (attempt %d): %s — 재시도", attempt + 1, e)
                if attempt < 2:
                    await asyncio.sleep(5 * (attempt + 1))
                else:
                    logger.error("DBA API 최종 실패")
                    return ""
        return ""
