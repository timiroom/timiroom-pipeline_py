"""문서 AI 편집 — EXAONE이 현재 문서를 읽고 섹션/항목 단위 수정안을 만든다.

이 라우터는 문서를 직접 고치지 않는다. 수정안과 그 diff만 반환하고,
실제 적용 여부는 프론트에서 사용자가 제안문과 diff를 보고 승인한 뒤에 결정한다.

지원 문서: PRD / 기능 명세서 / API 명세서 / ERD (아래 _PROFILES)

동작 방식 — 4단계 호출:
  1) 분류    : 요청이 '질문(chat)'인지 '수정(edit)'인지 판별하고, 수정이면 대상 섹션을 고른다.
  2) 범위    : 리스트 섹션이면 손댈 '항목'과 연산(edit/add/remove)까지 좁힌다.
  3) 재작성  : 고른 항목(또는 섹션)만 다시 쓴다. 나머지는 원본 객체를 그대로 재사용한다.
               - edit  : 고른 항목만 재작성해 제자리에 끼워 넣는다
               - remove: 모델을 부르지 않고 Python에서 해당 항목만 뺀다
               - add   : 새 항목만 만들게 하고 기존 목록 뒤에 이어붙인다
               셋 다 기존 항목 객체가 모델을 거치지 않으므로 변형될 수 없다.
  4) 제안문  : 확정된 변경 내용을 근거로 "무엇을 어떤 값으로 바꾸는지" 문장을 만든다.

한 번의 호출로 "JSON Patch 경로를 직접 만들어라"고 시키지 않는 이유는 EXAONE이
경로 표현식을 안정적으로 못 만들기 때문이다. 섹션/항목을 통째로 재작성시키면
출력이 원본과 같은 모양이라 검증(타입·필수키 일치)만으로 안전성을 확보할 수 있다.
"""
import asyncio
import difflib
import json
import logging
import re
import unicodedata
from dataclasses import dataclass, field

from fastapi import APIRouter, HTTPException
from openai import InternalServerError, APITimeoutError, APIConnectionError
from pydantic import BaseModel, Field

from common.api_response import ok
from phase2.json_utils import try_parse_json, has_suspicious_script, find_json_syntax_leak

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/document", tags=["document"])

# EXAONE 모델 카드 권장 샘플링 파라미터
# https://huggingface.co/LGAI-EXAONE/K-EXAONE-236B-A23B
_TEMPERATURE = 1.0
_TOP_P = 0.95
_PRESENCE_PENALTY = 0.0

# 한 번의 요청으로 고칠 수 있는 섹션 수 상한.
# 사용자가 "전체 다 고쳐줘"라고 해도 diff가 검토 불가능할 만큼 커지지 않게 막는다.
_MAX_TARGET_SECTIONS = 3

# 한 섹션에서 항목 단위로 동시에 재작성할 수 있는 항목 수 상한.
# 항목마다 EXAONE 호출이 하나씩 나가므로, 상한이 없으면 20개짜리 목록에서
# 20개 호출이 한꺼번에 뜬다(섹션 3개면 60개). 검토 가능한 diff 크기와도 맞물린다.
_MAX_TARGET_ITEMS = 5

# 분류 단계에 넣을 섹션 미리보기 길이 — 문서 전체를 넣으면 토큰이 폭발한다.
_PREVIEW_CHARS = 300

_MIN_REWRITE_TOKENS = 8000
_MAX_REWRITE_TOKENS = 16000

# 섹션 전체 재작성을 시도할 수 있는 원본 크기 상한(문자).
# 교체본은 원본보다 길어지는데 출력은 _MAX_REWRITE_TOKENS에 묶여 있어서,
# 이 선을 넘으면 어떤 예산을 잡아도 응답이 잘린다(finish_reason="length").
# 3번 다 잘릴 게 뻔한 호출을 던지느니 시작 전에 접는다.
_MAX_REWRITE_CHARS = 12000

# 요청 문서 크기 상한(직렬화 문자 수). 넘으면 프롬프트가 컨텍스트를 넘겨
# EXAONE이 400을 뱉는다 — 재시도 대상이 아니라 그대로 실패하므로 입구에서 막는다.
_MAX_DOCUMENT_CHARS = 60000

# 프롬프트에 실을 직전 대화 — 턴 수와 턴당 길이 모두 제한한다.
_MAX_HISTORY_TURNS = 6
_HISTORY_TURN_CHARS = 800

# 라벨 키를 지정하지 않은(또는 맞지 않는) 중첩 객체에 쓰는 렌더링 기본값
_GENERIC_LABEL_KEYS = ("name", "metric", "milestone", "title", "persona")


def _exaone_endpoint_id() -> str:
    from main import settings
    return settings.exaone_endpoint_id


# ── 문서 프로필 ──────────────────────────────────────────────────────

@dataclass(frozen=True)
class SectionSpec:
    """섹션 하나의 정의.

    label         사용자에게 보여줄 이름
    spec          EXAONE에게 알려줄 형식 규격
    label_keys    리스트 항목을 사람이 알아볼 수 있게 가리키는 키들 (여러 개면 공백으로 이어붙임)
    item_required 리스트 항목(dict)이 반드시 채우고 있어야 하는 키 — 잘림·오염 탐지용
    dict_required 객체 섹션이 반드시 채우고 있어야 하는 키
    """
    label: str
    spec: str
    label_keys: tuple[str, ...] = ()
    item_required: tuple[str, ...] = ()
    dict_required: tuple[str, ...] = ()


@dataclass(frozen=True)
class DocProfile:
    label: str
    sections: dict[str, SectionSpec] = field(default_factory=dict)


# prd_agent.SECTION_PROMPTS가 만들어내는 최상위 키와 1:1 대응
_PRD_PROFILE = DocProfile(
    label="PRD",
    sections={
        "projectOverview": SectionSpec(
            label="프로젝트 개요",
            spec="서비스 한 줄 개요 문자열. 누구를 위해 무엇을 어떻게 제공하는가. 50자 이상.",
        ),
        "background": SectionSpec(
            label="배경",
            spec="200자 이상 서술형 문단 문자열. 시장 현황 → 문제점 → 기회 순서. 수치에는 (출처: 기관명, YYYY년) 표기.",
        ),
        "goals": SectionSpec(
            label="목표",
            spec='문자열 배열. 각 항목은 "~을 통해 ~을 달성하여 ~에 기여한다" 형식, 60자 이상.',
        ),
        "kpi": SectionSpec(
            label="KPI",
            spec='객체 배열. 각 항목 키: metric, target("현재값 → 목표값"), basis(출처 "기관명, YYYY년"), measurementMethod, frequency.',
            label_keys=("metric",),
            item_required=("metric", "target", "basis"),
        ),
        "coreFeatures": SectionSpec(
            label="핵심 기능",
            spec="객체 배열. 각 항목 키: name, description(100자 이상), priority(P0/P1/P2), requirements(문자열 배열, 각 60자 이상).",
            label_keys=("name",),
            item_required=("name", "description"),
        ),
        "userPersonas": SectionSpec(
            label="사용자 페르소나",
            spec="객체 배열. 각 항목 키: name, age, job, techLevel, goal, painPoint, usagePattern.",
            label_keys=("name",),
            item_required=("name", "goal", "painPoint"),
        ),
        "mvpScope": SectionSpec(
            label="MVP 범위",
            spec="객체. 키: included(문자열 배열), excluded(문자열 배열), rationale(100자 이상 문자열).",
            dict_required=("rationale",),
        ),
        "techStack": SectionSpec(
            label="기술 스택",
            spec="객체. 키: backend, frontend, database, cache, messageQueue, cdn, monitoring, auth. 각 값은 기술명과 선택 이유.",
        ),
        "releaseSchedule": SectionSpec(
            label="릴리즈 일정",
            spec="객체 배열. 각 항목 키: date, milestone, description(100자 이상), deliverables(문자열 배열 3개 이상).",
            label_keys=("milestone",),
            item_required=("milestone", "description"),
        ),
    },
)

# 기능 목록은 객체 배열(coreFeatures 형태)일 수도, 단순 문자열 배열일 수도 있다.
# FeaturesPanel이 두 형태를 모두 지원하므로 여기서도 강제하지 않는다.
_FEATURES_PROFILE = DocProfile(
    label="기능 명세서",
    sections={
        "features": SectionSpec(
            label="기능 목록",
            spec=(
                "객체 배열. 각 항목 키: name(기능명), "
                'description(100자 이상, "사용자가 ~할 때 시스템이 ~한다" 구조), '
                "priority(P0/P1/P2 중 하나), requirements(문자열 배열). "
                "단, 원본이 단순 문자열 배열이면 같은 형태(기능명 문자열 배열)를 유지하세요."
            ),
            label_keys=("name",),
            item_required=("name", "description"),
        ),
    },
)

_API_PROFILE = DocProfile(
    label="API 명세서",
    sections={
        "endpoints": SectionSpec(
            label="엔드포인트",
            spec=(
                "객체 배열. 각 항목 키: method(GET/POST/PUT/DELETE/PATCH 중 하나), "
                "path(/api/v1/영문-리소스명, 한글 금지), description(이 API가 하는 일 한 줄), "
                "authRequired(true 또는 false), "
                "parameters(객체 배열: in은 query 또는 path, name, type, required, description), "
                "requestBody(본문이 없으면 \"없음\"), successResponse(반환 필드들), "
                'errorCodes("401 — 인증 실패, 404 — 리소스 없음" 형식으로 2개 이상). '
                "path에 {id} 같은 자리표시자가 있으면 parameters에 in=\"path\"로 반드시 포함하세요."
            ),
            label_keys=("method", "path"),
            item_required=("method", "path", "description"),
        ),
    },
)

_ERD_PROFILE = DocProfile(
    label="ERD 명세서",
    sections={
        "tables": SectionSpec(
            label="테이블",
            spec=(
                "객체 배열. 각 항목 키: name(영문 소문자 복수형), description, "
                "columns(객체 배열: name, type, constraints), indexes(문자열 배열). "
                "constraints는 PRIMARY_KEY / FOREIGN_KEY / NOT_NULL / UNIQUE / AUTO_INCREMENT 등을 "
                "콤마로 구분한 문자열입니다. 타입은 PostgreSQL 기준으로 작성하세요."
            ),
            label_keys=("name",),
            item_required=("name", "columns"),
        ),
        "relationships": SectionSpec(
            label="관계",
            spec='문자열 배열. 각 항목은 "테이블A (1:N) 테이블B" 형식. 관계 종류는 1:1, 1:N, N:M 중 하나.',
        ),
    },
)

# ── ERD 섹션 간 정합 ─────────────────────────────────────────────────
# tables와 relationships는 별개 섹션이지만 서로를 참조한다. 테이블 이름을 바꾸거나
# 테이블을 지우면 관계 문자열이 옛 이름을 가리킨 채 남는다(실측: settlement_records를
# 지웠는데 "spaces (1:N) settlement_records" 2개가 그대로 남았다).
#
# 모델에게 "관계도 같이 고쳐라"고 시키지 않는다 — 이건 문자열 치환이라 Python이
# 정확하고, LLM에 맡기면 멀쩡한 관계까지 다시 쓰다 흘린다.

_RELATION_RE = re.compile(r"^(.+?)\s*\((1:1|1:N|N:1|N:M)\)\s*(.+?)$")


def _table_renames(before: list, after: list) -> tuple[dict[str, str], set[str]]:
    """(바뀐 이름 매핑, 사라진 테이블 이름).

    이름 매핑은 길이가 같을 때만 자리로 짝지어 뽑는다. 항목이 추가·삭제되면
    자리 대응이 깨져서 엉뚱한 테이블을 '이름이 바뀐 것'으로 오인하기 때문이다.
    """
    def names(tables):
        return [t.get("name") for t in tables if isinstance(t, dict) and t.get("name")]

    renames: dict[str, str] = {}
    if len(before) == len(after):
        for b, a in zip(before, after):
            if not (isinstance(b, dict) and isinstance(a, dict)):
                continue
            old, new = b.get("name"), a.get("name")
            if old and new and old != new:
                renames[old] = new

    gone = set(names(before)) - set(names(after)) - set(renames)
    return renames, gone


def _singular(name: str) -> str:
    """FK 컬럼 접두사를 만들기 위한 단수형 (dba_agent._table_name_variants와 같은 규칙)."""
    if name.endswith("ies"):
        return name[:-3] + "y"
    if name.endswith("s") and not name.endswith("ss"):
        return name[:-1]
    return name


def _identifier_renames(renames: dict[str, str]) -> list[tuple[str, str]]:
    """테이블 개명 → 문서 안에서 함께 바뀌어야 할 식별자 쌍.

    바꾸는 것은 딱 세 가지뿐이다. `spaces_id` / `space_id` / `spaces`.
    단수형(`space`)만 따로 바꾸지 않는 이유가 중요하다 — `space_schedules`는
    이름이 겹칠 뿐 전혀 다른 테이블인데, 단수형을 치환 대상에 넣으면 그것까지
    `venue_schedules`로 망가진다. 마찬가지로 `is_space_owner`는 FK가 아니라
    그냥 이름에 space가 든 컬럼이라 건드리면 안 된다.

    긴 것부터 적용해야 `space_id`를 바꾸기 전에 `spaces`가 먼저 걸리지 않는다.
    """
    pairs: list[tuple[str, str]] = []
    for old, new in renames.items():
        pairs.append((f"{old}_id", f"{new}_id"))
        old_s, new_s = _singular(old), _singular(new)
        if old_s != old:
            pairs.append((f"{old_s}_id", f"{new_s}_id"))
        pairs.append((old, new))
    return sorted(set(pairs), key=lambda p: -len(p[0]))


def _token_sub(text: str, pairs: list[tuple[str, str]]) -> str:
    """식별자 경계에서만 치환한다. `_`는 경계로 본다 —
    `idx_reservations_space_id` 의 뒤쪽 `space_id`는 바꾸되,
    `space_schedules` 안의 `space`는 (그런 쌍이 없으므로) 건드리지 않는다."""
    if not isinstance(text, str):
        return text
    for old, new in pairs:
        text = re.sub(rf"(?<![A-Za-z0-9]){re.escape(old)}(?![A-Za-z0-9])", new, text)
    return text


def _sync_fk_references(tables: list, renames: dict[str, str]) -> tuple[list, list[str]]:
    """테이블 개명을 FK 컬럼명·제약·인덱스 문자열에 반영한다. (새 목록, 바뀐 내역).

    원본 dict를 제자리에서 고치지 않는다 — 항목 단위 수정 경로는 손대지 않은 테이블의
    원본 객체를 그대로 재사용하므로, 제자리 수정은 before까지 같이 바꿔 diff를 지워버린다.
    """
    if not isinstance(tables, list) or not renames:
        return tables, []

    pairs = _identifier_renames(renames)
    fk_forms = {}
    for old, new in renames.items():
        fk_forms[f"{old}_id"] = f"{new}_id"
        old_s, new_s = _singular(old), _singular(new)
        if old_s != old:
            fk_forms[f"{old_s}_id"] = f"{new_s}_id"

    changes: list[str] = []
    out = []
    for table in tables:
        if not isinstance(table, dict):
            out.append(table)
            continue
        name = table.get("name")
        new_table = dict(table)

        columns = []
        for col in table.get("columns") or []:
            if not isinstance(col, dict):
                columns.append(col)
                continue
            new_col = dict(col)
            # 컬럼명은 FK 형태와 '정확히' 같을 때만 바꾼다
            if col.get("name") in fk_forms:
                new_col["name"] = fk_forms[col["name"]]
                changes.append(f"{name}.{col['name']} → {new_col['name']}")
            if isinstance(col.get("constraints"), str):
                fixed = _token_sub(col["constraints"], pairs)
                if fixed != col["constraints"]:
                    new_col["constraints"] = fixed
                    changes.append(f"{name}.{new_col['name']} 제약")
            columns.append(new_col)
        if columns:
            new_table["columns"] = columns

        indexes = table.get("indexes")
        if isinstance(indexes, list):
            fixed_idx = [_token_sub(i, pairs) for i in indexes]
            if fixed_idx != indexes:
                new_table["indexes"] = fixed_idx
                changes.append(f"{name} 인덱스")

        out.append(new_table)
    return out, changes


def _sync_relationships(relationships, renames: dict[str, str], gone: set[str]) -> list:
    """관계 문자열의 테이블 이름을 갱신하고, 사라진 테이블을 가리키는 관계는 버린다."""
    if not isinstance(relationships, list):
        return relationships

    synced = []
    for rel in relationships:
        if not isinstance(rel, str):
            synced.append(rel)
            continue
        m = _RELATION_RE.match(rel.strip())
        if not m:
            synced.append(rel)
            continue

        left, kind, right = m.group(1).strip(), m.group(2), m.group(3).strip()
        if left in gone or right in gone:
            continue  # 참조 대상이 문서에서 사라졌다 — 관계도 성립하지 않는다
        synced.append(f"{renames.get(left, left)} ({kind}) {renames.get(right, right)}")
    return synced


_PROFILES: dict[str, DocProfile] = {
    "prd": _PRD_PROFILE,
    "features": _FEATURES_PROFILE,
    "api": _API_PROFILE,
    "erd": _ERD_PROFILE,
}


# ── 섹션 값 → 사람이 읽는 줄 목록 ────────────────────────────────────
# diff는 raw JSON이 아니라 이 렌더링 결과로 만든다. 사용자가 사이드바에서
# 검토하는 대상이므로 중괄호·따옴표가 아니라 문서처럼 보여야 한다.

def _dict_label(item: dict, label_keys: tuple[str, ...]) -> tuple[str, tuple[str, ...]]:
    """항목을 가리키는 제목과, 그 제목에 쓰인 키들을 반환.
    섹션이 지정한 키를 먼저 보고, 맞는 게 없으면 일반 키로 폴백한다
    (endpoints 안의 parameters처럼 중첩 구조에서 필요)."""
    for keys in (label_keys, _GENERIC_LABEL_KEYS):
        if not keys:
            continue
        used, parts = [], []
        for k in keys:
            v = item.get(k)
            if isinstance(v, str) and v.strip():
                parts.append(v.strip())
                used.append(k)
        if parts:
            return " ".join(parts), tuple(used)
    return "", ()


def _value_to_lines(value, label_keys: tuple[str, ...] = (), indent: int = 0) -> list[str]:
    pad = "  " * indent
    if value is None:
        return []
    if isinstance(value, str):
        return [pad + line for line in value.split("\n") if line.strip()] or [pad + value]
    if isinstance(value, bool):
        return [f"{pad}{'true' if value else 'false'}"]
    if isinstance(value, (int, float)):
        return [f"{pad}{value}"]
    if isinstance(value, list):
        lines: list[str] = []
        for item in value:
            if isinstance(item, dict):
                head, used = _dict_label(item, label_keys)
                lines.append(f"{pad}▸ {head}" if head else f"{pad}▸")
                rest = {k: v for k, v in item.items() if k not in used}
                lines.extend(_value_to_lines(rest, label_keys, indent + 1))
            else:
                lines.extend(f"{pad}- {ln.strip()}" for ln in _value_to_lines(item))
        return lines
    if isinstance(value, dict):
        lines = []
        for key, val in value.items():
            if isinstance(val, (dict, list)):
                lines.append(f"{pad}{key}:")
                lines.extend(_value_to_lines(val, label_keys, indent + 1))
            else:
                lines.extend(f"{pad}{key}: {ln.strip()}" for ln in _value_to_lines(val))
        return lines
    return [f"{pad}{value}"]


def _section_text(value, label_keys: tuple[str, ...] = ()) -> str:
    return "\n".join(_value_to_lines(value, label_keys))


def _build_diff(before, after, label_keys: tuple[str, ...] = ()) -> list[dict]:
    """섹션 값 두 개를 줄 단위로 비교해 프론트가 바로 색칠할 수 있는 형태로 반환.

    반환: [{"type": "same"|"add"|"del", "text": "..."}]
    """
    before_lines = _section_text(before, label_keys).split("\n") if before is not None else []
    after_lines = _section_text(after, label_keys).split("\n") if after is not None else []
    before_lines = [ln for ln in before_lines if ln.strip()]
    after_lines = [ln for ln in after_lines if ln.strip()]

    out: list[dict] = []
    matcher = difflib.SequenceMatcher(a=before_lines, b=after_lines, autojunk=False)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            out.extend({"type": "same", "text": ln} for ln in before_lines[i1:i2])
        else:
            out.extend({"type": "del", "text": ln} for ln in before_lines[i1:i2])
            out.extend({"type": "add", "text": ln} for ln in after_lines[j1:j2])
    return out


def _document_outline(profile: DocProfile, document: dict) -> str:
    """분류 단계용 문서 개요 — 섹션 키·라벨과 앞부분 미리보기만."""
    lines = []
    for key, sec in profile.sections.items():
        value = document.get(key)
        if value is None or (isinstance(value, (list, dict, str)) and not value):
            lines.append(f"- {key} ({sec.label}): (비어 있음)")
            continue
        preview = _section_text(value, sec.label_keys).replace("\n", " / ")
        if len(preview) > _PREVIEW_CHARS:
            preview = preview[:_PREVIEW_CHARS] + "…"
        count = f"[{len(value)}개] " if isinstance(value, list) else ""
        lines.append(f"- {key} ({sec.label}): {count}{preview}")
    return "\n".join(lines)


# ── 값 검증 ──────────────────────────────────────────────────────────

def _filled(value) -> bool:
    """값이 실제 내용을 담고 있는지 (빈 문자열·빈 배열·None 배제)."""
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, dict)):
        return bool(value)
    return value is not None


def _clean_section(sec: SectionSpec, value):
    """필수 키가 빈 '껍데기' 항목이 몇 개인지 세고, 걸러낸 값을 함께 돌려준다.

    ⚠️ 섹션 단위 흐름에서는 이 함수를 **개수 세기 용도로만** 쓴다. 걸러낸 값을 그대로
    채택하면 안 된다 — 설명이 비어 있는 기존 엔드포인트처럼 '느슨하지만 멀쩡한' 항목이
    사용자가 요청하지도 않았는데 문서에서 사라진다(실측: API 엔드포인트 5개 중 3개 소실).
    항목 하나를 새로 만들어낸 _rewrite_item에서만 걸러낸 값을 쓴다.

    반환: (걸러낸 값, 껍데기 개수)
    """
    if isinstance(value, list):
        kept = []
        for item in value:
            # 필수 키 검사는 dict 항목에만 적용한다 — 기능 목록처럼 문자열 배열을
            # 그대로 유지해야 하는 섹션이 통째로 걸러지면 안 된다
            if isinstance(item, dict) and sec.item_required:
                if all(_filled(item.get(k)) for k in sec.item_required):
                    kept.append(item)
            elif _filled(item):
                kept.append(item)
        return kept, len(value) - len(kept)

    if isinstance(value, dict):
        missing = [k for k in sec.dict_required if not _filled(value.get(k))]
        # 객체는 항목 단위로 버릴 수 없으므로 필수 키가 비면 통째로 실패 처리
        return value, len(missing) + (0 if value else 1)

    if isinstance(value, str):
        return value.strip(), (0 if value.strip() else 1)

    return value, (0 if _filled(value) else 1)


def _norm_label(text) -> str:
    """항목 라벨 비교용 정규화 — 유니코드 정규화 + 공백 축약 + 양끝 구두점 제거."""
    s = unicodedata.normalize("NFKC", str(text)).strip().lower()
    return re.sub(r"\s+", " ", s).strip(" .!?~-…。")


def _item_label_of(sec: SectionSpec, item) -> str:
    if isinstance(item, dict):
        label, _ = _dict_label(item, sec.label_keys)
        return label
    return str(item)


def _drop_duplicate_items(sec: SectionSpec, baseline, items):
    """같은 항목이 두 번 들어간 것을 정리한다.

    전체 재작성 경로에는 "원본 항목을 요청 없이 삭제하지 마세요"라는 지시가 있는데,
    EXAONE이 이를 '원본을 남겨둔 채 수정본을 새로 덧붙여라'로 해석하는 실패 모드가 있다
    (실측: 기능 명세서에서 같은 기능이 원본 + 수정본으로 2개가 됨).

    라벨이 정확히 같은 항목이 둘 이상이면 원본과 달라진(=수정된) 쪽을 남긴다.
    비교를 '정확히 일치'로 한정한 이유는 토큰 겹침으로 판정하면
    "GET /api/v1/ingredients"와 "POST /api/v1/ingredients"처럼 서로 다른 항목을
    같은 것으로 오인해 멀쩡한 엔드포인트를 지워버리기 때문이다.
    """
    if not isinstance(items, list):
        return items, 0

    baseline_list = baseline if isinstance(baseline, list) else []
    kept: list = []
    kept_labels: list[str] = []
    dropped = 0

    for item in items:
        label = _norm_label(_item_label_of(sec, item))
        if not label:
            kept.append(item)
            kept_labels.append("")
            continue

        try:
            pos = kept_labels.index(label)
        except ValueError:
            kept.append(item)
            kept_labels.append(label)
            continue

        # 중복 발견 — 원본 그대로인 쪽을 버리고 수정된 쪽을 남긴다
        existing_is_original = any(kept[pos] == b for b in baseline_list)
        item_is_original = any(item == b for b in baseline_list)
        if existing_is_original and not item_is_original:
            kept[pos] = item
        dropped += 1

    return kept, dropped


def _revert_colliding_edits(
    sec: SectionSpec, key: str, baseline: list, merged: list, touched: list[int],
) -> tuple[list, list[int]]:
    """수정 결과가 '다른 기존 항목'과 같은 라벨이 되면 그 수정만 원본으로 되돌린다.

    실측 두 가지:
      - "spaces-2가 spaces랑 중복이에요" → 모델이 spaces-2를 spaces로 고쳐 써서 2개가 됨
      - "검색 API에 필터 넣어주세요" → 모델이 엉뚱하게 POST /api/v1/spaces 의 method를
        GET으로 바꿔 GET /api/v1/spaces 와 같은 항목이 됨

    중복된 쪽을 지우는 방식으로 처리하면 공간 등록 API가 문서에서 사라진다 —
    사용자가 요청하지 않은 삭제다. 수정을 되돌리면 최악이라도 '그 항목이 안 고쳐졌을 뿐'이
    된다. 진짜 삭제 요청은 operation="remove" 경로가 따로 처리한다.

    반환: (정리된 목록, 실제로 남은 변경 인덱스)
    """
    base_labels = {_norm_label(_item_label_of(sec, item)) for item in baseline}
    reverted = []
    for i in touched:
        own = _norm_label(_item_label_of(sec, baseline[i]))
        new = _norm_label(_item_label_of(sec, merged[i]))
        # 자기 라벨을 유지했거나 완전히 새로운 라벨이면 문제없다.
        # 다른 기존 항목이 이미 쓰고 있던 라벨을 가져간 경우만 되돌린다.
        if new and new != own and new in base_labels:
            merged[i] = baseline[i]
            reverted.append(i)

    if reverted:
        logger.warning(
            "섹션 %s — 항목 %s 의 수정이 다른 기존 항목과 같은 이름이 되어 원본으로 되돌림",
            key, reverted,
        )
    return merged, [i for i in touched if merged[i] != baseline[i]]


def _same_shape(before, after) -> bool:
    """원본과 교체본의 최상위 자료 구조가 같은지. 문자열/숫자는 서로 허용."""
    if isinstance(before, list):
        return isinstance(after, list)
    if isinstance(before, dict):
        return isinstance(after, dict)
    return not isinstance(after, (list, dict))


def _entry_key(entry) -> str:
    """중첩 리스트 항목(컬럼·파라미터 등)을 가리키는 이름."""
    if isinstance(entry, dict):
        label, _ = _dict_label(entry, ())
        return _norm_label(label)
    return _norm_label(entry) if isinstance(entry, str) else ""


def _reconcile_with_original(before, after, known_keys: set | None = None) -> tuple:
    """교체본을 원본의 키 구성에 맞춘다. (정리본, 무엇을 손봤는지) 반환.

    두 가지를 한 번에 한다.
      - 흘린 필드는 원본 값으로 되살린다
      - 섹션 스키마에 없는 키는 떨어낸다 (실측: EXAONE이 엔드포인트에 `}{.;/ / / /`
        라는 키를 값 `'[]: ,,, ,,,'` 과 함께 만들어 붙였다. 값만 검사하는 오염 탐지기는
        이런 쓰레기 '키'를 못 본다.)

    known_keys는 같은 섹션의 **다른 항목들이 쓰고 있는 키**의 합집합이다. 이 항목에
    없던 키라도 형제 항목에 있으면 정당한 필드다 — 원본 users에 indexes가 없어도
    다른 테이블에 있으면 "인덱스 추가해줘"가 성립한다. 중첩 리스트 안쪽에서도 같은
    원리를 쓴다 — 그 리스트의 원본 항목들이 쓰는 키가 그 자리의 스키마다.

    K-EXAONE은 컬럼이 15개쯤 되는 항목을 다시 쓸 때 매번 '다른' 필드를 한두 개씩
    흘린다(실측: 3회 재시도가 전부 서로 다른 필드를 빠뜨림). 거부하고 재생성해봐야
    같은 일이 반복되므로 결과가 없다 — 모델이 바꾼 값은 그대로 두고, 언급조차 없이
    사라진 필드만 원본에서 되돌리는 편이 정확하고 확실하다.

    중첩 리스트가 짧아진 경우도 되살린다. 처음에는 "컬럼 하나 빼줘"를 존중하려고
    길이가 다르면 손대지 않았는데, 실측에서 '컬럼 추가' 요청에 users 테이블이
    19개 → 3개로 잘려 나온 채 통과했다. 자리(index)가 아니라 **이름**으로 짝지어
    사라진 항목만 뒤에 되돌려 붙이므로, 순서가 바뀌거나 항목이 늘어난 경우에도 안전하다.
    """
    restored: list[str] = []
    top_allowed = (known_keys or set()) | (set(before) if isinstance(before, dict) else set())

    def walk(b, a, path: str, allowed: set | None):
        if isinstance(b, dict) and isinstance(a, dict):
            merged = {}
            for key, value in a.items():
                if key in b:
                    merged[key] = walk(b[key], value, f"{path}{key}.", None)
                elif allowed is not None and key not in allowed:
                    restored.append(f"-{path}{key}")  # 이 자리 스키마에 없는 키 — 떨어냄
                else:
                    merged[key] = value
            for key, value in b.items():
                if key not in merged:
                    merged[key] = value
                    restored.append(f"{path}{key}")
            return merged

        if isinstance(b, list) and isinstance(a, list):
            keys_after = {_entry_key(x) for x in a}
            merged = list(a)
            for entry in b:
                name = _entry_key(entry)
                if name and name not in keys_after:
                    merged.append(entry)
                    restored.append(f"{path.rstrip('.')}[{name}]")
            # 같은 리스트의 원본 항목들이 쓰는 키가 이 자리의 스키마다.
            # 실측: parameters 항목 하나에 `descriptiontion:string, "required": ...}, {`
            # 라는 키가 통째로 생겨났다 — 최상위만 검사하면 이런 건 빠져나간다.
            entry_keys = {k for x in b if isinstance(x, dict) for k in x}
            by_name = {_entry_key(x): x for x in b if _entry_key(x)}
            return [
                walk(by_name[_entry_key(x)], x, path, entry_keys) if _entry_key(x) in by_name else x
                for x in merged
            ]

        return a

    return walk(before, after, "", top_allowed), restored


def _token_budget(current, attempt: int) -> int:
    """교체본을 담을 max_tokens. 원본 크기에 비례해 잡고, 재시도마다 늘린다.

    한국어 JSON은 대략 문자당 1~1.5토큰이지만, 교체본은 원본보다 길어지는 데다
    EXAONE이 장황해지는 경향이 있어 4배로 잡는다. 하한을 8000으로 둔 것은
    항목이 적은 섹션이라도 여유 없이 자르면 뒤쪽 항목이 빈 껍데기로 나오기 때문이다.
    """
    approx_chars = len(json.dumps(current, ensure_ascii=False)) if current is not None else 0
    budget = int(approx_chars * 4 * (1 + 0.5 * attempt))
    return max(_MIN_REWRITE_TOKENS, min(_MAX_REWRITE_TOKENS, budget))


# ── EXAONE 호출 ──────────────────────────────────────────────────────

def _messages(system: str, prompt: str, history: list[dict] | None = None) -> list[dict]:
    """system + 직전 대화 + 이번 프롬프트.

    대화를 분류 단계에만 넣으면 "아니 그거 말고 더 짧게" 같은 후속 요청에서
    재작성 단계가 무엇을 가리키는지 몰라 헛돈다. 다만 문서 본문이 이미 크므로
    턴 수와 턴당 길이를 모두 잘라서 싣는다.
    """
    messages = [{"role": "system", "content": system}]
    for turn in (history or [])[-_MAX_HISTORY_TURNS:]:
        content = turn["content"]
        if len(content) > _HISTORY_TURN_CHARS:
            content = content[:_HISTORY_TURN_CHARS] + "…"
        messages.append({"role": turn["role"], "content": content})
    messages.append({"role": "user", "content": prompt})
    return messages


async def _call_exaone(client, messages: list[dict], max_tokens: int) -> tuple[str, str]:
    """EXAONE 호출 (재시도 3회). (본문, finish_reason)을 함께 돌려준다.

    finish_reason이 "length"면 토큰 한도에서 잘린 응답이다. 이걸 무시하면
    try_parse_json이 잘린 JSON을 '부분 복구'해서 빈 껍데기 항목이 섞인 결과를
    성공처럼 돌려주므로(실측: kpi 항목이 metric만 있고 나머지가 빈 값), 호출부에서
    반드시 확인해야 한다.
    """
    for attempt in range(3):
        try:
            resp = await client.chat.completions.create(
                model=_exaone_endpoint_id(),
                max_tokens=max_tokens,
                temperature=_TEMPERATURE,
                top_p=_TOP_P,
                presence_penalty=_PRESENCE_PENALTY,
                frequency_penalty=0.3,
                response_format={"type": "json_object"},
                messages=messages,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
            choice = resp.choices[0]
            return (choice.message.content or ""), (choice.finish_reason or "")
        except (InternalServerError, APITimeoutError, APIConnectionError) as e:
            logger.warning("EXAONE 문서편집 일시 오류 (attempt %d): %s — 재시도", attempt + 1, e)
            if attempt < 2:
                await asyncio.sleep(3 * (attempt + 1))
            else:
                raise
    return "", ""


# ── 1단계: 의도 분류 + 대상 섹션 선택 ────────────────────────────────

_CLASSIFY_SYSTEM = """당신은 소프트웨어 기획 문서를 관리하는 시니어 PM입니다.
사용자의 요청이 '문서 수정 요청'인지 '단순 질문/상담'인지 판별하고,
수정이라면 어느 섹션을 고쳐야 하는지 고릅니다.

JSON 하나만 출력하세요. 설명·마크다운 코드블록 금지."""

_CLASSIFY_PROMPT = """## 문서 종류: {doc_label}

## 현재 섹션 목록
{outline}

## 사용자 요청
{instruction}

## 판단 기준
- intent가 "edit"인 경우: 문서 내용을 바꾸거나, 추가하거나, 지워달라는 요청
- intent가 "chat"인 경우: 문서에 대한 질문, 의견, 설명 요청 (문서를 바꾸지 않음)
- targets에는 실제로 내용이 바뀌어야 하는 섹션 키만 넣으세요. 최대 {max_targets}개.
- 반드시 위 '섹션 목록'에 있는 키만 사용하세요. 없는 키를 지어내지 마세요.
- 애매하면 가장 관련이 깊은 섹션 하나만 고르세요.

## 출력 형식
{{"intent": "edit", "targets": ["섹션키"], "reply": "무엇을 어떻게 고칠지 사용자에게 알리는 1~2문장"}}
또는
{{"intent": "chat", "targets": [], "reply": "질문에 대한 답변"}}

reply는 항상 한국어 존댓말로 작성하세요."""


async def _classify(client, profile: DocProfile, document: dict, instruction: str, history: list[dict]) -> dict:
    prompt = _CLASSIFY_PROMPT.format(
        doc_label=profile.label,
        outline=_document_outline(profile, document),
        instruction=instruction,
        max_targets=_MAX_TARGET_SECTIONS,
    )
    messages = _messages(_CLASSIFY_SYSTEM, prompt, history)

    raw, finish_reason = await _call_exaone(client, messages, max_tokens=1500)
    node = try_parse_json(raw)
    if not isinstance(node, dict):
        logger.warning(
            "문서편집 분류 파싱 실패 (finish=%s) — chat으로 처리 | raw: %.300s",
            finish_reason, raw,
        )
        return {"intent": "chat", "targets": [], "reply": ""}

    intent = node.get("intent") if node.get("intent") in ("edit", "chat") else "chat"
    raw_targets = node.get("targets")
    targets = []
    if isinstance(raw_targets, list):
        for t in raw_targets:
            # EXAONE이 존재하지 않는 키를 지어내는 경우가 있어 화이트리스트로 거른다
            if isinstance(t, str) and t in profile.sections and t not in targets:
                targets.append(t)
            elif isinstance(t, str):
                logger.warning("문서편집 — 알 수 없는 섹션 키 무시: %r", t)
    targets = targets[:_MAX_TARGET_SECTIONS]

    reply = node.get("reply") if isinstance(node.get("reply"), str) else ""
    if intent == "edit" and not targets:
        # 섹션이 하나뿐인 문서(기능/API/ERD)에서는 굳이 고르지 못했더라도
        # 대상이 자명하므로 그 섹션으로 진행한다
        if len(profile.sections) == 1:
            targets = list(profile.sections)
            logger.info("문서편집 — 섹션이 하나뿐이라 %s로 자동 지정", targets)
        else:
            logger.warning("문서편집 — intent=edit이지만 유효 섹션 없음 — chat으로 강등")
            intent = "chat"
    return {"intent": intent, "targets": targets, "reply": reply.strip()}


# ── 2단계: 리스트 섹션에서 손댈 항목 고르기 ──────────────────────────
# 섹션을 통째로 재작성시키면 "예약 완료율의 목표만 고쳐줘"라고 해도 EXAONE이 KPI 7개를
# 전부 다시 쓰면서 엉뚱한 항목까지 바꿔놓는다(실측). 프롬프트로 "나머지는 유지하라"고 해도
# 지켜지지 않으므로, 손댈 항목을 먼저 고른 뒤 그 항목만 재작성하고 나머지는 원본 객체를
# 그대로 재사용한다 — 건드리지 않은 항목이 '바뀌지 않았기를 기대'하는 게 아니라 실제로 동일해진다.

_SCOPE_SYSTEM = """당신은 소프트웨어 기획 문서를 관리하는 시니어 PM입니다.
사용자의 수정 요청이 '특정 항목 몇 개'를 겨냥한 것인지, '목록 전체'를 다시 짜야 하는 것인지 판별합니다.

JSON 하나만 출력하세요. 설명·마크다운 코드블록 금지."""

_SCOPE_PROMPT = """## 섹션: {key} ({label})

## 현재 항목 목록
{items}

## 사용자 요청
{instruction}

## scope — 기존 목록에서 특정 항목을 지목할 수 있는가
- "item": 기존 항목을 지목한 요청. indices에 그 항목의 번호를 넣으세요.
  (내용 수정: "예약 완료율 목표가 이상해요", "3번 설명 더 자세히")
  (삭제: "리텐션 지표 빼줘", "users 테이블 삭제해줘")
- "list": 항목을 새로 만들어 달라거나 목록 전체를 다시 짜야 하는 요청. indices는 빈 배열.
  (예: "엔드포인트 3개 더 추가해줘", "전부 다시 써줘")

## operation — 목록의 길이가 어떻게 바뀌어야 하는가
- "remove": 항목을 빼달라는 요청일 때만. scope는 "item"이고, indices에 뺄 항목 번호를 넣으세요.
            ("~ 삭제해줘", "~ 빼줘", "A가 B랑 중복이니 정리해줘" — 중복 정리는 한쪽을 빼는 것입니다)
- "add"   : 항목을 새로 추가해달라는 요청일 때 (예: "~ 추가해줘", "3개 더 만들어줘"). scope는 "list".
- "edit"  : 그 외 전부. 기존 항목의 내용만 바뀌고 개수는 그대로여야 합니다.

## 출력 형식
{{"scope": "item", "indices": [3], "operation": "edit"}}
또는
{{"scope": "item", "indices": [2], "operation": "remove"}}
또는
{{"scope": "list", "indices": [], "operation": "add"}}

indices는 위 '현재 항목 목록'에 적힌 번호이며 1번부터 시작합니다. 최대 {max_items}개까지,
요청과 직접 관련된 항목만 고르세요.
확실하지 않으면 가장 관련이 깊은 항목 하나만 고르고, operation은 "edit"으로 두세요."""


def _item_labels(sec: SectionSpec, items: list) -> str:
    """모델에게 보여줄 항목 목록. 번호는 1부터 — 사용자가 "3번"이라고 말할 때의
    번호와 같아야 모델이 엉뚱한 항목을 지목하지 않는다. 내부 인덱스는 0부터이므로
    파싱할 때 1을 뺀다(_select_item_indices)."""
    lines = []
    for i, item in enumerate(items, start=1):
        if isinstance(item, dict):
            text, _ = _dict_label(item, sec.label_keys)
            text = text or "(제목 없음)"
        else:
            text = str(item)
        if len(text) > 120:
            text = text[:120] + "…"
        lines.append(f"{i}. {text}")
    return "\n".join(lines)


# 전체 재작성이 불가능한 큰 목록에서 모델이 "목록 전체" 판정을 내렸을 때 덧붙이는 지시.
# 이게 없으면 "검색 API에 필터 더 넣어주세요" 같은 명백한 단일 항목 수정도
# scope="list"로 떨어져 크기 한도에 막히고 아무 제안도 못 만든다(실측).
_SCOPE_FORCE_ITEM_NOTE = """

## 중요
이 목록은 너무 커서 전체를 다시 쓸 수 없습니다.
scope는 반드시 "item"으로 하고, 요청과 가장 관련이 깊은 항목의 번호를 최소 하나 고르세요.
정말 어떤 항목과도 관련이 없을 때만 빈 배열을 두세요."""


async def _select_item_indices(
    client, sec: SectionSpec, key: str, items: list, instruction: str,
    history: list[dict] | None = None, force_item: bool = False,
) -> tuple[list[int] | None, str]:
    """(손댈 항목의 0-based 인덱스, operation)을 반환. 목록 전체를 다시 짜야 하면 인덱스는 None.

    operation은 목록 길이가 줄어도 되는지 판정하는 데 쓴다 — "remove"가 아닌데
    항목이 사라지면 모델이 목록을 흘린 것으로 본다. 판정에 실패하면 가장 보수적인
    "edit"으로 두어 축소를 허용하지 않는다.

    force_item은 '전체 재작성이 불가능한 크기'라는 뜻이다. 이때 모델이 목록 전체를
    고르면 그 경로는 막다른 길이므로, 항목을 고르라고 못 박아 한 번 더 묻는다.
    """
    indices, operation = await _ask_scope(client, sec, key, items, instruction, history, note="")
    if force_item and indices is None and operation != "add":
        logger.info("섹션 %s — 전체 재작성 불가 크기인데 목록 전체 판정 — 항목 지정을 요구해 재질의", key)
        retry_indices, retry_operation = await _ask_scope(
            client, sec, key, items, instruction, history, note=_SCOPE_FORCE_ITEM_NOTE,
        )
        if retry_indices is not None:
            return retry_indices, retry_operation
    return indices, operation


async def _ask_scope(
    client, sec: SectionSpec, key: str, items: list, instruction: str,
    history: list[dict] | None, note: str,
) -> tuple[list[int] | None, str]:
    prompt = _SCOPE_PROMPT.format(
        key=key, label=sec.label,
        items=_item_labels(sec, items),
        instruction=instruction,
        max_items=_MAX_TARGET_ITEMS,
    ) + note
    messages = _messages(_SCOPE_SYSTEM, prompt, history)
    try:
        raw, _ = await _call_exaone(client, messages, max_tokens=500)
    except Exception as e:
        logger.warning("섹션 %s 범위 판정 호출 실패: %s — 전체 재작성으로 진행", key, e)
        return None, "edit"

    node = try_parse_json(raw)
    if not isinstance(node, dict):
        return None, "edit"

    operation = node.get("operation")
    if operation not in ("edit", "add", "remove"):
        operation = "edit"

    if node.get("scope") != "item":
        return None, operation

    raw_indices = node.get("indices")
    if not isinstance(raw_indices, list):
        return None, operation

    indices = []
    for n in raw_indices:
        # EXAONE이 문자열 "3"으로 주는 경우가 있어 정수 변환을 시도한다
        try:
            idx = int(n) - 1  # 프롬프트는 1-based, 내부는 0-based
        except (TypeError, ValueError):
            continue
        if 0 <= idx < len(items) and idx not in indices:
            indices.append(idx)

    if not indices:
        logger.warning("섹션 %s 범위=item이지만 유효 인덱스 없음 — 전체 재작성으로 진행", key)
        return None, operation

    if len(indices) > _MAX_TARGET_ITEMS:
        # 항목마다 호출이 하나씩 나가므로 상한을 넘으면 잘라낸다
        logger.warning(
            "섹션 %s 대상 항목이 %d개 — 상한 %d개로 자름",
            key, len(indices), _MAX_TARGET_ITEMS,
        )
        indices = indices[:_MAX_TARGET_ITEMS]
    return indices, operation


# ── 3단계: 재작성 ────────────────────────────────────────────────────

_ITEM_REWRITE_SYSTEM = """당신은 10년 경력의 시니어 PM 겸 소프트웨어 아키텍트입니다.
문서 목록의 항목 하나를, 사용자 요청에 맞게 고친 교체본으로 다시 작성합니다.

JSON 하나만 출력하세요. 설명·인사말·마크다운 코드블록 금지.

절대 규칙:
- 사용자가 지적한 필드만 고치세요. 요청과 무관한 필드는 원본 값을 글자 그대로 두세요.
- 원본과 완전히 동일한 키 구성으로 출력하세요. 키를 빼거나 새로 만들지 마세요.
- placeholder($name, {value}, "기능명" 등)를 값으로 쓰지 마세요."""

_ITEM_REWRITE_PROMPT = """## 수정할 항목 — {key}({label}) 목록의 {number}번

## 이 목록의 형식 규격
{spec}

## 현재 값
{current}

## 사용자 요청
{instruction}

## 출력 형식
{{"value": <수정된 항목 하나>}}

이 항목 하나만 다루세요. 목록 전체를 출력하지 마세요."""

_REWRITE_SYSTEM = """당신은 10년 경력의 시니어 PM 겸 소프트웨어 아키텍트입니다.
문서의 특정 섹션 하나를, 사용자 요청에 맞게 수정한 '완성된 교체본'으로 다시 작성합니다.

JSON 하나만 출력하세요. 설명·인사말·마크다운 코드블록 금지.

절대 규칙:
- 사용자가 요청한 부분만 바꾸고, 나머지 내용은 원본 그대로 유지하세요.
- 원본과 완전히 동일한 자료 구조(배열은 배열, 객체는 객체, 문자열은 문자열)로 출력하세요.
- 원본에 있던 항목을 요청 없이 삭제하지 마세요.
- ⛔ 기존 항목을 고칠 때는 **그 항목 자리에서 값을 바꾸세요.** 원본을 그대로 둔 채
  수정본을 새 항목으로 덧붙이지 마세요. 같은 이름의 항목이 두 번 나오면 안 됩니다.
  (배열 길이는 항목을 추가·삭제해 달라는 요청이 없는 한 원본과 같아야 합니다.)
- placeholder($name, {value}, "기능명" 등)를 값으로 쓰지 마세요."""

_REWRITE_PROMPT = """## 수정할 섹션: {key} ({label})

## 이 섹션의 형식 규격
{spec}

## 현재 값
{current}

## 사용자 요청
{instruction}

## 출력 형식
{{"value": <수정된 섹션 값 전체>}}

value는 이 섹션의 **전체 교체본**입니다. 바뀐 부분만 담지 말고,
수정된 내용을 반영한 완전한 값을 담으세요."""


async def _rewrite_item(
    client, sec: SectionSpec, key: str, index: int, item, instruction: str,
    history: list[dict] | None = None, known_keys: set | None = None,
):
    """리스트 항목 하나만 재작성. 실패하면 None. index는 0-based(표시는 1-based)."""
    prompt = _ITEM_REWRITE_PROMPT.format(
        key=key, label=sec.label, number=index + 1, spec=sec.spec,
        current=json.dumps(item, ensure_ascii=False, indent=2),
        instruction=instruction,
    )
    messages = _messages(_ITEM_REWRITE_SYSTEM, prompt, history)

    for attempt in range(3):
        raw, finish_reason = await _call_exaone(client, messages, max_tokens=2500)
        if finish_reason == "length":
            logger.warning("섹션 %s 항목 %d 응답 잘림 (attempt %d) — 재생성", key, index, attempt + 1)
            continue

        # strict=True — 잘리거나 깨진 JSON을 '부분 복구'해서 받으면 원본의 멀쩡한
        # 내용이 조각으로 교체된다. 편집 경로에서는 복구본 대신 재생성이 맞다.
        node = try_parse_json(raw, strict=True)
        if not isinstance(node, dict) or "value" not in node:
            logger.warning("섹션 %s 항목 %d 파싱 실패 (attempt %d)", key, index, attempt + 1)
            continue

        value = node["value"]
        if has_suspicious_script(value):
            logger.warning("섹션 %s 항목 %d 스크립트 오염 (attempt %d)", key, index, attempt + 1)
            continue
        leak = find_json_syntax_leak(value)
        if leak:
            logger.warning(
                "섹션 %s 항목 %d 값에 JSON 조각이 섞임 (attempt %d) — 재생성 | %.120s",
                key, index, attempt + 1, leak,
            )
            continue
        if not _same_shape(item, value):
            logger.warning("섹션 %s 항목 %d 타입 불일치 (attempt %d)", key, index, attempt + 1)
            continue

        value, fixed = _reconcile_with_original(item, value, known_keys)
        if fixed:
            logger.warning(
                "섹션 %s 항목 %d 키 구성을 원본에 맞춤 (%d건, - 는 떨어낸 키): %s",
                key, index, len(fixed), fixed[:6],
            )

        # 항목 하나짜리 리스트로 감싸 기존 필수 키 검증을 그대로 재사용
        cleaned, dropped = _clean_section(sec, [value])
        if dropped or not cleaned:
            logger.warning("섹션 %s 항목 %d 필수 필드 누락 (attempt %d) — 재생성", key, index, attempt + 1)
            continue
        return cleaned[0]

    return None


_ADD_ITEMS_SYSTEM = """당신은 10년 경력의 시니어 PM 겸 소프트웨어 아키텍트입니다.
기존 문서 목록에 **새로 덧붙일 항목만** 작성합니다.

JSON 하나만 출력하세요. 설명·인사말·마크다운 코드블록 금지.

절대 규칙:
- 새로 추가할 항목만 출력하세요. 기존 항목은 절대 다시 쓰지 마세요.
- 아래 '기존 항목 예시'와 완전히 동일한 키 구성으로 출력하세요.
- 기존 목록에 이미 있는 것과 같은 항목을 만들지 마세요.
- placeholder($name, {value}, "기능명" 등)를 값으로 쓰지 마세요."""

_ADD_ITEMS_PROMPT = """## 항목을 추가할 목록: {key} ({label})

## 이 목록의 형식 규격
{spec}

## 이미 있는 항목 ({count}개) — 중복 금지
{items}

## 기존 항목 예시 (이 키 구성을 그대로 따르세요)
{example}

## 사용자 요청
{instruction}

## 출력 형식
{{"value": [<새로 추가할 항목들>]}}

value에는 **새 항목만** 담으세요. 기존 항목을 포함하면 문서가 망가집니다.
개수를 지정하지 않았다면 요청에 맞는 만큼만, 많아도 {max_add}개 이내로 만드세요."""

# 추가 요청 한 번에 만들 수 있는 항목 수 상한 — diff 검토 가능성과 토큰 예산 둘 다.
_MAX_ADDED_ITEMS = 10


async def _add_items(
    client, sec: SectionSpec, key: str, baseline: list, instruction: str,
    history: list[dict] | None = None,
) -> list | None:
    """새 항목만 생성해서 돌려준다. 실패하면 None.

    목록 전체를 다시 쓰게 하지 않는 이유는 두 가지다.
    (1) 엔드포인트 20개짜리 목록은 교체본이 _MAX_REWRITE_TOKENS를 넘어 매번 잘린다.
    (2) 기존 항목을 모델 손에 다시 맡길 이유가 없다 — 프롬프트에는 라벨만 넣고
        기존 항목 객체는 Python에서 그대로 이어붙이므로 변형될 수 없다.
    """
    example = next((it for it in baseline if isinstance(it, dict)), baseline[0] if baseline else None)
    prompt = _ADD_ITEMS_PROMPT.format(
        key=key, label=sec.label, spec=sec.spec,
        count=len(baseline),
        items=_item_labels(sec, baseline),
        example=json.dumps(example, ensure_ascii=False, indent=2),
        instruction=instruction,
        max_add=_MAX_ADDED_ITEMS,
    )
    messages = _messages(_ADD_ITEMS_SYSTEM, prompt, history)

    for attempt in range(3):
        max_tokens = min(_MAX_REWRITE_TOKENS, 4000 * (attempt + 1))
        raw, finish_reason = await _call_exaone(client, messages, max_tokens=max_tokens)
        if finish_reason == "length":
            logger.warning("섹션 %s 항목 추가 응답 잘림 (attempt %d) — 예산 늘려 재생성", key, attempt + 1)
            continue

        node = try_parse_json(raw, strict=True)
        if not isinstance(node, dict) or "value" not in node:
            logger.warning("섹션 %s 항목 추가 파싱 실패 (attempt %d)", key, attempt + 1)
            continue

        value = node["value"]
        # 항목 하나만 달랑 준 경우도 받아준다
        if not isinstance(value, list):
            value = [value]
        if has_suspicious_script(value):
            logger.warning("섹션 %s 항목 추가 스크립트 오염 (attempt %d)", key, attempt + 1)
            continue
        leak = find_json_syntax_leak(value)
        if leak:
            logger.warning(
                "섹션 %s 새 항목 값에 JSON 조각이 섞임 (attempt %d) — 재생성 | %.120s",
                key, attempt + 1, leak,
            )
            continue

        # 기존 항목과 모양이 다른 것은 버린다 (문자열 목록에 dict가 섞이는 등)
        shaped = [v for v in value if example is None or _same_shape(example, v)]
        cleaned, dropped = _clean_section(sec, shaped)
        if dropped:
            logger.warning("섹션 %s 새 항목 중 %d개가 필수 필드 누락 — 제외", key, dropped)
        if not cleaned:
            logger.warning("섹션 %s 항목 추가가 쓸 만한 항목을 못 만듦 (attempt %d)", key, attempt + 1)
            continue

        # 기존 항목과 라벨이 겹치는 것은 '추가'가 아니라 중복이므로 버린다
        existing = {_norm_label(_item_label_of(sec, it)) for it in baseline}
        fresh = []
        for it in cleaned[:_MAX_ADDED_ITEMS]:
            label = _norm_label(_item_label_of(sec, it))
            if label and label in existing:
                logger.info("섹션 %s 새 항목 %r 은 이미 있음 — 제외", key, label)
                continue
            existing.add(label)
            fresh.append(it)

        if not fresh:
            logger.warning("섹션 %s 새 항목이 전부 기존과 중복 (attempt %d)", key, attempt + 1)
            continue
        return fresh

    return None


async def _rewrite_section(
    client, profile: DocProfile, key: str, current, instruction: str,
    history: list[dict] | None = None,
):
    sec = profile.sections[key]

    # 원본은 절대 손대지 않고 그대로 모델에 넘긴다.
    #
    # 예전에는 여기서 '손상 항목'을 걸러낸 뒤 그 축소된 목록으로 재작성했는데,
    # 판정 기준(필수 키가 다 채워졌는가)이 '느슨하지만 멀쩡한' 항목까지 손상으로 몰아
    # 사용자가 요청하지도 않은 삭제가 일어났다 — 설명 없는 엔드포인트가 통째로 사라졌다.
    # 이제는 개수만 세어 두고, 응답이 그보다 더 망가졌을 때만 재생성의 근거로 쓴다.
    baseline = current
    _, baseline_bad = _clean_section(sec, current) if current is not None else (None, 0)
    if baseline_bad:
        logger.info(
            "섹션 %s 원본에 필수 키가 빈 항목 %d개 — 그대로 유지하고 응답 판정 기준으로만 사용",
            key, baseline_bad,
        )

    # 리스트 섹션은 먼저 '어느 항목을 손댈지' 좁힌다. 항목 단위로 처리되면
    # 나머지 항목은 원본 객체를 그대로 재사용하므로 절대 변형되지 않는다.
    operation = "edit"
    if isinstance(baseline, list) and baseline:
        # 전체 재작성이 불가능한 크기인지 먼저 알아야 범위 판정을 그에 맞게 몰 수 있다
        oversized = len(json.dumps(baseline, ensure_ascii=False, indent=2)) > _MAX_REWRITE_CHARS
        indices, operation = await _select_item_indices(
            client, sec, key, baseline, instruction, history, force_item=oversized,
        )

        # 삭제는 모델에게 목록을 다시 쓰게 할 일이 아니다. 뺄 항목을 골랐으면
        # 나머지는 원본 객체 그대로 남기고 Python에서 지운다.
        if operation == "remove" and indices:
            drop = set(indices)
            merged = [item for i, item in enumerate(baseline) if i not in drop]
            logger.info("섹션 %s — 항목 %s 삭제 (%d → %d개)", key, sorted(drop), len(baseline), len(merged))
            return merged

        # 추가도 마찬가지 — 새 항목만 만들게 하고 기존 항목 뒤에 이어붙인다.
        if operation == "add":
            fresh = await _add_items(client, sec, key, baseline, instruction, history)
            if fresh:
                logger.info("섹션 %s — 새 항목 %d개 추가 (%d → %d개)",
                            key, len(fresh), len(baseline), len(baseline) + len(fresh))
                return list(baseline) + fresh
            logger.warning("섹션 %s — 항목 추가 실패 — 전체 재작성으로 폴백", key)
            indices = None

        if indices is not None:
            logger.info("섹션 %s — 항목 단위 수정 대상 %s", key, indices)
            # 같은 섹션의 다른 항목들이 쓰는 키 = 이 섹션의 정당한 필드 목록
            known_keys = {k for it in baseline if isinstance(it, dict) for k in it}
            results = await asyncio.gather(*[
                _rewrite_item(client, sec, key, i, baseline[i], instruction, history, known_keys)
                for i in indices
            ], return_exceptions=True)

            merged = list(baseline)
            changed = []
            for i, result in zip(indices, results):
                if isinstance(result, Exception):
                    logger.error("섹션 %s 항목 %d 재작성 오류: %s", key, i, result)
                    continue
                if result is None:
                    continue
                if result != baseline[i]:
                    merged[i] = result
                    changed.append(i)

            if changed:
                merged, changed = _revert_colliding_edits(sec, key, baseline, merged, changed)
            if changed:
                logger.info("섹션 %s — 항목 %s 수정 완료 (%d개 유지)", key, changed, len(merged))
                return merged
            logger.warning("섹션 %s — 항목 단위 수정이 아무것도 바꾸지 못함 — 전체 재작성으로 폴백", key)

    current_json = (
        json.dumps(baseline, ensure_ascii=False, indent=2) if baseline is not None else "(비어 있음)"
    )

    # 교체본이 출력 한도에 들어갈 수 없는 크기면 시작하지 않는다. 그대로 진행하면
    # 세 번 다 finish_reason="length"로 잘려 시간과 토큰만 쓰고 같은 자리에 온다.
    if len(current_json) > _MAX_REWRITE_CHARS:
        logger.warning(
            "섹션 %s 원본이 %d자로 전체 재작성 한도(%d자)를 넘음 — 재작성 포기 "
            "(항목 단위 수정/추가/삭제로는 처리 가능)",
            key, len(current_json), _MAX_REWRITE_CHARS,
        )
        return None

    prompt = _REWRITE_PROMPT.format(
        key=key, label=sec.label, spec=sec.spec,
        current=current_json,
        instruction=instruction,
    )
    messages = _messages(_REWRITE_SYSTEM, prompt, history)

    for attempt in range(3):
        max_tokens = _token_budget(baseline, attempt)
        raw, finish_reason = await _call_exaone(client, messages, max_tokens=max_tokens)

        # 잘린 응답은 파싱 이전에 버린다 — try_parse_json이 조각을 복구해
        # 빈 항목이 섞인 '성공'을 만들어내기 때문에 여기서 끊어야 한다
        if finish_reason == "length":
            logger.warning(
                "섹션 %s 응답이 토큰 한도(%d)에서 잘림 (attempt %d) — 예산 늘려 재생성",
                key, max_tokens, attempt + 1,
            )
            continue

        node = try_parse_json(raw, strict=True)
        if not isinstance(node, dict) or "value" not in node:
            logger.warning("섹션 %s 재작성 파싱 실패 (attempt %d, finish=%s)", key, attempt + 1, finish_reason)
            continue

        value = node["value"]
        if has_suspicious_script(value):
            logger.warning("섹션 %s 스크립트 오염 감지 (attempt %d) — 재생성", key, attempt + 1)
            continue
        leak = find_json_syntax_leak(value)
        if leak:
            logger.warning(
                "섹션 %s 값에 JSON 조각이 섞임 (attempt %d) — 재생성 | %.120s",
                key, attempt + 1, leak,
            )
            continue
        if baseline is not None and not _same_shape(baseline, value):
            logger.warning(
                "섹션 %s 타입 불일치 (원본 %s → 응답 %s, attempt %d) — 재생성",
                key, type(baseline).__name__, type(value).__name__, attempt + 1,
            )
            continue

        # 응답이 원본보다 '더' 망가졌을 때만 잘림으로 본다. 원본에 이미 있던
        # 느슨한 항목을 결함으로 세면 멀쩡한 문서에서 영영 재생성만 돌게 된다.
        _, out_bad = _clean_section(sec, value)
        if out_bad > baseline_bad:
            logger.warning(
                "섹션 %s 껍데기 항목이 %d → %d로 늘어남 (attempt %d) — 잘린 응답으로 보고 재생성",
                key, baseline_bad, out_bad, attempt + 1,
            )
            continue
        if not _filled(value):
            logger.warning("섹션 %s 교체본이 비어 있음 (attempt %d) — 재생성", key, attempt + 1)
            continue

        # 원본을 남겨둔 채 수정본을 덧붙인 경우를 정리한다. 재생성으로 돌리지 않는 이유는
        # 이게 모델의 습관적 실패라 다시 시켜도 반복되기 때문 — 여기서 고치는 편이 확실하다.
        result, duplicated = _drop_duplicate_items(sec, baseline, value)
        if duplicated:
            logger.warning(
                "섹션 %s — 중복 항목 %d개 정리 (원본에 수정본을 덧붙인 것으로 보임)",
                key, duplicated,
            )

        # 삭제를 요청하지 않았는데 항목이 줄었다면 모델이 목록을 흘린 것이다.
        # diff로 사용자에게 떠넘기지 않는 이유: 엔드포인트가 수십 개인 문서의 diff는
        # 사람이 검토할 수 있는 분량이 아니어서 그대로 승인되고 데이터가 사라진다.
        if (
            operation != "remove"
            and isinstance(baseline, list) and isinstance(result, list)
            and len(result) < len(baseline)
        ):
            logger.warning(
                "섹션 %s 항목이 %d → %d로 줄었는데 삭제 요청이 아님 (attempt %d) — 재생성",
                key, len(baseline), len(result), attempt + 1,
            )
            continue

        return result

    return None


# ── 4단계: 확정된 변경 내용을 사람 말로 옮긴 제안문 ──────────────────
# 사용자는 diff를 보기 전에 "무엇을 왜 바꾸는지"를 문장으로 먼저 읽고 동의 여부를 정한다.
# 이 시점에는 재작성이 이미 끝나 있으므로, 제안문은 추측이 아니라 실제로 적용될
# 변경만을 근거로 쓰인다 — 제안과 결과가 어긋날 수 없다.

_PROPOSAL_SYSTEM = """당신은 소프트웨어 기획 문서를 관리하는 시니어 PM입니다.
문서에 적용할 변경 내용을 사용자에게 설명하고 동의를 구하는 문장을 씁니다.

JSON 하나만 출력하세요. 설명·마크다운 코드블록 금지."""

_PROPOSAL_PROMPT = """## 문서 종류: {doc_label}

## 사용자 요청
{instruction}

## 확정된 변경 내용
{changes}

## 작성 규칙
- 2~4문장, 한국어 존댓말.
- **무엇을 어떤 값으로 바꾸는지 구체적으로** 밝히세요. "적절히 조정하겠습니다" 같은 모호한 표현 금지.
- 왜 그렇게 바꾸는지 근거를 한 문장 덧붙이세요.
- 위 '확정된 변경 내용'에 없는 것을 지어내지 마세요.
- 마지막은 "진행할까요?"로 끝내세요.

## 출력 형식
{{"proposal": "..."}}"""

# 제안문 프롬프트에 넣을 변경 줄 수 상한 — 넘으면 요약만 넣는다
_MAX_PROPOSAL_LINES = 40


def _changes_digest(edits: list[dict]) -> str:
    """diff에서 실제로 바뀐 줄만 뽑아 제안문 생성용 컨텍스트로 만든다."""
    parts = []
    total = 0
    for edit in edits:
        changed = [d for d in edit["diff"] if d["type"] != "same"]
        parts.append(f'[{edit["label"]}]')
        for d in changed:
            if total >= _MAX_PROPOSAL_LINES:
                parts.append("… (이하 생략)")
                break
            parts.append(("- " if d["type"] == "del" else "+ ") + d["text"])
            total += 1
        if total >= _MAX_PROPOSAL_LINES:
            break
    return "\n".join(parts)


def _fallback_proposal(edits: list[dict]) -> str:
    """제안문 생성이 실패했을 때 쓰는 결정론적 문구.
    LLM 호출 실패로 흐름 전체가 막히면 안 되므로 변경 규모만이라도 알린다."""
    summaries = []
    for edit in edits:
        added = sum(1 for d in edit["diff"] if d["type"] == "add")
        removed = sum(1 for d in edit["diff"] if d["type"] == "del")
        summaries.append(f'{edit["label"]}({removed}줄 → {added}줄)')
    return f"{', '.join(summaries)}을(를) 수정하려 합니다. 아래 변경 내용을 확인해 주세요. 진행할까요?"


async def _build_proposal(
    client, profile: DocProfile, instruction: str, edits: list[dict],
    history: list[dict] | None = None,
) -> str:
    prompt = _PROPOSAL_PROMPT.format(
        doc_label=profile.label, instruction=instruction, changes=_changes_digest(edits),
    )
    messages = _messages(_PROPOSAL_SYSTEM, prompt, history)
    try:
        raw, _ = await _call_exaone(client, messages, max_tokens=800)
        node = try_parse_json(raw)
        if isinstance(node, dict):
            proposal = node.get("proposal")
            if isinstance(proposal, str) and len(proposal.strip()) >= 10:
                return proposal.strip()
        logger.warning("제안문 파싱 실패 — 결정론적 문구로 대체")
    except Exception as e:
        logger.warning("제안문 생성 실패: %s — 결정론적 문구로 대체", e)
    return _fallback_proposal(edits)


def _append_relationship_sync(profile: DocProfile, document: dict, edits: list[dict]) -> None:
    """tables 수정이 관계에 미치는 영향을 같은 제안에 실어 준다 — edits를 제자리 수정.

    이름이 바뀌거나 사라진 테이블을 관계 문자열에 반영한다. 사용자가 승인 버튼을
    한 번 누르면 두 섹션이 함께 적용되므로, 문서가 어긋난 중간 상태로 남지 않는다.
    """
    table_edit = next((e for e in edits if e["section"] == "tables"), None)
    if table_edit is None:
        return

    before_tables = table_edit["before"] if isinstance(table_edit["before"], list) else []
    after_tables = table_edit["after"] if isinstance(table_edit["after"], list) else []
    renames, gone = _table_renames(before_tables, after_tables)
    if not renames and not gone:
        return

    # 테이블 이름이 바뀌었으면 그 테이블을 가리키는 FK 컬럼·제약·인덱스도 따라가야 한다.
    # 관계 문자열과 같은 이유로 모델이 아니라 Python이 한다 — 식별자 치환이라 정확하고,
    # 모델에 맡기면 손대지 않아야 할 테이블까지 다시 쓰다 흘린다.
    if renames:
        synced_tables, fk_changes = _sync_fk_references(after_tables, renames)
        if fk_changes:
            logger.info("ERD 정합 — FK 참조 %d곳 갱신: %s", len(fk_changes), fk_changes[:6])
            table_edit["after"] = synced_tables
            table_edit["diff"] = _build_diff(
                table_edit["before"], synced_tables, profile.sections["tables"].label_keys,
            )

    # relationships가 이번 요청으로 이미 수정됐다면 그 결과 위에 얹는다
    rel_edit = next((e for e in edits if e["section"] == "relationships"), None)
    base = rel_edit["after"] if rel_edit else document.get("relationships")
    synced = _sync_relationships(base, renames, gone)
    if synced == base:
        return

    logger.info(
        "ERD 정합 — 테이블 이름 변경 %s, 삭제 %s → 관계 %d개 → %d개",
        renames or "없음", sorted(gone) or "없음",
        len(base) if isinstance(base, list) else 0, len(synced),
    )

    sec = profile.sections["relationships"]
    if rel_edit:
        rel_edit["after"] = synced
        rel_edit["diff"] = _build_diff(rel_edit["before"], synced, sec.label_keys)
    else:
        original = document.get("relationships")
        edits.append({
            "section": "relationships",
            "label": sec.label,
            "before": original,
            "after": synced,
            "diff": _build_diff(original, synced, sec.label_keys),
        })


# ── 엔드포인트 ───────────────────────────────────────────────────────

class ChatTurn(BaseModel):
    role: str
    content: str


class DocumentEditRequest(BaseModel):
    document: dict = Field(default_factory=dict, description="현재 문서 JSON")
    instruction: str = Field(min_length=1, description="사용자의 수정 요청")
    history: list[ChatTurn] = Field(default_factory=list, description="직전 대화 (최근 것부터 소수)")


@router.post("/{doc_type}/edit")
async def edit_document(doc_type: str, req: DocumentEditRequest) -> dict:
    """문서 수정 제안 생성. 문서를 저장하지는 않고 제안문과 diff만 반환한다.

    doc_type: prd | features | api | erd
    """
    from main import exaone_client

    profile = _PROFILES.get(doc_type)
    if profile is None:
        raise HTTPException(
            status_code=404,
            detail=f"지원하지 않는 문서 종류입니다: {doc_type} (가능: {', '.join(_PROFILES)})",
        )

    # 문서가 컨텍스트에 들어갈 수 있는 크기인지 먼저 본다. 넘으면 EXAONE이 400을 뱉는데
    # 재시도 대상이 아니라 그대로 실패하므로, 여기서 이유가 분명한 응답으로 끊는다.
    doc_chars = len(json.dumps(req.document, ensure_ascii=False))
    if doc_chars > _MAX_DOCUMENT_CHARS:
        logger.warning("문서편집 입력이 너무 큼(%s): %d자 > %d자", doc_type, doc_chars, _MAX_DOCUMENT_CHARS)
        raise HTTPException(
            status_code=413,
            detail=f"문서가 너무 큽니다 ({doc_chars:,}자). {_MAX_DOCUMENT_CHARS:,}자 이하로 줄여주세요.",
        )

    # history는 최근 몇 턴까지만 — 문서 본문이 이미 크므로 대화까지 길어지면 컨텍스트가 넘친다
    history = [
        {"role": t.role, "content": t.content}
        for t in req.history[-_MAX_HISTORY_TURNS:]
        if t.role in ("user", "assistant") and t.content.strip()
    ]

    try:
        decision = await _classify(exaone_client, profile, req.document, req.instruction, history)
    except Exception as e:
        logger.error("문서편집 분류 실패(%s): %s", doc_type, e, exc_info=True)
        return ok({"intent": "chat", "reply": "잠시 후 다시 시도해 주세요.", "edits": []})

    if decision["intent"] == "chat":
        return ok({
            "intent": "chat",
            "reply": decision["reply"] or "무엇을 수정할지 조금 더 구체적으로 알려주시겠어요?",
            "edits": [],
        })

    # 대상 섹션을 병렬로 재작성 — 섹션끼리 의존이 없으므로 순차로 돌 이유가 없다
    results = await asyncio.gather(*[
        _rewrite_section(exaone_client, profile, key, req.document.get(key), req.instruction, history)
        for key in decision["targets"]
    ], return_exceptions=True)

    edits = []
    for key, result in zip(decision["targets"], results):
        if isinstance(result, Exception):
            logger.error("섹션 %s 재작성 오류: %s", key, result)
            continue
        if result is None:
            logger.warning("섹션 %s 재작성 실패 — 제안에서 제외", key)
            continue

        sec = profile.sections[key]
        before = req.document.get(key)
        diff = _build_diff(before, result, sec.label_keys)
        if not any(d["type"] != "same" for d in diff):
            logger.info("섹션 %s — 변경 없음, 제안에서 제외", key)
            continue

        edits.append({
            "section": key,
            "label": sec.label,
            "before": before,
            "after": result,
            "diff": diff,
        })

    if doc_type == "erd":
        _append_relationship_sync(profile, req.document, edits)

    if not edits:
        return ok({
            "intent": "chat",
            "reply": (decision["reply"] + "\n\n" if decision["reply"] else "")
                     + "다만 이번엔 실제로 바뀌는 내용을 만들지 못했어요. 요청을 조금 더 구체적으로 알려주시겠어요?",
            "edits": [],
        })

    # 재작성이 끝난 뒤에 제안문을 만든다 — 실제 적용될 변경만 근거로 삼으므로
    # "이렇게 바꾸겠습니다"라고 말한 내용과 최종 결과가 어긋나지 않는다
    proposal = await _build_proposal(exaone_client, profile, req.instruction, edits, history)

    logger.info(
        "문서편집 제안 생성 (%s) — %d개 섹션: %s",
        doc_type, len(edits), [e["section"] for e in edits],
    )
    return ok({
        "intent": "edit",
        "reply": proposal,
        "edits": edits,
    })
