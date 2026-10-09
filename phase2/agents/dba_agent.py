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
from phase2.feature_scope import backend_features
from phase2.feature_registry import (
    missing_db_contract_features,
    normalize_feature_registry,
    registry_text,
)
from phase2.json_utils import try_parse_json, has_suspicious_script
from phase2.llm_runtime import LlmRuntime
from phase2.state import PipelineState
from phase2.quality_rules import (
    matching_table_features, relevance_score, required_field_concepts,
    scoped_unique_columns, self_check_passed,
)
from phase2.agent_contract import feature_relation_kind, requires_auth, normalize_api_path, route_contract_table

logger = logging.getLogger(__name__)

# 생성 샘플링 파라미터
_TEMPERATURE = 1.0
_TOP_P = 0.95
_PRESENCE_PENALTY = 0.0
# 테이블 목록(plan) 생성은 개수·커버리지 일관성이 중요한 구조 생성 단계 —
# temp=1.0은 실행마다 편차를 키우므로 이 단계만 낮춰 완성도/재현성을 높인다.
_PLAN_TEMPERATURE = 0.4

WORKER_SYSTEM = "JSON만 출력하세요. 설명·인사말·마크다운 코드블록 금지. { 로 시작해서 } 로 끝납니다."

MANAGER_SYSTEM = """당신은 시니어 DBA 겸 DB manager입니다.
JSON만 출력하세요. 설명·인사말·마크다운 코드블록 금지. { 로 시작해서 } 로 끝납니다."""

TARGETED_REPAIR_PROMPT = """현재 DB 스키마에서 검증 피드백이 지적한 테이블 또는 관계만 수정하세요.
문제없는 테이블과 관계는 절대 다시 작성하거나 삭제하지 마세요.
patches에는 수정할 테이블 전체 객체를, relationships에는 관계 변경이 필요한 경우에만 최종 관계 목록을 넣으세요.
피드백에 [STRUCTURED_REPAIR_TARGETS]가 있으면 artifactKey와 featureId가 일치하는 테이블만 패치하세요.
FK blocker는 source table.column과 target table.column을 모두 명시하고, 다른 테이블의 컬럼은 변경하지 마세요.
프로젝트 도메인에 없는 필수 컬럼을 임의로 추가하지 마세요.
JSON만 출력하세요.

[검증 피드백]
{feedback}

[현재 DB 스키마]
{schema}

출력 형식:
{{"patches":{{"table_name":{{"description":"...","columns":["..."],"indexes":[]}}}},"relationships":null}}
"""


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
    elif name.endswith("sses"):
        # classes -> class. 단순히 마지막 s 하나만 제거하면 classe가 되어
        # class_id FK를 classes 테이블에 연결하지 못한다.
        variants.add(name[:-2])
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


def _registry_table_names(registry: list[dict] | None) -> list[str]:
    names: list[str] = []
    for item in registry or []:
        contract = item.get("dbContract") if isinstance(item, dict) else None
        for name in (contract or {}).get("tables", []) if isinstance(contract, dict) else []:
            if isinstance(name, dict):
                name = name.get("name") or name.get("table")
            name = str(name or "").strip()
            if name and name not in names:
                names.append(name)
    return names


def normalize_table_contract_names(
    tables: list[dict], relationships: list | None, registry: list[dict] | None,
) -> tuple[list[dict], list]:
    """Registry의 테이블명을 최종 스키마와 관계의 단일 기준으로 만든다.

    LLM이 ``memberships``와 ``organization_memberships``처럼 같은 엔티티의 별칭을
    섞어 쓰는 경우 계약에 있는 이름을 canonical name으로 선택하고, 컬럼 REFERENCES,
    인덱스, relationships까지 함께 갱신한다.
    """
    if not isinstance(tables, list):
        return tables, relationships or []
    canonical = _registry_table_names(registry)
    if not canonical:
        return tables, relationships or []
    canonical_set = set(canonical)
    existing = {str(table.get("name") or "").strip() for table in tables if isinstance(table, dict)}
    aliases: dict[str, str] = {}
    for table in tables:
        if not isinstance(table, dict):
            continue
        actual = str(table.get("name") or "").strip()
        if not actual or actual in canonical_set:
            continue
        candidates = [name for name in canonical if _table_name_variants(name) & _table_name_variants(actual)]
        if len(candidates) != 1:
            tail = actual.split("_")[-1]
            candidates = [name for name in canonical if name.split("_")[-1] == tail]
        # 계약 이름이 이미 스키마에 있거나 다른 테이블이 먼저 차지했다면 별칭이 아니라 별개 테이블이다.
        if len(candidates) == 1 and candidates[0] not in existing and candidates[0] not in aliases.values():
            aliases[actual] = candidates[0]

    if not aliases:
        return tables, relationships or []
    for table in tables:
        if not isinstance(table, dict):
            continue
        old = str(table.get("name") or "")
        new = aliases.get(old)
        if not new:
            continue
        logger.warning("DBA 테이블명 계약 정규화: %s → %s", old, new)
        table["name"] = new
        for col in table.get("columns") or []:
            if not isinstance(col, dict):
                continue
            constraints = str(col.get("constraints") or "")
            for source, target in aliases.items():
                constraints = re.sub(rf"(REFERENCES\s+){re.escape(source)}(\s*\()", rf"\1{target}\2", constraints, flags=re.IGNORECASE)
            col["constraints"] = constraints
        table["indexes"] = [
            _rename_in_text(str(entry), list(aliases.items())) if isinstance(entry, (str, int)) else entry
            for entry in (table.get("indexes") or [])
        ]

    normalized_relationships = []
    for relationship in relationships or []:
        text = str(relationship)
        for source, target in aliases.items():
            text = re.sub(rf"\b{re.escape(source)}\b", target, text)
        if text not in normalized_relationships:
            normalized_relationships.append(text)
    return tables, normalized_relationships


def annotate_table_feature_ids(tables: list[dict], registry: list[dict] | None) -> list[dict]:
    """DB 테이블에도 어떤 featureId 계약을 구현하는지 명시한다."""
    by_table: dict[str, list[str]] = {}
    for item in registry or []:
        if not isinstance(item, dict):
            continue
        fid = str(item.get("featureId") or item.get("id") or "").strip()
        contract = item.get("dbContract") or {}
        for table in contract.get("tables", []) if isinstance(contract, dict) else []:
            if isinstance(table, dict):
                table = table.get("name") or table.get("table")
            if fid and table:
                by_table.setdefault(str(table).strip(), []).append(fid)
    for table in tables or []:
        if not isinstance(table, dict) or not table.get("name"):
            continue
        ids = by_table.get(str(table["name"]).strip(), [])
        if ids:
            table["featureIds"] = sorted(set(ids))
    return tables


def _fk_target(col_name, name_lookup: dict[str, str]) -> str | None:
    """FK 패턴 컬럼명(`<참조테이블>_id`)에서 참조 대상 테이블명을 해석. FK가 아니면 None."""
    if not isinstance(col_name, str) or col_name == "id" or not col_name.endswith("_id"):
        return None
    stem = col_name[:-3]
    direct = name_lookup.get(stem)
    if direct:
        return direct
    # refresh token rotation is an internal self-reference.
    if stem == "replaced_by_token":
        # 토큰 회전은 refresh_tokens라는 이름을 쓰기도 하고 password_reset_tokens
        # 같은 토큰 테이블에 self-FK로 표현되기도 한다. 실제 lookup에 있는 토큰
        # 테이블을 우선 사용하고, 둘 이상이면 표준 refresh_tokens를 선택한다.
        if "refresh_tokens" in name_lookup:
            return name_lookup["refresh_tokens"]
        token_tables = [
            table_name for variant, table_name in name_lookup.items()
            if variant.endswith("_tokens")
        ]
        if len(token_tables) == 1:
            return token_tables[0]
    # created_by_id, recorded_by_id, actor_user_id 같은 감사/행위자 FK는 users를 가리킨다.
    if stem.endswith("_user") or stem.endswith("_by") or stem in {
        "actor", "owner", "creator", "uploader", "assignee", "reviewer", "signer",
        "renter", "borrower", "requester", "provider", "buyer", "seller", "owner_user", "created_by_user", "updated_by_user", "recorded_by_user",
        "processed_by_user", "changed_by_user", "actor_user", "generated_by_user",
        "user", "student", "member", "customer", "client", "operator",
        "teacher", "instructor", "participant", "applicant", "requester_user",
        "recipient", "recipient_user", "beneficiary", "subscriber",
        "manager", "admin", "administrator", "staff", "coordinator", "organizer", "supervisor",
        "approver", "volunteer", "host", "leader",
    }:
        if "users" in name_lookup:
            return name_lookup["users"]
    # space_id -> kitchen_spaces처럼 도메인 접두사가 붙은 테이블도 허용한다.
    for variant, table_name in name_lookup.items():
        if stem.endswith(f"_{variant}"):
            return table_name
    # job_id, record_id, guardian_id처럼 복합 테이블의 마지막 명사만 남은
    # FK도 전체 테이블 목록에서 유일하게 매칭되면 실제 테이블로 연결한다.
    tail_matches = {
        table_name
        for variant, table_name in name_lookup.items()
        if variant.rsplit("_", 1)[-1] == stem
        or variant.rsplit("_", 1)[-1].removesuffix("s") == stem.removesuffix("s")
    }
    if not tail_matches:
        tail_matches = {
            table_name
            for variant, table_name in name_lookup.items()
            if stem in _table_name_variants(variant.rsplit("_", 1)[-1])
        }
    if not tail_matches:
        # class_time_slots 같은 복합 테이블은 time_slot_id처럼 마지막 두
        # 단어만 컬럼에 남는 경우가 많다. 테이블명의 모든 suffix를 단복수
        # 변형과 비교해 특정 도메인 이름을 하드코딩하지 않고 연결한다.
        stem_variants = _table_name_variants(stem)
        for variant, table_name in name_lookup.items():
            parts = variant.split("_")
            for start in range(len(parts)):
                suffix = "_".join(parts[start:])
                if stem in _table_name_variants(suffix) or suffix in stem_variants:
                    tail_matches.add(table_name)
    if not tail_matches:
        # evidence_file_id 같은 접두사가 붙은 파일 참조는 file/files 계열의
        # 유일한 테이블로 연결한다.
        stem_parts = set(stem.split("_"))
        if "file" in stem_parts or "document" in stem_parts:
            tail_matches = {
                table_name for variant, table_name in name_lookup.items()
                if variant.rsplit("_", 1)[-1].removesuffix("s") in {"file", "document"}
            }
    if not tail_matches:
        parts = {part for part in stem.split("_") if len(part) > 2}
        tail_matches = {
            actual for actual in set(name_lookup.values())
            if parts and parts <= {part.rstrip("s") for part in actual.split("_")}
        }
    if len(tail_matches) == 1:
        return next(iter(tail_matches))
    return None


def _contract_fk_targets(registry: list[dict] | None) -> dict[tuple[str, str], tuple[str, str]]:
    targets = {}
    for item in registry or []:
        contract = item.get("dbContract") if isinstance(item, dict) else None
        for ref in (contract or {}).get("foreignKeys", []) if isinstance(contract, dict) else []:
            if isinstance(ref, dict):
                column = str(ref.get("column") or "").strip()
                references = ref.get("references") if isinstance(ref.get("references"), dict) else {}
                target_table = str(references.get("table") or "").strip()
                target_column = str(references.get("column") or "id").strip()
                source_table = str(ref.get("table") or "").strip()
                if source_table and column and target_table:
                    targets[(source_table, column)] = (target_table, target_column)
                elif column and target_table:
                    # Older Feature Specs omitted source table. Keep the target
                    # as a column-level fallback for every table that owns it.
                    targets[("*", column)] = (target_table, target_column)
                continue
            match = re.match(r"^([A-Za-z_][\w]*)\.([A-Za-z_][\w]*)\s*->\s*([A-Za-z_][\w]*)\.([A-Za-z_][\w]*)$", str(ref).strip())
            if match:
                targets[(match.group(1), match.group(2))] = (match.group(3), match.group(4))
    return targets


def _registry_items(raw) -> list[dict]:
    if isinstance(raw, list):
        return [item for item in raw if isinstance(item, dict)]
    try:
        parsed = json.loads(raw or "[]")
    except (TypeError, ValueError):
        parsed = try_parse_json(raw or "[]")
    return [item for item in parsed if isinstance(item, dict)] if isinstance(parsed, list) else []


def _registry_fallback_table_names(registry: list[dict] | None) -> list[tuple[str, str]]:
    """Return declared table names plus FK endpoints for deterministic fallback planning."""
    names: list[tuple[str, str]] = []
    seen: set[str] = set()

    def add(name: str, feature_id: str = "") -> None:
        name = str(name or "").strip()
        if not name or not _VALID_TABLE_NAME_RE.match(name):
            return
        if name not in seen:
            seen.add(name)
            names.append((name, feature_id))

    for item in registry or []:
        if not isinstance(item, dict):
            continue
        feature_id = str(item.get("featureId") or item.get("id") or "").strip()
        contract = item.get("dbContract") or item.get("db") or {}
        for table in contract.get("tables", []) if isinstance(contract, dict) else []:
            if isinstance(table, dict):
                table = table.get("name") or table.get("table")
            add(str(table or ""), feature_id)
        for ref in contract.get("foreignKeys", []) if isinstance(contract, dict) else []:
            if isinstance(ref, dict):
                add(ref.get("table"), feature_id)
                references = ref.get("references") if isinstance(ref.get("references"), dict) else {}
                add(references.get("table"), "")
            else:
                match = re.match(
                    r"^([A-Za-z_][\w]*)\.[A-Za-z_][\w]*\s*->\s*([A-Za-z_][\w]*)\.[A-Za-z_][\w]*$",
                    str(ref).strip(),
                )
                if match:
                    add(match.group(1), feature_id)
                    add(match.group(2), "")
    return names


def _registry_fallback_schema(registry: list[dict] | None) -> tuple[list[dict], list[str]]:
    """Build a contract-preserving schema when upstream generation is unavailable."""
    table_ids = {name: [] for name, _ in _registry_fallback_table_names(registry)}
    for item in registry or []:
        if not isinstance(item, dict):
            continue
        feature_id = str(item.get("featureId") or item.get("id") or "").strip()
        contract = item.get("dbContract") or item.get("db") or {}
        for table in contract.get("tables", []) if isinstance(contract, dict) else []:
            if isinstance(table, dict):
                table = table.get("name") or table.get("table")
            if str(table or "").strip() in table_ids and feature_id:
                table_ids[str(table).strip()].append(feature_id)

    tables = {
        name: {
            "name": name,
            "description": "Registry 계약 fallback 스키마",
            "featureIds": sorted(set(table_ids.get(name, []))),
            "columns": [
                {"name": "id", "type": "BIGINT", "constraints": "PRIMARY_KEY AUTO_INCREMENT"},
                {"name": "created_at", "type": "TIMESTAMP", "constraints": "NOT_NULL"},
                {"name": "updated_at", "type": "TIMESTAMP", "constraints": "NOT_NULL"},
            ],
            "indexes": [],
        }
        for name, _ in _registry_fallback_table_names(registry)
    }
    relationships: list[str] = []
    for item in registry or []:
        if not isinstance(item, dict):
            continue
        contract = item.get("dbContract") or item.get("db") or {}
        for ref in contract.get("foreignKeys", []) if isinstance(contract, dict) else []:
            source = column = target = target_column = ""
            if isinstance(ref, dict):
                source = str(ref.get("table") or "").strip()
                column = str(ref.get("column") or "").strip()
                references = ref.get("references") if isinstance(ref.get("references"), dict) else {}
                target = str(references.get("table") or "").strip()
                target_column = str(references.get("column") or "id").strip()
            else:
                match = re.match(
                    r"^([A-Za-z_][\w]*)\.([A-Za-z_][\w]*)\s*->\s*([A-Za-z_][\w]*)\.([A-Za-z_][\w]*)$",
                    str(ref).strip(),
                )
                if match:
                    source, column, target, target_column = match.groups()
            if not source or not column or not target:
                continue
            if source not in tables:
                tables[source] = {"name": source, "description": "Registry FK fallback", "featureIds": [], "columns": [], "indexes": []}
            if target not in tables:
                tables[target] = {"name": target, "description": "Registry FK target fallback", "featureIds": [], "columns": [{"name": "id", "type": "BIGINT", "constraints": "PRIMARY_KEY"}], "indexes": []}
            columns = tables[source].setdefault("columns", [])
            if not any(str(col.get("name") or "") == column for col in columns if isinstance(col, dict)):
                columns.append({"name": column, "type": "BIGINT", "constraints": f"NOT_NULL FOREIGN_KEY REFERENCES {target}({target_column or 'id'})"})
            relationships.append(f"{target} (1:N) {source}")
    return list(tables.values()), sorted(set(relationships))


def _normalize_special_references(tables: list[dict]) -> None:
    """Normalize external and polymorphic IDs before FK validation."""
    if not isinstance(tables, list):
        return
    for table in tables:
        if not isinstance(table, dict):
            continue
        table_name = str(table.get("name") or "")
        columns = table.setdefault("columns", [])
        if table_name == "audit_logs" and isinstance(columns, list):
            names = {str(col.get("name") or "") for col in columns if isinstance(col, dict)}
            if "target_id" in names and "target_type" not in names:
                columns.append({"name": "target_type", "type": "VARCHAR(50)", "constraints": "NOT_NULL"})
        for col in columns:
            if not isinstance(col, dict) or str(col.get("name") or "") not in {"provider_message_id", "target_id"}:
                continue
            constraints = str(col.get("constraints") or "")
            constraints = re.sub(r"\bFOREIGN_KEY\b", "", constraints, flags=re.IGNORECASE)
            constraints = re.sub(r"\bREFERENCES\s+[A-Za-z_][\w]*\s*\([^)]*\)", "", constraints, flags=re.IGNORECASE)
            col["constraints"] = re.sub(r"\s+", " ", constraints).strip()


def _ensure_fk_references(tables: list[dict], registry: list[dict] | None = None) -> None:
    """FK 컬럼에 실제 REFERENCES를 보강한다. target_id polymorphic ID는 제외한다."""
    if not isinstance(tables, list):
        return
    lookup = _build_name_lookup(tables)
    contract_targets = _contract_fk_targets(registry)
    for table in tables:
        if not isinstance(table, dict):
            continue
        generic_fallback = bool(re.fullmatch(r"feature_\d+_records", str(table.get("name") or "")))
        for col in table.get("columns") or []:
            if not isinstance(col, dict):
                continue
            name = str(col.get("name") or "")
            if not name.endswith("_id") or name in {"id", "target_id"}:
                continue
            target_table, target_column = contract_targets.get(
                (str(table.get("name") or ""), name),
                contract_targets.get(("*", name), (None, None)),
            )
            target = target_table or _fk_target(name, lookup)
            if target_table:
                target = lookup.get(target_table, target_table if target_table in lookup.values() else None)
            if not target or (target == table.get("name") and not target_table and name != "replaced_by_token_id"):
                # 순번 기반 generic Feature Spec fallback은 실제 도메인 계약이
                # 아니므로, 존재하지 않는 임의 FK를 제약으로 남기지 않는다.
                # 명시된 Registry FK는 contract_targets 경로에서 보존된다.
                if generic_fallback and not target_table:
                    constraints = str(col.get("constraints") or "")
                    constraints = re.sub(r"\s*FOREIGN_KEY\s*", " ", constraints, flags=re.IGNORECASE)
                    constraints = re.sub(r"\s*REFERENCES\s+[A-Za-z_][\w]*\s*\([^)]*\)", "", constraints, flags=re.IGNORECASE)
                    col["constraints"] = re.sub(r"\s+", " ", constraints).strip()
                continue
            constraints = str(col.get("constraints") or "")
            referenced = re.search(
                r"\bREFERENCES\s+([A-Za-z_][\w]*)\s*\(([^)]*)\)",
                constraints,
                re.IGNORECASE,
            )
            if referenced:
                referenced_table = referenced.group(1)
                if referenced_table.casefold() in {value.casefold() for value in lookup.values()}:
                    continue
                # A stale/LLM-invented target must not shield the column from
                # deterministic repair. Remove only the invalid FK fragment;
                # the inferred or Registry target is appended below.
                constraints = re.sub(
                    r"\s*FOREIGN_KEY\s*",
                    " ",
                    constraints,
                    flags=re.IGNORECASE,
                )
                constraints = re.sub(
                    r"\s*REFERENCES\s+[A-Za-z_][\w]*\s*\([^)]*\)",
                    "",
                    constraints,
                    flags=re.IGNORECASE,
                )
                constraints = re.sub(r"\s+", " ", constraints).strip()
            fk_token = "" if re.search(r"\bFOREIGN_KEY\b", constraints, re.IGNORECASE) else "FOREIGN_KEY "
            col["constraints"] = f"{constraints} {fk_token}REFERENCES {target}({target_column or 'id'})".strip()
            logger.warning("DBA FK REFERENCES 보강 — %s.%s → %s(id)", table.get("name"), name, target)


def _ensure_contract_tables(tables: list[dict], registry: list[dict] | None) -> list[dict]:
    """PM dbContract에 명시된 테이블이 manager 누락으로 사라지지 않게 최소 스키마를 보장한다."""
    existing = {str(t.get("name")) for t in tables if isinstance(t, dict) and t.get("name")}
    for item in registry or []:
        fid = str(item.get("featureId") or item.get("id") or "")
        for name in (item.get("dbContract") or {}).get("tables", []) if isinstance(item.get("dbContract"), dict) else []:
            if isinstance(name, dict):
                name = name.get("name") or name.get("table")
            name = str(name).strip()
            if not name or name in existing or not _VALID_TABLE_NAME_RE.match(name):
                continue
            tables.append({
                "name": name,
                "description": f"PM 계약 {fid}가 요구하는 도메인 테이블",
                "columns": [
                    {"name": "id", "type": "BIGINT", "constraints": "PRIMARY_KEY AUTO_INCREMENT"},
                    {"name": "created_at", "type": "TIMESTAMP", "constraints": "NOT_NULL"},
                    {"name": "updated_at", "type": "TIMESTAMP", "constraints": "NOT_NULL"},
                ],
                "indexes": [],
            })
            existing.add(name)
            logger.warning("DBA dbContract 테이블 누락 — 최소 스키마 보강: %s", name)
    refresh_contracts = [item for item in registry or [] if any(
        isinstance(operation, dict) and str(operation.get("method") or "GET").upper() == "POST"
        and normalize_api_path(str(operation.get("path") or "")) == "/api/v1/auth/refresh"
        for operation in item.get("apiContract") or []
    )]
    if refresh_contracts:
        owners = {str((item.get("ownership") or {}).get("ownerEntity") or "users") for item in refresh_contracts}
        owners &= {str(table.get("name")) for table in tables}
        store_names = {"refresh_tokens"}
        for item in refresh_contracts:
            for declared in (item.get("dbContract") or {}).get("tables") or []:
                store_names.add(str(declared.get("name") or declared.get("table") or "") if isinstance(declared, dict) else str(declared))
        has_store = any(
            table.get("name") in store_names and (
                {"token_hash", "expires_at"} <= {column.get("name") for column in table.get("columns") or [] if isinstance(column, dict)}
                or {"refresh_token", "expires_at"} <= {column.get("name") for column in table.get("columns") or [] if isinstance(column, dict)}
            ) and any(
                re.search(rf"\bREFERENCES\s+{re.escape(owner)}\s*\(", str(column.get("constraints") or ""), re.I)
                for owner in owners for column in table.get("columns") or [] if isinstance(column, dict)
            )
            for table in tables
        )
        # Only the explicitly declared local refresh flow needs a token store.
        # Do not create identity/password tables for unrelated or external auth.
        if not has_store and len(owners) == 1:
            owner = next(iter(owners))
            store = next((table for table in tables if table.get("name") == "refresh_tokens"), None)
            if store is None:
                store = {"name": "refresh_tokens", "description": "선언된 세션 갱신의 만료·회수 토큰 저장", "columns": [], "indexes": []}
                tables.append(store)
            _ensure_column(store, "id", "BIGINT", "PRIMARY_KEY GENERATED_IDENTITY")
            _ensure_reference_column(store, owner)
            _ensure_column(store, "token_hash", "VARCHAR(255)", "NOT_NULL UNIQUE")
            _ensure_column(store, "expires_at", "TIMESTAMPTZ", "NOT_NULL")
            _ensure_column(store, "revoked_at", "TIMESTAMPTZ", "NULL")
            _ensure_index(store, (_singular_table_name(owner) + "_id",))
            _ensure_index(store, ("expires_at",))
    return tables


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
            explicit = re.search(
                r"REFERENCES\s+([a-z][a-z0-9_]*)", str(col.get("constraints") or ""), re.I
            ) if isinstance(col, dict) else None
            ref_table = (
                explicit.group(1).lower() if explicit and explicit.group(1).lower() in set(name_lookup.values())
                else _fk_target(col.get("name") if isinstance(col, dict) else None, name_lookup)
            )
            if not ref_table or ref_table == table_name:
                continue
            key = (ref_table, table_name)
            if key in seen:
                continue
            seen.add(key)
            relationships.append(f"{ref_table} (1:N) {table_name}")

    return relationships


def _normalize_relationships(relationships: list | None, tables: list[dict]) -> list[str]:
    """Persist relationships in the validator's single canonical format."""
    table_names = {
        str(table.get("name")) for table in tables or []
        if isinstance(table, dict) and table.get("name")
    }
    normalized: list[str] = []
    for raw in relationships or []:
        text = re.sub(r"\s+", " ", str(raw or "")).strip()
        if not text:
            continue
        # Accept common LLM output such as
        # ``reservations.user_id N:1 users.id`` and rewrite it as the
        # canonical parent-to-child representation used by Phase3.
        match = re.match(
            r"^([A-Za-z_][\w]*)\.[A-Za-z_][\w]*\s+(?:N:1|many:one)\s+([A-Za-z_][\w]*)\.[A-Za-z_][\w]*$",
            text,
            re.IGNORECASE,
        )
        if match:
            child, parent = match.groups()
            candidate = f"{parent} (1:N) {child}"
            if child in table_names and parent in table_names:
                text = candidate
        if not re.search(r"\(\s*[0-9Nn*]+\s*:\s*[0-9Nn*]+\s*\)", text):
            continue
        mentioned = re.findall(r"\b[A-Za-z_][\w]*\b", text)
        if len([name for name in mentioned if name in table_names]) < 2:
            continue
        if text not in normalized:
            normalized.append(text)

    for relationship in _synthesize_relationships(tables):
        if relationship not in normalized:
            normalized.append(relationship)
    return sorted(normalized)


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


def ensure_primary_keys(tables: list[dict]) -> list[dict]:
    """모든 테이블에 명시적 PRIMARY_KEY를 보장한다."""
    for table in tables or []:
        if not isinstance(table, dict) or not table.get("name"):
            continue
        columns = table.setdefault("columns", [])
        if not isinstance(columns, list):
            columns = table["columns"] = []
        has_pk = any(
            isinstance(col, dict)
            and "PRIMARY_KEY" in str(col.get("constraints") or "").upper()
            for col in columns
        )
        if has_pk:
            continue
        id_column = next(
            (col for col in columns if isinstance(col, dict) and str(col.get("name") or "").lower() == "id"),
            None,
        )
        if id_column is None:
            columns.insert(0, {"name": "id", "type": "BIGINT", "constraints": "PRIMARY_KEY AUTO_INCREMENT"})
            logger.warning("DBA PK 보강 — %s.id PRIMARY_KEY 추가", table["name"])
            continue
        constraints = _clean_constraint(id_column.get("constraints"))
        id_column["constraints"] = f"{constraints} PRIMARY_KEY".strip()
        id_column["type"] = id_column.get("type") or "BIGINT"
        logger.warning("DBA PK 보강 — %s.id에 PRIMARY_KEY 추가", table["name"])
    return tables


def _stable_schema_order(tables: list[dict]) -> list[dict]:
    """LLM 응답 순서와 무관하게 동일한 스키마 JSON 순서를 만든다."""
    normalized = [table for table in tables or [] if isinstance(table, dict) and table.get("name")]
    for table in normalized:
        columns = [col for col in table.get("columns") or [] if isinstance(col, dict) and col.get("name")]
        table["columns"] = sorted(
            columns,
            key=lambda col: (0 if str(col.get("name")).casefold() == "id" else 1, str(col.get("name")).casefold()),
        )
        indexes = [str(index).strip() for index in table.get("indexes") or [] if str(index).strip()]
        table["indexes"] = sorted(dict.fromkeys(indexes), key=str.casefold)
    return sorted(normalized, key=lambda table: str(table.get("name")).casefold())


def reconcile_fk_types(tables: list) -> list:
    """FK 컬럼(`<참조테이블>_id`)의 타입을 참조 대상 테이블의 PK 타입에 맞춘다 — 원본을 제자리 수정.

    worker 서브에이전트는 자기 테이블 하나만 보고 설계하므로 다른 테이블의 PK 타입을 알 수 없다.
    그래서 `users.id BIGINT`인데 `orders.user_id VARCHAR(255)`처럼 조인 불가능한 스키마가
    구조적으로 발생한다. 지금까지 이걸 잡는 건 manager_review(LLM)뿐이었고 리뷰 파싱이
    실패하면 그대로 통과했으므로, PK 타입을 단일 기준으로 삼아 결정론적으로 교정한다.
    참조 대상이 없는 `<name>_id`는 최종 문서에 저장하지 않는다. API는 이 함수 실행 이후
    최종 ERD에서 다시 조립되므로 잘못된 FK를 보존하는 것보다 제거하는 편이 안전하다."""
    if not isinstance(tables, list) or not tables:
        return tables

    _normalize_special_references(tables)
    _break_reciprocal_fk_cycles(tables)
    # Preserve worker-declared principals and roles. FK reconciliation below uses
    # actual REFERENCES targets; it must not invent actors from fixed name lists.
    name_lookup = _build_name_lookup(tables)
    pk_columns = {
        tbl["name"]: pk
        for tbl in tables
        if isinstance(tbl, dict) and tbl.get("name") and (pk := _pk_column(tbl))
    }

    table_names = set(pk_columns)
    explicit_edges: set[tuple[str, str]] = set()
    for source in tables:
        source_name = str(source.get("name") or "")
        for column in source.get("columns") or []:
            if not isinstance(column, dict):
                continue
            match = re.search(r"REFERENCES\s+([a-z][a-z0-9_]*)", str(column.get("constraints") or ""), re.I)
            if match and match.group(1).lower() in table_names:
                explicit_edges.add((source_name, match.group(1).lower()))

    fixed, dangling, blocked_cycles = 0, [], []
    for tbl in tables:
        if not isinstance(tbl, dict) or not tbl.get("name"):
            continue
        clean_columns = []
        for col in tbl.get("columns") or []:
            if not isinstance(col, dict):
                continue
            col_name = col.get("name")
            explicit = re.search(r"REFERENCES\s+([a-z][a-z0-9_]*)", str(col.get("constraints") or ""), re.I)
            explicit_target = explicit.group(1).lower() if explicit and explicit.group(1).lower() in table_names else None
            ref_table = explicit_target or _fk_target(col_name, name_lookup)
            if (ref_table and not explicit_target
                    and col_name != "replaced_by_token_id"
                    and _has_reference_path(explicit_edges, ref_table, str(tbl["name"]))):
                blocked_cycles.append(f"{tbl['name']}.{col_name}->{ref_table}")
                ref_table = None
            if not ref_table:
                constraints_text = str(col.get("constraints") or "").upper()
                if (
                    isinstance(col_name, str) and col_name != "id" and col_name.endswith("_id")
                    and ("FOREIGN_KEY" in constraints_text or "REFERENCES" in constraints_text)
                ):
                    dangling.append(f"{tbl['name']}.{col_name}")
                    continue
                clean_columns.append(col)
                continue
            ref_pk = pk_columns.get(ref_table)
            if not ref_pk:
                clean_columns.append(col)
                continue
            ref_pk_name, ref_pk_type = ref_pk
            cur = str(col.get("type") or "").strip()
            if cur.upper() == ref_pk_type.upper():
                pass
            else:
                logger.warning(
                    "DBA FK 타입 불일치 — %s.%s: %s → %s (%s.%s 기준)",
                    tbl["name"], col_name, cur or "(없음)", ref_pk_type, ref_table, ref_pk_name,
                )
                col["type"] = ref_pk_type
                fixed += 1
            constraints = _clean_constraint(str(col.get("constraints") or ""))
            reference = f"REFERENCES {ref_table}({ref_pk_name})"
            if "FOREIGN_KEY" not in constraints:
                constraints = (constraints + " FOREIGN_KEY").strip()
            if reference not in constraints:
                constraints = (constraints + " " + reference).strip()
            col["constraints"] = constraints
            clean_columns.append(col)
        tbl["columns"] = clean_columns

    # Column removal/retargeting happens after the first sanitize pass. Reconcile
    # indexes against the final columns and guarantee an index for every real FK,
    # so repeated selective repairs are idempotent instead of oscillating.
    for tbl in tables:
        _repair_indexes(tbl)
        for col in tbl.get("columns") or []:
            if isinstance(col, dict) and re.search(r"REFERENCES\s+", str(col.get("constraints") or ""), re.I):
                _ensure_index(tbl, (str(col.get("name")),))

    if fixed:
        logger.info("DBA FK 타입 %d개 교정 — 참조 PK 타입에 정합", fixed)
    if dangling:
        logger.warning("DBA 참조 대상 없는 FK 컬럼 %d개 제거: %s", len(dangling), dangling)
    if blocked_cycles:
        logger.warning("DBA 순환 방향을 만드는 추론 FK %d개 차단: %s", len(blocked_cycles), blocked_cycles)
    return tables


def _ensure_column(table: dict, name: str, col_type: str, constraints: str = "") -> None:
    columns = table.setdefault("columns", [])
    if any(isinstance(c, dict) and c.get("name") == name for c in columns):
        return
    columns.append({"name": name, "type": col_type, "constraints": constraints})


def _ensure_auth_contract_tables(tables: list[dict]) -> list[dict]:
    """Materialize the minimum deployable identity/session persistence contract."""
    users = next((table for table in tables if str(table.get("name") or "") == "users"), None)
    if users is None:
        users = {
            "name": "users", "description": "회원 가입 및 로그인; 내 정보 및 계정 관리",
            "columns": [], "indexes": [],
        }
        tables.append(users)
    _ensure_column(users, "id", "BIGINT", "PRIMARY_KEY GENERATED_IDENTITY")
    _ensure_column(users, "email", "VARCHAR(320)", "NOT_NULL UNIQUE")
    _ensure_column(users, "password_hash", "VARCHAR(255)", "NOT_NULL")
    _ensure_column(users, "status", "VARCHAR(20)", "NOT_NULL DEFAULT 'ACTIVE' CHECK (status IN ('ACTIVE','SUSPENDED','WITHDRAWN'))")
    _ensure_column(users, "created_at", "TIMESTAMPTZ", "NOT_NULL DEFAULT_NOW")
    _ensure_column(users, "updated_at", "TIMESTAMPTZ", "NOT_NULL DEFAULT_NOW")

    sessions = next((table for table in tables if str(table.get("name") or "") == "refresh_tokens"), None)
    if sessions is None:
        sessions = {
            "name": "refresh_tokens", "description": "인증 세션 관리; Refresh Token 회수와 만료",
            "columns": [], "indexes": [],
        }
        tables.append(sessions)
    _ensure_column(sessions, "id", "BIGINT", "PRIMARY_KEY GENERATED_IDENTITY")
    _ensure_reference_column(sessions, "users")
    _ensure_column(sessions, "token_hash", "VARCHAR(255)", "NOT_NULL UNIQUE")
    _ensure_column(sessions, "expires_at", "TIMESTAMPTZ", "NOT_NULL")
    _ensure_column(sessions, "revoked_at", "TIMESTAMPTZ", "NULL")
    _ensure_column(sessions, "created_at", "TIMESTAMPTZ", "NOT_NULL DEFAULT_NOW")
    _ensure_index(sessions, ("user_id",))
    _ensure_index(sessions, ("expires_at",))
    return tables


def _primary_contract_table(spec: dict, tables: list[dict], declared: list[str]) -> dict | None:
    """Pick the table a feature's API operates on from every table its DB contract lists.

    A contract lists owners and parents (users, teams) next to the resource itself, so
    taking the first table in schema order maps unrelated features to the same owner table.
    """
    by_name = {str(item.get("name") or ""): item for item in tables if isinstance(item, dict)}
    candidates = [name for name in dict.fromkeys(declared) if name in by_name]
    if len(candidates) <= 1:
        return by_name[candidates[0]] if candidates else None

    # 1) The most specific resource segment of the API path, mutating operations first.
    operations = sorted(
        (item for item in spec.get("apiContract") or [] if isinstance(item, dict)),
        key=lambda item: str(item.get("method") or "GET").upper() in {"GET", "DELETE"},
    )
    for operation in operations:
        named = route_contract_table(operation.get("path"), candidates)
        if named:
            return by_name[named]

    # 2) The dependent table of the contract: it references the others and nothing references it.
    foreign_keys = [
        item for item in (spec.get("dbContract") or {}).get("foreignKeys") or [] if isinstance(item, dict)
    ]
    sources = {str(item.get("table") or "") for item in foreign_keys}
    targets = {
        str((item.get("references") or {}).get("table") or "")
        for item in foreign_keys if isinstance(item.get("references"), dict)
    }
    dependents = [name for name in candidates if name in sources and name not in targets]
    if len(dependents) == 1:
        return by_name[dependents[0]]

    # 3) Keep schema order, but never prefer the identity principal over a business table.
    ordered = [item for item in tables if isinstance(item, dict) and item.get("name") in candidates]
    return next((item for item in ordered if not _is_principal_table(item)), ordered[0])


def build_feature_mappings(tables: list[dict], feature_specs: list[dict] | None) -> list[dict]:
    """Resolve each normalized PM feature to one physical table deterministically."""
    mappings: list[dict] = []
    for spec in feature_specs or []:
        if not isinstance(spec, dict) or not spec.get("name"):
            continue
        name = str(spec["name"])
        declared = (spec.get("dbContract") or {}).get("tables") if isinstance(spec.get("dbContract"), dict) else None
        names = [
            str(item.get("name") or item.get("table") or "") if isinstance(item, dict) else str(item)
            for item in declared or []
        ]
        if name in {"회원 가입 및 로그인", "내 정보 및 계정 관리", "사용자별 데이터 접근 제어"}:
            table = next((item for item in tables if item.get("name") == "users"), None)
        elif name == "인증 세션 관리":
            table = next((item for item in tables if item.get("name") == "refresh_tokens"), None)
        elif names:
            table = _primary_contract_table(spec, tables, names)
        else:
            eligible = [item for item in tables if not _is_principal_table(item) and item.get("name") != "refresh_tokens"]
            exact = [item for item in eligible if name in str(item.get("description") or "")]
            if len(exact) == 1:
                table = exact[0]
            else:
                feature_text = " ".join([
                    name, str(spec.get("rationale") or ""),
                    " ".join(str(value) for value in spec.get("actions") or []),
                    " ".join(str(value) for value in spec.get("dataRequirements") or []),
                ])
                scored = [
                    (max(
                        relevance_score(feature_text, f"{item.get('name', '')} {item.get('description', '')}"),
                        relevance_score(str(item.get("description") or ""), feature_text),
                    ), item)
                    for item in eligible
                ]
                ranked = sorted(scored, key=lambda pair: (-pair[0], str(pair[1].get("name") or "")))
                table = ranked[0][1] if ranked else None
        if table and name not in str(table.get("description") or ""):
            table["description"] = f"{table.get('description', '')}; 기능 매핑: {name}".strip("; ")
        ownership = spec.get("ownership") if isinstance(spec.get("ownership"), dict) else {}
        scope = str(ownership.get("scope") or "PUBLIC").upper()
        if table and scope in {"USER", "SHARED"} and table.get("name") not in {"users", "refresh_tokens"}:
            owner_entity = str(ownership.get("ownerEntity") or "users")
            if any(item.get("name") == owner_entity for item in tables):
                _ensure_reference_column(table, owner_entity)
                _ensure_index(table, (f"{_singular_table_name(owner_entity)}_id",))
        table_names = {str(item.get("name") or "") for item in tables if isinstance(item, dict)}
        mappings.append({
            "featureName": name,
            "featureId": str(spec.get("featureId") or spec.get("id") or ""),
            "table": str(table.get("name") or "") if table else "",
            # Every contract table, so a route of this feature can be checked against the one it names.
            "tables": [item for item in dict.fromkeys(names) if item in table_names],
            "ownership": ownership,
            "states": list(spec.get("states") or []),
            "stateTransitions": list(spec.get("stateTransitions") or []),
            "transactionRules": list(spec.get("transactionRules") or []),
        })
    return mappings


def enforce_schema_contracts(
    tables: list[dict], feature_list: list[str] | None = None, prd_document: str = ""
) -> list[dict]:
    """Apply cross-table contracts without relying on one service domain.

    A reference/catalog table owns identity and descriptive metadata. A user-scoped
    aggregate owns mutable state. Event/history tables point at that aggregate so a
    transaction can be reversed. Optional domain capabilities are never invented;
    only structures already produced for the requested service are normalized.
    """
    tables = [table for table in (tables or []) if isinstance(table, dict) and table.get("name")]
    _apply_prd_semantic_contracts(tables, prd_document)
    prd_text = str(prd_document or "").lower()
    preference_required = any(token in prd_text for token in (
        "알림 설정", "수신 설정", "notification setting", "alert setting",
    ))
    if preference_required:
        candidates = []
        for table in tables:
            names = {
                str(column.get("name") or "")
                for column in table.get("columns") or [] if isinstance(column, dict)
            }
            table_text = f"{table.get('name', '')} {table.get('description', '')}".lower()
            if "user_id" in names and any(token in table_text for token in (
                "notification", "alert", "알림",
            )):
                candidates.append(table)
        if candidates:
            owner = max(candidates, key=lambda table: relevance_score(
                "알림 수신 설정", f"{table.get('name', '')} {table.get('description', '')}",
            ))
            _ensure_column(owner, "is_enabled", "BOOLEAN", "NOT_NULL DEFAULT_TRUE")
    by_name = {str(table["name"]): table for table in tables}

    # Classify state and event roles from physical fields/FKs, never table names.
    name_lookup = _build_name_lookup(tables)
    occurrence_fields = {
        "occurred_at", "recorded_at", "event_at", "completed_at", "processed_at",
        "started_at", "ended_at", "effective_at",
    }
    event_table_names = {
        str(table.get("name") or "") for table in tables
        if (
            {str(c.get("name")) for c in table.get("columns") or [] if isinstance(c, dict)} & occurrence_fields
            or any(
                re.search(r"(?:^delta_|_delta$|_change$|_used$|^previous_|^new_)", str(c.get("name") or ""))
                for c in table.get("columns") or [] if isinstance(c, dict)
            )
            or (
                len({
                    _fk_target(c.get("name"), name_lookup)
                    for c in table.get("columns") or [] if isinstance(c, dict)
                    and _fk_target(c.get("name"), name_lookup)
                }) >= 2
                and bool({str(c.get("name")) for c in table.get("columns") or [] if isinstance(c, dict)} & {"status", "state"})
                and any(
                    {
                        str(c.get("name")) for c in by_name.get(target, {}).get("columns") or []
                        if isinstance(c, dict)
                    } & {"status", "state"}
                    for target in {
                        _fk_target(c.get("name"), name_lookup)
                        for c in table.get("columns") or [] if isinstance(c, dict)
                        and _fk_target(c.get("name"), name_lookup)
                    }
                )
            )
        )
        and any(
            isinstance(c, dict) and _fk_target(c.get("name"), name_lookup)
            for c in table.get("columns") or []
        )
    }
    aggregates: list[dict] = []
    aggregate_catalogs: dict[str, set[str]] = {}
    for table in tables:
        table_name = str(table.get("name") or "").lower()
        if table_name in event_table_names:
            continue
        columns = {str(c.get("name")) for c in table.get("columns") or [] if isinstance(c, dict)}
        mutable = columns & _MUTABLE_STATE_COLUMNS
        if not mutable:
            continue
        aggregates.append(table)
        aggregate_catalogs[table["name"]] = {
            target for c in table.get("columns") or []
            if isinstance(c, dict) and (target := _fk_target(c.get("name"), name_lookup))
            and target != table["name"]
        }

    # Catalogs keep descriptive metadata; duplicated mutable state belongs to aggregates.
    for aggregate in aggregates:
        aggregate_columns = {str(c.get("name")) for c in aggregate.get("columns") or [] if isinstance(c, dict)}
        for catalog_name in aggregate_catalogs.get(aggregate["name"], set()):
            catalog = by_name.get(catalog_name)
            if not catalog:
                continue
            duplicated = aggregate_columns & (_MUTABLE_STATE_COLUMNS - {"status", "state"})
            # A downstream delivery/history resource may snapshot timing and its
            # own status, but it must not become the owner of mutable business
            # quantities/balances that belong to the referenced aggregate.
            if any(token in str(aggregate.get("name") or "") for token in _DOWNSTREAM_ACTION_TOKENS):
                downstream_duplicates = duplicated - {"expires_at", "scheduled_at"}
                aggregate["columns"] = [
                    column for column in aggregate.get("columns") or []
                    if not (
                        isinstance(column, dict)
                        and str(column.get("name") or "") in downstream_duplicates
                    )
                ]
                aggregate_columns -= downstream_duplicates
                duplicated -= downstream_duplicates
            catalog["columns"] = [
                column for column in catalog.get("columns") or []
                if not (isinstance(column, dict) and column.get("name") in duplicated)
            ]

    # Event/history rows must identify the concrete aggregate they mutate, not only its catalog item.
    name_lookup = _build_name_lookup(tables)
    for table in tables:
        table_name = str(table.get("name") or "").lower()
        if table_name not in event_table_names:
            continue
        current_targets = {
            target for column in table.get("columns") or []
            if isinstance(column, dict) and (target := _fk_target(column.get("name"), name_lookup))
        }
        candidates = [
            aggregate for aggregate in aggregates
            if aggregate["name"] != table.get("name")
            and (aggregate_catalogs.get(aggregate["name"], set()) & current_targets)
        ]
        if len(candidates) > 1:
            event_columns = {
                str(column.get("name")) for column in table.get("columns") or [] if isinstance(column, dict)
            }
            scores = {
                candidate["name"]: sum(
                    1 for state_name in (
                        {str(column.get("name")) for column in candidate.get("columns") or [] if isinstance(column, dict)}
                        & _MUTABLE_STATE_COLUMNS
                    )
                    if any(event_name == state_name or event_name.startswith(f"{state_name}_") for event_name in event_columns)
                )
                for candidate in candidates
            }
            best_score = max(scores.values(), default=0)
            candidates = [candidate for candidate in candidates if scores[candidate["name"]] == best_score] if best_score else []
        if len(candidates) != 1:
            continue
        aggregate = candidates[0]
        aggregate_fk = f"{_singular_table_name(aggregate['name'])}_id"
        _ensure_column(table, aggregate_fk, "BIGINT", "NOT_NULL")
        obsolete_catalogs = aggregate_catalogs.get(aggregate["name"], set())
        table["columns"] = [
            column for column in table.get("columns") or []
            if not (
                isinstance(column, dict)
                and column.get("name", "").endswith("_id")
                and _fk_target(column.get("name"), name_lookup) in obsolete_catalogs
            )
        ]

    # Every FK gets an index; queue/scheduler scans get a composite status/time index.
    name_lookup = _build_name_lookup(tables)
    for table in tables:
        column_names = {str(c.get("name")) for c in table.get("columns") or [] if isinstance(c, dict)}
        for column_name in sorted(column_names):
            if _fk_target(column_name, name_lookup):
                _ensure_index(table, (column_name,))
        if {"status", "scheduled_at"}.issubset(column_names):
            _ensure_index(table, ("status", "scheduled_at"))
    return tables


def ensure_domain_columns(tables: list[dict]) -> list[dict]:
    """Normalize SQL syntax without inventing service-specific business fields."""
    for table in tables or []:
        if not isinstance(table, dict):
            continue
        for col in table.get("columns") or []:
            if isinstance(col, dict):
                col_type = re.sub(r"\s+", " ", str(col.get("type") or "").upper().strip())
                constraints = str(col.get("constraints") or "")
                if col_type in {"DATETIME", "TIMESTAMP WITH TIME ZONE"}:
                    col["type"] = "TIMESTAMPTZ"
                elif col_type == "INT":
                    col["type"] = "INTEGER"
                if col_type in {"TIMESTAMP", "TIMESTAMPTZ"} and re.search(r"\bWITH\s+TIME\s+ZONE\b", constraints, re.I):
                    col["type"] = "TIMESTAMPTZ"
                constraints = re.sub(r"\b(?:WITH|WITHOUT)\s+TIME\s+ZONE\b", "", constraints, flags=re.I)
                constraints = re.sub(r"\bGENERATED\s+BY\s+DEFAULT(?:\s+AS\s+IDENTITY)?\b", "GENERATED_IDENTITY", constraints, flags=re.I)
                constraints = re.sub(r"\bGENERATED\s+ALWAYS(?:\s+AS\s+IDENTITY)?\b", "GENERATED_IDENTITY", constraints, flags=re.I)
                constraints = re.sub(r"\bGENERATED\s+IDENTITY\b", "GENERATED_IDENTITY", constraints, flags=re.I)
                constraints = re.sub(r"\bDEFAULT\s+NOW\s*\(\s*\)", "DEFAULT_NOW", constraints, flags=re.I)
                name = str(col.get("name") or "").lower()
                temporal_name = bool(
                    name.endswith("_at") or re.search(
                        r"(?:^|_)(?:date|time|datetime|deadline|due|expiry|expiration|started|ended)(?:_|$)",
                        name,
                    )
                )
                if col_type in {"TIMESTAMP", "TIMESTAMPTZ"} and not temporal_name:
                    inferred_type, _ = _infer_column_type(name)
                    if inferred_type not in {"TIMESTAMP", "TIMESTAMPTZ"}:
                        col["type"] = inferred_type
                col["constraints"] = _clean_constraint(constraints)
    return tables


def _parse_table_plan_text(raw: str) -> list[dict]:
    result, seen = [], set()
    for line in (raw or "").splitlines():
        if not line.upper().startswith("TABLE:"):
            continue
        parts = [part.strip() for part in line.split("|||", 1)]
        name = parts[0].split(":", 1)[1].strip().lower()
        if _VALID_TABLE_NAME_RE.match(name) and name not in seen:
            seen.add(name)
            result.append({"name": name, "purpose": parts[1] if len(parts) > 1 else name})
    return result


def _parse_table_text(raw: str, name: str, purpose: str) -> dict | None:
    description, columns, indexes = purpose, [], []
    for line in (raw or "").splitlines():
        line = line.strip().strip("`*- ")
        upper = line.upper()
        if upper.startswith("DESCRIPTION:"):
            description = line.split(":", 1)[1].strip() or purpose
        elif upper.startswith("COLUMN:") and line.split(":", 1)[1].strip():
            columns.append(line.split(":", 1)[1].strip())
        elif upper.startswith("INDEX:") and line.split(":", 1)[1].strip():
            indexes.append(line.split(":", 1)[1].strip())
    return {"name": name, "description": description, "columns": columns, "indexes": indexes} if columns else None


def enforce_feature_relation_contracts(
    tables: list[dict], feature_list: list[str], auth_required: bool,
    prd_document: str = "",
) -> list[dict]:
    """Materialize manager-owned root/association/event contracts after worker output.

    Matching uses each table's retained feature description. No service entity name or
    actor vocabulary is assumed.
    """
    prd = try_parse_json(prd_document or "{}") or {}
    core_features = [item for item in prd.get("coreFeatures") or [] if isinstance(item, dict)]
    feature_tables: list[tuple[str, dict, str]] = []
    for feature in feature_list or []:
        if str(feature) in {
            "회원 가입 및 로그인", "인증 세션 관리", "내 정보 및 계정 관리", "사용자별 데이터 접근 제어",
        }:
            continue
        allow_principal = _feature_targets_principal(str(feature))
        kind = feature_relation_kind(str(feature))
        eligible = [
            table for table in tables
            if allow_principal or not _is_principal_table(table)
        ]
        exact = [
            table for table in eligible
            if str(table.get("description") or "").split(";", 1)[0].strip() == str(feature).strip()
        ]
        if len(exact) == 1:
            feature_tables.append((str(feature), exact[0], feature_relation_kind(str(feature))))
            continue
        contained = [
            table for table in eligible
            if str(feature).strip()
            and str(feature).strip() in str(table.get("description") or "")
            and not (
                kind == "aggregate"
                and any(
                    token in str(table.get("name") or "")
                    for token in _DOWNSTREAM_ACTION_TOKENS
                )
            )
        ]
        if len(contained) == 1:
            feature_tables.append((str(feature), contained[0], feature_relation_kind(str(feature))))
            continue
        scored = []
        for table in eligible:
            name = str(table.get("name") or "")
            text = f"{name} {table.get('description', '')}"
            score = max(relevance_score(str(feature), text), relevance_score(text, str(feature)))
            if kind == "association" and any(
                token in name for token in ("assignment", "membership", "link", "mapping")
            ):
                score += 1.0
            if kind == "event" and any(
                token in name for token in ("record", "history", "log", "event")
            ):
                score += 1.0
            if kind == "aggregate" and any(token in name for token in _DOWNSTREAM_ACTION_TOKENS):
                score -= 1.0
            scored.append((score, table))
        best = max((score for score, _table in scored), default=0.0)
        winners = [table for score, table in scored if score == best and score >= 0.25]
        if len(winners) == 1:
            feature_tables.append((str(feature), winners[0], kind))
    root = next((table for _feature, table, kind in feature_tables if kind == "aggregate"), None)
    if root is None:
        root_candidates = [
            table for table in tables
            if not _is_principal_table(table)
            and not any(
                token in str(table.get("name") or "")
                for token in (
                    "assignment", "membership", "link", "mapping", "record", "history", "log", "event",
                    *_DOWNSTREAM_ACTION_TOKENS,
                )
            )
        ]
        if len(root_candidates) == 1:
            root = root_candidates[0]
    principal = next((table for table in tables if _is_principal_table(table)), None)
    for _feature, table, kind in feature_tables:
        if (root and table is not root and kind in {"association", "event", "derived"}
                and (not core_features or _declares_upstream_relation(table, root, tables, core_features))):
            _ensure_reference_column(table, str(root["name"]))
        if auth_required and principal and table is not principal:
            _ensure_reference_column(table, str(principal["name"]))
            for column in table.get("columns") or []:
                if (
                    isinstance(column, dict)
                    and str(column.get("name") or "").endswith(("_by", "_to"))
                    and str(column.get("type") or "").upper() in {"BIGINT", "INT", "INTEGER", "UUID"}
                    and "REFERENCES" not in str(column.get("constraints") or "").upper()
                ):
                    column["constraints"] = f"NOT_NULL FOREIGN_KEY REFERENCES {principal['name']}(id)"
    if auth_required and principal:
        for table in tables:
            if table is principal:
                continue
            for column in table.get("columns") or []:
                if (
                    isinstance(column, dict)
                    and str(column.get("name") or "").endswith(("_by", "_to"))
                    and str(column.get("type") or "").upper() in {"BIGINT", "INT", "INTEGER", "UUID"}
                    and "REFERENCES" not in str(column.get("constraints") or "").upper()
                ):
                    column["constraints"] = f"NOT_NULL FOREIGN_KEY REFERENCES {principal['name']}(id)"
    # Numeric `<something>_id` columns are local relation identifiers. If no
    # physical FK contract exists after manager reconciliation, preserving them
    # creates an undeclared relation that API/ORM code cannot implement safely.
    for table in tables:
        table["columns"] = [
            column for column in table.get("columns") or []
            if not (
                isinstance(column, dict)
                and str(column.get("name") or "") != "id"
                and str(column.get("name") or "").endswith("_id")
                and str(column.get("type") or "").upper() in {"BIGINT", "INT", "INTEGER", "SMALLINT"}
                and "REFERENCES" not in str(column.get("constraints") or "").upper()
            )
        ]
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
    {"name": "id", "type": "BIGINT", "constraints": "PRIMARY_KEY GENERATED_IDENTITY"},
    {"name": "created_at", "type": "TIMESTAMPTZ", "constraints": "NOT_NULL DEFAULT_NOW"},
    {"name": "updated_at", "type": "TIMESTAMPTZ", "constraints": "NOT_NULL DEFAULT_NOW"},
]


_VALID_TYPE_RE = re.compile(
    r'^(BIGINT|INT|INTEGER|SMALLINT|VARCHAR|VARCHAR\(\d+\)|CHAR\(\d+\)|TEXT|BOOLEAN|DATE|TIMESTAMP|TIMESTAMPTZ|(?:DECIMAL|NUMERIC)\(\d+,\d+\)|REAL|DOUBLE PRECISION|JSONB|UUID)$',
    re.I,
)


# 공백으로 끊긴 제약조건 키워드 → 표준 언더스코어 형태 (예: 'NOT NULL' → 'NOT_NULL')
_CONSTRAINT_PHRASE_FIXES = [
    (re.compile(r'\bPRIMARY\s+KEY\b', re.I), 'PRIMARY_KEY'),
    (re.compile(r'\bAUTO[_\s]+INCREMENT\b', re.I), 'GENERATED_IDENTITY'),
    (re.compile(r'\bFOREIGN\s+KEY\b', re.I), 'FOREIGN_KEY'),
    (re.compile(r'\bNOT\s+NULL\b', re.I), 'NOT_NULL'),
    (re.compile(r'\bDEFAULT\s+FALSE\b', re.I), 'DEFAULT_FALSE'),
    (re.compile(r'\bDEFAULT\s+TRUE\b', re.I), 'DEFAULT_TRUE'),
]
# 개별 토큰 오타 → 표준형 (빈 문자열이면 제거). EXAONE 실측 오타: NOTULL/NOTNULL/NOT_EMPTY 등
_CONSTRAINT_TOKEN_FIXES = {
    'NOTULL': 'NOT_NULL', 'NOTNULL': 'NOT_NULL', 'NOTNUL': 'NOT_NULL', 'NOT_NUL': 'NOT_NULL',
    'NOT_EMPTY': 'NOT_NULL', 'NOT_HIDDEN': '',
    'PRIMARYKEY': 'PRIMARY_KEY', 'AUTOINCREMENT': 'GENERATED_IDENTITY', 'AUTO_INCREMENT': 'GENERATED_IDENTITY', 'FOREIGNKEY': 'FOREIGN_KEY',
    'DEFAULTFALSE': 'DEFAULT_FALSE', 'DEFAULTTRUE': 'DEFAULT_TRUE',
    'REQUIRED': 'NOT_NULL', 'OPTIONAL': 'NULL',
    'BOTH': '', '?': '', 'NONE': '', 'N/A': '',
}


def _clean_constraint(s) -> str:
    """Rebuild a constraint from the supported PostgreSQL grammar only."""
    if not isinstance(s, str):
        return ""
    normalized = s
    for rx, replacement in _CONSTRAINT_PHRASE_FIXES:
        normalized = rx.sub(replacement, normalized)
    for token, fixed in _CONSTRAINT_TOKEN_FIXES.items():
        normalized = re.sub(
            rf'(?<![A-Za-z0-9_]){re.escape(token)}(?![A-Za-z0-9_])',
            fixed, normalized, flags=re.I,
        )
    normalized = re.sub(r'\bDEFAULT\s+NOW\s*\(\s*\)', 'DEFAULT_NOW', normalized, flags=re.I)

    references = [
        target for target in re.findall(
            r'\bREFERENCES\s+([a-z][a-z0-9_]*)(?:\s*\([^)]*\))?', normalized, re.I
        )
        if target.lower() not in {"foreign_key", "primary_key", "key", "references"}
    ]
    reference = f"REFERENCES {references[-1]}(id)" if references else ""

    def balanced_check(text: str) -> str:
        match = re.search(r'\bCHECK\s*\(', text, re.I)
        if not match:
            return ""
        start, depth, quoted = match.start(), 0, False
        for index in range(match.end() - 1, len(text)):
            char = text[index]
            if char == "'":
                quoted = not quoted
            elif not quoted and char == "(":
                depth += 1
            elif not quoted and char == ")":
                depth -= 1
                if depth == 0:
                    return re.sub(r'\s+', ' ', text[start:index + 1]).strip()
        return ""

    upper = normalized.upper()
    parts: list[str] = []
    for token in ("PRIMARY_KEY", "GENERATED_IDENTITY"):
        if re.search(rf'\b{token}\b', upper):
            parts.append(token)
    if re.search(r'\bNOT_NULL\b', upper):
        parts.append("NOT_NULL")
    elif re.search(r'\b(?:NULL|NULLABLE)\b', upper):
        parts.append("NULL")
    if re.search(r'\bUNIQUE\b', upper):
        parts.append("UNIQUE")
    default = re.search(
        r"\b(DEFAULT_(?:FALSE|TRUE|NOW)|DEFAULT\s+(?:CURRENT_TIMESTAMP|TRUE|FALSE|'[^']*'|[-+]?\d+(?:\.\d+)?))",
        normalized, re.I,
    )
    if default:
        value = re.sub(r'\s+', ' ', default.group(1)).strip()
        parts.append(value.upper() if value.upper().startswith("DEFAULT_") else value)
    check = balanced_check(normalized)
    if check:
        parts.append(check)
    if reference:
        parts.extend(("FOREIGN_KEY", reference))
    on_delete = re.search(
        r'\bON\s+DELETE\s+(CASCADE|SET\s+NULL|RESTRICT|NO\s+ACTION)\b', normalized, re.I
    )
    if on_delete:
        action = re.sub(r'\s+', ' ', on_delete.group(1)).upper()
        parts.append(f"ON DELETE {action}")
    return " ".join(dict.fromkeys(parts))[:240]


def _infer_column_type(name: str) -> tuple[str, str]:
    """컬럼명으로 타입·제약조건을 결정론적으로 추론 — EXAONE이 타입 없이 컬럼명만 낸 경우 보강."""
    n = (name or "").lower()
    if n == "id":
        return "BIGINT", "PRIMARY_KEY GENERATED_IDENTITY"
    if n.endswith("_id"):
        return "BIGINT", "NOT_NULL"
    if n in ("created_at", "updated_at"):
        return "TIMESTAMPTZ", "NOT_NULL DEFAULT_NOW"
    if n.endswith("_at") or re.search(
        r"(?:^|_)(?:date|time|datetime|deadline|due|expiry|expiration|started|ended)(?:_|$)", n
    ):
        return "TIMESTAMPTZ", "NULL"
    if n.startswith("is_") or n.startswith("has_") or n.endswith("_flag"):
        return "BOOLEAN", "DEFAULT_FALSE"
    if any(k in n for k in ("quantity", "count", "amount", "stock", "qty", "price", "cost")):
        return "DECIMAL(10,2)", "NOT_NULL"
    if "email" in n:
        return "VARCHAR(255)", "UNIQUE"
    if any(k in n for k in ("description", "content", "memo", "note", "body", "text")):
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
        if not old:
            continue
        if _NON_ASCII_RE.search(old):
            new = _englishize_column(old, i, name_lookup)
        elif not _VALID_COLUMN_NAME_RE.match(old):
            new = re.sub(r'_+', '_', re.sub(r'[^a-z0-9_]+', '_', old.lower())).strip('_')
            new = {"updated": "updated_at", "created": "created_at"}.get(new, new)
            if not new or not _VALID_COLUMN_NAME_RE.match(new):
                new = f"col_{i}"
        else:
            continue
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

    kept, fixed, dropped, seen_normalized, seen_names = [], [], [], set(), set()
    for entry in indexes:
        text = str(entry)
        syntax = re.match(
            r"^CREATE\s+(?:UNIQUE\s+)?INDEX\s+([a-z][a-z0-9_]*)\s+ON\s+([a-z][a-z0-9_]*)\s*\(([^()]*)\)(?:\s+WHERE\s+.+)?;?$",
            text.strip(), re.I,
        )
        if not syntax or syntax.group(2).lower() != str(table.get("name") or "").lower():
            dropped.append(text)
            continue
        where_match = re.search(r"\bWHERE\s+(.+?);?$", text, re.I)
        if where_match:
            where_expr = where_match.group(1).strip()
            if (
                where_expr.count("(") != where_expr.count(")")
                or not re.search(r"(?:=|<>|!=|<=|>=|<|>|\bIS\s+(?:NOT\s+)?NULL\b|\bIN\s*\()", where_expr, re.I)
            ):
                dropped.append(text)
                continue
        match = syntax
        index_name = match.group(1).lower()
        if index_name in seen_names:
            dropped.append(text)
            continue
        referenced = [c.strip() for c in match.group(3).split(",") if c.strip()]
        unknown = [c for c in referenced if c not in col_names]
        if not unknown:
            normalized_entry = re.sub(r"\s+|;", "", text).lower()
            if normalized_entry not in seen_normalized:
                seen_normalized.add(normalized_entry)
                seen_names.add(index_name)
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
            seen_names.add(index_name)
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
                if not c["constraints"]:
                    _, inferred_constraints = _infer_column_type(str(c.get("name") or ""))
                    c["constraints"] = inferred_constraints
                column_name = str(c.get("name") or "")
                constraint_text = str(c.get("constraints") or "")
                for identifier in re.findall(
                    r"\b([a-z][a-z0-9_]*)\s*(?=(?:>=|<=|<>|!=|=|>|<|\bIN\s*\())",
                    constraint_text, re.I,
                ):
                    if identifier.lower() == column_name.lower():
                        continue
                    close = difflib.get_close_matches(identifier.lower(), [column_name.lower()], n=1, cutoff=0.6)
                    if close:
                        constraint_text = re.sub(
                            rf"\b{re.escape(identifier)}\b", column_name, constraint_text, flags=re.I,
                        )
                c["constraints"] = constraint_text
                if (
                    re.match(r"^(?:is|has|can)_[a-z0-9_]+$", column_name)
                    and str(c.get("type") or "").upper() in {"SMALLINT", "INTEGER", "INT", "BIGINT"}
                ):
                    c["type"] = "BOOLEAN"
                    c["constraints"] = re.sub(
                        r"\s*CHECK\s*\([^)]*\)", "", c["constraints"], flags=re.I,
                    ).strip()
                if (
                    str(c.get("name") or "") == "id"
                    and str(c.get("type") or "").upper() in {"BIGINT", "INT", "INTEGER", "SMALLINT"}
                    and "PRIMARY_KEY" in c["constraints"]
                    and "GENERATED_IDENTITY" not in c["constraints"]
                ):
                    c["constraints"] = f"{c['constraints']} GENERATED_IDENTITY"
            if filled:
                logger.warning("DBA 테이블 '%s' — 타입 없음/손상 컬럼 %d개 추론 보강", name, filled)
        # 인덱스 대조 전에 컬럼명을 영문화해야 인덱스가 '없는 컬럼 참조'로 오판돼 삭제되지 않는다
        _normalize_column_names(t, name_lookup)
        # 모델이 두 컬럼을 붙여 쓴 결과(예: status_idissuued_at)는 타입 추론으로
        # 복구할 수 없으므로 최종 스키마에 저장하지 않는다. 의미 복구는 해당 worker 재시도 대상이다.
        malformed_columns = []
        valid_columns = []
        for column in t.get("columns") or []:
            column_name = str(column.get("name") or "") if isinstance(column, dict) else ""
            if re.search(r"_id(?:issu+ed|created|updated|recorded|assigned|changed|due|status|at)(?:_at)?$", column_name):
                malformed_columns.append(column_name)
                continue
            valid_columns.append(column)
        if malformed_columns:
            logger.warning("DBA 테이블 '%s' — 병합 오염 컬럼 제거: %s", name, malformed_columns)
            t["columns"] = valid_columns
        _dedupe_columns(t)
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
- 기능 목록이 조직·동호회·팀 소속을 전제로 하면 단일 조직인지 확인하고, 다중 조직이면 organizations/clubs와 memberships 매핑 테이블을 계획에 포함하세요.
- 기능별로 필요한 테이블을 "featureId → 테이블명" 계약 매핑으로 함께 작성하세요. 상태 변화 기능은 상태 컬럼과 변경 이력 저장 위치를 명시하세요.
- PM/Feature Spec Registry의 각 dbContract.tables를 한 항목도 빠뜨리지 말고 plan에 반영하세요. Registry에 지정된 이름과 다른 테이블명을 새로 만들지 말고, plan 각 항목에 featureIds 배열을 기록하세요.
- Registry의 foreignKeys는 컬럼 설계에 1:1로 반영하세요. 모든 FK는 column과 references.table/references.column을 확인한 뒤 실제 컬럼 제약으로 작성해야 합니다.
- 테이블명은 영문 소문자 snake_case, 복수형 (예: users, ingredients, notifications)
- 최소 {min_tables}개 이상의 테이블을 설계하세요 (개수 상한 없음, 과도한 중복/분할은 지양)

응답 형식 (JSON만):
{{
  "plan": [
    {{"name": "테이블명", "purpose": "이 테이블이 어떤 기능을 지원하는지 한 줄 설명 (한글)", "featureIds": ["Registry의 featureId"]}}
  ]
}}

지시사항:
{instruction}

기능 목록:
{feature_str}

PM 기능 계약 Registry (각 featureId의 dbContract.tables와 foreignKeys를 빠짐없이 반영):
{feature_registry}

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

PM 기능 계약 Registry:
{feature_registry}

설계 규칙:
- 각 컬럼은 문자열 한 줄: "컬럼명 타입 제약조건들"
- ⚠️ 컬럼명은 반드시 영문 소문자 snake_case. 한글 컬럼명 절대 금지
  (예: '소모량' ❌ → quantity ⭕, '기록일시' ❌ → recorded_at ⭕, '재료_id' ❌ → ingredient_id ⭕)
- 인덱스 정의에 쓰는 컬럼명도 위에서 정의한 영문 컬럼명과 정확히 일치해야 함
- 허용 타입: BIGINT, VARCHAR(255), VARCHAR(100), VARCHAR(50), TEXT, BOOLEAN, TIMESTAMP, DATE, DECIMAL(10,2)
- 제약조건 키워드(공백 구분): PRIMARY_KEY, AUTO_INCREMENT, NOT_NULL, NULL, UNIQUE, FOREIGN_KEY, DEFAULT_FALSE, DEFAULT_TRUE
- "id BIGINT PRIMARY_KEY AUTO_INCREMENT", "created_at TIMESTAMP NOT_NULL", "updated_at TIMESTAMP NOT_NULL" 필수
- 다른 테이블을 참조하는 외래키 컬럼명은 참조테이블_id 형식 (예: user_id)
- 모든 *_id 외래키 컬럼에는 반드시 "FOREIGN_KEY REFERENCES 실제_테이블(id)"를 함께 표기하세요. relationships 배열에만 관계를 적고 컬럼 제약을 생략하지 마세요.
- Registry에 정의된 foreignKeys는 이름이 비슷하다는 이유로 생략하거나 다른 테이블로 추론하지 말고, column/reference 대상/nullable 여부를 그대로 반영하세요.
- created_by_user_id, updated_by_user_id, created_by_id, recorded_by_id, generated_by_id, checked_in_by_user_id, processed_by_user_id, actor_user_id처럼 역할 접두사가 있는 컬럼도 users(id)를 명시하세요.
- space_id처럼 실제 테이블명이 kitchen_spaces인 경우에도 전체 테이블 목록의 실제 이름을 REFERENCES 대상으로 사용하세요.
- target_id처럼 여러 테이블을 가리키는 polymorphic ID는 FOREIGN_KEY로 표시하지 말고 target_table과 함께 사용하는 이유를 description에 적으세요.
- provider_message_id처럼 외부 알림/결제 사업자가 발급한 식별자는 FOREIGN_KEY로 표시하지 말고 external_reference로 설명하세요.
- refresh_tokens.replaced_by_token_id는 refresh_tokens(id)를 참조하는 nullable self-FK로 명시하세요.
- audit_logs.target_id는 target_type + target_id 조합으로 저장하고 내부 FOREIGN_KEY는 만들지 마세요.

응답 형식 (JSON만, name/description은 위 값 그대로 유지):
{{
  "name": "{name}",
  "description": "{purpose}",
  "featureIds": ["Registry에서 이 테이블을 사용하는 featureId"],
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
PM 기능 계약 Registry:
{feature_registry}
PRD 요구사항:
{prd_document}
{missing_note}

검증 기준:
- 모든 FK 컬럼(예: user_id)이 실제 참조 대상 테이블의 PK 타입과 일치하는가? 존재하지 않는 테이블을 참조하지 않는가?
- 모든 *_id 컬럼에 실제 REFERENCES 대상이 명시되어 있고, relationships와 컬럼 제약이 서로 일치하는가?
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
    generation_blocker: str


class DbaAgent:

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

        source_registry = normalize_feature_registry(
            state.feature_registry, state.feature_list, preserve_extra=True,
        )
        registry_features = [str(item.get("name")) for item in source_registry if item.get("name")]
        scoped_features = backend_features(registry_features or state.feature_list)
        feature_str = "- " + "\n- ".join(scoped_features) if scoped_features else "(DB 대상 기능 없음)"
        registry = [item for item in source_registry if item.get("name") in scoped_features]
        graph_input: _DbaGraphState = {
            "plan": [],
            "tables": [],
            "final_tables": [],
            "relationships": [],
            "prd_issues": "",
            "ctx": {
                "instruction": state.dba_instruction or "",
                "feature_str": feature_str,
                "feature_list": scoped_features,
                "context": state.context_prompt or "",
                "feature_registry": registry_text(registry),
                "prd_document": state.prd_document or "{}",
                "dump": dump,
            },
            "generation_blocker": "",
        }

        result = await self._graph.ainvoke(graph_input)

        blocker = str(
            result.get("generation_blocker") or graph_input["ctx"].get("generation_blocker") or ""
        ).strip()
        tables = result.get("final_tables") or result.get("tables") or []
        relationships = result.get("relationships") or []
        registry_items = __import__("json").loads(graph_input["ctx"]["feature_registry"] or "[]")
        registry_items = registry_items if isinstance(registry_items, list) else []
        if blocker or not tables:
            blocker = blocker or "DBA_GENERATION_BLOCKER: upstream API 응답 후 유효한 산출물이 생성되지 않았습니다"
            tables, relationships = _registry_fallback_schema(registry_items)

        # 결정론적 커버리지 체크 — manager_plan 단계에서 이미 대부분 걸러지지만,
        # worker/manager_review 이후에도 기능이 반영 안 됐으면 targeted hint와 함께
        # manager_review를 최대 2회 더 재실행 (그래프 밖에서 직접 노드 재호출).
        # Phase2 전체 재실행(Phase3 재시도)은 feature_list 자체가 달라져 수렴하지 않으므로
        # 이 in-run 루프가 커버리지를 실질적으로 채우는 주된 수단이다.
        # 최초 전체 plan은 한 번만 확정하고, 누락분은 아래 manager_review patch로만 보정한다.
        # 전체 plan 재생성은 누락 기능이 하나 남을 때마다 모든 테이블 worker를 다시
        # 대기시키며 Phase2 timeout을 유발하므로 금지한다.
        _MAX_COVERAGE_ROUNDS = 2
        for round_ in range(0):
            missing = (
                missing_db_contract_features(registry_items, tables)
                if registry_items
                else uncovered_features(scoped_features, _tables_haystack(tables))
            )
            if not missing:
                break
            logger.warning(
                "DBA targeted patch 필요 (round %d): 미반영 기능 %d개 — %s",
                round_ + 1, len(missing), missing,
            )
            if round_ >= _MAX_COVERAGE_ROUNDS - 1:
                logger.warning("DBA targeted patch 한도 도달 — 미반영 기능은 QA blocker로 전달")
                break
            review_result = await self._manager_review_node({
                "ctx": {**graph_input["ctx"], "missing_note": missing_features_note(missing, "DB 스키마")},
                "tables": tables,
                "relationships": relationships,
            })
            tables = review_result.get("final_tables", tables)
            relationships = review_result.get("relationships", relationships)

        # PM dbContract 테이블은 LLM plan 누락과 무관하게 최소 스키마를 보장한다.
        tables, relationships = normalize_table_contract_names(
            tables, relationships, registry_items if isinstance(registry_items, list) else [],
        )
        tables = _ensure_contract_tables(tables, registry_items if isinstance(registry_items, list) else [])

        # 결정론적 백스톱: 문장형 테이블명 슬러그화 + 컬럼 0개 테이블 최소 컬럼 주입
        # (관계 합성이 테이블명을 참조하므로 그 전에 정규화)
        tables = sanitize_tables(tables)
        tables, relationships = normalize_table_contract_names(
            tables, relationships, registry_items if isinstance(registry_items, list) else [],
        )
        # 이름이 확정된 뒤에 리뷰가 덧붙인 '수정본' 중복 테이블을 병합
        tables = dedupe_meta_tables(tables)
        tables = ensure_primary_keys(tables)
        tables = annotate_table_feature_ids(
            tables, registry_items if isinstance(registry_items, list) else [],
        )
        _ensure_fk_references(tables, registry_items if isinstance(registry_items, list) else [])
        # FK 컬럼 타입을 참조 대상 PK 타입에 정합 — sanitize가 타입 추론/이름 슬러그화를
        # 끝낸 뒤에 실행해야 참조 해석과 비교 기준이 최종 상태를 반영한다.
        tables = ensure_domain_columns(tables)
        tables = enforce_schema_contracts(tables, scoped_features, state.prd_document)
        tables = reconcile_fk_types(tables)

        remaining_missing = (
            missing_db_contract_features(registry_items, tables)
            if registry_items
            else uncovered_features(scoped_features, _tables_haystack(tables))
        )
        if remaining_missing:
            # PRD rollback을 재귀적으로 유발하지 않고, Phase 3 직전 QA가 결정론적으로
            # blocker로 판정하도록 상태 산출물에만 보존한다.
            logger.warning(
                "DBA 최종 커버리지 미달 — QA blocker 전달: %s", remaining_missing,
            )

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

        mappings = build_feature_mappings(tables, state.feature_specs or registry_items)
        tables = reconcile_fk_types(tables)
        tables = _stable_schema_order(tables)
        relationships = _normalize_relationships(relationships, tables)
        clean = json.dumps({"tables": tables, "relationships": relationships, "featureMappings": mappings}, ensure_ascii=False)
        prd_issues = result.get("prd_issues", "")

        if prd_issues:
            logger.warning("DBA → PRD 피드백: %s", prd_issues)
        logger.info("DBA 에이전트 완료 — 테이블 %d개", len(tables))
        return state.copy(
            db_schema=clean,
            prd_feedback_from_dba=prd_issues,
            generation_blockers=(
                [*state.generation_blockers, blocker] if blocker else state.generation_blockers
            ),
            qa_db_blockers=(
                [*state.qa_db_blockers, blocker] if blocker else state.qa_db_blockers
            ),
            qa_approved=False if blocker else state.qa_approved,
            status_message="DBA 에이전트 완료 — DB 스키마 생성",
        )

    async def repair(self, state: PipelineState, feedback: str, dump=None) -> PipelineState:
        """기존 스키마를 보존하고 피드백에 해당하는 테이블/관계만 patch한다."""
        current = try_parse_json(state.db_schema)
        if not isinstance(current, dict) or not isinstance(current.get("tables"), list):
            blocker = "TARGETED_REPAIR_BLOCKER: DBA 기존 산출물이 없어 전체 재생성을 차단했습니다"
            logger.error(blocker)
            return state.copy(
                prd_feedback_from_dba=blocker,
                qa_db_blockers=[*state.qa_db_blockers, blocker],
                qa_approved=False,
                status_message="DBA targeted repair 차단 — 기존 산출물 없음",
            )
        prompt = TARGETED_REPAIR_PROMPT.format(
            feedback=str(feedback or "")[:8000],
            schema=json.dumps(current, ensure_ascii=False),
        )
        try:
            raw = await self._call(prompt, max_tokens=9000, system=MANAGER_SYSTEM)
            if dump:
                dump.log_raw("DBA_TARGETED_REPAIR", 1, raw)
            result = try_parse_json(raw)
            patches = result.get("patches") if isinstance(result, dict) else None
            by_name = {t.get("name"): t for t in current["tables"] if isinstance(t, dict) and t.get("name")}
            allowed_names = set(by_name)
            for item in normalize_feature_registry(
                state.feature_registry, state.feature_list, preserve_extra=True,
            ):
                contract = item.get("dbContract") or {}
                for declared in contract.get("tables", []) if isinstance(contract, dict) else []:
                    if isinstance(declared, dict):
                        declared = declared.get("name") or declared.get("table")
                    if declared:
                        allowed_names.add(str(declared))
            applied = 0
            if isinstance(patches, dict):
                for name, patch in patches.items():
                    if not isinstance(patch, dict):
                        continue
                    # Targeted repair may patch existing tables or tables
                    # explicitly declared by the Feature Registry only. Do
                    # not let blocker artifact keys such as feature_002 become
                    # accidental schema tables.
                    if str(name) not in allowed_names:
                        logger.warning("DBA targeted repair — 계약 외 테이블 patch 무시: %s", name)
                        continue
                    value = {**by_name.get(name, {}), **patch, "name": name}
                    value["columns"] = _parse_column_strings(value.get("columns"))
                    if not isinstance(value.get("indexes"), list):
                        value["indexes"] = [str(value["indexes"])] if value.get("indexes") else []
                    by_name[name] = value
                    applied += 1
            relationships = result.get("relationships")
            if not isinstance(relationships, list):
                relationships = current.get("relationships", [])
            if not applied and relationships == current.get("relationships", []):
                logger.warning("DBA targeted repair — 적용 가능한 patch 없음, 원본 유지")
                return state
            tables = ensure_primary_keys(
                dedupe_meta_tables(sanitize_tables(list(by_name.values())))
            )
            registry = normalize_feature_registry(
                state.feature_registry, state.feature_list, preserve_extra=True,
            )
            tables, relationships = normalize_table_contract_names(tables, relationships, registry)
            tables = _ensure_contract_tables(tables, registry)
            tables = annotate_table_feature_ids(tables, registry)
            _ensure_fk_references(tables, registry)
            mappings = build_feature_mappings(tables, state.feature_specs or registry)
            tables = reconcile_fk_types(tables)
            tables = _stable_schema_order(tables)
            relationships = _normalize_relationships(relationships, tables)
            repaired = json.dumps({
                **current, "tables": tables, "relationships": relationships,
                "featureMappings": mappings,
            }, ensure_ascii=False)
            logger.info("DBA targeted repair — %d개 테이블/관계 변경만 적용", applied)
            return state.copy(
                db_schema=repaired,
                prd_feedback_from_dba="",
                status_message="DBA 에이전트 완료 — 지적 테이블만 수정",
            )
        except Exception as e:
            logger.warning("DBA targeted repair 실패 — 원본 유지: %s", e)
            return state

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
                                   temperature=_PLAN_TEMPERATURE)
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
            feature_str=ctx.get("feature_str") or "- " + "\n- ".join(ctx.get("feature_list") or []),
            context=ctx["context"],
            min_tables=min_tables,
            feature_registry=ctx.get("feature_registry", "[]"),
        )

        data = None
        best: dict | None = None
        best_uncovered = None
        legacy_plain_plan = False
        for attempt in range(1):
            raw = await self._call(prompt, max_tokens=4000, system=MANAGER_SYSTEM, temperature=_PLAN_TEMPERATURE)
            if ctx.get("dump"):
                ctx["dump"].log_raw("DBA_MANAGER_PLAN", attempt + 1, raw)
            candidate = try_parse_json(raw)
            if not isinstance(candidate, dict):
                plain_plan = _parse_table_plan_text(raw)
                candidate = {"plan": plain_plan} if plain_plan else None
                legacy_plain_plan = bool(plain_plan)
            plan_list = candidate.get("plan") if candidate and isinstance(candidate.get("plan"), list) else []
            missing = uncovered_features(ctx["feature_list"], [p.get("purpose", "") for p in plan_list if isinstance(p, dict)])
            if best_uncovered is None or len(missing) < best_uncovered:
                best, best_uncovered = candidate, len(missing)
            if (
                candidate and isinstance(candidate.get("plan"), list)
                and len(candidate["plan"]) >= min_tables
                and not has_suspicious_script(candidate)
            ):
                if missing:
                    logger.warning(
                        "DBA manager_plan — 유효한 전체 plan 확정, 누락 기능은 targeted patch로 보정: %s",
                        missing,
                    )
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
        generation_blocker = ""
        registry_items = _registry_items(ctx.get("feature_registry", "[]"))
        if not data or not isinstance(data.get("plan"), list) or not data["plan"]:
            generation_blocker = (
                "DBA_GENERATION_BLOCKER: upstream API 응답 실패로 Registry 계약 기반 fallback을 사용했습니다"
            )
            logger.error("%s", generation_blocker)
            ctx["generation_blocker"] = generation_blocker
            data = {"plan": [
                {"name": name, "purpose": f"Registry 계약 fallback: {feature_id}"}
                for name, feature_id in _registry_fallback_table_names(registry_items)
            ]}

        plan = data.get("plan") or (
            self._fallback_plan(ctx["feature_list"])["plan"]
            if not generation_blocker else []
        )
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

        # Explicit registry tables remain the planning authority. For legacy
        # unregistered input, complete only omitted features without regenerating
        # a valid plan or replacing already generated domain entities.
        if legacy_plain_plan and not registry_items and not generation_blocker:
            features = ctx.get("feature_list") or []
            missing = uncovered_features(features, [str(item.get("purpose") or "") for item in plan])
            if missing:
                names = await asyncio.gather(*[
                    self._table_name_worker(feature, index, ctx.get("dump"))
                    for index, feature in enumerate(missing)
                ])
                additions = self._fallback_plan(missing, names)["plan"]
                existing = {item["name"]: item for item in plan}
                for addition in additions:
                    owner = existing.get(addition["name"])
                    if owner is None:
                        plan.append(addition)
                        existing[addition["name"]] = addition
                    elif addition["purpose"] not in str(owner.get("purpose") or ""):
                        owner["purpose"] = f"{owner.get('purpose', '')}; {addition['purpose']}".strip("; ")
            for feature in features:
                kind = feature_relation_kind(feature)
                tokens = ("record", "history", "log", "event") if kind == "event" else (
                    ("assignment", "membership", "link", "mapping") if kind == "association" else ()
                )
                if tokens and not any(
                    any(token in str(item.get("name") or "") for token in tokens)
                    and feature_relation_kind(str(item.get("purpose") or "")) == kind
                    for item in plan
                ):
                    name = await self._table_name_worker(feature, len(plan), ctx.get("dump"))
                    if not any(token in name for token in tokens) or any(item.get("name") == name for item in plan):
                        name = "event_records" if kind == "event" else "entity_assignments"
                        while any(item.get("name") == name for item in plan):
                            name += "_2"
                    plan.append({"name": name, "purpose": feature, "required_refs": []})

        logger.info("DBA manager_plan 완료 — 테이블 %d개 계획: %s", len(plan), [p.get("name") for p in plan if isinstance(p, dict)])
        return {"plan": plan, "generation_blocker": generation_blocker}

    def _fallback_plan(
        self, feature_list: list[str], names: list[str] | None = None,
        auth_required: bool = False,
    ) -> dict:
        plan = []
        seen: set[str] = set()
        root_name = names[0] if names else (_feature_table_name(feature_list[0], 0) if feature_list else "")
        if auth_required:
            plan.append({
                "name": "users", "purpose": "요구사항에 명시된 로그인 주체 식별 정보",
                "required_refs": [], "auth_principal": True,
            })
            seen.add("users")
        for i, f in enumerate(feature_list):
            name = names[i] if names and i < len(names) else _feature_table_name(f, i)
            kind = feature_relation_kind(f)
            if kind == "association" and root_name:
                name = f"{_singular_table_name(root_name)}_assignments"
            if name in seen:
                continue
            seen.add(name)
            refs: list[str] = []
            if root_name and (i == 0 or kind in {"association", "event"}):
                if i > 0:
                    refs.append(root_name)
                if auth_required:
                    refs.append("users")
            purpose = f
            if refs:
                purpose += f"; 필수 관계: {', '.join(refs)} FK"
            plan.append({"name": name, "purpose": purpose, "required_refs": refs})
        return {"plan": plan}

    def _dispatch(self, state: dict) -> list[Send]:
        if state.get("generation_blocker"):
            return []
        ctx = state["ctx"]
        all_table_names = [p.get("name") for p in state["plan"] if isinstance(p, dict)]
        sends = []
        for skeleton in state["plan"]:
            sends.append(Send("worker", {
                "skeleton": skeleton,
                "all_table_names": all_table_names,
                "context": ctx["context"],
                "feature_registry": ctx.get("feature_registry", "[]"),
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
            feature_registry=state.get("feature_registry", "[]"),
        )

        label = f"DBA_TABLE_{name}"
        data = None
        best: dict | None = None
        for attempt in range(1):
            raw = await self._call(prompt, max_tokens=1500, system=WORKER_SYSTEM)
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
                    "created_at TIMESTAMP NOT_NULL",
                    "updated_at TIMESTAMP NOT_NULL",
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

        # 전체 스키마를 manager LLM에 재전송하지 않는다. 계약 보강은 결정론적으로
        # 처리하고 남은 결함은 Phase3에서 해당 featureId만 targeted repair한다.
        registry_items = _registry_items(ctx.get("feature_registry", "[]"))
        tables = sanitize_tables(tables)
        tables, relationships = normalize_table_contract_names(
            tables, relationships, registry_items,
        )
        tables = _ensure_contract_tables(tables, registry_items)
        tables = ensure_primary_keys(tables)
        tables = annotate_table_feature_ids(tables, registry_items)
        _ensure_fk_references(tables, registry_items)
        tables = ensure_domain_columns(tables)
        tables = enforce_schema_contracts(tables, ctx.get("feature_list", []), ctx.get("prd_document", ""))
        if not registry_items:
            tables = enforce_feature_relation_contracts(
                tables, ctx.get("feature_list", []),
                requires_auth(ctx.get("prd_document", ""), ctx.get("feature_list", []), ctx.get("context", "")),
                ctx.get("prd_document", ""),
            )
        tables = reconcile_fk_types(tables)
        relationships = _normalize_relationships(relationships, tables)
        return {
            "final_tables": tables,
            "relationships": relationships,
            "prd_issues": "",
        }

    async def _call(self, user_prompt: str, max_tokens: int, system: str, temperature: float = _TEMPERATURE) -> str:
        for attempt in range(1):
            try:
                async def request():
                    return await self._client.chat.completions.create(
                        model=self._model,
                        temperature=temperature,
                        top_p=_TOP_P,
                        presence_penalty=_PRESENCE_PENALTY,
                        max_completion_tokens=max_tokens,
                        frequency_penalty=0.3,
                        messages=[
                            {"role": "system", "content": system},
                            {"role": "user", "content": user_prompt},
                        ],
                    )

                response = await self._runtime.call(request) if self._runtime else await request()
                return response.choices[0].message.content or ""
            except (InternalServerError, APITimeoutError, APIConnectionError, TimeoutError) as e:
                logger.warning("DBA API 일시 오류 (attempt %d): %s — 재시도", attempt + 1, e)
                if attempt < 2:
                    await asyncio.sleep(5 * (attempt + 1))
                else:
                    logger.error("DBA API 최종 실패")
                    return ""
        return ""

    async def _table_name_worker(self, feature: str, index: int, dump=None) -> str:
        prompt = (
            "아래 기능의 실제 데이터를 저장하는 PostgreSQL 테이블명을 영문 snake_case 복수 명사로 번역하세요. "
            "example_records, feature_N_records 같은 예시·자리표시자와 JSON은 금지합니다.\n"
            f"번역할 기능: {feature}\n첫 줄은 TABLE: 로 시작하고 마지막 줄은 SELF_CHECK: PASS로 끝내세요."
        )
        for attempt in range(2):
            raw = await self._call(prompt, max_tokens=120, system=(
                "You translate only the supplied Korean feature into one accurate PostgreSQL plural table name. "
                "Output exactly two lines: TABLE: snake_case_name and SELF_CHECK: PASS."
            ),
                                   temperature=_PLAN_TEMPERATURE)
            if dump:
                dump.log_raw(f"DBA_TABLE_NAME_{index + 1}", attempt + 1, raw)
            match = re.search(r"^\s*TABLE\s*:\s*([a-z][a-z0-9_]{1,40})\s*$", raw or "", re.I | re.M)
            name = match.group(1).lower() if match else ""
            if name and name != "example_records" and not name.startswith("feature_") and self_check_passed(raw):
                return name
            prompt += "\n이전 응답은 실제 기능 번역이 아니었습니다. 기능 의미가 드러나는 다른 이름으로 고치세요."
        return _feature_table_name(feature, index)

    async def _generate_missing_tables(
        self, tables: list[dict], missing: list[str], ctx: dict, round_: int = 0,
    ) -> list[dict]:
        """Generate concrete table specs for features missed by the first fan-out."""
        if not missing:
            return tables
        names = await asyncio.gather(*[
            self._table_name_worker(feature, index, ctx.get("dump"))
            for index, feature in enumerate(missing)
        ])
        additions = self._fallback_plan(missing, names, False)["plan"]
        existing_names = {
            str(table.get("name") or "") for table in tables if isinstance(table, dict)
        }
        additions = [item for item in additions if item.get("name") not in existing_names]
        if not additions:
            return tables
        all_names = sorted(existing_names | {str(item.get("name")) for item in additions})
        generated = await asyncio.gather(*[
            self._worker_node({
                "skeleton": skeleton,
                "all_table_names": all_names,
                "context": (
                    f"{ctx.get('context', '')}\n"
                    f"Coverage repair round {round_ + 1}; missing feature: {skeleton.get('purpose', '')}"
                ),
                "dump": ctx.get("dump"),
            })
            for skeleton in additions
        ])
        new_tables = [
            table
            for result in generated
            for table in (result.get("tables") or [])
            if isinstance(table, dict)
        ]
        return tables + new_tables


def _has_reference_path(edges: set[tuple[str, str]], start: str, goal: str) -> bool:
    pending, visited = [start], set()
    while pending:
        node = pending.pop()
        if node == goal:
            return True
        if node in visited:
            continue
        visited.add(node)
        pending.extend(target for source, target in edges if source == node)
    return False


def _break_reciprocal_fk_cycles(tables: list[dict]) -> None:
    """Remove one edge of a two-table FK cycle using physical topology only."""
    by_name = {str(table.get("name") or ""): table for table in tables if isinstance(table, dict)}
    edges: dict[tuple[str, str], list[dict]] = {}
    for source_name, table in by_name.items():
        for column in table.get("columns") or []:
            if not isinstance(column, dict):
                continue
            match = re.search(r"REFERENCES\s+([a-z][a-z0-9_]*)", str(column.get("constraints") or ""), re.I)
            if match and match.group(1).lower() in by_name:
                edges.setdefault((source_name, match.group(1).lower()), []).append(column)
    processed: set[frozenset[str]] = set()
    for (left, right), left_columns in list(edges.items()):
        pair = frozenset((left, right))
        if left == right or pair in processed or (right, left) not in edges:
            continue
        processed.add(pair)
        left_fk_count = sum(len(value) for (source, _target), value in edges.items() if source == left)
        right_fk_count = sum(len(value) for (source, _target), value in edges.items() if source == right)
        # A linking row usually has more outgoing references. Keep its edge to the
        # root; ties are resolved deterministically so the graph always converges.
        remove_source, remove_target = (
            (left, right) if left_fk_count <= right_fk_count else (right, left)
        )
        remove_ids = {id(column) for column in edges[(remove_source, remove_target)]}
        by_name[remove_source]["columns"] = [
            column for column in by_name[remove_source].get("columns") or []
            if id(column) not in remove_ids
        ]
        logger.warning("DBA 상호 FK 순환 제거 — %s → %s", remove_source, remove_target)


def _ensure_reference_column(table: dict, target: str, nullable: bool = False) -> None:
    """Ensure a physical FK contract, upgrading an existing weak `<target>_id` column."""
    name = f"{_singular_table_name(str(target))}_id"
    constraints = f"{'NULL' if nullable else 'NOT_NULL'} FOREIGN_KEY REFERENCES {target}(id)"
    for column in table.setdefault("columns", []):
        if isinstance(column, dict) and column.get("name") == name:
            column["type"] = "BIGINT"
            column["constraints"] = constraints
            return
    table["columns"].append({"name": name, "type": "BIGINT", "constraints": constraints})


def _set_column_constraints(table: dict, name: str, constraints: str) -> None:
    for column in table.get("columns") or []:
        if isinstance(column, dict) and column.get("name") == name:
            column["constraints"] = constraints
            return


def _singular_table_name(name: str) -> str:
    if name.endswith("ies"):
        return name[:-3] + "y"
    if name.endswith("s"):
        return name[:-1]
    return name


def _is_principal_table(table: dict) -> bool:
    """Identify an actor identity table from its role and identity columns."""
    name = _singular_table_name(str(table.get("name") or ""))
    columns = {
        str(column.get("name") or "")
        for column in table.get("columns") or []
        if isinstance(column, dict)
    }
    credential_fields = {"credential_hash", "password_hash", "login_id"}
    identity_fields = {"email", "username", "full_name", "display_name"}
    return bool(columns & credential_fields) or (
        name in _ACTOR_ID_BASES and bool(columns & identity_fields)
    )


def _feature_targets_principal(feature: str) -> bool:
    text = str(feature or "").lower()
    return any(token in text for token in (
        "로그인", "회원", "계정", "사용자 등록", "auth", "login", "account", "sign up",
    ))


def _ensure_index(table: dict, columns: tuple[str, ...]) -> None:
    table_name = str(table.get("name") or "")
    if not table_name or not columns:
        return
    index_name = f"idx_{table_name}_{'_'.join(columns)}"
    statement = f"CREATE INDEX {index_name} ON {table_name}({', '.join(columns)})"
    indexes = table.setdefault("indexes", [])
    normalized = {re.sub(r"\s+|;", "", str(index)).lower() for index in indexes}
    if re.sub(r"\s+|;", "", statement).lower() not in normalized:
        indexes.append(statement)


def _ensure_unique_index(table: dict, columns: tuple[str, ...]) -> None:
    if len(columns) < 2:
        return
    table_name = str(table.get("name") or "")
    statement = f"CREATE UNIQUE INDEX uq_{table_name}_{'_'.join(columns)} ON {table_name}({', '.join(columns)})"
    normalized = {re.sub(r"\s+|;", "", str(index)).lower() for index in table.setdefault("indexes", [])}
    if re.sub(r"\s+|;", "", statement).lower() not in normalized:
        table["indexes"].append(statement)


def _declares_upstream_relation(
    downstream: dict, upstream: dict, tables: list[dict], features: list[dict],
) -> bool:
    """Shared identifiers alone do not declare a business dependency."""
    upstream_features = matching_table_features(upstream, tables, features)
    for feature in matching_table_features(downstream, tables, features):
        prose = " ".join([str(feature.get("description") or ""), *map(str, feature.get("requirements") or [])])
        if not re.search(r"(?:\s후\b|이후|선행|참조|연결|종속|완료한|\b(?:after|following|requires?|references?|depends?\s+on)\b)", prose, re.I):
            continue
        if re.search(rf"\b{re.escape(str(upstream.get('name') or ''))}\b", prose, re.I):
            return True
        own_tokens = set(re.findall(r"[가-힣A-Za-z0-9]+", str(feature.get("name") or "")))
        for prior in upstream_features:
            prior_name = str(prior.get("name") or "")
            distinct = set(re.findall(r"[가-힣A-Za-z0-9]+", prior_name)) - own_tokens
            if prior_name and (prior_name in prose or any(len(token) >= 2 and token in prose for token in distinct)):
                return True
    return False


def _apply_prd_semantic_contracts(tables: list[dict], prd_document: str) -> None:
    """Project explicit PRD fields, duplicate rules and action chains into the ERD."""
    prd = try_parse_json(prd_document or "{}") or {}
    core_features = [item for item in prd.get("coreFeatures") or [] if isinstance(item, dict)]
    for feature in core_features:
        feature_name = str(feature.get("name") or "")
        content = f"{feature.get('description', '')} {' '.join(feature.get('requirements') or [])}"
        concepts = required_field_concepts(content)
        if not concepts:
            continue
        scored = [
            (relevance_score(feature_name, f"{table.get('name', '')} {table.get('description', '')}"), table)
            for table in tables
            if _feature_targets_principal(feature_name) or not _is_principal_table(table)
        ]
        best = max((score for score, _ in scored), default=0.0)
        winners = [table for score, table in scored if score == best and score >= 0.2]
        if len(winners) != 1:
            continue
        table = winners[0]
        for canonical, aliases in concepts.items():
            owner = table
            owner_name = str(owner.get("name") or "")
            role_container = any(token in owner_name for token in (
                "assigned", "assignment", "membership", "link", "mapping",
                "record", "history", "log", "event",
            ))
            if role_container and canonical in {"status", "state", "due_date", "priority"}:
                parent_names = []
                for column in owner.get("columns") or []:
                    if not isinstance(column, dict):
                        continue
                    match = re.search(r"REFERENCES\s+([a-z][a-z0-9_]*)", str(column.get("constraints") or ""), re.I)
                    if match and match.group(1).lower() in {str(t.get("name") or "") for t in tables}:
                        target = match.group(1).lower()
                        if _singular_table_name(target) not in _ACTOR_ID_BASES:
                            parent_names.append(target)
                parent_matches = [candidate for candidate in tables if candidate.get("name") in parent_names]
                if len(parent_matches) == 1:
                    owner = parent_matches[0]
            names = {str(column.get("name")) for column in owner.get("columns") or [] if isinstance(column, dict)}
            if names & aliases:
                continue
            col_type, constraints = _infer_column_type(canonical)
            _ensure_column(owner, canonical, col_type, constraints or "NULL")
            names.add(canonical)
            logger.info("DBA PRD 요구 필드 보강 — %s.%s", owner.get("name"), canonical)

    # Relationship inference below is intentionally PRD-driven.  Table names
    # alone (for example, a generic "applications" table) are not enough
    # evidence to invent business foreign keys.
    if not core_features:
        return

    auth_required = requires_auth(prd)
    for table in tables:
        table_name = str(table.get("name") or "")
        if not any(token in table_name for token in _PRIMARY_ACTION_TOKENS):
            continue
        names = {str(column.get("name")) for column in table.get("columns") or [] if isinstance(column, dict)}
        if auth_required and not any(name.endswith("_id") and name[:-3] in _ACTOR_ID_BASES for name in names):
            if any(str(candidate.get("name") or "") == "users" for candidate in tables):
                _ensure_reference_column(table, "users")
            else:
                _ensure_column(table, "user_id", "BIGINT", "NOT_NULL")
            names.add("user_id")
            logger.info("DBA 인증 행위 주체 보강 — %s.user_id", table_name)
        action_parts = {part.rstrip("s") for part in table_name.split("_")} - set(_PRIMARY_ACTION_TOKENS)
        candidates = []
        for candidate in tables:
            candidate_name = str(candidate.get("name") or "")
            if candidate is table or any(token in candidate_name for token in _PRIMARY_ACTION_TOKENS + _DOWNSTREAM_ACTION_TOKENS):
                continue
            candidate_parts = {part.rstrip("s") for part in candidate_name.split("_")}
            overlap = len(action_parts & candidate_parts)
            if overlap:
                candidates.append((overlap, candidate))
        if not candidates:
            root_candidates = []
            for candidate in tables:
                candidate_name = str(candidate.get("name") or "")
                if candidate is table or any(
                    token in candidate_name
                    for token in _PRIMARY_ACTION_TOKENS + _DOWNSTREAM_ACTION_TOKENS
                ):
                    continue
                if _singular_table_name(candidate_name) in _ACTOR_ID_BASES:
                    continue
                root_candidates.append(candidate)
            if len(root_candidates) == 1:
                candidates.append((1, root_candidates[0]))
        best = max((score for score, _ in candidates), default=0)
        winners = [candidate for score, candidate in candidates if score == best]
        if best and len(winners) == 1:
            target = winners[0]
            fk_name = f"{_singular_table_name(str(target['name']))}_id"
            _ensure_column(
                table, fk_name, "BIGINT",
                f"NOT_NULL FOREIGN_KEY REFERENCES {target['name']}(id)",
            )
            logger.info("DBA 행위 대상 보강 — %s.%s", table_name, fk_name)

    for table in tables:
        unique_columns = scoped_unique_columns(table, tables, core_features)
        if unique_columns:
            _ensure_unique_index(table, unique_columns)

    # Shared identifiers can support an explicitly requested action dependency;
    # they cannot establish that dependency on their own.
    for downstream in tables:
        down_name = str(downstream.get("name") or "")
        if not any(token in down_name for token in _DOWNSTREAM_ACTION_TOKENS):
            continue
        down_ids = {str(c.get("name")) for c in downstream.get("columns") or [] if isinstance(c, dict) and str(c.get("name", "")).endswith("_id")}
        if not down_ids:
            continue
        candidates = []
        for upstream in tables:
            up_name = str(upstream.get("name") or "")
            if upstream is downstream or any(token in up_name for token in _DOWNSTREAM_ACTION_TOKENS):
                continue
            if not _declares_upstream_relation(downstream, upstream, tables, core_features):
                continue
            up_ids = {str(c.get("name")) for c in upstream.get("columns") or [] if isinstance(c, dict) and str(c.get("name", "")).endswith("_id")}
            shared = down_ids & up_ids
            down_targets = {
                match.group(1).lower()
                for column in downstream.get("columns") or [] if isinstance(column, dict)
                if (match := re.search(
                    r"REFERENCES\s+([a-z][a-z0-9_]*)", str(column.get("constraints") or ""), re.I
                ))
            }
            up_targets = {
                match.group(1).lower()
                for column in upstream.get("columns") or [] if isinstance(column, dict)
                if (match := re.search(
                    r"REFERENCES\s+([a-z][a-z0-9_]*)", str(column.get("constraints") or ""), re.I
                ))
            }
            down_has_actor = any(name[:-3] in _ACTOR_ID_BASES for name in down_ids)
            up_has_actor = any(name[:-3] in _ACTOR_ID_BASES for name in up_ids)
            shared_target = any(name[:-3] not in _ACTOR_ID_BASES for name in shared)
            if ((shared and ((down_has_actor and up_has_actor and shared_target)
                            or (shared_target and any(token in up_name for token in _PRIMARY_ACTION_TOKENS))
                            or any(name[:-3] in _ACTOR_ID_BASES for name in shared)))
                    or (down_targets & up_targets
                        and any(token in up_name for token in _PRIMARY_ACTION_TOKENS))):
                candidates.append(upstream)
        if len(candidates) == 1:
            upstream = candidates[0]
            fk_name = f"{_singular_table_name(str(upstream['name']))}_id"
            upstream_parts = {part.rstrip("s") for part in str(upstream["name"]).split("_")}
            has_alias = any(
                name.endswith("_id")
                and (alias_parts := {part.rstrip("s") for part in name[:-3].split("_")}) <= upstream_parts
                and bool(alias_parts & set(_PRIMARY_ACTION_TOKENS))
                for name in down_ids
            )
            if not has_alias:
                _ensure_column(
                    downstream, fk_name, "BIGINT",
                    f"NOT_NULL FOREIGN_KEY REFERENCES {upstream['name']}(id)",
                )
                logger.info("DBA 선행·후속 행위 연결 보강 — %s.%s", down_name, fk_name)


def _dedupe_columns(table: dict) -> None:
    """Collapse repeated identifier fragments and duplicate column definitions."""
    columns = table.get("columns")
    if not isinstance(columns, list):
        return
    kept: list[dict] = []
    by_name: dict[str, dict] = {}
    renames: list[tuple[str, str]] = []
    dropped: list[str] = []
    for column in columns:
        if not isinstance(column, dict) or not column.get("name"):
            continue
        old_name = str(column["name"])
        pieces = old_name.split("_")
        collapsed = [
            piece for index, piece in enumerate(pieces)
            if index == 0 or piece != pieces[index - 1]
        ]
        name = "_".join(collapsed)
        if name != old_name:
            column["name"] = name
            renames.append((old_name, name))
        if name in by_name:
            existing = by_name[name]
            if not str(existing.get("constraints") or "") and str(column.get("constraints") or ""):
                existing["constraints"] = column["constraints"]
            dropped.append(name)
            continue
        by_name[name] = column
        kept.append(column)
    entity_prefix = f"{_singular_table_name(str(table.get('name') or ''))}_"
    prefixed_renames = []
    if entity_prefix != "_":
        names = {str(column.get("name") or "") for column in kept}
        for column in list(kept):
            name = str(column.get("name") or "")
            suffix = name.removeprefix(entity_prefix)
            if name.startswith(entity_prefix) and not name.endswith("_id") and suffix in names:
                kept.remove(column)
                prefixed_renames.append((name, suffix))
                dropped.append(name)
    renames.extend(prefixed_renames)
    table["columns"] = kept
    if renames and isinstance(table.get("indexes"), list):
        table["indexes"] = [_rename_in_text(str(index), renames) for index in table["indexes"]]
    if dropped:
        logger.warning(
            "DBA 테이블 '%s' 중복 컬럼 %d개 제거: %s",
            table.get("name"), len(dropped), dropped,
        )


def _feature_table_name(feature: str, index: int) -> str:
    text = str(feature or "")
    ascii_name = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return ascii_name[:36] if ascii_name else f"feature_{index + 1}_records"


_ACTOR_ID_BASES = {
    "user", "student", "learner", "customer", "member", "owner", "instructor",
    "teacher", "employee", "staff", "assignee", "participant", "attendee",
    "patient", "provider", "author", "creator", "requester", "applicant", "personnel",
    "volunteer",
}


_DOWNSTREAM_ACTION_TOKENS = (
    "record", "history", "log", "return", "completion", "attendance", "payment",
    "shipment", "delivery", "approval", "resolution", "inspection", "check",
    "verification", "notification", "alert", "reminder",
)


_PRIMARY_ACTION_TOKENS = (
    "reservation", "booking", "rental", "loan", "borrow", "order", "application",
    "request", "ticket", "assignment", "enrollment",
)


_MUTABLE_STATE_COLUMNS = {
    "quantity", "amount", "balance", "status", "state", "unit", "expiry_date",
    "expires_at", "scheduled_at", "position", "progress", "stock", "remaining_count",
}


_VALID_COLUMN_NAME_RE = re.compile(r'^[a-z][a-z0-9_]*$')
