"""Feature Spec 기능을 PRD/API/DBA가 공유하는 안정적인 계약으로 정규화한다."""

from __future__ import annotations

import hashlib
import re
from typing import Any

from phase2.agent_contract import (
    canonical_api_method,
    endpoint_feature_ids,
    normalize_api_path,
)
from phase2.feature_scope import artifact_required

_ACTION_TERMS = (
    ("생성", "create"), ("등록", "create"), ("조회", "read"), ("검색", "read"),
    ("수정", "update"), ("변경", "update"), ("삭제", "delete"), ("취소", "cancel"),
    ("신청", "apply"), ("확정", "confirm"), ("승인", "approve"),
    ("기록", "record"), ("계산", "calculate"), ("다운로드", "download"),
)


def _normalize_table_refs(value: Any) -> list[str]:
    """LLM이 tables를 문자열 또는 {name: ...} 객체로 섞어 출력해도 이름 계약으로 통일한다."""
    if not isinstance(value, list):
        return []
    names: list[str] = []
    for table in value:
        if isinstance(table, dict):
            table = table.get("name") or table.get("table")
        name = str(table or "").strip()
        generic = re.fullmatch(r"feature_?(\d+)s?", name.casefold())
        if generic:
            name = f"feature_{int(generic.group(1)):03d}_records"
        if name and name not in names:
            names.append(name)
    return names


def _normalize_api_contracts(value: Any, feature_id: str) -> list[dict[str, Any]]:
    """Normalize contract paths before any downstream agent compares them."""
    if not isinstance(value, list):
        return []
    contracts: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for raw in value:
        if not isinstance(raw, dict):
            continue
        path = normalize_api_path(raw.get("path"))
        if not path or path == "/api/v1":
            continue
        method = canonical_api_method(raw.get("method"), path)
        key = (method, path)
        if key in seen:
            continue
        item = dict(raw)
        item.update({"method": method, "path": path, "featureId": feature_id})
        item["action"] = str(item.get("action") or "manage").strip()
        contracts.append(item)
        seen.add(key)
    return contracts


def _actions(name: str) -> list[str]:
    found: list[str] = []
    for term, action in _ACTION_TERMS:
        if term in name and action not in found:
            found.append(action)
    return found or ["manage"]


def canonical_feature_id(item: dict) -> str:
    """Preserve semantic IDs and replace positional IDs from physical contracts."""
    current = str(item.get("featureId") or item.get("id") or "").strip()
    generic = re.match(r"^(?:feature[_.-]?\d+|supporting(?:\.|$)|intent\.)", current)
    if current and not generic and re.fullmatch(r"[a-z][a-z0-9_-]*(?:\.[a-z][a-z0-9_-]*)+", current):
        return current
    contracts = item.get("apiContract") or item.get("api") or []
    actions = item.get("actions") or _actions(str(item.get("name") or ""))
    action = str(actions[0] if actions else "manage").lower()
    ignored = {"api", "v1", "v2", "validate", "validation", "uploads", "transitions", "search"}
    ignored.update(str(value) for value in actions)
    resource = ""
    for contract in contracts if isinstance(contracts, list) else []:
        if not isinstance(contract, dict):
            continue
        parts = [part for part in str(contract.get("path") or "").split("/")
                 if part and not part.startswith("{") and part not in ignored]
        if parts:
            resource = parts[-1].replace("-", "_")
            break
    db = item.get("dbContract") or item.get("db") or {}
    if not resource and isinstance(db, dict):
        refs = _normalize_table_refs(db.get("tables"))
        resource = next((name for name in refs if not re.match(r"feature_\d+", name)), "")
    digest = hashlib.sha256(str(item.get("name") or "").encode("utf-8")).hexdigest()[:10]
    if not re.fullmatch(r"[a-z][a-z0-9_]*", resource) or re.match(r"^(?:feature[_-]?\d+|supporting|intent)", resource):
        # Before Feature Spec supplies English resource contracts, use a stable
        # internal identity rather than an order-dependent feature_001 counter.
        return f"intent.{digest}.{action}"
    action = re.sub(r"[^a-z0-9_]+", "_", action).strip("_") or "manage"
    return f"{resource}.{action}"


def build_feature_registry(features: list[str] | None) -> list[dict[str, Any]]:
    """문자열 기능 목록을 deterministic registry로 변환한다."""
    registry = []
    for name in features or []:
        if not isinstance(name, str) or not name.strip():
            continue
        feature_id = canonical_feature_id({"name": name, "actions": _actions(name)})
        registry.append({
            "id": feature_id,
            "featureId": feature_id,
            "name": name.strip(),
            "actions": _actions(name),
            "apiContract": [],
            "dbContract": {"tables": [], "foreignKeys": []},
        })
    return registry


def normalize_feature_registry(
    raw: Any,
    features: list[str] | None,
    *,
    preserve_extra: bool = False,
) -> list[dict[str, Any]]:
    """registry를 기준 기능 목록에 맞춘다.

    기본값은 기존 PM/PRD 호환을 위해 기준 목록과 1:1로 맞춘다. Feature Spec 이후의
    API/DBA/QA 단계에서는 supporting 기능도 계약의 일부이므로 ``preserve_extra``로
    registry에만 존재하는 항목을 보존한다.
    """
    source = raw if isinstance(raw, list) else []
    by_name = {
        str(item.get("name", "")).strip(): item
        for item in source if isinstance(item, dict) and str(item.get("name", "")).strip()
    }
    fallback = build_feature_registry(features)
    normalized = []
    used_ids: set[str] = set()
    for index, item in enumerate(fallback):
        candidate = by_name.get(item["name"], {})
        value = {**item, **candidate}
        value["id"] = canonical_feature_id(value)
        if not re.match(r"^[a-z][a-z0-9_.-]{1,80}$", value["id"]):
            value["id"] = item["id"]
        if value["id"] in used_ids:
            digest = hashlib.sha256(item["name"].encode("utf-8")).hexdigest()[:8]
            value["id"] = f"{value['id']}_{digest}"
        suffix = 2
        base_id = value["id"]
        while value["id"] in used_ids:
            value["id"] = f"{base_id}.{suffix}"
            suffix += 1
        used_ids.add(value["id"])
        value["featureId"] = value["id"]
        value["name"] = item["name"]
        value["actions"] = [str(a) for a in value.get("actions", item["actions"]) if str(a).strip()] or item["actions"]
        value["roles"] = [str(role) for role in value.get("roles", []) if str(role).strip()]
        api = value.get("apiContract", value.get("api"))
        value["apiContract"] = _normalize_api_contracts(api, value["id"])
        db = value.get("dbContract", value.get("db"))
        db = db if isinstance(db, dict) else {}
        value["dbContract"] = {
            **db,
            "tables": _normalize_table_refs(db.get("tables")),
            "foreignKeys": db.get("foreignKeys") if isinstance(db.get("foreignKeys"), list) else [],
        }
        value["api"] = value["apiContract"]
        value["db"] = value["dbContract"]
        normalized.append(value)
    if preserve_extra:
        known_names = {item["name"] for item in normalized}
        for item in source:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name") or "").strip()
            if not name or name in known_names:
                continue
            value = dict(item)
            value["name"] = name
            value["id"] = canonical_feature_id(value)
            if not re.match(r"^[a-z][a-z0-9_.-]{1,80}$", value["id"]) or value["id"] in used_ids:
                value["id"] = canonical_feature_id({**value, "featureId": "", "id": ""})
                suffix = 2
                base_id = value["id"]
                while value["id"] in used_ids:
                    value["id"] = f"{base_id}.{suffix}"
                    suffix += 1
            value["featureId"] = value["id"]
            value["actions"] = [str(a) for a in value.get("actions", ["manage"]) if str(a).strip()] or ["manage"]
            value["roles"] = [str(role) for role in value.get("roles", []) if str(role).strip()]
            api = value.get("apiContract", value.get("api"))
            value["apiContract"] = _normalize_api_contracts(api, value["id"])
            db = value.get("dbContract", value.get("db"))
            db = db if isinstance(db, dict) else {}
            value["dbContract"] = {
                **db,
                "tables": _normalize_table_refs(db.get("tables")),
                "foreignKeys": db.get("foreignKeys") if isinstance(db.get("foreignKeys"), list) else [],
            }
            value["api"] = value["apiContract"]
            value["db"] = value["dbContract"]
            normalized.append(value)
            known_names.add(name)
            used_ids.add(value["id"])
    id_map = {
        str(original.get("featureId") or original.get("id") or ""): item["featureId"]
        for item in normalized for original in source
        if isinstance(original, dict) and original.get("name") == item["name"]
    }
    for item in normalized:
        parent = item.get("parentFeatureId")
        if parent in id_map:
            item["parentFeatureId"] = id_map[parent]
    # 여러 기능이 같은 사용자 작업을 공유할 수 있다. 예를 들어 목록 조회는
    # 검색·필터·페이지네이션 기능을 동시에 구현한다. Registry에서 한 소유자만
    # 남기면 지식 그래프가 끊기므로 중복 route 계약을 그대로 보존하고 API
    # 산출 단계에서 endpoint.featureIds로 병합한다.
    return normalized


def registry_text(registry: list[dict[str, Any]] | None) -> str:
    import json
    return json.dumps(registry or [], ensure_ascii=False, indent=2)


def missing_db_contract_features(
    registry: list[dict[str, Any]] | None,
    tables: list[dict[str, Any]] | None,
) -> list[str]:
    """Return features whose declared DB tables are absent from the final schema.

    This is intentionally contract-based. Feature names are user-facing prose and
    are not reliable identifiers for matching ``equipment`` to a Korean feature.
    """
    table_names = {
        str(table.get("name") or "").strip().casefold()
        for table in tables or [] if isinstance(table, dict) and table.get("name")
    }
    missing: list[str] = []
    for item in registry or []:
        if not isinstance(item, dict):
            continue
        if not artifact_required(item, "db"):
            continue
        contract = item.get("dbContract") or item.get("db") or {}
        if not isinstance(contract, dict):
            continue
        required = _normalize_table_refs(contract.get("tables"))
        if not required:
            continue
        feature_id = str(item.get("featureId") or item.get("id") or "").strip()
        if any(name.casefold() not in table_names for name in required):
            missing.append(feature_id or str(item.get("name") or "unknown"))
    return missing


def missing_api_contract_features(
    registry: list[dict] | None,
    endpoints: list[dict] | None,
) -> list[str]:
    """Return feature IDs whose declared API contracts are absent from the final spec."""
    endpoint_keys: dict[str, set[tuple[str, str]]] = {}
    unscoped_keys: set[tuple[str, str]] = set()
    for endpoint in endpoints or []:
        if not isinstance(endpoint, dict) or not endpoint.get("path"):
            continue
        key = (
            canonical_api_method(endpoint.get("method"), endpoint.get("path")),
            _normalize_api_path(endpoint.get("path")),
        )
        feature_ids = endpoint_feature_ids(endpoint)
        if not feature_ids:
            unscoped_keys.add(key)
        for feature_id in feature_ids:
            endpoint_keys.setdefault(feature_id, set()).add(key)
    missing: list[str] = []
    for item in registry or []:
        if not isinstance(item, dict):
            continue
        if not artifact_required(item, "api"):
            continue
        feature_id = str(item.get("featureId") or item.get("id") or "").strip()
        contracts = item.get("apiContract") or item.get("api") or []
        if not feature_id or not isinstance(contracts, list):
            continue
        for contract in contracts:
            if not isinstance(contract, dict):
                continue
            key = (
                canonical_api_method(contract.get("method"), contract.get("path")),
                _normalize_api_path(contract.get("path")),
            )
            if (
                key[1]
                and key not in endpoint_keys.get(feature_id, set())
                and key not in unscoped_keys
                and feature_id not in missing
            ):
                missing.append(feature_id)
                break
    return missing


def _normalize_api_path(path: Any) -> str:
    value = "/" + str(path or "").strip().lstrip("/")
    value = re.sub(r"^/api/v1", "", value, flags=re.IGNORECASE)
    return "/api/v1" + (value if value.startswith("/") else "/" + value)
