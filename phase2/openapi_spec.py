"""Deterministic OpenAPI 3.1 projection for the internal API contract."""

from __future__ import annotations

import re
from typing import Any

from phase2.agent_contract import endpoint_feature_ids, normalize_api_path
from phase2.json_utils import try_parse_json

_METHODS = {"get", "post", "put", "patch", "delete", "head", "options"}
_BODYLESS = {"GET", "DELETE", "HEAD"}
_FIELD_RE = re.compile(
    r"(?:^|[,;])\s*([A-Za-z_][A-Za-z0-9_]*)\s*:\s*([^,;—]+)(?:\s*—\s*([^,;]+))?"
)
_STATUS_RE = re.compile(r"\b([1-5]\d{2})\b\s*(?:—|-|:)?\s*([^,;]*)")
_PATH_PARAM_RE = re.compile(r"\{([A-Za-z_][A-Za-z0-9_-]*)\}")


def _schema_for_type(raw: str) -> dict[str, Any]:
    value = str(raw or "string").strip().casefold()
    if "array" in value or "배열" in value or "list" in value:
        return {"type": "array", "items": {"type": "object"}}
    if any(token in value for token in ("integer", "int", "long", "정수")):
        return {"type": "integer", "format": "int64"}
    if any(token in value for token in ("number", "decimal", "float", "숫자", "실수")):
        return {"type": "number"}
    if any(token in value for token in ("boolean", "bool", "불리언", "여부")):
        return {"type": "boolean"}
    if any(token in value for token in ("object", "객체")):
        return {"type": "object"}
    return {"type": "string"}


def _object_schema(contract: Any) -> dict[str, Any]:
    if isinstance(contract, dict):
        return contract if "type" in contract or "$ref" in contract else {
            "type": "object",
            "additionalProperties": True,
        }
    text = str(contract or "").strip()
    properties: dict[str, dict[str, Any]] = {}
    for match in _FIELD_RE.finditer(text):
        schema = _schema_for_type(match.group(2))
        if match.group(3):
            schema["description"] = match.group(3).strip()
        properties[match.group(1)] = schema
    if properties:
        return {"type": "object", "properties": properties}
    return {"type": "object", "additionalProperties": True}


def _parameter_schema(parameter: dict) -> dict[str, Any]:
    result = {
        "name": str(parameter.get("name") or "parameter"),
        "in": str(parameter.get("in") or "query").lower(),
        "required": bool(parameter.get("required", False)),
        "schema": _schema_for_type(parameter.get("type")),
    }
    if result["in"] == "path":
        result["required"] = True
    if str(parameter.get("description") or "").strip():
        result["description"] = str(parameter["description"]).strip()
    return result


def _operation_id(endpoint: dict, used: set[str]) -> str:
    method = str(endpoint.get("method") or "GET").upper()
    path = normalize_api_path(endpoint.get("path"))
    action = str(endpoint.get("action") or "").strip().casefold()
    default_action = {
        "GET": "get" if "{" in path else "list",
        "POST": "create",
        "PUT": "replace",
        "PATCH": "update",
        "DELETE": "delete",
    }.get(method, method.casefold())
    action = re.sub(r"[^a-z0-9]+", " ", action).strip() or default_action
    parts = [part for part in path.removeprefix("/api/v1/").split("/") if part]
    words = action.split()
    for part in parts:
        if part.startswith("{"):
            words.extend(("by", part.strip("{}").replace("_", " ")))
        else:
            words.append(part.replace("-", " "))
    flattened = [word for group in words for word in str(group).split() if word]
    operation_id = (flattened[0].casefold() if flattened else default_action) + "".join(
        word[:1].upper() + word[1:] for word in flattened[1:]
    )
    base = re.sub(r"[^A-Za-z0-9_]", "", operation_id) or "operation"
    candidate = base
    suffix = 2
    while candidate in used:
        candidate = f"{base}{suffix}"
        suffix += 1
    used.add(candidate)
    return candidate


def _responses(endpoint: dict) -> dict[str, Any]:
    method = str(endpoint.get("method") or "GET").upper()
    success_code = "201" if method == "POST" else "204" if method == "DELETE" else "200"
    responses: dict[str, Any] = {
        success_code: {
            "description": "요청 성공",
        }
    }
    if success_code != "204":
        responses[success_code]["content"] = {
            "application/json": {"schema": _object_schema(endpoint.get("successResponse"))}
        }
    for match in _STATUS_RE.finditer(str(endpoint.get("errorCodes") or "")):
        code = match.group(1)
        responses.setdefault(code, {"description": match.group(2).strip() or "요청 처리 실패"})
    return responses


def build_openapi_document(
    api_spec: str | dict,
    project_name: str = "Generated API",
    *,
    server_url: str = "/",
) -> dict[str, Any]:
    """Project the internal endpoint list into an importable OpenAPI 3.1 document."""
    internal = try_parse_json(api_spec) if isinstance(api_spec, str) else api_spec
    internal = internal if isinstance(internal, dict) else {}
    document: dict[str, Any] = {
        "openapi": "3.1.0",
        "info": {"title": f"{project_name or 'Generated'} API", "version": "1.0.0"},
        "servers": [{"url": server_url or "/"}],
        "paths": {},
        "components": {
            "schemas": {},
            "securitySchemes": {
                "bearerAuth": {"type": "http", "scheme": "bearer", "bearerFormat": "JWT"}
            },
        },
    }
    used_operation_ids: set[str] = set()
    for endpoint in internal.get("endpoints") or []:
        if not isinstance(endpoint, dict):
            continue
        method = str(endpoint.get("method") or "GET").casefold()
        path = normalize_api_path(endpoint.get("path"))
        if method not in _METHODS or path == "/api/v1":
            continue
        feature_ids = endpoint_feature_ids(endpoint)
        resource = next((part for part in path.removeprefix("/api/v1/").split("/") if not part.startswith("{")), "api")
        operation: dict[str, Any] = {
            "operationId": _operation_id(endpoint, used_operation_ids),
            "summary": str(endpoint.get("summary") or endpoint.get("description") or f"{method.upper()} {path}"),
            "description": str(endpoint.get("description") or ""),
            "tags": [resource.replace("-", " ").title()],
            "parameters": [
                _parameter_schema(parameter)
                for parameter in endpoint.get("parameters") or []
                if isinstance(parameter, dict)
            ],
            "responses": _responses(endpoint),
            "x-feature-ids": feature_ids,
        }
        if endpoint.get("authRequired"):
            operation["security"] = [{"bearerAuth": []}]
        request_body = str(endpoint.get("requestBody") or "").strip()
        if method.upper() not in _BODYLESS and request_body and request_body != "없음":
            operation["requestBody"] = {
                "required": True,
                "content": {"application/json": {"schema": _object_schema(endpoint.get("requestBody"))}},
            }
        document["paths"].setdefault(path, {})[method] = operation
    return document


def validate_openapi_document(document: dict | None) -> list[str]:
    """Validate the deterministic subset emitted by build_openapi_document."""
    if not isinstance(document, dict):
        return ["OpenAPI 문서가 객체가 아닙니다"]
    issues: list[str] = []
    if not str(document.get("openapi") or "").startswith("3.1."):
        issues.append("openapi 버전은 3.1.x여야 합니다")
    info = document.get("info")
    if not isinstance(info, dict) or not info.get("title") or not info.get("version"):
        issues.append("info.title/info.version이 필요합니다")
    paths = document.get("paths")
    if not isinstance(paths, dict) or not paths:
        issues.append("paths가 비어 있습니다")
        return issues
    operation_ids: set[str] = set()
    for path, item in paths.items():
        if normalize_api_path(path) != path:
            issues.append(f"정규화되지 않은 OpenAPI path: {path}")
        if not isinstance(item, dict):
            issues.append(f"path item이 객체가 아닙니다: {path}")
            continue
        for method, operation in item.items():
            if method not in _METHODS or not isinstance(operation, dict):
                issues.append(f"유효하지 않은 operation: {method.upper()} {path}")
                continue
            operation_id = str(operation.get("operationId") or "")
            if not operation_id or operation_id in operation_ids:
                issues.append(f"operationId 누락 또는 중복: {method.upper()} {path}")
            operation_ids.add(operation_id)
            if not isinstance(operation.get("responses"), dict) or not operation["responses"]:
                issues.append(f"responses 누락: {method.upper()} {path}")
            declared = {
                str(parameter.get("name") or "")
                for parameter in operation.get("parameters") or []
                if isinstance(parameter, dict) and parameter.get("in") == "path"
            }
            missing = set(_PATH_PARAM_RE.findall(path)) - declared
            if missing:
                issues.append(f"path parameter 누락: {method.upper()} {path} {sorted(missing)}")
    return list(dict.fromkeys(issues))
