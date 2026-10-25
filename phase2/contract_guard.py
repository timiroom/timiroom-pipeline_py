"""Validate generated contracts before consumers can invent physical resources."""

import re

from phase2.feature_scope import artifact_required


def semantic_identifier(value: str) -> bool:
    parts = str(value or "").split(".")
    return (
        len(parts) >= 2
        and all(re.fullmatch(r"[a-z][a-z0-9_]*", part) for part in parts)
        and parts[0] not in {"intent", "supporting", "feature"}
        and not any(re.fullmatch(r"feature_?\d+|[a-f0-9]{8,}s?", part) for part in parts)
    )


def feature_contract_issues(features, *, feature_pointer=None) -> list[str]:
    if not isinstance(features, list) or not features:
        return ["features: non-empty array required"]
    issues = []
    seen = set()
    for index, feature in enumerate(features):
        if not isinstance(feature, dict):
            issues.append(f"features[{index}]: object required")
            continue
        key = str(feature.get("featureId") or feature.get("id") or "")
        prefix = key or f"features[{index}]"
        if not semantic_identifier(key) or key in seen:
            issues.append(f"{prefix}: semantic unique featureId required")
        seen.add(key)
        if not all(isinstance(feature.get(field), str) and feature[field].strip() for field in ("name", "description")):
            issues.append(f"{prefix}: name/description required")
        requirements = feature.get("requirements")
        if not isinstance(requirements, list) or not requirements or not all(
            (isinstance(item, str) and bool(item.strip())) or (isinstance(item, dict) and bool(item)) for item in requirements
        ):
            issues.append(f"{prefix}: requirements required")
        operations = feature.get("apiContract")
        db = feature.get("dbContract")
        if not artifact_required(feature, "api"):
            if operations:
                issues.append(f"{prefix}: FEATURE_ARTIFACT_SCOPE_CONFLICT api=false")
            operations = []
        elif not isinstance(operations, list) or not operations:
            issues.append(f"{prefix}: apiContract required")
        else:
            for operation_index, operation in enumerate(operations):
                operation_prefix = f"{prefix}: {feature_pointer or f'/features/{index}'}/apiContract/{operation_index}"
                if not isinstance(operation, dict):
                    issues.append(f"{operation_prefix}: operation object required")
                    continue
                path = str(operation.get("path") or "")
                if operation.get("method") not in {"GET", "POST", "PATCH", "PUT", "DELETE", "HEAD", "OPTIONS"} or not path.startswith("/api/v1/"):
                    issues.append(f"{operation_prefix}: canonical method/path required")
                if any(re.fullmatch(r"feature_?\d+.*|[a-f0-9]{8,}s?", part) for part in path.split("/")):
                    issues.append(f"{operation_prefix}: placeholder resource forbidden")
                if not (isinstance(operation.get("responseSchema"), dict) and operation["responseSchema"]) and str(operation.get("successStatus")) != "204":
                    issues.append(f"{operation_prefix}: responseSchema required")
        if not artifact_required(feature, "db"):
            if isinstance(db, dict) and db.get("tables"):
                issues.append(f"{prefix}: FEATURE_ARTIFACT_SCOPE_CONFLICT db=false")
            continue
        if not isinstance(db, dict):
            issues.append(f"{prefix}: dbContract required")
            continue
        tables = db.get("tables")
        if not isinstance(tables, list) or not tables or not all(isinstance(name, str) and re.fullmatch(r"[a-z][a-z0-9_]*", name) for name in tables):
            issues.append(f"{prefix}: canonical tables required")
        elif db.get("primaryTable") not in tables:
            issues.append(f"{prefix}: primaryTable must be declared")
        if isinstance(tables, list) and any(isinstance(name, str) and re.fullmatch(r"feature_?\d+.*|[a-f0-9]{8,}s?", name) for name in tables):
            issues.append(f"{prefix}: placeholder table forbidden")
        fields = db.get("fields")
        if not isinstance(fields, list) or not fields or not all(
            isinstance(field, dict) and field.get("table") in (tables or [])
            and field.get("name") and field.get("type") for field in fields
        ):
            issues.append(f"{prefix}: typed DB fields required")
        if feature.get("sessionPolicy") is not None and not isinstance(feature["sessionPolicy"], dict):
            issues.append(f"{prefix}: sessionPolicy object required")
    return issues
