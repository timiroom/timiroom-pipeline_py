"""Executable repair allowlists; prompts alone do not protect sibling artifacts."""

import copy
import hashlib
import json
import re


def artifact_hash(document: dict) -> str:
    return hashlib.sha256(json.dumps(document, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def endpoint_key(endpoint: dict) -> str:
    return f"{str(endpoint.get('method', 'GET')).upper()} {endpoint.get('path', '')}"


def repair_targets(state, feedback: str, domain: str, current: dict) -> set[str]:
    """Prefer structured targets; conservatively recognize exact legacy artifact names."""
    text = str(feedback or "")
    details = [*state.qa_repair_issues, *state.qa_blocker_details, *state.validation_blockers]
    for detail in details:
        if isinstance(detail, dict):
            label = str(detail.get("domain") or detail.get("agent") or "").lower()
            if label in {domain, "dba" if domain == "db" else domain}:
                text += "\n" + json.dumps(detail, ensure_ascii=False)
    objects = []
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", text):
        try:
            value, _ = decoder.raw_decode(text[match.start():])
            if isinstance(value, dict):
                objects.append(value)
        except ValueError:
            continue
    feature_ids = set()
    for obj in objects:
        values = obj.get("changedFeatureIds") or obj.get("featureIds") or []
        feature_ids.update(str(value) for value in values if isinstance(value, str))
        if obj.get("featureId"):
            feature_ids.add(str(obj["featureId"]))
    registry = [item for item in state.feature_registry if isinstance(item, dict)]
    for feature in registry:
        key = str(feature.get("featureId") or feature.get("id") or "")
        name = str(feature.get("name") or "")
        if (key and key in text) or (name and name in text):
            feature_ids.add(key)
    if domain == "prd":
        # A missing top-level section can be added, but unknown sections cannot.
        from phase2.agents.prd_agent import SECTION_PROMPTS
        keys = set(current) | set(SECTION_PROMPTS) | {"coreFeatures", "goals", "mvpScope"}
        targets = {key for key in keys if re.search(rf"(?<![A-Za-z0-9_]){re.escape(key)}(?![A-Za-z0-9_])", text)}
        if feature_ids:
            targets.add("coreFeatures")
        return targets
    if domain == "pm":
        return {name for name in state.feature_list if name and name in text}
    if domain == "feature":
        return feature_ids
    if domain == "api":
        known_routes = {endpoint_key(endpoint) for endpoint in current.get("endpoints") or [] if isinstance(endpoint, dict)}
        known_routes.update(endpoint_key(operation) for feature in registry
                            for operation in feature.get("apiContract") or [] if isinstance(operation, dict))
        targets = {f"{method.upper()} {path}" for method, path in re.findall(
            r"\b(GET|POST|PATCH|PUT|DELETE|HEAD|OPTIONS)\s+(/[A-Za-z0-9_/{}/.\-]+)", text, re.IGNORECASE,
        )}
        for obj in objects:
            if obj.get("artifactKey") in known_routes:
                targets.add(obj["artifactKey"])
            match = re.match(r"^/endpoints/(\d+)(?:/|$)", str(obj.get("pointer") or ""))
            if match and int(match.group(1)) < len(current.get("endpoints") or []):
                endpoint = current["endpoints"][int(match.group(1))]
                if isinstance(endpoint, dict):
                    targets.add(endpoint_key(endpoint))
        for endpoint in current.get("endpoints") or []:
            if set(endpoint.get("featureIds") or [endpoint.get("featureId")]) & feature_ids:
                targets.add(endpoint_key(endpoint))
        for feature in registry:
            if str(feature.get("featureId") or feature.get("id")) in feature_ids:
                targets.update(endpoint_key(operation) for operation in feature.get("apiContract") or [] if isinstance(operation, dict))
        return targets & known_routes
    names = {str(table.get("name")) for table in current.get("tables") or [] if isinstance(table, dict)}
    for feature in registry:
        declared = (feature.get("dbContract") or {}).get("tables") or []
        names.update(str(name.get("name") or name.get("table")) if isinstance(name, dict) else str(name) for name in declared)
    targets = {name for name in names if re.search(rf"(?<![A-Za-z0-9_]){re.escape(name)}(?![A-Za-z0-9_])", text)}
    for obj in objects:
        match = re.match(r"^/tables/(\d+)(?:/|$)", str(obj.get("pointer") or ""))
        if match and int(match.group(1)) < len(current.get("tables") or []):
            table = current["tables"][int(match.group(1))]
            if isinstance(table, dict) and table.get("name"):
                targets.add(table["name"])
    for feature in registry:
        if str(feature.get("featureId") or feature.get("id")) in feature_ids:
            for name in (feature.get("dbContract") or {}).get("tables") or []:
                targets.add(str(name.get("name") or name.get("table")) if isinstance(name, dict) else str(name))
    return targets


def scope_prompt(current: dict, targets: set[str]) -> str:
    return "\n[REPAIR_SCOPE]\n" + json.dumps({
        "baseHash": artifact_hash(current), "allowedArtifactKeys": sorted(targets),
        "instruction": "Only patch allowed keys. Echo baseHash. Preserve unmentioned fields/columns. Use op=remove for explicit deletion.",
    }, ensure_ascii=False)


def valid_base(result, current: dict) -> bool:
    return isinstance(result, dict) and result.get("baseHash", artifact_hash(current)) == artifact_hash(current)


def preserve_unscoped(original: list, candidate: list, allowed: set[str], key) -> list:
    by_key = {key(item): item for item in candidate if isinstance(item, dict)}
    result = [copy.deepcopy(by_key.get(key(item), item) if key(item) in allowed else item)
              for item in original if key(item) not in allowed or key(item) in by_key]
    old_keys = {key(item) for item in original}
    result.extend(copy.deepcopy(item) for item in candidate if key(item) in allowed - old_keys)
    return result


def merge_columns(original: list, patches: list) -> list:
    result = {item["name"]: copy.deepcopy(item) for item in original if isinstance(item, dict) and item.get("name")}
    for patch in patches:
        name = patch.get("name")
        if not name:
            continue
        if patch.get("op") == "remove":
            result.pop(name, None)
        else:
            result[name] = {**result.get(name, {}), **{key: value for key, value in patch.items() if key != "op"}}
    return list(result.values())
