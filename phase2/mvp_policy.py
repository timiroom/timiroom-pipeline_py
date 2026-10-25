"""One release/priority policy for PRD and Feature Spec."""

MAX_MVP_FEATURES = 12
MAX_P0_FEATURES = 5


def dependency_issues(features):
    by_id = {str(item.get("featureId") or item.get("id") or item.get("name")): item for item in features}
    issues = []
    def visit(key, path, hierarchy=False):
        prefix = "FEATURE_PARENT" if hierarchy else "FEATURE_DEPENDENCY"
        if key in path:
            issues.append(f"{prefix}_CYCLE: {key}")
            return
        item = by_id[key]
        dependencies = ([item["parentFeatureId"]] if item.get("parentFeatureId") else []) if hierarchy else list(item.get("dependencies") or [])
        for dependency in dependencies:
            if dependency not in by_id:
                issues.append(f"{prefix}_MISSING: {key} -> {dependency}")
            else:
                visit(dependency, path | {key}, hierarchy)
    for key in by_id:
        visit(key, set())
        visit(key, set(), hierarchy=True)
    return list(dict.fromkeys(issues))


def apply_project_priorities(features, project_plan):
    plan = project_plan if isinstance(project_plan, dict) else {}
    priorities = plan.get("priorities") or {}
    for feature in features:
        name = str(feature.get("name") or "").strip().casefold()
        core = str(feature.get("source") or "prd_core") in {"prd_core", "core"}
        for bucket, priority in (("must", "P0" if core else "P1"), ("should", "P1"), ("could", "P2"), ("wont", "EXCLUDE")):
            if name in {str(value).strip().casefold() for value in priorities.get(bucket) or []}:
                feature["priority"] = priority
                feature["releaseScope"] = "excluded" if priority == "EXCLUDE" else "mvp"


def normalize_priorities(features, scope=None):
    scope = scope if isinstance(scope, dict) else {}
    included = {str(name).strip().casefold() for name in scope.get("included") or []}
    excluded = {str(name).strip().casefold() for name in scope.get("excluded") or []}
    count = 0
    for feature in features:
        name = str(feature.get("name") or "").strip().casefold()
        source = str(feature.get("source") or "prd_core").lower()
        core = source in {"core", "prd_core"}
        raw = str(feature.get("priority") or "").upper()
        raw = {"MUST": "P0" if core else "P1", "SHOULD": "P1", "COULD": "P2"}.get(raw, raw)
        if name in excluded or raw in {"WONT", "WON'T", "EXCLUDE"} or str(feature.get("releaseScope") or "").lower() in {"excluded", "backlog"}:
            feature["releaseScope"] = "excluded"
            feature["priority"] = "P3"
            continue
        if raw not in {"P0", "P1", "P2", "P3"}:
            raw = "P0" if core and name in included else "P1" if core else "P2"
        if raw == "P0" and (not core or count >= MAX_P0_FEATURES):
            raw = "P1"
        count += raw == "P0"
        feature["priority"] = raw
        feature["releaseScope"] = "mvp"


def select_mvp(features):
    normalize_priorities(features)
    if dependency_issues(features):
        return []
    by_id = {str(item.get("featureId") or item.get("id") or item.get("name")): item for item in features}
    selected, visiting = {}, set()

    def closure(key):
        if key in visiting:
            # Parent is a grouping relation, not an execution prerequisite.
            # Both individual graphs were checked above; a mixed back-edge
            # simply means the features must be selected together.
            return {}
        if key not in by_id or by_id[key].get("releaseScope") != "mvp":
            return None
        visiting.add(key)
        item = by_id[key]
        dependencies = [*(item.get("dependencies") or [])]
        if item.get("parentFeatureId"):
            dependencies.insert(0, item["parentFeatureId"])
        result = {}
        for dependency in dependencies:
            required = closure(dependency)
            if required is None:
                visiting.remove(key)
                return None
            result.update(required)
        visiting.remove(key)
        result[key] = item
        return result

    ordered = sorted(enumerate(features), key=lambda pair: (
        {"P0": 0, "P1": 1, "P2": 2, "P3": 3}.get(pair[1].get("priority"), 4),
        -int(pair[1].get("priorityScore") or 0), pair[0],
    ))
    for _, feature in ordered:
        key = str(feature.get("featureId") or feature.get("id") or feature.get("name"))
        required = closure(key)
        if required is not None and len(set(selected) | set(required)) <= MAX_MVP_FEATURES:
            selected.update(required)
    return list(selected.values())
