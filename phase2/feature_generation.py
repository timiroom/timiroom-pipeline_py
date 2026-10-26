"""Bounded Feature Spec generation, strict parsing and source identity tracking."""

import copy
import hashlib
import json
import logging
import re
import time
from pathlib import Path
from typing import Literal

from pydantic import Field, ValidationError

from phase2.artifact_shapes import DbContract, Endpoint, Feature, Registry, shape_issues
from phase2.contract_guard import feature_contract_issues, semantic_identifier
from phase2.feature_draft import DraftDocument, compile_document
from phase2.feature_draft import merge_patch as merge_draft_patch
from phase2.feature_draft import repair_scope as draft_repair_scope
from phase2.mvp_policy import dependency_issues
from phase2.operation_contract import (
    cross_contract_issues,
    operation_registry,
    route_key,
)
from phase2.registry_contract_guard import (
    contract_artifacts,
    ownership_contract_issues,
    registry_contract_issues,
)
from phase2.workflow_contract import normalize_boolean_workflows

logger = logging.getLogger(__name__)


class GeneratedOperation(Endpoint):
    method: Literal["GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"]
    path: str
    action: str
    featureId: str
    responseSchema: dict
    successStatus: int
    contentType: str
    fieldMappings: list[dict]


class GeneratedDbContract(DbContract):
    primaryTable: str | None
    fields: list[dict]
    tables: list[str]


class GeneratedFeature(Feature):
    featureId: str
    sourceFeatureId: str | None
    source: Literal["prd_core", "supporting"]
    parentFeatureId: str | None
    name: str
    description: str
    requirements: list[str] = Field(min_length=1)
    actions: list[str] = Field(min_length=1)
    priority: Literal["P0", "P1", "P2", "P3"]
    reason: str
    applicability: dict[str, bool]
    nonApplicabilityReasons: dict[str, str]
    dependencies: list[str]
    apiContract: list[GeneratedOperation]
    dbContract: GeneratedDbContract


class GeneratedDocument(Registry):
    features: list[GeneratedFeature] = Field(min_length=1, max_length=12)


def output_schema_prompt(draft=False) -> str:
    """Use the same model schema for provider guidance and prompt-only mode."""
    model = DraftDocument if draft else GeneratedDocument
    schema = model.model_json_schema()
    return json.dumps(_schema_without_titles(schema) if draft else schema, ensure_ascii=False, separators=(",", ":"))


def _schema_without_titles(value):
    if isinstance(value, list):
        return [_schema_without_titles(item) for item in value]
    if not isinstance(value, dict):
        return value
    result = {}
    for key, child in value.items():
        if key == "title":
            continue
        if key in {"properties", "$defs", "definitions", "patternProperties"}:
            result[key] = {name: _schema_without_titles(schema) for name, schema in child.items()}
        elif key in {"default", "const", "enum", "examples"}:
            result[key] = copy.deepcopy(child)
        else:
            result[key] = _schema_without_titles(child)
    return result


def response_format(mode, draft=False):
    if mode == "json_object":
        return {"type": "json_object"}
    # Embedded JSON Schemas have dynamic property names. Non-strict provider
    # guidance is paired with strict local validation, never advertised as a
    # provider-enforced strict schema guarantee.
    return {"type": "json_schema", "json_schema": {
        "name": "feature_spec", "strict": False,
        "schema": (DraftDocument if draft else GeneratedDocument).model_json_schema(),
    }}


def source_cores(prd):
    cores = copy.deepcopy(prd.get("coreFeatures") or [])
    for core in cores:
        source = core.get("sourceFeatureId") or core.get("featureId") or core.get("id")
        if not source:
            canonical = json.dumps(core, ensure_ascii=False, sort_keys=True)
            source = "source." + hashlib.sha256(canonical.encode()).hexdigest()[:16]
        core["sourceFeatureId"] = source
    return cores


def generation_input(items):
    """Keep temporary source identities out of the final-ID example slots."""
    result = copy.deepcopy(items)
    for item in result:
        previous = item.get("featureId") or item.get("id")
        if previous and not semantic_identifier(previous):
            if not item.get("sourceFeatureId"):
                item["sourceFeatureId"] = previous
            item.pop("featureId", None)
            item.pop("id", None)
        for operation in item.get("apiContract") or []:
            if isinstance(operation, dict) and operation.get("featureId") and not semantic_identifier(operation["featureId"]):
                operation.pop("featureId")
    return result


def bind_sources(features, cores):
    """Legacy exact-name/id binding only; explicit source IDs survive renames."""
    out = copy.deepcopy(features)
    for item in out:
        if not isinstance(item, dict) or item.get("sourceFeatureId"):
            continue
        matches = [core for core in cores if
                   item.get("featureId") == core.get("featureId")
                   or item.get("name") == core.get("name")]
        if len(matches) == 1 and item.get("source") != "supporting":
            item["sourceFeatureId"] = matches[0]["sourceFeatureId"]
    return out


def document_issues(features, cores, *, preflight=False):
    if not isinstance(features, list) or not features:
        return ["FEATURE_SPEC_EMPTY: /features: non-empty array required"]
    issues = []
    for index, item in enumerate(features):
        shape = shape_issues("registry", [item])
        if shape:
            issues.extend(f"FEATURE_SPEC_SHAPE: {entry['pointer'].replace('/features/0', f'/features/{index}', 1)}: {entry['message']}"
                          for entry in shape)
            continue
        try:
            GeneratedFeature.model_validate(item)
        except ValidationError as exc:
            issues.extend(f"FEATURE_SPEC_REQUIRED: /features/{index}/{'/'.join(map(str, error['loc']))}: {error['msg']}"
                          for error in exc.errors(include_url=False, include_input=False))
        issues.extend(f"FEATURE_SPEC_CONTRACT: /features/{index}: {message}"
                      for message in feature_contract_issues([item], feature_pointer=f"/features/{index}"))
        for domain in ("ui", "api", "db"):
            if (item.get("applicability") or {}).get(domain) is False and not (
                item.get("nonApplicabilityReasons") or {}
            ).get(domain):
                issues.append(f"FEATURE_SPEC_SCOPE: /features/{index}: {domain}=false requires a reason")
    ids = [item.get("featureId") for item in features if isinstance(item, dict) and isinstance(item.get("featureId"), str)]
    if len(ids) != len(set(ids)):
        issues.append("FEATURE_SPEC_ID_DUPLICATE: /features: unique featureId required")
    if len(features) > 12:
        issues.append("FEATURE_SPEC_BUDGET: /features: at most 12 MVP features")
    expected = {core["sourceFeatureId"] for core in cores}
    for core in cores:
        matches = [item for item in features if isinstance(item, dict)
                   and item.get("sourceFeatureId") == core["sourceFeatureId"]
                   and item.get("source", "prd_core") == "prd_core"]
        if len(matches) != 1:
            issues.append(f"FEATURE_SPEC_CORE_COVERAGE: {core['sourceFeatureId']}: expected exactly one core feature")
    for index, item in enumerate(features):
        if not isinstance(item, dict):
            continue
        source = item.get("sourceFeatureId")
        if not isinstance(source, (str, type(None))):
            continue
        if item.get("source", "prd_core") == "prd_core" and source not in expected:
            issues.append(f"FEATURE_SPEC_SOURCE: /features/{index}: unknown core sourceFeatureId")
        if item.get("source") == "supporting" and (not item.get("parentFeatureId") or not str(item.get("reason") or "").strip()):
            issues.append(f"FEATURE_SPEC_SUPPORT: /features/{index}: parentFeatureId/reason required")
        for reference in [item.get("parentFeatureId"), *(item.get("dependencies") or [])]:
            if reference and reference not in ids:
                issues.append(f"FEATURE_SPEC_DEPENDENCY: /features/{index}: unknown dependency {reference}")
        for operation_index, operation in enumerate(item.get("apiContract") if isinstance(item.get("apiContract"), list) else []):
            if isinstance(operation, dict) and operation.get("featureId") != item.get("featureId"):
                issues.append(f"FEATURE_SPEC_OPERATION_OWNER: /features/{index}/apiContract/{operation_index}: operation featureId must match its owner")
    if not shape_issues("registry", features):
        issues.extend(registry_contract_issues(features))
        if preflight:
            issues.extend(_preflight_issues(features, cores))
        for issue in dependency_issues(features):
            key = issue.split(": ", 1)[-1].split(" -> ", 1)[0]
            for index, item in enumerate(features):
                if item.get("featureId") == key:
                    issues.append(f"{issue}: /features/{index}/dependencies")
    return issues


def _preflight_issues(features, cores):
    issues = ownership_contract_issues(features)
    db, api = contract_artifacts(features)
    _, operation_conflicts = operation_registry(features)
    for message in operation_conflicts:
        for index, feature in enumerate(features):
            for position, operation in enumerate(feature.get("apiContract") or []):
                method, path = route_key(operation)
                if message.startswith(f"API_OPERATION_CONTRACT_CONFLICT: {method} {path} "):
                    issues.append(f"API_OPERATION_CONTRACT_CONFLICT: /features/{index}/apiContract/{position}: {message}")
    by_source = {item.get("sourceFeatureId"): item for item in features if item.get("source") == "prd_core"}
    projected_cores = [{**core, "featureId": by_source[core["sourceFeatureId"]]["featureId"]}
                       for core in cores if core["sourceFeatureId"] in by_source]
    for code, _domain, message in cross_contract_issues(features, db, api, {"coreFeatures": projected_cores}):
        index = next((i for i, feature in enumerate(features) if message.startswith(feature["featureId"] + ":")), None)
        if index is not None:
            issues.append(f"{code}: /features/{index}: {message}")
    method_actions = {"create": "POST", "read": "GET", "list": "GET", "search": "GET", "update": "PATCH", "edit": "PATCH", "delete": "DELETE"}
    for index, feature in enumerate(features):
        if (feature.get("applicability") or {}).get("db") is not False and any(
            operation.get("authRequired") is True for operation in feature.get("apiContract") or []
        ) and not all((feature.get("ownership") or {}).get(key) for key in ("scope", "access")):
            issues.append(f"FEATURE_OWNERSHIP_REQUIRED: /features/{index}/ownership: authenticated stored resources require an explicit ownership scope and access contract")
        declared = {operation["method"] for operation in feature.get("apiContract") or []}
        if (feature.get("applicability") or {}).get("api") is not False:
            for action in feature.get("actions") or []:
                required = method_actions.get(action)
                if required and required not in declared and not (required == "PATCH" and "PUT" in declared):
                    issues.append(f"FEATURE_ACTION_CONTRACT_MISSING: /features/{index}/actions: {action} requires {required}")
        for op_index, operation in enumerate(feature.get("apiContract") or []):
            if operation.get("method") in {"POST", "PATCH", "PUT", "DELETE"} and not str(operation.get("transactionRules") or "").strip():
                issues.append(f"FEATURE_TRANSACTION_REQUIRED: /features/{index}/apiContract/{op_index}/transactionRules: explicit write/failure boundary required")
    return issues


def strict_payload(raw, finish_reason="stop", *, allow_patch=False):
    if finish_reason not in {"stop", None}:
        return None, [f"FEATURE_SPEC_FINISH: {finish_reason}"]
    try:
        # Preserve object pairs until validation so duplicate-key errors retain
        # a complete pointer. No duplicate value is selected or recovered.
        class ObjectPairs(list):
            pass

        def convert(value, pointer=""):
            if isinstance(value, ObjectPairs):
                result = {}
                for key, child in value:
                    path = pointer + "/" + key.replace("~", "~0").replace("/", "~1")
                    if key in result:
                        raise ValueError("duplicate key: " + path)
                    result[key] = convert(child, path)
                return result
            if isinstance(value, list):
                return [convert(child, pointer + "/" + str(index)) for index, child in enumerate(value)]
            return value

        data = json.loads(raw, object_pairs_hook=ObjectPairs,
                          parse_constant=lambda value: (_ for _ in ()).throw(ValueError("nonfinite JSON number")))
        data = convert(data)
    except json.JSONDecodeError as exc:
        return None, [f"FEATURE_SPEC_JSON: line={exc.lineno} column={exc.colno} offset={exc.pos}"]
    except ValueError as exc:
        reason = str(exc) if str(exc).startswith("duplicate key: /") else "invalid JSON value"
        return None, ["FEATURE_SPEC_JSON: " + reason]
    except (TypeError, RecursionError):
        return None, ["FEATURE_SPEC_JSON: invalid JSON value or nesting depth"]
    if not isinstance(data, dict) or ("features" in data or not allow_patch) and not isinstance(data.get("features"), list):
        return None, ["FEATURE_SPEC_SHAPE: /features: array required"]
    return data, []


def strict_document(raw, finish_reason="stop"):
    data, issues = strict_payload(raw, finish_reason)
    return (data["features"] if data is not None else None), issues


def safe_text(raw):
    text = re.sub(r'(?i)("(?:api[_-]?key|access[_-]?token|refresh[_-]?token|password|secret|authorization)"\s*:\s*)"[^"\n]*"', r'\1"[REDACTED]"', raw)
    text = re.sub(r"\b(?:sk-|xox[baprs]-|xapp-)[A-Za-z0-9_-]+", "[REDACTED]", text)
    return re.sub(r"(?i)bearer\s+[A-Za-z0-9._~-]+", "Bearer [REDACTED]", text)


def unsupported_schema(exc):
    if getattr(exc, "status_code", None) not in {400, 422}:
        return False
    message = str(exc).lower()
    return ("response_format" in message or "json_schema" in message) and any(
        word in message for word in ("unsupported", "not supported", "unknown parameter", "unrecognized")
    )


def retryable_transport(exc):
    return (type(exc).__name__ in {"APITimeoutError", "APIConnectionError", "TimeoutError", "ConnectError"}
            or getattr(exc, "status_code", 0) in {408, 429, 500, 502, 503, 504})


def normalize_parameter_maps(features):
    """Lossless adapter for valid JSON location/name maps; never repair syntax."""
    result = copy.deepcopy(features)
    changes = []
    for index, feature in enumerate(result):
        if not isinstance(feature, dict):
            continue
        for operation_index, operation in enumerate(feature.get("apiContract") or []):
            if not isinstance(operation, dict):
                continue
            raw = operation.get("parameters")
            if not isinstance(raw, dict) or not raw or set(raw) - {"path", "query", "header", "cookie"}:
                continue
            converted, valid = [], True
            for location, named in raw.items():
                if not isinstance(named, dict):
                    valid = False
                    break
                for name, definition in named.items():
                    if not isinstance(name, str) or not name or not isinstance(definition, dict):
                        valid = False
                        break
                    if "schema" in definition:
                        parameter = {**definition, "in": location, "name": name}
                    else:
                        schema = copy.deepcopy(definition)
                        required = schema.pop("required") if isinstance(schema.get("required"), bool) else False
                        parameter = {"in": location, "name": name, "schema": schema, "required": required}
                    if location == "path":
                        parameter["required"] = True
                    converted.append(parameter)
            if valid:
                operation["parameters"] = converted
                changes.append(f"/features/{index}/apiContract/{operation_index}/parameters")
    return result, changes


def _patchable_draft(payload):
    if not isinstance(payload, dict) or payload.get("version") != "contract-draft-v1":
        return False
    if set(payload) - {"version", "features", "tables", "schemas"}:
        return False
    for collection, identity in (("features", "featureId"), ("tables", "name"), ("schemas", "name")):
        entries = payload.get(collection)
        if not isinstance(entries, list):
            return False
        keys = [item.get(identity) for item in entries if isinstance(item, dict)]
        if len(keys) != len(entries) or any(not isinstance(key, str) or not key for key in keys):
            return False
        if len(set(keys)) != len(keys):
            return False
    return bool(payload["features"])


def normalize_response_array_maps(features):
    """Resolve an omitted array wrapper only when the declared DTO is unambiguous."""
    from phase2.operation_contract import _field_schema

    result, changes = copy.deepcopy(features), []
    for index, feature in enumerate(result):
        for position, operation in enumerate(feature.get("apiContract") or []):
            schema = operation.get("responseSchema") or {}
            if schema.get("type") != "object":
                continue
            for mapping_index, mapping in enumerate(operation.get("fieldMappings") or []):
                field = mapping.get("field", "")
                if mapping.get("source") != "response" or mapping.get("pointer") or not field.startswith("[].") or _field_schema(schema, field) is not None:
                    continue
                candidates = [name + field for name, child in schema.get("properties", {}).items()
                              if child.get("type") == "array" and _field_schema(schema, name + field) is not None]
                if len(candidates) != 1:
                    continue
                mapping["field"] = candidates[0]
                changes.append({"pointer": f"/features/{index}/apiContract/{position}/fieldMappings/{mapping_index}/field",
                                "before": field, "after": candidates[0]})
    return result, changes


def _ownership_repair_candidates(payload, scope):
    tables = {table["name"]: table for table in payload["tables"]}
    candidates = []
    for feature in payload["features"]:
        if feature["featureId"] not in scope["featureIds"]:
            continue
        primary = feature.get("primaryTable")
        table = tables.get(primary, {}) if isinstance(primary, str) else {}
        foreign_keys = table.get("foreignKeys")
        links = []
        for foreign_key in foreign_keys if isinstance(foreign_keys, list) else []:
            if not isinstance(foreign_key, dict) or not isinstance(foreign_key.get("references"), dict):
                continue
            if isinstance(foreign_key.get("column"), str) and foreign_key["column"]:
                links.append({"ownerKey": foreign_key["column"], "ownerEntity": foreign_key["references"].get("table")})
        candidates.append({"featureId": feature["featureId"], "primaryTable": primary,
                           "existingOwnership": feature.get("ownership"), "foreignKeys": links})
    return candidates


def _feature_patch_scope(baseline, issues, cores):
    bad_indices = {int(value) for issue in issues for value in re.findall(r"/features/(\d+)", issue)}
    if any("ID_DUPLICATE" in issue for issue in issues):
        ids = [item.get("featureId") for item in baseline if isinstance(item, dict)]
        bad_indices.update(index for index, item in enumerate(baseline)
                           if isinstance(item, dict) and ids.count(item.get("featureId")) > 1)
    missing = {core["sourceFeatureId"] for core in cores if any(
        "CORE_COVERAGE" in issue and core["sourceFeatureId"] in issue for issue in issues
    )}
    return bad_indices, missing


def _merge_patch(baseline, patch, issues, cores):
    """Only invalid feature keys and missing core sources may be replaced."""
    bad_indices, missing = _feature_patch_scope(baseline, issues, cores)
    bad_keys = {str(baseline[index].get("featureId")) for index in bad_indices
                if index < len(baseline) and isinstance(baseline[index], dict)}
    bad_sources = {str(baseline[index].get("sourceFeatureId")) for index in bad_indices
                   if index < len(baseline) and isinstance(baseline[index], dict) and baseline[index].get("sourceFeatureId")}
    merged = copy.deepcopy(baseline)
    for item in patch:
        if not isinstance(item, dict):
            continue
        matches = [index for index, old in enumerate(baseline) if isinstance(old, dict)
                   and (index in bad_indices or (old.get("sourceFeatureId") in missing and old.get("source") == "prd_core"))
                   and ((str(old.get("featureId")) in bad_keys and old.get("featureId") == item.get("featureId"))
                        or (str(old.get("sourceFeatureId")) in (missing | bad_sources) and old.get("sourceFeatureId") == item.get("sourceFeatureId"))
                        or (index in bad_indices and old.get("name") == item.get("name")))]
        if matches:
            merged[matches[0]] = copy.deepcopy(item)
        elif item.get("sourceFeatureId") in missing:
            merged.append(copy.deepcopy(item))
        else:
            raise ValueError("FEATURE_SPEC_REPAIR_SCOPE: unrequested feature patch")
    return merged


async def generate_document(client, model, runtime, prompt, cores, *, mode="auto", diagnostics_dir="", max_tokens=16000, draft=False, reasoning_effort=None, timeout_seconds=None, max_attempts=2):
    diagnostics = []
    current_mode = "json_schema" if mode == "auto" else mode
    # Keep other model families' existing request contract unchanged.
    applied_effort = reasoning_effort if model.startswith("gpt-5.4") else None
    sampling = {"reasoning_effort": applied_effort} if applied_effort and applied_effort != "none" else {"temperature": 0.2}
    messages = [{"role": "system", "content": "Return one complete JSON object. Never truncate or emit Markdown."},
                {"role": "user", "content": prompt}]
    if draft:
        messages[0]["content"] += (
            " contract-draft-v1 has sibling top-level arrays features, tables and schemas. "
            "Never put a document envelope, tables or schemas inside the features array."
        )
    if max_attempts < 1:
        raise ValueError("max_attempts must be >= 1")
    baseline, failures, previous_issues = None, [], []
    draft_baseline, active_scope = None, None
    # SDK retries are disabled here: this owner provides the bounded retries.
    # Each retry is a targeted patch over the previous draft, so extra attempts
    # shrink the remaining violations instead of regenerating from scratch.
    with_options = getattr(client, "with_options", None)
    options = {"max_retries": 0}
    if timeout_seconds is not None:
        options["timeout"] = timeout_seconds
    transport = with_options(**options) if callable(with_options) else client
    for attempt in range(max_attempts):
        normalizations = []
        response_normalizations = []
        boolean_normalizations = []
        start = time.perf_counter()
        raw, finish, usage, provider_model = "", None, None, None
        candidate, payload = None, None
        try:
            async def request(response_mode=current_mode, request_messages=messages):
                return await transport.chat.completions.create(
                    model=model, **sampling, max_completion_tokens=max_tokens,
                    messages=request_messages, response_format=response_format(response_mode, draft=draft),
                )
            try:
                runtime_options = {"timeout_seconds": timeout_seconds} if timeout_seconds is not None else {}
                response = await runtime.call(request, **runtime_options) if runtime else await request()
            except Exception as exc:
                if mode != "auto" or current_mode != "json_schema" or not unsupported_schema(exc):
                    raise
                diagnostics.append({"attempt": attempt, "code": "STRUCTURED_OUTPUT_UNSUPPORTED", "mode": current_mode})
                current_mode = "json_object"
                async def json_request():
                    return await request("json_object")
                response = await runtime.call(json_request, **runtime_options) if runtime else await json_request()
            choice = response.choices[0]
            provider_model = getattr(response, "model", None)
            raw = choice.message.content or ""
            finish = getattr(choice, "finish_reason", None)
            value = getattr(response, "usage", None)
            usage = {key: getattr(value, key, None) for key in ("prompt_tokens", "completion_tokens", "total_tokens")}
            if getattr(choice.message, "refusal", None):
                failures = ["FEATURE_SPEC_REFUSAL: provider refused generation"]
                candidate = None
            else:
                if draft:
                    payload, failures = strict_payload(raw, finish, allow_patch=draft_baseline is not None)
                    if payload is not None:
                        if draft_baseline is not None:
                            payload = merge_draft_patch(draft_baseline, payload, active_scope)
                        candidate = compile_document(payload)
                else:
                    candidate, failures = strict_document(raw, finish)
                if candidate is not None:
                    candidate, normalizations = normalize_parameter_maps(candidate)
                    candidate, response_normalizations = normalize_response_array_maps(candidate)
                    candidate, boolean_normalizations = normalize_boolean_workflows(candidate)
                    candidate = bind_sources(candidate, cores)
                    if baseline is not None and not draft:
                        candidate = _merge_patch(baseline, candidate, previous_issues, cores)
                    failures = document_issues(candidate, cores, preflight=draft)
        except Exception as exc:  # noqa: BLE001 - provider boundary; record the type, never the sensitive response
            logger.warning("Feature Spec bounded generation error: %s", type(exc).__name__)
            failures = [f"FEATURE_SPEC_TRANSPORT: {type(exc).__name__}"] if not isinstance(exc, ValueError) else [str(exc)]
            candidate = None
            terminal = not retryable_transport(exc) if not isinstance(exc, ValueError) else False
        else:
            terminal = any("REFUSAL" in item for item in failures)
        record = {"attempt": attempt, "mode": current_mode, "finishReason": finish,
                  "requestedModel": model, "providerModel": provider_model, "reasoningEffort": applied_effort,
                  "requestTimeoutSeconds": timeout_seconds,
                  "durationSeconds": round(time.perf_counter() - start, 3), "usage": usage,
                  "responseBytes": len(raw.encode()), "responseHash": hashlib.sha256(raw.encode()).hexdigest(),
                  "issues": failures, "partialRecoveryUsed": False, "parameterMapNormalizations": normalizations,
                  "responseMapNormalizations": response_normalizations,
                  "booleanPolicyNormalizations": boolean_normalizations}
        if isinstance(payload, dict) and isinstance(payload.get("features"), list):
            record["featureRefs"] = [{"index": index, **{key: value for key in ("featureId", "sourceFeatureId")
                                                        if isinstance(value := item.get(key), str)}}
                                     for index, item in enumerate(payload["features"]) if isinstance(item, dict)]
        diagnostics.append(record)
        if diagnostics_dir and raw:
            # Raw diagnostics are local opt-in only and excluded from SSE.
            directory = Path(diagnostics_dir)
            directory.mkdir(parents=True, exist_ok=True)
            path = directory / f"feature-{time.time_ns()}-{attempt}.json"
            path.write_text(safe_text(raw), encoding="utf-8")
        logger.info("Feature Spec attempt=%d mode=%s finish=%s bytes=%d issues=%d elapsed=%.3f",
                    attempt, current_mode, finish, record["responseBytes"], len(failures), record["durationSeconds"])
        if not failures:
            return candidate, diagnostics, []
        if terminal or attempt == max_attempts - 1:
            break
        if mode == "auto" and current_mode == "json_schema" and any(item.startswith("FEATURE_SPEC_JSON:") for item in failures):
            # Some compatible gateways accept schema guidance without enforcing JSON syntax.
            # Reuse the one existing retry with JSON-object transport, not partial recovery.
            current_mode = "json_object"
            diagnostics.append({"attempt": attempt, "code": "INVALID_SCHEMA_JSON_RETRY_AS_JSON_OBJECT", "mode": current_mode})
        previous_issues = failures
        baseline = candidate
        patch_scope = {}
        if baseline is not None:
            bad_indices, missing = _feature_patch_scope(baseline, failures, cores)
            patch_scope = {"invalidFeatures": [
                {"featureId": item.get("featureId"), "sourceFeatureId": item.get("sourceFeatureId"), "name": item.get("name"), "index": index}
                for index, item in enumerate(baseline) if index in bad_indices and isinstance(item, dict)
            ], "missingSourceFeatureIds": sorted(missing)}
        if draft and _patchable_draft(payload):
            draft_baseline = payload
            active_scope = draft_repair_scope(payload, failures, cores)
            patch_scope = {**active_scope, "ownershipCandidates": _ownership_repair_candidates(payload, active_scope)}
        elif draft:
            # Without keyed root collections no valid contract exists to patch.
            # Use only the existing single retry; never salvage nested resources.
            diagnostics[-1]["retryKind"] = "UNUSABLE_DRAFT_RETRY"
        if draft:
            envelope = '{"version":"contract-draft-v1","features":[],"tables":[],"schemas":[]}'
            instruction = (
                "Return only replacements for keys explicitly allowed in repairScope. Preserve every other feature/table/DTO. "
                "You may omit unchanged feature fields and envelope version. For existing routes, apiContract may contain ONLY the affected operations keyed by their original method/path; omitted operations are preserved. "
                "For operationIndices, keep the original endpoint count/order and change only those indexed operations; "
                "preserve every other endpoint and its referenced DTO definitions. "
                "Exception: missingOperationMethods allows appending only those missing methods after the existing operations. "
                "missingOperationKeys permits appending ONLY the exact named method/path lookup or role-provision operation, with directly referenced DTOs; it does not authorize other routes. "
                "emptyApiFeatureIds permits filling that feature's currently empty apiContract and defining only its directly referenced NEW DTOs and schema-ref dependencies. "
                "For duplicateCoreSources, keep exactly ONE existing featureId as prd_core. Reclassify the other listed featureIds as supporting with a parentFeatureId, reason and non-P0 priority; preserve their original sourceFeatureId provenance and operations. "
                "Do not add or rename a third core for a duplicate source. Missing and duplicated cores are different repairs. "
                "An affected/appended operation may declare NEW DTO definitions that it directly references, including their explicit schema-ref dependencies. "
                "protectedSchemaNames are used by unaffected operations and MUST NOT be returned or changed. If the affected operation needs extra output fields, create a NEW operation-specific DTO (or wrapper/item DTO), reference it from only that operation, and preserve shared DTOs exactly. schemaNames lists only editable existing DTOs. "
                "When adding a missingSourceFeatureId core, you may also declare NEW tables/DTOs directly referenced by that core, "
                "plus their explicit FK/schema-ref dependencies. Reuse existing definitions without returning or modifying them. "
                "Do not add unrelated resources. The merged draft must pass every contract check. "
                "The arrays features, tables and schemas are siblings at the top level: a DTO patch belongs in schemas, NEVER inside features or a feature object. "
                "Every features entry needs featureId, every tables/schemas entry needs name, and each key may occur only once in its array. "
                "For ownership errors, use ownershipCandidates to select the actual primary-table FK ownerKey and referenced ownerEntity; preserve the access policy. "
                "Do not clear ownership or set it to null to bypass a reference error. Authenticated stored resources still need an explicit scope/access contract. "
                "For workflow issues, align statePolicies/intervalPolicies/readDependencies/sessionPolicy with actual operations and DTOs. "
                "readDependencies is a sparse keyed patch by method/path/responseField: return only new or corrected entries; all existing promised entries are preserved. Never use an empty array to remove promised reads. "
                "Preserve promised lookups/transitions: implement the missing named operation or effect rather than deleting the requirement to silence validation. "
                "Bearer issuance must return its tokenField; cookie issuance must declare Set-Cookie. Never change the transport only in prose. "
                "API_REQUIRED_OUTPUT_MISSING requires fixing an operation's responseType and its schemas.nodes (including a NEW DTO when needed), not repeating outputRequirements. "
                "A 204 operation cannot return the promised output: use a suitable response status and explicit DTO when the requirement needs returned fields. Do not remove the promised output to silence validation. "
                if draft_baseline is not None else "The previous JSON is unusable. Return one complete shallow draft, never partial JSON. "
            )
            prior = [*messages, {"role": "assistant", "content": raw}] if draft_baseline is not None else messages[:2]
            messages = [*prior, {"role": "user", "content":
                "Correct once using the same shallow draft format: " + envelope + ". " + instruction
                + json.dumps({"issues": failures, "repairScope": patch_scope}, ensure_ascii=False)
                + " Do not emit requestSchema/responseSchema or dbContract. Use schemas.nodes, table references and scalar mappings."}]
            continue
        messages = [*messages, {"role": "assistant", "content": raw}, {"role": "user", "content":
            "Correct these errors once. Return JSON {\"features\":[...]}. "
            + ("Return only replacements for invalid features or missing core sources. Keep their featureId/sourceFeatureId. Do not return or modify valid siblings. " if baseline is not None else
               "Previous JSON could not be used. Return the complete document preserving every sourceFeatureId from the input. ")
            + json.dumps({"issues": failures, "repairScope": patch_scope}, ensure_ascii=False)
            + " Validate JSON syntax before responding: close each nested responseSchema/properties/items object before the next operation in apiContract. "
            + " JSON pointer indices refer to the complete original features array, including supporting features. "
            "Use repairScope.featureId/sourceFeatureId to select replacements; do not infer a different feature from the numeric index."}]
    return None, diagnostics, failures
