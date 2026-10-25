"""One acceptance policy for generated PRD items, assembly and QA."""

import copy
import hashlib
import json
import re

from pydantic import BaseModel, ConfigDict, ValidationError

from phase2.quality_rules import (
    contamination_reasons,
    has_placeholder,
    kpi_basis_issues,
)

DIMENSIONS = {
    "획득": "acquisition", "활성": "activation", "전환": "conversion",
    "단기 리텐션": "retention_7d", "장기 리텐션": "retention_30d",
    "품질": "quality", "비즈니스": "business",
}


class Measurement(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    event: str
    numerator: str
    denominator: str | None = None
    unit: str
    window: str


def assigned_dimension(direction):
    return DIMENSIONS.get(str(direction).split(":", 1)[0], str(direction))


def kpi_issues(item, direction=None, market_data="", require_measurement=False):
    if not isinstance(item, dict):
        return ["KPI object required"]
    issues = []
    for field in ("metric", "target", "basis", "measurementMethod", "frequency"):
        if not isinstance(item.get(field), str) or not item[field].strip():
            issues.append(f"KPI_REQUIRED:{field}")
    target = str(item.get("target") or "")
    values = target.split("→")
    if len(values) != 2 or any(not re.search(r"\d", value) for value in values):
        issues.append("KPI_TARGET: numeric baseline → target required")
    dimension = item.get("dimension")
    if require_measurement and not dimension:
        issues.append("KPI_DIMENSION: explicit dimension required")
    if dimension is not None and dimension not in DIMENSIONS.values():
        issues.append("KPI_DIMENSION: unknown dimension")
    if direction and dimension and dimension != assigned_dimension(direction):
        issues.append("KPI_DIMENSION: assigned dimension mismatch")
    if item.get("measurement") is not None:
        try:
            measurement = Measurement.model_validate(item["measurement"])
        except ValidationError:
            issues.append("KPI_MEASUREMENT: event/numerator/unit/window strings required")
        else:
            if any(not getattr(measurement, key).strip() for key in ("event", "numerator", "unit", "window")):
                issues.append("KPI_MEASUREMENT: empty measurement field")
            if measurement.unit.lower() in {"%", "percent", "percentage", "ratio", "비율"} and not measurement.denominator:
                issues.append("KPI_MEASUREMENT: ratio denominator required")
            average = " ".join(str(item.get(key) or "") for key in ("metric", "measurementMethod"))
            if re.search(r"평균|\b(?:average|mean)\b", average, re.IGNORECASE) and not measurement.denominator:
                issues.append("KPI_MEASUREMENT: average denominator required")
    elif require_measurement:
        issues.append("KPI_MEASUREMENT: structured measurement required")
    issues.extend(contamination_reasons(item))
    if has_placeholder(item):
        issues.append("KPI_PLACEHOLDER")
    issues.extend(kpi_basis_issues(item, market_data))
    return list(dict.fromkeys(issues))


def kpi_signature(item):
    measurement = item.get("measurement")
    if isinstance(measurement, dict):
        return json.dumps([item.get("dimension"), measurement], sort_keys=True, ensure_ascii=False)
    return re.sub(r"\s+", "", str(item.get("metric") or "")).casefold()


def age_interval(value):
    text = re.sub(r"\s+", "", str(value or "")).removeprefix("만")
    if text in {"연령무관", "무관"}:
        return None

    def endpoint(number, unit, qualifier):
        low = int(number)
        if unit == "대" and (low % 10 or low > 140):
            raise ValueError("PERSONA_AGE: invalid decade")
        high = low + 9 if unit == "대" else low
        if qualifier in {"초반", "중반", "후반"}:
            if unit != "대":
                raise ValueError("PERSONA_AGE: qualifier requires a decade")
            if qualifier == "초반":
                high = low + 3
            elif qualifier == "중반":
                low, high = low + 4, low + 6
            else:
                low += 7
        elif qualifier == "이상":
            high = 150
        elif qualifier == "이하":
            low = 0
        return low, high

    match = re.fullmatch(r"(\d{1,3})(대|세)?(초반|중반|후반)?[-~～](\d{1,3})(대|세)(초반|중반|후반)?", text)
    if match:
        if match[2] and match[2] != match[5]:
            raise ValueError("PERSONA_AGE: mixed age units")
        low = endpoint(match[1], match[2] or match[5], match[3])[0]
        high = endpoint(match[4], match[5], match[6])[1]
    else:
        match = re.fullmatch(r"(\d{1,3})(대|세)(초반|중반|후반|이상|이하)?", text)
        if not match:
            raise ValueError("PERSONA_AGE: unsupported age format")
        low, high = endpoint(*match.groups())
    if low > high:
        raise ValueError("PERSONA_AGE: reversed age interval")
    if not 0 <= low <= high <= 150:
        raise ValueError("PERSONA_AGE: outside supported age bounds")
    return low, high


def persona_issues(item, user_query=""):
    if not isinstance(item, dict):
        return ["PERSONA_OBJECT_REQUIRED"]
    issues = []
    for field in ("name", "age", "job", "techLevel", "goal", "painPoint", "usagePattern"):
        if not isinstance(item.get(field), str) or not item[field].strip():
            issues.append(f"PERSONA_REQUIRED:{field}")
    try:
        actual = age_interval(item.get("age"))
        match = re.search(
            r"(?<!\d)(?:만\s*)?\d{1,3}\s*(?:대|세)?\s*(?:초반|중반|후반)?\s*[-~～]\s*\d{1,3}\s*(?:대|세)(?:\s*(?:초반|중반|후반))?"
            r"|(?<!\d)(?:만\s*)?\d{1,3}\s*(?:대|세)(?:\s*(?:초반|중반|후반|이상|이하))?",
            user_query or "")
        expected = age_interval(match[0]) if match else None
        if actual and expected and not (expected[0] <= actual[0] <= actual[1] <= expected[1]):
            issues.append("PERSONA_AGE: outside user-confirmed range")
    except ValueError as exc:
        issues.append(str(exc))
    issues.extend(contamination_reasons(item))
    if has_placeholder(item):
        issues.append("PERSONA_PLACEHOLDER")
    return list(dict.fromkeys(issues))


def persona_signature(item):
    return tuple(re.sub(r"\s+", "", str(item.get(key) or "")).casefold()
                 for key in ("role", "goal", "painPoint", "usagePattern"))


def persona_directions(ctx):
    segments = ctx.get("persona_segments") or []
    if not segments:
        segments = [ctx.get("target_user_context") or "요구사항의 핵심 사용자"]
    return tuple(f"대상 사용자: {segment}; 명시된 사용자 역할과 사용 상황에 집중하고 연령·직업이 없으면 연령 무관/직업 무관으로 작성"
                 for segment in dict.fromkeys(str(value).strip() for value in segments if str(value).strip()))


def segment_id(direction):
    return "persona." + hashlib.sha256(str(direction).encode()).hexdigest()[:12]


def persona_coverage_issues(document):
    quality = document.get("qualityMetadata") or {}
    if not isinstance(quality, dict):
        return ["PERSONA_SEGMENT_CONTRACT_INVALID: quality metadata object required"]
    expected = quality.get("expectedPersonaSegments") or []
    if not isinstance(expected, list) or any(not isinstance(item, dict) or not isinstance(item.get("segmentId"), str)
                                            or not isinstance(item.get("role"), str) for item in expected):
        return ["PERSONA_SEGMENT_CONTRACT_INVALID: named segment array required"]
    present = {item["segmentId"] for item in document.get("userPersonas") or []
               if isinstance(item, dict) and isinstance(item.get("segmentId"), str)}
    return [f"PERSONA_SEGMENT_MISSING: {item['role']} ({item['segmentId']})" for item in expected if item["segmentId"] not in present]


def audit_items(sections, user_query, market_data):
    out = copy.deepcopy(sections)
    diagnostics = list(out.get("_qualityDiagnostics") or [])
    for field, validator, signature in (("kpi", lambda item: kpi_issues(item, market_data=market_data, require_measurement=True), kpi_signature),
                                        ("userPersonas", lambda item: persona_issues(item, user_query), persona_signature)):
        accepted, seen = [], set()
        for index, item in enumerate(out.get(field) or []):
            reasons = validator(item)
            if not reasons:
                identity = signature(item)
                if identity in seen:
                    reasons = ["DUPLICATE_MEASUREMENT" if field == "kpi" else "DUPLICATE_USER_JOURNEY"]
            if reasons:
                diagnostics.append({"section": field, "index": index, "stage": "assembly", "reasons": reasons})
                continue
            seen.add(identity)
            if field == "userPersonas":
                item = {**item, "assumption": True}
            accepted.append(item)
        out[field] = accepted
        if not accepted:
            diagnostics.append({"section": field, "stage": "assembly", "reasons": ["NO_VALID_ITEMS: evidence or manual completion required"]})
    out["_qualityDiagnostics"] = diagnostics
    return out


def align_project_scope(document, plan, excluded_features=()):
    scope = document.setdefault("mvpScope", {})
    plan_scope = plan.get("scope") or {}
    priorities = plan.get("priorities") or {}
    exclusions = list(dict.fromkeys([*(scope.get("excluded") or []), *(plan_scope.get("out") or []),
                                    *(priorities.get("wont") or []), *excluded_features]))
    scope["excluded"] = exclusions
    scope["included"] = [value for value in scope.get("included") or [] if value not in exclusions]
    conflicts = [item.get("name") for item in document.get("coreFeatures") or [] if item.get("name") in exclusions]
    if conflicts:
        quality = document.setdefault("qualityMetadata", {})
        quality.setdefault("warnings", []).append("PROJECT_SCOPE_CONFLICT: " + ", ".join(conflicts))
    return document
