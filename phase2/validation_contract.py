"""Stable, revision-bound issues shared by QA and Phase3."""

import hashlib
import json
import re
from dataclasses import asdict, dataclass


def artifact_revision(*artifacts) -> str:
    def parsed(value):
        if isinstance(value, str):
            try:
                return json.loads(value)
            except ValueError:
                return value
        return value
    return hashlib.sha256(json.dumps([parsed(value) for value in artifacts], sort_keys=True,
                                     ensure_ascii=False, default=str).encode()).hexdigest()


@dataclass(frozen=True)
class ContractIssue:
    code: str
    domain: str
    artifactKey: str
    pointer: str
    message: str
    featureId: str = ""
    severity: str = "BLOCKER"
    repairable: bool = True
    revision: str = ""

    def to_dict(self):
        return {**asdict(self), "repairTarget": self.domain, "agent": {"db": "DBA", "api": "API", "prd": "PRD", "feature_spec": "FEATURE_SPEC"}.get(self.domain, "PM"),
                "reason": self.message, "problem": self.message}


def issue_detail(code, domain, message, **artifact):
    endpoint = re.search(r"\b(GET|POST|PUT|PATCH|DELETE|HEAD|OPTIONS)\s+(/[^\s,'\")]+)", message)
    columns = list(re.finditer(r"\b([a-z][a-z0-9_]*)\.([a-z][a-z0-9_]*)\b", message))
    column = columns[-1] if columns else None
    key = artifact.get("artifactKey") or (endpoint.group(0).rstrip(":") if endpoint else column.group(1) if column else "")
    pointer = artifact.get("pointer") or (f"/columns/{column.group(2)}" if column and domain == "db" else "")
    if str(code).startswith("API_FIELD_MAPPING") and not pointer:
        pointer = "/fieldMappings/" + message.rsplit(":", 1)[-1].strip()
    if str(code).startswith("API_OPERATION_CONTRACT") and not pointer:
        pointer = "/" + message.split(";", 1)[0].rsplit(" ", 1)[-1].strip()
    fid = artifact.get("featureId") or ""
    if not key:
        key = fid or hashlib.sha256(message.encode()).hexdigest()[:16]
    detail = ContractIssue(str(code), str(domain), str(key), str(pointer), str(message), str(fid)).to_dict()
    return {**detail, **artifact}


def issue_identity(detail):
    return tuple(str(detail.get(key) or "") for key in ("code", "domain", "artifactKey", "pointer", "featureId"))


def dedupe_issues(details):
    result = {}
    for detail in details:
        result.setdefault(issue_identity(detail), detail)
    return list(result.values())
