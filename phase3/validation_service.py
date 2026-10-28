import json
import logging
import re

from phase2.agent_contract import IssueSeverity, classify_issue
from phase2.execution_error import GENERATION_HALT_STAGES
from phase2.openapi_spec import build_openapi_document
from phase2.state import PipelineState
from phase3.schema_validator import SchemaValidator

logger = logging.getLogger(__name__)


class ValidationService:

    def __init__(self, validator: SchemaValidator):
        self._validator = validator

    @staticmethod
    def _canonical_blockers(items: list[str]) -> list[str]:
        """QA/구조 검증 메시지의 표현 차이를 제거하고 중복을 합친다."""
        normalized: list[str] = []
        seen: set[str] = set()
        for item in items:
            value = " ".join(str(item or "").split())
            if value.startswith("QA:"):
                value = value[3:].strip()
            if value and value not in seen:
                seen.add(value)
                normalized.append(value)
        return normalized

    @staticmethod
    def _halted_generation(state: PipelineState) -> PipelineState:
        """Report only the halt. A stage that never ran leaves no repairable defect.

        Without this, a PM/PRD/SEARCH transport failure was reported as three
        extra blockers ("DB 스키마: 값이 비어있습니다", ...) plus a PRD repair
        target, so Phase3 dispatched repairs for artifacts no agent had been
        asked to produce.
        """
        from phase2.validation_contract import issue_detail
        messages = list(dict.fromkeys(state.generation_blockers))
        details = [{**issue_detail(message.split(":", 1)[0], "prd", message),
                    "repairable": False, "failedStage": state.failed_stage}
                   for message in messages]
        return state.copy(
            validated=False, qa_approved=False,
            qa_blocker_details=details, qa_issue_details=details, qa_repair_issues=[],
            qa_db_blockers=[], qa_api_blockers=[], qa_prd_blockers=messages,
            qa_db_warnings=[], qa_api_warnings=[], qa_prd_warnings=[],
            validation_error_codes=sorted({detail["code"] for detail in details}),
            validation_repair_targets=[], validation_blockers=messages,
            validation_unresolved_blockers=messages,
            last_validation_error="\n".join(messages),
            status_message=f"{state.failed_stage} 생성 실패 — 이후 단계는 실행되지 않았습니다",
        )

    def validate(self, state: PipelineState) -> PipelineState:
        from phase2.validation_contract import (
            artifact_revision,
            dedupe_issues,
            issue_detail,
        )
        if state.failed_stage in GENERATION_HALT_STAGES and state.generation_blockers:
            if state.failed_stage != "FEATURE_SPEC":
                return self._halted_generation(state)
            # Missing downstream artifacts are consequences, not independent
            # repair targets. The producer already exhausted its own budget.
            diagnostic = next((item for item in reversed(state.generation_diagnostics) if item.get("issues")), {})
            refs = {item["index"]: item for item in diagnostic.get("featureRefs") or [] if isinstance(item, dict) and "index" in item}
            details = []
            for message in dict.fromkeys(diagnostic.get("issues") or state.generation_blockers):
                pointer = re.search(r"/(?:features|tables|schemas)(?:/[A-Za-z0-9_~.-]+)*", message)
                index = re.search(r"/features/(\d+)", message)
                feature = refs.get(int(index[1]), {}) if index else {}
                fid = feature.get("featureId") or feature.get("sourceFeatureId") or ""
                details.append({**issue_detail(message.split(":", 1)[0], "feature_spec", message,
                                               pointer=pointer[0] if pointer else "", featureId=fid, artifactKey=fid or "feature_spec"),
                                "repairable": False, "failedStage": "FEATURE_SPEC"})
            messages = [item["message"] for item in details]
            return state.copy(
                validated=False, qa_approved=False, qa_blocker_details=details,
                qa_issue_details=details, qa_repair_issues=[],
                qa_api_blockers=[], qa_db_blockers=[], qa_prd_blockers=messages,
                validation_error_codes=["FEATURE_SPEC_CONTRACT_BLOCKER"],
                validation_repair_targets=[], validation_blockers=messages,
                validation_unresolved_blockers=messages, last_validation_error="\n".join(messages),
                status_message="기능명세 생성 실패 — API/DBA는 실행되지 않았습니다",
            )
        result = self._validator.validate(
            feature_list=state.feature_list, db_schema=state.db_schema, api_spec=state.api_spec,
            prd_document=state.prd_document, feature_registry=state.feature_registry, feature_specs=state.feature_specs,
        )
        normalized = state.copy(
            db_schema=result.normalized_db_schema or state.db_schema,
            api_spec=result.normalized_api_spec or state.api_spec,
            prd_document=result.normalized_prd_document or state.prd_document,
        )
        revision = artifact_revision(normalized.db_schema, normalized.api_spec, normalized.prd_document, normalized.feature_registry)
        details = list(result.blocker_details)
        if not details and result.errors:
            details = [issue_detail(result.error_codes[index] if index < len(result.error_codes) else "VALIDATION_ERROR",
                                    result.repair_targets[0] if result.repair_targets else "prd", message)
                       for index, message in enumerate(result.errors)]
        for message in state.generation_blockers:
            domain = "db" if message.startswith("DBA_") else "api" if message.startswith("API_") else "prd"
            details.append(issue_detail(message.split(":", 1)[0], domain, message))
        # Compatibility for external/manual QA. Revision-bound pipeline QA is
        # always recomputed above, never copied from a previous snapshot.
        if not state.qa_validation_revision:
            for domain in ("db", "api", "prd"):
                warnings = set(getattr(state, f"qa_{domain}_warnings"))
                for message in getattr(state, f"qa_{domain}_blockers") or getattr(state, f"qa_{domain}_issues"):
                    if message not in warnings and classify_issue(message) != IssueSeverity.WARNING:
                        details.append(issue_detail("MANUAL_QA", domain, message))
            for detail in state.qa_issue_details:
                if detail.get("severity") in {"ERROR", "BLOCKER"}:
                    domain = {"DBA": "db", "API": "api", "PRD": "prd"}.get(detail.get("agent"), "prd")
                    details.append(issue_detail("MANUAL_QA", domain, f"QA {detail['severity']}: {detail.get('reason', '')}"))
        details = dedupe_issues([{**detail, "revision": revision} for detail in details])
        blockers = [detail for detail in details if detail.get("severity") not in {"WARNING", "INFO"}]
        messages = self._canonical_blockers([detail.get("message", "") for detail in blockers])
        previous = self._canonical_blockers(state.validation_unresolved_blockers or state.validation_blockers)
        resolved = self._canonical_blockers([*state.validation_resolved_blockers, *(value for value in previous if value not in messages)])
        targets = sorted({detail.get("repairTarget") or detail.get("domain") for detail in blockers
                          if detail.get("repairable", True)})
        fields = {}
        for domain in ("db", "api", "prd"):
            domain_details = [detail for detail in details if detail.get("domain") == domain]
            fields[f"qa_{domain}_issues"] = [detail["message"] for detail in domain_details]
            fields[f"qa_{domain}_blockers"] = [detail["message"] for detail in domain_details if detail.get("severity") not in {"WARNING", "INFO"}]
            fields[f"qa_{domain}_warnings"] = [detail["message"] for detail in domain_details if detail.get("severity") == "WARNING"]
        return normalized.copy(
            **fields, openapi_spec=json.dumps(build_openapi_document(normalized.api_spec, state.project_name), ensure_ascii=False),
            validated=not blockers, qa_approved=not blockers, qa_validation_revision=revision,
            qa_blocker_details=blockers, qa_issue_details=details, qa_repair_issues=blockers,
            validation_error_codes=sorted({detail["code"] for detail in blockers}),
            validation_repair_targets=targets, validation_blockers=messages,
            validation_unresolved_blockers=messages, validation_resolved_blockers=resolved,
            last_validation_error="\n".join(messages),
            status_message="Phase 3 완료 — 검증 통과" if not blockers else "Phase 3 검증 실패 — 현재 계약 blocker",
        )
