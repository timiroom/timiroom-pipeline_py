import asyncio
import hashlib
import json
import logging

from phase2.json_utils import try_parse_json
from phase2.state import PipelineState

logger = logging.getLogger(__name__)

_FEEDBACK_MARKER = "\n\n[PHASE3_VALIDATION_FEEDBACK]\n"


class RetryService:

    def __init__(self, max_retry: int = 3, max_targeted_repair_per_domain: int = 1):
        if max_retry <= 0:
            raise ValueError("max_retry는 1 이상이어야 합니다")
        if max_targeted_repair_per_domain <= 0:
            raise ValueError("max_targeted_repair_per_domain은 1 이상이어야 합니다")
        self._max_retry = max_retry
        self._max_targeted_repair_per_domain = max_targeted_repair_per_domain

    async def retry_with(
        self,
        state: PipelineState,
        orchestration_graph,
        validation_service,
        pipeline_id: str | None = None,
    ) -> PipelineState:
        current_retry = state.retry_count + 1

        if current_retry > self._max_retry:
            logger.warning("최대 재시도 횟수 초과 (%d/%d)", current_retry, self._max_retry)
            return state.copy(status_message="Human-in-the-Loop 필요 — 관리자 검토 요청")

        terminal_generation_blockers = [
            item for item in state.generation_blockers
            if str(item).startswith((
                "API_GENERATION_BLOCKER:",
                "DBA_GENERATION_BLOCKER:",
                "FEATURE_SPEC_CONTRACT_BLOCKER:",
                "REGISTRY_CONTRACT_BLOCKER:",
            ))
        ]
        if terminal_generation_blockers:
            logger.warning(
                "불변 generation blocker 감지 — Phase3 repair 재시도 중단: %s",
                terminal_generation_blockers,
            )
            return state.copy(
                validated=False,
                last_validation_error=(state.last_validation_error or "")
                + "\nGENERATION_BLOCKER_RETRY_SUPPRESSED",
                status_message="Human-in-the-Loop 필요 — 생성 blocker는 자동 재시도하지 않음",
            )

        fingerprint = self._failure_fingerprint(state)

        if fingerprint in state.validation_failure_fingerprints:
            logger.warning("동일 blocker fingerprint 반복 — 추가 repair 중단")
            return state.copy(
                validated=False,
                last_validation_error=(state.last_validation_error or "")
                + "\nSAME_BLOCKER_REPEATED",
                status_message="동일 blocker 재발 — 추가 자동 repair 중단, Phase3 blocker 확정",
            )

        targets = set(state.validation_repair_targets or [])
        exhausted = sorted(
            target for target in targets
            if state.targeted_repair_attempts.get(target, 0)
            >= self._max_targeted_repair_per_domain
        )
        if exhausted:
            message = "TARGETED_REPAIR_BUDGET_EXHAUSTED: " + ", ".join(exhausted)
            logger.warning("%s", message)
            return state.copy(
                validated=False,
                last_validation_error=(state.last_validation_error or "") + "\n" + message,
                status_message="targeted repair 예산 초과 — Phase3 blocker 확정",
            )

        logger.info("Phase 3 재시도 시작 (%d/%d)", current_retry, self._max_retry)

        retry_prompt = self._build_retry_prompt(state)
        repair_feedback = self._structured_feedback(state)

        # feature_list/project_name/platform 등 기존 state를 그대로 보존하고
        # context_prompt만 재시도 피드백으로 교체 — 문자열만으로 새 state를 만들면
        # Phase1/2에서 이미 구축된 정보가 전부 소실된다.
        retry_state = state.copy(
            context_prompt=retry_prompt,
            retry_count=current_retry,
            validation_failure_fingerprints=[*state.validation_failure_fingerprints, fingerprint],
            targeted_repair_attempts={
                **state.targeted_repair_attempts,
                **{
                    target: state.targeted_repair_attempts.get(target, 0) + 1
                    for target in targets
                },
            },
        )
        try:
            result = await orchestration_graph.repair(
                retry_state,
                repair_feedback,
                pipeline_id,
                repair_targets=state.validation_repair_targets,
            )
        except (asyncio.TimeoutError, TimeoutError) as exc:
            logger.error("Phase 3 선택적 복구 타임아웃 (%d/%d): %s", current_retry, self._max_retry, exc)
            timeout_state = retry_state.copy(
                validated=False,
                last_validation_error=(state.last_validation_error or "") + "\nREPAIR_TIMEOUT",
                status_message="Phase 3 복구 타임아웃 — 재시도 대기",
            )
            if current_retry >= self._max_retry:
                return timeout_state.copy(
                    status_message="Human-in-the-Loop 필요 — 최대 재시도 후 복구 타임아웃"
                )
            return await self.retry_with(
                timeout_state, orchestration_graph, validation_service, pipeline_id
            )

        validated = validation_service.validate(result)

        if validated.validated:
            logger.info("Phase 3 재시도 성공 (%d/%d)", current_retry, self._max_retry)
            return validated

        return await self.retry_with(validated, orchestration_graph, validation_service, pipeline_id)

    @staticmethod
    def _failure_fingerprint(state: PipelineState) -> str:
        # 결과 JSON은 LLM의 사소한 순서/표현 변화로 매번 달라진다. 오류 코드·대상·
        # 정규화된 메시지만 fingerprint에 포함해야 같은 결함의 무한 재시도를 막을 수 있다.
        codes = ",".join(sorted(set(state.validation_error_codes)))
        targets = ",".join(sorted(set(state.validation_repair_targets)))
        blockers = json.dumps(
            state.validation_unresolved_blockers
            or state.validation_blockers
            or state.qa_blocker_details
            or [],
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        # 자연어 error message는 retry마다 표현이 달라질 수 있으므로 fingerprint에서
        # 제외한다. 동일 blocker를 같은 결함으로 인식해야 반복 repair를 정확히 감지한다.
        payload = "\x1f".join([codes, targets, blockers])
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @staticmethod
    def _canonical_blockers(items: list[str]) -> list[str]:
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
    def _canonical_json(value: str) -> str:
        parsed = try_parse_json(value or "")
        if parsed is None:
            return value or ""
        return json.dumps(parsed, ensure_ascii=False, sort_keys=True, separators=(",", ":"))

    @staticmethod
    def _build_retry_prompt(state: PipelineState) -> str:
        base_prompt = (state.context_prompt or "").split(_FEEDBACK_MARKER, 1)[0]
        feedback = " ".join((state.last_validation_error or "").split())[:6000]
        return (
            base_prompt
            + _FEEDBACK_MARKER
            + feedback
            + "\n\n위 오류를 반드시 수정해서 다시 생성해주세요."
        )

    @staticmethod
    def _structured_feedback(state: PipelineState) -> str:
        """Send only current blockers, with artifact identity, to targeted repair."""
        details = [d for d in (state.qa_blocker_details or []) if isinstance(d, dict)]
        targets = set(state.validation_repair_targets or [])
        if targets:
            details = [d for d in details if d.get("repairTarget") in targets]
        payload = {
            "instruction": "Patch only these blockers; preserve all other artifacts unchanged.",
            "blockers": details,
            "unresolvedBlockers": state.validation_unresolved_blockers or [],
        }
        return (state.last_validation_error or "")[:6000] + "\n\n[STRUCTURED_REPAIR_TARGETS]\n" + json.dumps(payload, ensure_ascii=False)
