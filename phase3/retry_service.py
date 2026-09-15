import asyncio
import hashlib
import json
import logging

from phase2.json_utils import try_parse_json
from phase2.state import PipelineState

logger = logging.getLogger(__name__)

_FEEDBACK_MARKER = "\n\n[PHASE3_VALIDATION_FEEDBACK]\n"


class RetryService:

    def __init__(self, max_retry: int = 3):
        if max_retry <= 0:
            raise ValueError("max_retry는 1 이상이어야 합니다")
        self._max_retry = max_retry

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

        fingerprint = self._failure_fingerprint(state)
        if fingerprint in state.validation_failure_fingerprints:
            logger.warning("동일 산출물과 검증 오류가 반복되어 재시도를 조기 중단")
            return state.copy(status_message="Human-in-the-Loop 필요 — 동일 검증 실패 반복")

        logger.info("Phase 3 재시도 시작 (%d/%d)", current_retry, self._max_retry)

        retry_prompt = self._build_retry_prompt(state)

        # feature_list/project_name/platform 등 기존 state를 그대로 보존하고
        # context_prompt만 재시도 피드백으로 교체 — 문자열만으로 새 state를 만들면
        # Phase1/2에서 이미 구축된 정보가 전부 소실된다.
        retry_state = state.copy(
            context_prompt=retry_prompt,
            retry_count=current_retry,
            validation_failure_fingerprints=[*state.validation_failure_fingerprints, fingerprint],
        )
        try:
            result = await orchestration_graph.repair(
                retry_state,
                state.last_validation_error or "",
                pipeline_id,
                repair_targets=state.validation_repair_targets,
            )
        except (asyncio.TimeoutError, TimeoutError) as exc:
            logger.error("Phase 3 선택적 복구 타임아웃 (%d/%d): %s", current_retry, self._max_retry, exc)
            return retry_state.copy(
                validated=False,
                status_message="Phase 3 복구 타임아웃 — 관리자 검토 필요",
                last_validation_error=(state.last_validation_error or "") + "\nREPAIR_TIMEOUT",
            )

        validated = validation_service.validate(result)

        if validated.validated:
            logger.info("Phase 3 재시도 성공 (%d/%d)", current_retry, self._max_retry)
            return validated

        next_fingerprint = self._failure_fingerprint(validated)
        if next_fingerprint in validated.validation_failure_fingerprints:
            logger.warning("복구 직후 동일 검증 오류가 반복되어 추가 재시도를 생략")
            return validated.copy(status_message="Human-in-the-Loop 필요 — 동일 검증 실패 반복")

        return await self.retry_with(validated, orchestration_graph, validation_service, pipeline_id)

    @staticmethod
    def _failure_fingerprint(state: PipelineState) -> str:
        # 결과 JSON은 LLM의 사소한 순서/표현 변화로 매번 달라진다. 오류 코드·대상·
        # 정규화된 메시지만 fingerprint에 포함해야 같은 결함의 무한 재시도를 막을 수 있다.
        error = " ".join((state.last_validation_error or "").split())
        codes = ",".join(sorted(set(state.validation_error_codes)))
        targets = ",".join(sorted(set(state.validation_repair_targets)))
        payload = "\x1f".join([codes, targets, error[:6000]])
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

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
