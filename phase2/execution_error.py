"""Preserve the last trustworthy artifact snapshot across stage exceptions."""

from phase2.state import PipelineState

STAGES = ("SEARCH", "PM", "PRD", "FEATURE_SPEC", "DBA_API", "QA", "PHASE3", "PHASE4")
# Stages that run before any DB/API artifact is attempted. When one of these
# halts, the missing downstream artifacts are consequences of that halt, not
# independent repairable defects, so Phase3 must report the halt alone.
GENERATION_HALT_STAGES = ("SEARCH", "PM", "PRD", "FEATURE_SPEC")


def stage_snapshot(stage: str, state: PipelineState) -> PipelineState:
    """Name the stage that stopped and the stages it therefore never reached."""
    skipped = list(STAGES[STAGES.index(stage) + 1:]) if stage in STAGES else []
    return state.copy(failed_stage=stage, skipped_stages=skipped, validated=False)


class PipelineStageError(RuntimeError):
    def __init__(self, stage: str, state: PipelineState):
        super().__init__(f"Pipeline stage {stage} did not complete")
        self.pipeline_state = stage_snapshot(stage, state)


def mark_stage(error: BaseException, stage: str, state: PipelineState) -> BaseException:
    """Attach a snapshot to an exception that has to be re-raised unchanged.

    A cancelled stage must keep raising CancelledError so asyncio.timeout can
    convert it; replacing it with PipelineStageError would leave the cancelled
    task uncancelled and hide the timeout.
    """
    error.pipeline_state = stage_snapshot(stage, state)
    return error


def state_from_error(error, fallback):
    """asyncio.timeout wraps a cancelled stage; retain that stage's snapshot."""
    seen = set()
    while error is not None and id(error) not in seen:
        seen.add(id(error))
        state = getattr(error, "pipeline_state", None)
        if isinstance(state, PipelineState):
            return state
        error = error.__cause__ or error.__context__
    return fallback
