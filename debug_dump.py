"""파이프라인 실행 결과를 txt 파일에 저장하는 디버그 덤프 모듈."""
from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

from phase2.state import PipelineState

logger = logging.getLogger(__name__)

DUMP_DIR = Path(__file__).parent / "pipeline_dumps"
SEP = "=" * 70


class PipelineDump:
    """파이프라인 한 번 실행의 에이전트 raw 응답 + 상태를 단일 txt 파일에 기록."""

    def __init__(self, pipeline_id: str):
        DUMP_DIR.mkdir(exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:19]  # 밀리초 포함 → 충돌 방지
        safe_id = str(pipeline_id or "unknown")[:20].replace("/", "_").replace("\\", "_")
        self._path = DUMP_DIR / f"pipeline_{ts}_{safe_id}.txt"
        self._f = open(self._path, "w", encoding="utf-8")
        self._w(SEP)
        self._w(f"PIPELINE START  |  id={pipeline_id}")
        self._w(f"시작: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self._w(SEP + "\n")
        logger.info("파이프라인 덤프 파일 생성: %s", self._path)

    # ── raw 응답 기록 ────────────────────────────────────────────────
    def log_raw(self, agent: str, attempt: int, raw: str) -> None:
        self._w(f"\n{SEP}")
        self._w(f"[{agent}] RAW RESPONSE  |  attempt={attempt}  |  len={len(raw)}")
        self._w(SEP)
        self._w(raw)
        self._flush()

    # ── 에이전트 완료 후 state 스냅샷 ────────────────────────────────
    def log_state(self, label: str, state: PipelineState) -> None:
        self._w(f"\n{SEP}")
        self._w(f"[{label}] STATE SNAPSHOT  |  {datetime.now().strftime('%H:%M:%S')}")
        self._w(f"status: {state.status_message}")
        self._w(SEP)

        if state.feature_list:
            self._w(f"\n▶ feature_list ({len(state.feature_list)}개)")
            for i, f in enumerate(state.feature_list, 1):
                self._w(f"  {i}. {f}")

        if state.dba_instruction:
            self._section("dba_instruction", state.dba_instruction)

        if state.api_instruction:
            self._section("api_instruction", state.api_instruction)

        if state.market_research:
            self._section(f"market_research ({len(state.market_research)}자)", state.market_research)

        if state.prd_document and state.prd_document not in ("{}", ""):
            self._section(f"prd_document ({len(state.prd_document)}자)", state.prd_document)

        if state.db_schema:
            self._section(f"db_schema ({len(state.db_schema)}자)", state.db_schema)

        if state.api_spec:
            self._section(f"api_spec ({len(state.api_spec)}자)", state.api_spec)

        if state.last_validation_error:
            self._section("⚠ last_validation_error", state.last_validation_error)

        if state.prd_feedback_from_dba:
            self._section("prd_feedback_from_dba", state.prd_feedback_from_dba)

        if state.prd_feedback_from_api:
            self._section("prd_feedback_from_api", state.prd_feedback_from_api)

        self._flush()

    # ── 완료 처리 ────────────────────────────────────────────────────
    def close(self, final_state: PipelineState) -> None:
        self._w(f"\n{SEP}")
        self._w(f"PIPELINE END  |  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self._w(SEP)
        self.log_state("FINAL", final_state)
        self._f.flush()
        self._f.close()
        logger.info("파이프라인 덤프 완료: %s", self._path)

    # ── 내부 헬퍼 ───────────────────────────────────────────────────
    def _section(self, title: str, content: str) -> None:
        self._w(f"\n▶ {title}")
        self._w("-" * 50)
        self._w(content)

    def _w(self, text: str) -> None:
        self._f.write(text + "\n")

    def _flush(self) -> None:
        self._f.flush()
