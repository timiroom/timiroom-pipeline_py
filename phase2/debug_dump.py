"""Phase2 에이전트의 원문 LLM 응답과 단계별 state 스냅샷을 파일로 기록하는 진단용 덤프.

재생성/파싱 실패의 원인을 로그 메시지만으로는 알 수 없어(예: '파싱 실패'라고만 찍히고
실제 EXAONE이 뭘 뱉었는지는 안 남음) orchestration_graph.py의 dump.log_raw(...) 호출부에
연결해서 쓴다. 프로덕션 기본값은 dump=None(비활성화) — 필요할 때만 켠다.
"""
import json
import logging
import os
import time
from dataclasses import asdict, is_dataclass

logger = logging.getLogger(__name__)


class PipelineDump:

    def __init__(self, pipeline_id: str, base_dir: str):
        self._dir = os.path.join(base_dir, (pipeline_id or "unknown")[:8])
        os.makedirs(self._dir, exist_ok=True)
        self._raw_path = os.path.join(self._dir, "raw_calls.log")
        self._state_path = os.path.join(self._dir, "states.log")
        self._seq = 0
        logger.info("PipelineDump 활성화 — %s", self._dir)

    def log_raw(self, label: str, attempt: int, raw: str) -> None:
        self._seq += 1
        with open(self._raw_path, "a", encoding="utf-8") as f:
            f.write(
                f"\n{'=' * 100}\n"
                f"[{self._seq:04d}] {time.strftime('%H:%M:%S')} label={label} attempt={attempt} len={len(raw)}\n"
                f"{'=' * 100}\n"
            )
            f.write(raw or "(빈 응답)")
            f.write("\n")

    def log_state(self, stage: str, state) -> None:
        with open(self._state_path, "a", encoding="utf-8") as f:
            f.write(f"\n{'=' * 100}\n[{time.strftime('%H:%M:%S')}] STAGE={stage}\n{'=' * 100}\n")
            try:
                data = asdict(state) if is_dataclass(state) else vars(state)
                f.write(json.dumps(data, ensure_ascii=False, indent=2, default=str))
            except Exception as e:
                f.write(f"(state 직렬화 실패: {e})\n{state!r}")
            f.write("\n")

    def close(self, final_state) -> None:
        logger.info("PipelineDump 종료 — %s", self._dir)
