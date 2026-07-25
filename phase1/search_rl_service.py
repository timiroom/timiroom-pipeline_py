import logging
import random
import threading
from collections import deque
from dataclasses import dataclass

import psycopg2

logger = logging.getLogger(__name__)

ALPHA = 0.1
EPSILON = 0.15
BASELINE_WINDOW = 20
EXPLORE_SIGMA = 0.05
THRESHOLD_MIN = 0.1
THRESHOLD_MAX = 0.6


@dataclass(frozen=True)
class SearchParams:
    vector_weight: float
    keyword_weight: float
    similarity_threshold: float


def _clamp(val: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, val))


class SearchRLService:
    """Phase1 similarity_threshold 자동 튜닝 — Epsilon-Greedy + Gaussian + EMA.

    rag-pipeline의 SearchRLService(Java)와 동일한 알고리즘.
    vector_weight/keyword_weight는 1.0 고정, similarity_threshold만 튜닝한다.
    보상 신호는 리랭커 평균 관련도 점수(0.0~1.0), baseline은 최근 20회 슬라이딩 윈도우 평균.
    """

    def __init__(self, db_url: str):
        self._db_url = db_url
        self._lock = threading.Lock()
        self._baseline_window: deque[float] = deque(maxlen=BASELINE_WINDOW)
        self._current_threshold = 0.3
        self._load_from_db()

    def _get_conn(self):
        return psycopg2.connect(self._db_url)

    def _load_from_db(self) -> None:
        try:
            conn = self._get_conn()
            try:
                with conn.cursor() as cur:
                    cur.execute("SELECT similarity_threshold FROM rl_params WHERE id = 1")
                    row = cur.fetchone()
                    if row and row[0] is not None:
                        self._current_threshold = float(row[0])
                        logger.info("Phase1 RL 파라미터 로드 — threshold:%.4f", self._current_threshold)
            finally:
                conn.close()
        except Exception as e:
            logger.warning("Phase1 RL 파라미터 로드 실패 — 기본값 사용: %s", e)

    def get_params(self) -> SearchParams:
        with self._lock:
            base = self._current_threshold
        if random.random() < EPSILON:
            noise = random.gauss(0, EXPLORE_SIGMA)
            threshold = _clamp(round((base + noise) * 10000.0) / 10000.0, THRESHOLD_MIN, THRESHOLD_MAX)
            logger.debug("Phase1 RL 탐색 — base=%.4f noise=%.4f → explored=%.4f", base, noise, threshold)
        else:
            threshold = base
        return SearchParams(1.0, 1.0, threshold)

    def get_current_params(self) -> SearchParams:
        with self._lock:
            return SearchParams(1.0, 1.0, self._current_threshold)

    def log_search(self, pipeline_id: str | None, params: SearchParams, chunk_count: int) -> None:
        if not pipeline_id:
            return
        try:
            conn = self._get_conn()
            try:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO rl_search_log
                            (pipeline_id, vector_weight, keyword_weight, similarity_threshold, chunk_count)
                        VALUES (%s, %s, %s, %s, %s)
                        ON CONFLICT (pipeline_id) DO UPDATE
                            SET similarity_threshold = EXCLUDED.similarity_threshold,
                                chunk_count          = EXCLUDED.chunk_count
                        """,
                        (pipeline_id, params.vector_weight, params.keyword_weight,
                         params.similarity_threshold, chunk_count),
                    )
                conn.commit()
            finally:
                conn.close()
        except Exception as e:
            logger.warning("Phase1 RL 로그 저장 실패 — pipelineId: %s, 원인: %s", pipeline_id, e)

    def apply_rerank_score(self, pipeline_id: str | None, avg_score: float) -> None:
        """리랭커 평균 관련도 점수로 threshold 업데이트.

        advantage = avg_score - baseline
        new_threshold = current + sign(advantage) * ALPHA * |advantage| * (used - current)
        """
        if not pipeline_id:
            return

        with self._lock:
            self._baseline_window.append(avg_score)
            baseline = sum(self._baseline_window) / len(self._baseline_window)
        advantage = avg_score - baseline

        logger.info("Phase1 RL 피드백 — score:%.3f baseline:%.3f advantage:%.3f",
                     avg_score, baseline, advantage)

        try:
            conn = self._get_conn()
            try:
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT similarity_threshold FROM rl_search_log WHERE pipeline_id = %s",
                        (pipeline_id,),
                    )
                    row = cur.fetchone()
                    if not row or row[0] is None:
                        logger.warning("Phase1 RL 로그 없음 — 업데이트 스킵: %s", pipeline_id)
                        return
                    used_threshold = float(row[0])

                    cur.execute(
                        "UPDATE rl_search_log SET avg_cohere_score = %s WHERE pipeline_id = %s",
                        (avg_score, pipeline_id),
                    )

                    with self._lock:
                        current = self._current_threshold
                        sign = 1.0 if advantage >= 0 else -1.0
                        new_threshold = _clamp(
                            current + sign * ALPHA * abs(advantage) * (used_threshold - current),
                            THRESHOLD_MIN, THRESHOLD_MAX,
                        )
                        self._current_threshold = new_threshold

                    cur.execute(
                        """
                        UPDATE rl_params SET
                            similarity_threshold = %s,
                            total_runs           = total_runs + 1,
                            updated_at           = NOW()
                        WHERE id = 1
                        """,
                        (new_threshold,),
                    )
                conn.commit()
                logger.info("Phase1 RL 업데이트 — advantage:%.3f → threshold:%.4f", advantage, new_threshold)
            finally:
                conn.close()
        except Exception as e:
            logger.error("Phase1 RL 업데이트 실패 — pipelineId: %s, 원인: %s", pipeline_id, e)
