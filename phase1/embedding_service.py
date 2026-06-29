import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor

import numpy as np

logger = logging.getLogger(__name__)

EMBED_DIM = 1024

try:
    from sentence_transformers import SentenceTransformer
    _ST_AVAILABLE = True
except ImportError:
    _ST_AVAILABLE = False
    logger.warning(
        "sentence_transformers 미설치 — EmbeddingService 비활성화 (Phase 1 skip 모드에서는 영향 없음). "
        "활성화하려면: pip install sentence-transformers"
    )


class EmbeddingService:
    """
    sentence-transformers 기반 로컬 임베딩 서비스.
    sentence_transformers 미설치 시 stub으로 동작 (Phase 1 skip 모드 전용).
    """

    def __init__(self, model_name: str = "nlpai-lab/KURE-v1"):
        if _ST_AVAILABLE:
            logger.info("임베딩 모델 로딩 시작: %s", model_name)
            self._model = SentenceTransformer(model_name)
            self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="embed")
            logger.info("임베딩 모델 로딩 완료")
        else:
            self._model = None
            self._executor = None

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not _ST_AVAILABLE or self._model is None:
            raise RuntimeError("sentence_transformers 미설치 — 임베딩 불가. pip install sentence-transformers")
        loop = asyncio.get_event_loop()
        vecs: np.ndarray = await loop.run_in_executor(
            self._executor,
            lambda: self._model.encode(
                texts,
                normalize_embeddings=True,
                show_progress_bar=False,
            ),
        )
        return vecs.tolist()

    async def embed_one(self, text: str) -> list[float]:
        results = await self.embed([text])
        return results[0]
