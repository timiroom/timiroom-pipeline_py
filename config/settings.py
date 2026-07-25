from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # ── Upstage Solar 임베딩 ─────────────────────────────────────
    # query/passage 전용 모델이 분리되어 있음 (동일 벡터 공간)
    upstage_api_key: str = ""
    solar_embedding_query_model: str = "solar-embedding-2-query"
    solar_embedding_passage_model: str = "solar-embedding-2-passage"

    # ── K-EXAONE (Friendli.ai) ────────────────────────────────────
    exaone_api_key: str = ""
    exaone_endpoint_id: str = ""

    # ── PostgreSQL ────────────────────────────────────────────────
    db_url: str = "postgresql://localhost:5432/timiroom"

    # ── Kafka ─────────────────────────────────────────────────────
    kafka_bootstrap_servers: str = "localhost:9092"
    kafka_topic_pipeline_result: str = "rag.pipeline.result"
    kafka_topic_dead_letter: str = "rag.pipeline.result.DLT"
    kafka_consumer_group_id: str = "rag-pipeline-group"

    # ── 로컬 Ko-Reranker ──────────────────────────────────────────
    # Dongjin-kr/ko-reranker: bge-reranker-large 기반 한국어 파인튜닝
    # 빈 문자열로 설정 시 비활성화
    ko_reranker_model: str = "Dongjin-kr/ko-reranker"

    # ── RAG ───────────────────────────────────────────────────────
    rag_chunk_size: int = 512
    rag_chunk_overlap: int = 64
    rag_top_k_vector: int = 20
    rag_top_k_keyword: int = 20
    rag_top_k_final: int = 5
    rag_reranker_enabled: bool = True
    rag_similarity_threshold: float = 0.3
    rag_min_threshold: float = 0.1
    rag_min_results: int = 5
    rag_threshold_step: float = 0.1

    # ── Agent 타임아웃 (초) ───────────────────────────────────────
    agent_stream_timeout: int = 90
    agent_sync_timeout: int = 60

    # ── 검증 ─────────────────────────────────────────────────────
    validation_max_retry: int = 3

    # ── CORS ──────────────────────────────────────────────────────
    allowed_origins: str = "http://localhost:3000,http://localhost:5500,http://127.0.0.1:5500"

    def get_allowed_origins(self) -> list[str]:
        return [o.strip() for o in self.allowed_origins.split(",")]


settings = Settings()
