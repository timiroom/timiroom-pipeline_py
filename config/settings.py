import re

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def validate_sql_identifier(value: str) -> str:
    """환경변수로 받은 테이블명이 SQL 식별자로 안전한지 확인한다."""
    if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", value) is None:
        raise ValueError("RAG_DOCUMENT_TABLE은 영문, 숫자, 밑줄로만 구성해야 합니다")
    return value


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # ── OpenAI ────────────────────────────────────────────────────
    openai_api_key: str = ""
    openai_base_url: str = "https://api.openai.com/v1"
    openai_chat_model: str = "gpt-5.4-mini"
    openai_request_timeout_seconds: float = 120.0
    openai_max_retries: int = 2

    # ── Phase 2 ───────────────────────────────────────────────────
    phase2_llm_max_concurrency: int = 8
    phase2_timeout_seconds: float = 900.0
    phase2_dba_resync_timeout_seconds: float = 120.0
    phase2_api_resync_timeout_seconds: float = 120.0
    phase2_web_search_enabled: bool = True

    # ── Upstage Solar 임베딩 ─────────────────────────────────────
    upstage_api_key: str = ""
    solar_embedding_query_model: str = "solar-embedding-2-query"
    solar_embedding_passage_model: str = "solar-embedding-2-passage"

    # ── Cohere Rerank ─────────────────────────────────────────────
    cohere_api_key: str = ""
    cohere_rerank_model: str = "rerank-v4.0-pro"
    cohere_base_url: str = "https://api.cohere.com"
    cohere_max_concurrency: int = 5

    # ── PostgreSQL ────────────────────────────────────────────────
    db_url: str = "postgresql://localhost:5432/timiroom"
    # Solar Embedding 2의 출력 차원은 1024다.
    rag_document_table: str = "document_chunks"

    # ── Kafka ─────────────────────────────────────────────────────
    kafka_bootstrap_servers: str = "localhost:9092"
    kafka_topic_pipeline_result: str = "rag.pipeline.result"
    kafka_topic_dead_letter: str = "rag.pipeline.result.DLT"
    kafka_consumer_group_id: str = "rag-pipeline-group"
    kafka_publish_max_retry: int = 3
    kafka_max_poll_interval_ms: int = 900000

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
    embedding_max_concurrency: int = 8
    embedding_batch_size: int = 64
    rag_db_max_concurrency: int = 10

    # ── 검증 ─────────────────────────────────────────────────────
    validation_max_retry: int = 3
    phase3_targeted_repair_max_per_domain: int = 1
    phase3_repair_timeout_seconds: float = 300.0

    # ── CORS ──────────────────────────────────────────────────────
    allowed_origins: str = "http://localhost:3000,http://localhost:5500,http://127.0.0.1:5500"

    @model_validator(mode="after")
    def validate_runtime_limits(self) -> "Settings":
        if self.rag_chunk_size <= 0:
            raise ValueError("RAG_CHUNK_SIZE는 1 이상이어야 합니다")
        if self.rag_chunk_overlap < 0 or self.rag_chunk_overlap >= self.rag_chunk_size:
            raise ValueError("RAG_CHUNK_OVERLAP은 0 이상 RAG_CHUNK_SIZE 미만이어야 합니다")
        if self.embedding_max_concurrency <= 0:
            raise ValueError("EMBEDDING_MAX_CONCURRENCY는 1 이상이어야 합니다")
        if self.embedding_batch_size <= 0:
            raise ValueError("EMBEDDING_BATCH_SIZE는 1 이상이어야 합니다")
        if self.rag_db_max_concurrency <= 0:
            raise ValueError("RAG_DB_MAX_CONCURRENCY는 1 이상이어야 합니다")
        if self.cohere_max_concurrency <= 0:
            raise ValueError("COHERE_MAX_CONCURRENCY는 1 이상이어야 합니다")
        if self.openai_request_timeout_seconds <= 0:
            raise ValueError("OPENAI_REQUEST_TIMEOUT_SECONDS는 0보다 커야 합니다")
        if self.openai_max_retries < 0:
            raise ValueError("OPENAI_MAX_RETRIES는 0 이상이어야 합니다")
        if self.phase2_llm_max_concurrency <= 0:
            raise ValueError("PHASE2_LLM_MAX_CONCURRENCY는 1 이상이어야 합니다")
        if self.phase2_timeout_seconds <= 0:
            raise ValueError("PHASE2_TIMEOUT_SECONDS는 0보다 커야 합니다")
        if self.phase2_dba_resync_timeout_seconds <= 0:
            raise ValueError("PHASE2_DBA_RESYNC_TIMEOUT_SECONDS는 0보다 커야 합니다")
        if self.phase2_api_resync_timeout_seconds <= 0:
            raise ValueError("PHASE2_API_RESYNC_TIMEOUT_SECONDS는 0보다 커야 합니다")
        if self.phase3_repair_timeout_seconds <= 0:
            raise ValueError("PHASE3_REPAIR_TIMEOUT_SECONDS는 0보다 커야 합니다")
        if self.phase3_targeted_repair_max_per_domain <= 0:
            raise ValueError("PHASE3_TARGETED_REPAIR_MAX_PER_DOMAIN은 1 이상이어야 합니다")
        if self.kafka_publish_max_retry <= 0:
            raise ValueError("KAFKA_PUBLISH_MAX_RETRY는 1 이상이어야 합니다")
        if self.kafka_max_poll_interval_ms <= 0:
            raise ValueError("KAFKA_MAX_POLL_INTERVAL_MS는 1 이상이어야 합니다")
        return self

    def get_allowed_origins(self) -> list[str]:
        return [o.strip() for o in self.allowed_origins.split(",")]

    def get_rag_document_table(self) -> str:
        return validate_sql_identifier(self.rag_document_table)


settings = Settings()
