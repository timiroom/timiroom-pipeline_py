from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # ── OpenAI ────────────────────────────────────────────────────
    openai_api_key: str
    openai_chat_model: str = "gpt-4o"
    openai_embedding_model: str = "text-embedding-3-large"
    openai_embedding_dimensions: int = 3072

    # ── Anthropic ─────────────────────────────────────────────────
    anthropic_api_key: str = ""
    anthropic_chat_model: str = "claude-sonnet-4-20250514"

    # ── PostgreSQL ────────────────────────────────────────────────
    db_url: str = "postgresql://localhost:5432/timiroom"

    # ── Kafka ─────────────────────────────────────────────────────
    kafka_bootstrap_servers: str = "localhost:9092"
    kafka_topic_pipeline_result: str = "rag.pipeline.result"
    kafka_topic_dead_letter: str = "rag.pipeline.result.DLT"
    kafka_consumer_group_id: str = "rag-pipeline-group"

    # ── Cohere ────────────────────────────────────────────────────
    cohere_api_key: str = ""
    cohere_rerank_model: str = "rerank-english-v3.0"

    # ── RAG ───────────────────────────────────────────────────────
    rag_chunk_size: int = 512
    rag_chunk_overlap: int = 64
    rag_top_k_vector: int = 20
    rag_top_k_keyword: int = 20
    rag_top_k_final: int = 5
    rag_reranker_enabled: bool = True

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
