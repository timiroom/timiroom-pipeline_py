import pytest

from config.settings import Settings, validate_sql_identifier


@pytest.mark.parametrize(
    "value",
    ["document_chunks", "document_chunks_ko", "rag2026"],
)
def test_validate_sql_identifier_accepts_safe_names(value: str) -> None:
    assert validate_sql_identifier(value) == value


@pytest.mark.parametrize("value", ["", "1table", "table-name", "table name", "x;drop"])
def test_validate_sql_identifier_rejects_unsafe_names(value: str) -> None:
    with pytest.raises(ValueError):
        validate_sql_identifier(value)


def test_model_provider_defaults() -> None:
    settings = Settings(_env_file=None)

    assert settings.openai_base_url == "https://api.openai.com/v1"
    assert settings.openai_chat_model == "gpt-5.4-mini"
    assert settings.solar_embedding_query_model == "solar-embedding-2-query"
    assert settings.solar_embedding_passage_model == "solar-embedding-2-passage"
    assert settings.cohere_rerank_model == "rerank-v4.0-pro"


@pytest.mark.parametrize(
    "overrides",
    [
        {"rag_chunk_size": 0},
        {"rag_chunk_size": 100, "rag_chunk_overlap": 100},
        {"rag_chunk_overlap": -1},
        {"embedding_max_concurrency": 0},
        {"embedding_batch_size": 0},
        {"rag_db_max_concurrency": 0},
        {"cohere_max_concurrency": 0},
        {"openai_request_timeout_seconds": 0},
        {"phase2_llm_max_concurrency": 0},
        {"phase2_timeout_seconds": 0},
        {"kafka_publish_max_retry": 0},
        {"kafka_max_poll_interval_ms": 0},
    ],
)
def test_invalid_runtime_limits_are_rejected(overrides: dict) -> None:
    with pytest.raises(ValueError):
        Settings(_env_file=None, **overrides)
