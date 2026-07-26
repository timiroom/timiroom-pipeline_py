import pytest

from config.settings import validate_sql_identifier


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
