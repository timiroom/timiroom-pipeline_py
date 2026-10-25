import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from phase1.document_ingestion import DocumentIngestionService
from phase4.storage_receipt import StorageReceiptStore


def connection(monkeypatch, row):
    cursor = MagicMock()
    cursor.fetchone.return_value = row
    conn = MagicMock()
    conn.cursor.return_value.__enter__.return_value = cursor
    monkeypatch.setattr("phase4.storage_receipt.psycopg2.connect", lambda *args, **kwargs: conn)
    conn.__enter__.return_value = conn
    return cursor


def test_manifest_keys_are_stable_unique_and_content_scoped():
    documents = [("text", {"pipeline_id": "p", "type": "PRD"})]
    keys = DocumentIngestionService.event_source_keys(documents)
    assert keys == DocumentIngestionService.event_source_keys(documents * 2)
    assert keys != DocumentIngestionService.event_source_keys([("other", documents[0][1])])
    assert keys != DocumentIngestionService.event_source_keys([("text", {"pipeline_id": "q", "type": "PRD"})])


def test_conflicting_manifest_never_replaces_expected_chunks(monkeypatch):
    cursor = connection(monkeypatch, None)
    with pytest.raises(ValueError, match="STORAGE_MANIFEST_CONFLICT"):
        StorageReceiptStore("db", object())._prepare("p", ["key"])
    statement, values = cursor.execute.call_args.args
    assert "WHERE pipeline_storage_receipts.source_keys=EXCLUDED.source_keys" in statement
    assert values[0] == "p"


def test_recreated_verifier_reads_manifest_and_actual_committed_keys(monkeypatch):
    connection(monkeypatch, (["a", "b"], True))
    ingestion = MagicMock()
    ingestion.storage_status_for_keys = AsyncMock(side_effect=[
        {"status": "pending", "expectedChunks": 2, "storedChunks": 1},
        {"status": "stored", "expectedChunks": 2, "storedChunks": 2},
    ])
    assert asyncio.run(StorageReceiptStore("db", ingestion).status("p"))["status"] == "pending"
    recovered = asyncio.run(StorageReceiptStore("db", ingestion).status("p"))
    assert recovered == {"status": "stored", "expectedChunks": 2, "storedChunks": 2, "kafkaPublished": True}
    ingestion.storage_status_for_keys.assert_awaited_with(["a", "b"])


def test_unpublished_or_unknown_manifest_cannot_confirm_completion(monkeypatch):
    connection(monkeypatch, (["a"], False))
    ingestion = MagicMock()
    ingestion.storage_status_for_keys = AsyncMock(return_value={"status": "stored"})
    assert asyncio.run(StorageReceiptStore("db", ingestion).status("p")) == {"status": "pending", "kafkaPublished": False}
    connection(monkeypatch, None)
    assert asyncio.run(StorageReceiptStore("db", ingestion).status("missing")) == {"status": "unknown", "kafkaPublished": False}
