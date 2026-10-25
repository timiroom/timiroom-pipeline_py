import asyncio
import os
import uuid

import psycopg2
import pytest
from psycopg2 import sql
from psycopg2.extensions import make_dsn

from migrate_db import remove_global_content_hash_uniqueness
from phase1.document_ingestion import DocumentIngestionService
from phase4.storage_receipt import StorageReceiptStore


@pytest.mark.skipif(os.getenv("TIMIROOM_TEST_STORAGE_POSTGRES") != "true", reason="Opt-in PostgreSQL integration test")
def test_hash_index_migration_preserves_source_identity_and_unrelated_tables(monkeypatch):
    from config.settings import settings

    schema = "receipt_test_" + uuid.uuid4().hex
    with psycopg2.connect(settings.db_url) as connection, connection.cursor() as cursor:
        cursor.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema)))
    dsn = make_dsn(settings.db_url, options=f"-c search_path={schema},public")
    try:
        with psycopg2.connect(dsn) as connection, connection.cursor() as cursor:
            cursor.execute("CREATE TABLE chunks (id UUID PRIMARY KEY, content TEXT, content_hash TEXT UNIQUE, source_key TEXT UNIQUE NOT NULL, metadata JSONB, embedding vector(1024), tokens TSVECTOR)")
            cursor.execute("CREATE UNIQUE INDEX legacy_partial_hash ON chunks(content_hash) WHERE content_hash IS NOT NULL")
            cursor.execute("CREATE UNIQUE INDEX arbitrary_hash_name ON chunks(content_hash)")
            cursor.execute("CREATE TABLE unrelated (content_hash TEXT UNIQUE)")
            remove_global_content_hash_uniqueness(cursor, "chunks")
            remove_global_content_hash_uniqueness(cursor, "chunks")
            cursor.execute("SELECT count(*) FROM pg_index WHERE indrelid='unrelated'::regclass AND indisunique")
            assert cursor.fetchone()[0] == 1
        ingestion = DocumentIngestionService(dsn, "chunks", object())
        monkeypatch.setattr(ingestion, "_tokenize_text", lambda text: text)
        records = [("same content", {"pipeline_id": pipeline, "type": kind, "chunk_index": 0})
                   for pipeline, kind in (("first", "prd"), ("first", "api"), ("second", "prd"))]
        assert ingestion._store_records_to_db(records, [[0.0] * 1024] * 3) == 3
        assert ingestion._store_records_to_db(records, [[0.0] * 1024] * 3) == 0
        with psycopg2.connect(dsn) as connection, connection.cursor() as cursor:
            cursor.execute("SELECT count(*), count(DISTINCT content_hash), count(DISTINCT source_key) FROM chunks")
            assert cursor.fetchone() == (3, 1, 3)
    finally:
        assert schema.startswith("receipt_test_") and len(schema) == 45
        with psycopg2.connect(settings.db_url) as connection, connection.cursor() as cursor:
            cursor.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(schema)))


@pytest.mark.skipif(os.getenv("TIMIROOM_TEST_STORAGE_POSTGRES") != "true", reason="Opt-in PostgreSQL integration test")
def test_durable_manifest_verifies_only_committed_chunks_after_restart():
    from config.settings import settings

    schema = "receipt_test_" + uuid.uuid4().hex
    with psycopg2.connect(settings.db_url) as connection, connection.cursor() as cursor:
        cursor.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema)))
    dsn = make_dsn(settings.db_url, options=f"-c search_path={schema}")
    try:
        with psycopg2.connect(dsn) as connection, connection.cursor() as cursor:
            cursor.execute("CREATE TABLE pipeline_storage_receipts (pipeline_id TEXT PRIMARY KEY, source_keys JSONB NOT NULL, published BOOLEAN NOT NULL DEFAULT FALSE)")
            cursor.execute("CREATE TABLE receipt_chunks (source_key TEXT PRIMARY KEY)")

        async def scenario():
            ingestion = DocumentIngestionService(dsn, "receipt_chunks", object())
            store = StorageReceiptStore(dsn, ingestion)
            documents = [("first", {"pipeline_id": "p", "type": "prd"}), ("second", {"pipeline_id": "p", "type": "api"})]
            await store.prepare("p", documents)
            await store.prepare("p", documents)
            await store.mark_published("p")
            keys = ingestion.event_source_keys(documents)
            with pytest.raises(ValueError, match="MANIFEST_CONFLICT"):
                await store.prepare("p", [("different", documents[0][1])])
            assert (await store.status("p"))["storedChunks"] == 0
            with psycopg2.connect(dsn) as connection, connection.cursor() as cursor:
                cursor.execute("INSERT INTO receipt_chunks VALUES (%s)", (keys[0],))
                # A consumer transaction that has not committed is not an ACK.
                assert (await store.status("p"))["status"] == "pending"
            assert (await store.status("p"))["storedChunks"] == 1
            with psycopg2.connect(dsn) as connection, connection.cursor() as cursor:
                cursor.execute("INSERT INTO receipt_chunks VALUES (%s)", (keys[1],))
            recovered = StorageReceiptStore(dsn, DocumentIngestionService(dsn, "receipt_chunks", object()))
            assert await recovered.status("p") == {"status": "stored", "kafkaPublished": True, "expectedChunks": 2, "storedChunks": 2}

        asyncio.run(scenario())
    finally:
        assert schema.startswith("receipt_test_") and len(schema) == 45
        with psycopg2.connect(settings.db_url) as connection, connection.cursor() as cursor:
            cursor.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(schema)))
