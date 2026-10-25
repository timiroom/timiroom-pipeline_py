"""Durable expected-chunk manifest for late consumer completion."""

import asyncio
import json

import psycopg2


class StorageReceiptStore:
    def __init__(self, db_url, ingestion):
        self._db_url, self._ingestion = db_url, ingestion

    async def prepare(self, pipeline_id, documents):
        keys = self._ingestion.event_source_keys(documents)
        await asyncio.to_thread(self._prepare, pipeline_id, keys)

    def _prepare(self, pipeline_id, keys):
        with psycopg2.connect(self._db_url, connect_timeout=5) as conn, conn.cursor() as cur:
            cur.execute("""INSERT INTO pipeline_storage_receipts(pipeline_id, source_keys)
                VALUES (%s,%s::jsonb) ON CONFLICT(pipeline_id) DO UPDATE
                SET source_keys=EXCLUDED.source_keys WHERE pipeline_storage_receipts.source_keys=EXCLUDED.source_keys
                RETURNING pipeline_id""", (pipeline_id, json.dumps(keys)))
            if not cur.fetchone():
                raise ValueError("STORAGE_MANIFEST_CONFLICT")

    async def mark_published(self, pipeline_id):
        await asyncio.to_thread(self._mark_published, pipeline_id)

    def _mark_published(self, pipeline_id):
        with psycopg2.connect(self._db_url, connect_timeout=5) as conn, conn.cursor() as cur:
            cur.execute("UPDATE pipeline_storage_receipts SET published=true WHERE pipeline_id=%s", (pipeline_id,))

    async def status(self, pipeline_id):
        value = await asyncio.to_thread(self._load, pipeline_id)
        if value is None:
            return {"status": "unknown", "kafkaPublished": False}
        keys, published = value
        result = await self._ingestion.storage_status_for_keys(keys)
        return {**result, "status": result["status"] if published else "pending", "kafkaPublished": published}

    def _load(self, pipeline_id):
        with psycopg2.connect(self._db_url, connect_timeout=5) as conn, conn.cursor() as cur:
            cur.execute("SELECT source_keys,published FROM pipeline_storage_receipts WHERE pipeline_id=%s", (pipeline_id,))
            return cur.fetchone()
