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


def test_consumer_writes_exactly_the_keys_the_producer_expects():
    """저장 확인의 핵심 계약 — ingest_event가 쓰는 source_key 집합이
    event_source_keys(= publish 시점의 기대 명세)와 정확히 같아야 한다.

    둘이 어긋나면 consumer가 정상 저장했더라도 confirm_storage가 영원히
    pending에 머물고, 파이프라인은 '저장 확인 대기'로 끝나 결과가 안 나온다.
    """
    from phase1.document_ingestion import KAFKA_CHUNK_SIZE
    from phase2.state import PipelineState
    from phase4.documents import event_documents
    from phase4.kafka_producer import KafkaProducerService

    # 여러 청크로 쪼개지는 길이여야 chunk_index 정렬까지 검증된다.
    long_prd = "가" * (KAFKA_CHUNK_SIZE * 2 + 17)
    state = PipelineState(
        project_name="타이미룸",
        feature_list=["목록 조회", "공유"],
        prd_document=long_prd,
        db_schema='{"tables":[{"name":"items"}]}',
        api_spec='{"endpoints":[{"method":"GET","path":"/api/v1/items"}]}',
        openapi_spec='{"openapi":"3.1.0"}',
        feature_spec_document='{"features":[]}',
        feature_registry=[{"featureId": "items.read", "name": "목록 조회"}],
        market_research="조사 결과",
    )
    documents = event_documents(KafkaProducerService._event(state, "run-1"))
    assert len(documents) > 4, "산출물 종류가 충분히 포함돼야 의미 있는 검증이 된다"

    expected = DocumentIngestionService.event_source_keys(documents)
    written: list[str] = []

    service = DocumentIngestionService("db", "chunks", MagicMock())
    service._embedder.embed = AsyncMock(side_effect=lambda texts: [[0.0]] * len(texts))
    service._missing_records = lambda records: records
    service._store_records_to_db = lambda records, _embeddings: written.extend(
        DocumentIngestionService._source_key(text, metadata) for text, metadata in records
    ) or len(records)

    asyncio.run(service.ingest_event(documents))

    assert sorted(written) == expected
    assert len(written) == len(set(written)), "source_key 중복 — ON CONFLICT로 청크가 소실된다"


def test_partially_stored_event_never_reports_stored():
    """일부 청크만 커밋된 상태를 stored로 보고하면 저장이 안 된 결과가 완료로 나간다."""
    service = DocumentIngestionService("db", "chunks", MagicMock())
    keys = ["a", "b", "c"]
    service._stored_source_keys = lambda requested: {"a", "b"}

    partial = asyncio.run(service.storage_status_for_keys(keys))
    assert partial == {"status": "pending", "expectedChunks": 3, "storedChunks": 2}

    service._stored_source_keys = lambda requested: {"a", "b", "c", "unrelated"}
    assert asyncio.run(service.storage_status_for_keys(keys))["status"] == "stored"

    # 기대 명세가 비어 있으면 '저장할 게 없음'을 완료로 착각해선 안 된다.
    assert asyncio.run(service.storage_status_for_keys([]))["status"] == "pending"
