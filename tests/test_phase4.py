import asyncio
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from phase1.document_ingestion import DocumentIngestionService
from phase2.state import PipelineState
from phase4.event import PipelineResultEvent
from phase4.kafka_consumer import KafkaConsumerService
from phase4.kafka_producer import KafkaProducerService


class FakeOutbox:
    def __init__(self, pending=None):
        self.rows = list(pending or [])
        self.puts = []
        self.deleted = []
        self.errors = []

    async def put(self, pipeline_id, topic, event):
        self.puts.append((pipeline_id, topic, event))

    async def pending(self, limit=100):
        return self.rows[:limit]

    async def delete(self, pipeline_id):
        self.deleted.append(pipeline_id)

    async def mark_error(self, pipeline_id, error):
        self.errors.append((pipeline_id, error))


class FakeProducer:
    def __init__(self, error=None):
        self.error = error
        self.sent = []
        self.stopped = False

    async def send_and_wait(self, topic, key=None, value=None):
        if self.error:
            raise self.error
        self.sent.append((topic, key, value))

    async def stop(self):
        self.stopped = True


def _state() -> PipelineState:
    return PipelineState(
        project_name="테스트 프로젝트",
        feature_list=["로그인"],
        prd_document="{}",
        db_schema="{}",
        api_spec="{}",
    )


def _event() -> dict:
    return {
        "schemaVersion": 1,
        "pipelineId": "pipeline-1",
        "projectName": "테스트 프로젝트",
        "platform": "WEB",
        "techStack": [],
        "userQuery": "로그인 서비스",
        "featureList": ["로그인"],
        "prdDocument": "{}",
        "marketResearch": "시장 조사",
        "dbSchema": "{}",
        "apiSpec": "{}",
        "retryCount": 0,
        "createdAt": "2026-09-14T00:00:00+00:00",
    }


def test_producer_preserves_request_pipeline_id_and_clears_outbox():
    async def scenario():
        outbox = FakeOutbox()
        producer = FakeProducer()
        service = KafkaProducerService("kafka:9092", "results", "db", outbox=outbox)
        service._producer = producer
        service._ready = True

        result = await service.publish(_state(), "pipeline-1")
        return result, outbox, producer

    result, outbox, producer = asyncio.run(scenario())
    assert result == "pipeline-1"
    assert outbox.puts[0][2]["pipelineId"] == "pipeline-1"
    assert producer.sent[0][1] == "pipeline-1"
    assert outbox.deleted == ["pipeline-1"]


def test_producer_failure_keeps_outbox_and_raises():
    async def scenario():
        outbox = FakeOutbox()
        producer = FakeProducer(RuntimeError("broker down"))
        service = KafkaProducerService(
            "kafka:9092", "results", "db", max_publish_retry=1, outbox=outbox
        )
        service._producer = producer
        service._ready = True
        with pytest.raises(RuntimeError, match="outbox에 보관됨"):
            await service.publish(_state(), "pipeline-1")
        return outbox

    outbox = asyncio.run(scenario())
    assert outbox.deleted == []
    assert outbox.errors and outbox.errors[0][0] == "pipeline-1"


def test_consumer_is_configured_for_manual_commit(monkeypatch):
    captured = {}

    def factory(*topics, **kwargs):
        captured.update(kwargs)
        return SimpleNamespace(topics=topics)

    monkeypatch.setattr("phase4.kafka_consumer.AIOKafkaConsumer", factory)
    service = KafkaConsumerService("kafka:9092", "results", "group", object())

    service._create_consumer()

    assert captured["enable_auto_commit"] is False
    assert captured["max_poll_interval_ms"] == 900_000


def test_consumer_commits_only_after_processing():
    order = []
    message = SimpleNamespace(topic="results", partition=0, offset=7, value=_event())

    class Consumer:
        def __aiter__(self):
            async def iterator():
                yield message
            return iterator()

        async def commit(self, offsets):
            order.append(("commit", offsets))

    async def scenario():
        service = KafkaConsumerService("kafka:9092", "results", "group", object())
        service._consumer = Consumer()

        async def process(_event):
            order.append(("process", None))

        service._process_with_retry = process
        await service._consume_loop()

    asyncio.run(scenario())
    assert [item[0] for item in order] == ["process", "commit"]
    committed = next(iter(order[1][1].values()))
    assert committed.offset == 8


def test_dlq_failure_propagates_so_offset_is_not_committed():
    class DlqProducer:
        async def send_and_wait(self, *_args, **_kwargs):
            raise RuntimeError("DLQ down")

    async def scenario():
        service = KafkaConsumerService("kafka:9092", "results", "group", object(), "dlt")
        service._dlq_producer = DlqProducer()
        await service._publish_to_dlq(_event(), "ingestion failed")

    with pytest.raises(RuntimeError, match="DLQ down"):
        asyncio.run(scenario())


def test_event_contract_rejects_missing_and_unknown_fields():
    malformed = _event()
    malformed.pop("pipelineId")
    malformed["unexpected"] = True

    with pytest.raises(ValidationError):
        PipelineResultEvent.model_validate(malformed)


def test_consumer_ingests_whole_event_in_one_batch():
    class Ingestion:
        def __init__(self):
            self.documents = None

        async def ingest_event(self, documents):
            self.documents = documents
            return 5

    async def scenario():
        ingestion = Ingestion()
        service = KafkaConsumerService("kafka:9092", "results", "group", ingestion)
        await service._process(_event())
        return ingestion.documents

    documents = asyncio.run(scenario())
    assert len(documents) == 5
    assert {metadata["type"] for _, metadata in documents} == {
        "prd", "erd", "api", "features", "market_research"
    }


def test_source_key_is_scoped_by_pipeline_type_and_chunk():
    text = "동일한 문장"
    key_a = DocumentIngestionService._source_key(
        text, {"pipeline_id": "a", "type": "prd", "chunk_index": 0}
    )
    key_b = DocumentIngestionService._source_key(
        text, {"pipeline_id": "b", "type": "prd", "chunk_index": 0}
    )
    key_c = DocumentIngestionService._source_key(
        text, {"pipeline_id": "a", "type": "api", "chunk_index": 0}
    )

    assert len({key_a, key_b, key_c}) == 3


def test_document_batch_rolls_back_on_any_insert_failure():
    class Cursor:
        def __init__(self):
            self.calls = 0

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def execute(self, *_args):
            self.calls += 1
            if self.calls == 2:
                raise RuntimeError("insert failed")

    class Connection:
        def __init__(self):
            self.cursor_value = Cursor()
            self.committed = False
            self.rolled_back = False

        def cursor(self):
            return self.cursor_value

        def commit(self):
            self.committed = True

        def rollback(self):
            self.rolled_back = True

        def close(self):
            pass

    connection = Connection()
    service = object.__new__(DocumentIngestionService)
    service._document_table = "document_chunks"
    service._get_conn = lambda: connection
    records = [
        ("첫 문장", {"pipeline_id": "p", "type": "prd", "chunk_index": 0}),
        ("둘째 문장", {"pipeline_id": "p", "type": "prd", "chunk_index": 1}),
    ]

    with pytest.raises(RuntimeError, match="insert failed"):
        service._store_records_to_db(records, [[0.1], [0.2]])

    assert connection.rolled_back is True
    assert connection.committed is False
