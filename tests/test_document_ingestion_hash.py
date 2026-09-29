from phase1.document_ingestion import DocumentIngestionService


def test_pipeline_output_hash_is_scoped_by_pipeline_and_document_type():
    text = "동일한 결과 문서"
    first = DocumentIngestionService._content_hash(text, {"pipeline_id": "p1", "type": "features"})
    second = DocumentIngestionService._content_hash(text, {"pipeline_id": "p2", "type": "features"})
    other_type = DocumentIngestionService._content_hash(text, {"pipeline_id": "p1", "type": "market_research"})

    assert len({first, second, other_type}) == 3


def test_source_ingestion_keeps_global_content_deduplication():
    text = "원본 자료"
    assert DocumentIngestionService._content_hash(text, {}) == DocumentIngestionService._content_hash(
        text, {"source": "another-file.pdf"}
    )
