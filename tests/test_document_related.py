from routers.document import _messages, _related_block, _related_context


def test_related_block_skips_current_and_empty():
    block = _related_block({"api": "GET /posters", "erd": "", "prd": "목표"}, "api")
    assert "[PRD]" in block and "GET /posters" not in block
    assert "[ERD]" not in block


def test_related_block_is_capped():
    block = _related_block({"erd": "가" * 50000}, "api")
    assert len(block) < 13000


def test_messages_include_related_context():
    token = _related_context.set(_related_block({"erd": "TABLE users"}, "api"))
    try:
        msgs = _messages("system", "prompt")
        assert "TABLE users" in msgs[0]["content"]
    finally:
        _related_context.reset(token)
    assert "TABLE users" not in _messages("system", "prompt")[0]["content"]
