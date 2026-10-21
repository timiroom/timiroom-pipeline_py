from routers.document import _messages, _related_block, _related_context


def test_related_block_skips_current_and_empty():
    block = _related_block({"api": "GET /posters", "erd": "", "prd": "목표"}, "api")
    assert "[PRD]" in block and "GET /posters" not in block
    assert "[ERD]" not in block


def test_related_block_is_capped():
    block = _related_block({"erd": "가" * 50000}, "api")
    assert len(block) < 17000


def test_messages_include_related_context():
    token = _related_context.set(_related_block({"erd": "TABLE users"}, "api"))
    try:
        msgs = _messages("system", "prompt")
        assert msgs[0]["content"] == "system"
        assert "TABLE users" in msgs[-1]["content"] and msgs[-1]["content"].endswith("prompt")
    finally:
        _related_context.reset(token)
    assert _messages("system", "prompt")[-1]["content"] == "prompt"
