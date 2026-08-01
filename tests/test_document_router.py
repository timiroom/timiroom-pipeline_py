"""routers/document.py — EXAONE 응답을 스텁으로 대체하고 결정론적 부분만 검증한다.

실제 모델을 부르지 않는다. 스텁은 '어떤 프롬프트가 왔는지'로 단계를 구분해
미리 정해둔 JSON을 돌려주고, 호출 기록을 남겨 호출 횟수·동시성 상한을 확인한다.
"""
import asyncio
import json
from types import SimpleNamespace

import pytest

from routers import document as doc

# ── EXAONE 스텁 ──────────────────────────────────────────────────────

class StubClient:
    """chat.completions.create만 흉내 내는 최소 클라이언트."""

    def __init__(self, responder):
        self.calls: list[dict] = []
        self._responder = responder
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    async def _create(self, **kwargs):
        self.calls.append(kwargs)
        content, finish = self._responder(kwargs)
        await asyncio.sleep(0)  # 실제 호출처럼 한 번 양보
        return SimpleNamespace(
            choices=[SimpleNamespace(
                message=SimpleNamespace(content=content),
                finish_reason=finish,
            )]
        )

    def prompts(self) -> list[str]:
        return [c["messages"][-1]["content"] for c in self.calls]


def _stage(kwargs: str) -> str:
    """마지막 user 프롬프트로 어느 단계인지 판별."""
    prompt = kwargs["messages"][-1]["content"]
    if "## scope" in prompt or "현재 항목 목록" in prompt:
        return "scope"
    if "수정할 항목" in prompt:
        return "item"
    if "항목을 추가할 목록" in prompt:
        return "add"
    if "확정된 변경 내용" in prompt:
        return "proposal"
    if "수정할 섹션" in prompt:
        return "section"
    return "classify"


@pytest.fixture(autouse=True)
def _no_main_import(monkeypatch):
    """_exaone_endpoint_id가 main을 import하지 않게 막는다."""
    monkeypatch.setattr(doc, "_exaone_endpoint_id", lambda: "stub-endpoint")


API_SEC = doc._API_PROFILE.sections["endpoints"]


def _endpoint(method: str, path: str) -> dict:
    return {
        "method": method,
        "path": path,
        "description": f"{path} 를 {method} 한다",
        "authRequired": True,
    }


def _endpoints(n: int) -> list[dict]:
    return [_endpoint("GET", f"/api/v1/res{i}") for i in range(n)]


# ── 3번: 항목 번호는 1-based로 보여주고 파싱할 때 되돌린다 ────────────

def test_item_labels_are_one_based():
    items = _endpoints(3)
    lines = doc._item_labels(API_SEC, items).split("\n")
    assert lines[0].startswith("1. ")
    assert lines[2].startswith("3. ")


def test_selected_index_is_converted_to_zero_based():
    # 사용자가 "3번"이라고 말하고 모델이 3을 고르면 → 내부 인덱스 2 (세 번째 항목)
    client = StubClient(lambda k: ('{"scope":"item","indices":[3],"operation":"edit"}', "stop"))
    indices, operation = asyncio.run(
        doc._select_item_indices(client, API_SEC, "endpoints", _endpoints(5), "3번 설명 고쳐줘")
    )
    assert indices == [2]
    assert operation == "edit"


def test_out_of_range_one_based_index_is_rejected():
    # 5개 목록에서 6번은 없는 번호 → 유효 인덱스 없음 → 전체 재작성으로 폴백
    client = StubClient(lambda k: ('{"scope":"item","indices":[6],"operation":"edit"}', "stop"))
    indices, _ = asyncio.run(
        doc._select_item_indices(client, API_SEC, "endpoints", _endpoints(5), "6번 고쳐줘")
    )
    assert indices is None


# ── 1번: 항목 단위 재작성 동시 호출 상한 ─────────────────────────────

def test_selected_indices_are_capped():
    picked = list(range(1, 21))  # 1-based로 20개 전부 지목
    client = StubClient(lambda k: (json.dumps({"scope": "item", "indices": picked, "operation": "edit"}), "stop"))
    indices, _ = asyncio.run(
        doc._select_item_indices(client, API_SEC, "endpoints", _endpoints(20), "전부 손봐줘")
    )
    assert len(indices) == doc._MAX_TARGET_ITEMS


def test_item_rewrite_fanout_respects_cap():
    """모델이 항목 20개를 지목해도 항목 재작성 호출은 상한만큼만 나간다."""
    seq = {"n": 0}

    def responder(kwargs):
        stage = _stage(kwargs)
        if stage == "scope":
            return json.dumps({"scope": "item", "indices": list(range(1, 21)), "operation": "edit"}), "stop"
        if stage == "item":
            # 항목마다 서로 다른 교체본 — 같은 값을 주면 중복 정리가 끼어들어 개수가 달라진다
            seq["n"] += 1
            return json.dumps({"value": _endpoint("POST", f"/api/v1/changed{seq['n']}")}), "stop"
        return '{}', "stop"

    client = StubClient(responder)
    result = asyncio.run(
        doc._rewrite_section(client, doc._API_PROFILE, "endpoints", _endpoints(20), "설명 다 고쳐줘")
    )
    item_calls = [c for c in client.calls if _stage(c) == "item"]
    assert len(item_calls) == doc._MAX_TARGET_ITEMS
    # 손대지 않은 항목은 원본 객체 그대로
    assert len(result) == 20
    assert result[doc._MAX_TARGET_ITEMS:] == _endpoints(20)[doc._MAX_TARGET_ITEMS:]


# ── 2번: add / remove 는 목록 전체를 다시 쓰지 않는다 ─────────────────

def test_remove_deletes_in_python_without_a_rewrite_call():
    baseline = _endpoints(20)
    client = StubClient(lambda k: ('{"scope":"item","indices":[2],"operation":"remove"}', "stop"))

    result = asyncio.run(
        doc._rewrite_section(client, doc._API_PROFILE, "endpoints", baseline, "두 번째 엔드포인트 빼줘")
    )

    assert len(client.calls) == 1  # 범위 판정 한 번뿐 — 재작성 호출 없음
    assert len(result) == 19
    assert baseline[1] not in result
    assert result == baseline[:1] + baseline[2:]  # 나머지는 원본 객체 그대로


def test_add_on_a_large_list_appends_only_new_items():
    """전체 재작성이었다면 잘려서 실패했을 크기에서도 추가가 성공해야 한다."""
    baseline = _endpoints(20)
    assert len(json.dumps(baseline, ensure_ascii=False)) > 0

    def responder(kwargs):
        stage = _stage(kwargs)
        if stage == "scope":
            return '{"scope":"list","indices":[],"operation":"add"}', "stop"
        if stage == "add":
            return json.dumps({"value": [
                _endpoint("POST", "/api/v1/new-a"),
                _endpoint("POST", "/api/v1/new-b"),
            ]}), "stop"
        raise AssertionError(f"예상치 못한 단계 호출: {stage}")

    client = StubClient(responder)
    result = asyncio.run(
        doc._rewrite_section(client, doc._API_PROFILE, "endpoints", baseline, "엔드포인트 2개 더 추가해줘")
    )

    assert result[:20] == baseline  # 기존 항목은 손대지 않음
    assert [e["path"] for e in result[20:]] == ["/api/v1/new-a", "/api/v1/new-b"]
    # 추가 프롬프트에는 원본 20개의 JSON이 아니라 라벨 목록 + 예시 항목 하나만 실린다
    add_prompt = next(c["messages"][-1]["content"] for c in client.calls if _stage(c) == "add")
    assert add_prompt.count('"authRequired"') == 1
    assert len(add_prompt) < len(json.dumps(baseline, ensure_ascii=False, indent=2))


def test_add_drops_items_that_duplicate_existing_ones():
    baseline = _endpoints(3)

    def responder(kwargs):
        stage = _stage(kwargs)
        if stage == "scope":
            return '{"scope":"list","indices":[],"operation":"add"}', "stop"
        if stage == "add":
            return json.dumps({"value": [
                _endpoint("GET", "/api/v1/res0"),   # 이미 있는 것
                _endpoint("POST", "/api/v1/fresh"),
            ]}), "stop"
        raise AssertionError(stage)

    client = StubClient(responder)
    result = asyncio.run(
        doc._rewrite_section(client, doc._API_PROFILE, "endpoints", baseline, "추가해줘")
    )
    assert len(result) == 4
    assert [e["path"] for e in result[3:]] == ["/api/v1/fresh"]


# ── 5번 + 2번: 전체 재작성 크기 가드 ─────────────────────────────────

def test_oversized_section_skips_the_doomed_rewrite():
    """출력 한도에 못 들어갈 크기면 호출을 던지지 않고 접는다."""
    big = [
        {"metric": f"지표{i}", "target": "1 → 2", "basis": "기관, 2025년", "note": "가" * 400}
        for i in range(40)
    ]
    assert len(json.dumps(big, ensure_ascii=False)) > doc._MAX_REWRITE_CHARS

    # 범위 판정은 '전체 재작성'으로 답하게 해서 폴백 경로로 밀어넣는다
    client = StubClient(lambda k: ('{"scope":"list","indices":[],"operation":"edit"}', "stop"))
    result = asyncio.run(
        doc._rewrite_section(client, doc._PRD_PROFILE, "kpi", big, "전부 다시 써줘")
    )

    assert result is None
    assert all(_stage(c) == "scope" for c in client.calls)  # 재작성 호출은 없었다


def test_document_size_guard_returns_413():
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    app = FastAPI()
    app.include_router(doc.router)
    client = TestClient(app, raise_server_exceptions=False)

    huge = {"endpoints": [{"method": "GET", "path": "/x", "description": "가" * 1000} for _ in range(100)]}
    assert len(json.dumps(huge, ensure_ascii=False)) > doc._MAX_DOCUMENT_CHARS

    resp = client.post("/api/v1/document/api/edit", json={"document": huge, "instruction": "고쳐줘"})
    assert resp.status_code == 413
    assert "너무 큽니다" in resp.json()["detail"]


def test_unknown_doc_type_still_404():
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    app = FastAPI()
    app.include_router(doc.router)
    client = TestClient(app, raise_server_exceptions=False)
    resp = client.post("/api/v1/document/nope/edit", json={"document": {}, "instruction": "고쳐줘"})
    assert resp.status_code == 404


# ── 4번: 직전 대화가 모든 단계에 실린다 ──────────────────────────────

def test_history_reaches_every_stage():
    history = [
        {"role": "user", "content": "첫 요청입니다"},
        {"role": "assistant", "content": "이렇게 바꿀까요?"},
    ]

    def responder(kwargs):
        stage = _stage(kwargs)
        if stage == "scope":
            return '{"scope":"item","indices":[1],"operation":"edit"}', "stop"
        if stage == "item":
            return json.dumps({"value": _endpoint("POST", "/api/v1/res0")}), "stop"
        return '{}', "stop"

    client = StubClient(responder)
    asyncio.run(
        doc._rewrite_section(client, doc._API_PROFILE, "endpoints", _endpoints(3), "그거 말고 더 짧게", history)
    )

    assert client.calls, "호출이 없다"
    for call in client.calls:
        roles = [m["role"] for m in call["messages"]]
        assert roles[0] == "system"
        assert roles[1:3] == ["user", "assistant"], f"{_stage(call)} 단계에 history가 없다"
        assert call["messages"][1]["content"] == "첫 요청입니다"


def test_history_turns_are_trimmed():
    long_turn = [{"role": "user", "content": "가" * 5000}]
    messages = doc._messages("sys", "prompt", long_turn)
    assert len(messages[1]["content"]) == doc._HISTORY_TURN_CHARS + 1  # 잘린 표시 '…' 포함


def test_history_turn_count_is_capped():
    turns = [{"role": "user", "content": f"턴{i}"} for i in range(20)]
    messages = doc._messages("sys", "prompt", turns)
    assert len(messages) == doc._MAX_HISTORY_TURNS + 2  # system + history + prompt


# ── 기존 동작이 깨지지 않았는지 ──────────────────────────────────────

def test_edit_path_keeps_untouched_items_identical():
    baseline = _endpoints(5)

    def responder(kwargs):
        stage = _stage(kwargs)
        if stage == "scope":
            return '{"scope":"item","indices":[2],"operation":"edit"}', "stop"
        if stage == "item":
            return json.dumps({"value": _endpoint("POST", "/api/v1/res1")}), "stop"
        return '{}', "stop"

    client = StubClient(responder)
    result = asyncio.run(
        doc._rewrite_section(client, doc._API_PROFILE, "endpoints", baseline, "2번 POST로 바꿔줘")
    )
    assert result[1]["method"] == "POST"
    assert result[0] is baseline[0] and result[2] is baseline[2]


def test_shrunk_list_without_remove_request_is_regenerated():
    baseline = _endpoints(5)
    attempts = {"n": 0}

    def responder(kwargs):
        stage = _stage(kwargs)
        if stage == "scope":
            return '{"scope":"list","indices":[],"operation":"edit"}', "stop"
        if stage == "section":
            attempts["n"] += 1
            return json.dumps({"value": _endpoints(3)}), "stop"  # 요청 없이 줄어든 목록
        return '{}', "stop"

    client = StubClient(responder)
    result = asyncio.run(
        doc._rewrite_section(client, doc._API_PROFILE, "endpoints", baseline, "설명 다듬어줘")
    )
    assert result is None
    assert attempts["n"] == 3  # 3번 다 거부


# ── 손실 있는 '부분 복구' 응답을 편집에 쓰지 않는다 ──────────────────
# 실측: users 테이블 재작성 응답이 잘려 나왔는데 try_parse_json이 그걸 복구해
# constraints 값이 "DEFAULT_FALSE'}, {: : NOT NULL" 이 되고 컬럼 다수가 소실됐다.
# 구조는 멀쩡해서 타입·필수키 검증을 전부 통과했다.

TRUNCATED = '{"value": {"method": "GET", "path": "/api/v1/res0", "description": "설명", "authRequired": true'


def test_strict_parse_rejects_lossy_repair_but_keeps_safe_fixes():
    from phase2.json_utils import try_parse_json

    assert try_parse_json(TRUNCATED) is not None       # 기존(관대) 동작 — 파이프라인용
    assert try_parse_json(TRUNCATED, strict=True) is None

    # 내용을 잃지 않는 문법 교정은 strict에서도 살아 있어야 한다
    assert try_parse_json('{"a": [1, 2,], }', strict=True) == {"a": [1, 2]}
    assert try_parse_json('{"a": "열린 문자열\n"b": "값"}', strict=True) == {"a": "열린 문자열", "b": "값"}


def test_item_rewrite_rejects_truncated_response():
    client = StubClient(lambda k: (TRUNCATED, "stop"))
    result = asyncio.run(
        doc._rewrite_item(client, API_SEC, "endpoints", 0, _endpoint("GET", "/api/v1/res0"), "고쳐줘")
    )
    assert result is None
    assert len(client.calls) == 3  # 3번 재시도 후 포기


def test_item_rewrite_rejects_json_fragments_in_values():
    poisoned = _endpoint("GET", "/api/v1/res0")
    poisoned["description"] = "DEFAULT_FALSE'}, {: : NOT NULL"
    client = StubClient(lambda k: (json.dumps({"value": poisoned}), "stop"))
    result = asyncio.run(
        doc._rewrite_item(client, API_SEC, "endpoints", 0, _endpoint("GET", "/api/v1/res0"), "고쳐줘")
    )
    assert result is None


def test_item_rewrite_restores_dropped_fields():
    """실측: 필터 추가 요청에 requestBody·successResponse·errorCodes가 통째로 사라졌다.

    재생성해도 매번 다른 필드를 흘리므로, 거부 대신 원본 값으로 되살린다.
    """
    original = _endpoint("GET", "/api/v1/res0")
    original["parameters"] = [{"in": "query", "name": "page", "type": "integer"}]
    original["errorCodes"] = "401 — 인증 실패"

    partial = {"method": "GET", "path": "/api/v1/res0", "description": "새 설명", "authRequired": True,
               "parameters": [{"in": "query", "name": "page"}]}  # errorCodes + 중첩 type 소실
    client = StubClient(lambda k: (json.dumps({"value": partial}), "stop"))

    result = asyncio.run(doc._rewrite_item(client, API_SEC, "endpoints", 0, original, "고쳐줘"))
    assert result["description"] == "새 설명"          # 모델이 바꾼 값은 유지
    assert result["errorCodes"] == "401 — 인증 실패"   # 흘린 필드는 원본에서 복구
    assert result["parameters"][0]["type"] == "integer"
    assert len(client.calls) == 1                      # 재생성 없이 한 번에 끝난다


def test_restore_brings_back_dropped_list_entries_by_name():
    """실측: '컬럼 추가' 요청에 users 테이블이 컬럼 19개 → 3개로 잘려 나왔다.

    자리(index)가 아니라 이름으로 짝지어 되살리므로 순서가 바뀌어도, 모델이 새로
    넣은 항목이 있어도 안전해야 한다.
    """
    before = {
        "name": "users",
        "description": "이용자",
        "columns": [{"name": f"c{i}", "type": "T", "constraints": "X"} for i in range(19)],
        "indexes": ["idx_a"],
    }
    after = {
        "name": "users",
        "columns": [
            {"name": "c1", "type": "T", "constraints": "X"},          # 순서 바뀜
            {"name": "c0", "type": "T2", "constraints": "X"},         # 모델이 고친 값
            {"name": "phone", "type": "VARCHAR(20)", "constraints": "NULL"},  # 새로 추가
        ],
    }
    merged, restored = doc._reconcile_with_original(before, after)

    names = [c["name"] for c in merged["columns"]]
    assert len(names) == 20 and len(set(names)) == 20      # 19개 + phone, 중복 없음
    assert "phone" in names                                 # 모델이 추가한 건 유지
    assert merged["columns"][1]["type"] == "T2"             # 모델이 고친 값도 유지
    assert merged["indexes"] == ["idx_a"]                   # 사라진 키 복구
    assert merged["description"] == "이용자"
    assert restored                                          # 무엇을 되살렸는지 로그용


def test_restore_fills_subfields_of_matched_entries():
    before = {"name": "t", "columns": [{"name": "a", "type": "T", "constraints": "C"}]}
    after = {"name": "t", "columns": [{"name": "a", "type": "T2"}]}   # constraints 소실
    merged, _ = doc._reconcile_with_original(before, after)
    assert merged["columns"][0] == {"name": "a", "type": "T2", "constraints": "C"}


def test_legit_api_values_are_not_treated_as_corruption():
    """successResponse에 흔한 배열 표기가 오탐되면 영영 재생성만 돈다."""
    from phase2.json_utils import find_json_syntax_leak

    assert find_json_syntax_leak("favorites: [공간 객체 배열], total: number") is None
    assert find_json_syntax_leak("reviews: [리뷰 객체], count: integer") is None
    assert find_json_syntax_leak("/api/v1/spaces/{spaceId}/favorites") is None
    assert find_json_syntax_leak("DEFAULT_FALSE'}, {: : NOT NULL") is not None


def test_edit_that_collides_with_another_item_is_reverted_not_deleted():
    """실측: 필터 추가 요청에 모델이 엉뚱하게 POST /spaces 를 GET /spaces 로 바꿔
    기존 항목과 같아졌다. 중복을 지우면 공간 등록 API가 문서에서 사라진다."""
    baseline = [_endpoint("GET", "/api/v1/spaces"), _endpoint("GET", "/api/v1/other"),
                _endpoint("POST", "/api/v1/spaces")]

    def responder(kwargs):
        stage = _stage(kwargs)
        if stage == "scope":
            return '{"scope":"item","indices":[2,3],"operation":"edit"}', "stop"
        if stage == "item":
            prompt = kwargs["messages"][-1]["content"]
            if "3번" in prompt:   # POST /spaces 를 GET /spaces 로 바꿔버리는 응답
                return json.dumps({"value": _endpoint("GET", "/api/v1/spaces")}), "stop"
            return json.dumps({"value": dict(baseline[1], description="정상 수정")}), "stop"
        return '{}', "stop"

    client = StubClient(responder)
    result = asyncio.run(
        doc._rewrite_section(client, doc._API_PROFILE, "endpoints", baseline, "필터 넣어주세요")
    )
    assert len(result) == 3                       # 아무것도 사라지지 않는다
    assert result[2] == baseline[2]               # 충돌한 수정만 되돌아간다
    assert result[1]["description"] == "정상 수정"  # 멀쩡한 수정은 유지


def test_edit_keeping_its_own_label_is_not_reverted():
    baseline = [_endpoint("GET", "/api/v1/a"), _endpoint("GET", "/api/v1/b")]
    merged = [dict(baseline[0], description="새 설명"), baseline[1]]
    kept, changed = doc._revert_colliding_edits(API_SEC, "endpoints", baseline, merged, [0])
    assert changed == [0]
    assert kept[0]["description"] == "새 설명"


def test_edit_to_a_brand_new_label_is_not_reverted():
    """자리표시자를 의미 있는 경로로 바꾸는 건 정상 수정이다."""
    baseline = [_endpoint("GET", "/api/v1/resource-19"), _endpoint("GET", "/api/v1/b")]
    merged = [_endpoint("GET", "/api/v1/spaces/available-times"), baseline[1]]
    kept, changed = doc._revert_colliding_edits(API_SEC, "endpoints", baseline, merged, [0])
    assert changed == [0]
    assert kept[0]["path"] == "/api/v1/spaces/available-times"


def test_oversized_list_re_asks_for_an_item_instead_of_giving_up():
    """실측: 12,745자짜리 엔드포인트 목록에서 scope=list가 나오면 크기 한도에 막혀
    아무 제안도 못 만들었다. 이때는 항목을 고르라고 못 박아 한 번 더 묻는다."""
    baseline = [
        {"method": "GET", "path": f"/api/v1/res{i}", "description": "설명 " + "가" * 500}
        for i in range(30)
    ]
    assert len(json.dumps(baseline, ensure_ascii=False, indent=2)) > doc._MAX_REWRITE_CHARS

    scope_calls = {"n": 0}

    def responder(kwargs):
        stage = _stage(kwargs)
        if stage == "scope":
            scope_calls["n"] += 1
            if scope_calls["n"] == 1:
                return '{"scope":"list","indices":[],"operation":"edit"}', "stop"
            assert "전체를 다시 쓸 수 없습니다" in kwargs["messages"][-1]["content"]
            return '{"scope":"item","indices":[1],"operation":"edit"}', "stop"
        if stage == "item":
            return json.dumps({"value": dict(baseline[0], description="새 설명")}), "stop"
        return '{}', "stop"

    client = StubClient(responder)
    result = asyncio.run(
        doc._rewrite_section(client, doc._API_PROFILE, "endpoints", baseline, "검색 API에 필터 넣어주세요")
    )
    assert scope_calls["n"] == 2
    assert result[0]["description"] == "새 설명"
    assert result[1:] == baseline[1:]


def test_add_request_on_oversized_list_is_not_re_asked():
    """add는 목록 전체를 다시 쓰지 않으므로 scope=list가 정상 — 재질의하면 낭비다."""
    baseline = [{"method": "GET", "path": f"/api/v1/res{i}", "description": "가" * 500} for i in range(30)]
    scope_calls = {"n": 0}

    def responder(kwargs):
        stage = _stage(kwargs)
        if stage == "scope":
            scope_calls["n"] += 1
            return '{"scope":"list","indices":[],"operation":"add"}', "stop"
        if stage == "add":
            return json.dumps({"value": [_endpoint("POST", "/api/v1/new")]}), "stop"
        return '{}', "stop"

    client = StubClient(responder)
    result = asyncio.run(
        doc._rewrite_section(client, doc._API_PROFILE, "endpoints", baseline, "하나 추가해줘")
    )
    assert scope_calls["n"] == 1
    assert len(result) == 31


def test_garbage_keys_invented_by_the_model_are_dropped():
    """실측: 엔드포인트에 `}{.;/ / / /` 라는 키가 값 `'[]: ,,, ,,,'` 과 함께 붙어 나왔다.
    값만 보는 오염 탐지기는 이런 '키'를 못 본다."""
    before = _endpoint("GET", "/api/v1/res0")
    after = dict(before, description="새 설명")
    after["}{.;/ / / / / / /"] = "[]: ,,, ,,,"
    merged, fixed = doc._reconcile_with_original(before, after)
    assert "}{.;/ / / / / / /" not in merged
    assert merged["description"] == "새 설명"
    assert any(f.startswith("-") for f in fixed)


def test_a_field_other_items_already_use_may_be_added():
    """원본 users에 indexes가 없어도 다른 테이블이 쓰고 있으면 정당한 필드다 —
    "인덱스 추가해줘"가 성립해야 한다."""
    before = {"name": "users", "columns": [{"name": "id", "type": "BIGINT"}]}
    after = {"name": "users", "columns": [{"name": "id", "type": "BIGINT"}, {"name": "phone"}],
             "indexes": ["new"]}

    merged, restored = doc._reconcile_with_original(
        before, after, known_keys={"name", "description", "columns", "indexes"},
    )
    assert merged == after
    assert restored == []

    # 섹션 어디에도 없는 키였다면 떨어진다
    merged2, fixed2 = doc._reconcile_with_original(before, after, known_keys={"name", "columns"})
    assert "indexes" not in merged2
    assert fixed2 == ["-indexes"]


def test_legitimate_values_are_not_flagged_as_corrupted():
    from phase2.json_utils import has_json_syntax_leak

    assert not has_json_syntax_leak({"description": "이용자 정보 (1:N 관계)"})
    assert not has_json_syntax_leak({"errorCodes": "401 — 인증 실패, 404 — 리소스 없음"})
    # 실측 오탐: 파라미터 설명에 예시 값 목록을 적거나 JSON 예시를 쓰는 건 정상이다
    assert not has_json_syntax_leak({"description": "['projector', 'screen'] 등 장비 보유 여부"})
    assert not has_json_syntax_leak({"successResponse": "favorites: [공간 객체 배열], total: number"})
    assert not has_json_syntax_leak({"requestBody": '{ "spaceId": "integer" }'})


def test_corrupted_constraint_fragment_is_caught():
    """실측: 재작성 결과에 `constraints: \" '},  \"` 가 남아 통과했다."""
    from phase2.json_utils import has_json_syntax_leak

    assert has_json_syntax_leak({"constraints": " '},  "})
    assert has_json_syntax_leak({"constraints": "DEFAULT_FALSE'}, {: : NOT NULL"})
    # 값이 아니라 키가 깨진 유형
    assert has_json_syntax_leak({'descriptiontion:string, "required"}, {': "v"})
    assert not has_json_syntax_leak({"requestBody": "userId: string — 요청 사용자 ID, spaceId: string — 공간 ID"})
    assert has_json_syntax_leak({"constraints": "DEFAULT_FALSE'}, {: : NOT NULL"})


# ── ERD 섹션 간 정합: 테이블명 변경/삭제가 관계에 반영된다 ────────────

def _table(name: str) -> dict:
    return {"name": name, "description": f"{name} 테이블", "columns": [{"name": "id", "type": "BIGINT"}]}


def test_renamed_table_updates_relationship_strings():
    before = [_table("users"), _table("spaces"), _table("reservations")]
    after = [_table("users"), _table("venues"), _table("reservations")]   # spaces → venues
    renames, gone = doc._table_renames(before, after)
    assert renames == {"spaces": "venues"}
    assert gone == set()

    rels = ["users (1:N) spaces", "spaces (1:N) reservations", "users (1:N) reservations"]
    assert doc._sync_relationships(rels, renames, gone) == [
        "users (1:N) venues", "venues (1:N) reservations", "users (1:N) reservations",
    ]


def test_deleted_table_drops_relationships_referencing_it():
    """실측: settlement_records를 지웠는데 그걸 가리키는 관계 2개가 남았다."""
    before = [_table("users"), _table("spaces"), _table("settlement_records")]
    after = [_table("users"), _table("spaces")]
    renames, gone = doc._table_renames(before, after)
    assert gone == {"settlement_records"}

    rels = ["users (1:N) spaces", "spaces (1:N) settlement_records",
            "reservations (1:N) settlement_records"]
    assert doc._sync_relationships(rels, renames, gone) == ["users (1:N) spaces"]


def test_rename_detection_skipped_when_list_length_changed():
    """항목이 늘거나 줄면 자리 대응이 깨져 엉뚱한 테이블을 개명으로 오인한다."""
    before = [_table("users"), _table("spaces")]
    after = [_table("users"), _table("spaces"), _table("reviews")]
    renames, gone = doc._table_renames(before, after)
    assert renames == {} and gone == set()


def test_fk_columns_follow_a_table_rename():
    tables = [
        {"name": "spaces", "columns": [{"name": "id", "type": "BIGINT"}],
         "indexes": ["INDEX idx_spaces_name ON spaces(name)"]},
        {"name": "reservations",
         "columns": [{"name": "id", "type": "BIGINT"},
                     {"name": "space_id", "type": "BIGINT",
                      "constraints": "NOT_NULL FOREIGN_KEY spaces(id)"}],
         "indexes": ["INDEX idx_reservations_space_id ON reservations(space_id)"]},
    ]
    out, changes = doc._sync_fk_references(tables, {"spaces": "venues"})

    res = out[1]["columns"][1]
    assert res["name"] == "venue_id"
    assert res["constraints"] == "NOT_NULL FOREIGN_KEY venues(id)"
    assert out[1]["indexes"] == ["INDEX idx_reservations_venue_id ON reservations(venue_id)"]
    assert out[0]["indexes"] == ["INDEX idx_venues_name ON venues(name)"]
    assert changes


def test_fk_sync_does_not_touch_a_different_table_with_a_similar_name():
    """space_schedules 는 이름이 겹칠 뿐 전혀 다른 테이블이다 —
    단순 문자열 치환이면 venue_schedules 로 망가진다."""
    tables = [
        {"name": "spaces", "columns": [{"name": "id"}]},
        {"name": "space_schedules",
         "columns": [{"name": "space_id", "constraints": "NOT_NULL FOREIGN_KEY spaces(id)"}],
         "indexes": ["INDEX idx_space_schedules_space_id ON space_schedules(space_id)"]},
        {"name": "users", "columns": [{"name": "is_space_owner", "type": "BOOLEAN"}]},
    ]
    out, _ = doc._sync_fk_references(tables, {"spaces": "venues"})

    assert out[1]["name"] == "space_schedules"                    # 테이블명 그대로
    assert out[1]["columns"][0]["name"] == "venue_id"             # FK만 갱신
    assert out[1]["indexes"] == [
        "INDEX idx_space_schedules_venue_id ON space_schedules(venue_id)"
    ]
    assert out[2]["columns"][0]["name"] == "is_space_owner"       # FK가 아니므로 무시


def test_fk_sync_does_not_mutate_the_original_tables():
    """항목 단위 경로는 손대지 않은 테이블의 원본 객체를 그대로 재사용한다 —
    제자리 수정하면 before까지 바뀌어 diff가 빈다."""
    original = [{"name": "reservations", "columns": [{"name": "space_id"}]}]
    snapshot = json.loads(json.dumps(original))
    doc._sync_fk_references(original, {"spaces": "venues"})
    assert original == snapshot


def test_relationship_sync_preserves_unparseable_and_other_kinds():
    rels = ["users (1:1) profiles", "spaces (N:M) tags", "형식이 이상한 줄"]
    synced = doc._sync_relationships(rels, {"spaces": "venues"}, set())
    assert synced == ["users (1:1) profiles", "venues (N:M) tags", "형식이 이상한 줄"]


def test_endpoint_appends_a_relationship_edit_when_a_table_is_renamed():
    document = {
        "tables": [_table("users"), _table("spaces")],
        "relationships": ["users (1:N) spaces"],
    }
    edits = [{
        "section": "tables", "label": "테이블",
        "before": document["tables"],
        "after": [_table("users"), _table("venues")],
        "diff": [],
    }]
    doc._append_relationship_sync(doc._ERD_PROFILE, document, edits)

    assert [e["section"] for e in edits] == ["tables", "relationships"]
    rel = edits[1]
    assert rel["before"] == ["users (1:N) spaces"]
    assert rel["after"] == ["users (1:N) venues"]
    assert any(d["type"] != "same" for d in rel["diff"])


def test_relationship_sync_layers_on_top_of_an_existing_relationship_edit():
    document = {"tables": [_table("users"), _table("spaces")],
                "relationships": ["users (1:N) spaces"]}
    edits = [
        {"section": "tables", "label": "테이블", "before": document["tables"],
         "after": [_table("users"), _table("venues")], "diff": []},
        {"section": "relationships", "label": "관계", "before": document["relationships"],
         "after": ["users (1:N) spaces", "spaces (1:1) settings"], "diff": []},
    ]
    doc._append_relationship_sync(doc._ERD_PROFILE, document, edits)

    assert len(edits) == 2                       # 새로 만들지 않고 기존 편집 위에 얹는다
    assert edits[1]["after"] == ["users (1:N) venues", "venues (1:1) settings"]


def test_no_relationship_edit_when_table_names_are_unchanged():
    document = {"tables": [_table("users")], "relationships": ["users (1:N) spaces"]}
    edits = [{"section": "tables", "label": "테이블", "before": document["tables"],
              "after": [dict(_table("users"), description="설명만 수정")], "diff": []}]
    doc._append_relationship_sync(doc._ERD_PROFILE, document, edits)
    assert [e["section"] for e in edits] == ["tables"]


def test_diff_marks_only_changed_lines():
    before = {"included": ["A", "B"], "excluded": ["C"], "rationale": "이유"}
    after = {"included": ["A", "B2"], "excluded": ["C"], "rationale": "이유"}
    diff = doc._build_diff(before, after)
    changed = [d for d in diff if d["type"] != "same"]
    assert {d["text"].strip() for d in changed} == {"- B", "- B2"}
