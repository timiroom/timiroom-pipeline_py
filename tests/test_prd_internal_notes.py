from phase2.agents.prd_agent import _normalize_prd_keys, strip_internal_notes


def test_strip_registry_note_from_description():
    text = "참여 기회를 놓치지 않도록 돕는다. (API: Registry apiContract 없음, DB: Registry tables 없음, foreignKeys 없음)"
    assert strip_internal_notes(text) == "참여 기회를 놓치지 않도록 돕는다."


def test_keep_normal_parentheses():
    text = "사진(포스터)을 올리면 일정(날짜, 장소)을 추출한다."
    assert strip_internal_notes(text) == text


def test_normalize_cleans_core_features():
    doc = {"coreFeatures": [{"name": "a", "description": "설명 (featureId: f1)", "requirements": ["조건 [Registry 매핑]"]}]}
    out = _normalize_prd_keys(doc)
    assert out["coreFeatures"][0]["description"] == "설명"
    assert out["coreFeatures"][0]["requirements"] == ["조건"]
