from phase2.agents.prd_agent import PrdAgent, _build_plain_section_prompt
from phase2.state import PipelineState
from phase2.agents.dba_agent import _parse_table_plan_text, _parse_table_text
from phase2.agents.api_agent import _parse_endpoint_text
from phase2.agents.qa_agent import _apply_review_patches, _derive_patches_from_full_output


def test_goal_accepts_plain_text_when_model_omits_label():
    raw = "재고 자동화를 통해 사용자 확인 시간을 단축하여 식재료 낭비 감소와 반복 사용 증가에 기여한다."

    item = PrdAgent._parse_item_text("goal", raw, "", 0)

    assert item == {"value": raw}
    assert PrdAgent._valid_item("goal", item)


def test_dba_plan_and_table_are_parsed_from_plain_text():
    plan = _parse_table_plan_text(
        "TABLE: users ||| 사용자 계정\nTABLE: inventory_items ||| 냉장고 재고 자동 등록"
    )
    table = _parse_table_text(
        "DESCRIPTION: 냉장고 재고\nCOLUMN: id BIGINT PRIMARY_KEY AUTO_INCREMENT\n"
        "COLUMN: user_id BIGINT NOT_NULL FOREIGN_KEY\nINDEX: INDEX idx_inventory_user ON inventory_items(user_id)",
        "inventory_items", "재고",
    )

    assert [item["name"] for item in plan] == ["users", "inventory_items"]
    assert table["columns"][1].startswith("user_id BIGINT")


def test_api_endpoint_is_parsed_from_plain_text_and_bound_to_skeleton():
    skeleton = {"method": "POST", "path": "/api/v1/items", "description": "재료 등록", "authRequired": True}
    item = _parse_endpoint_text(
        "METHOD: GET\nPATH: /wrong\nDESCRIPTION: 재료를 등록합니다\nAUTH_REQUIRED: true\n"
        "REQUEST_BODY: name: string\nSUCCESS_RESPONSE: id: integer\nERROR_CODES: 400 - 잘못된 요청",
        skeleton,
    )

    assert item["method"] == "POST"
    assert item["path"] == "/api/v1/items"
    assert item["requestBody"] == "name: string"


def test_api_get_endpoint_fills_optional_blank_fields_without_retry():
    skeleton = {"method": "GET", "path": "/api/v1/items", "description": "item list", "authRequired": True}
    item = _parse_endpoint_text(
        "METHOD: GET\nPATH: /api/v1/items\nDESCRIPTION: item list\n"
        "AUTH_REQUIRED: true\nREQUEST_BODY:\nSUCCESS_RESPONSE:\nERROR_CODES:",
        skeleton,
    )
    assert item["requestBody"] == "없음"
    assert item["successResponse"] == "성공 응답"
    assert item["errorCodes"].startswith("400")


def test_api_endpoint_tolerates_common_label_typos():
    skeleton = {"method": "POST", "path": "/api/v1/items", "description": "create item", "authRequired": True}
    item = _parse_endpoint_text(
        "DESCRIPTION: create item\nREQUEST_BODY_BODY: name string\n"
        "SUCCESS_RESPONSE: created\nERROR_COUDES: 400",
        skeleton,
    )
    assert item["requestBody"] == "name string"
    assert item["errorCodes"] == "400"


def test_api_endpoint_treats_colonless_error_text_as_continuation():
    skeleton = {"method": "GET", "path": "/api/v1/items", "description": "item list", "authRequired": True}
    item = _parse_endpoint_text(
        "DESCRIPTION: item list\nSUCCESS_RESPONSE: items returned\n"
        "ERROR_Codes may be supplied in the response body",
        skeleton,
    )
    assert item["successResponse"].endswith("response body")
    assert item["errorCodes"].startswith("400")


def test_kpi_delimited_text_is_parsed_without_json():
    raw = """METRIC: 7일 리텐션
TARGET: 0% → 35%
BASIS: 모바일인덱스, 2024년 추정치
MEASUREMENT_METHOD: 가입 후 7일차 재방문 사용자 비율
FREQUENCY: 주간"""

    item = PrdAgent._parse_item_text("kpi", raw, "리텐션", 0)

    assert PrdAgent._valid_item("kpi", item)
    assert item["metric"] == "7일 리텐션"
    assert item["measurementMethod"] == "가입 후 7일차 재방문 사용자 비율"


def test_release_item_uses_deterministic_order_and_stage():
    raw = """DATE: 잘못된 기간
MILESTONE: 잘못된 단계
DESCRIPTION: 요구사항과 데이터 모델, API 계약을 확정하고 이해관계자의 승인을 받아 이후 개발 단계의 완료 조건을 명확하게 고정합니다.
DELIVERABLES: 요구사항 명세서 ||| ERD ||| API 설계서"""

    item = PrdAgent._parse_item_text("release", raw, "", 0)

    assert item["date"] == "1개월차"
    assert item["milestone"] == "요구사항 확정 및 설계"
    assert item["deliverables"] == ["요구사항 명세서", "ERD", "API 설계서"]
    assert PrdAgent._valid_item("release", item)


def test_core_feature_name_is_bound_to_requested_feature():
    raw = """NAME: 모델이 바꾼 이름
DESCRIPTION: 사용자가 냉장고 사진을 올리면 시스템이 이미지 분석으로 재료를 식별하고 재고 목록에 반영하여 수동 입력 시간을 줄이고 정확한 재고 확인을 가능하게 합니다.
PRIORITY: MUST
REQUIREMENTS: 사진 업로드 시 이미지 형식을 검증한 뒤 인식 모델을 호출하고 결과를 사용자 확인 후 재고에 저장해야 합니다."""

    item = PrdAgent._parse_item_text("core_feature", raw, "냉장고 사진 자동 인식", 0)

    assert item["name"] == "냉장고 사진 자동 인식"
    assert item["priority"] == "P0"
    assert PrdAgent._valid_item("core_feature", item)


def test_prd_prompts_include_phase1_rag_and_priority_context():
    ctx = {
        "user_query": "예약 서비스를 만든다",
        "feature_str": "- 예약 생성",
        "market_data": "시장 조사",
        "rollback_section": "",
        "rag_context": "대화에서 예약 중복 방지 정책을 확정함",
        "priority_context": "Must: 예약 생성\nCould: 추천",
    }

    plain = _build_plain_section_prompt("mvpScope", ctx)
    item = PrdAgent._build_item_prompt(ctx, "core_feature", 0, "예약 생성")

    for prompt in (plain, item):
        assert "예약 중복 방지 정책" in prompt
        assert "Must: 예약 생성" in prompt


def test_phase1_moscow_priorities_override_generated_mvp_priorities():
    parsed = {
        "coreFeatures": [
            {"name": "예약 생성", "priority": "P2"},
            {"name": "예약 알림", "priority": "P0"},
            {"name": "맞춤 추천", "priority": "P0"},
        ],
        "mvpScope": {
            "included": ["맞춤 추천"],
            "excluded": ["예약 생성", "예약 알림"],
        },
    }
    state = PipelineState(
        must_features=["예약 생성"],
        should_features=["예약 알림"],
        could_features=["맞춤 추천"],
    )

    PrdAgent._apply_phase1_priorities(parsed, state)

    priorities = {item["name"]: item["priority"] for item in parsed["coreFeatures"]}
    assert priorities == {"예약 생성": "P0", "예약 알림": "P1", "맞춤 추천": "P2"}
    assert "예약 생성" in parsed["mvpScope"]["included"]
    assert "예약 생성" not in parsed["mvpScope"]["excluded"]


def test_inline_requirements_marker_is_split_from_description():
    raw = """NAME: ignored
DESCRIPTION: 사용자가 사진을 올리면 시스템이 재료를 인식하고 저장하여 수동 입력을 줄입니다. ||| REQUIREMENTS: 이미지 형식을 검증해야 합니다. ||| 인식 결과를 저장해야 합니다.
PRIORITY: P0"""

    item = PrdAgent._parse_item_text("core_feature", raw, "사진 인식", 0)

    assert item["description"].endswith("줄입니다.")
    assert item["requirements"] == ["이미지 형식을 검증해야 합니다.", "인식 결과를 저장해야 합니다."]


def test_parallel_items_are_sorted_and_internal_order_is_removed():
    sections = {
        "releaseSchedule": [
            {"milestone": "두 번째", "_order": 1},
            {"milestone": "첫 번째", "_order": 0},
        ]
    }

    result = PrdAgent._finalize_parallel_sections(sections)

    assert [x["milestone"] for x in result["releaseSchedule"]] == ["첫 번째", "두 번째"]
    assert all("_order" not in x for x in result["releaseSchedule"])


def test_persona_accepts_use_pattern_alias():
    raw = (
        "NAME: Mina\nAGE: 28\nJOB: Designer\nTECH_LEVEL: high\n"
        "GOAL: Manage inventory. Reduce waste.\n"
        "PAIN_POINT: Manual checks take time. Expired food is wasted.\n"
        "USE_PATTERN: Checks the dashboard daily. Responds to alerts immediately."
    )
    item = PrdAgent._parse_item_text("persona", raw, "", 0)
    assert item["usagePattern"].startswith("Checks")
    assert PrdAgent._valid_item("persona", item)


def test_kpi_worker_keeps_only_first_block_when_model_overproduces():
    raw = (
        "METRIC: activation\nTARGET: 0% → 60%\nBASIS: internal estimate, 2026\n"
        "MEASUREMENT_METHOD: event count\nFREQUENCY: weekly\n"
        "METRIC: unrelated second metric\nTARGET: 0 → 1"
    )
    item = PrdAgent._parse_item_text("kpi", raw, "", 0)
    assert item["metric"] == "activation"
    assert item["frequency"] == "weekly"


def test_api_patch_replaces_matching_endpoint_and_preserves_others():
    draft = {
        "endpoints": [
            {"method": "GET", "path": "/api/v1/items", "description": "old"},
            {"method": "POST", "path": "/api/v1/items", "description": "create"},
        ]
    }
    patches = {
        "endpoints": [
            {"method": "GET", "path": "/api/v1/items", "description": "new"},
            {"method": "GET", "path": "/api/v1/recipes", "description": "recipes"},
        ]
    }

    result = _apply_review_patches("api", draft, patches)

    assert len(result["endpoints"]) == 3
    assert next(x for x in result["endpoints"] if x["method"] == "GET" and x["path"] == "/api/v1/items")["description"] == "new"
    assert any(x["path"] == "/api/v1/recipes" for x in result["endpoints"])


def test_prd_patch_merges_list_items_without_deleting_originals():
    draft = {
        "coreFeatures": [
            {"name": "재료 인식", "description": "old"},
            {"name": "유통기한 알림", "description": "keep"},
        ]
    }
    patches = {
        "coreFeatures": [
            {"name": "재료 인식", "description": "fixed"},
            {"name": "레시피 추천", "description": "added"},
        ]
    }

    result = _apply_review_patches("prd", draft, patches)

    assert len(result["coreFeatures"]) == 3
    assert next(x for x in result["coreFeatures"] if x["name"] == "재료 인식")["description"] == "fixed"
    assert any(x["name"] == "유통기한 알림" for x in result["coreFeatures"])


def test_legacy_full_api_output_is_reduced_to_non_destructive_patches():
    draft = {"endpoints": [{"method": "GET", "path": "/old"}]}
    fixed = {"endpoints": [{"method": "GET", "path": "/new"}]}

    patches = _derive_patches_from_full_output("api", draft, fixed)
    result = _apply_review_patches("api", draft, patches)

    assert {x["path"] for x in result["endpoints"]} == {"/old", "/new"}


def test_prd_audit_always_restores_three_distinct_personas():
    audited = PrdAgent._audit_sections(
        {"userPersonas": []}, ["문서 등록"], "웹에서 문서를 등록하는 서비스", "",
    )

    personas = audited["userPersonas"]
    assert len(personas) == 3
    assert len({item["name"] for item in personas}) == 3
