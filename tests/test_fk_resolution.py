from phase2.agents.dba_agent import (
    _build_name_lookup,
    _ensure_fk_references,
    _fk_target,
    dba_self_check,
)
from phase2.agents.feature_spec_agent import (
    FeatureSpecAgent,
    _ensure_feature_narrative,
    _ensure_required_contracts,
)
from phase2.agents.qa_agent import QaAgent
from phase2.state import PipelineState


def test_fk_resolution_handles_irregular_plural_and_prefixed_columns():
    tables = [
        {
            "name": "comparison_searches",
            "columns": [{"name": "id", "type": "BIGINT", "constraints": "PRIMARY_KEY"}],
        },
        {
            "name": "comparison_search_items",
            "columns": [{"name": "search_id", "type": "BIGINT", "constraints": "FOREIGN_KEY"}],
        },
        {
            "name": "restaurants",
            "columns": [{"name": "id", "type": "BIGINT", "constraints": "PRIMARY_KEY"}],
        },
        {
            "name": "platform_restaurant_listings",
            "columns": [{
                "name": "external_restaurant_id",
                "type": "VARCHAR(255)",
                "constraints": "FOREIGN_KEY",
            }],
        },
    ]

    lookup = _build_name_lookup(tables)
    assert _fk_target("search_id", lookup) == "comparison_searches"
    assert _fk_target("external_restaurant_id", lookup) == "restaurants"

    _ensure_fk_references(tables)
    assert "REFERENCES comparison_searches(id)" in tables[1]["columns"][0]["constraints"]
    assert "REFERENCES restaurants(id)" in tables[3]["columns"][0]["constraints"]


def test_supporting_feature_does_not_require_prd_core_membership():
    registry = [{
        "featureId": "feature_001.platform-retry",
        "name": "플랫폼 재시도",
        "source": "supporting",
        "priority": "P0",
        "dbContract": {"tables": [], "foreignKeys": []},
    }]

    _, _, prd_issues = QaAgent._check_registry_contracts(
        registry,
        {"tables": []},
        {"endpoints": []},
        {"coreFeatures": []},
    )
    assert prd_issues == []


def test_feature_spec_backfills_description_and_requirements():
    result = _ensure_feature_narrative({"name": "예약 생성"}, {}, {})
    assert result["description"]
    assert result["requirements"]


def test_feature_spec_adds_plan_supporting_features_and_manage_contracts():
    state = PipelineState(project_plan={"priorities": {"should": ["여행 일정 저장"]}})
    result = FeatureSpecAgent._plan_supporting_specs(
        state,
        [{"featureId": "travel.search", "name": "여행지 검색", "source": "prd_core"}],
    )
    assert result[0]["name"] == "여행 일정 저장"
    contract = _ensure_required_contracts({"featureId": "travel.save", "name": "여행 일정 저장", "actions": ["manage"]})
    assert len(contract["apiContract"]) == 2


def test_feature_spec_derives_two_implementation_subfeatures_per_core():
    result = FeatureSpecAgent._derive_missing_subfeatures([
        {
            "featureId": "travel.course",
            "name": "여행 코스 추천",
            "source": "prd_core",
            "actions": ["create"],
        },
    ])
    assert len(result) == 2
    assert all(item["source"] == "supporting" for item in result)
    assert all(item["parentFeatureId"] == "travel.course" for item in result)
    assert all(item["description"] and item["requirements"] for item in result)


def test_fk_resolution_handles_role_prefixed_and_self_referencing_columns():
    tables = [
        {"name": "recommended_courses", "columns": [
            {"name": "id", "type": "BIGINT", "constraints": "PRIMARY_KEY"},
            {"name": "parent_course_id", "type": "BIGINT", "constraints": "FOREIGN_KEY"},
        ]},
        {"name": "travel_places", "columns": [
            {"name": "id", "type": "BIGINT", "constraints": "PRIMARY_KEY"},
        ]},
        {"name": "route_segments", "columns": [
            {"name": "id", "type": "BIGINT", "constraints": "PRIMARY_KEY"},
            {"name": "from_place_id", "type": "BIGINT", "constraints": "FOREIGN_KEY"},
            {"name": "to_place_id", "type": "BIGINT", "constraints": "FOREIGN_KEY"},
        ]},
    ]

    _ensure_fk_references(tables)
    course_constraints = tables[0]["columns"][1]["constraints"]
    segment_constraints = [column["constraints"] for column in tables[2]["columns"][1:]]
    assert "REFERENCES recommended_courses(id)" in course_constraints
    assert all("REFERENCES travel_places(id)" in value for value in segment_constraints)


def test_user_role_foreign_keys_resolve_to_users():
    tables = [
        {"name": "users", "columns": [{"name": "id", "type": "BIGINT", "constraints": "PRIMARY_KEY"}]},
        {"name": "messages", "columns": [{"name": "id", "type": "BIGINT", "constraints": "PRIMARY_KEY"}, {"name": "sender_id", "type": "BIGINT", "constraints": "FOREIGN_KEY"}]},
        {"name": "messages_2", "columns": [{"name": "id", "type": "BIGINT", "constraints": "PRIMARY_KEY"}, {"name": "sender_id", "type": "BIGINT", "constraints": "FOREIGN_KEY"}]},
        {"name": "reports", "columns": [{"name": "id", "type": "BIGINT", "constraints": "PRIMARY_KEY"}, {"name": "reporter_id", "type": "BIGINT", "constraints": "FOREIGN_KEY"}]},
        {"name": "reviews", "columns": [{"name": "id", "type": "BIGINT", "constraints": "PRIMARY_KEY"}, {"name": "reviewee_id", "type": "BIGINT", "constraints": "FOREIGN_KEY"}]},
    ]

    _ensure_fk_references(tables)
    for table in tables[1:]:
        assert "REFERENCES users(id)" in table["columns"][1]["constraints"]


def test_dba_self_check_repairs_and_reports_only_unresolved_fk_rules():
    tables = [
        {"name": "users", "columns": [{"name": "id", "type": "BIGINT", "constraints": "PRIMARY_KEY"}]},
        {"name": "messages", "columns": [{"name": "id", "type": "BIGINT", "constraints": "PRIMARY_KEY"}, {"name": "sender_id", "type": "BIGINT", "constraints": "FOREIGN_KEY"}]},
        {"name": "external_events", "columns": [{"name": "id", "type": "BIGINT", "constraints": "PRIMARY_KEY"}, {"name": "provider_message_id", "type": "VARCHAR(100)", "constraints": "FOREIGN_KEY"}]},
    ]

    checked, relationships, issues = dba_self_check(tables, [])
    assert "REFERENCES users(id)" in checked[1]["columns"][1]["constraints"]
    assert issues == []
    assert any("users (1:N) messages" == value for value in relationships)


def test_dba_self_check_blocks_unresolved_id_without_foreign_key_constraint():
    tables = [
        {"name": "messages", "columns": [
            {"name": "id", "type": "BIGINT", "constraints": "PRIMARY_KEY"},
            {"name": "sender_id", "type": "BIGINT", "constraints": "NOT_NULL"},
        ]},
    ]

    _, _, issues = dba_self_check(tables, [])
    assert issues == [
        "DBA_SELF_CHECK_BLOCKER: messages.sender_id의 FK 참조 대상을 찾을 수 없습니다"
    ]
