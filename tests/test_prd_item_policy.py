import asyncio
import copy
import json
from unittest.mock import AsyncMock

import pytest

from phase2.agents.prd_agent import (
    _KPI_DIMENSIONS,
    PrdAgent,
    _fallback_section,
    _persona_directions,
)
from phase2.agents.qa_agent import QaAgent
from phase2.prd_policy import (
    age_interval,
    align_project_scope,
    audit_items,
    kpi_issues,
    persona_issues,
)
from phase2.prd_quality import reconcile_prd_quality
from phase2.state import PipelineState


def metric():
    return {"metric": "첫 식물 등록 완료율", "dimension": "activation", "target": "0% → 60%",
            "basis": "제품 출시 전 기준선 0에서 시작하는 내부 운영 목표", "frequency": "주간",
            "measurementMethod": "가입자 중 첫 식물을 등록한 사용자 비율을 집계한다.",
            "measurement": {"event": "plant_created", "numerator": "첫 식물 등록 사용자 수",
                            "denominator": "가입 사용자 수", "unit": "%", "window": "가입 후 7일"}}


def persona(name="김하늘", goal="식물 등록", pattern="등록 후 관리 일정을 확인한다."):
    return {"name": name, "age": "연령 무관", "job": "반려식물 재배자", "role": "개인 재배자",
            "techLevel": "중간", "goal": goal, "painPoint": "기록이 분산돼 확인이 어렵다.", "usagePattern": pattern}


def test_activation_is_valid_without_specific_korean_keyword():
    assert not PrdAgent._item_quality_issues("kpi", metric(), "", _KPI_DIMENSIONS[1])
    assert audit_items({"kpi": [metric()]}, "", "")["kpi"] == [metric()]


def test_prd_machine_release_scope_enum_is_not_prose_contamination():
    document = {"coreFeatures": [{"name": "Item", "releaseScope": "mvp"}]}
    issues = object.__new__(QaAgent)._check_prd_completeness(document, 1)
    assert not any("오염 문자열" in issue for issue in issues)


def test_machine_segment_hash_does_not_trigger_source_identifier_prose_gate():
    item = {**persona(), "segmentId": "persona.012345abcdef"}
    assert not persona_issues(item)
    assert len(audit_items({"userPersonas": [item]}, "", "")["userPersonas"]) == 1
    item["goal"] = "사용 통계 식별자 011468을 노출한다"
    assert "source identifier leak" in persona_issues(item)


def test_active_kpi_worker_parses_typed_measurement_without_fallback_or_retry():
    agent = PrdAgent(object())
    agent._call = AsyncMock(return_value=(
        "DIMENSION: activation\nMETRIC: 첫 식물 등록 완료율\nTARGET: 0% → 60%\n"
        "BASIS: 검증할 내부 운영 목표\nMEASUREMENT_METHOD: 가입자 중 첫 식물을 등록한 사용자 비율\n"
        "FREQUENCY: 주간\nEVENT: plant_created\nNUMERATOR: 첫 식물 등록 사용자 수\nDENOMINATOR: 가입 사용자 수\n"
        "UNIT: %\nWINDOW: 가입 후 7일\nSELF_CHECK: PASS"
    ))
    result = asyncio.run(agent._item_worker_node({"item_kind": "kpi", "item_index": 1, "item_context": _KPI_DIMENSIONS[1],
                                                  "prompt": "typed KPI", "section": "goalsKpi"}))
    assert agent._call.await_count == 1
    assert result["sections"]["kpi"][0]["measurement"]["event"] == "plant_created"
    assert not result["sections"].get("_qualityFallbacks")


def test_active_persona_worker_preserves_provided_role_and_marks_hypothesis():
    agent = PrdAgent(object())
    agent._call = AsyncMock(return_value=(
        "ROLE: 임의 역할\nNAME: 김하늘\nAGE: 20대 초반\nJOB: 학생\nTECH_LEVEL: 중간\n"
        "GOAL: 물주기 기록을 확인하고 싶다.\nPAIN_POINT: 날짜를 자주 잊는다.\n"
        "USAGE_PATTERN: 물을 준 뒤 기록한다.\nSELF_CHECK: PASS"
    ))
    result = asyncio.run(agent._item_worker_node({"item_kind": "persona", "item_index": 0,
        "item_context": "대상 사용자: 초보 재배자; 근거 기반 작성", "prompt": "persona", "section": "userPersonas",
        "user_query": "20~30대 사용자의 식물 관리"}))
    item = result["sections"]["userPersonas"][0]
    assert agent._call.await_count == 1 and item["role"] == "초보 재배자"
    assert item["assumption"] and item["segmentId"].startswith("persona.")


def test_time_unit_is_accepted_by_worker_assembly_and_qa_policy():
    value = metric()
    value.update(metric="평균 처리 시간", dimension="business", target="10초 → 5초")
    value["measurement"].update(unit="초", denominator="완료된 처리 건수")
    assert not kpi_issues(value, require_measurement=True)
    assert not PrdAgent._item_quality_issues("kpi", value, "", _KPI_DIMENSIONS[6])
    assert audit_items({"kpi": [value]}, "", "")["kpi"] == [value]
    qa = object.__new__(QaAgent)
    assert not any("KPI" in issue for issue in qa._check_prd_completeness({"kpi": [value]}, 0))


def test_average_measurement_requires_count_denominator_but_sum_does_not():
    item = metric()
    item.update(metric="평균 처리 시간", measurementMethod="전체 처리 시간의 합계를 완료 건수로 나눈다")
    item["measurement"].update(unit="초", denominator=None)
    assert "KPI_MEASUREMENT: average denominator required" in kpi_issues(item)
    item["measurement"]["denominator"] = "완료된 처리 건수"
    assert not kpi_issues(item)
    item.update(metric="총 처리 시간", measurementMethod="전체 처리 시간의 합계를 집계한다")
    item["measurement"]["denominator"] = None
    assert not kpi_issues(item)


def test_ratio_requires_denominator_in_all_active_checks():
    value = metric()
    value["measurement"]["denominator"] = None
    assert any("denominator" in issue for issue in kpi_issues(value, require_measurement=True))
    assert audit_items({"kpi": [value]}, "", "")["kpi"] == []


def test_persona_dedup_uses_journey_not_age_job_or_display_name():
    values = [persona(), persona(goal="사진 비교", pattern="주말마다 사진을 올려 비교한다.")]
    result = audit_items({"userPersonas": values}, "", "")
    assert len(result["userPersonas"]) == 2
    assert all(item["assumption"] for item in result["userPersonas"])
    assert len(audit_items({"userPersonas": [values[0], copy.deepcopy(values[0])]}, "", "")["userPersonas"]) == 1


def test_age_subranges_are_normalized_and_outside_range_is_rejected():
    value = persona()
    value["age"] = "20대 초반"
    assert age_interval(value["age"]) == (20, 23)
    assert not persona_issues(value, "20~30대 식물 재배자")
    value["age"] = "40대"
    assert any("outside" in issue for issue in persona_issues(value, "20~30대 식물 재배자"))


@pytest.mark.parametrize(("value", "expected"), [
    ("20대~30대", (20, 39)), ("20~30대", (20, 39)), ("20대 ~ 30대 중반", (20, 36)),
    ("20세~30세", (20, 30)), ("20~30세", (20, 30)), ("만 25세", (25, 25)),
    ("40대 이상", (40, 150)), ("20대 후반", (27, 29)), ("30세 이하", (0, 30)),
])
def test_age_range_notations_share_the_same_validation(value, expected):
    assert age_interval(value) == expected


@pytest.mark.parametrize("value", ["30~20세", "30세~20세", "30대~20대", "20대 후반~20대 초반", "25대", "20세~30대", "151세"])
def test_invalid_or_reversed_age_intervals_are_rejected(value):
    with pytest.raises(ValueError, match="PERSONA_AGE"):
        age_interval(value)


@pytest.mark.parametrize("query", ["20대~30대 대상", "20~30대 대상", "20세~39세 대상", "20~39세 대상"])
def test_persona_age_checks_confirmed_range_for_each_supported_notation(query):
    item = {**persona(), "age": "20대 초반"}
    assert not persona_issues(item, query)
    item["age"] = "40대"
    assert any("outside" in issue for issue in persona_issues(item, query))


def test_no_fictional_cardinality_fill_and_diagnostics_survive_public_quality_metadata():
    result = audit_items({"kpi": [], "userPersonas": []}, "", "")
    public = reconcile_prd_quality(result)
    assert public["kpi"] == [] and public["userPersonas"] == []
    assert public["qualityMetadata"]["itemDiagnostics"]
    assert _fallback_section("goalsKpi", ["식물 등록"], "")["kpi"] == []
    assert _fallback_section("userPersonas", ["식물 등록"], "")["userPersonas"] == []


def test_persona_worker_count_comes_from_actual_segments():
    assert len(_persona_directions({"persona_segments": ["초보 재배자"]})) == 1
    assert len(_persona_directions({"persona_segments": ["초보 재배자", "전문 재배자"]})) == 2


def test_scope_preserves_pm_and_user_exclusions_not_just_existing_feature_names():
    document = {"coreFeatures": [], "mvpScope": {"included": ["기록"], "excluded": []}}
    align_project_scope(document, {"scope": {"out": ["거래"]}, "priorities": {"wont": ["센서 연동"]}}, ["광고"])
    assert document["mvpScope"]["excluded"] == ["거래", "센서 연동", "광고"]
    assert document["mvpScope"]["included"] == ["기록"]


def test_retention_keeps_declared_event_and_segment_keeps_its_journey():
    kpi = metric()
    kpi.update(metric="7일 재방문율", dimension="retention_7d")
    kpi["measurement"]["event"] = "actual_repeat_event"
    person = {**persona(), "segmentId": "persona.user"}
    document = {"coreFeatures": [{"name": "다른 표현의 핵심 기능"}], "kpi": [kpi], "userPersonas": [person]}
    result = reconcile_prd_quality(document)
    assert result["kpi"][0]["measurementDefinition"]["returnEvent"] == "actual_repeat_event"
    assert result["kpi"][0]["measurementMethod"] == kpi["measurementMethod"]
    assert result["userPersonas"][0]["usagePattern"] == person["usagePattern"]


def test_kpi_targeted_repair_resolves_history_and_preserves_other_sections():
    document = {"goals": ["Preserve original goal"], "kpi": [],
                "qualityMetadata": {"itemDiagnostics": [{"section": "kpi", "reasons": ["unknown dimension"]}]}}
    agent = PrdAgent(object())
    agent._call = AsyncMock(return_value=json.dumps({"patches": {"kpi": [metric()]}}))
    result = asyncio.run(agent.repair(PipelineState(prd_document=json.dumps(document)), "kpi measurement 보완"))
    repaired = json.loads(result.prd_document)
    assert repaired["goals"] == document["goals"]
    assert repaired["qualityMetadata"]["itemDiagnostics"][0]["status"] == "resolved_by_targeted_repair"
    issues = object.__new__(QaAgent)._check_prd_completeness(repaired, 0)
    assert not any("placeholder" in issue or "오염 문자열" in issue or "KPI 검증" in issue for issue in issues)


def test_invalid_targeted_measurement_does_not_replace_baseline():
    document = {"kpi": [metric()]}
    state = PipelineState(prd_document=json.dumps(document))
    agent = PrdAgent(object())
    invalid = metric()
    invalid["measurement"]["denominator"] = None
    agent._call = AsyncMock(return_value=json.dumps({"patches": {"kpi": [invalid]}}))
    result = asyncio.run(agent.repair(state, "kpi measurement 보완"))
    assert result.prd_document == state.prd_document
    assert result.targeted_repair_results["prd"] == "rejected_validation"


def test_qa_does_not_accept_one_persona_if_an_explicit_target_segment_is_missing():
    document = {"userPersonas": [{**persona(), "segmentId": "persona.one"}],
                "qualityMetadata": {"expectedPersonaSegments": [
                    {"segmentId": "persona.one", "role": "재배자"}, {"segmentId": "persona.two", "role": "운영자"},
                ]}}
    issues = object.__new__(QaAgent)._check_prd_completeness(document, 0)
    assert any("PERSONA_SEGMENT_MISSING: 운영자" in issue for issue in issues)
