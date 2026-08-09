import asyncio
from types import SimpleNamespace

import main
from phase2.agents.prd_agent import _parse_plain_section
from routers.chat import (
    ChatMessageDto,
    ChatRequest,
    _clean_stage_suggestions,
    _collect_interview_state,
    _contextual_fallback_suggestions,
    _extract_service_subject,
    _is_valid_generated_project_name,
    _parse_labeled_text,
    _synthesize_form_data,
    message,
)


def test_chat_labeled_text_collects_repeated_suggestions():
    parsed = _parse_labeled_text(
        "MESSAGE: 어떤 플랫폼이 좋으신가요?\n"
        "SUGGESTION: 웹으로 만들고 싶어요\n"
        "SUGGESTION: 앱으로 만들고 싶어요\n"
        "SUGGESTION: 웹과 앱 모두 필요해요",
        {"SUGGESTION"},
    )
    assert parsed["MESSAGE"].endswith("?")
    assert len(parsed["SUGGESTION"]) == 3


def test_chat_endpoint_returns_contextual_question_and_complete_platform_enum_choices(monkeypatch):
    raw = (
        "MESSAGE: 팀 할 일이 메신저에 흩어진다고 하셨는데, 이 서비스를 웹과 앱 중 어디에서 가장 자주 쓰실까요?\n"
        "SUGGESTION: 팀원 모두가 브라우저에서 쓰는 웹 서비스가 좋아요\n"
        "SUGGESTION: 이동 중 확인할 수 있는 모바일 앱이 좋아요\n"
        "SUGGESTION: 사무실에서는 웹, 외부에서는 앱을 함께 쓰고 싶어요"
    )

    class FakeCompletions:
        async def create(self, **_kwargs):
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=raw))])

    fake_client = SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions()))
    monkeypatch.setattr(main, "exaone_client", fake_client)
    response = asyncio.run(message(ChatRequest(messages=[
        ChatMessageDto(role="assistant", content="어떤 서비스를 만들고 싶으신가요?"),
        ChatMessageDto(role="user", content="팀 할 일이 메신저에 흩어지는 문제를 해결하고 싶어요"),
    ])))
    data = response["data"]
    assert "팀 할 일이 메신저에 흩어지는 문제" in data["message"]
    assert "웹, 모바일 앱" in data["message"]
    assert data["suggestions"] == [
        "웹사이트(WEB)로 만들고 싶어요",
        "모바일 앱(APP)으로 만들고 싶어요",
        "웹과 앱 둘 다(WEB_APP) 필요해요",
    ]
    assert data["stage"] == "플랫폼"


def test_form_data_is_assembled_from_fixed_interview_order_without_llm_json():
    answers = [
        "냉장고 재고 관리 서비스를 만들고 싶어요",
        "웹사이트로 만들고 싶어요",
        "재고를 매번 눈으로 확인하기 불편해요",
        "메모 앱에 기록해요",
        "재고와 유통기한을 자동으로 알고 싶어요",
        "20-30대 1인 가구",
        "재료 추가, 유통기한 알림, 소비 기록",
        "냉프래시",
    ]
    messages = [ChatMessageDto(role="user", content=value) for value in answers]
    result = asyncio.run(_synthesize_form_data(messages, client=None))

    assert result["projectName"] == "냉프래시"
    assert result["platform"] == "WEB"
    assert [x["featureName"] for x in result["featureDefinition"]["customFeatures"]] == [
        "재료 추가", "유통기한 알림", "소비 기록",
    ]


def test_chat_does_not_advance_when_answer_is_an_unfilled_template():
    messages = [ChatMessageDto(role="user", content=value) for value in [
        "대학생용 일정 관리 앱을 만들고 싶어요",
        "모바일 앱으로 만들고 싶어요",
        "시험 일정과 과제가 겹치면 우선순위를 정하기 어려워요",
        "카카오톡 단체방과 메모 앱으로 확인해요",
        "~가 한눈에 보이면 좋겠어요",
    ]]

    state = _collect_interview_state(messages)

    assert state["stage"] == 3
    assert state["invalid"]["stage"] == 3
    assert "완성되지 않은" in state["invalid"]["reason"]


def test_chat_requires_three_distinct_features_before_naming():
    base_answers = [
        "대학생용 일정 관리 앱을 만들고 싶어요",
        "모바일 앱으로 만들고 싶어요",
        "시험 일정과 과제가 겹치면 우선순위를 정하기 어려워요",
        "카카오톡 단체방과 메모 앱으로 확인해요",
        "과제와 시험 일정 및 우선순위가 한 화면에 보이면 좋겠어요",
        "여러 과목을 수강하는 대학 재학생",
    ]
    one_feature = [*base_answers, "과목별 일정 자동 분류"]
    three_features = [*base_answers, "일정 등록, 과목별 자동 분류, 마감 알림"]

    invalid_state = _collect_interview_state([
        ChatMessageDto(role="user", content=value) for value in one_feature
    ])
    valid_state = _collect_interview_state([
        ChatMessageDto(role="user", content=value) for value in three_features
    ])

    assert invalid_state["stage"] == 5
    assert "1개만" in invalid_state["invalid"]["reason"]
    assert valid_state["stage"] == 6


def test_feature_suggestions_are_complete_three_feature_bundles():
    suggestions = _clean_stage_suggestions([
        "일정 등록",
        "일정 등록, 과목별 자동 분류, 마감 알림",
        "과제 공유, 담당자 지정, 완료 상태 확인",
        "기능1, 기능2, 기능3",
    ], 5)

    assert suggestions == [
        "일정 등록, 과목별 자동 분류, 마감 알림",
        "과제 공유, 담당자 지정, 완료 상태 확인",
    ]


def test_solution_rejects_a_repeated_pain_without_tool_or_action():
    messages = [ChatMessageDto(role="user", content=value) for value in [
        "대학생용 일정 관리 앱을 만들고 싶어요",
        "모바일 앱으로 만들고 싶어요",
        "마감일을 자주 놓치는 게 걱정돼요",
        "중요한 마감일을 놓치는 게 걱정돼요",
    ]]

    state = _collect_interview_state(messages)

    assert state["stage"] == 2
    assert "도구나 처리 행동" in state["invalid"]["reason"]


def test_persona_fallback_is_anchored_to_the_collected_service_role():
    state = {
        "idea": "대학생이 과제와 시험 일정을 관리하는 앱",
        "answers": [
            "모바일 앱",
            "과제와 시험 마감일을 자주 놓쳐요",
            "캘린더에 직접 기록해요",
            "마감 일정이 한눈에 보이면 좋겠어요",
        ],
    }

    suggestions = _contextual_fallback_suggestions(4, state)

    assert all("대학생" in suggestion for suggestion in suggestions)
    assert not any("1인 가구" in suggestion or "사장님" in suggestion for suggestion in suggestions)


def test_collaboration_persona_fallback_has_distinct_concrete_roles():
    state = {
        "idea": "팀 할 일이 메신저에 흩어지는 문제를 해결하고 싶어요",
        "answers": [
            "웹과 모바일 앱을 모두 만들고 싶어요",
            "담당자와 마감일이 여러 대화방에 흩어져서 업무를 놓쳐요",
            "캘린더와 메모 앱에 직접 기록해요",
            "대화에서 할 일을 자동으로 모아 한눈에 확인하고 싶어요",
        ],
    }

    suggestions = _contextual_fallback_suggestions(4, state)

    assert len(suggestions) == 3
    assert any("팀원" in suggestion for suggestion in suggestions)
    assert any("팀 리더" in suggestion for suggestion in suggestions)
    assert any("프로젝트 관리자" in suggestion for suggestion in suggestions)
    assert not any("해당 문제를 반복해서" in suggestion for suggestion in suggestions)


def test_feature_fallback_is_anchored_to_the_service_subject():
    state = {
        "idea": "대학생이 과제와 시험 일정을 함께 관리하는 앱",
        "answers": [
            "모바일 앱",
            "과제와 시험 마감일을 자주 놓쳐요",
            "캘린더에 직접 기록해요",
            "과제와 시험 일정이 자동으로 정리되면 좋겠어요",
            "여러 과목을 수강하는 대학생",
        ],
    }

    suggestions = _contextual_fallback_suggestions(5, state)

    assert all("과제와 시험 일정" in suggestion for suggestion in suggestions)
    assert all(len(suggestion.split(",")) == 3 for suggestion in suggestions)


def test_service_subject_extracts_object_before_automatic_action():
    state = {
        "idea": "팀 업무 관리 서비스",
        "answers": [
            "웹과 앱",
            "업무를 자주 놓쳐요",
            "메모 앱에 기록해요",
            "대화에서 할 일을 자동으로 모으고 담당자와 마감일을 확인하고 싶어요",
        ],
    }

    assert _extract_service_subject(state) == "할 일"
    assert all(
        "할 일" in suggestion
        for suggestion in _contextual_fallback_suggestions(5, state)
    )


def test_generated_project_name_rejects_unstructured_korean_coinage():
    assert _is_valid_generated_project_name("팀플로우")
    assert _is_valid_generated_project_name("TaskPulse")
    assert not _is_valid_generated_project_name("업무수퍼고")
    assert not _is_valid_generated_project_name("업무를 놓치지 않는 서비스")


def test_form_description_does_not_attach_problem_to_polite_sentence():
    answers = [
        "팀 업무 관리 서비스를 만들고 싶어요",
        "웹과 모바일 앱을 모두 만들고 싶어요",
        "담당자와 마감일이 흩어져서 업무를 자주 놓쳐요",
        "캘린더와 메모 앱에 직접 기록해요",
        "할 일을 자동으로 모아 한눈에 확인하고 싶어요",
        "메신저로 협업하는 직장인 팀원",
        "할 일 자동 추출, 담당자 지정, 마감 알림",
        "팀플로우",
    ]

    result = asyncio.run(_synthesize_form_data([
        ChatMessageDto(role="user", content=value) for value in answers
    ], client=None))

    assert "놓쳐요 문제" not in result["projectDescription"]
    assert "현재의 어려움" in result["projectDescription"]
    assert "라는 목표" in result["projectDescription"]


def test_prd_plain_sections_are_assembled_into_nested_python_objects():
    overview = _parse_plain_section(
        "projectOverview", "OVERVIEW: 냉장고 재고를 관리하는 서비스\nBACKGROUND: 시장 배경 설명", [], "요구사항",
    )
    scope = _parse_plain_section(
        "mvpScope", "RATIONALE: 사용자 가치가 높은 기능부터 구현합니다.", ["추가", "알림", "기록", "추천"], "요구사항",
    )
    tech = _parse_plain_section(
        "techStack", "BACKEND: FastAPI를 사용합니다.\nDATABASE: PostgreSQL을 사용합니다.", [], "요구사항",
    )

    assert overview["projectOverview"].startswith("냉장고")
    assert scope["mvpScope"]["included"] == ["추가", "알림", "기록"]
    assert scope["mvpScope"]["excluded"] == ["추천"]
    assert tech["techStack"]["backend"].startswith("FastAPI")
    assert tech["techStack"]["auth"]
