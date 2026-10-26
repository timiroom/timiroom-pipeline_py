import asyncio
import sys
from types import SimpleNamespace

import main
import routers.chat as chat_module
from phase2.agents.prd_agent import _parse_plain_section
from routers.chat import (
    ChatMessageDto,
    ChatRequest,
    _clean_stage_suggestions,
    _collect_interview_state,
    _contextual_fallback_suggestions,
    _extract_service_subject,
    _fallback_project_name_candidates,
    _is_valid_generated_project_name,
    _parse_labeled_text,
    _synthesize_form_data,
    message,
)


def test_stdio_configuration_does_not_replace_stream_objects(monkeypatch):
    class ReconfigurableStream:
        def __init__(self):
            self.calls = []

        def reconfigure(self, **kwargs):
            self.calls.append(kwargs)

    stdout = ReconfigurableStream()
    stderr = ReconfigurableStream()
    monkeypatch.setattr(sys, "stdout", stdout)
    monkeypatch.setattr(sys, "stderr", stderr)

    main._configure_utf8_stdio()

    assert sys.stdout is stdout
    assert sys.stderr is stderr
    assert stdout.calls == [{"encoding": "utf-8", "errors": "replace"}]
    assert stderr.calls == [{"encoding": "utf-8", "errors": "replace"}]


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
    monkeypatch.setattr(main, "openai_client", fake_client)
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


def test_chat_synthesis_hands_a_runnable_form_to_the_pipeline():
    """채팅 합성 결과가 그대로 /orchestration/generate 입력으로 성립해야 한다."""
    import json

    from phase1.form_to_query import FormToQueryService
    from phase1.models import FormData

    messages = [ChatMessageDto(role="user", content=text) for text in (
        "대화에서 할 일을 자동으로 정리해주는 서비스를 만들고 싶어요",
        "웹과 앱 둘 다(WEB_APP) 필요해요",
        "여러 대화방에 흩어진 할 일을 매번 직접 찾아야 해서 번거로워요",
        "메모 앱에 그때그때 기록해 둬요",
        "대화에서 할 일이 자동으로 정리되면 좋겠어요",
        "여러 대화방에서 담당 업무를 확인하는 팀원",
        "할 일 자동 추출, 담당자 지정, 마감일 알림",
        "플로우메이트",
    )]

    state = _collect_interview_state(messages)
    assert state["stage"] == 6
    assert state["extras"] == ["플로우메이트"]

    payload = asyncio.run(_synthesize_form_data(messages, client=None))
    assert payload is not None
    assert payload["projectName"] == "플로우메이트"

    form = FormData.model_validate_json(json.dumps(payload, ensure_ascii=False))
    service = FormToQueryService()
    assert form.platform.value == "WEB_APP"
    assert service.extract_must_features(form) == ["할 일 자동 추출", "담당자 지정", "마감일 알림"]
    assert service.extract_all_included_features(form)
    assert service.synthesize(form).strip()


def test_quoted_user_wording_gets_the_right_korean_particle():
    """질문은 사용자 문장을 그대로 인용하므로 조사가 종성에 맞아야 한다."""
    from routers.chat import _josa

    assert _josa("번거로워요", "이라고", "라고") == "라고"
    assert _josa("자동 정리됨", "이라고", "라고") == "이라고"
    assert _josa("한눈에 보이는 통합 화면", "을", "를") == "을"
    assert _josa("정리되면 좋겠어요", "을", "를") == "를"
    # anchor()가 붙이는 말줄임표는 건너뛰고 마지막 한글로 판단한다.
    assert _josa("아주 긴 문장…", "이라고", "라고") == "이라고"
    assert _josa("", "이라고", "라고") == "라고"

    messages = [ChatMessageDto(role="user", content=value) for value in (
        "할 일 정리 서비스", "웹", "할 일이 여기저기 흩어져서 번거로워요", "메모 앱에 기록해요",
        "한눈에 보이는 통합 화면",
    )]
    assert "번거로워요’라고 하셨는데" in chat_module._contextual_question(messages, 2)
    assert "통합 화면’을 가장 필요로" in chat_module._contextual_question(messages, 4)


def test_service_subject_rejects_a_whole_clause_and_uses_the_managed_noun():
    """관리 대상 자리에 절이 잡히면 기능 제안과 프로젝트명이 모두 망가진다.

    실측(사내 비품 요청·승인 라이브): '필요할 때 요청하고 관리자가 승인하는 과정'이
    주어로 잡혀 '필요할때요청하고관리메이트'가 후보로 제시됐다.
    """
    state = {
        "idea": "회사에서 비품이 필요할 때 요청하고 관리자가 승인하는 과정을 관리하는 서비스를 만들고 싶어요",
        "answers": [
            "웹사이트(WEB)로 만들고 싶어요",
            "비품 요청을 메신저로 보내서 추적이 안 되고 승인 여부도 흐려져요",
            "엑셀 파일에 요청 내역을 적고 메신저로 승인 여부를 주고받아요",
            "요청부터 승인, 지급까지 상태가 한 화면에서 보이고 담당자가 바뀌어도 이력이 남으면 좋겠어요",
            "비품을 요청하는 일반 직원과 요청을 승인하는 총무 담당자",
            "비품 요청 등록, 요청 승인 및 반려, 재고 수량 관리",
        ],
    }

    assert _extract_service_subject(state) == "비품"
    assert _fallback_project_name_candidates(state) == ["비품메이트", "비품플로우", "비품허브"]
    assert all("비품" in suggestion for suggestion in _contextual_fallback_suggestions(5, state))


def test_subject_phrase_filter_keeps_noun_phrases_and_drops_clauses():
    from routers.chat import _is_subject_phrase

    for value in ("비품", "할 일", "과제와 시험 일정", "재고 수량"):
        assert _is_subject_phrase(value), value
    for value in (
        "필요할 때 요청하고 관리자가 승인하는 과정",
        "담당자가 바뀌어도 남는 이력",
        "자동으로 정리되는 목록",
        "요청하는 비품",
        "",
    ):
        assert not _is_subject_phrase(value), value
