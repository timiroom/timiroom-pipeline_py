import asyncio
import json
from types import SimpleNamespace

from phase2.agents.prd_agent import PrdAgent
from phase2.llm_runtime import LlmRuntime
from phase2.state import PipelineState
from routers import chat


def test_prd_execute_assembles_worker_text_before_pm_and_keeps_confirmed_priorities():
    class Completions:
        async def create(self, **kwargs):
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=""))])

    client = SimpleNamespace(chat=SimpleNamespace(completions=Completions()))
    state = PipelineState(
        user_query="대학생 과제와 시험 일정을 관리하는 서비스",
        must_features=["일정 등록", "마감 알림"],
        should_features=["과목 분류"],
        could_features=["일정 공유"],
    )

    result = asyncio.run(PrdAgent(client).execute(state))
    document = json.loads(result.prd_document)

    assert result.feature_list == ["일정 등록", "마감 알림", "과목 분류", "일정 공유"]
    assert {item["name"]: item["priority"] for item in document["coreFeatures"]} == {
        "일정 등록": "P0", "마감 알림": "P0", "과목 분류": "P1", "일정 공유": "P2",
    }
    assert len(document["kpi"]) == 7
    assert len(document["userPersonas"]) == 3
    assert len(document["releaseSchedule"]) == 6


def test_chat_keeps_invalid_answer_at_same_stage_and_accepts_its_correction(monkeypatch):
    async def fake_call(*_args, **_kwargs):
        return json.dumps({"message": "관계 없는 질문인가요?", "suggestions": []})

    monkeypatch.setattr(chat, "_call_openai", fake_call)
    answers = [
        "대학생 과제 관리 서비스를 만들고 싶어요", "모바일 앱",
        "시험과 과제 마감일을 자주 놓쳐요", "메모 앱에 기록해요",
        "~가 한눈에 보이면 좋겠어요",
    ]
    messages = [chat.ChatMessageDto(role="user", content=value) for value in answers]
    invalid = asyncio.run(chat.message(chat.ChatRequest(messages=messages)))["data"]
    corrected = asyncio.run(chat.message(chat.ChatRequest(messages=[
        *messages, chat.ChatMessageDto(role="user", content="과제와 시험 일정이 한눈에 보이면 좋겠어요"),
    ])))["data"]

    assert invalid["stage"] == "원하는 이상적 상태"
    assert "완성되지 않은" in invalid["message"]
    assert corrected["stage"] == "타겟 유저"
    assert all("대학생" in suggestion for suggestion in corrected["suggestions"])


def test_prd_worker_assembly_preserves_registry_contracts_through_bounded_runtime():
    class Completions:
        async def create(self, **kwargs):
            raw = ""
            if "NAME: 기능명" in kwargs["messages"][1]["content"]:
                raw = (
                    "NAME: 모델이 제안한 다른 이름\n"
                    "DESCRIPTION: 사용자가 일정 등록을 요청하면 입력을 검증하고 저장하여 과제와 시험 일정을 추적합니다.\n"
                    "PRIORITY: P2\nREQUIREMENTS: 일정 등록 전에 제목과 마감일을 검증해야 합니다.\n"
                    "SELF_CHECK: PASS"
                )
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=raw))])

    state = PipelineState(
        user_query="일정 관리 서비스", feature_list=["일정 등록"], must_features=["일정 등록"],
        feature_registry=[{
            "featureId": "schedule_create", "name": "일정 등록", "actions": ["create"],
            "apiContract": [{"method": "POST", "path": "/api/v1/schedules"}],
            "dbContract": {"tables": ["schedules"], "foreignKeys": []},
        }],
    )
    client = SimpleNamespace(chat=SimpleNamespace(completions=Completions()))
    agent = PrdAgent(client, runtime=LlmRuntime(2, 5))

    result = asyncio.run(agent.execute(state))
    feature = json.loads(result.prd_document)["coreFeatures"][0]

    assert feature["featureId"] == "schedule_create"
    assert feature["name"] == "일정 등록"
    assert "과제와 시험" in feature["description"]
    assert feature["actions"] == ["create"]
    assert feature["apiContract"][0]["method"] == "POST"
    assert feature["apiContract"][0]["path"] == "/api/v1/schedules"
    assert feature["dbContract"] == {"tables": ["schedules"], "foreignKeys": []}


def test_chat_name_candidates_reject_generated_sentences_and_use_contextual_fallback(monkeypatch):
    async def fake_call(*_args, **_kwargs):
        return json.dumps({"names": ["업무수퍼고", "업무를 놓치지 않는 서비스", "TaskPulse"]})

    monkeypatch.setattr(chat, "_call_openai", fake_call)
    answers = [
        "대학생 과제 관리 서비스를 만들고 싶어요", "모바일 앱",
        "시험과 과제 마감일을 자주 놓쳐요", "메모 앱에 기록해요",
        "과제와 시험 일정이 한눈에 보이면 좋겠어요", "여러 과목을 수강하는 대학생",
        "일정 등록, 과목 분류, 마감 알림",
    ]
    result = asyncio.run(chat.message(chat.ChatRequest(messages=[
        chat.ChatMessageDto(role="user", content=value) for value in answers
    ])))["data"]

    assert result["stage"] == "naming"
    assert len(result["suggestions"]) == 3
    assert all("과제와시험일정" in name for name in result["suggestions"])
