import asyncio

import pytest
from fastapi import HTTPException

from routers import qa_scenario


def test_parse_scenarios_keeps_valid_items_and_normalizes_type():
    text = """```json
    {"scenarios": [
      {"type": "normal", "title": "정상 등록", "given": "로그인", "when": "등록", "then": "201"},
      {"type": "weird", "title": "알 수 없는 유형", "then": "기본값"},
      {"type": "EXCEPTION", "title": "", "then": "제목 없음은 버린다"},
      {"type": "BOUNDARY", "title": "예상 결과 없음"}
    ]}
    ```"""
    scenarios = qa_scenario.parse_scenarios(text)
    assert [s["title"] for s in scenarios] == ["정상 등록", "알 수 없는 유형"]
    assert [s["type"] for s in scenarios] == ["NORMAL", "NORMAL"]


def test_parse_verdict_falls_back_to_inconclusive():
    result = qa_scenario.parse_verdict('설명 {"verdict": "maybe", "evidence": ["a.py", " "]} 끝')
    assert result["verdict"] == "INCONCLUSIVE"
    assert result["evidence"] == ["a.py"]
    assert result["reasoning"]


def test_generate_rejects_empty_scenarios(monkeypatch):
    async def fake_ask(system_prompt, context):
        assert "[FEATURE]" in context and "회원 가입" in context
        return '{"scenarios": []}', "test-model"

    monkeypatch.setattr(qa_scenario, "_ask", fake_ask)
    with pytest.raises(HTTPException) as error:
        asyncio.run(qa_scenario.generate(qa_scenario.GenerateRequest(featureName="회원 가입")))
    assert error.value.status_code == 502


def test_evaluate_returns_backend_contract(monkeypatch):
    async def fake_ask(system_prompt, context):
        assert "### UserService.java" in context
        return '{"verdict": "PASS", "reasoning": "근거 확인", "evidence": ["UserService.java"]}', "test-model"

    monkeypatch.setattr(qa_scenario, "_ask", fake_ask)
    result = asyncio.run(qa_scenario.evaluate(qa_scenario.EvaluateRequest(
        scenarioTitle="가입", then="201", codeFiles=[qa_scenario.CodeFile(filename="UserService.java", content="...")])))
    assert result == {"agent": "QA_SCENARIO_AGENT", "model": "test-model", "verdict": "PASS",
                      "reasoning": "근거 확인", "evidence": ["UserService.java"]}
