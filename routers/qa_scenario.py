"""QA 시나리오 에이전트 — 기능 명세로 테스트 시나리오를 만들고, 시나리오가 통과할지 판정한다.

백엔드 QA 도메인(QaScenarioService)이 부른다. 판정은 코드를 실행하지 않고, 받은 API·DB 명세와
관련 코드 조각만 근거로 추론한다. 근거가 모자라면 PASS/FAIL 대신 INCONCLUSIVE로 답한다.
"""
import asyncio
import json
import logging

from fastapi import APIRouter, HTTPException
from openai import (
    APIConnectionError,
    APITimeoutError,
    InternalServerError,
    RateLimitError,
)
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/agents/qa-scenario", tags=["qa-scenario"])

AGENT_NAME = "QA_SCENARIO_AGENT"
SCENARIO_TYPES = {"NORMAL", "EXCEPTION", "BOUNDARY"}
VERDICTS = {"PASS", "FAIL", "INCONCLUSIVE"}

GENERATE_SYSTEM_PROMPT = """당신은 소프트웨어 QA 엔지니어 에이전트입니다. 주어진 기능 명세(설명·요구사항)와
API/DB 명세를 근거로 실행 가능한 테스트 시나리오를 생성하세요.

각 시나리오는 Given-When-Then 형식으로 작성하고, 다음 세 유형을 각각 최소 1개 이상 포함하세요:
- NORMAL(정상): 요구사항이 정상적으로 충족되는 경우
- EXCEPTION(예외): 잘못된 입력이나 실패 조건
- BOUNDARY(경계): 입력 길이·개수·범위 등 한계값

명세에 없는 세부사항은 추측하지 말고, 명세에서 확인 가능한 내용을 근거로 작성하세요.
시나리오는 모두 한국어로 쓰고 3~8개로 만드세요.

JSON 외 텍스트는 출력하지 마세요.
{"scenarios": [{"type": "NORMAL|EXCEPTION|BOUNDARY", "title": "...", "given": "...", "when": "...", "then": "..."}]}"""

EVALUATE_SYSTEM_PROMPT = """당신은 시니어 QA 엔지니어 에이전트입니다. 주어진 테스트 시나리오(Given-When-Then)가
현재 구현된 코드와 명세를 근거로 통과할지 판단하세요.

실제로 코드를 실행하지 않습니다. 제공된 코드 조각·API 명세·DB 명세만 근거로
논리적으로 추론하세요. 근거 없이 추측하지 마세요.

판정 기준:
- PASS: 제공된 근거에서 시나리오의 then(예상 결과)이 실제로 충족됨을 확인함
- FAIL: 제공된 근거에서 given/when 대비 then이 충족되지 않음을 확인함
- INCONCLUSIVE: 판단에 필요한 코드나 명세가 부족해 확정할 수 없음 — 코드 조각이
  전혀 제공되지 않았다면 명세만으로 단정하지 말고 우선 INCONCLUSIVE를 고려하세요

JSON 외 텍스트는 출력하지 마세요.
{"verdict": "PASS|FAIL|INCONCLUSIVE", "reasoning": "판단 근거를 설명하는 한국어 문장", "evidence": ["근거로 사용한 구체적 코드/명세 조각 또는 파일명"]}"""


class GenerateRequest(BaseModel):
    projectName: str | None = None
    featureName: str | None = None
    featureDescription: str | None = None
    requirements: str | None = None
    apiSpec: str | None = None
    dbSchema: str | None = None


class CodeFile(BaseModel):
    filename: str | None = None
    content: str | None = None


class EvaluateRequest(BaseModel):
    projectName: str | None = None
    featureName: str | None = None
    scenarioTitle: str | None = None
    given: str | None = None
    when: str | None = None
    then: str | None = None
    apiSpec: str | None = None
    dbSchema: str | None = None
    codeFiles: list[CodeFile] = Field(default_factory=list)


def _text(value: str | None, limit: int) -> str:
    if not value or not value.strip():
        return "(없음)"
    return value if len(value) <= limit else value[:limit] + "\n...[truncated]"


def _extract_json(text: str) -> dict:
    clean = (text or "").replace("```json", "").replace("```", "").strip()
    start, end = clean.find("{"), clean.rfind("}")
    if start < 0 or end < start:
        raise ValueError("JSON을 찾을 수 없습니다")
    return json.loads(clean[start:end + 1])


def parse_scenarios(text: str) -> list[dict]:
    """모델 답에서 쓸 수 있는 시나리오만 고른다. 제목·예상 결과가 없으면 버린다."""
    scenarios = []
    for node in _extract_json(text).get("scenarios") or []:
        if not isinstance(node, dict):
            continue
        kind = str(node.get("type") or "NORMAL").upper()
        title, then = str(node.get("title") or "").strip(), str(node.get("then") or "").strip()
        if not title or not then:
            continue
        scenarios.append({
            "type": kind if kind in SCENARIO_TYPES else "NORMAL",
            "title": title,
            "given": str(node.get("given") or "").strip(),
            "when": str(node.get("when") or "").strip(),
            "then": then,
        })
    return scenarios


def parse_verdict(text: str) -> dict:
    result = _extract_json(text)
    verdict = str(result.get("verdict") or "INCONCLUSIVE").upper()
    evidence = [str(item).strip() for item in result.get("evidence") or [] if str(item).strip()]
    return {
        "verdict": verdict if verdict in VERDICTS else "INCONCLUSIVE",
        "reasoning": str(result.get("reasoning") or "").strip() or "판정 근거가 제공되지 않았습니다.",
        "evidence": evidence,
    }


async def _ask(system_prompt: str, context: str) -> tuple[str, str]:
    """서버 기본 모델로 부른다. 사용자가 고른 AI 품질과 사용량 기록은 common.usage가 처리한다."""
    from main import openai_client, settings

    model = settings.openai_chat_model
    for attempt in range(3):
        try:
            resp = await openai_client.chat.completions.create(
                model=model,
                max_completion_tokens=4000,
                messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": context}],
            )
            return resp.choices[0].message.content or "", resp.model or model
        except (InternalServerError, APITimeoutError, APIConnectionError, RateLimitError) as e:
            logger.warning("QA 시나리오 에이전트 일시 오류 (attempt %d): %s", attempt + 1, e)
            if attempt == 2:
                raise  # 공통 처리기가 AI_BUSY·AI_TIMEOUT 같은 코드와 안내로 바꾼다
            await asyncio.sleep(3 * (attempt + 1))
    return "", model


@router.post("/generate")
async def generate(req: GenerateRequest) -> dict:
    context = (
        f"[PROJECT]\n{req.projectName or '(없음)'}\n\n"
        f"[FEATURE]\nname: {req.featureName or '(없음)'}\n"
        f"description: {_text(req.featureDescription, 2_000)}\n"
        f"requirements: {_text(req.requirements, 3_000)}\n\n"
        f"[API_SPEC]\n{_text(req.apiSpec, 10_000)}\n\n"
        f"[DB_SCHEMA]\n{_text(req.dbSchema, 10_000)}\n"
    )
    raw, model = await _ask(GENERATE_SYSTEM_PROMPT, context)
    try:
        scenarios = parse_scenarios(raw)
    except (ValueError, json.JSONDecodeError) as e:
        raise HTTPException(status_code=502, detail="QA 시나리오 응답을 읽지 못했습니다.") from e
    if not scenarios:
        raise HTTPException(status_code=502, detail="QA 시나리오를 만들지 못했습니다. 다시 시도해 주세요.")
    logger.info("QA 시나리오 생성 완료 — %d개", len(scenarios))
    return {"agent": AGENT_NAME, "model": model, "scenarios": scenarios}


@router.post("/evaluate")
async def evaluate(req: EvaluateRequest) -> dict:
    files = "".join(f"\n### {f.filename or '(이름 없음)'}\n{_text(f.content, 6_000)}" for f in req.codeFiles)
    context = (
        f"[PROJECT]\n{req.projectName or '(없음)'}\n\n"
        f"[SCENARIO]\nfeature: {req.featureName or '(없음)'}\ntitle: {req.scenarioTitle or '(없음)'}\n"
        f"given: {_text(req.given, 1_000)}\nwhen: {_text(req.when, 1_000)}\nthen: {_text(req.then, 1_000)}\n\n"
        f"[API_SPEC]\n{_text(req.apiSpec, 10_000)}\n\n"
        f"[DB_SCHEMA]\n{_text(req.dbSchema, 10_000)}\n\n"
        f"[RELATED_CODE]\n{_text(files, 20_000) if files else '(관련 코드를 찾지 못했습니다 — 명세만 근거로 판단하세요)'}\n"
    )
    raw, model = await _ask(EVALUATE_SYSTEM_PROMPT, context)
    try:
        result = parse_verdict(raw)
    except (ValueError, json.JSONDecodeError) as e:
        raise HTTPException(status_code=502, detail="QA 판정 응답을 읽지 못했습니다.") from e
    logger.info("QA 시나리오 판정 완료 — %s", result["verdict"])
    return {"agent": AGENT_NAME, "model": model, **result}
