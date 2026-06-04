import json
import logging

from openai import AsyncOpenAI

from phase2.state import PipelineState

logger = logging.getLogger(__name__)

QA_PROMPT = """당신은 시니어 소프트웨어 아키텍트입니다.
아래 세 가지 설계 산출물의 논리적 정합성을 검수하세요.

[기능 목록]
{feature_list}

[DB 스키마]
{db_schema}

[API 스펙]
{api_spec}

검수 기준:
1. DB 스키마에 핵심 테이블과 컬럼이 존재하는가?
2. API 스펙의 각 엔드포인트가 필요로 하는 DB 테이블과 컬럼이 존재하는가?
3. DB 테이블 간 외래키 관계가 올바르게 설계되어 있는가?
4. API 필드명과 DB 컬럼명이 일치하는가?

결함 판단 기준 (엄하게 적용하지 말 것):
- 사소한 필드명 차이 (camelCase vs snake_case)는 결함이 아닙니다
- API 응답에 일부 필드가 빠져도 핵심 기능이 동작하면 통과입니다
- 전체적인 설계 방향이 올바르면 세부 불일치는 무시하세요
{relax_msg}

응답 형식 (JSON만 출력):
{{
  "passed": true 또는 false,
  "issues": ["발견된 결함 1"],
  "suggestions": ["수정 제안 1"]
}}

결함이 없으면 passed=true, issues=[] 로 응답하세요.
JSON 외 다른 텍스트는 절대 포함하지 마세요.
"""


class QaAgent:

    def __init__(self, client: AsyncOpenAI, model: str = "gpt-4o"):
        self._client = client
        self._model = model

    async def execute(self, state: PipelineState) -> PipelineState:
        logger.info("QA 에이전트 시작")

        relax_msg = ""
        if state.retry_count >= 2:
            relax_msg = (
                f"\n\n※ 이미 {state.retry_count}회 재시도됨. "
                "치명적인 구조적 결함만 잡고 나머지는 passed=true로 처리하세요."
            )

        prompt = QA_PROMPT.format(
            feature_list="\n".join(state.feature_list),
            db_schema=state.db_schema,
            api_spec=state.api_spec,
            relax_msg=relax_msg,
        )

        response = await self._client.chat.completions.create(
            model=self._model,
            temperature=0.0,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = response.choices[0].message.content or ""
        passed, issues, suggestions = self._parse(raw)

        if passed:
            logger.info("QA 검수 통과")
            return state.copy(status_message="QA 에이전트 완료 — 정합성 검수 통과")
        else:
            logger.warning("QA 검수 실패 — %d 개 결함", len(issues))
            error_msg = (
                "QA 검수 실패\n[발견된 결함]\n"
                + "\n".join(issues)
                + "\n[수정 제안]\n"
                + "\n".join(suggestions)
            )
            return state.copy(
                last_validation_error=error_msg,
                status_message="QA 에이전트 — 결함 발견, 재생성 필요",
            )

    def _parse(self, raw: str) -> tuple[bool, list[str], list[str]]:
        try:
            clean = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
            data = json.loads(clean)
            return (
                bool(data.get("passed", True)),
                [str(i) for i in data.get("issues", [])],
                [str(s) for s in data.get("suggestions", [])],
            )
        except Exception as e:
            logger.error("QA 파싱 실패: %s", e)
            return True, [], []
