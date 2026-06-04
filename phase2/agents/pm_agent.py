import json
import logging

from openai import AsyncOpenAI

from common.pm_skills import PmSkillsLoader
from phase2.state import PipelineState

logger = logging.getLogger(__name__)

PM_PROMPT = """당신은 시니어 소프트웨어 아키텍트이자 PM입니다.
아래 요구사항을 분석하여 JSON 형식으로만 응답하세요.
{skills_section}
응답 형식:
{{
  "featureList": ["기능1", "기능2", "기능3"],
  "dbaInstruction": "DBA에게 전달할 DB 설계 지시사항",
  "apiInstruction": "API 개발자에게 전달할 API 설계 지시사항"
}}

규칙:
- featureList는 서비스를 구성하는 모든 핵심·부가 기능을 빠짐없이 열거하세요 (개수 제한 없음, 최대한 상세하게)
- dbaInstruction은 필요한 테이블, 관계, 제약조건을 명시하세요
- apiInstruction은 필요한 엔드포인트와 인증 방식을 명시하세요
- JSON 외 다른 텍스트는 절대 포함하지 마세요

요구사항:
{context}
"""


class PmAgent:

    def __init__(self, client: AsyncOpenAI, skills_loader: PmSkillsLoader, model: str = "gpt-4o"):
        self._client = client
        self._skills = skills_loader
        self._model = model

    async def execute(self, state: PipelineState) -> PipelineState:
        logger.info("PM 에이전트 시작")

        if self._skills.has_skills():
            skills_section = await self._skills.find_relevant_skills(state.context_prompt, 5)
            logger.info("PM 스킬 주입 완료")
        else:
            skills_section = ""
            logger.warning("PM 스킬 미적용")

        prompt = PM_PROMPT.format(
            skills_section=skills_section,
            context=state.context_prompt,
        )

        response = await self._client.chat.completions.create(
            model=self._model,
            temperature=0.1,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = response.choices[0].message.content or ""

        feature_list, dba_instruction, api_instruction = self._parse(raw)
        logger.info("PM 에이전트 완료 — %d 기능 도출", len(feature_list))

        return state.copy(
            feature_list=feature_list,
            dba_instruction=dba_instruction,
            api_instruction=api_instruction,
            status_message="PM 에이전트 완료 — 기능 목록 도출",
        )

    def _parse(self, raw: str) -> tuple[list[str], str, str]:
        try:
            clean = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
            data = json.loads(clean)
            return (
                data.get("featureList", []),
                data.get("dbaInstruction", ""),
                data.get("apiInstruction", ""),
            )
        except Exception as e:
            logger.error("PM 에이전트 파싱 실패: %s", e)
            msg = f"파싱 실패: {e}"
            return (["파싱 실패"], msg, msg)
