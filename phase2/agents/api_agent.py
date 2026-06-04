import json
import logging

from openai import AsyncOpenAI

from phase2.state import PipelineState

logger = logging.getLogger(__name__)

API_PROMPT = """당신은 시니어 백엔드 개발자입니다.
아래 지시사항을 바탕으로 REST API 스펙을 JSON 형식으로만 설계하세요.

응답 형식:
{{
  "endpoints": [
    {{
      "method": "HTTP 메서드",
      "path": "/api/경로",
      "description": "엔드포인트 설명",
      "request": {{
        "headers": {{"헤더명": "타입"}},
        "body": {{"필드명": "타입"}}
      }},
      "response": {{
        "success": {{"필드명": "타입"}},
        "error": "에러 설명"
      }}
    }}
  ],
  "authentication": "인증 방식 설명",
  "prdIssues": "누락된 API 또는 불명확한 요구사항 (없으면 빈 문자열 \\"\\")"
}}

규칙:
- JSON 외 다른 텍스트는 절대 포함하지 마세요
- RESTful 설계 원칙을 따르세요
- 인증이 필요한 엔드포인트는 Authorization 헤더를 명시하세요
- 기능 목록의 모든 기능에 대한 엔드포인트를 반드시 포함하세요
- 누락된 기능이 있으면 prdIssues 필드에 명시하세요, 없으면 빈 문자열

지시사항:
{instruction}

기능 목록:
{feature_list}

컨텍스트:
{context}
"""


class ApiAgent:

    def __init__(self, client: AsyncOpenAI, model: str = "gpt-4o-mini"):
        self._client = client
        self._model = model

    async def execute(self, state: PipelineState) -> PipelineState:
        logger.info("API 에이전트 시작")

        feature_str = "- " + "\n- ".join(state.feature_list)
        prompt = API_PROMPT.format(
            instruction=state.api_instruction,
            feature_list=feature_str,
            context=state.context_prompt or "",
        )

        response = await self._client.chat.completions.create(
            model=self._model,
            temperature=0.1,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = response.choices[0].message.content or ""
        clean = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()

        prd_issues = ""
        try:
            data = json.loads(clean)
            prd_issues = data.pop("prdIssues", "") or ""
            clean = json.dumps(data, ensure_ascii=False)
            if prd_issues:
                logger.warning("API → PRD 피드백: %s", prd_issues)
            else:
                logger.info("API 검증 통과")
        except Exception as e:
            logger.warning("API 응답 파싱 실패: %s", e)

        logger.info("API 에이전트 완료")
        return state.copy(
            api_spec=clean,
            prd_feedback_from_api=prd_issues,
            status_message="API 에이전트 완료 — API 스펙 생성",
        )
