import json
import logging

from openai import AsyncOpenAI

from phase2.state import PipelineState

logger = logging.getLogger(__name__)

DBA_PROMPT = """당신은 시니어 DBA입니다.
아래 지시사항을 바탕으로 DB 스키마를 JSON 형식으로만 설계하세요.

응답 형식:
{{
  "tables": [
    {{
      "name": "테이블명",
      "columns": [
        {{"name": "컬럼명", "type": "타입", "constraints": "제약조건"}}
      ],
      "indexes": ["인덱스 설명"]
    }}
  ],
  "relationships": ["관계 설명"],
  "prdIssues": "PRD에서 누락되거나 불명확한 기능 (없으면 빈 문자열 \\"\\")"
}}

규칙:
- JSON 외 다른 텍스트는 절대 포함하지 마세요
- 각 테이블에 반드시 id(PK), created_at, updated_at 컬럼을 포함하세요
- 외래키 관계를 명확히 표현하세요
- 기능 목록의 모든 기능이 테이블에 반영되어야 합니다
- relationships 배열에 반드시 카디널리티를 명시하세요
- 누락된 기능이 있으면 prdIssues 필드에 명시하세요, 없으면 빈 문자열

지시사항:
{instruction}

기능 목록:
{feature_list}

컨텍스트:
{context}
"""


class DbaAgent:

    def __init__(self, client: AsyncOpenAI, model: str = "gpt-4o-mini"):
        self._client = client
        self._model = model

    async def execute(self, state: PipelineState) -> PipelineState:
        logger.info("DBA 에이전트 시작")

        feature_str = "- " + "\n- ".join(state.feature_list)
        prompt = DBA_PROMPT.format(
            instruction=state.dba_instruction,
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
                logger.warning("DBA → PRD 피드백: %s", prd_issues)
            else:
                logger.info("DBA 검증 통과")
        except Exception as e:
            logger.warning("DBA 응답 파싱 실패: %s", e)

        logger.info("DBA 에이전트 완료")
        return state.copy(
            db_schema=clean,
            prd_feedback_from_dba=prd_issues,
            status_message="DBA 에이전트 완료 — DB 스키마 생성",
        )
