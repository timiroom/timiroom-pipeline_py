import asyncio
import json
import logging

from openai import AsyncOpenAI, InternalServerError, APITimeoutError, APIConnectionError

from phase2.json_utils import try_parse_json
from phase2.state import PipelineState

logger = logging.getLogger(__name__)

API_PROMPT = """당신은 시니어 백엔드 개발자입니다.
아래 지시사항을 바탕으로 REST API 스펙을 JSON 형식으로만 설계하세요.

응답 형식:
{{
  "endpoints": [
    {{
      "method": "GET 또는 POST 또는 PUT 또는 DELETE 또는 PATCH",
      "path": "/api/v1/리소스명",
      "description": "기능명: 이 API가 하는 일 한 줄 설명",
      "authRequired": true,
      "requestBody": "field1: 타입 (설명), field2: 타입 (설명) — 없으면 없음",
      "successResponse": "field1: 타입, field2: 타입 — 반환 데이터 설명",
      "errorCodes": "401 — 인증 실패, 404 — 리소스 없음"
    }}
  ],
  "authentication": "JWT Bearer 토큰 방식. 로그인 후 accessToken을 Authorization: Bearer {{token}} 헤더로 전송.",
  "prdIssues": "누락된 API 또는 불명확한 요구사항 (없으면 빈 문자열 \\"\\")"
}}

설계 규칙:
- JSON 외 다른 텍스트는 절대 포함하지 마세요
- requestBody와 successResponse는 반드시 문자열로 작성 (객체 금지)
- RESTful 설계: 명사형 복수형 경로 (예: /api/v1/users, /api/v1/posts)
- authRequired: 로그인 필요 엔드포인트는 true, 공개 API는 false
- 기능 목록의 모든 기능에 대한 엔드포인트를 반드시 포함 (조회·생성·수정·삭제 각각)
- DB 스키마의 테이블명을 API 경로에 반영 (users 테이블 → /api/v1/users)
- 목록 조회에 page, size 쿼리 파라미터 포함

⚠️ 설계 완료 후 반드시 체크:
- [ ] 기능 목록의 모든 기능에 해당하는 API 엔드포인트가 있는가?
- [ ] authRequired가 올바르게 설정되었는가?
- [ ] DB 스키마 테이블명과 API 경로가 일치하는가?

누락된 기능이 있으면 prdIssues 필드에 명시하세요. 없으면 빈 문자열.

지시사항:
{instruction}

기능 목록:
{feature_list}

컨텍스트 (DB 스키마 포함):
{context}
"""


class ApiAgent:

    def __init__(self, client: AsyncOpenAI, model: str = "gpt-4o-mini"):
        self._client = client
        self._model = model

    async def execute(self, state: PipelineState, dump=None) -> PipelineState:
        logger.info("API 에이전트 시작")

        feature_str = "- " + "\n- ".join(state.feature_list)
        prompt = API_PROMPT.format(
            instruction=state.api_instruction,
            feature_list=feature_str,
            context=state.context_prompt or "",
        )

        data = None
        for attempt in range(3):
            try:
                response = await self._client.chat.completions.create(
                    model=self._model,
                    temperature=0.1,
                    max_tokens=6000,
                    frequency_penalty=0.5,
                    messages=[
                        {"role": "system", "content": "JSON만 출력하세요. 설명·인사말·마크다운 코드블록 금지. { 로 시작해서 } 로 끝납니다."},
                        {"role": "user", "content": prompt},
                    ],
                    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                )
            except (InternalServerError, APITimeoutError, APIConnectionError) as e:
                logger.warning("API API 일시 오류 (attempt %d): %s — 재시도", attempt + 1, e)
                if attempt < 2:
                    await asyncio.sleep(5 * (attempt + 1))
                    continue
                break
            raw = response.choices[0].message.content or ""
            if dump:
                dump.log_raw("API", attempt + 1, raw)
            data = try_parse_json(raw)
            if data and isinstance(data, dict):
                break
            logger.warning("API 파싱 실패 (attempt %d) — raw:\n%s", attempt + 1, raw)

        prd_issues = ""
        if data and isinstance(data, dict):
            prd_issues = data.pop("prdIssues", "") or ""
            clean = json.dumps(data, ensure_ascii=False)
            if prd_issues:
                logger.warning("API → PRD 피드백: %s", prd_issues)
            else:
                logger.info("API 검증 통과")
        else:
            logger.error("API 최종 파싱 실패 — 최소 스펙으로 대체")
            clean = json.dumps({"endpoints": [], "authentication": "JWT Bearer 토큰"}, ensure_ascii=False)

        logger.info("API 에이전트 완료")
        return state.copy(
            api_spec=clean,
            prd_feedback_from_api=prd_issues,
            status_message="API 에이전트 완료 — API 스펙 생성",
        )
