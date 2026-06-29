import asyncio
import json
import logging

from openai import AsyncOpenAI, InternalServerError, APITimeoutError, APIConnectionError

from phase2.json_utils import try_parse_json
from phase2.state import PipelineState

logger = logging.getLogger(__name__)

DBA_PROMPT = """당신은 시니어 DBA입니다.
아래 지시사항과 기능 목록을 바탕으로 DB 스키마를 JSON으로 설계하세요.

## 출력 형식 — 반드시 아래 구조를 그대로 사용하세요

{{
  "tables": {{
    "users": {{
      "description": "사용자 계정 정보",
      "columns": [
        "id BIGINT PRIMARY_KEY AUTO_INCREMENT",
        "email VARCHAR(255) NOT_NULL UNIQUE",
        "password_hash VARCHAR(255) NOT_NULL",
        "created_at DATETIME NOT_NULL",
        "updated_at DATETIME NOT_NULL"
      ],
      "indexes": ["INDEX idx_users_email ON users(email)"]
    }},
    "posts": {{
      "description": "게시물 정보",
      "columns": [
        "id BIGINT PRIMARY_KEY AUTO_INCREMENT",
        "user_id BIGINT NOT_NULL FOREIGN_KEY",
        "title VARCHAR(255) NOT_NULL",
        "body TEXT NOT_NULL",
        "created_at DATETIME NOT_NULL",
        "updated_at DATETIME NOT_NULL"
      ],
      "indexes": ["INDEX idx_posts_user_id ON posts(user_id)"]
    }}
  }},
  "relationships": [
    "users (1:N) posts"
  ],
  "prdIssues": ""
}}

## 컬럼 작성 규칙
- 각 컬럼은 문자열 한 줄: "컬럼명 타입 제약조건들"
- 허용 타입: BIGINT, VARCHAR(255), VARCHAR(100), VARCHAR(50), TEXT, BOOLEAN, DATETIME, DECIMAL(10,2)
- 제약조건 키워드(공백 구분): PRIMARY_KEY, AUTO_INCREMENT, NOT_NULL, NULL, UNIQUE, FOREIGN_KEY, DEFAULT_FALSE, DEFAULT_TRUE

## 스키마 규칙
- tables는 배열이 아닌 오브젝트: 키=테이블명, 값=description/columns/indexes
- 테이블 수는 최대 6개
- 모든 테이블에 "id BIGINT PRIMARY_KEY AUTO_INCREMENT", "created_at DATETIME NOT_NULL", "updated_at DATETIME NOT_NULL" 필수
- 외래키 컬럼명은 참조테이블_id 형식 (예: user_id, recipe_id)
- 인증/회원 기능이 있으면 users 테이블 필수
- relationships: "테이블A (1:N) 테이블B", "테이블A (N:M) 테이블B", "테이블A (1:1) 테이블B" 형식
- prdIssues: PRD에서 누락된 기능이 있으면 한 줄, 없으면 빈 문자열 ""

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

    async def execute(self, state: PipelineState, dump=None) -> PipelineState:
        logger.info("DBA 에이전트 시작")

        feature_str = "- " + "\n- ".join(state.feature_list)
        prompt = DBA_PROMPT.format(
            instruction=state.dba_instruction,
            feature_list=feature_str,
            context=state.context_prompt or "",
        )

        data = None
        for attempt in range(3):
            try:
                response = await self._client.chat.completions.create(
                    model=self._model,
                    temperature=0.1,
                    max_tokens=4000,
                    frequency_penalty=0.3,
                    messages=[
                        {"role": "system", "content": "JSON만 출력하세요. 설명·인사말·마크다운 코드블록 금지. { 로 시작해서 } 로 끝납니다."},
                        {"role": "user", "content": prompt},
                    ],
                    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                )
            except (InternalServerError, APITimeoutError, APIConnectionError) as e:
                logger.warning("DBA API 일시 오류 (attempt %d): %s — 재시도", attempt + 1, e)
                if attempt < 2:
                    await asyncio.sleep(5 * (attempt + 1))
                    continue
                break
            raw = response.choices[0].message.content or ""
            if dump:
                dump.log_raw("DBA", attempt + 1, raw)
            data = try_parse_json(raw)
            if data and isinstance(data, dict):
                break
            logger.warning("DBA 파싱 실패 (attempt %d) — raw:\n%s", attempt + 1, raw)

        prd_issues = ""
        if data and isinstance(data, dict):
            prd_issues = data.pop("prdIssues", "") or ""
            clean = json.dumps(data, ensure_ascii=False)
            if prd_issues:
                logger.warning("DBA → PRD 피드백: %s", prd_issues)
            else:
                logger.info("DBA 검증 통과")
        else:
            logger.error("DBA 최종 파싱 실패 — 최소 스키마로 대체")
            clean = json.dumps({"tables": {}, "relationships": []}, ensure_ascii=False)

        logger.info("DBA 에이전트 완료")
        return state.copy(
            db_schema=clean,
            prd_feedback_from_dba=prd_issues,
            status_message="DBA 에이전트 완료 — DB 스키마 생성",
        )
