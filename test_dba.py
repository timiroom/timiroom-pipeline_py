"""DBA 에이전트 단독 테스트"""
import asyncio
import json
from openai import AsyncOpenAI
from config.settings import settings
from phase2.agents.dba_agent import DbaAgent
from phase2.state import PipelineState

FEATURE_LIST = [
    "재료 등록 및 관리",
    "레시피 자동 추천",
    "회원가입/로그인",
    "유통기한 알림",
    "장보기 목록 자동 생성",
]

DBA_INSTRUCTION = (
    "냉장고 재료 기반 레시피 추천 앱의 DB를 설계하세요. "
    "React Native + FastAPI + PostgreSQL 스택. "
    "사용자별 재료 관리, 레시피-재료 N:M 관계, 유통기한 알림 저장이 핵심입니다."
)


async def main():
    client = AsyncOpenAI(
        api_key=settings.exaone_api_key,
        base_url="https://api.friendli.ai/dedicated/v1",
    )
    agent = DbaAgent(client, settings.exaone_endpoint_id)

    state = PipelineState(
        feature_list=FEATURE_LIST,
        dba_instruction=DBA_INSTRUCTION,
        context_prompt="냉장고 레시피 앱 - APP 플랫폼, 20대 자취생 타겟",
    )

    result = await agent.execute(state)

    print("=== DB Schema ===")
    try:
        parsed = json.loads(result.db_schema)
        tables = parsed.get("tables", {})
        print(f"테이블 수: {len(tables)}")
        for tname, tinfo in tables.items():
            cols = tinfo.get("columns", [])
            print(f"\n  [{tname}] - {tinfo.get('description', '')}")
            for c in cols:
                print(f"    {c}")
        print(f"\n관계: {parsed.get('relationships', [])}")
        if parsed.get("prdIssues"):
            print(f"PRD 이슈: {parsed['prdIssues']}")
        print("\n[PASS] 파싱 성공")
    except Exception as e:
        print(f"[FAIL] 파싱 실패: {e}")
        print(result.db_schema[:500])

    if result.prd_feedback_from_dba:
        print(f"\nPRD 피드백: {result.prd_feedback_from_dba}")


if __name__ == "__main__":
    asyncio.run(main())
