import asyncio
import json
import logging
import re
from typing import Annotated, TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.types import Send
from openai import AsyncOpenAI, InternalServerError, APITimeoutError, APIConnectionError

from phase2.agent_contract import contract_prompt, normalize_target_arrow
from phase2.json_utils import try_parse_json, has_suspicious_script
from phase2.quality_rules import (
    contamination_reasons, feature_semantic_issues, has_placeholder, kpi_basis_issues,
    near_duplicate, relevance_score,
    retry_prompt, self_check_passed, source_evidence_issues,
)
from phase2.state import PipelineState
from phase2.llm_concurrency import llm_slot

logger = logging.getLogger(__name__)

# 섹션별 최소 항목 수 — SECTION_PROMPTS가 스스로 요구하는 기준과 동일.
# _generate_section에서 이 기준 미달 시 파싱 성공이라도 재생성을 강제한다.
# coreFeatures 생성 배치 크기 — 한 worker가 담당할 기능 수.
# 기능 전체를 한 호출에 넣으면 출력이 잘려 3회 재생성이 모두 실패한다(실측 21개).
_CORE_FEATURE_BATCH_SIZE = 1

_SECTION_MIN_COUNTS: dict[str, dict[str, int]] = {
    "goalsKpi": {"kpi": 7},
    "userPersonas": {"userPersonas": 3},
    "releaseSchedule": {"releaseSchedule": 6},
}

_KPI_DIMENSIONS = (
    "획득: 서비스에 처음 유입된 사용자 또는 조직",
    "활성: 사용자가 처음으로 핵심 가치를 경험한 행동",
    "전환: 이 서비스의 핵심 업무를 성공적으로 완료한 비율",
    "단기 리텐션: 7일 재방문",
    "장기 리텐션: 30일 재방문",
    "품질: 요구사항의 핵심 처리 성공률 또는 정확도",
    "비즈니스: 사용자 요구에 맞는 시간·비용 절감 또는 수익 성과",
)

_RELEASE_STAGES = (
    ("1개월차", "요구사항 확정 및 설계"),
    ("2~3개월차", "핵심 기능 개발"),
    ("4개월차", "통합 및 QA"),
    ("5개월차", "클로즈드 베타"),
    ("6개월차", "정식 출시"),
    ("출시 후 1~3개월", "안정화 및 고도화"),
)

# EXAONE 모델 카드 권장 샘플링 파라미터
# https://huggingface.co/LGAI-EXAONE/K-EXAONE-236B-A23B
_TEMPERATURE = 1.0
_TOP_P = 0.95
_PRESENCE_PENALTY = 0.0

WORKER_SYSTEM = """JSON만 출력하세요. 설명·인사말·마크다운 코드블록 금지. { 로 시작해서 } 로 끝납니다.

당신은 10년 경력의 시니어 PM이자 소프트웨어 아키텍트입니다.
제공된 시장 데이터를 최대한 활용하여 투자자와 개발팀이 신뢰할 수 있는 최고 수준의 PRD 섹션을 작성하세요.

══ 작성 원칙 ══
- 모든 텍스트 필드는 최소 기준 글자수를 반드시 충족해야 합니다.
- 완성된 문장으로 작성하세요 (키워드 나열 금지).
- 수치에는 출처(기관명 + 연도)를 괄호로 명시하세요.
- 각 항목이 서로 다른 관점을 다루도록 중복 내용 금지.
- "~을 제공합니다", "~을 지원합니다" 같은 단순 서술 금지 —
  반드시 "누가 / 무엇을 / 어떻게 / 왜" 구조로 구체적으로 작성하세요.

══ 절대 금지 ══
- 출처 없는 수치 사용
- "내부 목표", "내부 설문조사" 표현
- "약", "대략" 등 모호한 표현 (실제 추정치 제외)
- A사, B사 등 가상 서비스명 — 실제 한국 서비스명 사용
- $name, $serviceName, {name}, {value} 등 placeholder를 값으로 사용하는 것
- 같은 단어나 공백을 연속으로 반복 생성하는 것
- thinking 태그(</think>, <think>) 를 JSON 값 안에 포함하는 것

반드시 JSON만 출력하고 다른 텍스트는 절대 포함하지 마세요."""

ITEM_WORKER_SYSTEM = """당신은 PRD 단일 항목 작성 worker입니다.
사용자가 지정한 영문 라벨과 평문 값만 출력하세요. JSON, 배열, 마크다운, 설명, 인사말은 금지합니다.
각 라벨은 한 번만 사용하고 모든 필드를 빠짐없이 작성하세요.
담당 항목 이외의 기능을 설명하지 말고 마지막 줄에 SELF_CHECK: PASS를 출력하세요."""

MANAGER_SYSTEM = """당신은 10년 경력의 시니어 PM 겸 PRD manager입니다.
sub-agent 7명이 작성한 PRD 섹션 초안을 검토하여, 기준 미달이거나 내용이 이상한 섹션만 골라
직접 교정본을 작성하세요. 문제 없는 섹션은 절대 건드리지 마세요.

JSON만 출력하세요. 설명·인사말·마크다운 코드블록 금지. { 로 시작해서 } 로 끝납니다."""

# 섹션별 프롬프트 — 기존 PART_1~3 프롬프트의 글자수/개수 기준을 그대로 유지
SECTION_PROMPTS: dict[str, str] = {
    "projectOverview": """수집된 시장 데이터:
=== 시장 데이터 ===
{market_data}
=================
{rollback_section}

아래 JSON 형식으로 PRD의 서비스 개요를 작성하세요.

══ 필드별 최소 기준 (반드시 충족) ══
▸ projectOverview: 서비스 한 줄 개요 — 누구를 위해 무엇을 어떻게 제공하는가 (50자 이상).
▸ background     : 200자 이상. 완성된 2~3문단. 시장 현황 → 문제점 → 기회 순서로 전개.
                   수치를 인용할 때는 실제로 존재하고 검증 가능한 **국내** 출처만 사용하고
                   수치마다 (출처: 기관명, YYYY년) 표기. 확실하지 않으면 '(추정치)'로 명시.
                   ⛔ 회사명·브랜드명·기관명을 여러 개 나열하지 마세요. 반드시 완성된 문장으로만
                   서술하고, 같은 유형의 단어(회사명 등)를 콤마/공백으로 길게 열거하지 마세요.

검증 체크리스트 (생성 후 스스로 확인):
[ ] projectOverview가 50자 이상인가?
[ ] background가 200자 이상이며 완성된 문장으로만 쓰였는가? (단어 나열 없음)
[ ] 인용 수치에 검증 가능한 국내 출처가 명시되어 있는가?

JSON:
{{
  "projectOverview": "서비스 한 줄 개요",
  "background": "200자 이상 서술형 문단"
}}

사용자 요구사항: {user_query}
""",
    "goalsKpi": """수집된 시장 데이터:
=== 시장 데이터 ===
{market_data}
=================
{rollback_section}

아래 JSON 형식으로 PRD의 목표·KPI 부분을 작성하세요.

══ 필드별 최소 기준 (반드시 충족) ══
▸ goals: ⭐3개 이상⭐. 각 항목은 "~을 통해 ~을 달성하여 ~에 기여한다" 형식, 60자 이상.
▸ kpi  : ⭐⭐반드시 7개 이상⭐⭐. 획득·활성·전환·리텐션·품질·수익 등 서로 다른 관점의 지표 7개.
         target은 "현재값 → 목표값" 형식. basis는 업계 평균 수치 출처를 "기관명, YYYY년"으로 명시.

⚠️ 출력하기 전에 kpi 배열의 원소를 직접 1,2,3...로 세어 7개 이상인지 확인하고, 부족하면 채운 뒤 출력하세요.
⚠️ 출처(basis)는 실제로 존재하고 검증 가능한 **국내** 기관·리포트를 사용하세요
   (예: 통계청, 한국인터넷진흥원(KISA), 한국소비자원, 오픈서베이, 모바일인덱스, 와이즈앱리테일).
   아래 예시의 출처·수치를 그대로 복사하지 말고 이 서비스와 국내 시장에 맞게 교체하세요.
   확실하지 않은 수치는 지어내지 말고 basis에 '(추정치)'라고 명시하세요.

JSON (아래 goals 3개·kpi 7개는 개수/형식 예시입니다 — 이 서비스에 맞는 실제 지표로 goals 3개 이상, kpi 7개 이상을 반드시 채우세요):
{{
  "goals": [
    "핵심 기능 자동화를 통해 사용자 입력 시간을 단축하여 일일 활성 사용자 증가에 기여한다",
    "적시 알림을 통해 사용자의 반복 이탈을 줄여 재방문율 향상에 기여한다",
    "개인화 추천을 통해 서비스 체류 시간을 늘려 장기 리텐션 향상에 기여한다"
  ],
  "kpi": [
    {{"metric":"월간 활성 사용자(MAU)","target":"0명 → 10,000명","basis":"국내 모바일 앱 이용 현황 (모바일인덱스, 2023년, 추정치)","measurementMethod":"월간 고유 로그인 사용자 집계","frequency":"월간"}},
    {{"metric":"신규 가입자 수","target":"0명 → 3,000명","basis":"국내 동종 서비스 초기 획득 벤치마크 (와이즈앱리테일, 2023년, 추정치)","measurementMethod":"신규 계정 생성 수","frequency":"주간"}},
    {{"metric":"온보딩 전환율","target":"0% → 60%","basis":"국내 앱 온보딩 완료율 (오픈서베이 모바일 리포트, 2023년, 추정치)","measurementMethod":"가입 대비 핵심 행동 1회 완료 비율","frequency":"주간"}},
    {{"metric":"7일 리텐션","target":"0% → 35%","basis":"국내 유틸리티 앱 평균 리텐션 (모바일인덱스, 2023년, 추정치)","measurementMethod":"가입 후 7일차 재방문 비율","frequency":"주간"}},
    {{"metric":"30일 리텐션","target":"0% → 20%","basis":"국내 유틸리티 앱 평균 리텐션 (모바일인덱스, 2023년, 추정치)","measurementMethod":"가입 후 30일차 재방문 비율","frequency":"월간"}},
    {{"metric":"핵심 기능 사용률","target":"0% → 50%","basis":"앱 기능 채택률 (오픈서베이, 2023년, 추정치)","measurementMethod":"활성 사용자 중 핵심 기능 사용 비율","frequency":"주간"}},
    {{"metric":"앱 스토어 평점","target":"0.0 → 4.5","basis":"국내 상위 유틸리티 앱 평균 평점 (구글플레이/앱스토어, 2023년)","measurementMethod":"스토어 평균 별점","frequency":"월간"}}
  ]
}}

검증 체크리스트 (생성 후 스스로 확인):
[ ] kpi 배열 원소를 직접 세었을 때 7개 이상인가?
[ ] goals가 3개 이상인가?
[ ] KPI basis에 "내부 목표" 표현이 없는가?

사용자 요구사항: {user_query}
""",
    "coreFeatures": """수집된 시장 데이터:
=== 시장 데이터 ===
{market_data}
=================
{rollback_section}

아래 JSON 형식으로 PRD의 핵심기능 부분을 작성하세요.
⚠️⭐ coreFeatures 배열은 반드시 정확히 {feature_count}개 — 아래 '기능 목록'의 각 기능마다 1:1로 대응하는 항목을 하나씩 만드세요. 여러 기능을 하나로 묶거나 빠뜨리지 마세요. ⭐

══ 필드별 최소 기준 (반드시 충족) ══
▸ coreFeatures             : 정확히 {feature_count}개. 각 항목 name은 기능 목록의 각 기능명을 그대로(또는 거의 동일하게) 사용.
▸ coreFeatures.description : 100자 이상. "사용자가 ~상황에서 ~을 하면 시스템이 ~을 수행하여 ~효과를 낸다" 구조.
▸ coreFeatures.requirements: 각 항목 60자 이상. "~할 때 → ~처리 → ~결과" 구조.

⚠️ 출력하기 전에 coreFeatures 원소 수가 기능 목록 개수({feature_count}개)와 같은지 직접 세어 확인하세요.

JSON (아래는 형식 예시 2개일 뿐 — 기능 목록의 모든 기능에 대해 정확히 {feature_count}개를 채우세요):
{{
  "coreFeatures": [
    {{"name":"기능 목록의 1번 기능명","description":"사용자가 특정 상황에서 이 기능을 사용하면 시스템이 해당 처리를 수행하여 구체적 효과를 내는 방식을 100자 이상으로 서술","priority":"P0","requirements":["특정 조건일 때 어떤 처리를 거쳐 어떤 결과를 내는지 60자 이상 서술"]}},
    {{"name":"기능 목록의 2번 기능명","description":"두 번째 기능의 사용 상황·시스템 동작·효과를 100자 이상으로 서술","priority":"P1","requirements":["처리 조건 → 처리 → 결과를 60자 이상 서술"]}}
  ]
}}

검증 체크리스트 (생성 후 스스로 확인):
[ ] coreFeatures 항목 수가 기능 목록 개수({feature_count}개)와 정확히 일치하고 묶음 처리가 없는가?
[ ] coreFeatures.requirements 각 항목이 60자 이상인가?

사용자 요구사항: {user_query}
기능 목록 (아래 {feature_count}개 기능 각각을 coreFeatures 1개 항목으로 작성):
{feature_str}
""",
    "userPersonas": """수집된 시장 데이터:
=== 시장 데이터 ===
{market_data}
=================
{rollback_section}

아래 JSON 형식으로 PRD의 사용자 페르소나 부분을 작성하세요.

══ 필드별 최소 기준 (반드시 충족) ══
▸ userPersonas: ⭐⭐반드시 정확히 3개⭐⭐ (2개 이하 절대 금지). 연령대/직업/기술수준이 서로 다른 3명.
                goal/painPoint/usagePattern 각 2문장 이상.

⚠️ 출력하기 전에 userPersonas 배열이 정확히 3개인지 세어 확인하세요.

JSON (반드시 서로 다른 3명 — 아래 형식으로 3개 항목을 채우세요):
{{
  "userPersonas": [
    {{"name":"이름1","age":"20대 중반","job":"직업1","techLevel":"높음","goal":"사용 목표 2문장 이상","painPoint":"주요 불편함 2문장 이상","usagePattern":"사용 패턴 2문장 이상"}},
    {{"name":"이름2","age":"30대 초반","job":"직업2","techLevel":"중간","goal":"사용 목표 2문장 이상","painPoint":"주요 불편함 2문장 이상","usagePattern":"사용 패턴 2문장 이상"}},
    {{"name":"이름3","age":"40대","job":"직업3","techLevel":"낮음","goal":"사용 목표 2문장 이상","painPoint":"주요 불편함 2문장 이상","usagePattern":"사용 패턴 2문장 이상"}}
  ]
}}

검증 체크리스트 (생성 후 스스로 확인):
[ ] userPersonas가 정확히 3개인가? (2개 이하면 실패)

사용자 요구사항: {user_query}
""",
    "mvpScope": """수집된 시장 데이터:
=== 시장 데이터 ===
{market_data}
=================
{rollback_section}

아래 JSON 형식으로 PRD의 MVP 범위 부분을 작성하세요.

══ 필드별 최소 기준 (반드시 충족) ══
▸ mvpScope.rationale: 100자 이상. 포함/제외 기준 논리와 비즈니스 근거 서술.

JSON:
{{
  "mvpScope": {{"included":["기능명"],"excluded":["기능명"],"rationale":"MVP 범위 선정 이유"}}
}}

사용자 요구사항: {user_query}
기능 목록:
{feature_str}
""",
    "techStack": """수집된 시장 데이터:
=== 시장 데이터 ===
{market_data}
=================
{rollback_section}

아래 JSON 형식으로 PRD의 기술스택 부분을 작성하세요.

══ 필드별 최소 기준 (반드시 충족) ══
▸ techStack: 각 레이어 선택 이유 2문장 이상 (기술 특성 + 이 서비스에 적합한 이유).

JSON:
{{
  "techStack": {{"backend":"기술명과 선택 이유","frontend":"기술명과 선택 이유","database":"기술명과 선택 이유","cache":"기술명과 선택 이유","messageQueue":"기술명과 선택 이유","cdn":"기술명과 선택 이유","monitoring":"기술명과 선택 이유","auth":"인증 방식"}}
}}

사용자 요구사항: {user_query}
기능 목록: {feature_str}
""",
    "releaseSchedule": """수집된 시장 데이터:
=== 시장 데이터 ===
{market_data}
=================
{rollback_section}

아래 JSON 형식으로 PRD의 릴리스일정 부분을 작성하세요.

══ 필드별 최소 기준 (반드시 충족) ══
▸ releaseSchedule: ⭐⭐반드시 6개 이상⭐⭐. 아래 6단계 흐름을 기본 골격으로 삼되 서비스에 맞게 조정하세요:
    ① 요구사항 확정·설계 ② 핵심기능 개발 ③ 통합·QA ④ 클로즈드 베타 ⑤ 정식 출시 ⑥ 안정화·고도화
  각 항목 description은 완료 기준(Done Criteria) 포함 100자 이상, deliverables 3개 이상.

⚠️ 출력하기 전에 releaseSchedule 원소가 6개 이상인지 직접 세어 확인하세요.

JSON (아래는 형식 예시 2개일 뿐 — 위 6단계에 따라 반드시 6개 이상을 채우세요):
{{
  "releaseSchedule": [
    {{"date":"1개월차","milestone":"요구사항 확정 및 설계","description":"핵심 요구사항과 데이터 모델을 확정하고 완료 기준을 명시한 100자 이상 서술","deliverables":["요구사항 명세서","ERD","API 설계서"]}},
    {{"date":"2~3개월차","milestone":"핵심기능 개발","description":"MVP 핵심 기능 구현 완료 기준을 포함한 100자 이상 서술","deliverables":["핵심 기능 구현체","단위 테스트","개발 문서"]}}
  ]
}}

검증 체크리스트 (생성 후 스스로 확인):
[ ] releaseSchedule이 6개 이상인가? (직접 세어 확인)

사용자 요구사항: {user_query}
기능 목록: {feature_str}
""",
}

# 매니저 리뷰용 섹션별 최소 기준 요약 (프롬프트에 그대로 첨부)
_SECTION_CRITERIA = """- projectOverview: 50자 이상
- background: 200자 이상, 수치마다 출처 표기
- goals: 각 항목 60자 이상
- kpi: 7개 이상, target은 "현재값 → 목표값" 형식
- coreFeatures: featureList와 정확히 {feature_count}개 1:1 대응, description 100자 이상
- userPersonas: 정확히 3개
- mvpScope: rationale 100자 이상
- techStack: 각 레이어 선택 이유 2문장 이상
- releaseSchedule: 6개 이상"""

MANAGER_REVIEW_PROMPT = """아래는 sub-agent 7명이 작성한 PRD 섹션 초안입니다.

=== 섹션 초안 ===
{sections_json}
=================

=== 섹션별 최소 기준 ===
{criteria}
=================

사용자 요구사항: {user_query}
기능 목록 ({feature_count}개): {feature_str}

위 기준을 충족하지 못했거나 내용이 이상한(placeholder 남음, 기준 미달, 사실관계 오류 등) 섹션이 있으면
해당 섹션만 새로 작성해 patches에 담으세요. 문제 없는 섹션은 patches에 포함하지 마세요.

JSON:
{{
  "patches": {{
    "섹션키(예: techStack)": "교정된 값 (원본과 동일한 타입/구조)"
  }}
}}

문제되는 섹션이 하나도 없으면 반드시 {{"patches": {{}}}}로 응답하세요.
"""


# EXAONE이 스키마 영문 키 대신 한글/변형/손상 키로 출력하는 것을 표준 키로 되돌리는 백스톱.
# api_agent._normalize_endpoint_keys와 같은 목적 — PRD엔 그동안 이게 없어서 releaseSchedule이
# '출시일정', goals가 'goal', deliverables가 'deliver합니다' 등으로 새어나가 소비자(프론트/Kafka)
# 에게 해당 섹션이 통째로 빈 값으로 보이는 문제가 있었다.
_PRD_TOP_KEY_ALIASES = {
    "출시일정": "releaseSchedule", "릴리스일정": "releaseSchedule", "릴리즈일정": "releaseSchedule",
    "goal": "goals", "목표": "goals",
    "kpis": "kpi", "KPIs": "kpi",
    "coreFeature": "coreFeatures", "핵심기능": "coreFeatures",
    "userPersona": "userPersonas", "personas": "userPersonas", "사용자페르소나": "userPersonas",
    "프로젝트개요": "projectOverview", "서비스개요": "projectOverview",
    "배경": "background", "기술스택": "techStack",
    "mvp범위": "mvpScope", "mvpscope": "mvpScope",
}
_RELEASE_ITEM_ALIASES = {
    "날짜": "date", "기간": "date",
    "마일스톤명": "milestone", "마일스톤": "milestone",
    "산출물": "deliverables", "설명": "description",
}
_KPI_ITEM_ALIASES = {
    "지표명": "metric", "지표": "metric", "목표값": "target",
    "출처": "basis", "근거": "basis", "측정방법": "measurementMethod",
    "측정주기": "frequency", "주기": "frequency",
}
_CORE_ITEM_ALIASES = {
    "기능명": "name", "이름": "name", "설명": "description",
    "우선순위": "priority", "요구사항": "requirements",
}


def _remap_keys(d: dict, alias: dict) -> dict:
    """dict의 키를 표준 키로 remap. 표준 키가 이미 있으면 덮어쓰지 않아(우선) 원본을 보존한다.
    'deliver'로 시작하는 손상 키(deliver합니다 등)는 deliverables로 복구한다."""
    if not isinstance(d, dict):
        return d
    out: dict = {}
    # 1차: 이미 표준인 키 우선 확보
    for k, v in d.items():
        if k not in alias and not (isinstance(k, str) and k.lower().startswith("deliver")):
            out[k] = v
    # 2차: 별칭/손상 키 매핑 (표준 키가 아직 없을 때만)
    for k, v in d.items():
        nk = alias.get(k, k)
        if isinstance(k, str) and k.lower().startswith("deliver"):
            nk = "deliverables"
        if nk not in out:
            out[nk] = v
    return out


def _normalize_prd_keys(doc: dict) -> dict:
    """PRD 문서 전체의 키를 표준 스키마 키로 정규화 (최상위 + 리스트 항목 내부)."""
    if not isinstance(doc, dict):
        return doc
    doc = _remap_keys(doc, _PRD_TOP_KEY_ALIASES)
    for field, item_alias in (
        ("releaseSchedule", _RELEASE_ITEM_ALIASES),
        ("kpi", _KPI_ITEM_ALIASES),
        ("coreFeatures", _CORE_ITEM_ALIASES),
    ):
        val = doc.get(field)
        if isinstance(val, list):
            doc[field] = [_remap_keys(it, item_alias) if isinstance(it, dict) else it for it in val]
    return doc


# 재탕 판정 시 토큰에서 떼어낼 한국어 조사 (긴 것부터: '으로'가 '로'보다 먼저 매칭돼야 함)
_JOSA_SUFFIXES = ("으로", "로", "을", "를", "이", "가", "은", "는", "의", "에", "와", "과", "도", "만")


def _looks_like_runaway(text: str) -> bool:
    """문장부호 없이 짧은 토큰이 길게 나열되는 EXAONE 폭주(예: 회사명 수십 개 나열)를 탐지.
    정상 한국어 산문은 마침표/물음표/느낌표로 문장이 끊기므로, 한 구간이 비정상적으로 길고
    공백 토큰이 매우 많으면 나열형 오염으로 본다 (보수적 임계로 오탐 최소화)."""
    if not isinstance(text, str):
        return False
    for seg in re.split(r'[.!?\n]', text):
        seg = seg.strip()
        if len(seg) > 220 and seg.count(' ') >= 25:
            return True
    return False


def _has_runaway_text(node) -> bool:
    """파싱된 섹션 트리 안의 문자열에 나열형 폭주 텍스트가 있으면 True."""
    if isinstance(node, str):
        return _looks_like_runaway(node)
    if isinstance(node, dict):
        return any(_has_runaway_text(v) for v in node.values())
    if isinstance(node, list):
        return any(_has_runaway_text(v) for v in node)
    return False


def _merge_sections(a: dict, b: dict) -> dict:
    """섹션 결과 병합. 같은 키가 양쪽 다 리스트면 이어붙인다 —
    coreFeatures는 기능 배치별로 여러 worker가 나눠 만들므로 덮어쓰면 안 된다."""
    merged = dict(a or {})
    for key, value in (b or {}).items():
        prev = merged.get(key)
        if isinstance(prev, list) and isinstance(value, list):
            merged[key] = prev + value
        else:
            merged[key] = value
    return merged


_VALID_PRIORITIES = ("P0", "P1", "P2")
_MOSCOW_TO_PRIORITY = {"MUST": "P0", "SHOULD": "P1", "COULD": "P2", "WONT": "P2", "WON'T": "P2"}
_PRIORITY_RE = re.compile(r'^P\s*([0-2])$')


def _normalize_priority(value) -> str | None:
    """P0/P1/P2 또는 MoSCoW 표기를 P0/P1/P2로 정규화. 알아볼 수 없으면 None."""
    text = str(value or "").strip().upper()
    if text in _VALID_PRIORITIES:
        return text
    if text in _MOSCOW_TO_PRIORITY:
        return _MOSCOW_TO_PRIORITY[text]
    match = _PRIORITY_RE.match(text)
    return f"P{match.group(1)}" if match else None


def _fallback_section(section_key: str, feature_list: list[str], user_query: str) -> dict:
    """섹션 생성이 3회 모두 실패했을 때 문서에서 완전히 빠지지 않도록 최소 콘텐츠로 대체."""
    features = feature_list or ["핵심 기능"]
    if section_key == "projectOverview":
        subject = re.split(r"\n\s*\[", user_query or "사용자의 반복 업무와 정보 누락 문제", maxsplit=1)[0]
        subject = re.sub(r"\s+", " ", subject).strip()[:140]
        feature_summary = ", ".join(features[:3])
        return {
            "projectOverview": f"사용자 요구를 바탕으로 {feature_summary} 기능을 하나의 흐름으로 제공해 반복 확인과 정보 누락을 줄이는 서비스입니다.",
            "background": (
                "현재 사용자는 필요한 상태를 반복해서 확인하고 관련 정보를 여러 위치에 따로 기록해야 합니다. "
                "이 과정에서는 입력 누락과 오래된 정보가 발생하기 쉬우며, 적절한 행동 시점을 놓쳐 시간과 자원이 낭비될 수 있습니다. "
                f"본 프로젝트는 {feature_summary} 기능을 연결해 입력부터 상태 확인, 후속 행동까지 일관된 흐름으로 관리합니다. "
                "사용자가 최신 상태와 다음 행동을 즉시 이해하도록 만들어 반복 확인 비용을 낮추고 핵심 업무의 완료율을 높이는 것이 시장 기회입니다."
            ),
        }
    if section_key == "goalsKpi":
        metrics = (
            ("월간 활성 이용 주체", "0건 → 1,000건", "제품 분석 이벤트의 월간 고유 이용 주체 집계", "월간"),
            ("핵심 작업 시작률", "0% → 55%", "서비스 진입 대비 핵심 작업 시작 비율", "주간"),
            ("핵심 기능 활성화율", "0% → 60%", "최초 이용 후 24시간 내 핵심 기능 1회 완료 비율", "주간"),
            ("7일 반복 이용률", "0% → 35%", "첫 이용 코호트의 7일차 재이용 비율", "주간"),
            ("30일 반복 이용률", "0% → 20%", "첫 이용 코호트의 30일차 재이용 비율", "월간"),
            ("핵심 처리 성공률", "0% → 95%", "핵심 기능 요청 중 성공 응답 비율", "일간"),
            ("사용자당 문제 해결 건수", "0건 → 월 10건", "사용자별 핵심 업무 완료 이벤트 합계", "월간"),
        )
        return {
            "goals": [f"'{f}' 기능을 통해 사용자 핵심 문제를 해결하고 측정 가능한 서비스 성과에 기여한다" for f in features[:3]],
            "kpi": [{"metric": metric, "target": target, "basis": "제품 출시 전 기준선 0에서 시작하는 운영 목표",
                     "measurementMethod": method, "frequency": frequency}
                    for metric, target, method, frequency in metrics],
        }
    if section_key == "userPersonas":
        range_match = re.search(r"(\d{2})\s*[-~～]\s*(\d{2})대", user_query or "")
        if range_match:
            start, end = int(range_match.group(1)), int(range_match.group(2))
            allowed_ages = [f"{age}대" for age in range(start, end + 1, 10)] or [f"{start}대"]
        else:
            allowed_ages = ["연령 무관"]
        persona_templates = (
            ("김가람", "서비스 핵심 이용자", "핵심 업무를 빠르게 끝내고 반복 입력을 줄이려 합니다.", "현재 상태를 반복 확인하며 기록을 일관되게 유지하기 어렵습니다."),
            ("이도윤", "서비스 반복 이용자", "필요한 정보를 한곳에서 확인하고 누락과 중복 작업을 줄이려 합니다.", "여러 위치의 기록이 달라 최신 상태를 파악하기 어렵습니다."),
            ("박서윤", "서비스 초기 이용자", "복잡한 설정 없이 중요한 상태와 다음 행동을 쉽게 확인하려 합니다.", "입력 단계가 많거나 낯선 용어가 나오면 작업을 중단하게 됩니다."),
        )
        return {
            "userPersonas": [
                {"name": name, "age": allowed_ages[index % len(allowed_ages)], "job": job,
                 "techLevel": ("높음", "중간", "낮음")[index], "goal": goal, "painPoint": pain,
                 "usagePattern": "웹 UI에서 상태를 확인하고 필요한 작업과 알림에 반응합니다."}
                for index, (name, job, goal, pain) in enumerate(persona_templates)
            ],
        }
    if section_key == "mvpScope":
        split = max(1, min(3, len(features)))
        return {"mvpScope": {"included": features[:split], "excluded": features[split:],
                             "rationale": "핵심 사용자 흐름을 완성하는 선행 기능을 MVP에 포함하고 독립적으로 후속 제공 가능한 기능은 출시 이후로 분리합니다."}}
    if section_key == "techStack":
        return {"techStack": {
            "backend": "FastAPI와 Python", "frontend": "React와 TypeScript", "database": "PostgreSQL",
            "cache": "Redis", "messageQueue": "Kafka", "cdn": "표준 CDN",
            "monitoring": "OpenTelemetry와 Grafana",
            "auth": "사용자 요구사항에 인증이 명시된 경우에만 적합한 인증 방식을 적용합니다.",
        }}
    if section_key == "releaseSchedule":
        return {"releaseSchedule": [{"date": date, "milestone": milestone,
            "description": f"{milestone} 단계의 범위와 완료 조건을 확인하고 다음 단계로 이관할 산출물을 검수합니다.",
            "deliverables": [f"{milestone} 결과서", f"{milestone} 검증 기록", f"{milestone} 승인 체크리스트"]}
            for date, milestone in _RELEASE_STAGES]}
    if section_key == "coreFeatures":
        # priority를 비워 두는 것이 핵심 — 예전엔 여기서 "P0"를 박아 넣어서 생성이 실패한
        # 기능까지 전부 최우선으로 표기됐다. 비워두면 뒤의 MVP 범위 기반 재배정이 채운다.
        items = []
        for feature in features:
            description = f"사용자가 {feature} 기능이 필요한 상황에서 요청하면 시스템이 입력을 검증하고 처리 결과와 실패 사유를 명확하게 반환하여 핵심 업무를 중단 없이 완료하게 합니다."
            requirements = [f"{feature} 요청의 필수 입력과 권한을 요구사항에 따라 검증하고, 상태 변경이 있으면 원자적으로 처리하며 성공 결과와 오류 코드를 기록해야 합니다."]
            items.append({"name": feature, "description": description, "priority": "", "requirements": requirements})
        return {"coreFeatures": items}
    return {}


class _PrdGraphState(TypedDict):
    sections: Annotated[dict, _merge_sections]
    prd_document: str
    ctx: dict


_PLAIN_SECTIONS = {"projectOverview", "mvpScope", "techStack"}
_TECH_DEFAULTS = {
    "backend": "FastAPI 기반 비동기 API 서버를 사용하며 Python AI 생태계와 쉽게 통합합니다.",
    "frontend": "React 기반 웹 UI를 사용해 컴포넌트 재사용성과 빠른 사용자 피드백을 지원합니다.",
    "database": "PostgreSQL을 사용해 관계형 데이터의 정합성과 확장 가능한 질의를 지원합니다.",
    "cache": "Redis를 사용해 반복 조회와 세션성 데이터를 빠르게 처리합니다.",
    "messageQueue": "Kafka를 사용해 비동기 이벤트 처리와 서비스 간 결합도 완화를 지원합니다.",
    "cdn": "CDN을 사용해 정적 자산을 사용자와 가까운 위치에서 빠르게 제공합니다.",
    "monitoring": "구조화 로그와 메트릭 모니터링으로 장애 원인을 추적하고 운영 상태를 관찰합니다.",
    "auth": "사용자 요구사항에 인증이 명시된 경우에만 서비스 특성에 맞는 인증 방식을 적용합니다.",
}


def _target_user_context(target_users) -> str:
    values = []
    for target in target_users or []:
        if hasattr(target, "model_dump"):
            target = target.model_dump(by_alias=True)
        if isinstance(target, dict):
            values.append(" / ".join(str(target.get(key) or "").strip() for key in (
                "persona", "usageEnvironment", "biggestPainPoint",
            ) if str(target.get(key) or "").strip()))
        elif str(target or "").strip():
            values.append(str(target).strip())
    return " | ".join(value for value in values if value)


def _persona_directions(ctx: dict) -> tuple[str, str, str]:
    target = str(ctx.get("target_user_context") or "").strip()
    base = target or "사용자 요구사항에 명시된 핵심 사용자"
    return (
        f"핵심 사용자: {base}; 요구사항에 명시된 연령·직업만 사용하고 없으면 연령 무관으로 작성",
        f"협업자 또는 운영 이해관계자: {base}; 실제 요구사항에 해당 역할이 없으면 핵심 사용자의 다른 사용 상황으로 구분",
        f"간헐적이거나 기술 숙련도가 낮은 사용자: {base}; 연령을 임의로 추정하지 말고 사용 행태로 구분",
    )


def _build_plain_section_prompt(section: str, ctx: dict) -> str:
    common = (
        f"Phase 1 search/conversation evidence:\n{ctx.get('rag_context', '')[:6000]}\n"
        f"User-confirmed MoSCoW priorities:\n{ctx.get('priority_context', '')}\n"
        "JSON·배열·마크다운을 출력하지 말고 지정된 영문 라벨 평문만 작성하세요.\n"
        f"사용자 요구사항: {ctx['user_query']}\n기능 목록: {ctx['feature_str']}\n"
        f"시장 데이터 요약: {ctx['market_data'][:5000]}\n{ctx['rollback_section']}\n"
        "담당 섹션 밖의 내용을 섞지 말고 마지막 줄에 SELF_CHECK: PASS를 출력하세요.\n"
    )
    if section == "projectOverview":
        return common + (
            "OVERVIEW: 누구를 위해 무엇을 어떻게 제공하는지 50자 이상\n"
            "BACKGROUND: 시장 현황, 문제, 기회를 완성된 문장으로 200자 이상"
        )
    if section == "mvpScope":
        return common + "RATIONALE: MVP 포함·제외 기준과 비즈니스 근거를 100자 이상"
    return common + (
        "BACKEND: 기술명과 선택 이유\nFRONTEND: 기술명과 선택 이유\nDATABASE: 기술명과 선택 이유\n"
        "CACHE: 기술명과 선택 이유\nMESSAGE_QUEUE: 기술명과 선택 이유\nCDN: 기술명과 선택 이유\n"
        "MONITORING: 기술명과 선택 이유\nAUTH: 인증 방식과 선택 이유"
    )


def _parse_plain_section(section: str, raw: str, feature_list: list[str], user_query: str) -> dict | None:
    labels = {}
    current = None
    for raw_line in (raw or "").replace("\r", "").splitlines():
        line = raw_line.strip().strip("`*- ")
        if not line:
            continue
        match = re.match(r"^([A-Za-z_]+)\s*:\s*(.*)$", line)
        if match:
            current = match.group(1).upper()
            labels[current] = match.group(2).strip()
        elif current:
            labels[current] = f"{labels[current]} {line}".strip()

    if section == "projectOverview":
        overview = labels.get("OVERVIEW") or user_query
        background = labels.get("BACKGROUND") or f"{user_query}에서 확인된 문제를 해결할 시장 기회가 있습니다."
        return {"projectOverview": overview, "background": background} if overview and background else None
    if section == "mvpScope":
        features = [f for f in feature_list if isinstance(f, str) and f.strip()]
        split = max(1, min(3, len(features)))
        rationale = labels.get("RATIONALE") or "사용자 가치와 구현 의존도가 높은 기능을 우선 포함하고 후속 검증이 필요한 기능은 제외합니다."
        return {"mvpScope": {"included": features[:split], "excluded": features[split:], "rationale": rationale}}
    if section == "techStack":
        mapping = {
            "BACKEND": "backend", "FRONTEND": "frontend", "DATABASE": "database", "CACHE": "cache",
            "MESSAGE_QUEUE": "messageQueue", "CDN": "cdn", "MONITORING": "monitoring", "AUTH": "auth",
        }
        tech = dict(_TECH_DEFAULTS)
        for label, field in mapping.items():
            if labels.get(label):
                tech[field] = labels[label]
        return {"techStack": tech}
    return None


class PrdAgent:

    def __init__(
        self,
        client: AsyncOpenAI,
        model: str = "gpt-4o",
    ):
        self._client = client
        self._model = model
        self._graph = self._build_graph()

    def _build_graph(self):
        graph = StateGraph(_PrdGraphState)
        graph.add_node("worker", self._worker_node)
        graph.add_node("manager_review", self._manager_review_node)
        graph.add_conditional_edges(START, self._dispatch, ["worker"])
        graph.add_edge("worker", "manager_review")
        graph.add_edge("manager_review", END)
        return graph.compile()

    @staticmethod
    def _priority_context(state: PipelineState) -> str:
        groups = (
            ("Must", state.must_features),
            ("Should", state.should_features),
            ("Could", state.could_features),
            ("Excluded", state.excluded_features),
        )
        return "\n".join(
            f"{label}: {', '.join(str(item) for item in values if str(item).strip()) or '(none)'}"
            for label, values in groups
        )

    async def execute(self, state: PipelineState, dump=None) -> PipelineState:
        logger.info("PRD 에이전트 시작 (manager + 7 sub-agent 서브그래프)")
        is_rollback = bool(
            (state.prd_feedback_from_dba or "").strip()
            or (state.prd_feedback_from_api or "").strip()
        )
        if is_rollback:
            logger.warning("PRD rollback 모드 (%d회째)", state.rollback_count)

        try:
            market_data = state.market_research or "시장 데이터 없음"
            feature_str = "- " + "\n- ".join(state.feature_list)
            feature_count = len(state.feature_list)
            rollback_section = self._build_rollback_section(state) if is_rollback else ""

            graph_input = {
                "sections": {},
                "prd_document": "",
                "ctx": {
                    "user_query": state.user_query,
                    "feature_str": feature_str,
                    "feature_count": feature_count,
                    "feature_list": state.feature_list,
                    "rag_context": state.context_prompt or "",
                    "priority_context": self._priority_context(state),
                    "target_user_context": _target_user_context(state.target_users),
                    "market_data": market_data,
                    "rollback_section": rollback_section,
                    "dump": dump,
                },
            }
            result = await self._graph.ainvoke(graph_input)
            prd_document = result["prd_document"]

            # EXAONE이 한글/변형 키로 출력한 섹션을 표준 스키마 키로 되돌린다 (소비자가 못 읽는 문제 방어)
            parsed = try_parse_json(prd_document)
            if isinstance(parsed, dict):
                before_keys = set(parsed.keys())
                parsed = _normalize_prd_keys(parsed)
                remapped = before_keys - set(parsed.keys())
                if remapped:
                    logger.warning("PRD 키 정규화 — 비표준 키 교정: %s", remapped)

                # coreFeatures 누락 시 fallback — feature_list로 최소 구성
                if not parsed.get("coreFeatures"):
                    logger.warning("coreFeatures 누락 — feature_list로 fallback 구성")
                    parsed.update(_fallback_section("coreFeatures", state.feature_list, state.user_query))
                else:
                    # 배치 병합 과정에서 같은 기능이 두 번 들어왔거나 껍데기가 섞였을 수 있어 정리
                    parsed["coreFeatures"] = self._dedup_list(
                        self._drop_empty_features(parsed["coreFeatures"])
                    )

                # 배치별 개수 백스톱(_pad_shortfall)은 "개수"만 맞추므로, top-up이 이미 채워진
                # 기능과 겹치는 항목을 만들어 개수는 맞지만 다른 기능 하나가 통째로 안 채워지는
                # 사례가 실측됐다(예: 8개 기능인데 7개만 매칭, 특정 1개는 대응 항목 없음).
                # uncovered_features(전체 haystack 블롭 안에 토큰이 '어딘가에' 있으면 커버로 침)는
                # "예약 취소/환불 정책 자동 적용"처럼 흔한 낱말(자동·적용 등)로만 이뤄진 기능명이
                # 다른 항목 설명에 우연히 그 낱말이 섞여 있다는 이유로 오탐(거짓 커버)되는 걸
                # 실측으로 확인했다. 그래서 여기서는 항목 단위 근접중복 판정(_is_label_dup,
                # 같은 클래스가 병합 단계에서 이미 쓰는 65% 토큰 겹침 기준)으로 기능마다 실제로
                # 대응하는 coreFeatures 항목이 하나라도 있는지 개별 확인한다.
                existing_token_sets = [
                    self._label_tokens(c.get("name", ""))
                    for c in parsed.get("coreFeatures") or [] if isinstance(c, dict)
                ]
                missing = [
                    f for f in state.feature_list
                    if not self._is_label_dup(self._label_tokens(f), existing_token_sets)
                ]
                if missing:
                    logger.warning(
                        "PRD coreFeatures — 배치 병합 후에도 대응 없는 기능 %d개 — placeholder로 강제 보강: %s",
                        len(missing), missing,
                    )
                    parsed["coreFeatures"] = (parsed.get("coreFeatures") or []) + _fallback_section(
                        "coreFeatures", missing, state.user_query
                    )["coreFeatures"]

                # 우선순위는 항상 마지막에 정리 — MVP 범위가 확정된 뒤라야 도출할 수 있다
                self._reconcile_priorities(parsed.get("coreFeatures"), parsed.get("mvpScope"))
                self._apply_phase1_priorities(parsed, state)
                prd_document = json.dumps(parsed, ensure_ascii=False)

            logger.info("PRD 에이전트 완료 — %d chars", len(prd_document))

            return state.copy(
                prd_document=prd_document,
                prd_feedback_from_dba="",
                prd_feedback_from_api="",
                status_message="PRD 에이전트 완료",
            )
        except Exception as e:
            logger.error("PRD 에이전트 실패: %s", e)
            return state.copy(
                prd_document="{}",
                status_message=f"PRD 에이전트 실패: {e}",
            )

    async def repair(self, state: PipelineState, issues: list[dict], dump=None) -> PipelineState:
        """Regenerate only PRD list items named by QA findings."""
        parsed = try_parse_json(state.prd_document or "{}") or {}
        if not isinstance(parsed, dict):
            return state
        jobs: list[tuple[str, int, str, dict]] = []
        reason_text = "\n".join(str(issue.get("reason") or "") for issue in issues)
        if "일정 문장 잘림 또는 미완결" in reason_text:
            for item in parsed.get("releaseSchedule") or []:
                if not isinstance(item, dict):
                    continue
                description = str(item.get("description") or "").split("subtitle:", 1)[0].strip()
                if description and description in reason_text:
                    description = description.rstrip(" .") + ". 해당 단계의 완료 조건과 산출물을 검증합니다."
                    item["description"] = description
        for index, item in enumerate(parsed.get("releaseSchedule") or []):
            if isinstance(item, dict) and str(item.get("description") or "") in reason_text:
                jobs.append(("release", index, f"기간={item.get('date', '')}; 단계={item.get('milestone', '')}", item))
        for index, item in enumerate(parsed.get("coreFeatures") or []):
            if isinstance(item, dict) and str(item.get("name") or "") in reason_text:
                jobs.append(("core_feature", index, str(item.get("name") or ""), item))
        if not jobs:
            parsed = self._audit_sections(
                parsed, state.feature_list, state.user_query, state.market_research or ""
            )
            self._apply_phase1_priorities(parsed, state)
            return state.copy(prd_document=json.dumps(parsed, ensure_ascii=False))
        ctx = {
            "user_query": state.user_query,
            "feature_str": "- " + "\n- ".join(state.feature_list),
            "feature_count": len(state.feature_list),
            "feature_list": state.feature_list,
            "rag_context": state.context_prompt or "",
            "priority_context": self._priority_context(state),
            "market_data": state.market_research or "시장 데이터 없음",
            "rollback_section": "QA 실패 원인:\n" + reason_text,
            "dump": dump,
        }
        for kind, index, item_context, _old in jobs:
            result = await self._item_worker_node({
                "section": "releaseSchedule" if kind == "release" else "coreFeatures",
                "item_kind": kind,
                "item_index": index,
                "item_context": item_context,
                "prompt": self._build_item_prompt(ctx, kind, index, item_context),
                "dump": dump,
                "feature_list": state.feature_list,
                "feature_count": len(state.feature_list),
                "user_query": state.user_query,
            })
            field = "releaseSchedule" if kind == "release" else "coreFeatures"
            generated = (result.get("sections") or {}).get(field) or []
            if generated and index < len(parsed.get(field) or []):
                parsed[field][index] = generated[0]
        self._reconcile_priorities(parsed.get("coreFeatures"), parsed.get("mvpScope"))
        parsed = self._audit_sections(
            parsed, state.feature_list, state.user_query, state.market_research or ""
        )
        self._apply_phase1_priorities(parsed, state)
        return state.copy(prd_document=json.dumps(parsed, ensure_ascii=False))

    def _dispatch(self, state: dict) -> list[Send]:
        ctx = state["ctx"]
        sends = []
        for section_key, template in SECTION_PROMPTS.items():
            if section_key in {"goalsKpi", "coreFeatures", "userPersonas", "releaseSchedule"}:
                sends.extend(self._item_sends(ctx, section_key))
                continue
            prompt = _build_plain_section_prompt(section_key, ctx)
            sends.append(Send("worker", {
                "section": section_key,
                "prompt": prompt,
                "plain_section": True,
                "dump": ctx.get("dump"),
                "feature_list": ctx.get("feature_list") or [],
                "feature_count": ctx["feature_count"],
                "user_query": ctx["user_query"],
            }))
        return sends

    def _item_sends(self, ctx: dict, section: str) -> list[Send]:
        """목록형 PRD 섹션을 항목 하나당 worker 하나로 병렬 분배한다.

        worker는 배열 JSON을 만들지 않고 라벨 기반 평문 한 항목만 반환한다. Python이
        파싱·검증·정렬 후 최종 배열을 조립하므로 한 항목의 문법 손상이 다른 항목을
        잘라내지 않는다.
        """
        specs: list[tuple[str, int, str]] = []
        if section == "goalsKpi":
            specs.extend(("goal", i, direction) for i, direction in enumerate((
                "사용자 시간 절약과 핵심 문제 해결",
                "반복 사용과 사용자 리텐션 향상",
                f"{ctx.get('target_user_context') or '목표 사용자'}의 업무·생활 성과 개선",
            )))
            specs.extend(("kpi", i, direction) for i, direction in enumerate(_KPI_DIMENSIONS))
        elif section == "userPersonas":
            specs.extend(("persona", i, direction) for i, direction in enumerate(_persona_directions(ctx)))
        elif section == "releaseSchedule":
            specs.extend(
                ("release", i, f"기간={date}; 단계={milestone}")
                for i, (date, milestone) in enumerate(_RELEASE_STAGES)
            )
        elif section == "coreFeatures":
            specs.extend(("core_feature", i, feature) for i, feature in enumerate(ctx.get("feature_list") or []))

        logger.info("PRD %s — 항목 %d개를 단일 항목 worker로 병렬 분할", section, len(specs))
        return [
            Send("worker", {
                "section": section,
                "item_kind": kind,
                "item_index": index,
                "item_context": item_context,
                "prompt": self._build_item_prompt(ctx, kind, index, item_context),
                "dump": ctx.get("dump"),
                "feature_list": ctx.get("feature_list") or [],
                "feature_count": ctx["feature_count"],
                "user_query": ctx["user_query"],
            })
            for kind, index, item_context in specs
        ]

    @staticmethod
    def _build_item_prompt(ctx: dict, kind: str, index: int, item_context: str) -> str:
        common = (
            f"Phase 1 search/conversation evidence:\n{ctx.get('rag_context', '')[:6000]}\n"
            f"User-confirmed MoSCoW priorities:\n{ctx.get('priority_context', '')}\n"
            "아래 PRD 항목 하나만 작성하세요. JSON·배열·마크다운을 출력하지 마세요. "
            "각 필드는 반드시 한 줄이며 지정된 영문 라벨로 시작하세요.\n"
            f"사용자 요구사항: {ctx['user_query']}\n"
            f"기능 목록: {ctx['feature_str']}\n"
            f"항목 방향: {item_context}\n"
            f"시장 데이터 요약: {ctx['market_data'][:5000]}\n"
            f"{ctx['rollback_section']}\n"
            "담당 항목만 작성하고 마지막 줄에 SELF_CHECK: PASS를 출력하세요.\n"
        )
        formats = {
            "goal": (
                "GOAL: 담당 방향의 문제·행동·기대 성과가 드러나는 단일 목표 문장"
            ),
            "kpi": (
                "METRIC: 항목 방향과 직접 연결된 지표명\nTARGET: 현재값 → 목표값\n"
                "BASIS: 외부 수치를 쓰면 제공된 시장 데이터의 원문 URL·수치를 함께 쓰고, 그렇지 않으면 "
                "'제품 출시 전 기준선 0에서 시작하는 내부 운영 목표'라고 작성\n"
                "MEASUREMENT_METHOD: 측정 방법\nFREQUENCY: 일간/주간/월간 중 하나"
            ),
            "persona": (
                "NAME: 가상 인물 이름\nAGE: 연령대\nJOB: 직업\nTECH_LEVEL: 높음/중간/낮음\n"
                "GOAL: 사용 목표 2문장\nPAIN_POINT: 주요 불편 2문장\nUSAGE_PATTERN: 사용 패턴 2문장"
            ),
            "release": (
                "DATE: 지정된 기간\nMILESTONE: 지정된 단계\nDESCRIPTION: 완료 기준을 포함한 구체적인 설명\n"
                "DELIVERABLES: 산출물 3개를 ||| 로 구분"
            ),
            "core_feature": (
                "NAME: 기능명을 항목 방향과 정확히 동일하게 작성\n"
                "DESCRIPTION: 사용 상황→시스템 동작→효과를 포함한 구체적인 설명\n"
                "PRIORITY: P0/P1/P2 중 하나\nREQUIREMENTS: 검증 가능한 요구사항을 ||| 로 구분"
            ),
        }
        return (
            common + contract_prompt("PRD Worker", item_context)
            + "\n출력 형식:\n" + formats[kind] + "\nSELF_CHECK: PASS"
        )

    def _core_feature_sends(self, ctx: dict, template: str) -> list[Send]:
        """coreFeatures만 기능 배치로 쪼개 여러 worker에 나눠 맡긴다.

        기능 21개 × (description 100자 + requirements 60자)를 한 호출에 요구하면 출력이 잘려
        3회 재생성이 전부 파싱 실패하고, 결국 우선순위가 전부 P0인 자리표시자 fallback이
        통째로 들어갔다(실측). 배치당 {n}개면 요구 분량이 토큰 예산 안에 들어온다.
        """.replace("{n}", str(_CORE_FEATURE_BATCH_SIZE))
        features = ctx.get("feature_list") or []
        batches = [
            features[i:i + _CORE_FEATURE_BATCH_SIZE]
            for i in range(0, len(features), _CORE_FEATURE_BATCH_SIZE)
        ] or [[]]
        logger.info("PRD coreFeatures — 기능 %d개를 %d개 배치로 분할", len(features), len(batches))

        sends = []
        for i, batch in enumerate(batches):
            prompt = template.format(
                market_data=ctx["market_data"],
                rollback_section=ctx["rollback_section"],
                user_query=ctx["user_query"],
                feature_str="- " + "\n- ".join(batch) if batch else ctx["feature_str"],
                feature_count=len(batch) or ctx["feature_count"],
            )
            sends.append(Send("worker", {
                "section": "coreFeatures",
                "label_suffix": f"_{i + 1}",
                "prompt": prompt,
                "dump": ctx.get("dump"),
                "feature_list": batch,
                "feature_count": len(batch),
                "user_query": ctx["user_query"],
            }))
        return sends

    async def _worker_node(self, state: dict) -> dict:
        section = state["section"]
        dump = state.get("dump")
        if state.get("item_kind"):
            return await self._item_worker_node(state)
        if state.get("plain_section"):
            label = f"PRD_{section}"
            data = None
            retry_reasons: list[str] = []
            # projectOverview만 한 번의 형식 교정 기회를 주고, 기계적으로 조립 가능한
            # MVP 범위·기술 스택은 실패 즉시 결정론적 값으로 전환한다.
            max_attempts = 2 if section == "projectOverview" else 1
            for attempt in range(max_attempts):
                attempt_prompt = retry_prompt(state["prompt"], retry_reasons if attempt else [], attempt)
                raw = await self._call(
                    attempt_prompt, max_tokens=2200, system=ITEM_WORKER_SYSTEM, enable_thinking=False,
                )
                if dump:
                    dump.log_raw(label, attempt + 1, raw)
                data = _parse_plain_section(
                    section, raw, state.get("feature_list") or [], state.get("user_query") or "",
                )
                retry_reasons = contamination_reasons(data or raw)
                if not data:
                    retry_reasons.append("필수 라벨 파싱 실패")
                if data and not retry_reasons and not has_suspicious_script(data) and not _has_runaway_text(data):
                    break
                logger.warning("%s 평문 섹션 검증 실패 (attempt %d) — %s", label, attempt + 1, retry_reasons)
                data = None
            if not data:
                logger.warning("%s 평문 섹션 최종 검증 실패 — 결정론적 fallback 사용", label)
                data = _fallback_section(section, state.get("feature_list") or [], state.get("user_query") or "")
            return {"sections": data}
        feature_count = state.get("feature_count") or 0
        label = f"PRD_{section}{state.get('label_suffix', '')}"

        min_counts = dict(_SECTION_MIN_COUNTS.get(section, {}))
        if section == "coreFeatures" and feature_count:
            min_counts["coreFeatures"] = feature_count

        # 항목이 많은 섹션(kpi 7+, coreFeatures N개, releaseSchedule 6+)은 토큰 잘림으로
        # 개수가 깎이지 않도록 출력 여유를 늘린다.
        max_tokens = 6000 if section in ("goalsKpi", "coreFeatures", "releaseSchedule") else 4000
        data = await self._generate_section(
            label, state["prompt"], dump=dump, min_counts=min_counts, max_tokens=max_tokens,
            section=section, feature_list=state.get("feature_list") or [], user_query=state.get("user_query") or "",
        )
        if not data:
            logger.error("%s 최종 파싱 실패 또는 기준 미달 — fallback 콘텐츠로 대체", label)
            return {"sections": _fallback_section(section, state.get("feature_list") or [], state.get("user_query") or "")}
        return {"sections": data}

    async def _item_worker_node(self, state: dict) -> dict:
        kind = state["item_kind"]
        index = int(state.get("item_index", 0))
        label = f"PRD_{kind.upper()}_{index + 1}"
        item = None
        retry_reasons: list[str] = []
        # 의미 이탈은 같은 모델이 반복하는 경향이 강하므로 즉시 fallback하고,
        # 라벨/SELF_CHECK 누락처럼 형식 교정 가능한 경우에만 한 번 재시도한다.
        for attempt in range(2):
            raw = await self._call(
                retry_prompt(state["prompt"], retry_reasons, attempt),
                max_tokens=1600, system=ITEM_WORKER_SYSTEM, enable_thinking=False,
            )
            if state.get("dump"):
                state["dump"].log_raw(label, attempt + 1, raw)
            parsed = self._parse_item_text(kind, raw, state.get("item_context", ""), index)
            retry_reasons = self._item_quality_issues(kind, parsed, raw, state.get("item_context", ""))
            if not retry_reasons:
                item = parsed
                break
            logger.warning("%s 단일 항목 검증 실패 (attempt %d) — %s", label, attempt + 1, retry_reasons)
            semantic_failure = any(
                marker in reason for reason in retry_reasons
                for marker in ("의미가 불일치", "관점과 지표", "연령대와 불일치", "동작이 설명되지 않음")
            )
            if semantic_failure:
                break

        if item is None:
            logger.warning("%s 단일 항목 최종 검증 실패 — 결정론적 fallback 사용", label)
            item = self._fallback_item(kind, index, state.get("item_context", ""), state)

        item["_order"] = index
        target = {
            "goal": "goals", "kpi": "kpi", "persona": "userPersonas",
            "release": "releaseSchedule", "core_feature": "coreFeatures",
        }[kind]
        value = item.pop("value") if kind == "goal" else item
        return {"sections": {target: [value]}}

    @staticmethod
    def _parse_item_text(kind: str, raw: str, item_context: str, index: int) -> dict | None:
        labels = {
            "goal": {"GOAL": "value"},
            "kpi": {"METRIC": "metric", "TARGET": "target", "BASIS": "basis",
                    "MEASUREMENT_METHOD": "measurementMethod", "FREQUENCY": "frequency"},
            "persona": {"NAME": "name", "AGE": "age", "JOB": "job", "TECH_LEVEL": "techLevel",
                        "GOAL": "goal", "PAIN_POINT": "painPoint", "USAGE_PATTERN": "usagePattern"},
            "release": {"DATE": "date", "MILESTONE": "milestone", "DESCRIPTION": "description",
                        "DELIVERABLES": "deliverables"},
            "core_feature": {"NAME": "name", "DESCRIPTION": "description", "PRIORITY": "priority",
                             "REQUIREMENTS": "requirements"},
        }[kind]
        # 모델이 `DESCRIPTION: ... ||| REQUIREMENTS: ...`처럼 다음 라벨을 같은 줄에
        # 이어 붙이는 경우가 잦다. 라벨 앞 구분자를 줄바꿈으로 바꾼 뒤 파싱한다.
        marker_pattern = "|".join(re.escape(marker) for marker in labels)
        normalized_raw = re.sub(
            rf"\s*\|\|\|\s*(?=(?:{marker_pattern})\s*:)", "\n", raw or "", flags=re.I,
        )
        normalized_raw = re.sub(r"^\s*SELF_CHECK\s*:\s*PASS\s*$", "", normalized_raw, flags=re.I | re.M)
        result: dict = {}
        current = None
        for raw_line in normalized_raw.replace("\r", "").split("\n"):
            line = raw_line.strip().strip("`*")
            if not line:
                continue
            matched = False
            for marker, field in labels.items():
                prefix = marker + ":"
                if line.upper().startswith(prefix):
                    # A single KPI worker occasionally emits several KPI
                    # blocks. Keep the first complete item and ignore the rest.
                    if kind == "kpi" and marker == "METRIC" and result.get("metric"):
                        return result
                    result[field] = line[len(prefix):].strip().strip('"')
                    current = field
                    matched = True
                    break
            if not matched and current:
                result[current] = (str(result[current]) + " " + line).strip()

        # 단일 목표는 모델이 GOAL 라벨을 생략해도 응답 전체가 곧 값이므로 안전하게 수용한다.
        if kind == "goal" and not result.get("value"):
            plain = " ".join(line.strip() for line in normalized_raw.splitlines() if line.strip())
            if plain and not plain.lstrip().startswith(("{", "[")):
                result["value"] = plain

        # 자주 관찰되는 persona 별칭을 표준 필드로 흡수한다.
        if kind == "persona" and not result.get("age"):
            age_match = re.search(r"^AGE_GROUP\s*:\s*(.+)$", normalized_raw, re.I | re.M)
            if age_match:
                result["age"] = age_match.group(1).strip()
        if kind == "persona" and not result.get("usagePattern"):
            usage_match = re.search(r"^USE_PATTERN\s*:\s*(.+)$", normalized_raw, re.I | re.M)
            if usage_match:
                result["usagePattern"] = usage_match.group(1).strip()

        for field in ("deliverables", "requirements"):
            if isinstance(result.get(field), str):
                result[field] = [x.strip() for x in re.split(r"\s*\|\|\|\s*|\s*;\s*", result[field]) if x.strip()]
        if kind == "kpi" and result.get("target"):
            result["target"] = normalize_target_arrow(result["target"])
        if kind == "core_feature" and not result.get("requirements") and "|||" in str(result.get("description") or ""):
            parts = [x.strip() for x in str(result["description"]).split("|||") if x.strip()]
            result["description"] = parts[0]
            result["requirements"] = parts[1:]
        if kind == "core_feature":
            result["name"] = item_context
            result["priority"] = _normalize_priority(result.get("priority")) or ""
        if kind == "release" and index < len(_RELEASE_STAGES):
            result["date"], result["milestone"] = _RELEASE_STAGES[index]
            if len(result.get("deliverables") or []) < 3:
                result["deliverables"] = [
                    f"{result['milestone']} 완료 보고서",
                    f"{result['milestone']} 검증 결과",
                    f"{result['milestone']} 산출물 패키지",
                ]
        return result or None

    @classmethod
    def _item_quality_issues(cls, kind: str, item: dict | None, raw: str, item_context: str) -> list[str]:
        issues: list[str] = []
        if not cls._valid_item(kind, item):
            issues.append("필수 필드 누락 또는 형식 오류")
        issues.extend(contamination_reasons(item or raw))
        if item and has_placeholder(item):
            issues.append("placeholder 또는 미정 값 포함")
        if kind == "kpi" and item:
            if "→" not in str(item.get("target") or ""):
                issues.append("KPI target이 현재값 → 목표값 형식이 아님")
            direction = item_context.split(":", 1)[0]
            expected = {
                "획득": ("신규", "유입", "획득"), "활성": ("활성", "온보딩", "핵심 기능"),
                "전환": ("전환", "완료", "성공"), "단기 리텐션": ("7일", "단기", "재방문"),
                "장기 리텐션": ("30일", "장기", "리텐션"), "품질": ("정확", "성공", "품질"),
                "비즈니스": ("시간", "비용", "매출", "성과", "절감"),
            }.get(direction, ())
            metric_text = f"{item.get('metric', '')} {item.get('measurementMethod', '')}"
            if expected and not any(token in metric_text for token in expected):
                issues.append("할당된 KPI 관점과 지표 내용이 불일치")
        if kind == "core_feature" and item:
            content = f"{item.get('description', '')} {' '.join(item.get('requirements') or [])}"
            if relevance_score(item_context, content) < 0.2:
                issues.append("담당 기능과 설명·요구사항의 의미가 불일치")
            issues.extend(feature_semantic_issues(item_context, str(item.get("description") or "")))
        return list(dict.fromkeys(issues))

    @staticmethod
    def _valid_item(kind: str, item: dict | None) -> bool:
        if not isinstance(item, dict) or has_suspicious_script(item) or _has_runaway_text(item):
            return False
        required = {
            "goal": ("value",),
            "kpi": ("metric", "target", "basis", "measurementMethod", "frequency"),
            "persona": ("name", "age", "job", "techLevel", "goal", "painPoint", "usagePattern"),
            "release": ("date", "milestone", "description", "deliverables"),
            "core_feature": ("name", "description", "requirements"),
        }[kind]
        if any(not item.get(field) for field in required):
            return False
        if kind == "release" and len(item.get("deliverables") or []) < 3:
            return False
        if kind == "core_feature" and not item.get("requirements"):
            return False
        return True

    @staticmethod
    def _fallback_item(kind: str, index: int, item_context: str, state: dict) -> dict:
        fallback = _fallback_section(state["section"], state.get("feature_list") or [], state.get("user_query") or "")
        field = {"goal": "goals", "kpi": "kpi", "persona": "userPersonas",
                 "release": "releaseSchedule", "core_feature": "coreFeatures"}[kind]
        values = fallback.get(field) or []
        value = values[min(index, len(values) - 1)] if values else "자동 생성 실패 — 수동 보완 필요"
        return {"value": value} if kind == "goal" else dict(value)

    @staticmethod
    def _finalize_parallel_sections(sections: dict) -> dict:
        finalized = dict(sections)
        for field in ("kpi", "userPersonas", "releaseSchedule", "coreFeatures"):
            values = finalized.get(field)
            if isinstance(values, list):
                values = sorted(values, key=lambda x: x.get("_order", 0) if isinstance(x, dict) else 0)
                for value in values:
                    if isinstance(value, dict):
                        value.pop("_order", None)
                finalized[field] = values
        return finalized

    async def _manager_review_node(self, state: dict) -> dict:
        ctx = state["ctx"]
        sections = self._finalize_parallel_sections(state["sections"])
        sections = self._audit_sections(
            sections, ctx.get("feature_list") or [], ctx.get("user_query") or "", ctx.get("market_data") or ""
        )
        # 각 worker 출력은 이미 Python에서 필드별 검증·fallback 처리됐다. 전체 문서를
        # LLM manager가 다시 출력하거나 JSON patch를 만들지 않고, 최종 조립만 수행한다.
        logger.info("PRD manager 결정론적 조립 완료 — 섹션 %d개", len(sections))

        return {"prd_document": json.dumps(sections, ensure_ascii=False)}

    @classmethod
    def _audit_sections(
        cls, sections: dict, feature_list: list[str], user_query: str, market_data: str = ""
    ) -> dict:
        """PRD manager's deterministic semantic gate; invalid worker items never enter the document."""
        out = dict(sections or {})
        fallback = _fallback_section("goalsKpi", feature_list, user_query)

        overview_fallback = _fallback_section("projectOverview", feature_list, user_query)
        overview = str(out.get("projectOverview") or "").strip()
        background = str(out.get("background") or "").strip()
        narrative_leak = re.compile(r"[가-힣][A-Za-z]{3,}|[A-Za-z]{8,}(?:\s|[.,]|$)")
        if (len(overview) < 50 or len(overview) > 300 or "\n" in overview
                or contamination_reasons(overview) or has_placeholder(overview)):
            overview = overview_fallback["projectOverview"]
        if (len(background) < 160 or contamination_reasons(background) or has_placeholder(background)
                or narrative_leak.search(background)):
            background = overview_fallback["background"]
        source_blocks = re.split(r"(?=SOURCE\s+\d+)", market_data or "")
        for block in source_blocks:
            if "provider: KOSIS" not in block or "evidence: KOSIS 실제 통계값" not in block:
                continue
            url_match = re.search(r"^url:\s*(https://\S+)", block, re.M)
            id_match = re.search(r"^identifier:\s*([^\n]+)", block, re.M)
            evidence_match = re.search(r"^evidence:\s*(KOSIS 실제 통계값[^\n]+)", block, re.M)
            if url_match and id_match and evidence_match:
                citation = f" 공식 통계 근거: {evidence_match.group(1)} [KOSIS {id_match.group(1).strip()}] {url_match.group(1)}"
                # 숫자가 포함된 시장 서술은 모델의 부가 추정을 보존하지 않는다. 검증된
                # 문제 배경과 공식 API가 반환한 수치·식별자·원문 URL만 Python이 조립한다.
                background = overview_fallback["background"] + citation
                break
        if source_evidence_issues(background):
            background = overview_fallback["background"]
        out["projectOverview"], out["background"] = overview, background

        goals: list[str] = []
        for goal in out.get("goals") or []:
            text = str(goal or "").strip()
            if not text or contamination_reasons(text) or has_placeholder(text):
                continue
            if any(near_duplicate(text, prior, 0.45) for prior in goals):
                continue
            goals.append(text)
        for goal in fallback["goals"]:
            if len(goals) >= 3:
                break
            if not any(near_duplicate(goal, prior, 0.45) for prior in goals):
                goals.append(goal)
        feature_summary = ", ".join(feature_list[:3]) or "핵심 기능"
        deterministic_goals = (
            f"{feature_summary} 흐름을 통해 사용자의 반복 확인과 작업 누락을 줄여 핵심 문제 해결 시간을 단축한다",
            "핵심 처리의 성공률과 데이터 정합성을 높여 사용자가 서비스를 안정적으로 반복 이용하도록 만든다",
            "출시 후 활성 사용자와 재방문 지표를 측정하고 개선하여 지속 가능한 서비스 운영 기반을 확보한다",
        )
        for goal in deterministic_goals:
            if len(goals) >= 3:
                break
            if not any(near_duplicate(goal, prior, 0.45) for prior in goals):
                goals.append(goal)
        out["goals"] = goals[:3]

        kpis, metric_names = [], []
        approved_scope = " ".join(feature_list)
        for item in out.get("kpi") or []:
            if not isinstance(item, dict) or contamination_reasons(item) or has_placeholder(item):
                continue
            metric = str(item.get("metric") or "").strip()
            target = str(item.get("target") or "")
            if not metric or "→" not in target or not re.search(r"[%명건회원]", target):
                continue
            kpi_text = f"{metric} {item.get('basis', '')} {item.get('measurementMethod', '')}"
            expansion_tokens = ("인식", "OCR", "카메라", "배달", "레시피", "추천")
            if any(token in kpi_text and token not in approved_scope for token in expansion_tokens):
                continue
            if any(near_duplicate(metric, prior, 0.6) for prior in metric_names):
                continue
            basis = str(item.get("basis") or "")
            unsupported_basis = any(
                token in basis for token in ("보고서", "통계", "조사", "연구", "백서", "추정", "업계", "증가 추세")
            ) and "http" not in basis
            if unsupported_basis or kpi_basis_issues(item, market_data):
                item = dict(item)
                item["basis"] = "제품 출시 전 기준선 0에서 시작하는 내부 운영 목표"
            metric_names.append(metric)
            kpis.append(item)
        for item in fallback["kpi"]:
            if len(kpis) >= 7:
                break
            if not any(near_duplicate(item["metric"], prior, 0.6) for prior in metric_names):
                metric_names.append(item["metric"])
                kpis.append(item)
        out["kpi"] = kpis[:7]

        personas, signatures, persona_names = [], set(), set()
        target_range = re.search(r"(\d{2})\s*[-~～]\s*(\d{2})대", user_query or "")
        allowed_ages = (
            {f"{age}대" for age in range(int(target_range.group(1)), int(target_range.group(2)) + 1, 10)}
            if target_range else set()
        )
        for item in out.get("userPersonas") or []:
            if not isinstance(item, dict) or contamination_reasons(item) or has_placeholder(item):
                continue
            age_value = str(item.get("age") or "").strip()
            if not re.match(r"^(?:연령\s*무관|무관|\d{1,2}(?:-\d{1,2})?세|\d{1,2}대(?:\s*(?:초반|중반|후반))?)$", age_value):
                continue
            if allowed_ages and age_value not in allowed_ages:
                continue
            persona_name = str(item.get("name") or "").strip()
            if not persona_name or persona_name in persona_names:
                continue
            sig = (str(item.get("age") or "").strip(), str(item.get("job") or "").strip())
            signature_text = " ".join(sig)
            if any(near_duplicate(signature_text, " ".join(prior), 0.75) for prior in signatures):
                continue
            signatures.add(sig)
            persona_names.add(persona_name)
            personas.append(item)
        for item in _fallback_section("userPersonas", feature_list, user_query)["userPersonas"]:
            if len(personas) >= 3:
                break
            sig = (item["age"], item["job"])
            if item["name"] not in persona_names and not any(
                near_duplicate(" ".join(sig), " ".join(prior), 0.75) for prior in signatures
            ):
                signatures.add(sig)
                persona_names.add(item["name"])
                personas.append(item)
        if len(personas) < 3:
            # Similarity filtering protects diversity, but it must never reduce a
            # required section below its structural minimum. The deterministic
            # fallback already contains distinct named personas, so use remaining
            # names as the final cardinality guard.
            for item in _fallback_section("userPersonas", feature_list, user_query)["userPersonas"]:
                if len(personas) >= 3:
                    break
                if item["name"] not in persona_names:
                    persona_names.add(item["name"])
                    personas.append(item)
        out["userPersonas"] = personas[:3]

        tech_fallback = _fallback_section("techStack", feature_list, user_query)["techStack"]
        tech_stack = out.get("techStack") if isinstance(out.get("techStack"), dict) else {}
        out["techStack"] = {
            key: (value if value and not has_placeholder(value) and not contamination_reasons(value) else tech_fallback[key])
            for key, value in ((key, tech_stack.get(key)) for key in tech_fallback)
        }

        release_fallback = _fallback_section("releaseSchedule", feature_list, user_query)["releaseSchedule"]
        releases = []
        incomplete_tail = re.compile(
            r"(?:하는|되는|위한|통한|해소하는|구현하는|제공하는|"
            r"태스크\s*관리|기능\s*구현|서비스\s*설계)\s*[.]?$"
        )
        for index, expected in enumerate(release_fallback):
            item = (out.get("releaseSchedule") or [])[index] if index < len(out.get("releaseSchedule") or []) else None
            description = str(item.get("description") or "").strip() if isinstance(item, dict) else ""
            if (not isinstance(item, dict) or contamination_reasons(item) or has_placeholder(item)
                    or narrative_leak.search(description) or len(description) < 15 or incomplete_tail.search(description)):
                item = expected
            item["date"], item["milestone"] = expected["date"], expected["milestone"]
            releases.append(item)
        out["releaseSchedule"] = releases

        feature_by_name = {
            str(item.get("name") or "").strip(): item
            for item in out.get("coreFeatures") or [] if isinstance(item, dict)
        }
        clean_core = []
        for feature in feature_list:
            item = feature_by_name.get(feature)
            content = "" if not item else f"{item.get('description', '')} {' '.join(item.get('requirements') or [])}"
            description = "" if not item else str(item.get("description") or "")
            if (not item or contamination_reasons(item) or has_placeholder(item)
                    or relevance_score(feature, content) < 0.2 or feature_semantic_issues(feature, description)):
                item = _fallback_section("coreFeatures", [feature], user_query)["coreFeatures"][0]
            item["name"] = feature
            clean_core.append(item)
        out["coreFeatures"] = clean_core

        scope = out.get("mvpScope") if isinstance(out.get("mvpScope"), dict) else {}
        included = [f for f in scope.get("included") or [] if f in feature_list]
        excluded = [f for f in scope.get("excluded") or [] if f in feature_list and f not in included]
        for feature in feature_list:
            if feature not in included and feature not in excluded:
                excluded.append(feature)
        if not included and feature_list:
            included, excluded = feature_list[:min(3, len(feature_list))], feature_list[min(3, len(feature_list)):]
        out["mvpScope"] = {
            "included": included,
            "excluded": excluded,
            "rationale": scope.get("rationale") or _fallback_section("mvpScope", feature_list, user_query)["mvpScope"]["rationale"],
        }
        cls._reconcile_priorities(out["coreFeatures"], out["mvpScope"])
        return out

    async def _generate_section(
        self, label: str, prompt: str, dump=None, min_counts: dict[str, int] | None = None,
        max_tokens: int = 4000, section: str | None = None,
        feature_list: list[str] | None = None, user_query: str = "",
    ) -> dict | None:
        """파싱 가능한 JSON이 나올 때까지, 그리고 최소 항목 수·스크립트 오염 기준을
        충족할 때까지 최대 3회 재생성 (토큰 반복 루프·응답 잘림·EXAONE 스크립트 혼입 대응).
        3회 모두 기준 미달이면 그중 항목 수가 가장 많았던 결과를 반환한다."""
        best: dict | None = None
        best_score = -1
        for attempt in range(3):
            raw = await self._call(prompt, max_tokens=max_tokens, system=WORKER_SYSTEM, enable_thinking=False)
            if dump:
                dump.log_raw(label, attempt + 1, raw)
            data = try_parse_json(raw)
            if not (data and isinstance(data, dict)):
                logger.warning("%s 파싱 실패 (attempt %d) — 재생성", label, attempt + 1)
                continue
            if has_suspicious_script(data):
                logger.warning("%s 스크립트 오염 감지 (attempt %d) — 재생성", label, attempt + 1)
                continue
            if _has_runaway_text(data):
                logger.warning("%s 나열형 폭주/오염 텍스트 감지 (attempt %d) — 재생성", label, attempt + 1)
                continue
            # 개수 판정 전에 재탕(near-dup) 제거 — 가짜로 부풀린 개수가 아니라 실제 고유 항목 수로 판단
            data = self._dedup_section_fields(data, min_counts)
            score = self._count_score(data, min_counts)
            if score > best_score:
                best, best_score = data, score
            shortfall = self._count_shortfall(data, min_counts)
            if shortfall:
                logger.warning("%s 최소 개수 미달 (attempt %d): %s — 재생성", label, attempt + 1, shortfall)
                continue
            return data
        # 3회 재생성으로도 개수가 부족하면, 이미 만든 항목은 그대로 두고 EXAONE에 '부족분만'
        # 추가로 요청해 서비스 특화 내용으로 채운다. (결정론적 템플릿 채움은 서비스와 무관하므로 쓰지 않음)
        if best is not None:
            shortfall = self._count_shortfall(best, min_counts)
            if shortfall:
                best = await self._topup_short_fields(label, prompt, best, min_counts, max_tokens)
                shortfall = self._count_shortfall(best, min_counts)
            # top-up까지 거치고도 여전히 부족하면(실측: coreFeatures 8개 요구에 4개, userPersonas
            # 3명 요구에 1명만 나온 사례) 조용히 미달 상태로 나가지 않도록 결정론적 placeholder로
            # 강제 보강한다 — 사람이 눈으로 "부족분이 있다"를 바로 알아볼 수 있게 표시해 둔다.
            if shortfall and section:
                best = self._pad_shortfall(section, best, shortfall, feature_list or [], user_query)
        return best

    @staticmethod
    def _pad_shortfall(
        section: str, data: dict, shortfall: dict[str, tuple[int, int]],
        feature_list: list[str], user_query: str,
    ) -> dict:
        """재생성 3회 + top-up 3라운드를 다 거치고도 최소 개수에 못 미치는 필드를
        `_fallback_section`의 placeholder로 강제로 채운다 (완전 실패시의 fallback과 동일한
        모양을 재사용). coreFeatures는 feature_list와 1:1이어야 하므로, 아직 대응하는 항목이
        없는 기능만 골라 채운다 — 이미 채워진 기능에 중복 placeholder를 추가하지 않는다."""
        fallback = _fallback_section(section, feature_list, user_query)
        for field, (have, need) in shortfall.items():
            template = fallback.get(field)
            if not isinstance(template, list) or not template:
                continue
            existing = data.get(field) if isinstance(data.get(field), list) else []
            if field == "coreFeatures" and feature_list:
                present = {str(c.get("name") or "").strip() for c in existing if isinstance(c, dict)}
                candidates = [t for t in template if isinstance(t, dict) and str(t.get("name") or "").strip() not in present]
            else:
                candidates = template
            added = candidates[: need - have]
            if added:
                logger.warning(
                    "%s — top-up 이후에도 %s 부족(%d/%d) — placeholder %d개로 강제 보강",
                    section, field, have, need, len(added),
                )
                data[field] = existing + added
        return data

    @staticmethod
    def _count_shortfall(data: dict, min_counts: dict[str, int] | None) -> dict[str, tuple[int, int]]:
        if not min_counts:
            return {}
        shortfall = {}
        for field, minimum in min_counts.items():
            val = data.get(field)
            n = len(val) if isinstance(val, list) else 0
            if n < minimum:
                shortfall[field] = (n, minimum)
        return shortfall

    @staticmethod
    def _count_score(data: dict, min_counts: dict[str, int] | None) -> int:
        if not min_counts:
            return 0
        total = 0
        for field in min_counts:
            val = data.get(field)
            total += len(val) if isinstance(val, list) else 0
        return total

    async def _topup_short_fields(
        self, label: str, base_prompt: str, data: dict, min_counts: dict[str, int], max_tokens: int,
    ) -> dict:
        """개수가 부족한 리스트 필드를, 이미 만든 항목은 유지한 채 EXAONE에 '부족분만 추가로'
        재요청해서 채운다. 같은 서비스 맥락(base_prompt)을 재사용하므로 채운 항목도 서비스 특화된다.
        결정론적 템플릿 채움은 하지 않으므로 개수를 100% 보장하진 않지만(EXAONE 한계), 매 라운드
        진짜 항목만 늘어나 절대 나빠지지 않는다. 최대 3라운드."""
        for round_ in range(3):
            shortfall = self._count_shortfall(data, min_counts)
            if not shortfall:
                break
            parts = []
            for field, (have, need) in shortfall.items():
                existing = data.get(field) if isinstance(data.get(field), list) else []
                parts.append(
                    f'- "{field}": 현재 {have}개 → {need}개 필요. 아래 기존 항목과 절대 겹치지 않는 '
                    f'새 항목을 정확히 {need - have}개 더 만드세요.\n'
                    f'  기존 항목(이건 다시 넣지 말 것): {json.dumps(existing, ensure_ascii=False)}'
                )
            topup_prompt = (
                base_prompt
                + "\n\n════ 부족분 추가 생성 요청 ════\n"
                + "위와 완전히 동일한 서비스/맥락에서, 아래 필드의 부족한 항목만 추가로 생성하세요.\n"
                + "\n".join(parts)
                + "\n\n응답은 부족한 필드만 담은 JSON이며, 각 필드 값은 '새로 만든 추가 항목들'의 배열입니다"
                + " (기존 항목은 포함하지 마세요). 예: {\"kpi\": [ ...새 항목만... ]}"
            )
            raw = await self._call(topup_prompt, max_tokens=max_tokens, system=WORKER_SYSTEM, enable_thinking=False)
            add = try_parse_json(raw)
            if not (add and isinstance(add, dict)) or has_suspicious_script(add):
                # 일시적 파싱 실패/오염 한 번에 전체를 포기하지 않고 다음 라운드 재시도
                logger.warning("%s top-up 파싱 실패/오염 (round %d) — 다음 라운드 재시도", label, round_ + 1)
                continue
            merged_any = False
            for field, (_have, need) in shortfall.items():
                new_items = add.get(field)
                if isinstance(new_items, list) and new_items:
                    base_list = data.get(field) if isinstance(data.get(field), list) else []
                    # 목표치(need)에 도달하면 멈춰 과도한 초과를 막는다 (예: coreFeatures가 기능 수를 넘겨 부풀지 않도록)
                    merged = self._merge_dedup(base_list, new_items, cap=need)
                    if len(merged) > len(base_list):
                        data[field] = merged
                        merged_any = True
            logger.info(
                "%s top-up round %d — 병합 후 개수: %s",
                label, round_ + 1, {f: len(data.get(f) or []) for f in shortfall},
            )
            if not merged_any:
                # 이번 라운드가 전부 중복이어도 다음 라운드는 temperature 변동으로 새 항목이 나올 수 있어 계속
                logger.warning("%s top-up round %d — 새로 병합된 항목 없음(전부 중복) — 다음 라운드 재시도", label, round_ + 1)
        return data

    # 재탕(거의 동일한 항목) 판정 — name/metric/milestone 라벨 토큰의 겹침 계수(짧은 쪽 기준)가
    # 이 비율 이상이면 같은 항목의 재작성으로 보고 제거한다. top-up이 살짝 바꿔 쓴 중복을
    # 채워넣던 문제(기능 5개가 가짜 9개로, 페르소나·MAU 중복)를 막는다.
    _DEDUP_OVERLAP = 0.65
    _LABEL_FIELDS = ("name", "metric", "milestone", "title")

    @classmethod
    def _label_of(cls, item):
        if isinstance(item, dict):
            for f in cls._LABEL_FIELDS:
                v = item.get(f)
                if isinstance(v, str) and v.strip():
                    return v
        return None

    @staticmethod
    def _exact_key(item):
        return json.dumps(item, ensure_ascii=False, sort_keys=True) if isinstance(item, (dict, list)) else str(item)

    @staticmethod
    def _label_tokens(text) -> set[str]:
        cleaned = re.sub(r"[()\[\]{}:,./·—\-!?'\"]", " ", str(text).lower())
        toks = set()
        for t in cleaned.split():
            # 한국어 조사 제거로 '스캔을'과 '스캔'을 같은 토큰으로 매칭
            for j in _JOSA_SUFFIXES:
                if t.endswith(j) and len(t) - len(j) >= 2:
                    t = t[: -len(j)]
                    break
            if len(t) >= 2:
                toks.add(t)
        return toks

    @classmethod
    def _is_label_dup(cls, tokens: set, prior_token_sets: list) -> bool:
        if not tokens:
            return False
        for pl in prior_token_sets:
            if not pl:
                continue
            denom = min(len(tokens), len(pl))
            if denom and len(tokens & pl) / denom >= cls._DEDUP_OVERLAP:
                return True
        return False

    @classmethod
    def _dedup_list(cls, items: list) -> list:
        """리스트 내부의 재탕 제거. 라벨(name/metric/milestone)이 있는 항목은 토큰 겹침으로
        near-dup까지 제거하고, 라벨 없는 항목(예: goals 문자열)은 완전 일치만 제거한다."""
        out, exact, label_sets = [], set(), []
        for it in items or []:
            ek = cls._exact_key(it)
            if ek in exact:
                continue
            lbl = cls._label_of(it)
            if lbl is not None:
                lt = cls._label_tokens(lbl)
                if cls._is_label_dup(lt, label_sets):
                    continue
                label_sets.append(lt)
            out.append(it)
            exact.add(ek)
        return out

    @classmethod
    def _merge_dedup(cls, base: list, new: list, cap: int | None = None) -> list:
        """base 내부 중복을 먼저 제거한 뒤, new에서 재탕이 아닌 항목만 이어붙인다.
        cap이 주어지면 결과 길이가 cap에 도달한 순간 멈춘다 (top-up 과도 초과 방지)."""
        out = cls._dedup_list(base)
        exact = {cls._exact_key(x) for x in out}
        label_sets = [cls._label_tokens(cls._label_of(x)) for x in out if cls._label_of(x)]
        for item in new:
            if cap is not None and len(out) >= cap:
                break
            ek = cls._exact_key(item)
            if ek in exact:
                continue
            lbl = cls._label_of(item)
            if lbl is not None:
                lt = cls._label_tokens(lbl)
                if cls._is_label_dup(lt, label_sets):
                    continue
                label_sets.append(lt)
            out.append(item)
            exact.add(ek)
        return out

    @classmethod
    def _best_overlap(cls, tokens: set, candidates: list[set]) -> float:
        """tokens와 후보들 중 가장 잘 맞는 것의 겹침 비율 (짧은 쪽 기준)."""
        best = 0.0
        for cand in candidates:
            denom = min(len(tokens), len(cand))
            if denom:
                best = max(best, len(tokens & cand) / denom)
        return best

    @staticmethod
    def _drop_empty_features(core_features) -> list:
        """이름이 비어 있는 coreFeatures 항목 제거.
        실측: {"name":"","description":"","requirements":[]} 껍데기가 최종 PRD까지 나갔고,
        우선순위 재배정이 거기에 P1까지 붙여 정상 항목처럼 보이게 만들었다."""
        if not isinstance(core_features, list):
            return core_features
        kept = [
            f for f in core_features
            if isinstance(f, dict) and str(f.get("name") or "").strip()
        ]
        if len(kept) != len(core_features):
            logger.warning(
                "coreFeatures 빈 항목 %d개 제거 (%d → %d)",
                len(core_features) - len(kept), len(core_features), len(kept),
            )
        return kept

    @classmethod
    def _apply_phase1_priorities(cls, parsed: dict, state: PipelineState) -> None:
        """Make the user's Phase 1 MoSCoW decision authoritative in the PRD."""
        core_features = parsed.get("coreFeatures") or []
        groups = (
            ("P0", state.must_features),
            ("P1", state.should_features),
            ("P2", state.could_features),
            ("P2", state.excluded_features),
        )
        candidates = [
            (priority, cls._label_tokens(name))
            for priority, names in groups
            for name in (names or [])
            if str(name).strip()
        ]
        if not candidates:
            return

        for item in core_features:
            if not isinstance(item, dict):
                continue
            tokens = cls._label_tokens(item.get("name") or "")
            ranked = [
                (cls._best_overlap(tokens, [candidate_tokens]), priority)
                for priority, candidate_tokens in candidates
            ]
            score, priority = max(ranked, default=(0.0, ""))
            if score >= cls._DEDUP_OVERLAP:
                item["priority"] = priority

        # Must features cannot be omitted from the MVP by a downstream worker.
        scope = parsed.get("mvpScope")
        if isinstance(scope, dict):
            must = [str(value) for value in state.must_features if str(value).strip()]
            included = [str(value) for value in scope.get("included") or []]
            excluded = [str(value) for value in scope.get("excluded") or []]
            for feature in must:
                if feature not in included:
                    included.append(feature)
                excluded = [value for value in excluded if value != feature]
            scope["included"] = included
            scope["excluded"] = excluded

    @classmethod
    def _reconcile_priorities(cls, core_features, mvp_scope) -> None:
        """coreFeatures.priority를 정규화하고, 구분이 없거나 비어 있으면 MVP 범위에서 도출한다.

        실측: 21개 기능이 전부 P0로 표기됐다 — coreFeatures 생성이 실패해 priority를 P0로
        박아둔 fallback이 통째로 들어갔기 때문이다. mvpScope(included/excluded)는 정상적으로
        생성되므로 이를 MoSCoW 기준으로 삼는다: MVP 포함=P0(Must), 언급 없음=P1(Should),
        제외=P2(Could). LLM이 이미 우선순위를 실제로 구분해 놓았다면 그 판단을 존중하고
        비어 있는 항목만 채운다. 원본을 제자리 수정."""
        if not isinstance(core_features, list) or not core_features:
            return

        items = [f for f in core_features if isinstance(f, dict)]
        resolved = {}
        for item in items:
            priority = _normalize_priority(item.get("priority"))
            resolved[id(item)] = priority
            if priority:
                item["priority"] = priority

        distinct = {p for p in resolved.values() if p}
        # MVP 범위는 제품 manager의 최종 결정이므로 worker가 낸 priority보다 항상 우선한다.
        # 이를 전량 재계산해야 P0이 excluded에 남는 직접 모순을 막을 수 있다.
        targets = items

        included, excluded = [], []
        if isinstance(mvp_scope, dict):
            included = [x for x in (mvp_scope.get("included") or []) if isinstance(x, str) and x.strip()]
            excluded = [x for x in (mvp_scope.get("excluded") or []) if isinstance(x, str) and x.strip()]
        if not included and not excluded:
            logger.warning(
                "coreFeatures 우선순위 구분 없음(%s)이지만 mvpScope가 비어 도출 불가 — 그대로 둠",
                distinct or "(전부 미지정)",
            )
            return

        inc_tokens = [cls._label_tokens(x) for x in included]
        exc_tokens = [cls._label_tokens(x) for x in excluded]
        counts = {"P0": 0, "P1": 0, "P2": 0}
        for item in targets:
            tokens = cls._label_tokens(item.get("name") or "")
            inc_score = cls._best_overlap(tokens, inc_tokens)
            exc_score = cls._best_overlap(tokens, exc_tokens)
            if max(inc_score, exc_score) < cls._DEDUP_OVERLAP:
                priority = "P1"          # 어느 쪽에도 명시되지 않음 → Should
            elif inc_score >= exc_score:
                priority = "P0"          # MVP 포함 → Must
            else:
                priority = "P2"          # MVP 제외 → Could
            item["priority"] = priority
            counts[priority] += 1

        logger.warning(
            "coreFeatures 우선순위 %d개를 MVP 범위 기준으로 재배정 (기존 구분: %s) — %s",
            len(targets), distinct or "없음", counts,
        )

    @classmethod
    def _dedup_section_fields(cls, data: dict, min_counts: dict | None) -> dict:
        """섹션 데이터의 개수-민감 리스트 필드에서 재탕을 제거 (개수 판정 전에 호출)."""
        if not isinstance(data, dict):
            return data
        fields = set(min_counts or {}) | {"coreFeatures", "kpi", "userPersonas", "releaseSchedule", "goals"}
        for f in fields:
            v = data.get(f)
            if isinstance(v, list):
                data[f] = cls._dedup_list(v)
        return data

    async def _call(self, user_prompt: str, max_tokens: int, system: str, enable_thinking: bool) -> str:
        for attempt in range(2):
            try:
                async with llm_slot():
                    resp = await self._client.chat.completions.create(
                        model=self._model, temperature=_TEMPERATURE, top_p=_TOP_P,
                        presence_penalty=_PRESENCE_PENALTY, max_tokens=max_tokens,
                        frequency_penalty=0.5,
                        messages=[{"role": "system", "content": system}, {"role": "user", "content": user_prompt}],
                        extra_body={"chat_template_kwargs": {"enable_thinking": enable_thinking}},
                    )
                return resp.choices[0].message.content or ""
            except (InternalServerError, APITimeoutError, APIConnectionError) as e:
                logger.warning("PRD API 일시 오류 (attempt %d): %s — 재시도", attempt + 1, e)
                if attempt < 1:
                    await asyncio.sleep(1)
                else:
                    logger.error("PRD API 최종 실패")
                    return ""
        return ""

    def _build_rollback_section(self, state: PipelineState) -> str:
        parts = ["\n⚠️ ROLLBACK 모드 — 아래 피드백을 반드시 반영하세요:"]
        if state.prd_feedback_from_dba:
            parts.append(f"DBA 에이전트 피드백:\n{state.prd_feedback_from_dba}")
        if state.prd_feedback_from_api:
            parts.append(f"API 에이전트 피드백:\n{state.prd_feedback_from_api}")
        return "\n".join(parts)


def reconcile_core_features(core_features, mvp_scope) -> list:
    """coreFeatures 정리(빈 항목 제거 + 우선순위 재배정)의 모듈 레벨 진입점 —
    QA 최종 백스톱에서 재사용한다. (dba_agent.reconcile_fk_types와 동일한 역할)"""
    cleaned = PrdAgent._drop_empty_features(core_features)
    PrdAgent._reconcile_priorities(cleaned, mvp_scope)
    return cleaned
