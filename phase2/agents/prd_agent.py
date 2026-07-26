import asyncio
import json
import logging
import re
from typing import Annotated, TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.types import Send
from openai import AsyncOpenAI, InternalServerError, APITimeoutError, APIConnectionError

from phase2.json_utils import try_parse_json, has_suspicious_script
from phase2.state import PipelineState

logger = logging.getLogger(__name__)

# 섹션별 최소 항목 수 — SECTION_PROMPTS가 스스로 요구하는 기준과 동일.
# _generate_section에서 이 기준 미달 시 파싱 성공이라도 재생성을 강제한다.
_SECTION_MIN_COUNTS: dict[str, dict[str, int]] = {
    "goalsKpi": {"kpi": 7},
    "userPersonas": {"userPersonas": 3},
    "releaseSchedule": {"releaseSchedule": 6},
}

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
    merged = dict(a or {})
    merged.update(b or {})
    return merged


def _fallback_section(section_key: str, feature_list: list[str], user_query: str) -> dict:
    """섹션 생성이 3회 모두 실패했을 때 문서에서 완전히 빠지지 않도록 최소 콘텐츠로 대체."""
    features = feature_list or ["핵심 기능"]
    if section_key == "projectOverview":
        return {
            "projectOverview": (user_query or "서비스 개요")[:200] or "서비스 개요 생성 실패",
            "background": f"'{(user_query or '이 서비스')[:80]}' 요구사항을 기반으로 한 서비스입니다. "
                          "자동 생성에 실패하여 상세 배경 설명이 채워지지 않았습니다. 수동 보완이 필요합니다.",
        }
    if section_key == "goalsKpi":
        return {
            "goals": [f"{f}을(를) 통해 사용자 핵심 문제를 해결하여 서비스 목표에 기여한다" for f in features[:3]],
            "kpi": [{
                "metric": f"핵심 지표 {i}", "target": "0% → 목표치 미정", "basis": "자동 생성 실패 — 수동 보완 필요",
                "measurementMethod": "미정", "frequency": "미정",
            } for i in range(1, 8)],
        }
    if section_key == "userPersonas":
        return {
            "userPersonas": [
                {"name": f"페르소나 {i}", "age": "미정", "job": "미정", "techLevel": "중간",
                 "goal": "미정", "painPoint": "미정", "usagePattern": "미정"}
                for i in range(1, 4)
            ],
        }
    if section_key == "mvpScope":
        return {"mvpScope": {"included": features, "excluded": [], "rationale": "자동 생성 실패 — 수동 보완 필요"}}
    if section_key == "techStack":
        return {"techStack": {
            "backend": "미정", "frontend": "미정", "database": "미정", "cache": "미정",
            "messageQueue": "미정", "cdn": "미정", "monitoring": "미정", "auth": "JWT Bearer 토큰",
        }}
    if section_key == "releaseSchedule":
        return {"releaseSchedule": [
            {"date": f"{i}차", "milestone": "미정", "description": "자동 생성 실패 — 수동 보완 필요", "deliverables": ["미정"]}
            for i in range(1, 7)
        ]}
    if section_key == "coreFeatures":
        return {"coreFeatures": [
            {"name": f, "description": f"{f} 기능", "priority": "P0", "requirements": [f"{f}를 처리한다"]}
            for f in features
        ]}
    return {}


class _PrdGraphState(TypedDict):
    sections: Annotated[dict, _merge_sections]
    prd_document: str
    ctx: dict


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
                    parsed["coreFeatures"] = [
                        {"name": f, "description": f"{f} 기능", "priority": "P0", "requirements": [f"{f}를 처리한다"]}
                        for f in state.feature_list
                    ]
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

    def _dispatch(self, state: dict) -> list[Send]:
        ctx = state["ctx"]
        sends = []
        for section_key, template in SECTION_PROMPTS.items():
            prompt = template.format(
                market_data=ctx["market_data"],
                rollback_section=ctx["rollback_section"],
                user_query=ctx["user_query"],
                feature_str=ctx["feature_str"],
                feature_count=ctx["feature_count"],
            )
            sends.append(Send("worker", {
                "section": section_key,
                "prompt": prompt,
                "dump": ctx.get("dump"),
                "feature_list": ctx.get("feature_list") or [],
                "feature_count": ctx["feature_count"],
                "user_query": ctx["user_query"],
            }))
        return sends

    async def _worker_node(self, state: dict) -> dict:
        section = state["section"]
        dump = state.get("dump")
        feature_count = state.get("feature_count") or 0
        label = f"PRD_{section}"

        min_counts = dict(_SECTION_MIN_COUNTS.get(section, {}))
        if section == "coreFeatures" and feature_count:
            min_counts["coreFeatures"] = feature_count

        # 항목이 많은 섹션(kpi 7+, coreFeatures N개, releaseSchedule 6+)은 토큰 잘림으로
        # 개수가 깎이지 않도록 출력 여유를 늘린다.
        max_tokens = 6000 if section in ("goalsKpi", "coreFeatures", "releaseSchedule") else 4000
        data = await self._generate_section(label, state["prompt"], dump=dump, min_counts=min_counts, max_tokens=max_tokens)
        if not data:
            logger.error("%s 최종 파싱 실패 또는 기준 미달 — fallback 콘텐츠로 대체", label)
            return {"sections": _fallback_section(section, state.get("feature_list") or [], state.get("user_query") or "")}
        return {"sections": data}

    async def _manager_review_node(self, state: dict) -> dict:
        ctx = state["ctx"]
        sections = state["sections"]
        dump = ctx.get("dump")

        prompt = MANAGER_REVIEW_PROMPT.format(
            sections_json=json.dumps(sections, ensure_ascii=False),
            criteria=_SECTION_CRITERIA.format(feature_count=ctx["feature_count"]),
            user_query=ctx["user_query"],
            feature_count=ctx["feature_count"],
            feature_str=ctx["feature_str"],
        )

        try:
            raw = await self._call(prompt, max_tokens=16384, system=MANAGER_SYSTEM, enable_thinking=False)
            if dump:
                dump.log_raw("PRD_MANAGER_REVIEW", 1, raw)
            review = try_parse_json(raw)
            if review is None or not isinstance(review, dict):
                logger.warning("PRD manager 리뷰 파싱 실패 — 원본 섹션 유지")
            else:
                patches = review.get("patches")
                if isinstance(patches, dict) and patches:
                    # 패치 키도 표준화(goal→goals 등)해서 원본 섹션과 정확히 대응시킨다
                    patches = _remap_keys(patches, _PRD_TOP_KEY_ALIASES)
                    applied: dict = {}
                    for key, val in patches.items():
                        orig = sections.get(key)
                        # manager가 리스트 섹션을 원본보다 줄이면(항목 누락) 거부 — QA 축소 방어막과 동일 원칙.
                        # PRD manager는 '교정'이지 '삭감'이 아니므로 개수 감소는 항상 결함으로 본다.
                        if isinstance(orig, list) and isinstance(val, list) and len(val) < len(orig):
                            logger.warning(
                                "PRD manager 패치가 %s를 축소(%d→%d, EXAONE 누락 추정) — 거부, 원본 유지",
                                key, len(orig), len(val),
                            )
                            continue
                        applied[key] = val
                    if applied:
                        logger.info("PRD manager 리뷰 — %d개 섹션 패치: %s", len(applied), list(applied.keys()))
                        sections = {**sections, **applied}
                    else:
                        logger.info("PRD manager 리뷰 — 유효 패치 없음, 원본 유지")
                else:
                    logger.info("PRD manager 리뷰 — 패치 없음, 원본 유지")
        except Exception as e:
            logger.warning("PRD manager 리뷰 실패 — 원본 섹션 유지: %s", e)

        return {"prd_document": json.dumps(sections, ensure_ascii=False)}

    async def _generate_section(
        self, label: str, prompt: str, dump=None, min_counts: dict[str, int] | None = None,
        max_tokens: int = 4000,
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
        if best is not None and self._count_shortfall(best, min_counts):
            best = await self._topup_short_fields(label, prompt, best, min_counts, max_tokens)
        return best

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
        for attempt in range(3):
            try:
                resp = await self._client.chat.completions.create(
                    model=self._model,
                    temperature=_TEMPERATURE,
                    top_p=_TOP_P,
                    presence_penalty=_PRESENCE_PENALTY,
                    max_tokens=max_tokens,
                    frequency_penalty=0.5,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user_prompt},
                    ],
                    extra_body={"chat_template_kwargs": {"enable_thinking": enable_thinking}},
                )
                return resp.choices[0].message.content or ""
            except (InternalServerError, APITimeoutError, APIConnectionError) as e:
                logger.warning("PRD API 일시 오류 (attempt %d): %s — 재시도", attempt + 1, e)
                if attempt < 2:
                    await asyncio.sleep(5 * (attempt + 1))
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
