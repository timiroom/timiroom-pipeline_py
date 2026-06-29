import asyncio
import json
import logging

from openai import AsyncOpenAI, InternalServerError, APITimeoutError, APIConnectionError

from phase2.json_utils import try_parse_json
from phase2.state import PipelineState

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """JSON만 출력하세요. 설명·인사말·마크다운 코드블록 금지. { 로 시작해서 } 로 끝납니다.

당신은 10년 경력의 시니어 PM이자 소프트웨어 아키텍트입니다.
제공된 시장 데이터를 최대한 활용하여 투자자와 개발팀이 신뢰할 수 있는 최고 수준의 PRD를 작성하세요.

══ 작성 원칙 ══
- 완성된 문장으로 작성하세요 (키워드 나열 금지).
- 수치에는 출처(기관명 + 연도)를 괄호로 명시하세요.
- background는 반드시 문자열로 작성 (배열 금지).

══ 절대 금지 ══
- A사, B사 등 가상 서비스명 — 실제 한국 서비스명 사용
- Shopify, Magento, WooCommerce 등 글로벌 플랫폼 — 반드시 한국 서비스로 작성
- $name, $serviceName, {name}, {value} 등 placeholder를 값으로 사용하는 것
- competitors 배열에서 동일한 serviceName 반복 — 서로 다른 실존 서비스 5개
- 같은 단어나 공백을 연속으로 반복 생성하는 것
- thinking 태그(</think>, <think>) 를 JSON 값 안에 포함하는 것

반드시 JSON만 출력하고 다른 텍스트는 절대 포함하지 마세요."""

# Part A: 핵심 기능 정보 먼저 → 이후 시장·경쟁 데이터
# coreFeatures를 앞쪽에 배치하여 EXAONE 반복 루프가 뒤쪽에서 발생해도 핵심 정보는 보존
PART_A_PROMPT = """수집된 시장 데이터:
=== 시장 데이터 ===
{market_data}
=================
{rollback_section}

아래 JSON 형식으로 PRD 앞부분을 작성하세요.
⚠️ coreFeatures는 기능 목록의 각 기능과 반드시 1:1 대응하여 작성하세요 (총 {feature_count}개).
⚠️ competitors의 serviceName에는 실제 존재하는 한국 서비스명만 작성하세요 (당근마켓, 배달의민족, 토스, 카카오, 네이버 등).
⚠️ $name, $serviceName 등 placeholder를 값으로 절대 사용하지 마세요.

JSON:
{{
  "projectOverview": "서비스 한 줄 개요",
  "background": "200자 이상 서술형 문단",
  "goals": ["달성 목표 1문장"],
  "coreFeatures": [{{"name":"기능명","description":"이 기능이 해결하는 문제와 동작 방식 설명","priority":"P0/P1/P2","requirements":["처리 조건 1개 이상"]}}],
  "mvpScope": {{"included":["기능명"],"excluded":["기능명"],"rationale":"MVP 범위 선정 이유"}},
  "userPersonas": [{{"name":"이름","age":"나이","job":"직업","techLevel":"낮음/중간/높음","goal":"사용 목표","painPoint":"주요 불편함","usagePattern":"사용 패턴"}}],
  "kpi": [{{"metric":"지표명","target":"현재값 → 목표값","basis":"출처"}}],
  "marketData": {{
    "marketSize": "시장 규모 수치 + 출처",
    "growthRate": "성장률 + 출처",
    "competitors": [{{"serviceName":"실존 한국 서비스명 (당근마켓·배달의민족 등 서로 다른 실존 서비스 5개)","feature":"핵심 강점","weakness":"주요 약점"}}],
    "userPainPoints": [{{"point":"Pain Point","data":"수치 + 출처","impact":"영향"}}]
  }},
  "competitiveMatrix": {{"criteria": ["기준 3개"],"competitors": [{{"serviceName":"서비스명","scores":["상/중/하"]}}]}},
  "userFlow": ["1. 단계명 — 사용자 행동 → 시스템 반응"],
  "techStack": {{"backend":"기술명과 선택 이유","frontend":"기술명과 선택 이유","database":"기술명과 선택 이유","auth":"인증 방식"}},
  "nonFunctionalRequirements": {{"performance":"성능 요구사항","security":"보안 요구사항","scalability":"확장성 요구사항"}},
  "releaseSchedule": [{{"date":"기간","milestone":"마일스톤명","description":"주요 내용","deliverables":["산출물"]}}]
}}

사용자 요구사항: {user_query}
기능 목록 (coreFeatures를 아래 {feature_count}개 기능 기준으로 작성):
{feature_str}
"""

PART_B_PROMPT = """수집된 시장 데이터:
=== 시장 데이터 ===
{market_data}
=================

=== Part A KPI 목록 (successMetrics와 반드시 1:1 대응) ===
{part_a_kpi}
=================
{rollback_section}

아래 JSON 형식으로 PRD 뒷부분을 작성하세요.

JSON:
{{
  "risks": [{{"title":"리스크명","probability":"상/중/하","impact":"상/중/하","description":"리스크 설명","strategy":"대응 전략","contingency":"단계별 대응 시나리오"}}],
  "fsd": [{{"id":"FSD_001","category":"기능 분류","description":"기능 상세 설명","action":"① 사용자 행동 → ② 시스템 처리 → ③ 결과","acceptanceCriteria":"완료 조건","note":"비고"}}],
  "operationPolicy": [{{"id":"POL_001","category":"정책 분류","description":"정책 설명","action":"처리 절차","legalBasis":"관련 법령","note":"비고"}}],
  "successMetrics": [{{"metric":"지표명","target":"현재값 → 목표값","measurementTool":"도구명","owner":"담당팀"}}]
}}

서비스 요구사항: {user_query}
기능 목록: {feature_str}
"""


class PrdAgent:

    def __init__(self, client: AsyncOpenAI, model: str = "gpt-4o"):
        self._client = client
        self._model = model

    async def execute(self, state: PipelineState, dump=None) -> PipelineState:
        logger.info("PRD 에이전트 시작")
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

            logger.info("STEP 2A: PRD 앞부분 생성 중...")
            part_a = await self._call(
                PART_A_PROMPT.format(
                    market_data=market_data,
                    rollback_section=rollback_section,
                    user_query=state.user_query,
                    feature_str=feature_str,
                    feature_count=feature_count,
                ),
                max_tokens=6000,
            )
            if dump:
                dump.log_raw("PRD_PART_A", 1, part_a)

            logger.info("STEP 2B: PRD 뒷부분 생성 중...")
            part_b = await self._call(
                PART_B_PROMPT.format(
                    market_data=market_data,
                    part_a_kpi=self._extract_kpi_summary(part_a),
                    rollback_section=rollback_section,
                    user_query=state.user_query,
                    feature_str=feature_str,
                ),
                max_tokens=8000,
            )
            if dump:
                dump.log_raw("PRD_PART_B", 1, part_b)

            prd_document = self._merge(part_a, part_b)

            # coreFeatures 누락 시 fallback — feature_list로 최소 구성
            parsed = try_parse_json(prd_document)
            if parsed and not parsed.get("coreFeatures"):
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

    async def _call(self, user_prompt: str, max_tokens: int) -> str:
        for attempt in range(3):
            try:
                resp = await self._client.chat.completions.create(
                    model=self._model,
                    temperature=0.1,
                    max_tokens=max_tokens,
                    frequency_penalty=0.5,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                )
                return resp.choices[0].message.content or ""
            except (InternalServerError, APITimeoutError, APIConnectionError) as e:
                logger.warning("PRD API 일시 오류 (attempt %d): %s — 재시도", attempt + 1, e)
                if attempt < 2:
                    await asyncio.sleep(5 * (attempt + 1))
                else:
                    raise
        return ""

    def _extract_kpi_summary(self, part_a: str) -> str:
        node = try_parse_json(part_a)
        if not node:
            return "Part A KPI 파싱 실패 — successMetrics를 서비스 목표에 맞게 독립 작성"
        kpi_list = node.get("kpi", [])
        if not kpi_list:
            return "Part A KPI 없음 — successMetrics를 서비스 목표에 맞게 독립 작성"
        lines = ["아래 KPI와 successMetrics를 1:1 대응하여 동일한 metric명·target값을 사용하세요:"]
        for kpi in kpi_list:
            lines.append(
                f"- metric: {kpi.get('metric')} | target: {kpi.get('target')} | basis: {kpi.get('basis')}"
            )
        return "\n".join(lines)

    def _build_rollback_section(self, state: PipelineState) -> str:
        parts = ["\n⚠️ ROLLBACK 모드 — 아래 피드백을 반드시 반영하세요:"]
        if state.prd_feedback_from_dba:
            parts.append(f"DBA 에이전트 피드백:\n{state.prd_feedback_from_dba}")
        if state.prd_feedback_from_api:
            parts.append(f"API 에이전트 피드백:\n{state.prd_feedback_from_api}")
        return "\n".join(parts)

    def _merge(self, part_a: str, part_b: str) -> str:
        logger.debug("Part A raw (첫 500자): %s", part_a[:500])
        logger.debug("Part B raw (첫 500자): %s", part_b[:500])
        a = try_parse_json(part_a)
        b = try_parse_json(part_b)
        if a and b:
            a.update(b)
            return json.dumps(a, ensure_ascii=False)
        if a:
            logger.warning("Part B 파싱 실패 — Part A만 사용")
            return json.dumps(a, ensure_ascii=False)
        if b:
            logger.warning("Part A 파싱 실패 — Part B만 사용")
            return json.dumps(b, ensure_ascii=False)
        logger.error("Part A/B 모두 파싱 실패\nPart A raw: %s\nPart B raw: %s", part_a[:1000], part_b[:1000])
        return part_a
