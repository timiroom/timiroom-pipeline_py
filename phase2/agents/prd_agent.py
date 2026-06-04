import json
import logging

import httpx

from phase2.state import PipelineState

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """당신은 10년 경력의 시니어 PM이자 소프트웨어 아키텍트입니다.
제공된 시장 데이터를 최대한 활용하여 투자자와 개발팀이 신뢰할 수 있는 최고 수준의 PRD를 작성하세요.

══ 작성 품질 원칙 ══
- 모든 텍스트 필드는 최소 기준 글자수를 반드시 충족해야 합니다.
- 단순 키워드 나열 금지 — 반드시 완성된 문장으로 작성하세요.
- 숫자/수치가 포함된 모든 항목에는 반드시 출처(기관명 + 연도)를 괄호로 명시하세요.
- 각 항목이 서로 다른 관점을 다루도록 중복 내용 금지.

══ 절대 금지 ══
- 출처 없는 수치 사용
- "내부 목표", "내부 설문조사" 표현
- A사, B사 등 가상 서비스명
- background는 반드시 문자열로 작성 (배열 금지)
- Shopify, Magento, WooCommerce 등 글로벌 플랫폼 (반드시 한국 서비스로 작성)
- releaseSchedule은 정확히 6개 이상 작성
- fsd는 정확히 12개 이상 작성

반드시 JSON만 출력하고 다른 텍스트는 절대 포함하지 마세요."""

PART_A_PROMPT = """수집된 시장 데이터:
=== 시장 데이터 ===
{market_data}
=================
{rollback_section}

아래 JSON 형식으로 PRD 앞부분을 작성하세요.

(필드별 최소 기준은 Spring Boot 버전의 프롬프트와 동일하게 유지)

JSON:
{{
  "projectOverview": "서비스 한 줄 개요 (50자 이상)",
  "background": "500자 이상 서술형 문단",
  "goals": ["~을 통해 ~을 달성하여 ~에 기여한다 (60자 이상)"],
  "kpi": [{{"metric":"지표명","target":"현재값 → 목표값","basis":"출처","measurementMethod":"측정 방법","frequency":"측정 주기"}}],
  "marketData": {{
    "marketSize": "시장 규모 수치 + 출처",
    "growthRate": "성장률 + 출처",
    "competitors": [{{"name":"한국 서비스명","revenue":"매출액 + 출처","marketShare":"점유율","feature":"강점 2문장","weakness":"약점 2문장","differentiator":"차별화 포인트"}}],
    "userPainPoints": [{{"point":"Pain Point","data":"수치 + 출처","impact":"영향 2문장"}}]
  }},
  "userPersonas": [{{"name":"이름","age":"나이","job":"직업","techLevel":"낮음/중간/높음","goal":"목표 2문장","painPoint":"페인포인트 2문장","usagePattern":"사용 패턴 2문장"}}],
  "competitiveMatrix": {{"criteria": ["기준 5개"],"competitors": [{{"name":"서비스명","scores":["상/중/하"]}}]}},
  "coreFeatures": [{{"name":"기능명","description":"100자 이상","priority":"P0/P1/P2","requirements":["60자 이상 조건"]}}],
  "mvpScope": {{"included":["기능명"],"excluded":["기능명"],"rationale":"100자 이상"}},
  "userFlow": ["1. 단계명 — 사용자 행동 → 시스템 반응 (80자 이상)"],
  "uxConsiderations": [{{"category":"카테고리","detail":"150자 이상","reference":"UX 원칙"}}],
  "techStack": {{"backend":"이유 2문장","frontend":"이유 2문장","database":"이유 2문장","cache":"이유 2문장","messageQueue":"이유 2문장","cdn":"이유 2문장","monitoring":"이유 2문장","auth":"이유 2문장"}},
  "nonFunctionalRequirements": {{"performance":"100자 이상","security":"100자 이상","legal":"100자 이상","scalability":"100자 이상","availability":"100자 이상"}},
  "releaseSchedule": [{{"date":"기간","milestone":"마일스톤명","description":"100자 이상","team":"담당팀","deliverables":["산출물"]}}]
}}

사용자 요구사항: {user_query}
기능 목록: {feature_str}
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
  "risks": [{{"title":"리스크명","probability":"상/중/하","impact":"상/중/하","description":"150자 이상","strategy":"100자 이상","contingency":"1단계: ~ → 4단계 이상"}}],
  "fsd": [{{"id":"FSD_001","category":"기능 분류","description":"80자 이상","action":"① ~ → ② ~ → ③ ~","acceptanceCriteria":"조건 2개 이상","note":"비고"}}],
  "operationPolicy": [{{"id":"POL_001","category":"정책 분류","description":"100자 이상","action":"1) ~ → 4단계 이상","legalBasis":"법령명 제X조 — 내용","note":"비고"}}],
  "successMetrics": [{{"metric":"지표명","target":"현재값 → 목표값","measurementTool":"도구명","frequency":"측정 주기","owner":"담당팀","baselineNote":"기준값 수립 방법"}}]
}}

서비스 요구사항: {user_query}
기능 목록: {feature_str}
"""


class PrdAgent:

    def __init__(self, openai_api_key: str, model: str = "gpt-4o"):
        self._api_key = openai_api_key
        self._model = model

    async def execute(self, state: PipelineState) -> PipelineState:
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
            rollback_section = self._build_rollback_section(state) if is_rollback else ""

            logger.info("STEP 2A: PRD 앞부분 생성 중...")
            part_a = await self._call(
                PART_A_PROMPT.format(
                    market_data=market_data,
                    rollback_section=rollback_section,
                    user_query=state.user_query,
                    feature_str=feature_str,
                ),
                max_tokens=8000,
            )

            logger.info("STEP 2B: PRD 뒷부분 생성 중...")
            part_b = await self._call(
                PART_B_PROMPT.format(
                    market_data=market_data,
                    part_a_kpi=self._extract_kpi_summary(part_a),
                    rollback_section=rollback_section,
                    user_query=state.user_query,
                    feature_str=feature_str,
                ),
                max_tokens=6000,
            )

            prd_document = self._merge(part_a, part_b)
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
        body = {
            "model": self._model,
            "temperature": 0.1,
            "max_tokens": max_tokens,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
        }
        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {self._api_key}"},
                json=body,
            )
            resp.raise_for_status()
            data = resp.json()
        return data["choices"][0]["message"]["content"]

    def _extract_kpi_summary(self, part_a: str) -> str:
        try:
            node = json.loads(part_a)
            kpi_list = node.get("kpi", [])
            if not kpi_list:
                return "Part A KPI 없음 — successMetrics를 서비스 목표에 맞게 독립 작성"
            lines = ["아래 KPI와 successMetrics를 1:1 대응하여 동일한 metric명·target값을 사용하세요:"]
            for kpi in kpi_list:
                lines.append(
                    f"- metric: {kpi.get('metric')} | target: {kpi.get('target')} | basis: {kpi.get('basis')}"
                )
            return "\n".join(lines)
        except Exception:
            return "Part A KPI 파싱 실패 — successMetrics를 서비스 목표에 맞게 독립 작성"

    def _build_rollback_section(self, state: PipelineState) -> str:
        parts = ["\n⚠️ ROLLBACK 모드 — 아래 피드백을 반드시 반영하세요:"]
        if state.prd_feedback_from_dba:
            parts.append(f"DBA 에이전트 피드백:\n{state.prd_feedback_from_dba}")
        if state.prd_feedback_from_api:
            parts.append(f"API 에이전트 피드백:\n{state.prd_feedback_from_api}")
        return "\n".join(parts)

    def _merge(self, part_a: str, part_b: str) -> str:
        try:
            a = json.loads(part_a)
            b = json.loads(part_b)
            a.update(b)
            return json.dumps(a, ensure_ascii=False)
        except Exception as e:
            logger.error("JSON 합치기 실패: %s", e)
            return part_a
