import asyncio
import json
import logging
import re

from openai import AsyncOpenAI, InternalServerError, APITimeoutError, APIConnectionError

from common.pm_skills import PmSkillsLoader
from phase2.json_utils import has_suspicious_script, try_parse_json
from phase2.quality_rules import (
    contamination_reasons, dedupe_labels, retry_prompt, self_check_passed,
)
from phase2.state import PipelineState
from phase2.agent_contract import (
    auth_feature_specs, contract_prompt, feature_requires_user_scope, requires_auth,
)
from phase2.llm_concurrency import llm_slot

logger = logging.getLogger(__name__)

# 길이는 손상 신호로는 약하다 — 아래 패턴 검사가 스크램블을 직접 잡으므로, 정상적인 긴
# 기능명(괄호 안 상세 설명 포함)이 잘려 DBA·API가 참고할 정보를 잃지 않도록 여유를 둔다.
# (실측: '보유 재료 기반 레시피 추천 (AI 기반 재료 조합 분석, 조리 시간, 난이도, ...)' 62자가
#  멀쩡한데도 부연이 통째로 잘렸다)
_MAX_FEATURE_NAME_LEN = 100
_MIN_MEANINGFUL_CHAR_RATIO = 0.5
_MAX_HYPHEN_SEGMENTS = 4
_MAX_HYPHEN_CHAIN = 3
_MIN_TRIMMED_NAME_LEN = 4

_HANGUL = "가-힣"
# EXAONE 토큰 손상 패턴 (전부 실측 사례)
# - '푸0일 전 푸시 알림'  : 한글 낱말 사이에 숫자가 끼어듦. '1일 전', '20-30대'처럼
#                          숫자가 낱말 앞·뒤에 오는 정상 표기는 걸리지 않는다.
_DIGIT_IN_HANGUL_RE = re.compile(rf"[{_HANGUL}]\d+[{_HANGUL}]")
# - '누Guest 재료 기반'   : 한글 바로 뒤에 영문이 붙음. 반대 방향(영문 뒤 한글 조사,
#                          예: 'API를', 'OCR로')은 정상 한국어 표기라 검사하지 않는다.
_HANGUL_THEN_LATIN_RE = re.compile(rf"[{_HANGUL}][A-Za-z]")
# - '_recipe_auto_suggest_via_ingredients_' : 내부 식별자가 기능명 자리로 유출
_IDENTIFIER_LIKE_RE = re.compile(r"^[A-Za-z0-9]*_[A-Za-z0-9_]*$")


def _corruption_reason(name: str) -> str | None:
    """손상으로 판단되는 이유. 정상이면 None."""
    if len(name) > _MAX_FEATURE_NAME_LEN:
        return "길이 초과"
    if has_suspicious_script(name):
        return "스크립트 오염"
    meaningful = sum(1 for ch in name if ch.isalnum())
    non_space = sum(1 for ch in name if not ch.isspace())
    if non_space and meaningful / non_space < _MIN_MEANINGFUL_CHAR_RATIO:
        return "기호 비율 과다"
    if name.count("-") > _MAX_HYPHEN_SEGMENTS:
        return "하이픈 과다"
    # 공백 없이 하이픈으로만 이어붙인 이름은 정상 한국어 기능명이 아니다
    # (실측: '기구-재물-현황-한다그-보'). 괄호 부연을 떼어낸 뒤에도 이 검사가 걸리므로
    # 통짜로 스크램블된 이름이 '복구됨'으로 살아남지 않는다.
    if name.count("-") >= _MAX_HYPHEN_CHAIN and " " not in name:
        return "하이픈 연결 나열"
    if _IDENTIFIER_LIKE_RE.match(name):
        return "식별자 유출"
    if _DIGIT_IN_HANGUL_RE.search(name):
        return "한글 사이 숫자 혼입"
    if _HANGUL_THEN_LATIN_RE.search(name):
        return "한글 뒤 영문 혼입"
    return None


def _is_plausible_feature(name) -> bool:
    """EXAONE이 토큰 손상으로 뱉는 스크램블된 기능명을 걸러내는 보수적 결정론적 필터.
    (예: '기구-재물-현황-한다그-보(목록-밀동-2 엘드폰)' 같은 실제 관찰된 손상 사례)"""
    if not isinstance(name, str) or not name.strip():
        return False
    return _corruption_reason(name.strip()) is None


def repair_feature_name(name) -> str | None:
    """손상된 기능명을 살릴 수 있으면 살리고, 못 살리면 None.

    손상은 주로 괄호 안 부연 설명에서 발생한다
    (실측: '유통기한 등록 및 임박 알림 (푸0일 전 푸시 알림, 위젯 노출)').
    이럴 때 기능 자체를 버리면 그 기능이 PRD·DB·API에서 통째로 사라지므로,
    괄호 앞부분이 멀쩡하면 부연만 떼어내고 기능은 유지한다."""
    if not isinstance(name, str) or not name.strip():
        return None
    stripped = name.strip()
    reason = _corruption_reason(stripped)
    if reason is None:
        return stripped

    head = stripped.split("(", 1)[0].strip(" ,-–—")
    if len(head) >= _MIN_TRIMMED_NAME_LEN and _corruption_reason(head) is None:
        logger.warning("PM 기능명 부연 손상(%s) — 괄호 이후 제거: %r → %r", reason, stripped, head)
        return head

    logger.warning("PM 기능명 손상(%s) — 제외: %r", reason, stripped)
    return None

# EXAONE 모델 카드 권장 샘플링 파라미터
# https://huggingface.co/LGAI-EXAONE/K-EXAONE-236B-A23B
_TEMPERATURE = 1.0
_TOP_P = 0.95
_PRESENCE_PENALTY = 0.0

PM_PROMPT = """당신은 시니어 소프트웨어 아키텍트이자 PM입니다.
아래 요구사항을 분석하여 지정된 평문 라벨 형식으로만 응답하세요.
{skills_section}
응답 형식:
FEATURE: 기능명
PARENT: 사용자 원본 기능명 또는 공통
ORIGIN: USER 또는 DERIVED
SOURCE: problem, persona, workflow, data, permission, exception 중 하나 이상
RATIONALE: 이 기능이 필요한 구체적인 이유
PRIORITY: P0 또는 P1 또는 P2
ACTIONS: 사용자가 수행하거나 시스템이 보장할 행위
DATA: 저장·조회해야 할 데이터
RULES: 권한, 상태 전이, 중복 방지 등 업무 규칙
ERRORS: 검증 실패, 권한 없음, 대상 없음, 충돌 등 예외
ACCEPTANCE: 검증 가능한 완료 조건
(기능마다 위 블록을 반복)
DBA_INSTRUCTION: 필요한 테이블, 주요 관계, 중요 컬럼·제약조건을 두 문장 이상의 한 줄로 작성
API_INSTRUCTION: 기능별 API 그룹, 인증 방식, 특별한 계약 요구사항을 두 문장 이상의 한 줄로 작성
SELF_CHECK: PASS

규칙:
- 폼 기능 목록은 반드시 보존해야 하는 사용자 원본 기능이지 전체 기능의 상한이 아닙니다.
- 문제 정의, persona, 이상적 사용자 흐름, 데이터 생명주기, 인증·권한, 상태 전이, 실패·복구 흐름에서
  구현에 필요한 상위 기능과 독립적으로 검증 가능한 기능을 추가할 수 있습니다.
- 원본 기능은 ORIGIN: USER로, 추가 기능은 ORIGIN: DERIVED로 표시하고 PARENT와 SOURCE와 RATIONALE을 반드시 적으세요.
- 근거가 없는 추천, 결제, 소셜, AI, 관리자 기능이나 단순 화면 요소는 추가하지 마세요.
- 요구사항이 충분하면 전체 기능을 6~15개 수준으로 구체화하되, 숫자를 맞추기 위한 중복 기능은 만들지 마세요.
- 이번 입력의 최소 기능 수는 {min_feature_count}개입니다. 원본 기능이 3개 이상이면 문제·흐름·권한·예외에서
  최소 3개의 파생 기능을 추가하되, 각 파생 기능의 필요성을 독립적으로 설명할 수 있어야 합니다.
- 의미가 같은 기능의 표현만 다른 중복을 만들지 마세요.
- 기능명에는 내부 enum, 영문 런타임 문장, 템플릿 라벨, 구분자(|||)를 포함하지 마세요.
- dbaInstruction: 필요한 테이블명, 주요 관계(1:N, N:M), 중요 컬럼이나 제약조건 명시 (2문장 이상)
- apiInstruction: 기능별 API 그룹, 인증 방식, 특별한 설계 요구사항 명시 (2문장 이상)
- JSON·배열·마크다운·구분자는 사용하지 마세요.
- 생성 후 기능 범위·중복·오염 여부를 스스로 검토하고 마지막 줄을 SELF_CHECK: PASS로 끝내세요.

요구사항:
{context}
{form_features_hint}"""


_FEATURE_FIELDS = {
    "PARENT": "parentFeature", "ORIGIN": "origin", "SOURCE": "source",
    "RATIONALE": "rationale", "PRIORITY": "priority", "ACTIONS": "actions",
    "DATA": "dataRequirements", "RULES": "permissionRules", "ERRORS": "errorCases",
    "ACCEPTANCE": "acceptanceCriteria", "OWNERSHIP": "ownership",
    "STATES": "states", "TRANSITIONS": "stateTransitions", "TRANSACTIONS": "transactionRules",
}


DISCOVERY_PROMPT = """당신은 제품 요구사항을 기능 단위로 분해하는 PM입니다.
사용자가 직접 제시한 기능은 아래에 별도로 제공되므로 반복 출력하지 마세요.
문제, 사용자 여정, 데이터 생명주기, 권한, 상태 전이, 실패 복구 관점에서
독립적으로 구현하고 검증할 가치가 있는 파생 기능을 3~9개 발굴하세요.

각 기능은 정확히 다음 6줄만 사용하세요.
FEATURE: 짧고 명확한 기능명
PARENT: 원본 기능명 하나 또는 공통
ORIGIN: DERIVED
SOURCE: problem, persona, workflow, data, permission, exception 중 하나 이상
RATIONALE: 입력 근거와 이 기능이 필요한 이유를 구체적으로 설명
PRIORITY: P0 또는 P1 또는 P2

결제, 추천, 소셜, AI, 관리자 기능처럼 입력 근거가 없는 범위를 만들지 마세요.
화면 요소나 표현만 다른 중복 기능을 만들지 마세요.
JSON, 마크다운, 서문, 결론을 출력하지 마세요.

[원본 기능]
{original_features}

[제품 근거]
{context}
"""


DETAIL_PROMPT = """당신은 제품 기능 하나를 구현 가능한 명세로 구체화하는 PM입니다.
아래 기능의 범위를 벗어나지 말고 정확히 다음 9줄만 출력하세요.
ACTIONS: 사용자의 행동과 시스템 처리 2~4개
DATA: 저장하거나 검증해야 할 핵심 데이터 2~5개
RULES: 권한, 상태 전이, 중복 방지 등 업무 규칙 1~3개
ERRORS: 검증 실패, 권한 없음, 대상 없음, 충돌 중 해당되는 예외 1~3개
ACCEPTANCE: 테스트로 확인 가능한 완료 조건 2~4개
OWNERSHIP: PUBLIC, USER, SHARED 중 하나와 판단 근거
STATES: 저장 상태가 있으면 상태값, 없으면 NONE
TRANSITIONS: from → to: trigger 형식의 상태 전이, 없으면 NONE
TRANSACTIONS: 함께 성공하거나 롤백되어야 하는 변경과 멱등성·충돌 규칙 1~3개

각 줄의 항목은 세미콜론으로 구분하세요. JSON이나 마크다운은 출력하지 마세요.

기능명: {name}
상위 기능: {parent}
출처: {source}
필요 근거: {rationale}
기존 요구사항: {requirements}
제품 근거: {context}
"""


def _parse_feature_detail(raw: str) -> dict[str, list[str]]:
    result = {
        "actions": [], "dataRequirements": [], "permissionRules": [],
        "errorCases": [], "acceptanceCriteria": [], "ownership": [],
        "states": [], "stateTransitions": [], "transactionRules": [],
    }
    labels = {
        "ACTIONS": "actions", "DATA": "dataRequirements",
        "RULES": "permissionRules", "ERRORS": "errorCases",
        "ACCEPTANCE": "acceptanceCriteria",
        "OWNERSHIP": "ownership", "STATES": "states",
        "TRANSITIONS": "stateTransitions", "TRANSACTIONS": "transactionRules",
    }
    for raw_line in (raw or "").replace("\r", "").splitlines():
        line = raw_line.strip().strip("`*- ")
        if ":" not in line:
            continue
        label, value = line.split(":", 1)
        key = labels.get(label.strip().upper())
        if key:
            result[key] = _split_detail(value)
    return result


def _split_detail(value: str) -> list[str]:
    return [part.strip() for part in re.split(r"\s*[;,]\s*", str(value or "")) if part.strip()]


def _parse_plain_pm(raw: str, form_features: list[str]) -> dict:
    features: list[str] = []
    feature_specs: list[dict] = []
    current: dict | None = None
    dba_instruction = ""
    api_instruction = ""
    for raw_line in (raw or "").replace("\r", "").splitlines():
        line = raw_line.strip().strip("`*- ")
        upper = line.upper()
        if upper.startswith("FEATURE:"):
            if current:
                feature_specs.append(current)
            value = line.split(":", 1)[1].strip()
            if value:
                features.append(value)
                current = {"name": value}
        elif any(upper.startswith(label + ":") for label in _FEATURE_FIELDS):
            label, value = line.split(":", 1)
            key = _FEATURE_FIELDS[label.strip().upper()]
            if current is not None:
                value = value.strip()
                current[key] = (
                    _split_detail(value)
                    if key in {
                        "source", "actions", "dataRequirements", "permissionRules", "errorCases",
                        "acceptanceCriteria", "ownership", "states", "stateTransitions", "transactionRules",
                    }
                    else value
                )
        elif upper.startswith("DBA_INSTRUCTION:"):
            dba_instruction = line.split(":", 1)[1].strip()
        elif upper.startswith("API_INSTRUCTION:"):
            api_instruction = line.split(":", 1)[1].strip()
    if current:
        feature_specs.append(current)
    feature_summary = ", ".join(form_features or features) or "승인된 기능"
    if len(dba_instruction) < 30:
        dba_instruction = (
            f"{feature_summary} 구현에 필요한 사용자 소유 데이터 테이블과 기능별 테이블을 설계합니다. "
            "모든 외래키, 유일성 제약조건, 생성·수정 시각과 조회 인덱스를 명시합니다."
        )
    if len(api_instruction) < 30:
        api_instruction = (
            f"{feature_summary} 기능별 REST API를 제공합니다. 인증은 사용자 요구사항에 명시된 경우에만 적용하고, "
            "요청·응답 필드는 최종 ERD와 일치시키고 표준 4xx·5xx 오류 계약을 명시합니다."
        )
    return {
        "featureList": features,
        "featureSpecs": feature_specs,
        "dbaInstruction": dba_instruction,
        "apiInstruction": api_instruction,
        "selfCheck": "PASS" if self_check_passed(raw) else "FAIL",
    }


class PmAgent:

    def __init__(self, client: AsyncOpenAI, skills_loader: PmSkillsLoader, model: str = "gpt-4o"):
        self._client = client
        self._skills = skills_loader
        self._model = model

    async def execute(self, state: PipelineState, dump=None) -> PipelineState:
        logger.info("PM 시작: 기능 발굴과 기능별 상세화를 분리 실행")
        originals = dedupe_labels([
            name for value in (state.feature_list or [])
            if (name := repair_feature_name(value))
        ])
        context = self._planning_context(state)
        derived: list[dict] = []
        seen = set(originals)
        discovery_prompt = DISCOVERY_PROMPT.format(
            original_features="\n".join(f"- {name}" for name in originals) or "- 없음",
            context=context,
        )

        # 유효한 블록만 누적하므로 응답 후반부 하나가 깨져도 앞의 정상 기능은 보존된다.
        for attempt in range(3):
            try:
                raw = await self._complete(discovery_prompt, max_tokens=1400)
            except (InternalServerError, APITimeoutError, APIConnectionError) as exc:
                logger.warning("PM 기능 발굴 API 오류: %s", exc)
                break
            if dump:
                dump.log_raw("PM_DISCOVERY", attempt + 1, raw)
            parsed = _parse_plain_pm(raw, originals)
            for candidate in parsed.get("featureSpecs") or []:
                spec = self._valid_derived_spec(candidate, originals)
                if not spec or spec["name"] in seen:
                    continue
                seen.add(spec["name"])
                derived.append(spec)
            if len(derived) >= 3:
                break
            discovery_prompt += (
                "\n\n유효한 파생 기능이 아직 부족합니다. "
                f"확보된 이름({', '.join(seen)})과 겹치지 않는 기능을 추가하세요."
            )

        parsed_prd = try_parse_json(state.prd_document or "{}") or {}
        existing_features = {
            str(item.get("name") or ""): item
            for item in parsed_prd.get("coreFeatures", [])
            if isinstance(item, dict) and item.get("name")
        } if isinstance(parsed_prd, dict) else {}
        feature_specs = [self._original_spec(name, existing_features.get(name)) for name in originals]
        feature_specs.extend(derived)
        feature_specs = await asyncio.gather(*[
            self._enrich_feature(spec, existing_features.get(spec["name"]), context, dump, index)
            for index, spec in enumerate(feature_specs, 1)
        ])
        auth_required = requires_auth(
            state.prd_document, [spec["name"] for spec in feature_specs],
            f"{state.user_query} {state.context_prompt}",
        ) or any(feature_requires_user_scope(spec) for spec in feature_specs)
        if auth_required:
            existing_names = {str(spec.get("name") or "") for spec in feature_specs}
            feature_specs.extend(
                spec for spec in auth_feature_specs() if spec["name"] not in existing_names
            )
        feature_specs = [self._normalize_contract(spec, auth_required) for spec in feature_specs]
        feature_list = [spec["name"] for spec in feature_specs]
        dba_instruction = (
            "최종 기능 카탈로그 전체에 필요한 엔터티, 소유 관계, 상태 이력과 기능 간 참조를 설계하고 "
            "각 테이블의 PK·FK·유일성·필수값·삭제 정책 및 조회 인덱스를 명시합니다."
        )
        api_instruction = (
            "최종 기능 카탈로그의 사용자 행동과 완료 조건을 API 작업으로 매핑하고 인증·권한, "
            "요청·응답 필드, 상태 코드, 오류 계약과 멱등성·트랜잭션 경계를 명시합니다."
        )
        prd_document = self._merge_feature_specs(state.prd_document, feature_specs, originals)
        logger.info("PM 기능 카탈로그: 원본 %d개 + 파생 %d개", len(originals), len(derived))
        return state.copy(
            feature_list=feature_list,
            feature_specs=feature_specs,
            prd_document=prd_document,
            dba_instruction=dba_instruction,
            api_instruction=api_instruction,
            status_message="PM 에이전트 완료 — 기능 카탈로그 상세화",
        )

    async def _complete(self, prompt: str, max_tokens: int) -> str:
        async with llm_slot():
            response = await self._client.chat.completions.create(
                model=self._model,
                temperature=0.7,
                top_p=_TOP_P,
                presence_penalty=_PRESENCE_PENALTY,
                frequency_penalty=0.3,
                max_tokens=max_tokens,
                messages=[
                    {"role": "system", "content": "요청한 평문 라벨과 내용만 출력하세요."},
                    {"role": "user", "content": prompt},
                ],
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
        return response.choices[0].message.content or ""

    @staticmethod
    def _valid_derived_spec(candidate: dict, originals: list[str]) -> dict | None:
        if not isinstance(candidate, dict):
            return None
        name = repair_feature_name(candidate.get("name"))
        parent = str(candidate.get("parentFeature") or "").strip()
        rationale = str(candidate.get("rationale") or "").strip()
        source = [str(value).strip().lower() for value in candidate.get("source") or [] if str(value).strip()]
        allowed_sources = {"problem", "persona", "workflow", "data", "permission", "exception"}
        raw_origin = str(candidate.get("origin") or "").strip()
        # 발굴 프롬프트는 파생 기능만 요청한다. 모델이 SOURCE 값을 ORIGIN 줄에
        # 잘못 배치한 경우 허용된 열거값에 한해 출처로 복구한다.
        if raw_origin.lower() in allowed_sources:
            source = dedupe_labels([*source, raw_origin.lower()])
            origin = "DERIVED"
        else:
            origin = raw_origin.upper()
        if (
            not name or name in originals
            or origin != "DERIVED"
            or parent not in set(originals) | {"공통", "ROOT", "root"}
            or len(rationale) < 12
            or not set(source) & allowed_sources
        ):
            return None
        priority = str(candidate.get("priority") or "P1").upper()
        return {
            "name": name, "parentFeature": parent, "origin": "DERIVED",
            "source": source, "rationale": rationale,
            "priority": priority if priority in {"P0", "P1", "P2"} else "P1",
        }

    @staticmethod
    def _original_spec(name: str, existing: dict | None) -> dict:
        existing = existing or {}
        return {
            "name": name, "parentFeature": "", "origin": "USER", "priority": "P0",
            "source": ["problem", "workflow"],
            "rationale": str(existing.get("description") or "").strip()
                         or f"사용자가 직접 요청한 핵심 기능: {name}",
        }

    async def _enrich_feature(
        self, spec: dict, existing: dict | None, context: str, dump, index: int,
    ) -> dict:
        existing = existing or {}
        requirements = [str(value) for value in existing.get("requirements") or [] if str(value).strip()]
        prompt = DETAIL_PROMPT.format(
            name=spec["name"], parent=spec.get("parentFeature") or "없음",
            source=", ".join(spec.get("source") or []), rationale=spec.get("rationale") or "",
            requirements="; ".join(requirements) or "없음", context=context[:2800],
        )
        detail = {}
        try:
            raw = await self._complete(prompt, max_tokens=750)
            if dump:
                dump.log_raw(f"PM_DETAIL_{index}", 1, raw)
            detail = _parse_feature_detail(raw)
        except (InternalServerError, APITimeoutError, APIConnectionError) as exc:
            logger.warning("PM 기능 상세화 API 오류(%s): %s", spec["name"], exc)
        enriched = dict(spec)
        for key, fallback in self._fallback_detail(spec["name"], requirements).items():
            generated = [str(value) for value in detail.get(key) or [] if str(value).strip()]
            enriched[key] = generated if generated else fallback
        return enriched

    @staticmethod
    def _fallback_detail(name: str, requirements: list[str]) -> dict[str, list[str]]:
        return {
            "actions": requirements[:4] or [f"{name} 요청을 처리하고 처리 결과를 조회한다"],
            "dataRequirements": [f"{name} 처리 입력값", "처리 상태", "생성·수정 시각"],
            "permissionRules": ["제품 요구에 명시된 공개·보호 범위와 업무 규칙을 적용한다"],
            "errorCases": ["필수 입력값 누락", "허용 범위 위반", "대상 데이터 없음 또는 상태 충돌"],
            "acceptanceCriteria": [
                f"유효한 입력으로 {name} 처리가 완료되고 결과를 조회할 수 있다",
                "잘못된 입력이나 권한 없는 요청은 데이터 변경 없이 명시적 오류로 종료된다",
            ],
            "ownership": [], "states": [], "stateTransitions": [],
            "transactionRules": [f"{name}의 연관 변경은 한 트랜잭션으로 처리하고 실패 시 롤백한다"],
        }

    @staticmethod
    def _normalize_contract(spec: dict, auth_required: bool) -> dict:
        normalized = dict(spec)
        raw_ownership = normalized.get("ownership")
        if isinstance(raw_ownership, dict):
            scope = str(raw_ownership.get("scope") or "").upper()
        else:
            ownership_text = " ".join(str(value) for value in raw_ownership or []).upper()
            scope = next((value for value in ("USER", "SHARED", "PUBLIC") if value in ownership_text), "")
        if feature_requires_user_scope(normalized):
            scope = "USER"
        if scope not in {"USER", "SHARED", "PUBLIC", "SYSTEM"}:
            scope = "USER" if auth_required or feature_requires_user_scope(normalized) else "PUBLIC"
        normalized["ownership"] = {
            "scope": scope,
            "ownerEntity": "users" if scope in {"USER", "SHARED"} else "",
            "ownerKey": "user_id" if scope in {"USER", "SHARED"} else "",
            "access": (
                "인증 주체의 소유권 또는 명시된 역할을 검증"
                if scope in {"USER", "SHARED"} else "인증 없이 허용된 공개 범위"
            ),
        }
        for key in ("states", "stateTransitions", "transactionRules"):
            values = [
                str(value).strip() for value in normalized.get(key) or []
                if str(value).strip() and str(value).strip().upper() != "NONE"
            ]
            normalized[key] = list(dict.fromkeys(values))
        if normalized["states"] and not normalized["stateTransitions"]:
            normalized["stateTransitions"] = [
                "현재 상태 → 요청 상태: 기능별 선행 조건과 권한 검증 성공"
            ]
        if not normalized["transactionRules"]:
            normalized["transactionRules"] = [
                f"{normalized.get('name', '기능')}의 연관 데이터 변경은 한 트랜잭션으로 처리하고 실패 시 롤백한다"
            ]
        return normalized

    async def _execute_legacy(self, state: PipelineState, dump=None) -> PipelineState:
        logger.info("PM 에이전트 시작")

        if self._skills.has_skills():
            skills_section = await self._skills.find_relevant_skills(state.context_prompt, 5)
            logger.info("PM 스킬 주입 완료")
        else:
            skills_section = ""
            logger.warning("PM 스킬 미적용")

        # 폼에서 추출된 기능 목록을 프롬프트에 명시하여 PM 에이전트가 빠뜨리지 않도록 힌트 제공
        form_features = state.feature_list or []
        form_features_hint = (
            "\n\n[사용자 원본 기능 — 반드시 USER 기능으로 보존하고, 필요한 기능을 근거와 함께 확장하세요]\n"
            + "\n".join(f"- {f}" for f in form_features)
        ) if form_features else ""

        min_feature_count = len(form_features) + (3 if len(form_features) >= 3 else 0)
        prompt = PM_PROMPT.format(
            skills_section=(skills_section or "")[:1800],
            context=self._planning_context(state),
            form_features_hint=form_features_hint,
            min_feature_count=min_feature_count,
        )

        data = None
        retry_reasons: list[str] = []
        for attempt in range(3):
            try:
                response = await self._client.chat.completions.create(
                    model=self._model,
                    temperature=_TEMPERATURE,
                    top_p=_TOP_P,
                    presence_penalty=_PRESENCE_PENALTY,
                    frequency_penalty=0.3,
                    max_tokens=3500,
                    messages=[
                        {"role": "system", "content": "지정된 평문 라벨만 출력하세요. JSON·배열·마크다운·인사말은 금지합니다."},
                        {"role": "user", "content": retry_prompt(prompt, retry_reasons, attempt)},
                    ],
                    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                )
            except (InternalServerError, APITimeoutError, APIConnectionError) as e:
                logger.warning("PM API 오류: %s — 결정론적 지시로 대체", e)
                break
            raw = response.choices[0].message.content or ""
            if dump:
                dump.log_raw("PM", attempt + 1, raw)
            data = _parse_plain_pm(raw, form_features)
            retry_reasons = self._quality_issues(data, raw, form_features)
            if not retry_reasons:
                break
            logger.warning("PM 검증 실패 (attempt %d) — %s", attempt + 1, retry_reasons)

        if retry_reasons:
            logger.error("PM 최종 검증 실패 — 검증되지 않은 생성 결과 폐기")
            data = None

        feature_list, dba_instruction, api_instruction = self._extract(data)
        feature_specs = list(data.get("featureSpecs") or []) if isinstance(data, dict) else []

        if feature_list:
            # 손상된 이름은 먼저 복구를 시도하고(대개 괄호 안 부연만 깨져 있다),
            # 복구 불가능한 것만 제외한다 — 통째로 버리면 그 기능이 산출물 전체에서 사라진다
            repaired, dropped, seen = [], 0, set()
            for f in feature_list:
                name = repair_feature_name(f)
                if name is None:
                    dropped += 1
                    continue
                if name in seen:  # 부연 제거로 앞 항목과 같아질 수 있다
                    continue
                seen.add(name)
                repaired.append(name)
            if dropped or len(repaired) != len(feature_list):
                logger.warning(
                    "PM featureList 정리 — %d개 → %d개 (복구 불가 %d개 제외)",
                    len(feature_list), len(repaired), dropped,
                )
            feature_list = dedupe_labels(repaired)

        # 사용자 원본은 반드시 남기되, 근거가 검증된 파생 기능은 보존한다.
        if form_features:
            authoritative = [name for f in form_features if (name := repair_feature_name(f))]
            authoritative = dedupe_labels(authoritative)
            generated = set(feature_list)
            feature_list = authoritative + [name for name in feature_list if name not in set(authoritative)]
            by_name = {
                str(spec.get("name") or ""): dict(spec)
                for spec in feature_specs if isinstance(spec, dict) and spec.get("name")
            }
            for original in authoritative:
                spec = by_name.setdefault(original, {"name": original})
                spec.update({"origin": "USER", "parentFeature": "", "priority": "P0"})
                spec.setdefault("source", ["problem", "workflow"])
                spec.setdefault("rationale", f"사용자가 직접 요청한 핵심 기능: {original}")
            feature_specs = [by_name[name] for name in feature_list if name in by_name]
            if generated - set(authoritative):
                logger.info("PM 근거 기반 기능 확장 — 원본 %d개 + 파생 %d개", len(authoritative), len(generated - set(authoritative)))

        # 파싱 완전 실패 또는 전량 필터링 시 기존 state의 feature_list 유지
        if not feature_list:
            feature_list = state.feature_list or []
            logger.warning("PM feature_list 도출 실패 — 기존 feature_list 유지 (%d개)", len(feature_list))
        if not feature_specs:
            feature_specs = [
                {
                    "name": name, "origin": "USER", "parentFeature": "", "priority": "P0",
                    "source": ["problem", "workflow"],
                    "rationale": f"사용자가 직접 요청한 핵심 기능: {name}",
                    "actions": [], "dataRequirements": [], "permissionRules": [],
                    "errorCases": [], "acceptanceCriteria": [],
                }
                for name in feature_list
            ]

        prd_document = self._merge_feature_specs(
            state.prd_document, feature_specs, form_features,
        )

        logger.info("PM 에이전트 완료 — %d 기능 도출", len(feature_list))
        return state.copy(
            feature_list=feature_list,
            feature_specs=feature_specs,
            prd_document=prd_document,
            dba_instruction=dba_instruction or "DB 설계 진행",
            api_instruction=api_instruction or "REST API 설계 진행",
            status_message="PM 에이전트 완료 — 기능 목록 도출",
        )

    def _extract(self, data) -> tuple[list[str], str, str]:
        if data and isinstance(data, dict):
            return (
                data.get("featureList", []),
                data.get("dbaInstruction", ""),
                data.get("apiInstruction", ""),
            )
        return ([], "", "")

    @staticmethod
    def _planning_context(state: PipelineState) -> str:
        """Keep feature planning focused on product evidence, not release/KPI noise."""
        prd = try_parse_json(state.prd_document or "{}") or {}
        if not isinstance(prd, dict):
            prd = {}
        compact = {
            "projectOverview": prd.get("projectOverview"),
            "background": prd.get("background"),
            "coreFeatures": prd.get("coreFeatures"),
            "userPersonas": prd.get("userPersonas"),
            "mvpScope": prd.get("mvpScope"),
        }
        return (
            "[사용자 요구와 Phase1 근거]\n"
            f"{(state.user_query or state.context_prompt or '')[:4500]}\n\n"
            "[PRD 제품 근거]\n"
            f"{json.dumps(compact, ensure_ascii=False)[:6500]}"
        )

    @staticmethod
    def _merge_feature_specs(prd_document: str, specs: list[dict], original_features: list[str]) -> str:
        prd = try_parse_json(prd_document or "{}") or {}
        if not isinstance(prd, dict):
            prd = {}
        existing = {
            str(item.get("name") or ""): dict(item)
            for item in prd.get("coreFeatures") or []
            if isinstance(item, dict) and item.get("name")
        }
        original_set = set(original_features or [])
        merged = []
        for spec in specs:
            if not isinstance(spec, dict) or not spec.get("name"):
                continue
            name = str(spec["name"])
            item = existing.get(name, {})
            origin = "USER" if name in original_set else str(spec.get("origin") or "DERIVED").upper()
            priority = str(spec.get("priority") or ("P0" if origin == "USER" else "P1")).upper()
            actions = [str(x) for x in spec.get("actions") or [] if str(x).strip()]
            rules = [str(x) for x in spec.get("permissionRules") or [] if str(x).strip()]
            errors = [str(x) for x in spec.get("errorCases") or [] if str(x).strip()]
            acceptance = [str(x) for x in spec.get("acceptanceCriteria") or [] if str(x).strip()]
            requirements = list(dict.fromkeys([
                *[str(x) for x in item.get("requirements") or [] if str(x).strip()],
                *actions, *rules, *errors,
            ]))
            item.update({
                "name": name,
                "description": str(
                    item.get("description")
                    or f"{name}: {spec.get('rationale') or '요구사항 흐름을 완성하기 위해 필요한 기능'}"
                ),
                "priority": priority if priority in {"P0", "P1", "P2"} else "P1",
                "requirements": requirements,
                "parentFeature": str(spec.get("parentFeature") or ""),
                "origin": origin,
                "source": [str(x) for x in spec.get("source") or []],
                "rationale": str(spec.get("rationale") or ""),
                "actions": actions,
                "dataRequirements": [str(x) for x in spec.get("dataRequirements") or []],
                "permissionRules": rules,
                "ownership": spec.get("ownership") or {},
                "states": [str(x) for x in spec.get("states") or []],
                "stateTransitions": [str(x) for x in spec.get("stateTransitions") or []],
                "transactionRules": [str(x) for x in spec.get("transactionRules") or []],
                "errorCases": errors,
                "acceptanceCriteria": acceptance,
            })
            merged.append(item)
        prd["coreFeatures"] = merged
        scope = prd.get("mvpScope") if isinstance(prd.get("mvpScope"), dict) else {}
        scope["included"] = [item["name"] for item in merged if item.get("priority") == "P0"]
        scope["excluded"] = [item["name"] for item in merged if item.get("priority") in {"P1", "P2"}]
        scope["rationale"] = (
            "사용자 원본 기능과 출시 필수 의존 기능은 P0에 포함하고, "
            "근거가 있으나 후속 검증 가능한 기능은 P1/P2로 분리합니다."
        )
        prd["mvpScope"] = scope
        return json.dumps(prd, ensure_ascii=False)

    @staticmethod
    def _quality_issues(data, raw: str, form_features: list[str]) -> list[str]:
        if not isinstance(data, dict) or not isinstance(data.get("featureList"), list):
            return ["FEATURE 라벨 파싱 실패"]
        issues = contamination_reasons(data)
        features = [str(x).strip() for x in data.get("featureList") or [] if str(x).strip()]
        if len(dedupe_labels(features)) != len(features):
            issues.append("의미상 중복 기능 존재")
        invalid = [f for f in features if not _is_plausible_feature(f)]
        if invalid:
            issues.append(f"오염되거나 비정상적인 기능명: {invalid}")
        specs = [x for x in data.get("featureSpecs") or [] if isinstance(x, dict)]
        by_name = {str(x.get("name") or ""): x for x in specs}
        missing_originals = [name for name in form_features if name not in features]
        if missing_originals:
            issues.append(f"사용자 원본 기능 누락: {missing_originals}")
        if len(form_features) >= 3 and len(features) < len(dedupe_labels(form_features)) + 3:
            issues.append("문제·흐름·권한·예외에서 도출한 기능이 최소 3개보다 적음")
        valid_parents = set(features) | {"공통", "ROOT", "root"}
        for feature in features:
            spec = by_name.get(feature) or {}
            if feature not in form_features:
                if str(spec.get("origin") or "").upper() != "DERIVED":
                    issues.append(f"파생 기능 origin 누락: {feature}")
                if not str(spec.get("parentFeature") or "").strip():
                    issues.append(f"파생 기능 parentFeature 누락: {feature}")
                elif str(spec.get("parentFeature")) not in valid_parents:
                    issues.append(f"파생 기능 parentFeature 불일치: {feature}")
                if len(str(spec.get("rationale") or "").strip()) < 12:
                    issues.append(f"파생 기능 근거 부족: {feature}")
            if not (spec.get("actions") or spec.get("acceptanceCriteria")):
                issues.append(f"기능 행위·수용 기준 부족: {feature}")
        if len(str(data.get("dbaInstruction") or "")) < 30:
            issues.append("DBA_INSTRUCTION 내용 부족")
        if len(str(data.get("apiInstruction") or "")) < 30:
            issues.append("API_INSTRUCTION 내용 부족")
        return issues
