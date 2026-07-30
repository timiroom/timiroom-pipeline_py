import asyncio
import json
import logging
import re
import unicodedata

from fastapi import APIRouter
from openai import InternalServerError, APITimeoutError, APIConnectionError
from pydantic import BaseModel, Field

from common.api_response import ok
from phase1.models import FormData, MoSCoW, PlatformType
from phase2.json_utils import try_parse_json

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/chat", tags=["chat"])

# EXAONE 모델 카드 권장 샘플링 파라미터
# https://huggingface.co/LGAI-EXAONE/K-EXAONE-236B-A23B
_TEMPERATURE = 1.0
_TOP_P = 0.95
_PRESENCE_PENALTY = 0.0


def _exaone_endpoint_id() -> str:
    from main import settings
    return settings.exaone_endpoint_id


# ── 수집 항목 정의 ───────────────────────────────────────────────────
# 질문 순서·프롬프트 가이드·결정론적 fallback의 단일 출처.
# 이전에는 questions 리스트가 세 곳(_build_context_string, _generate_dynamic_suggestions,
# _static_fallback)에 중복 정의돼 있어 인덱스가 어긋날 여지가 있었다.
#
# fallback_question은 프롬프트 어디에도 등장하지 않는 문장이어야 한다 —
# 프롬프트에 실린 문장은 EXAONE이 그대로 복사(에코)하므로 검증 대상이지 대체재가 될 수 없다.
_QUESTIONS = [
    {
        "label": "플랫폼",
        "guideline": "WEB / APP / WEB_APP 중 하나를 고르게 하는 질문.",
        "suggestion_format": "세 플랫폼을 각각 하나씩 제시하고 '~으로 만들고 싶어요' 형식으로 끝낸다.",
        "fallback_question": "어떤 형태로 만들고 싶으신가요? 웹, 모바일 앱, 아니면 둘 다 필요하신가요?",
        "fallback_suggestions": [
            "웹사이트(WEB)로 만들고 싶어요",
            "모바일 앱(APP)으로 만들고 싶어요",
            "웹과 앱 둘 다(WEB_APP) 필요해요",
        ],
    },
    {
        "label": "현재 불편한 점",
        "guideline": (
            "사용자가 실제로 겪는 문제·불편을 구체적으로 끌어내는 질문. "
            "해결책이나 기능을 먼저 제시하지 말 것."
        ),
        "suggestion_format": "'~해서 번거로워요', '~하기가 어려워요'처럼 상황이 드러나는 문장.",
        "fallback_question": "지금 그 일을 하면서 가장 번거롭거나 답답한 순간은 언제인가요?",
        "fallback_suggestions": [
            "매번 직접 하나씩 확인해야 해서 번거로워요",
            "정보가 여기저기 흩어져 있어 찾기가 어려워요",
            "같은 작업을 반복하느라 시간이 너무 오래 걸려요",
        ],
    },
    {
        "label": "현재 해결 방법",
        "guideline": (
            "그 불편을 지금은 어떤 도구·습관으로 버티고 있는지 묻는 질문. "
            "또 다른 불편함이나 이상적인 상태를 묻지 말 것."
        ),
        "suggestion_format": "'~를 사용해요', '~로 관리해요'처럼 현재 수단이 드러나는 문장.",
        "fallback_question": "그 상황을 지금은 어떤 방법이나 도구로 버티고 계신가요?",
        "fallback_suggestions": [
            "엑셀이나 스프레드시트로 직접 관리해요",
            "메모 앱에 그때그때 기록해 둬요",
            "따로 정리하지 않고 그냥 기억에 의존해요",
        ],
    },
    {
        "label": "원하는 이상적 상태",
        "guideline": "문제가 해결됐을 때의 모습을 묘사하게 하는 질문. 불편함이나 현재 방법을 다시 묻지 말 것.",
        "suggestion_format": "'~가 자동으로 되면 좋겠어요', '~가 한눈에 보이면 좋겠어요' 형식.",
        "fallback_question": "이 문제가 완전히 해결된다면, 하루가 어떻게 달라져 있으면 좋겠나요?",
        "fallback_suggestions": [
            "한 화면에서 전부 한눈에 보이면 좋겠어요",
            "반복 작업이 자동으로 처리되면 좋겠어요",
            "필요한 시점에 알림으로 알려주면 좋겠어요",
        ],
    },
    {
        "label": "타겟 유저",
        "guideline": "이 서비스를 실제로 쓸 사람이 누구인지 묻는 질문.",
        "suggestion_format": "'20-30대 자취생'처럼 연령·직업·상황이 드러나는 구체적인 집단.",
        "fallback_question": "이 서비스를 가장 반가워할 사람은 어떤 분들일까요?",
        "fallback_suggestions": [
            "혼자 사는 20-30대 1인 가구",
            "같은 업무를 매일 반복하는 직장인",
            "소규모로 가게나 사업을 운영하는 사장님",
        ],
    },
    {
        "label": "핵심 기능 3가지",
        "guideline": (
            "꼭 필요한 기능 3가지를 기능명 위주로 짧게 답하게 하는 질문. "
            "MUST/SHOULD 같은 우선순위 표기나 긴 설명은 요구하지 말 것."
        ),
        "suggestion_format": "'재료 추가', '유통기한 알림'처럼 짧은 기능명 형태.",
        "fallback_question": "이것만큼은 반드시 있어야 한다 싶은 기능 세 가지만 꼽아주신다면요?",
        "fallback_suggestions": [
            "회원가입과 로그인",
            "목록 조회 및 검색",
            "알림 받기",
        ],
    },
]

_QUESTION_COUNT = len(_QUESTIONS)


def _collection_guide(q_idx: int) -> str:
    """전체 수집 순서를 보여주되 '지금 물을 항목'을 명시 — 모델이 단계를 스스로 세다
    질문을 건너뛰는 것을 막는다 (관측된 결함: 6번 핵심기능 질문이 통째로 누락)."""
    lines = []
    for i, q in enumerate(_QUESTIONS):
        mark = "◀ 지금 물어볼 항목" if i == q_idx else ("완료" if i < q_idx else "")
        lines.append(f"{i + 1}. {q['label']} {mark}".rstrip())
    return "\n".join(lines)


# ── 수집 프롬프트: 6가지 질문 (프로젝트명 제외) ───────────────────────
COLLECTION_PROMPT_TEMPLATE = """## 출력 형식 — 절대 규칙
JSON 하나만 출력하세요. 다른 텍스트 절대 금지.
키는 정확히 isComplete(false), message(문자열), suggestions(문자열 3개 배열) 세 개입니다.

## 역할
스타트업 기획 인터뷰어 AI.
지금 물어볼 항목은 아래에 지정돼 있습니다. 그 항목 하나만 질문하세요.

## 수집 순서
{guide}

## 지금 물어볼 항목: {label}
{guideline}

## message 작성 규칙 (매우 중요!)
- 한 문장, 친근한 존댓말 질문. 물음표로 끝나며 그 뒤에 아무것도 붙이지 말 것
- 아래 'suggestions 형식'은 suggestions에만 적용됩니다. 그 형식 문구를 질문에 이어 붙이지 마세요
- **사용자가 앞서 말한 표현을 최소 한 번 그대로 인용해서** 질문에 녹일 것
  (예: 사용자가 "여행지를 계획하느라 30분"이라고 했다면 → "여행지 계획에 매일 30분씩 쓰신다고 하셨는데, ...")
- 아래 문장들은 누구에게나 쓸 수 있는 껍데기 질문이므로 **절대 그대로 쓰지 말 것**:
  "현재는 어떻게 해결하고 있나요?", "이렇게 되면 좋겠다는 것은?", "주로 누가 사용할까요?"
- 이 지시문에 적힌 예시 문구를 복사해서 message에 넣지 말 것

## suggestions 생성 규칙 (매우 중요!)
- 형식: {suggestion_format}
- 반드시 정확히 3개
- **지금 물어볼 항목({label})에만** 맞는 답변. 이전·다음 항목의 답변을 넣지 말 것
- 사용자가 이미 말한 맥락(아래 참고 정보)에 맞게 구체적으로. 서로 다른 방향 3가지
- 클릭하면 그대로 전송 가능한 완성된 문장
- "답변예시1", "핵심 기능 1" 같은 자리표시자 금지 — 실제 내용이어야 함

## 참고 정보 (이미 수집한 내용)
{context}
"""

# ── 합성 프롬프트 ───────────────────────────────────────────────────
SYNTHESIS_PROMPT = """대화를 분석해서 아래 형식의 JSON을 출력하세요.
JSON 하나만 출력하세요. 마크다운·설명 금지. {{ 로 시작해서 }} 로 끝납니다.

{{"projectName": "이름", "projectDescription": "설명 1-2문장", "platform": "WEB", "techStack": [], "problemDefinition": {{"currentPainPoint": "불편함", "currentSolution": "현재방법", "idealState": "이상적상태", "businessImpact": null, "motivation": null, "competitorGap": null}}, "targetUsers": [{{"persona": "유저설명", "usageEnvironment": "사용환경", "biggestPainPoint": "주요불편함"}}], "featureDefinition": {{"commonFeatures": [], "customFeatures": [{{"featureName": "기능명", "description": "기능설명", "priority": "MUST"}}]}}}}

규칙:
- platform: WEB, APP, WEB_APP 중 하나
- priority: MUST 또는 SHOULD로 자동 할당 (사용자가 순서대로 말한 기능이므로, 처음 3개는 우선순위를 자동으로 분배)
- projectName: 대화에서 유추 (사용자가 명시하지 않으면 기능과 불편함으로부터 추론)
- projectDescription: 현재 불편함과 이상적 상태로부터 생성
- 대화에서 언급 없는 필드는 합리적으로 추론해서 채우기
- 빈 문자열("") 금지"""


# ── EXAONE 프롬프트 에코 방어 ─────────────────────────────────────────
# EXAONE에는 프롬프트에 적힌 예시·자리표시자를 그대로 출력하는 실패 모드가 있다.
# (실측: message가 "질문 1문장", 그리고 질문 3개가 프롬프트 예시문과 문자 단위로 동일)
# 프롬프트 문구를 고쳐도 재발하므로 출력 쪽에 결정론적 필터를 둔다.

def _normalize(text) -> str:
    """비교용 정규화 — 유니코드 정규화 + 공백 축약 + 양끝 구두점 제거."""
    if not isinstance(text, str):
        return ""
    s = unicodedata.normalize("NFKC", text).strip().lower()
    return re.sub(r"\s+", " ", s).strip(" .!?~-…。")


_ECHO_PHRASES = {_normalize(p) for p in (
    # 응답 형식 예시에 쓰인 자리표시자
    "질문 1문장", "답변예시1", "답변예시2", "답변예시3",
    "완성된_답변1", "완성된_답변2", "완성된_답변3",
    "핵심 기능 1", "핵심 기능 2", "핵심 기능 3",
    "기능명", "기능설명", "유저설명", "사용환경", "주요불편함",
    "불편함", "현재방법", "이상적상태", "설명 1-2문장",
    "이름", "이름1", "이름2", "이름3",
    # 프롬프트에 예시로 실려 있어 그대로 복사되던 껍데기 질문들
    "문제/불편함이 구체적으로 뭔가요?",
    "현재는 어떻게 해결하고 있나요?",
    "이렇게 되면 좋겠다는 것은?",
    "주로 누가 사용할까요?",
    "꼭 필요한 기능 3가지를 말씀해 주세요",
)}

# "답변예시2", "핵심 기능 3", "이름1", "suggestion 2"처럼 번호만 붙은 자리표시자
_PLACEHOLDER_RE = re.compile(
    r"^(답변\s*예시|예시\s*답변|예시|답변|핵심\s*기능|기능|이름|항목|"
    r"suggestion|answer|item|option|placeholder)[\s_-]*\d*$",
    re.I,
)

_MIN_MESSAGE_LEN = 10
_MIN_SUGGESTION_LEN = 4


def _is_echo(text) -> bool:
    """프롬프트 리터럴·자리표시자를 그대로 뱉은 출력인지 판정."""
    norm = _normalize(text)
    if not norm:
        return True
    return norm in _ECHO_PHRASES or bool(_PLACEHOLDER_RE.match(norm))


# 질문 끝에 suggestions 형식 힌트가 딸려 나온 경우 (실측: "...만들고 싶어요? ~으로 만들고 싶어요.")
# — 프롬프트에서 항목을 분리해도 EXAONE이 이어 붙이므로 결정론적으로 잘라낸다
_TRAILING_FORMAT_HINT_RE = re.compile(r'\?\s*[~〜][^?]*$')


def _strip_format_hint(text: str) -> str:
    stripped = _TRAILING_FORMAT_HINT_RE.sub("?", text.strip())
    if stripped != text.strip():
        logger.info("질문 끝의 형식 힌트 제거: %r → %r", text.strip(), stripped)
    return stripped


def _is_valid_question(text) -> bool:
    return isinstance(text, str) and len(text.strip()) >= _MIN_MESSAGE_LEN and not _is_echo(text)


def _clean_suggestions(raw) -> list[str]:
    """자리표시자·에코·중복·과도하게 짧은 항목을 제거하고 최대 3개로 정리."""
    if not isinstance(raw, list):
        return []
    out, seen = [], set()
    for item in raw:
        if not isinstance(item, str):
            continue
        s = item.strip()
        if len(s) < _MIN_SUGGESTION_LEN or _is_echo(s):
            continue
        key = _normalize(s)
        if key in seen:
            continue
        seen.add(key)
        out.append(s)
        if len(out) == 3:
            break
    return out


class ChatMessageDto(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: list[ChatMessageDto] = Field(min_length=1, description="메시지 목록이 비어있습니다")


async def _call_exaone(client, messages: list[dict], max_tokens: int) -> str:
    """EXAONE 호출 (재시도 3회)"""
    for attempt in range(3):
        try:
            resp = await client.chat.completions.create(
                model=_exaone_endpoint_id(),
                max_tokens=max_tokens,
                temperature=_TEMPERATURE,
                top_p=_TOP_P,
                presence_penalty=_PRESENCE_PENALTY,
                frequency_penalty=0.3,
                response_format={"type": "json_object"},
                messages=messages,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
            return resp.choices[0].message.content or ""
        except (InternalServerError, APITimeoutError, APIConnectionError) as e:
            logger.warning("EXAONE 채팅 일시 오류 (attempt %d): %s — 재시도", attempt + 1, e)
            if attempt < 2:
                await asyncio.sleep(3 * (attempt + 1))
            else:
                raise
    return ""


def _get_user_message_count(messages: list[ChatMessageDto]) -> int:
    """사용자 메시지 개수 반환"""
    return sum(1 for m in messages if m.role == "user")


def _question_index(messages: list[ChatMessageDto]) -> int:
    """지금 물어야 할 수집 항목의 인덱스. 첫 user 메시지는 초기 아이디어(수집 답변 아님)이므로
    -1 오프셋. 이 값 하나만 단계 판정의 기준으로 쓴다 — 이전에는 호출부마다 따로 계산해서
    질문이 통째로 건너뛰어지는 어긋남이 있었다."""
    return _get_user_message_count(messages) - 1


def _build_context_string(messages: list[ChatMessageDto]) -> str:
    """이미 수집한 내용을 문자열로 변환 (프로젝트명 제외)"""
    if _get_user_message_count(messages) == 0:
        return "(아직 수집한 정보 없음)"

    lines = []
    user_idx = 0

    for m in messages:
        if m.role == "user":
            q_idx = user_idx - 1  # 첫 메시지는 초기 아이디어, 컬렉션 답변 아님
            if q_idx < 0:
                lines.append(f"0. 처음 말한 아이디어: {m.content}")
            elif q_idx < _QUESTION_COUNT:
                lines.append(f"{q_idx + 1}. {_QUESTIONS[q_idx]['label']}: {m.content}")
            user_idx += 1

    return "\n".join(lines) if lines else "(아직 수집한 정보 없음)"


def _normalize_form_data(node: dict) -> dict:
    """EXAONE 출력의 흔한 포맷 오류를 FormData 스키마에 맞게 정규화."""
    # platform 정규화
    platform_map = {
        "앱": "APP", "모바일": "APP", "MOBILE": "APP",
        "웹": "WEB", "WEB ONLY": "WEB",
        "앱+웹": "WEB_APP", "웹+앱": "WEB_APP", "BOTH": "WEB_APP",
    }
    valid_platforms = {p.value for p in PlatformType}
    p = node.get("platform", "APP")
    if isinstance(p, str) and p not in valid_platforms:
        node["platform"] = platform_map.get(p, "APP")

    # featureDefinition 정규화
    fd = node.get("featureDefinition")
    if isinstance(fd, dict):
        # priority 정규화
        priority_map = {
            "필수": "MUST", "HIGH": "MUST", "높음": "MUST",
            "권장": "SHOULD", "MEDIUM": "SHOULD", "중간": "SHOULD",
            "선택": "COULD", "LOW": "COULD", "낮음": "COULD",
        }
        valid_priorities = {m.value for m in MoSCoW}
        for cf in fd.get("customFeatures") or []:
            if isinstance(cf, dict):
                pri = cf.get("priority", "MUST")
                if pri not in valid_priorities:
                    cf["priority"] = priority_map.get(pri, "MUST")

        # commonFeatures: 문자열 아이템 → 객체 변환
        common = fd.get("commonFeatures") or []
        if common and isinstance(common[0], str):
            fd["commonFeatures"] = [{"featureName": f} for f in common]

    # problemDefinition 필수 필드 보완
    pd = node.get("problemDefinition")
    if isinstance(pd, dict):
        if not pd.get("currentPainPoint"):
            pd["currentPainPoint"] = pd.get("painPoint") or pd.get("problem") or "정보 없음"
        if not pd.get("currentSolution"):
            pd["currentSolution"] = pd.get("solution") or "정보 없음"
        if not pd.get("idealState"):
            pd["idealState"] = pd.get("goal") or pd.get("ideal") or "정보 없음"

    # targetUsers 필수 필드 보완
    for tu in node.get("targetUsers") or []:
        if isinstance(tu, dict):
            if not tu.get("persona"):
                tu["persona"] = "일반 사용자"
            if not tu.get("usageEnvironment"):
                tu["usageEnvironment"] = tu.get("environment") or "일반적인 환경"
            if not tu.get("biggestPainPoint"):
                tu["biggestPainPoint"] = tu.get("painPoint") or "정보 없음"

    return node


async def _generate_dynamic_suggestions(question_idx: int, context: str, client) -> list[str]:
    """
    사용자 컨텍스트를 기반으로 동적 suggestions 생성
    """
    if question_idx < 0 or question_idx >= _QUESTION_COUNT:
        return []

    question = _QUESTIONS[question_idx]

    # 동적 suggestions 프롬프트 - 매우 명확하게
    dynamic_prompt = f"""사용자의 다음 질문에 대한 좋은 답변 예시 3개를 JSON으로 생성하세요.

현재까지 수집한 정보:
{context}

다음 질문 #{question_idx + 1}: {question['label']}

질문 가이드:
{question['guideline']}

답변 형식: {question['suggestion_format']}

답변 형식: {{"suggestions": ["...", "...", "..."]}}

중요한 규칙:
- 절대 이전 질문의 답변으로 혼동하지 말 것
- 위 '수집한 정보'에 나온 사용자의 실제 상황에 맞춘 구체적인 내용일 것
- 각 suggestion은 사용자가 그대로 입력할 수 있는 완성된 문장
- 반드시 정확히 3개, 서로 다른 방향으로
- 자리표시자(예: "답변1", "기능명") 금지 — 실제 내용이어야 함"""

    try:
        messages_for_suggestions = [
            {"role": "system", "content": "사용자의 맥락을 기반으로 각 질문에 정확히 맞는 suggestions을 생성합니다. JSON만 출력하세요."},
            {"role": "user", "content": dynamic_prompt}
        ]

        raw = await _call_exaone(client, messages_for_suggestions, max_tokens=500)
        node = try_parse_json(raw)

        if node and isinstance(node, dict):
            suggestions = _clean_suggestions(node.get("suggestions"))
            if len(suggestions) == 3:
                logger.info("동적 suggestions 생성 성공 (질문: %s)", question["label"])
                return suggestions
            logger.warning(
                "동적 suggestions 유효 항목 부족 (%d/3, 질문: %s)", len(suggestions), question["label"],
            )

    except Exception as e:
        logger.warning("동적 suggestions 생성 실패: %s", e)

    return []


async def _regenerate_question(
    client, collection_messages: list[dict], question_idx: int,
) -> tuple[str | None, list[str]]:
    """message가 프롬프트 예시의 에코로 판정됐을 때, 무엇이 잘못됐는지 명시해 한 번 더 생성.
    (프롬프트만으로는 안 잡히는 EXAONE 특성상 재요청도 실패할 수 있어 결과는 다시 검증한다)"""
    label = _QUESTIONS[question_idx]["label"]
    retry_messages = collection_messages + [{
        "role": "user",
        "content": (
            "방금 응답의 message가 지시문에 있던 예시 문장을 그대로 복사한 것이었습니다. "
            f"'{label}' 항목을, 사용자가 앞서 실제로 사용한 표현을 인용하면서 "
            "완전히 새로운 한 문장으로 다시 질문하세요. suggestions 3개도 함께 다시 만드세요. "
            "JSON만 출력하세요."
        ),
    }]
    try:
        raw = await _call_exaone(client, retry_messages, max_tokens=800)
        node = try_parse_json(raw)
        if not (node and isinstance(node, dict)):
            return None, []
        msg = node.get("message")
        cleaned = _strip_format_hint(msg) if isinstance(msg, str) else None
        return (cleaned if _is_valid_question(cleaned) else None), _clean_suggestions(node.get("suggestions"))
    except Exception as e:
        logger.warning("질문 재생성 실패: %s", e)
        return None, []


async def _generate_project_name_candidates(messages: list[ChatMessageDto], client) -> list[str]:
    """
    사용자의 답변을 분석해서 프로젝트 이름 3가지 후보 생성
    """
    lines = []
    for m in messages:
        prefix = "사용자" if m.role == "user" else "AI"
        lines.append(f"{prefix}: {m.content}")
    conversation = "\n".join(lines)
    
    name_prompt = f"""다음 대화를 분석해서 프로젝트의 핵심을 반영하는 프로젝트 이름 3가지를 생성하세요.

대화:
{conversation}

요구사항:
- 프로젝트의 핵심 가치나 문제 해결을 반영
- 기억하기 쉽고 멋있는 이름
- 한국어 또는 영어 모두 가능
- 실제로 사용할 수 있는 이름

응답 형식 (JSON만):
{{"names": ["이름1", "이름2", "이름3"], "reasoning": "왜 이 이름들을 제안했는지 간단한 설명"}}

예시:
- 불편함: "냉장고 뭐가 있는지 몰라서 낭비", 해결: "냉장고 재고 자동 추적"
  → "FridgeTracker", "냉동고", "FreshKeep"
"""

    try:
        name_msgs = [
            {"role": "system", "content": "사용자의 프로젝트 정보를 분석해서 창의적이고 의미있는 프로젝트 이름 3가지를 생성합니다. JSON만 출력하세요."},
            {"role": "user", "content": name_prompt}
        ]
        
        raw = await _call_exaone(client, name_msgs, max_tokens=500)
        node = try_parse_json(raw)
        
        if node and isinstance(node, dict):
            # "이름1" 같은 자리표시자 에코를 걸러낸 뒤 3개가 남을 때만 채택
            names = _clean_suggestions(node.get("names"))
            if len(names) == 3:
                logger.info("프로젝트 이름 후보 생성 성공: %s", names)
                return names
            logger.warning("프로젝트 이름 후보 유효 항목 부족 (%d/3)", len(names))
    
    except Exception as e:
        logger.warning("프로젝트 이름 생성 실패: %s", e)
    
    return []


async def _synthesize_form_data(messages: list[ChatMessageDto], client) -> dict | None:
    """대화 내용에서 중첩 FormData를 직접 합성 (최대 2회 시도)."""
    lines = []
    for m in messages:
        prefix = "사용자" if m.role == "user" else "AI"
        lines.append(f"{prefix}: {m.content}")
    conversation = "\n".join(lines)

    synthesis_msgs = [
        {"role": "system", "content": SYNTHESIS_PROMPT},
        {"role": "user", "content": f"대화:\n{conversation}"},
    ]

    for attempt in range(2):
        try:
            raw = await _call_exaone(client, synthesis_msgs, max_tokens=4000)
            logger.debug("Synthesis attempt %d raw (%.500s)", attempt + 1, raw)

            node = try_parse_json(raw)
            if node and isinstance(node, dict) and node.get("projectName") and node.get("featureDefinition"):
                node = _normalize_form_data(node)
                try:
                    FormData.model_validate(node)
                    logger.info("Synthesis OK (Pydantic 검증 통과) | attempt=%d", attempt + 1)
                    return node
                except Exception as ve:
                    logger.warning(
                        "Synthesis attempt %d: Pydantic 검증 실패 — %s | node keys: %s",
                        attempt + 1, ve, list(node.keys()),
                    )
                    continue

            logger.warning("Synthesis attempt %d: 파싱 실패 또는 필수 필드 없음", attempt + 1)
        except Exception as e:
            logger.error("합성 LLM 호출 실패 (attempt %d): %s", attempt + 1, e, exc_info=True)

    return None


async def _finish_collection(
    messages: list[ChatMessageDto], user_msg_count: int, client,
) -> dict:
    """6개 항목 수집 완료 후 단계 — 프로젝트명 후보 제시(1회) → FormData 합성."""
    logger.info("합성 단계 진입 | user_msgs=%d", user_msg_count)

    # 1단계: 프로젝트 이름 후보 제시 (수집 직후 1회)
    if user_msg_count == _QUESTION_COUNT + 1:
        project_name_candidates = await _generate_project_name_candidates(messages, client)

        if project_name_candidates:
            logger.info("프로젝트 이름 후보 제시 | candidates=%s", project_name_candidates)
            return ok({
                "message": "좋아요! 충분한 정보가 모였어요. 이 프로젝트에 어울리는 이름이 뭘까요?",
                "isComplete": False,
                "suggestions": project_name_candidates,
                "formData": None,
                "stage": "naming",
            })

        logger.warning("프로젝트 이름 생성 실패 — 바로 합성 진행")

    # 2단계: FormData 합성
    form_data = await _synthesize_form_data(messages, client)
    if form_data:
        # 이름 후보 제시 직후의 답변만 프로젝트명으로 채택
        # (그 이후는 합성 실패 후 추가로 수집한 보충 설명이므로 이름이 아님 — LLM 합성 결과를 신뢰)
        if user_msg_count == _QUESTION_COUNT + 2:
            last_user_msg = next((m.content for m in reversed(messages) if m.role == "user"), None)
            if last_user_msg:
                form_data["projectName"] = last_user_msg

        return ok({
            "message": "좋아요! 충분한 정보가 모였어요. 지금 바로 프로젝트를 시작할게요!",
            "isComplete": True,
            "suggestions": [],
            "formData": form_data,
            "stage": "complete",
        })

    # 합성 실패 — 수집 계속
    logger.warning("합성 실패 — 수집 계속 (user_msgs=%d)", user_msg_count)
    return ok({
        "message": "조금 더 자세히 알려주시면 더 잘 기획할 수 있어요. 핵심 기능이나 목표 유저에 대해 추가로 말씀해 주세요.",
        "isComplete": False,
        "suggestions": [
            "핵심 기능을 더 자세히 설명할게요",
            "타겟 유저를 더 좁혀서 말할게요",
            "참고할 만한 경쟁 서비스가 있어요",
        ],
        "formData": None,
    })


@router.post("/message")
async def message(req: ChatRequest) -> dict:
    from main import exaone_client

    user_msg_count = _get_user_message_count(req.messages)
    q_idx = _question_index(req.messages)
    context = _build_context_string(req.messages)
    logger.info("채팅 수집 단계 | user_msgs=%d, question_idx=%d", user_msg_count, q_idx)

    # 수집이 끝난 뒤(naming/synthesis 단계)에는 질문 항목이 없으므로 마지막 항목 기준으로 포맷만 맞춘다
    prompt_idx = min(max(q_idx, 0), _QUESTION_COUNT - 1)
    collection_prompt = COLLECTION_PROMPT_TEMPLATE.format(
        guide=_collection_guide(q_idx),
        label=_QUESTIONS[prompt_idx]["label"],
        guideline=_QUESTIONS[prompt_idx]["guideline"],
        suggestion_format=_QUESTIONS[prompt_idx]["suggestion_format"],
        context=context,
    )

    # assistant 메시지를 완전히 래핑 (suggestions 포함)
    collection_messages = [{"role": "system", "content": collection_prompt}]

    for m in req.messages:
        if m.role not in ("user", "assistant"):
            continue
        if m.role == "assistant":
            # assistant 메시지 완전 래핑
            try:
                existing = json.loads(m.content)
                wrapped = json.dumps(existing, ensure_ascii=False)
            except:
                # JSON 파싱 실패 시 그냥 메시지만 래핑
                wrapped = json.dumps(
                    {"isComplete": False, "message": m.content},
                    ensure_ascii=False,
                )
            collection_messages.append({"role": "assistant", "content": wrapped})
        else:
            collection_messages.append({"role": "user", "content": m.content})

    try:
        # 6개 항목을 모두 받았으면 질문 생성 없이 곧장 naming/synthesis로 —
        # 수집용 LLM 호출을 낭비하지 않고, 단계 판정 기준도 q_idx 하나로 통일된다
        if q_idx >= _QUESTION_COUNT:
            return await _finish_collection(req.messages, user_msg_count, exaone_client)

        raw = await _call_exaone(exaone_client, collection_messages, max_tokens=800)
        logger.debug("Collection raw (%.400s)", raw)

        node = try_parse_json(raw)
        if not (node and isinstance(node, dict)):
            logger.warning("JSON 파싱 실패 — 재생성으로 복구 시도 | raw: %.300s", raw)
            node = {}

        question = node.get("message")
        if isinstance(question, str):
            question = _strip_format_hint(question)
        suggestions = _clean_suggestions(
            node.get("suggestions")
            or node.get("sugations")   # EXAONE 오타 방어
            or node.get("suggestion")  # 단수형 방어
        )

        # 프롬프트 에코 방어 — 지시문의 예시 문장을 그대로 복사한 질문이면 이유를 명시해 재생성
        if not _is_valid_question(question):
            logger.warning("질문이 프롬프트 에코/무효 (idx=%d): %r — 재생성", q_idx, question)
            question, retry_suggestions = await _regenerate_question(
                exaone_client, collection_messages, q_idx,
            )
            if len(retry_suggestions) > len(suggestions):
                suggestions = retry_suggestions

        # suggestions 보강 1단계: 맥락 기반 동적 생성
        if len(suggestions) < 3:
            logger.warning("suggestions 부족 (%d/3, idx=%d) — 동적 생성 시도", len(suggestions), q_idx)
            dynamic = await _generate_dynamic_suggestions(q_idx, context, exaone_client)
            if len(dynamic) > len(suggestions):
                suggestions = dynamic

        # 2단계: 그래도 부족하면 실제로 생성된 항목은 살리고 나머지만 정적 항목으로 채운다
        if len(suggestions) < 3:
            logger.warning("suggestions 생성 실패 (%d/3, idx=%d) — 정적 fallback 보충", len(suggestions), q_idx)
            existing = {_normalize(s) for s in suggestions}
            for s in _QUESTIONS[q_idx]["fallback_suggestions"]:
                if len(suggestions) >= 3:
                    break
                if _normalize(s) not in existing:
                    suggestions.append(s)

        if not question:
            logger.warning("질문 재생성도 실패 — 결정론적 fallback 질문 사용 (idx=%d)", q_idx)
            question = _QUESTIONS[q_idx]["fallback_question"]

        return ok({
            "message": question,
            "isComplete": False,
            "suggestions": suggestions,
            "formData": None,
        })

    except Exception as e:
        logger.error("EXAONE 호출 실패: %s", e, exc_info=True)
        return ok({"message": "잠시 후 다시 시도해 주세요.", "isComplete": False, "suggestions": [], "formData": None})