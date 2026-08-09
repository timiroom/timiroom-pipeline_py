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
        "suggestion_format": (
            "각 suggestion 하나에 서로 다른 핵심 기능 3개를 쉼표로 묶는다. "
            "예: '일정 등록, 우선순위 설정, 마감 알림'."
        ),
        "fallback_question": "이것만큼은 반드시 있어야 한다 싶은 기능 세 가지만 꼽아주신다면요?",
        "fallback_suggestions": [
            "항목 등록 및 수정, 목록 조회 및 검색, 진행 상태 관리",
            "분류 및 필터, 우선순위 설정, 마감 일정 관리",
            "공유 및 협업, 변경 알림, 활동 이력 조회",
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
JSON과 마크다운은 출력하지 마세요. 아래 라벨 평문만 출력합니다.
MESSAGE: 질문 한 문장
SUGGESTION: 답변 예시 1
SUGGESTION: 답변 예시 2
SUGGESTION: 답변 예시 3

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

_UNFILLED_TEMPLATE_RE = re.compile(
    r"(^|[\s'\"(])(?:~|〜)\s*(?:가|이|을|를|은|는|로|에서|에게)?(?:\s|$)|"
    r"(?:무엇|어떤 것|뭔가|아무거나)\s*(?:이|가)?\s*(?:좋겠|필요)|"
    r"^(?:잘|좋게|편하게|자동으로|한눈에)\s*(?:되면|보이면)?\s*좋겠어요[.!]?$",
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


def _has_unfilled_template(text: str) -> bool:
    """사용자가 그대로 선택하면 의미가 비는 '~가 ...' 류 템플릿을 차단한다."""
    return not isinstance(text, str) or bool(_UNFILLED_TEMPLATE_RE.search(text.strip()))


# 질문 끝에 suggestions 형식 힌트가 딸려 나온 경우 (실측: "...만들고 싶어요? ~으로 만들고 싶어요.")
# — 프롬프트에서 항목을 분리해도 EXAONE이 이어 붙이므로 결정론적으로 잘라낸다
_TRAILING_FORMAT_HINT_RE = re.compile(r'\?\s*[~〜][^?]*$')


def _strip_format_hint(text: str) -> str:
    stripped = _TRAILING_FORMAT_HINT_RE.sub("?", text.strip())
    if stripped != text.strip():
        logger.info("질문 끝의 형식 힌트 제거: %r → %r", text.strip(), stripped)
    return stripped


def _is_valid_question(text) -> bool:
    return (
        isinstance(text, str)
        and len(text.strip()) >= _MIN_MESSAGE_LEN
        and text.strip().endswith("?")
        and not _is_echo(text)
    )


def _extract_question_from_raw(raw: str) -> str | None:
    """Recover a valid question when the model omitted the MESSAGE label."""
    try:
        parsed = json.loads((raw or "").strip().strip("`"))
    except (TypeError, ValueError, json.JSONDecodeError):
        parsed = None
    if isinstance(parsed, dict):
        for key in ("MESSAGE", "message", "question"):
            value = parsed.get(key)
            if isinstance(value, str) and _is_valid_question(_strip_format_hint(value)):
                return _strip_format_hint(value)
    for raw_line in (raw or "").replace("\r", "").splitlines():
        line = raw_line.strip().strip("`*- ")
        line = re.sub(r"^(?:MESSAGE|질문)\s*[:：]\s*", "", line, flags=re.I)
        line = _strip_format_hint(line)
        if _is_valid_question(line):
            return line
    return None


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


def _split_features(text: str) -> list[str]:
    """사용자 입력 한 문장에서 중복 없는 기능명을 추출한다."""
    out, seen = [], set()
    for value in re.split(r"[,，;/|\n·]+", text or ""):
        name = re.sub(r"^\s*\d+[.)]\s*", "", value).strip(" -•")
        key = _normalize(name)
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(name)
    return out


def _validate_stage_answer(stage: int, text: str) -> tuple[bool, str, str]:
    """형식은 Python에서 정규화하고, 의미가 비거나 담당 항목과 다른 답만 거절한다."""
    value = re.sub(r"\s+", " ", (text or "")).strip()
    if not value or _is_echo(value) or _has_unfilled_template(value):
        return False, value, "내용이 완성되지 않은 보기 또는 자리표시자입니다."

    if stage == 0:
        upper = value.upper()
        has_web = "WEB" in upper or "웹" in value or "브라우저" in value
        has_app = "APP" in upper or "앱" in value or "모바일" in value
        if not (has_web or has_app):
            return False, value, "웹, 모바일 앱, 또는 웹과 앱 모두 중 하나를 알려주세요."
    elif stage == 1:
        if len(value) < 10:
            return False, value, "실제로 불편한 상황을 한 문장으로 조금 더 구체적으로 알려주세요."
    elif stage == 2:
        if len(value) < 4:
            return False, value, "현재 사용하는 도구나 처리 방법을 알려주세요. 사용하지 않는다면 '아직 별도 도구가 없어요'라고 답할 수 있어요."
        if "?" in value or any(phrase in value for phrase in ("어떤 것을", "무엇을 사용", "알려주세요", "사용하나요")):
            return False, value, "답변 대신 질문이 들어왔습니다. 지금 쓰는 도구나 처리 방법을 문장으로 알려주세요."
        pain_words = ("걱정", "어려", "불편", "번거", "놓치", "막막", "혼란")
        solution_words = (
            "사용", "기록", "관리", "정리", "저장", "확인", "공유", "처리", "입력",
            "적어", "켜 두", "의존", "없", "안 하", "수동",
        )
        if any(word in value for word in pain_words) and not any(word in value for word in solution_words):
            return False, value, "불편함이 반복됐습니다. 지금 실제로 사용하는 도구나 처리 행동을 알려주세요."
    elif stage == 3:
        if len(value) < 10:
            return False, value, "문제가 해결됐을 때 무엇이 어떻게 달라지는지 완성된 문장으로 알려주세요."
    elif stage == 4:
        generic_personas = {"사용자", "사람", "사람들", "모두", "누구나", "일반 사용자"}
        if len(value) < 4 or _normalize(value) in generic_personas:
            return False, value, "연령, 직업, 역할 또는 사용 상황이 드러나는 구체적인 이용자를 알려주세요."
    elif stage == 5:
        features = _split_features(value)
        valid_features = [
            feature for feature in features
            if 2 <= len(feature) <= 50 and not _is_echo(feature) and not _has_unfilled_template(feature)
        ]
        if len(valid_features) < 3:
            return False, value, f"핵심 기능이 {len(valid_features)}개만 확인됐습니다. 서로 다른 기능을 쉼표로 구분해 3개 이상 알려주세요."
        value = ", ".join(valid_features)

    return True, value, ""


def _collect_interview_state(messages: list["ChatMessageDto"]) -> dict:
    """재시도 메시지를 포함한 대화에서 검증을 통과한 답만 순서대로 수집한다."""
    user_answers = [m.content.strip() for m in messages if m.role == "user" and m.content.strip()]
    if not user_answers:
        return {"idea": "", "answers": [], "stage": 0, "invalid": None, "extras": []}

    accepted: list[str] = []
    invalid = None
    extras: list[str] = []
    for value in user_answers[1:]:
        stage = len(accepted)
        if stage >= _QUESTION_COUNT:
            extras.append(value)
            continue
        valid, normalized, reason = _validate_stage_answer(stage, value)
        if valid:
            accepted.append(normalized)
            invalid = None
        else:
            invalid = {"stage": stage, "value": value, "reason": reason}

    return {
        "idea": user_answers[0],
        "answers": accepted,
        "stage": len(accepted),
        "invalid": invalid,
        "extras": extras,
    }


def _content_tokens(text: str) -> set[str]:
    stop = {
        "사용자", "서비스", "기능", "현재", "관련", "필요", "통해", "위해", "하고", "있는",
        "싶어요", "좋겠어요", "만들고", "문제", "상황", "정보", "합니다", "있어요",
    }
    out = set()
    for raw_token in re.findall(r"[가-힣A-Za-z0-9]{2,}", text or ""):
        token = raw_token.lower()
        if token in stop:
            continue
        out.add(token)
        stem = re.sub(r"(?:으로|에서|에게|까지|부터|처럼|보다|하고|하며|이랑|랑|의|이|가|은|는|을|를|과|와|도)$", "", token)
        if len(stem) >= 2 and stem not in stop:
            out.add(stem)
    return out


def _is_context_relevant(value: str, context: str) -> bool:
    context_tokens = _content_tokens(context)
    value_tokens = _content_tokens(value)
    return not context_tokens or bool(context_tokens & value_tokens)


def _clean_stage_suggestions(raw, stage: int, context: str = "") -> list[str]:
    """현재 단계에서 실제 답으로 채택 가능한 보기만 남긴다."""
    out = []
    for suggestion in _clean_suggestions(raw):
        valid, normalized, _ = _validate_stage_answer(stage, suggestion)
        relevant = stage not in (4, 5) or _is_context_relevant(normalized, context)
        if valid and relevant and _normalize(normalized) not in {_normalize(item) for item in out}:
            out.append(normalized)
    return out[:3]


def _contextual_fallback_suggestions(stage: int, state: dict) -> list[str]:
    """정적 예시가 다른 서비스의 Persona를 주입하지 않도록 사용자 원문으로 fallback을 만든다."""
    if stage not in (4, 5):
        return list(_QUESTIONS[stage]["fallback_suggestions"])

    if stage == 5:
        subject = _extract_service_subject(state)
        return [
            f"{subject} 등록, {subject} 목록 조회, {subject} 수정 및 삭제",
            f"{subject} 자동 분류, {subject} 우선순위 설정, {subject} 변경 알림",
            f"{subject} 검색 및 필터, {subject} 공유, {subject} 변경 이력 조회",
        ]

    source = f"{state.get('idea', '')} {' '.join(state.get('answers', []))}"
    roles = re.findall(
        r"(?:대학원생|대학생|재학생|학생|직장인|팀원|팀장|관리자|운영자|사장님|사업자|교사|강사|"
        r"학부모|개발자|디자이너|환자|보호자|고객|회원|주민|여행자|사용자)",
        source,
    )
    role = roles[0] if roles else "실사용자"
    collaborative = (
        bool(re.search(r"(?:^|\s)팀(?:\s|$)|팀원|팀장", source))
        or any(token in source for token in ("업무", "담당자", "프로젝트", "협업", "메신저"))
    )
    if collaborative:
        return [
            f"여러 대화방에서 담당 업무를 확인하는 {role if role != '실사용자' else '팀원'}",
            "여러 프로젝트의 담당자와 마감일을 관리하는 팀 리더",
            "팀의 업무 진행 상황을 조율하는 프로젝트 관리자",
        ]

    subject = _extract_service_subject(state)
    third_role = role if role != "실사용자" else "관리자"
    return [
        f"{subject} 정보를 자주 확인하고 직접 관리하는 {role}",
        f"여러 도구에 흩어진 {subject} 정보를 정리해야 하는 {role}",
        f"{subject} 변경 사항을 놓치지 않아야 하는 {third_role}",
    ]


def _extract_service_subject(state: dict) -> str:
    """기능명과 이름 fallback에 쓸 서비스의 핵심 관리 대상을 사용자 문장에서 추출한다."""
    idea = state.get("idea", "")
    answers = state.get("answers", [])
    ideal = answers[3] if len(answers) > 3 else ""
    sources = (ideal, idea)
    patterns = (
        r"^(.{2,32}?)(?:이|가)\s+(?:자동|한눈|통합|정리|표시|보이)",
        r"^(.{2,32}?)(?:을|를)\s+(?:자동|한눈|통합|정리|표시|모으|보이)",
        r"(?:이|가)\s+(.{2,32}?)(?:을|를)\s+(?:함께\s+)?(?:관리|기록|예약|공유|추적|확인)",
    )
    for source in sources:
        for pattern in patterns:
            match = re.search(pattern, source)
            if match:
                subject = re.sub(r"\s+", " ", match.group(1)).strip(" '.,!?…")
                # "대화에서 할 일을 자동으로"처럼 출처가 앞에 붙은 경우에는
                # 관리 대상만 남긴다.
                subject = re.sub(r"^.*(?:에서|으로)\s+", "", subject).strip()
                if 2 <= len(subject) <= 32:
                    return subject
    return "핵심 항목"


def _fallback_project_name_candidates(state: dict) -> list[str]:
    subject = _extract_service_subject(state)
    stem_tokens = [
        token for token in re.findall(r"[가-힣A-Za-z0-9]+", subject)
        if token not in {"핵심", "항목", "정보", "관리"}
    ]
    stem = "".join(stem_tokens)[:10] or "프로젝트"
    return [f"{stem}메이트", f"{stem}플로우", f"{stem}허브"]


def _is_valid_project_name(text: str) -> bool:
    value = (text or "").strip()
    return (
        2 <= len(value) <= 80
        and not _is_echo(value)
        and not _has_unfilled_template(value)
        and "?" not in value
    )


def _is_valid_generated_project_name(text: str) -> bool:
    """모델이 만든 이름에서 문장·임의 조어처럼 보이는 후보를 제거한다."""
    value = (text or "").strip()
    if not _is_valid_project_name(value) or len(value) > 24 or " " in value:
        return False
    if re.fullmatch(r"[A-Za-z][A-Za-z0-9-]*", value):
        return True
    approved_suffixes = (
        "메이트", "플로우", "허브", "보드", "노트", "링크", "체크", "매니저", "톡", "업", "온",
    )
    return bool(re.fullmatch(r"[가-힣A-Za-z0-9]+", value)) and value.endswith(approved_suffixes)


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


def _parse_labeled_text(raw: str, repeated: set[str] | None = None) -> dict[str, str | list[str]]:
    """LLM 평문 라벨을 파싱한다. repeated 라벨은 배열로 누적한다."""
    repeated = {x.upper() for x in (repeated or set())}
    result: dict[str, str | list[str]] = {}
    current = None
    for raw_line in (raw or "").replace("\r", "").splitlines():
        line = raw_line.strip().strip("`*- ")
        if not line:
            continue
        match = re.match(r"^([A-Za-z_]+)\s*:\s*(.*)$", line)
        if match:
            key, value = match.group(1).upper(), match.group(2).strip()
            current = key
            if key in repeated:
                result.setdefault(key, [])
                if value:
                    result[key].append(value)
            else:
                result[key] = value
            continue
        if current and current not in repeated:
            result[current] = f"{result.get(current, '')} {line}".strip()
    return result


def _get_user_message_count(messages: list[ChatMessageDto]) -> int:
    """사용자 메시지 개수 반환"""
    return sum(1 for m in messages if m.role == "user")


def _question_index(messages: list[ChatMessageDto]) -> int:
    """메시지 개수가 아니라 의미 검증을 통과한 답변 개수로 현재 단계를 결정한다."""
    return _collect_interview_state(messages)["stage"]


def _build_context_string(messages: list[ChatMessageDto]) -> str:
    """이미 수집한 내용을 문자열로 변환 (프로젝트명 제외)"""
    state = _collect_interview_state(messages)
    if not state["idea"]:
        return "(아직 수집한 정보 없음)"

    lines = [f"0. 처음 말한 아이디어: {state['idea']}"]
    for idx, value in enumerate(state["answers"]):
        lines.append(f"{idx + 1}. {_QUESTIONS[idx]['label']}: {value}")

    return "\n".join(lines) if lines else "(아직 수집한 정보 없음)"


def _contextual_question(messages: list[ChatMessageDto], q_idx: int) -> str:
    """Build a stage-safe question while retaining the user's own wording."""
    state = _collect_interview_state(messages)
    answers = [state["idea"], *state["answers"]]

    def anchor(index: int, fallback: str) -> str:
        value = answers[index] if index < len(answers) else fallback
        value = re.sub(r"\s+", " ", value).strip(" \"'.,!?…")
        return value if len(value) <= 64 else value[:61].rstrip() + "…"

    idea = anchor(0, "말씀하신 서비스")
    templates = (
        f"‘{idea}’ 아이디어는 웹, 모바일 앱, 또는 둘 다 중 어떤 형태로 만들고 싶으신가요?",
        f"‘{idea}’ 서비스를 구상하실 때 지금 가장 번거롭거나 답답한 순간은 언제인가요?",
        f"‘{anchor(2, '그 불편한 상황')}’이라고 하셨는데, 지금은 어떤 방법이나 도구로 처리하고 계신가요?",
        f"지금은 ‘{anchor(3, '현재 사용 중인 방법')}’ 방식으로 해결하고 계신데, 문제가 해결되면 어떤 모습이면 좋을까요?",
        f"‘{anchor(4, '원하는 변화')}’를 가장 필요로 하는 사용자는 구체적으로 어떤 분들인가요?",
        f"‘{anchor(5, '말씀하신 사용자')}’에게 꼭 필요한 핵심 기능 세 가지는 무엇인가요?",
    )
    question = templates[q_idx] if 0 <= q_idx < len(templates) else _QUESTIONS[0]["fallback_question"]
    invalid = state.get("invalid")
    if invalid and invalid.get("stage") == q_idx:
        return f"{invalid['reason']} {question}"
    return question


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
    dynamic_prompt = f"""사용자의 다음 질문에 대한 좋은 답변 예시 3개를 평문으로 생성하세요.

현재까지 수집한 정보:
{context}

다음 질문 #{question_idx + 1}: {question['label']}

질문 가이드:
{question['guideline']}

답변 형식: {question['suggestion_format']}

답변 형식:
SUGGESTION: 첫 번째 답변
SUGGESTION: 두 번째 답변
SUGGESTION: 세 번째 답변

중요한 규칙:
- 절대 이전 질문의 답변으로 혼동하지 말 것
- 위 '수집한 정보'에 나온 사용자의 실제 상황에 맞춘 구체적인 내용일 것
- 각 suggestion은 사용자가 그대로 입력할 수 있는 완성된 문장
- 반드시 정확히 3개, 서로 다른 방향으로
- 자리표시자(예: "답변1", "기능명") 금지 — 실제 내용이어야 함"""

    try:
        messages_for_suggestions = [
            {"role": "system", "content": "사용자의 맥락에 맞는 답변 예시를 SUGGESTION 라벨 평문으로만 생성합니다."},
            {"role": "user", "content": dynamic_prompt}
        ]

        raw = await _call_exaone(client, messages_for_suggestions, max_tokens=500)
        node = _parse_labeled_text(raw, {"SUGGESTION"})
        suggestions = _clean_stage_suggestions(node.get("SUGGESTION"), question_idx, context)
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
            "MESSAGE와 SUGGESTION 라벨 평문만 출력하세요."
        ),
    }]
    try:
        raw = await _call_exaone(client, retry_messages, max_tokens=800)
        node = _parse_labeled_text(raw, {"SUGGESTION"})
        msg = node.get("MESSAGE")
        cleaned = _strip_format_hint(msg) if isinstance(msg, str) else None
        return (cleaned if _is_valid_question(cleaned) else None), _clean_suggestions(node.get("SUGGESTION"))
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

응답 형식:
NAME: 첫 번째 이름
NAME: 두 번째 이름
NAME: 세 번째 이름

예시:
- 불편함: "냉장고 뭐가 있는지 몰라서 낭비", 해결: "냉장고 재고 자동 추적"
  → "FridgeTracker", "냉동고", "FreshKeep"
"""

    try:
        name_msgs = [
            {"role": "system", "content": "프로젝트 이름 3개를 NAME 라벨 평문으로만 생성합니다."},
            {"role": "user", "content": name_prompt}
        ]
        
        raw = await _call_exaone(client, name_msgs, max_tokens=500)
        node = _parse_labeled_text(raw, {"NAME"})
        names = [
            name for name in _clean_suggestions(node.get("NAME"))
            if _is_valid_generated_project_name(name)
        ]
        if len(names) == 3:
            logger.info("프로젝트 이름 후보 생성 성공: %s", names)
            return names
        logger.warning("프로젝트 이름 후보 유효 항목 부족 (%d/3)", len(names))
    
    except Exception as e:
        logger.warning("프로젝트 이름 생성 실패: %s", e)
    
    return []


async def _synthesize_form_data(messages: list[ChatMessageDto], client) -> dict | None:
    """고정된 인터뷰 순서의 사용자 답변을 FormData로 결정론적으로 조립한다."""
    state = _collect_interview_state(messages)
    if not state["idea"] or state["stage"] < _QUESTION_COUNT:
        return None

    idea = state["idea"]
    platform_text, pain, solution, ideal, persona, feature_text = state["answers"]
    platform = "WEB_APP" if ("WEB_APP" in platform_text.upper() or ("웹" in platform_text and "앱" in platform_text)) else (
        "APP" if ("APP" in platform_text.upper() or "앱" in platform_text or "모바일" in platform_text) else "WEB"
    )
    project_name = state["extras"][0] if state["extras"] else ""
    project_name = re.sub(r"^(프로젝트\s*)?이름(은|으로|:)\s*", "", project_name).strip()

    features = _split_features(feature_text)
    if len(features) < 3:
        logger.warning("FormData 조립 거부 — 핵심 기능 부족(%d/3)", len(features))
        return None
    if not project_name:
        naming_context = f"{idea} {pain} {feature_text}"
        if any(token in naming_context for token in ("냉장고", "재료", "유통기한")):
            project_name = "냉프레시"
        elif any(token in naming_context for token in ("일정", "할 일", "업무")):
            project_name = "플로우메이트"
        else:
            stem = re.sub(r"[^가-힣A-Za-z0-9]", "", features[0])[:12] or "서비스"
            project_name = f"{stem} 프로젝트"

    node = {
        "projectName": project_name,
        "projectDescription": (
            f"사용자가 말한 ‘{pain}’라는 현재의 어려움을 줄이고, "
            f"‘{ideal}’라는 목표를 지원하는 서비스입니다."
        ),
        "platform": platform,
        "techStack": [],
        "problemDefinition": {
            "currentPainPoint": pain,
            "currentSolution": solution,
            "idealState": ideal,
            "businessImpact": None,
            "motivation": idea,
            "competitorGap": None,
        },
        "targetUsers": [{
            "persona": persona,
            "usageEnvironment": f"{platform} 환경에서 일상적으로 사용",
            "biggestPainPoint": pain,
        }],
        "featureDefinition": {
            "commonFeatures": [],
            "customFeatures": [
                {"featureName": name, "description": f"사용자가 '{name}' 기능을 사용할 수 있도록 지원합니다.", "priority": "MUST"}
                for name in features
            ],
        },
    }
    validated = FormData.model_validate(node)
    logger.info("FormData Python 조립 완료 — 기능 %d개", len(features))
    return validated.model_dump(by_alias=True)


async def _finish_collection(
    messages: list[ChatMessageDto], user_msg_count: int, client,
) -> dict:
    """6개 항목 수집 완료 후 단계 — 프로젝트명 후보 제시(1회) → FormData 합성."""
    state = _collect_interview_state(messages)
    logger.info(
        "합성 단계 진입 | user_msgs=%d, accepted=%d, naming_answers=%d",
        user_msg_count, state["stage"], len(state["extras"]),
    )

    # 1단계: 프로젝트 이름 후보 제시 (수집 직후 1회)
    if not state["extras"]:
        project_name_candidates = await _generate_project_name_candidates(messages, client)

        if len(project_name_candidates) < 3:
            project_name_candidates = _fallback_project_name_candidates(state)

        if project_name_candidates:
            logger.info("프로젝트 이름 후보 제시 | candidates=%s", project_name_candidates)
            return ok({
                "message": "좋아요! 충분한 정보가 모였어요. 이 프로젝트에 어울리는 이름이 뭘까요?",
                "isComplete": False,
                "suggestions": project_name_candidates,
                "formData": None,
                "stage": "naming",
            })
    elif not _is_valid_project_name(state["extras"][0]):
        return ok({
            "message": "프로젝트 이름이 비어 있거나 완성되지 않았어요. 사용할 이름을 2자 이상으로 입력해 주세요.",
            "isComplete": False,
            "suggestions": [],
            "formData": None,
            "stage": "naming",
        })

    # 2단계: FormData 합성
    form_data = await _synthesize_form_data(messages, client)
    if form_data:
        if state["extras"]:
            form_data["projectName"] = state["extras"][0].strip()

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

        # 필수 수집 항목과 순서는 제품 계약으로 유지하되, 실제 질문 문장과 보기는
        # 지금까지의 사용자 답변을 반영해 매 턴 생성한다. 출력이 깨진 경우에만 아래의
        # 단계별 fallback을 사용하므로 프론트는 고정된 보기 대신 동적 suggestions를 받는다.
        raw = await _call_exaone(exaone_client, collection_messages, max_tokens=800)
        logger.debug("Collection raw (%.400s)", raw)

        node = _parse_labeled_text(raw, {"SUGGESTION"})
        # 모델이 다른 수집 단계를 질문하는 경우가 있어 질문 문장은 Python이 현재
        # 단계와 사용자 원문을 조합한다. 모델은 맥락별 보기 생성에 집중한다.
        question = _contextual_question(req.messages, q_idx)
        suggestions = _clean_stage_suggestions(node.get("SUGGESTION"), q_idx, context)
        if q_idx == 0:
            # 플랫폼은 자유 생성 대상이 아니라 FormData enum 계약이다.
            # 세 보기가 한 방향으로 몰리지 않도록 WEB/APP/WEB_APP를 항상 하나씩 보장한다.
            suggestions = list(_QUESTIONS[0]["fallback_suggestions"])

        # 질문 한 항목 때문에 모델 전체 호출을 반복하지 않는다. 라벨 누락은 위에서 복구하고,
        # 여전히 무효면 현재 단계의 검증된 질문만 결정론적으로 대체한다.
        # 첫 응답의 보기가 부족하면 보기 생성만 한 번 보충한다. 질문 전체를 다시
        # 생성하지 않아 단계가 바뀌거나 같은 질문을 반복하는 문제를 막는다.
        if len(suggestions) < 3:
            logger.info("suggestions 부족 (%d/3, idx=%d) — 동적 보기만 보충", len(suggestions), q_idx)
            regenerated = await _generate_dynamic_suggestions(q_idx, context, exaone_client)
            if len(regenerated) == 3:
                suggestions = regenerated

        # 모델 응답이 두 번 모두 깨진 경우에만 안전한 정적 후보로 빈 자리만 채운다.
        if len(suggestions) < 3:
            existing = {_normalize(s) for s in suggestions}
            state = _collect_interview_state(req.messages)
            for s in _contextual_fallback_suggestions(q_idx, state):
                if len(suggestions) >= 3:
                    break
                valid, normalized, _ = _validate_stage_answer(q_idx, s)
                if valid and _normalize(normalized) not in existing:
                    suggestions.append(normalized)
                    existing.add(_normalize(normalized))

        return ok({
            "message": question,
            "isComplete": False,
            "suggestions": suggestions,
            "formData": None,
            "stage": _QUESTIONS[q_idx]["label"],
        })

    except Exception as e:
        logger.error("EXAONE 호출 실패: %s", e, exc_info=True)
        return ok({"message": "잠시 후 다시 시도해 주세요.", "isComplete": False, "suggestions": [], "formData": None})
