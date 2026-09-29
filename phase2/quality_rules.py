"""Shared deterministic quality gates for Phase 2 agents.

LLM self-review is useful context, but it is never the acceptance criterion.  Every
agent uses these rules before an item can enter the shared pipeline state.
"""
from __future__ import annotations

import json
import re
from collections.abc import Iterable
from urllib.parse import urlparse

from phase2.agent_contract import semantic_relevance


_CONTAMINATION_PATTERNS = (
    (re.compile(r"\bXVG\b", re.I), "XVG token"),
    (re.compile(r"DELIV(?:ER|THELIVER)ABLES?", re.I), "prompt label leak"),
    (re.compile(r"(?:abspath|banned\s+token|unexpected\s+error)", re.I), "runtime/meta text"),
    (re.compile(r"IMAGE_TO_TEXT|PRODUCT_CLASSIFICATION|FOOD_SAFETY_ALERT", re.I), "internal enum leak"),
    (re.compile(r"</?think>|```|\$\{?\w+\}?|\{(?:name|value|serviceName)\}", re.I), "template/script leak"),
    (re.compile(r"\.(?:toml|ya?ml|ini|env)(?![A-Za-z0-9_])", re.I), "config/file fragment leak"),
    (re.compile(r"\*\*\.\\|\\\*\*|\|{2,}"), "delimiter/markdown leak"),
    (re.compile(r"\]\s*(?:으로|로|을|를|이|가|은|는)(?:\s|$)"), "stray bracket Korean splice"),
    (re.compile(r"\b(?:entrega|cualquier\s+idioma|mysteries\s+of|the\s+player)\b", re.I), "foreign sentence leak"),
    (re.compile(r"(?:FREQU\b|watch\s+the\s+daily|모델\s*추론\s*오류|email://)", re.I), "meta/foreign phrase leak"),
    (re.compile(r"(?:시장\s*현황\s*,\s*문제\s*,\s*기회|\d+자\s*이상|출력\s*형식|마지막\s*줄에\s*SELF_CHECK)", re.I), "prompt instruction echo"),
    (re.compile(r"(?:uttered\s+response|per\s+instruction|instruction\s+guidelines?)", re.I), "model instruction leak"),
    (re.compile(r"\b(?:IDENTITY|USE_PATTERN|PAIN_POINT|TECH_LEVEL)\s*:", re.I), "internal field label leak"),
    (re.compile(r"(?:\d+\s*문장(?:을|이|으로)?|문장을?\s*(?:넘어서|이내|이상)|subtitle\s*:)", re.I), "length/prompt instruction leak"),
    (re.compile(r"(?:[-–—]\s*)?\d+\s*자(?:\s*(?:이내|이상))?\s*$", re.I), "length annotation leak"),
    (re.compile(r"\b(?:jury|thereof|wherein|lang-en)\b|\b(?:DEVELOP|OUTPUT|INPUT)\s*단계|\buser의\b", re.I), "short foreign fragment leak"),
    (re.compile(r"(?:^|\n)\s*(?:입력|출력|요구사항|설명)\s*[>:]", re.I), "prompt delimiter leak"),
    (re.compile(r"(?:을\(를\)|이\(가\)|은\(는\)|\(으\)로)"), "Korean particle template leak"),
    (re.compile(r"(?:있습니다가|합니다가|됩니다가|이다가\s+된다)"), "Korean grammar collision"),
    (re.compile(r"(?:찔끔|대충\s*처리|아무튼)"), "informal/noise fragment"),
    (re.compile(r"(?:확인해|처리해|관리해|하려고|하고\s*싶어)\s*[.]$"), "incomplete informal Korean ending"),
    (re.compile(r"(?:관리|설계|구현|제공|완료)\s*[.](?:\s+(?=[가-힣])|\s*$)"), "nominal Korean sentence ending"),
    (re.compile(r"(?<![A-Za-z])mvp(?![A-Za-z])"), "lowercase acronym style leak"),
    (re.compile(r"(?:이|가)\s*(?:기록|저장|관리|처리|구현)\s*[.]$"), "incomplete Korean predicate"),
    (re.compile(r"(?:SELF_CHECK\s*:\s*(?:PASS|FAIL)|이전\s*시도에서|출처\s*규칙\s*:|최종\s*작성\s*예시|지시가\s*있으나|조건을\s*충족한\s*상태)", re.I), "retry/prompt text leak"),
    (re.compile(r"\[(?:문제\s*정의|타겟\s*유저|기능\s*정의)\]", re.I), "interview context leak"),
    (re.compile(r"(?:URL://|https?://\S+://)", re.I), "malformed URL"),
    (re.compile(r"(?:\b[A-Za-z]+[\s,;:'()–—-]+){8,}\b[A-Za-z]+"), "foreign-language sentence leak"),
    (re.compile(r"[\u0400-\u04FF]|\ufffd"), "unexpected script/replacement character"),
    (re.compile(r"(?<!\d)0\d{5,}(?!\d)"), "source identifier leak"),
)
_PLACEHOLDER_RE = re.compile(
    r"(?:자동 생성 실패|수동 보완|목표치 미정|^미정$|설명 없음|요청 데이터$|성공 응답$|unknown)", re.I,
)
_SELF_CHECK_RE = re.compile(r"^\s*SELF_CHECK\s*:\s*PASS\s*$", re.I | re.M)
_TOKEN_RE = re.compile(r"[가-힣A-Za-z0-9]+")
_SOURCE_URL_RE = re.compile(r"https?://[^\s)>\]]+", re.I)
_SOURCE_ID_RE = re.compile(r"(?:보고서|통계|공시|자료명|DOI|ISBN|보도자료|조사명)", re.I)


def flatten_text(value) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return " ".join(flatten_text(v) for v in value.values())
    if isinstance(value, Iterable) and not isinstance(value, (bytes, bytearray)):
        return " ".join(flatten_text(v) for v in value)
    return str(value or "")


def _string_values(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _string_values(item)
    elif isinstance(value, Iterable) and not isinstance(value, (bytes, bytearray)):
        for item in value:
            yield from _string_values(item)


def contamination_reasons(value) -> list[str]:
    text = flatten_text(value)
    reasons = [label for pattern, label in _CONTAMINATION_PATTERNS if pattern.search(text)]
    # A truncated Korean token followed by an unrelated CamelCase/long English
    # word is a frequent model splice (for example "스프레 SharePoint"). Keep
    # established technical names usable in tech-stack prose.
    allowed = {
        "api", "jwt", "oauth", "oauth2", "postgresql", "redis", "react",
        "node", "express", "rabbitmq", "cloudflare", "prometheus", "grafana",
        "web", "ui", "mvp", "qa", "pwa", "json", "sql", "fastapi", "python",
        "typescript", "kafka", "cdn", "opentelemetry", "bearer", "token", "unique",
    }
    mixed = [
        token
        for item in _string_values(value)
        for token in re.findall(r"[\uac00-\ud7a3]{1,4}\s+([A-Za-z][A-Za-z0-9_]{4,})\b", item)
    ]
    if any("_" not in token and token.lower() not in allowed for token in mixed):
        reasons.append("mixed-language splice")
    if " / " in text and "KOSIS 실제 통계값" not in text:
        reasons.append("inline delimiter leak")
    return list(dict.fromkeys(reasons))


def kpi_basis_issues(item: dict, market_data: str = "") -> list[str]:
    """Reject unsupported external evidence for a numeric product target.

    A launch baseline or an explicitly-labelled internal operating target needs no
    external citation. An external benchmark must carry a direct URL that is also
    present in the verified market-research payload. Laws may justify compliance
    requirements, but do not by themselves justify a conversion or retention goal.
    """
    if not isinstance(item, dict):
        return ["KPI 형식 오류"]
    metric = str(item.get("metric") or "")
    target = str(item.get("target") or "")
    basis = str(item.get("basis") or "").strip()
    if not metric or not target or not basis:
        return ["KPI 근거 필드 누락"]
    internal = any(token in basis for token in (
        "내부 운영 목표", "운영 목표", "출시 전 기준선", "제품 출시 전 기준선", "내부 측정",
    ))
    if internal:
        return []
    if any(token in basis for token in ("법령", "법적 의무", "개인정보 보호법", "시행령")):
        return ["법령을 제품 KPI 수치 근거로 사용"]
    urls = _SOURCE_URL_RE.findall(basis)
    if not urls:
        return ["외부 KPI 근거에 원문 URL 없음"]
    verified = market_data or ""
    if not any(url.rstrip(".,") in verified for url in urls):
        return ["KPI 근거 URL이 검증된 시장자료에 없음"]
    # The benchmark number must be visible in the cited basis. A source name alone
    # cannot explain why the goal is 55%, 1,000 users, and so on.
    goal_numbers = re.findall(r"\d[\d,.]*", target.split("→")[-1])
    if goal_numbers and not any(number in basis for number in goal_numbers):
        return ["KPI 목표 수치와 외부 근거 수치가 연결되지 않음"]
    return []


_REQUIREMENT_FIELD_CONCEPTS = (
    ("name", re.compile(r"(?:이름|명칭|상품명|장비명|반려동물명)"), {"name", "title"}),
    ("species", re.compile(r"(?:동물\s*)?종류|품종"), {"species", "animal_type", "breed"}),
    ("birth_date", re.compile(r"(?:생년월일|출생일|생일)"), {"birth_date", "date_of_birth", "born_at"}),
    ("gender", re.compile(r"(?:성별)"), {"gender", "sex"}),
    ("image_url", re.compile(r"(?:사진|이미지)"), {"image_url", "photo_url", "image", "photo"}),
    ("title", re.compile(r"(?:제목|강의명|상품명|장비명|티켓명|이름을?\s*(?:입력|등록|저장))"), {"title", "name"}),
    ("start_time", re.compile(r"(?:시작\s*(?:시간|일시|시각)|대여\s*(?:시작\s*시간|가능\s*시간))"), {"start_time", "starts_at", "start_at"}),
    ("end_time", re.compile(r"(?:종료\s*(?:시간|일시|시각)|반납\s*(?:예정|시간)|대여\s*종료)"), {"end_time", "ends_at", "end_at", "due_at"}),
    ("date", re.compile(r"(?:날짜|일자|예약일|대여일)"), {"date", "scheduled_date", "schedule_date", "reservation_date", "registered_at", "created_at"}),
    ("weekday", re.compile(r"요일"), {"weekday", "day_of_week"}),
    ("capacity", re.compile(r"(?:정원|수용\s*인원|최대\s*인원)"), {"capacity", "max_participants", "limit_count"}),
    ("status", re.compile(r"(?:상태|단계|처리\s*결과)"), {"status", "state"}),
    ("location", re.compile(r"(?:장소|위치|주소를?\s*(?:입력|등록|저장))"), {"location", "address", "place"}),
    ("quantity", re.compile(r"(?:수량|개수|재고량)"), {"quantity", "count", "stock"}),
    ("amount", re.compile(r"(?:금액|가격|비용|요금)"), {"amount", "price", "cost", "fee"}),
    ("priority", re.compile(r"우선순위"), {"priority"}),
    ("due_date", re.compile(r"(?:마감일|기한|완료\s*예정일)"), {"due_date", "deadline", "due_at"}),
    ("content", re.compile(r"(?:내용|상세\s*설명|메모를?\s*(?:입력|등록|저장))"), {"content", "description", "memo", "body"}),
)


def required_field_concepts(value) -> dict[str, set[str]]:
    """Return high-confidence business fields explicitly requested by prose."""
    text = flatten_text(value)
    return {name: aliases for name, pattern, aliases in _REQUIREMENT_FIELD_CONCEPTS if pattern.search(text)}


def has_placeholder(value) -> bool:
    return bool(_PLACEHOLDER_RE.search(flatten_text(value).strip()))


def self_check_passed(raw: str) -> bool:
    return bool(_SELF_CHECK_RE.search(raw or ""))


def retry_prompt(base_prompt: str, reasons: list[str], attempt: int) -> str:
    if not reasons:
        return base_prompt
    unique = list(dict.fromkeys(str(r) for r in reasons if str(r).strip()))
    return (
        base_prompt
        + f"\n\n[이전 시도 {attempt} 실패 사유]\n- "
        + "\n- ".join(unique)
        + "\n위 실패 항목만 교정하여 처음부터 다시 작성하고 마지막 줄에 SELF_CHECK: PASS를 출력하세요."
    )


def content_tokens(text: str) -> set[str]:
    stop = {"기능", "사용자", "지원", "제공", "관리", "기반", "자동", "통한", "대한", "및"}
    return {
        token.lower() for token in _TOKEN_RE.findall(text or "")
        if len(token) >= 2 and token.lower() not in stop
    }


def relevance_score(subject: str, content: str) -> float:
    return semantic_relevance(subject, content)


def feature_semantic_issues(feature: str, content: str) -> list[str]:
    """Detect semantic drift without assuming a particular service domain."""
    feature, content = str(feature or ""), str(content or "")
    issues: list[str] = []
    if semantic_relevance(feature, content) < 0.2:
        issues.append("담당 기능의 행위·대상 개념이 설명에 반영되지 않음")
    if not any(token in feature for token in ("UI", "접근성", "디자인", "반응형")) and any(
        token in content for token in ("최소 너비", "px 기준", "반응형", "디자인 시스템", "접근성 지원")
    ):
        issues.append("담당 기능과 무관한 UI·디자인 범위가 추가됨")
    return issues


def near_duplicate(left: str, right: str, threshold: float = 0.65) -> bool:
    left_numbers = set(re.findall(r"\d+(?:\.\d+)?", str(left or "")))
    right_numbers = set(re.findall(r"\d+(?:\.\d+)?", str(right or "")))
    if left_numbers and right_numbers and left_numbers != right_numbers:
        return False
    a, b = content_tokens(left), content_tokens(right)
    if not a or not b:
        return left.strip().lower() == right.strip().lower()
    return len(a & b) / min(len(a), len(b)) >= threshold


def dedupe_labels(values: list[str]) -> list[str]:
    out: list[str] = []
    for value in values:
        text = str(value or "").strip()
        if text and not any(near_duplicate(text, prior) for prior in out):
            out.append(text)
    return out


def source_evidence_issues(text: str) -> list[str]:
    """Require direct evidence in every paragraph that makes a numeric claim."""
    claim_re = re.compile(r"(?:\d[\d,.~～-]*\s*(?:%|명|가구|개|원|억\s*원|조\s*원|건|회|배|ms|초|시간)|제\s*\d+\s*조)", re.I)
    # Models often omit blank lines, so a new bracketed claim starts a new evidence unit.
    paragraphs = [
        part.strip() for part in re.split(r"\n\s*\n|(?=^\[(?!출처\]))", text or "", flags=re.M)
        if part.strip()
    ]
    unsupported = 0
    for paragraph in paragraphs:
        if not claim_re.search(paragraph):
            continue
        urls = _SOURCE_URL_RE.findall(paragraph)
        has_direct_url = any(
            (lambda parsed: bool(parsed.netloc and (parsed.path not in ("", "/") or parsed.query)))(urlparse(url.rstrip(".,")))
            for url in urls
        )
        if not has_direct_url or not _SOURCE_ID_RE.search(paragraph):
            unsupported += 1
    if unsupported:
        return [f"수치 주장 {unsupported}개 문단에 원문 세부 URL 또는 출처 식별자 없음"]
    return []


def stable_json(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)
