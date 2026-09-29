"""Shared generation and validation contracts for Phase 2 agents.

Prompts, deterministic assemblers and QA must make decisions from the same
vocabulary.  This module intentionally contains no service-specific entity
names; it models user intent (authentication, actions and issue severity).
"""
from __future__ import annotations

import json
import re
from enum import StrEnum
from typing import Any


class IssueSeverity(StrEnum):
    BLOCKER = "BLOCKER"
    ERROR = "ERROR"
    WARNING = "WARNING"


def contract_prompt(role: str, assigned_scope: str = "") -> str:
    """Render the same behavioral contract into every active agent prompt."""
    scope = assigned_scope.strip() or "전달받은 담당 항목"
    return (
        f"\n공통 AgentContract ({role}):\n"
        f"- 담당 범위는 '{scope}'이며 다른 기능이나 섹션을 추가하지 않습니다.\n"
        "- 오염 문자열, 템플릿 라벨, 다른 언어의 무관한 조각을 출력하지 않습니다.\n"
        "- 형식은 Python이 정규화하므로 의미와 업무 계약의 정확성을 우선합니다.\n"
        "- 자기검토에서 담당 범위·중복·참조 무결성을 확인합니다.\n"
        "- 재시도 요청을 받으면 전달된 실패 원인에 해당하는 항목만 수정합니다.\n"
    )


_SEMANTIC_GROUPS = (
    {"등록", "생성", "추가"},
    {"배정", "할당", "지정"},
    {"기록", "이력", "로그", "내역"},
    {"조회", "목록", "검색", "확인", "보기"},
    {"수정", "변경", "갱신", "편집", "업데이트"},
    {"삭제", "제거", "해제", "취소", "철회"},
    {"완료", "종료", "마감", "처리"},
    {"담당자", "팀원", "직원", "작업자", "assignee", "staff"},
    {"사용자", "회원", "고객", "user", "member", "customer"},
    {"예약", "신청", "접수", "booking", "reservation", "application"},
    {"수량", "재고", "잔여", "quantity", "stock", "balance"},
)
_TOKEN_RE = re.compile(r"[가-힣A-Za-z0-9]+")


def semantic_tokens(value: str) -> set[str]:
    tokens = {token.lower() for token in _TOKEN_RE.findall(str(value or "")) if len(token) >= 2}
    expanded = set(tokens)
    for group in _SEMANTIC_GROUPS:
        lowered = {token.lower() for token in group}
        if tokens & lowered:
            expanded.update(lowered)
    return expanded


def semantic_relevance(subject: str, content: str) -> float:
    expected = semantic_tokens(subject)
    if not expected:
        return 1.0
    actual = semantic_tokens(content)
    matched = 0
    for token in expected:
        if token in actual or any(
            token.startswith(other) or other.startswith(token)
            for other in actual if len(other) >= 2
        ):
            matched += 1
    return matched / len(expected)


def normalize_target_arrow(value: Any) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    text = re.sub(r"\s*(?:->|⇒|=>|에서)\s*", " → ", text)
    return text


def _prd_business_text(prd_document: str | dict) -> str:
    if isinstance(prd_document, str):
        try:
            data = json.loads(prd_document)
        except Exception:
            data = {}
    else:
        data = prd_document if isinstance(prd_document, dict) else {}
    # techStack.auth is an implementation suggestion, not proof that the product
    # requires identity.  Only product requirements may activate auth contracts.
    selected = {
        key: data.get(key)
        for key in ("projectOverview", "background", "coreFeatures", "mvpScope", "userPersonas")
        if key in data
    }
    return json.dumps(selected, ensure_ascii=False).lower()


def requires_auth(
    prd_document: str | dict = "", feature_list: list[str] | None = None,
    requirement_text: str = "",
) -> bool:
    text = (
        _prd_business_text(prd_document) + " " + " ".join(feature_list or []).lower()
        + " " + str(requirement_text or "").lower()
    )
    for public_phrase in ("로그인 없이", "비회원", "익명 공개", "인증 없이", "public access"):
        text = text.replace(public_phrase, " ")
    explicit = (
        "회원가입", "로그인", "계정", "본인 인증", "접근 권한", "역할 기반", "소유권",
        "사용자별", "회원별", "개인별", "본인만", "자신의", "내 데이터", "내 정보",
        "권한 없음", "요청 주체", "user-specific", "user owned", "owner_id",
        "oauth", "sso", "authentication", "authorization",
    )
    if any(token in text for token in explicit):
        return True
    return False


def feature_requires_user_scope(spec: dict[str, Any]) -> bool:
    """Return whether one feature explicitly owns or protects per-user data."""
    ownership = spec.get("ownership") if isinstance(spec, dict) else None
    if isinstance(ownership, dict) and str(ownership.get("scope") or "").upper() == "USER":
        return True
    if isinstance(ownership, (list, tuple, str)) and "USER" in str(ownership).upper():
        return True
    selected = {
        key: spec.get(key)
        for key in ("name", "rationale", "actions", "dataRequirements", "permissionRules")
    }
    text = json.dumps(selected, ensure_ascii=False).lower()
    return any(token in text for token in (
        "사용자별", "회원별", "개인별", "본인만", "자신의", "내 데이터", "내 정보",
        "소유자", "소유권", "권한 없음", "요청 주체", "user-specific", "user owned",
    )) or bool(re.search(
        r"사용자(?:가|의|는|에게)?.{0,18}(?:등록|저장|기록|관리|설정|소유|수정|삭제|알림|재고)",
        text,
    ))


def auth_feature_specs() -> list[dict[str, Any]]:
    """Canonical identity capabilities injected only when product evidence requires them."""
    common = {
        "parentFeature": "공통", "origin": "DERIVED", "source": ["permission"],
        "priority": "P0",
        "ownership": {
            "scope": "USER", "ownerEntity": "users", "ownerKey": "user_id",
            "access": "본인 또는 명시적으로 허용된 역할만 접근",
        },
    }
    return [
        {
            **common, "name": "회원 가입 및 로그인",
            "rationale": "사용자별 데이터를 안전하게 식별하고 접근 주체를 확정하기 위한 선행 기능",
            "actions": ["회원 가입", "자격 증명으로 로그인"],
            "dataRequirements": ["로그인 식별자", "단방향 해시 자격 증명", "계정 상태"],
            "permissionRules": ["가입은 공개하되 로그인 성공 전 보호 데이터에 접근할 수 없다"],
            "states": ["ACTIVE", "SUSPENDED", "WITHDRAWN"],
            "stateTransitions": ["ACTIVE → SUSPENDED: 계정 정지", "ACTIVE → WITHDRAWN: 회원 탈퇴"],
            "transactionRules": ["회원과 자격 증명을 한 트랜잭션으로 생성하고 중복 식별자는 거부한다"],
            "errorCases": ["로그인 식별자 중복", "자격 증명 불일치", "비활성 계정"],
            "acceptanceCriteria": ["가입 후 로그인할 수 있다", "잘못된 자격 증명은 보호 데이터 변경 없이 거부된다"],
        },
        {
            **common, "name": "인증 세션 관리",
            "rationale": "로그인 이후의 인증 유지, 갱신, 로그아웃과 토큰 재사용 방지를 일관되게 처리하기 위한 기능",
            "actions": ["접근 토큰 갱신", "로그아웃 및 세션 회수"],
            "dataRequirements": ["사용자 식별자", "Refresh Token 해시", "만료 시각", "회수 시각"],
            "permissionRules": ["유효하고 회수되지 않은 본인 세션만 갱신하거나 종료할 수 있다"],
            "states": ["ACTIVE", "REVOKED", "EXPIRED"],
            "stateTransitions": ["ACTIVE → REVOKED: 로그아웃", "ACTIVE → EXPIRED: 만료 시각 경과"],
            "transactionRules": ["토큰 갱신 시 기존 토큰을 회수하고 새 토큰을 원자적으로 발급한다"],
            "errorCases": ["토큰 만료", "회수된 토큰 재사용", "세션 소유자 불일치"],
            "acceptanceCriteria": ["유효 세션은 갱신된다", "로그아웃 뒤 동일 토큰 재사용은 거부된다"],
        },
        {
            **common, "name": "내 정보 및 계정 관리",
            "rationale": "인증된 사용자가 자신의 계정 정보를 조회·변경·탈퇴할 수 있어야 하기 위한 기능",
            "actions": ["내 정보 조회", "내 정보 수정", "회원 탈퇴"],
            "dataRequirements": ["사용자 식별자", "프로필 정보", "계정 상태", "수정 시각"],
            "permissionRules": ["본인 계정만 조회·수정·탈퇴할 수 있다"],
            "states": ["ACTIVE", "WITHDRAWN"],
            "stateTransitions": ["ACTIVE → WITHDRAWN: 회원 탈퇴"],
            "transactionRules": ["탈퇴 상태 전환과 활성 세션 회수를 한 트랜잭션으로 처리한다"],
            "errorCases": ["인증 없음", "본인 계정 아님", "이미 탈퇴한 계정"],
            "acceptanceCriteria": ["인증 사용자는 자신의 정보를 조회·수정할 수 있다", "탈퇴 후 보호 API 접근이 거부된다"],
        },
        {
            **common, "name": "사용자별 데이터 접근 제어",
            "rationale": "각 기능 데이터의 소유자를 확정하고 다른 사용자의 데이터 접근을 차단하기 위한 공통 기능",
            "actions": ["요청 주체와 데이터 소유자 비교", "허용된 범위만 조회·변경"],
            "dataRequirements": ["사용자 식별자", "기능 데이터의 user_id", "접근 결정 결과"],
            "permissionRules": ["모든 사용자 소유 데이터는 인증 주체의 user_id로 범위를 제한한다"],
            "states": ["ALLOWED", "DENIED"],
            "stateTransitions": ["요청 검증 → ALLOWED 또는 DENIED: 소유권·역할 검사"],
            "transactionRules": ["변경 쿼리의 소유권 조건과 데이터 변경을 같은 트랜잭션 경계에서 처리한다"],
            "errorCases": ["인증 없음", "소유자 불일치", "허용되지 않은 역할"],
            "acceptanceCriteria": ["다른 사용자의 식별자로 조회·변경해도 데이터가 노출되지 않는다", "허용된 본인 요청은 정상 처리된다"],
        },
    ]


def feature_methods(feature: str) -> tuple[str, ...]:
    """Derive a minimal REST action set from explicit user verbs.

    Read access is included for persisted resources. DELETE is never invented
    unless deletion/cancellation is explicitly requested.
    """
    text = str(feature or "").lower()
    if any(token in text for token in ("회원 가입", "회원가입", "로그인", "sign up", "signup", "login")):
        return ("POST",)
    if any(token in text for token in ("인증 세션", "토큰 갱신", "로그아웃", "auth session", "token refresh", "logout")):
        return ("POST",)
    if any(token in text for token in ("내 정보", "계정 관리", "my profile", "account management")):
        return ("GET", "PATCH", "DELETE")
    if any(token in text for token in ("접근 제어", "access control")):
        return ("GET",)
    read = any(token in text for token in (
        "조회", "목록", "검색", "보기", "확인", "통계", "분석", "추천", "현황", "대시보드", "모니터링",
    ))
    create = any(token in text for token in (
        "등록", "생성", "추가", "신청", "접수", "배정", "할당", "기록", "업로드", "발행", "공유",
    ))
    update = any(token in text for token in (
        "수정", "변경", "갱신", "편집", "설정", "배정", "할당", "지정", "승인", "처리", "완료", "분류", "전환",
    ))
    delete = any(token in text for token in ("삭제", "제거", "해제", "취소", "철회"))
    methods: list[str] = []
    if read or create or update or delete:
        methods.append("GET")
    if create:
        methods.append("POST")
    if update:
        methods.append("PATCH")
    if delete:
        methods.append("DELETE")
    return tuple(methods or ("GET", "POST"))


def feature_relation_kind(feature: str) -> str:
    """Classify a feature by behavior, independent of service entity names."""
    tokens = semantic_tokens(feature)
    if tokens & {"배정", "할당", "지정", "assignee"}:
        return "association"
    if tokens & {"기록", "이력", "로그", "내역"}:
        return "event"
    if tokens & {"감지", "분석", "계산", "추천", "예측", "표시", "집계", "평가", "우선순위"}:
        return "derived"
    return "aggregate"


def classify_issue(message: str) -> IssueSeverity:
    text = str(message or "")
    blocker_tokens = (
        "비어 있거나 파싱", "유효하지 않은 json", "참조 대상 없는 fk", "fk 순환",
        "api 요청 필드가 erd와 불일치", "api post 필수 필드", "rest 경로 형식 위반",
        "기능 목록이 비어", "문서가 비어", "기능 관계 fk 누락", "배정 주체 fk 누락",
        "엔드포인트가 없어", "crud/행위 계약 누락", "기능 미반영", "인덱스 문법 오류",
        "check 제약조건 오류",
    )
    warning_tokens = (
        "문장 잘림", "미완결", "persona", "kpi 근거", "kpi 지표", "오염 문자열",
        "placeholder", "스케줄러 인덱스", "fk 인덱스", "의미상 중복", "관련성이 낮",
    )
    lowered = text.lower()
    if any(token in lowered for token in blocker_tokens):
        return IssueSeverity.BLOCKER
    if "persona" in lowered and any(token in lowered for token in ("개뿐", "최소", "누락", "부족")):
        return IssueSeverity.ERROR
    if any(token in lowered for token in warning_tokens):
        return IssueSeverity.WARNING
    return IssueSeverity.ERROR
