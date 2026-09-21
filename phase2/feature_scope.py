"""기능별 산출물 담당 영역을 결정하는 공통 규칙.

기능 목록은 PRD와 사용자에게 보여줄 전체 요구사항이므로 그대로 보존한다.
DBA/API처럼 특정 산출물만 담당하는 에이전트는 프론트엔드 전용 기능을 제외해야
불필요한 테이블이나 endpoint를 억지로 만들지 않는다.
"""

from __future__ import annotations


# 저장/통신 모델 없이 화면 표현만 바꾸는 기능의 대표적인 표현.
_FRONTEND_ONLY_MARKERS = (
    "반응형",
    "다크모드",
    "다크 모드",
    "접근성",
    "레이아웃",
    "화면 디자인",
    "UI 디자인",
    "UX 디자인",
    "모바일 최적화",
    "태블릿 최적화",
)


def is_frontend_only_feature(feature: object) -> bool:
    """DB/API 저장이나 통신 계약이 필요하지 않은 화면 전용 기능인지 판정한다."""
    if not isinstance(feature, str):
        return False
    normalized = " ".join(feature.casefold().split())
    return any(marker.casefold() in normalized for marker in _FRONTEND_ONLY_MARKERS)


def backend_features(feature_list: list[str] | None) -> list[str]:
    """DBA/API가 설계해야 하는 기능만 반환한다. 원본 목록은 변경하지 않는다."""
    return [feature for feature in (feature_list or []) if not is_frontend_only_feature(feature)]
