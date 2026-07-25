"""기능 목록(featureList) 항목이 DB 스키마/API 스펙에 실제로 반영됐는지 확인하는
결정론적(비-LLM) 커버리지 체커. LLM의 "다 반영했다"는 자기평가만으로는 표/엔드포인트가
통째로 누락되는 경우를 못 잡아서, 기능명 키워드가 산출물의 한국어 설명 텍스트 어디에도
전혀 등장하지 않는 경우를 결정론적으로 탐지한다.

완벽한 의미 매칭은 아니고 "이 기능에 대응하는 흔적이 전혀 없음"만 잡는 최소한의 안전망이다
— 공유 인프라(예: 인증 테이블 하나로 여러 기능을 지원)로 커버되는 기능은 오탐될 수 있으므로
회귀 방지용 하드 게이트가 아니라 LLM 재검토를 유도하는 힌트로 사용해야 한다.
"""
import re

# 매칭에 무의미한 범용 단어 — 이 단어만으로는 커버리지 판단 근거로 삼지 않는다
_STOPWORDS = {
    "기능", "기능명", "관리", "설정", "제공", "가능", "지원", "및", "또는", "통한",
    "기반", "위한", "이용", "사용", "구현", "처리", "정보", "데이터", "시스템",
}
_MIN_TOKEN_LEN = 2
# 기능 목록 내에서 이 비율 이상 등장하는 토큰은 "재료/등록"처럼 도메인 전체에 흔한 단어로
# 보고 단독 매칭 근거로 인정하지 않는다 (예: 냉장고 앱에서 "재료"가 기능 대부분에 등장 →
# 이 단어 하나로 "레시피 추천"이 "재료 등록" 엔드포인트에 매칭돼버리는 오탐 방지).
_GENERIC_TOKEN_DOC_FREQ_RATIO = 0.34


def _feature_tokens(feature: str) -> set[str]:
    """기능명에서 매칭용 키워드 추출 — 괄호/구두점 제거 후 공백 분리, 불용어·짧은 토큰 제외."""
    if not isinstance(feature, str):
        return set()
    cleaned = re.sub(r"[()\[\]{}:,./·—\-!?'\"]", " ", feature)
    tokens = [t for t in re.split(r"\s+", cleaned) if len(t) >= _MIN_TOKEN_LEN]
    return {t for t in tokens if t not in _STOPWORDS}


def uncovered_features(feature_list: list, haystack_texts: list) -> list[str]:
    """feature_list 중 haystack_texts(테이블/엔드포인트 설명 등) 어디에도 키워드가
    전혀 매칭되지 않는 기능만 반환. 토큰이 하나도 추출되지 않는 기능(전부 불용어/기호)은
    판단 불가로 보고 통과시킨다.

    기능 목록 전체에서 흔하게 등장하는 토큰(도메인 공통어)은 그 토큰 하나만으로는
    매칭 근거로 삼지 않는다 — 서로 다른 기능인데 공통 단어 하나 때문에 "커버됨"으로
    오판되는 것을 막기 위함."""
    combined = " ".join(t for t in haystack_texts if isinstance(t, str))
    feature_list = feature_list or []
    tokens_per_feature = [_feature_tokens(f) for f in feature_list]

    doc_freq: dict[str, int] = {}
    for tokens in tokens_per_feature:
        for tok in tokens:
            doc_freq[tok] = doc_freq.get(tok, 0) + 1
    n = len(feature_list) or 1
    generic = {tok for tok, freq in doc_freq.items() if freq / n > _GENERIC_TOKEN_DOC_FREQ_RATIO}

    uncovered = []
    for feature, tokens in zip(feature_list, tokens_per_feature):
        if not tokens:
            continue
        specific = tokens - generic
        match_pool = specific or tokens  # 전부 범용 단어뿐이면 어쩔 수 없이 원래 토큰 사용
        if not any(tok in combined for tok in match_pool):
            uncovered.append(feature)
    return uncovered


def missing_features_note(missing: list[str], target_label: str) -> str:
    """리뷰 프롬프트에 주입할 '누락 기능' 힌트 텍스트. missing이 비어있으면 빈 문자열."""
    if not missing:
        return ""
    items = "\n".join(f"- {f}" for f in missing)
    return (
        f"\n[결정론적 커버리지 검사 — 반드시 확인]\n"
        f"아래 기능들은 현재 {target_label} 어디에도 반영된 흔적이 없습니다. "
        f"이미 다른 요소로 커버되고 있다면 무시해도 되지만, 실제로 빠졌다면 반드시 추가하세요:\n"
        f"{items}"
    )
