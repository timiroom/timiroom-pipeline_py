from phase2.feature_scope import backend_features, is_frontend_only_feature


def test_responsive_ui_is_frontend_only() -> None:
    assert is_frontend_only_feature("반응형 웹 화면 지원(PC·태블릿)")


def test_backend_features_preserves_backend_requirements() -> None:
    features = ["반응형 웹 화면 지원(PC·태블릿)", "예약 등록", "고객 알림"]
    assert backend_features(features) == ["예약 등록", "고객 알림"]
    assert features == ["반응형 웹 화면 지원(PC·태블릿)", "예약 등록", "고객 알림"]
