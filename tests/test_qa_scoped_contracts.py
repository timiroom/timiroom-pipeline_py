import pytest

from phase2.agents.qa_agent import QaAgent


def _schema():
    def table(name, description, extra_columns=()):
        return {
            "name": name, "description": description,
            "columns": [{"name": "id", "type": "BIGINT", "constraints": "PRIMARY_KEY"}, *extra_columns],
            "indexes": [],
        }

    def reference(name, target):
        return {"name": name, "type": "BIGINT", "constraints": f"FOREIGN_KEY REFERENCES {target}(id)"}

    return {"tables": [
        table("users", "회원"), table("documents", "문서"),
        table("favorites", "즐겨찾기 중복 방지", [reference("user_id", "users"), reference("document_id", "documents")]),
        table("document_views", "문서 조회 이력: 같은 회원이 같은 문서를 볼 때마다 기록", [
            reference("user_id", "users"), reference("document_id", "documents"),
            {"name": "viewed_at", "type": "TIMESTAMPTZ", "constraints": "NOT_NULL"},
        ]),
    ]}


def _prd():
    return {"coreFeatures": [
        {"name": "즐겨찾기", "requirements": ["동일 사용자의 동일 문서 중복 즐겨찾기 방지"]},
        {"name": "문서 조회 이력", "requirements": ["같은 회원이 같은 문서를 볼 때마다 새 조회 이력을 저장한다"]},
    ]}


def _unique_issues(db):
    issues, _, _ = QaAgent(object())._check_cross_document_semantics(_prd(), db, {"endpoints": []}, [])
    return [issue for issue in issues if "중복 방지 UNIQUE" in issue]


def test_duplicate_prevention_does_not_block_repeatable_sibling_events():
    issues = _unique_issues(_schema())

    assert issues == ["중복 방지 UNIQUE 제약 누락: favorites"]


def test_favorite_uniqueness_satisfies_its_own_feature_without_constraining_views():
    db = _schema()
    db["tables"][2]["indexes"] = [
        "CREATE UNIQUE INDEX uq_favorites_pair ON favorites (user_id, document_id)",
    ]

    assert _unique_issues(db) == []


def test_unrelated_unique_and_separate_indexes_do_not_satisfy_composite_contract():
    db = _schema()
    db["tables"][2]["indexes"] = [
        "CREATE UNIQUE INDEX uq_favorites_id ON favorites (id)",
        "CREATE INDEX idx_favorite_user ON favorites (user_id)",
        "CREATE INDEX idx_favorite_document ON favorites (document_id)",
    ]

    assert "중복 방지 UNIQUE 제약 누락: favorites" in _unique_issues(db)


@pytest.mark.parametrize("metadata", [
    {"goals": ["사용자 유입 증가"]},
    {"mvpScope": {"excluded": ["알림 설정"]}},
])
def test_goals_and_excluded_features_do_not_create_business_storage_requirements(metadata):
    prd = {"coreFeatures": [{"name": "문서 조회", "requirements": ["공개 문서를 조회한다"]}], **metadata}
    db = {"tables": [{"name": "documents", "description": "문서 조회", "columns": [
        {"name": "id", "type": "BIGINT", "constraints": "PRIMARY_KEY"},
    ]}]}

    issues, _, _ = QaAgent(object())._check_cross_document_semantics(prd, db, {"endpoints": []}, [])

    assert not any("상태 컬럼이 없음" in issue or "수신 설정 구조가 없음" in issue for issue in issues)


def test_active_state_mutation_requirement_still_requires_stored_state():
    prd = {"coreFeatures": [{"name": "재고 차감", "requirements": ["사용한 재고 수량을 차감한다"]}]}
    db = {"tables": [{"name": "items", "columns": [
        {"name": "id", "type": "BIGINT", "constraints": "PRIMARY_KEY"},
    ]}]}

    issues, _, _ = QaAgent(object())._check_cross_document_semantics(prd, db, {"endpoints": []}, [])

    assert any("상태 컬럼이 없음" in issue for issue in issues)
