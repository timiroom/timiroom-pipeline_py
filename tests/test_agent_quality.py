import json

from phase2.agents.api_agent import _align_endpoints_to_db, _feature_methods, _feature_resource_slug
from phase2.agents.dba_agent import ensure_domain_columns, reconcile_fk_types, sanitize_tables
from phase2.agents.pm_agent import PmAgent, _parse_feature_detail, _parse_plain_pm
from phase2.agents.prd_agent import PrdAgent
from phase2.agents.qa_agent import QaAgent
from phase1.hybrid_search import _SEARCH_TYPE_SQL
from phase2.quality_rules import (
    contamination_reasons, feature_semantic_issues, kpi_basis_issues, source_evidence_issues,
)


def test_shared_contamination_gate_rejects_observed_model_leaks():
    assert contamination_reasons("장 XVG 냉장 효과 데이터")
    assert contamination_reasons("ipv4 DELIVTHELIVERABLES: API")
    assert contamination_reasons("완료. ||| IMAGE_TO_TEXT")
    assert contamination_reasons("시장 현황, 문제, 기회를 완성된 문장으로 200자 이상")
    assert contamination_reasons("요구사항 A || 요구사항 B")
    assert contamination_reasons(
        "Many users struggle to monitor refrigerator inventory efficiently and need reliable daily notifications."
    )
    assert contamination_reasons("[문제 정의] 현재 문제")


def test_contamination_gate_rejects_instruction_fragments_inside_prd_values():
    assert contamination_reasons("빠른 조회를 지원합니다 uttered response in Korean per instruction guidelines.")
    assert contamination_reasons("장부를 찾는 과정이 번거로2문장을 넘어서요.")
    assert contamination_reasons("정상 문장입니다. subtitle: 두 문장으로 작성")
    assert contamination_reasons("OAuth jury 기반 인증을 사용합니다")
    assert contamination_reasons("현재 문제가 있습니다가 된다")
    assert contamination_reasons("입력> 시스템은 값을 저장합니다")
    assert contamination_reasons("공구의 기본 정보와 관리 상태가 기록.")
    assert contamination_reasons("완료 상태를 기록합니다. – 144자")
    assert contamination_reasons("메모와 스프레 SharePoint 등에 흩어져 있습니다")
    assert contamination_reasons("권한 확인 / 마감일 추적")
    assert contamination_reasons("기능을(를) 연결합니다")
    assert contamination_reasons("IDENTITY: 서비스 목표")
    assert contamination_reasons("Redis 캐시로 응답 속도 향상 ]으로")
    assert not contamination_reasons("선행 행위 equipment_reservations의 상태를 확인합니다")
    assert contamination_reasons("불편함입니다. USE_PATTERN: 매일 사용")
    assert not contamination_reasons(["정상적인 한국어 문장입니다", "FastAPI 기술을 사용합니다"])
    assert contamination_reasons("사용 통계 식별자 011468")
    assert contamination_reasons("전체.toml에 상태를 기록합니다")
    assert contamination_reasons("찔끔 기능을 추가합니다")
    assert contamination_reasons("매일 상태를 확인해.")
    assert contamination_reasons("기능을 설계. 사용자가 검수합니다.")
    assert contamination_reasons("mvp에 필수 포함합니다")


def test_kpi_basis_requires_verified_numeric_evidence_or_internal_goal():
    internal = {
        "metric": "30일 리텐션", "target": "0% → 20%",
        "basis": "제품 출시 전 기준선 0에서 시작하는 내부 운영 목표",
    }
    law = {
        "metric": "가입 완료율", "target": "0% → 90%",
        "basis": "국가법령정보, 개인정보 보호법상 법적 의무 반영",
    }
    verified = {
        "metric": "활성화율", "target": "0% → 60%",
        "basis": "공식 통계의 60% 벤치마크 https://example.org/report/2025",
    }
    assert not kpi_basis_issues(internal)
    assert kpi_basis_issues(law)
    assert not kpi_basis_issues(verified, "SOURCE\nurl: https://example.org/report/2025")
    assert kpi_basis_issues(verified, "")


def test_phase1_global_search_excludes_previous_pipeline_artifacts():
    assert "pipeline_id" in _SEARCH_TYPE_SQL
    assert "= ''" in _SEARCH_TYPE_SQL


def test_feature_semantic_gate_rejects_cross_feature_content():
    wrong = "재료를 직접 추가하고 유통기한 알림을 설정합니다. 입력값을 저장합니다."
    assert feature_semantic_issues("소비 기록", wrong)
    assert not feature_semantic_issues(
        "소비 기록", "소비 사용량을 기록하고 재고를 차감해 잔여 수량을 갱신합니다."
    )
    assert feature_semantic_issues("유통기한 알림", "유통기한 알림을 제공하며 최소 너비 3276px 반응형 UI를 지원합니다.")
    assert not feature_semantic_issues("재료 추가", "재료를 등록하면 유통기한을 계산하고 알림을 전송합니다.")
    assert feature_semantic_issues("유통기한 알림", "재료명과 구매일을 입력해 데이터베이스에 저장합니다.")


def test_search_numeric_claim_requires_url_and_publication_identifier():
    assert source_evidence_issues("시장 규모는 4.2조 원이다")
    assert source_evidence_issues("시장 규모 통계 보고서: 4.2조 원 https://example.org")
    assert not source_evidence_issues("시장 규모 통계 보고서: 4.2조 원 https://example.org/report/2025")
    assert source_evidence_issues(
        "시장 규모 통계 보고서: 4.2조 원 https://example.org/report\n\n"
        "이용률은 78%로 조사됐다."
    )


def test_pm_rejects_untraced_expansion_and_duplicate_features():
    data = {
        "featureList": ["재료 추가", "재료 추가 기능", "유통기한 알림", "새 추천 기능"],
        "dbaInstruction": "tables",
        "apiInstruction": "apis",
        "selfCheck": "PASS",
    }
    issues = PmAgent._quality_issues(data, "{}", ["재료 추가", "유통기한 알림"])
    assert "의미상 중복 기능 존재" in issues
    assert "파생 기능 origin 누락: 새 추천 기능" in issues
    assert "파생 기능 parentFeature 누락: 새 추천 기능" in issues
    assert "파생 기능 근거 부족: 새 추천 기능" in issues


def test_pm_plain_output_is_assembled_without_json_parser():
    raw = (
        "FEATURE: 재료 추가\n"
        "PARENT: \n"
        "ORIGIN: USER\n"
        "SOURCE: problem, workflow\n"
        "RATIONALE: 사용자가 직접 요청한 핵심 재료 등록 기능입니다\n"
        "PRIORITY: P0\n"
        "ACTIONS: 재료를 등록한다\n"
        "DATA: 재료명, 수량\n"
        "RULES: 소유자만 수정한다\n"
        "ERRORS: 필수값 누락\n"
        "ACCEPTANCE: 저장 후 목록에서 조회된다\n"
        "FEATURE: 유통기한 알림\n"
        "PARENT: 재료 추가\n"
        "ORIGIN: DERIVED\n"
        "SOURCE: workflow, exception\n"
        "RATIONALE: 유통기한을 놓치는 문제를 예방하기 위해 필요합니다\n"
        "PRIORITY: P1\n"
        "ACTIONS: 임박 재료를 알린다\n"
        "DATA: 유통기한, 알림 시각\n"
        "RULES: 중복 알림을 방지한다\n"
        "ERRORS: 알림 대상 없음\n"
        "ACCEPTANCE: 임박 시점에 한 번만 알림이 생성된다\n"
        "DBA_INSTRUCTION: users와 ingredients 테이블을 만들고 소유권 외래키를 둡니다. 조회 인덱스를 추가합니다.\n"
        "API_INSTRUCTION: JWT 인증 API와 재료 CRUD API를 제공합니다. 요청과 응답 필드를 ERD와 일치시킵니다.\n"
        "SELF_CHECK: PASS"
    )
    result = _parse_plain_pm(raw, ["재료 추가"])
    assert result["featureList"] == ["재료 추가", "유통기한 알림"]
    assert result["featureSpecs"][1]["origin"] == "DERIVED"
    assert result["featureSpecs"][1]["parentFeature"] == "재료 추가"
    assert result["featureSpecs"][1]["acceptanceCriteria"]
    assert result["selfCheck"] == "PASS"


def test_pm_discovery_keeps_valid_item_when_neighbor_is_invalid():
    originals = ["문서 등록", "문서 조회", "문서 수정"]
    valid = {
        "name": "문서 접근 권한", "parentFeature": "문서 조회", "origin": "DERIVED",
        "source": ["permission"], "rationale": "허가되지 않은 문서 조회와 변경을 차단하기 위해 필요하다",
        "priority": "P0",
    }
    invalid = {
        "name": "AI 추천", "parentFeature": "문서 조회", "origin": "USER",
        "source": [], "rationale": "짧음", "priority": "P1",
    }

    assert PmAgent._valid_derived_spec(valid, originals)["name"] == "문서 접근 권한"
    assert PmAgent._valid_derived_spec(invalid, originals) is None

    misplaced_source = dict(valid, origin="permission", source=[])
    repaired = PmAgent._valid_derived_spec(misplaced_source, originals)
    assert repaired["origin"] == "DERIVED"
    assert repaired["source"] == ["permission"]


def test_pm_feature_detail_parser_and_fallback_are_complete():
    parsed = _parse_feature_detail(
        "ACTIONS: 요청을 등록한다; 결과를 조회한다\n"
        "DATA: 요청 식별자; 처리 상태\n"
        "RULES: 소유자만 수정한다\n"
        "ERRORS: 필수값 누락; 권한 없음\n"
        "ACCEPTANCE: 등록 후 조회된다; 권한 없는 수정은 거절된다"
    )
    assert len(parsed["actions"]) == 2
    assert len(parsed["dataRequirements"]) == 2
    assert len(parsed["acceptanceCriteria"]) == 2

    fallback = PmAgent._fallback_detail("문서 등록", [])
    assert all(fallback[key] for key in (
        "actions", "dataRequirements", "permissionRules", "errorCases", "acceptanceCriteria",
    ))


def test_pm_feature_catalog_enriches_prd_with_traceable_derived_feature():
    prd = json.dumps({
        "coreFeatures": [{
            "name": "문서 등록", "description": "문서를 등록한다", "priority": "P0",
            "requirements": ["제목을 입력한다"],
        }],
        "mvpScope": {"included": ["문서 등록"], "excluded": []},
    }, ensure_ascii=False)
    specs = [{
        "name": "문서 등록", "origin": "USER", "priority": "P0",
        "source": ["problem"], "rationale": "사용자가 직접 요청한 기능",
        "actions": ["문서를 생성한다"], "dataRequirements": ["제목"],
        "permissionRules": ["소유자만 수정한다"], "errorCases": ["제목 누락"],
        "acceptanceCriteria": ["저장 후 조회된다"],
    }, {
        "name": "문서 접근 권한", "parentFeature": "문서 등록", "origin": "DERIVED", "priority": "P0",
        "source": ["permission"], "rationale": "다른 사용자의 문서 변경을 막기 위해 필요합니다",
        "actions": ["접근 권한을 검증한다"], "dataRequirements": ["소유자 식별자"],
        "permissionRules": ["소유자만 변경한다"], "errorCases": ["권한 없음"],
        "acceptanceCriteria": ["권한 없는 변경은 거부된다"],
    }]

    merged = json.loads(PmAgent._merge_feature_specs(prd, specs, ["문서 등록"]))
    derived = next(item for item in merged["coreFeatures"] if item["name"] == "문서 접근 권한")

    assert derived["parentFeature"] == "문서 등록"
    assert derived["origin"] == "DERIVED"
    assert derived["dataRequirements"] == ["소유자 식별자"]
    assert derived["acceptanceCriteria"] == ["권한 없는 변경은 거부된다"]
    assert "문서 접근 권한" in merged["mvpScope"]["included"]


def test_qa_rejects_untraceable_derived_feature_details():
    qa = object.__new__(QaAgent)
    prd = {
        "coreFeatures": [{
            "name": "문서 권한", "description": "문서 권한을 검증한다", "requirements": ["접근을 검사한다"],
            "origin": "DERIVED", "parentFeature": "", "source": [], "rationale": "",
            "actions": [], "dataRequirements": [], "acceptanceCriteria": [],
        }],
        "goals": ["목표 A", "목표 B", "목표 C"],
        "kpi": [], "userPersonas": [], "releaseSchedule": [], "mvpScope": {},
    }

    issues = qa._check_prd_completeness(prd, 1)

    assert any("상위 기능 누락" in issue for issue in issues)
    assert any("요구사항 근거 부족" in issue for issue in issues)
    assert any("수용 기준 누락" in issue for issue in issues)


def test_api_fallback_slug_is_domain_neutral_and_methods_follow_behavior():
    assert _feature_resource_slug("소비 기록 (재고 자동 갱신)", 0) == "feature-1"
    assert _feature_resource_slug("완전히 다른 서비스 기능", 1) == "feature-2"
    assert set(_feature_methods("사용자 맞춤 알림 설정")) == {"GET", "PATCH"}


def test_dba_postgres_normalization_does_not_invent_domain_columns():
    tables = [{
        "name": "inventory_items",
        "description": "사용자 냉장고 재고와 유통기한 관리",
        "columns": [
            {"name": "id", "type": "BIGINT", "constraints": "PRIMARY_KEY AUTO_INCREMENT"},
            {"name": "created_at", "type": "DATETIME", "constraints": "NOT_NULL"},
            {"name": "updated_at", "type": "DATETIME", "constraints": "NOT_NULL ON UPDATE"},
        ],
        "indexes": [],
    }]
    result = ensure_domain_columns(sanitize_tables(tables))[0]
    by_name = {column["name"]: column for column in result["columns"]}
    assert by_name["created_at"]["type"] == "TIMESTAMPTZ"
    assert "AUTO_INCREMENT" not in by_name["id"]["constraints"]
    assert not {"user_id", "ingredient_id", "quantity", "unit", "expiry_date"} & set(by_name)


def test_dba_does_not_invent_fields_from_table_name_or_description():
    tables = [{"name": "ingredients", "description": "재료 등록 후 소비 기록과 알림에 활용",
               "columns": [{"name": "id", "type": "BIGINT", "constraints": "PRIMARY_KEY"}], "indexes": []}]
    result = ensure_domain_columns(tables)[0]
    names = {column["name"] for column in result["columns"]}
    assert names == {"id"}
    assert "quantity_used" not in names
    assert "notification_type" not in names


def test_dba_removes_fk_columns_without_a_real_target():
    tables = [
        {"name": "users", "columns": [{"name": "id", "type": "BIGINT", "constraints": "PRIMARY_KEY"}]},
        {"name": "notifications", "columns": [
            {"name": "id", "type": "BIGINT", "constraints": "PRIMARY_KEY"},
            {"name": "user_id", "type": "INTEGER", "constraints": "FOREIGN_KEY"},
            {"name": "notifying_item_id", "type": "BIGINT", "constraints": "FOREIGN_KEY"},
        ]},
    ]
    result = reconcile_fk_types(tables)
    columns = {column["name"] for column in result[1]["columns"]}
    assert "user_id" in columns
    assert "notifying_item_id" not in columns


def test_api_manager_aligns_contract_fields_to_erd():
    db = '{"tables":[{"name":"consumption_records","columns":[' \
         '{"name":"id","type":"BIGINT"},{"name":"user_id","type":"BIGINT"},' \
         '{"name":"ingredient_id","type":"BIGINT"},{"name":"quantity_used","type":"DECIMAL(10,2)"}]}]}'
    endpoints = [{"method": "POST", "path": "/api/v1/consumption-records", "description": "소비 기록",
                  "authRequired": True, "requestBody": "food_id: bigint", "successResponse": "ok", "errorCodes": "500"}]
    result = _align_endpoints_to_db(endpoints, db)[0]
    assert "ingredient_id" in result["requestBody"]
    assert "food_id" not in result["requestBody"]
    assert "user_id" not in result["requestBody"]


def test_prd_manager_replaces_placeholders_and_resolves_p0_mvp_conflict():
    feature_list = ["재료 추가", "유통기한 알림", "소비 기록", "공유 냉장고"]
    sections = {
        "goals": ["동일 목표", "동일 목표", "자동 생성 실패 — 수동 보완 필요"],
        "kpi": [{"metric": "핵심 지표 3", "target": "0% → 목표치 미정", "basis": "미정",
                 "measurementMethod": "미정", "frequency": "미정"}],
        "userPersonas": [], "releaseSchedule": [],
        "coreFeatures": [
            {"name": feature, "description": f"{feature} 처리", "requirements": [f"{feature} 검증"], "priority": "P0"}
            for feature in feature_list
        ],
        "mvpScope": {"included": feature_list[:3], "excluded": feature_list[3:], "rationale": "범위"},
    }
    result = PrdAgent._audit_sections(sections, feature_list, "서비스")
    assert len(result["kpi"]) == 7
    assert all("미정" not in str(item) for item in result["kpi"])
    assert result["coreFeatures"][-1]["priority"] == "P2"
    assert "소비 기록" in result["coreFeatures"][2]["description"]
    assert "차감" not in result["coreFeatures"][2]["description"]


def test_prd_manager_replaces_prompt_dump_in_project_overview():
    result = PrdAgent._audit_sections(
        {"projectOverview": "서비스 설명\n[문제 정의]\n" + "반복 문장 " * 80, "background": "배경 " * 100},
        ["재료 추가"],
        "냉장고 재고를 관리하는 서비스입니다.\n[문제 정의]\n내부 프롬프트",
    )
    assert "\n" not in result["projectOverview"]
    assert len(result["projectOverview"]) <= 300


def test_prd_manager_removes_unlinked_external_kpi_basis():
    sections = {
        "kpi": [{"metric": "신규 가입자 수", "target": "0명 → 1,000명",
                 "basis": "한국정보화진흥원 2025 이용실태조사 보고서",
                 "measurementMethod": "가입 완료 이벤트 집계", "frequency": "월간"}],
    }
    result = PrdAgent._audit_sections(sections, ["재료 추가"], "서비스")
    first = next(item for item in result["kpi"] if item["metric"] == "신규 가입자 수")
    assert "보고서" not in first["basis"]


def test_prd_manager_drops_kpi_for_unapproved_feature_expansion():
    sections = {"kpi": [{"metric": "재료 인식 정확도", "target": "80% → 95%",
                         "basis": "내부 테스트", "measurementMethod": "카메라 이미지 비교", "frequency": "월간"}]}
    result = PrdAgent._audit_sections(sections, ["재료 추가", "유통기한 알림"], "서비스")
    assert all("인식" not in item["metric"] for item in result["kpi"])


def test_prd_manager_replaces_near_duplicate_persona_segments():
    sections = {"userPersonas": [
        {"name": "A", "age": "20대", "job": "의류 위탁 매장 관리자", "techLevel": "높음", "goal": "정산", "painPoint": "오류", "usagePattern": "매일"},
        {"name": "B", "age": "30대", "job": "중고 의류 위탁 매장 관리자", "techLevel": "중간", "goal": "정산", "painPoint": "오류", "usagePattern": "매일"},
        {"name": "C", "age": "40대 이상", "job": "위탁자", "techLevel": "낮음", "goal": "확인", "painPoint": "지연", "usagePattern": "주간"},
    ]}
    audited = PrdAgent._audit_sections(sections, ["정산 처리"], "위탁 판매 서비스")
    issues = object.__new__(QaAgent)._check_prd_completeness(audited, 1)
    assert not any("Persona" in issue for issue in issues)


def test_qa_semantic_checks_do_not_false_approve_broken_documents():
    qa = QaAgent.__new__(QaAgent)
    api_issues = qa._check_api_completeness({
        "authentication": "JWT",
        "endpoints": [{"method": "GET", "path": "/api/v1/resource-1", "description": "재료 추가",
                       "authRequired": True, "requestBody": "없음", "successResponse": "성공 응답",
                       "errorCodes": "500"}],
    }, ["재료 추가"])
    db_issues = qa._check_db_completeness({
        "tables": [{"name": "inventory", "description": "재고", "columns": [
            {"name": "id", "type": "BIGINT", "constraints": "AUTO_INCREMENT"},
        ], "indexes": []}], "relationships": [],
    }, ["재고 관리"])
    assert any("의미 없는 리소스 경로" in issue for issue in api_issues)
    assert any("PostgreSQL 비호환" in issue for issue in db_issues)
