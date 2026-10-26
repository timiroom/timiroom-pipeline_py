import asyncio
import json
from unittest.mock import AsyncMock

import pytest

from phase2.agents.prd_agent import PrdAgent
from phase2.agents.qa_agent import QaAgent
from phase2.state import PipelineState


def test_bodyless_post_transition_is_not_checked_as_a_new_row_insert():
    db = {"tables": [{"name": "allocations", "columns": [
        {"name": "id", "type": "UUID", "constraints": "PRIMARY_KEY"},
        {"name": "asset_id", "type": "UUID", "constraints": "NOT_NULL"},
        {"name": "status", "type": "TEXT", "constraints": "NOT_NULL DEFAULT 'active'"},
    ]}]}
    endpoint = {"method": "POST", "path": "/api/v1/allocations/{allocationId}/cancel", "successStatus": 204,
                "requestBody": "없음", "effects": [{"table": "allocations", "column": "status", "kind": "state_transition", "from": "active", "to": "cancelled"}],
                "fieldMappings": [{"source": "path", "field": "allocationId", "table": "allocations", "column": "id"}]}
    _, issues, _ = QaAgent(object())._check_cross_document_semantics({}, db, {"endpoints": [endpoint]}, [])
    assert not any("API POST 필수 필드" in issue for issue in issues)
    endpoint["effects"] = [{"table": "allocations", "kind": "insert"}]
    _, issues, _ = QaAgent(object())._check_cross_document_semantics({}, db, {"endpoints": [endpoint]}, [])
    assert any("asset_id" in issue for issue in issues)


@pytest.mark.parametrize('wrong', [False, True])
def test_qa_uses_explicit_feature_table_instead_of_route_parent(wrong):
    db = {'tables': [
        {'name': 'todos', 'columns': [{'name': 'title', 'type': 'TEXT', 'constraints': 'NOT_NULL'}]},
        {'name': 'validation_errors', 'columns': [{'name': 'error_code', 'type': 'TEXT', 'constraints': 'NOT_NULL'}]},
    ], 'featureMappings': [{'featureId': 'validation', 'featureName': '입력 검증', 'table': 'validation_errors'}]}
    api = {'endpoints': [{'method': 'POST', 'path': '/api/v1/todos/validations/title',
                         'featureId': 'validation', 'featureName': '입력 검증',
                         'requestBody': 'title: string' if wrong else 'error_code: string'}]}
    _, issues, _ = QaAgent(object())._check_cross_document_semantics({}, db, api, [])
    field_issues = [issue for issue in issues if 'ERD' in issue]
    assert bool(field_issues) is wrong
    if wrong:
        assert any('error_code' in issue for issue in field_issues)


def test_prd_repair_does_not_invent_unpatched_registry_extras():
    registry = [
        {'featureId': 'todos.create', 'name': '할 일 생성', 'source': 'prd_core', 'priority': 'P0'},
        {'featureId': 'errors.notify', 'name': '오류 안내', 'source': 'supporting', 'priority': 'P0'},
        {'featureId': 'colors.update', 'name': '색상 설정', 'source': 'supporting', 'priority': 'P2'},
    ]
    original = {'coreFeatures': [{'featureId': 'todos.create', 'name': '할 일 생성'}], 'mvpScope': {}}
    agent = object.__new__(PrdAgent)
    agent._call = AsyncMock(return_value=json.dumps({'patches': {'coreFeatures': [{'name': '할 일 생성'}]}}))
    state = PipelineState(feature_list=['할 일 생성'], feature_registry=registry, prd_document=json.dumps(original))
    result = json.loads(asyncio.run(agent.repair(state, '필수 오류 안내 계약 복구')).prd_document)
    ids = {feature.get('featureId') for feature in result['coreFeatures']}
    assert ids == {'todos.create'}
    assert 'colors.update' not in ids


def test_declared_refresh_contract_has_durable_token_storage_without_password_invention():
    from phase2.agents.dba_agent import _ensure_contract_tables
    tables = [{'name': 'members', 'columns': [{'name': 'id', 'type': 'UUID', 'constraints': 'PRIMARY_KEY'}]}]
    registry = [{'featureId': 'refresh', 'name': '세션 갱신',
                 'ownership': {'scope': 'USER', 'ownerEntity': 'members'},
                 'dbContract': {'tables': ['members'], 'foreignKeys': []},
                 'apiContract': [{'method': 'POST', 'path': '/api/v1/auth/refresh'}]}]
    result = _ensure_contract_tables(tables, registry)
    stores = [table for table in result if {'token_hash', 'expires_at', 'revoked_at'} <=
              {column['name'] for column in table['columns']}]
    assert len(stores) == 1
    assert not any(column['name'] == 'password_hash' for table in result for column in table['columns'])
    assert len(_ensure_contract_tables(result, registry)) == len(result)


def test_public_contract_does_not_invent_token_storage():
    from phase2.agents.dba_agent import _ensure_contract_tables
    tables = [{'name': 'items', 'columns': [{'name': 'id', 'type': 'UUID', 'constraints': 'PRIMARY_KEY'}]}]
    result = _ensure_contract_tables(tables, [{'featureId': 'read', 'name': '목록',
        'apiContract': [{'method': 'GET', 'path': '/api/v1/items'}]}])
    assert [table['name'] for table in result] == ['items']


def test_password_reset_tokens_do_not_satisfy_refresh_contract():
    from phase2.agents.dba_agent import _ensure_contract_tables
    tables = [
        {'name': 'users', 'columns': [{'name': 'id', 'type': 'BIGINT', 'constraints': 'PRIMARY_KEY'}]},
        {'name': 'password_reset_tokens', 'columns': [
            {'name': 'token_hash', 'type': 'TEXT'}, {'name': 'expires_at', 'type': 'TIMESTAMPTZ'}]},
    ]
    registry = [{'featureId': 'refresh', 'name': '세션 갱신',
                 'apiContract': [{'method': 'POST', 'path': '/api/v1/auth/refresh'}]}]
    result = _ensure_contract_tables(tables, registry)
    assert any(table['name'] == 'refresh_tokens' for table in result)
    assert next(table for table in result if table['name'] == 'password_reset_tokens')['columns'] == tables[1]['columns']


@pytest.mark.parametrize('broken', ['', 'missing_expiry', 'missing_revocation', 'sequential_id', 'wrong_store', 'refresh'])
def test_declared_uuid_bearer_session_store_supports_logout_without_refresh_tokens(broken):
    columns = [
        {'name': 'id', 'type': 'UUID', 'constraints': 'PRIMARY_KEY'},
        {'name': 'expires_at', 'type': 'TIMESTAMPTZ'},
        {'name': 'revoked_at', 'type': 'TIMESTAMPTZ'},
    ]
    if broken in {'missing_expiry', 'missing_revocation'}:
        missing = 'expires_at' if broken == 'missing_expiry' else 'revoked_at'
        columns = [column for column in columns if column['name'] != missing]
    if broken == 'sequential_id':
        columns[0]['type'] = 'BIGINT'
    policy = {'strategy': 'bearer_session', 'sessionTable': 'other' if broken == 'wrong_store' else 'sessions'}
    endpoints = [{'method': 'POST', 'path': '/api/v1/auth/logout', 'sessionPolicy': policy}]
    if broken == 'refresh':
        endpoints.append({'method': 'POST', 'path': '/api/v1/auth/refresh', 'sessionPolicy': policy})
    issues, _, _ = QaAgent(object())._check_cross_document_semantics(
        {}, {'tables': [{'name': 'sessions', 'columns': columns}]}, {'endpoints': endpoints}, [],
    )
    assert any('토큰 저장 구조' in issue for issue in issues) is bool(broken)


@pytest.mark.parametrize('link', ['principal', 'missing', 'cycle'])
def test_notification_preferences_can_belong_to_user_through_parent_foreign_keys(link):
    target = 'accounts' if link == 'principal' else 'reminders'
    owner_constraint = '' if link == 'missing' else f'FOREIGN_KEY REFERENCES {target}(id)'
    db = {'tables': [
        {'name': 'accounts', 'columns': [{'name': 'id'}, {'name': 'email'}]},
        {'name': 'assets', 'columns': [{'name': 'id'}, {'name': 'owner_id', 'constraints': owner_constraint}]},
        {'name': 'reminders', 'columns': [{'name': 'id'}, {'name': 'enabled'},
            {'name': 'asset_id', 'constraints': 'FOREIGN_KEY REFERENCES assets(id)'}]},
    ]}
    prd = {'coreFeatures': [{'name': '알림 설정', 'requirements': ['자산별 알림 설정 변경']} ]}
    issues, _, _ = QaAgent(object())._check_cross_document_semantics(prd, db, {'endpoints': []}, [])
    assert any('수신 설정 구조' in issue for issue in issues) is (link != 'principal')


def _guest_intake_feature():
    """비회원 접수: 공개 생성 + 접수번호 공개 조회 + 사장님 전용 목록."""
    return {
        "name": "수거 접수 등록",
        "featureId": "pickup.request",
        "applicability": {"ui": True, "api": True, "db": True},
        "ownership": {"scope": "SHARED", "ownerEntity": "users", "ownerKey": "owner_id",
                      "access": "고객은 receiptNumber로만 조회하고 OWNER는 자신의 접수만 조회한다"},
        "apiContract": [
            {"method": "POST", "path": "/api/v1/pickup-requests", "authRequired": False,
             "accessScope": "PUBLIC", "effects": [{"table": "pickup_requests", "kind": "insert"}]},
            {"method": "GET", "path": "/api/v1/public/pickup-requests/{receiptNumber}",
             "authRequired": False, "accessScope": "PUBLIC"},
        ],
    }


def test_explicitly_declared_guest_intake_create_is_allowed():
    from phase2.contract_quality import public_write_contract

    feature = _guest_intake_feature()
    endpoint = {"method": "POST", "path": "/api/v1/pickup-requests", "authRequired": False}
    assert public_write_contract(feature, endpoint)


def test_careless_public_writes_are_still_blocked():
    from phase2.contract_quality import public_write_contract

    endpoint = {"method": "POST", "path": "/api/v1/pickup-requests", "authRequired": False}

    # accessScope 미선언 — ownership.scope 상속으로는 인정하지 않는다.
    feature = _guest_intake_feature()
    feature["apiContract"][0].pop("accessScope")
    assert not public_write_contract(feature, endpoint)

    # 공개 조회 수단이 없는 반쪽 흐름.
    feature = _guest_intake_feature()
    feature["apiContract"] = feature["apiContract"][:1]
    assert not public_write_contract(feature, endpoint)

    # 기존 행을 바꾸는 effects.
    feature = _guest_intake_feature()
    feature["apiContract"][0]["effects"] = [{"table": "pickup_requests", "kind": "update"}]
    assert not public_write_contract(feature, endpoint)

    # 생성이 아닌 수정·삭제는 공개로 열지 않는다.
    feature = _guest_intake_feature()
    for method in ("PATCH", "DELETE", "PUT"):
        feature["apiContract"][0]["method"] = method
        assert not public_write_contract(
            feature, {"method": method, "path": "/api/v1/pickup-requests", "authRequired": False})

    # 인증이 필요한 operation은 애초에 대상이 아니다.
    feature = _guest_intake_feature()
    feature["apiContract"][0]["authRequired"] = True
    assert not public_write_contract(
        feature, {"method": "POST", "path": "/api/v1/pickup-requests", "authRequired": True})
