import asyncio
import json
from unittest.mock import AsyncMock

import pytest

from phase2.agents.prd_agent import PrdAgent
from phase2.agents.qa_agent import QaAgent
from phase2.state import PipelineState


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


def test_prd_repair_restores_only_required_registry_extras():
    registry = [
        {'featureId': 'todo', 'name': '할 일 생성', 'source': 'prd_core', 'priority': 'P0'},
        {'featureId': 'error', 'name': '오류 안내', 'source': 'supporting', 'priority': 'P0'},
        {'featureId': 'optional', 'name': '색상 설정', 'source': 'supporting', 'priority': 'P2'},
    ]
    original = {'coreFeatures': [{'featureId': 'todo', 'name': '할 일 생성'}], 'mvpScope': {}}
    agent = object.__new__(PrdAgent)
    agent._call = AsyncMock(return_value=json.dumps({'patches': {'coreFeatures': [{'name': '할 일 생성'}]}}))
    state = PipelineState(feature_list=['할 일 생성'], feature_registry=registry, prd_document=json.dumps(original))
    result = json.loads(asyncio.run(agent.repair(state, '필수 오류 안내 계약 복구')).prd_document)
    ids = {feature.get('featureId') for feature in result['coreFeatures']}
    assert {'todo', 'error'} <= ids
    assert 'optional' not in ids


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
