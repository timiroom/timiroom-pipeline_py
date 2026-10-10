import asyncio
import json
from unittest.mock import AsyncMock

import pytest

from phase2.agents.api_agent import ApiAgent
from phase2.agents.dba_agent import DbaAgent
from phase2.state import PipelineState


def test_targeted_db_repair_preserves_registry_name_mapping_and_metadata():
    registry = [{
        'featureId': 'feature_003.error-handling', 'name': '오류 기록',
        'actions': ['create'], 'dbContract': {'tables': ['feature_003_records'], 'foreignKeys': []},
    }]
    original = {
        'dialect': 'PostgreSQL',
        'tables': [{'name': 'feature_003_records', 'description': '오류 기록', 'columns': [
            {'name': 'id', 'type': 'BIGINT', 'constraints': 'PRIMARY_KEY'},
            {'name': 'message', 'type': 'TEXT', 'constraints': 'NOT_NULL'},
        ], 'indexes': []}],
        'relationships': [],
        'featureMappings': [{'featureId': 'feature_003.error-handling', 'featureName': '오류 기록', 'table': 'feature_003_records'}],
    }
    agent = object.__new__(DbaAgent)
    agent._call = AsyncMock(return_value=json.dumps({'patches': {
        'feature_003_records': {'description': '오류 기록과 원인 안내'},
    }}))
    state = PipelineState(feature_list=['오류 기록'], feature_registry=registry,
                          feature_specs=registry, db_schema=json.dumps(original))
    result = json.loads(asyncio.run(agent.repair(state, '오류 기록 설명 보완')).db_schema)
    assert result.get('dialect') == 'PostgreSQL'
    assert result['tables'][0]['name'] == 'feature_003_records'
    assert result.get('featureMappings')
    assert result['featureMappings'][0]['table'] == 'feature_003_records'
    assert result['tables'][0]['description'] == '오류 기록과 원인 안내'


@pytest.mark.parametrize('action', ['login', 'signup'])
def test_auth_contract_and_targeted_repair_keep_transaction_rules(action):
    route = '/api/v1/auth/' + action
    registry = [{'featureId': 'auth.' + action, 'name': action, 'actions': [action],
                 'apiContract': [{'method': 'POST', 'path': route, 'action': action}],
                 'dbContract': {'tables': ['users'], 'foreignKeys': []}}]
    schema = json.dumps({'tables': [{'name': 'users', 'columns': [
        {'name': 'id', 'type': 'BIGINT', 'constraints': 'PRIMARY_KEY'}]}]})
    agent = object.__new__(ApiAgent)
    managed = asyncio.run(agent._manager_review_node({
        'ctx': {'feature_registry': json.dumps(registry), 'db_schema': schema},
        'endpoints': [{'method': 'POST', 'path': route, 'featureId': 'auth.' + action, 'action': action}],
        'authentication': 'JWT',
    }))
    endpoint = json.loads(managed['api_spec'])['endpoints'][0]
    assert endpoint.get('transactionRules')
    agent._call = AsyncMock(return_value=json.dumps({'patches': [
        {'method': 'POST', 'path': route, 'transactionRules': ''}]}))
    state = PipelineState(feature_list=[action], feature_registry=registry, db_schema=schema,
                          api_spec=json.dumps({'endpoints': [endpoint], 'metadata': 'preserved'}))
    repaired = json.loads(asyncio.run(agent.repair(state, '응답 계약 보완')).api_spec)
    assert repaired['metadata'] == 'preserved'
    assert len(repaired['endpoints']) == 1
    assert repaired['endpoints'][0]['transactionRules']
    assert repaired['endpoints'][0]['featureId'] == 'auth.' + action
    assert repaired.get('featureMappings')

@pytest.mark.parametrize('method,path', [('PATCH', '/api/v1/users/me'), ('PUT', '/api/v1/users/me'), ('DELETE', '/api/v1/users/me')])
def test_account_mutations_have_transaction_contract_without_extra_routes(method, path):
    from phase2.agents.api_agent import _align_endpoints_to_db
    items = _align_endpoints_to_db([{'method': method, 'path': path, 'featureId': 'account',
                                   'action': 'update', 'requestBody': 'nickname: string'}], '{}')
    assert len(items) == 1
    assert items[0]['transactionRules']
    assert items[0]['featureId'] == 'account'
    assert items[0]['requestBody'] == 'nickname: string'


def test_blank_repair_rule_does_not_erase_existing_custom_transaction():
    agent = object.__new__(ApiAgent)
    agent._call = AsyncMock(return_value=json.dumps({'patches': [
        {'method': 'POST', 'path': '/api/v1/auth/login', 'transactionRules': ''}]}))
    state = PipelineState(api_spec=json.dumps({'endpoints': [{
        'method': 'POST', 'path': '/api/v1/auth/login', 'transactionRules': 'custom atomic rule',
        'requestBody': 'otp: string', 'featureId': 'auth.login', 'action': 'login',
    }]}))
    result = json.loads(asyncio.run(agent.repair(state, '설명 보완')).api_spec)
    assert result['endpoints'][0]['transactionRules'] == 'custom atomic rule'
    assert result['endpoints'][0]['requestBody'] == 'otp: string'



def test_repair_mapping_keeps_uuid_owner_fk_consistent():
    registry = [{'featureId': 'tasks', 'name': '업무 생성', 'actions': ['create'],
                 'ownership': {'scope': 'USER', 'ownerEntity': 'users', 'ownerKey': 'user_id'},
                 'dbContract': {'tables': ['tasks'], 'foreignKeys': []}}]
    original = {'tables': [
        {'name': 'users', 'columns': [{'name': 'id', 'type': 'UUID', 'constraints': 'PRIMARY_KEY'}]},
        {'name': 'tasks', 'description': '업무 생성', 'columns': [
            {'name': 'id', 'type': 'BIGINT', 'constraints': 'PRIMARY_KEY'},
            {'name': 'user_id', 'type': 'UUID', 'constraints': 'REFERENCES users(id)'}]},
    ], 'relationships': ['users (1:N) tasks']}
    agent = object.__new__(DbaAgent)
    agent._call = AsyncMock(return_value=json.dumps({'patches': {'tasks': {'description': '업무 생성 설명'}}}))
    state = PipelineState(feature_list=['업무 생성'], feature_registry=registry,
                          feature_specs=registry, db_schema=json.dumps(original))
    result = json.loads(asyncio.run(agent.repair(state, '설명 보완')).db_schema)
    task = next(table for table in result['tables'] if table['name'] == 'tasks')
    owner = next(column for column in task['columns'] if column['name'] == 'user_id')
    assert owner['type'] == 'UUID'
