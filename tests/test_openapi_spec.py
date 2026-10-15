import json

from phase2.agent_contract import normalize_api_path
from phase2.agents.api_agent import _api_feature_mappings, api_self_check
from phase2.agents.feature_spec_agent import _ensure_required_contracts
from phase2.feature_registry import normalize_feature_registry
from phase2.openapi_spec import build_openapi_document, validate_openapi_document
from phase3.schema_validator import SchemaValidator


def test_path_normalization_uses_kebab_case_and_one_version_prefix():
    assert normalize_api_path("/api/v1/api/v1/user_profiles/{user_id}") == (
        "/api/v1/user-profiles/{user_id}"
    )


def test_supporting_contract_does_not_expose_feature_id_in_path():
    feature = _ensure_required_contracts({
        "featureId": "supporting.supporting.14",
        "name": "입력값 검증",
        "source": "supporting",
        "actions": ["validate"],
        "apiContract": [],
        "dbContract": {"tables": [], "foreignKeys": []},
    })

    paths = {contract["path"] for contract in feature["apiContract"]}
    assert paths == {"/api/v1/validation-rules"}
    assert all("supporting" not in path and "feature-" not in path for path in paths)


def test_shared_route_keeps_all_feature_ids():
    registry = normalize_feature_registry([
        {
            "featureId": "items.search",
            "name": "물품 검색",
            "apiContract": [{"method": "GET", "path": "/api/v1/items", "action": "search"}],
        },
        {
            "featureId": "items.filter",
            "name": "물품 필터",
            "apiContract": [{"method": "GET", "path": "/api/v1/items", "action": "filter"}],
        },
    ], ["물품 검색", "물품 필터"], preserve_extra=True)

    endpoints, issues = api_self_check([{
        "method": "GET",
        "path": "/api/v1/items",
        "featureId": "items.search",
        "featureIds": ["items.search", "items.filter"],
        "description": "물품 검색 및 필터",
        "requestBody": "없음",
        "successResponse": "items: array — 검색 결과",
        "errorCodes": "400 — 잘못된 필터, 500 — 서버 오류",
    }], registry)

    assert not issues
    assert len(endpoints) == 1
    assert endpoints[0]["featureIds"] == ["items.search", "items.filter"]


def test_openapi_projection_is_valid_and_preserves_feature_links():
    internal = {
        "authentication": "JWT Bearer",
        "endpoints": [{
            "method": "POST",
            "path": "/api/v1/major_items",
            "featureId": "items.create",
            "featureIds": ["items.create", "items.validate"],
            "action": "create",
            "description": "전공 물품 등록",
            "authRequired": True,
            "requestBody": "name: string — 물품명, price: integer — 가격",
            "successResponse": "id: integer — 생성 ID, name: string — 물품명",
            "errorCodes": "400 — 요청 오류, 409 — 중복 등록, 500 — 서버 오류",
        }],
    }

    document = build_openapi_document(json.dumps(internal, ensure_ascii=False), "Major Mate")

    assert validate_openapi_document(document) == []
    assert document["openapi"] == "3.1.0"
    operation = document["paths"]["/api/v1/major-items"]["post"]
    assert operation["operationId"] == "createMajorItems"
    assert operation["x-feature-ids"] == ["items.create", "items.validate"]
    assert operation["requestBody"]["content"]["application/json"]["schema"]["properties"]["price"]["type"] == "integer"
    assert {"201", "400", "409", "500"} <= set(operation["responses"])


def test_feature_mappings_include_every_shared_endpoint_link():
    endpoints = [{
        "method": "GET",
        "path": "/api/v1/items",
        "featureId": "items.search",
        "featureIds": ["items.search", "items.filter"],
    }]
    registry = [
        {"featureId": "items.search", "name": "물품 검색"},
        {"featureId": "items.filter", "name": "물품 필터"},
    ]
    db_schema = json.dumps({
        "tables": [{"name": "items", "featureIds": ["items.search", "items.filter"]}],
    })

    mappings = _api_feature_mappings(endpoints, db_schema, [], registry)

    assert all(mapping["table"] == "items" for mapping in mappings)
    assert all(mapping["operations"] == [{"method": "GET", "path": "/api/v1/items"}] for mapping in mappings)


def test_phase3_contract_graph_accepts_shared_endpoint_feature_ids():
    registry = [
        {
            "featureId": "items.search",
            "apiContract": [{"method": "GET", "path": "/api/v1/items"}],
            "dbContract": {"tables": ["items"]},
        },
        {
            "featureId": "items.filter",
            "apiContract": [{"method": "GET", "path": "/api/v1/items"}],
            "dbContract": {"tables": ["items"]},
        },
    ]
    db = {
        "tables": [{
            "name": "items",
            "featureIds": ["items.search", "items.filter"],
            "columns": [{"name": "id", "type": "BIGINT", "constraints": "PRIMARY_KEY"}],
        }],
    }
    api = {
        "endpoints": [{
            "method": "GET",
            "path": "/api/v1/items",
            "featureId": "items.search",
            "featureIds": ["items.search", "items.filter"],
        }],
    }
    errors = []

    SchemaValidator._check_contract_graph(
        db, api, registry, lambda code, target, message: errors.append((code, target, message)),
    )

    assert errors == []
