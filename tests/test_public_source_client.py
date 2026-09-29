import asyncio
from types import SimpleNamespace

import httpx

from phase2.agents.search_agent import _select_public_search_keyword
from phase2.public_source_client import (
    CollectionReport,
    PublicSourceClient,
    SourceEvidence,
    _host_allowed,
    format_collection_report,
)


def test_allowlist_rejects_http_credentials_ip_and_lookalike_domains():
    allowed = ("go.kr", "kosis.kr")
    assert _host_allowed("https://kosis.kr/openapi/index.jsp", allowed)
    assert _host_allowed("https://www.example.go.kr/report", allowed)
    assert not _host_allowed("http://kosis.kr/report", allowed)
    assert not _host_allowed("https://user:pass@kosis.kr/report", allowed)
    assert not _host_allowed("https://127.0.0.1/report", allowed)
    assert not _host_allowed("https://kosis.kr.evil.example/report", allowed)


def test_report_is_assembled_by_python_with_source_evidence():
    report = CollectionReport(
        sources=(
            SourceEvidence(
                provider="KOSIS",
                title="1인 가구 통계표",
                url="https://kosis.kr/statHtml/statHtml.do?orgId=101&tblId=T1",
                identifier="101/T1",
                published_date="20260101",
                evidence="KOSIS 통합검색에서 확인된 통계표: 1인 가구 통계표",
            ),
        ),
        notices=("외부 AI 검색 API를 사용하지 않았습니다.",),
    )
    text = format_collection_report(report)
    assert "provider: KOSIS" in text
    assert "identifier: 101/T1" in text
    assert "외부 AI 검색 API를 사용하지 않고" in text


def test_kosis_catalog_response_is_parsed_without_llm():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.params["searchNm"] == "1인 가구 식품 관리"
        return httpx.Response(
            200,
            json=[{
                "TBL_NM": "가구원수별 가구",
                "ORG_ID": "101",
                "TBL_ID": "DT_TEST",
                "SEND_DE": "20260701",
                "LINK_URL": "http://kosis.kr/statHtml/statHtml.do?orgId=101&tblId=DT_TEST",
            }, {
                "TBL_NM": "단독가입상품 약정기간",
                "ORG_ID": "405",
                "TBL_ID": "DT_IRRELEVANT",
                "LINK_URL": "http://kosis.kr/statHtml/statHtml.do?orgId=405&tblId=DT_IRRELEVANT",
            }],
        )

    async def run_test():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
            client = PublicSourceClient(kosis_api_key="key", law_open_api_oc="", client=http_client)
            return await client._collect_kosis(http_client, "1인 가구 식품 관리")

    records = asyncio.run(run_test())
    assert records[0].provider == "KOSIS"
    assert records[0].identifier == "101/DT_TEST"
    assert records[0].url.startswith("https://kosis.kr/statHtml/")
    assert len(records) == 1


def test_kosis_numeric_cells_are_added_to_evidence():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("statisticsSearch.do"):
            return httpx.Response(200, json=[{
                "TBL_NM": "1인 가구 현황", "ORG_ID": "101", "TBL_ID": "DT_TEST",
                "LINK_URL": "https://kosis.kr/statHtml/statHtml.do?orgId=101&tblId=DT_TEST",
            }])
        assert request.url.path.endswith("statisticsParameterData.do")
        return httpx.Response(200, json=[{
            "DT": "8214.8", "PRD_DE": "2025", "ITM_NM": "가구수",
            "C1_NM": "1인가구", "UNIT_NM": "천가구",
        }])

    async def run_test():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
            client = PublicSourceClient(kosis_api_key="key", law_open_api_oc="", client=http_client)
            return await client._collect_kosis(http_client, "1인 가구")

    records = asyncio.run(run_test())
    assert "2025" in records[0].evidence
    assert "8214.8천가구" in records[0].evidence


def test_settings_factory_keeps_credentials_out_of_report_structure():
    settings = SimpleNamespace(
        kosis_api_key="secret-kosis",
        law_open_api_oc="secret-law",
        public_source_timeout_seconds=3,
        get_public_source_urls=list,
        get_public_source_domains=lambda: ["go.kr"],
    )
    client = PublicSourceClient.from_settings(settings)
    assert client._kosis_api_key == "secret-kosis"
    assert client._law_open_api_oc == "secret-law"


def test_statistical_keyword_prefers_target_group_over_product_domain():
    query = "혼자 사는 20-30대 1인 가구가 냉장고 재고를 관리하는 서비스"
    assert _select_public_search_keyword(query, "냉장 관리 플랫폼") == "1인 가구"


def test_law_selection_does_not_add_commerce_or_finance_without_matching_scope():
    assert PublicSourceClient._law_names_for_query("냉장고 재고와 유통기한을 관리하는 웹 서비스") == ("개인정보 보호법",)
    assert "전자상거래 등에서의 소비자보호에 관한 법률" in PublicSourceClient._law_names_for_query("상품 구매와 결제를 제공하는 쇼핑몰")
