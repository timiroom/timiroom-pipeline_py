"""비-AI 공식 자료 수집기.

외부 생성형/검색 AI를 호출하지 않는다. KOSIS와 국가법령정보 공동활용 API,
그리고 명시적으로 허용된 공식 URL만 HTTP로 조회한다. 수집 결과는 LLM이 아니라
Python 코드가 검증하고 직렬화한다.
"""

from __future__ import annotations

import asyncio
import html
import ipaddress
import logging
import re
from dataclasses import dataclass
from html.parser import HTMLParser
from typing import Any
from urllib.parse import quote, urlparse

import httpx

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SourceEvidence:
    provider: str
    title: str
    url: str
    evidence: str
    identifier: str = ""
    published_date: str = ""


@dataclass(frozen=True)
class CollectionReport:
    sources: tuple[SourceEvidence, ...]
    notices: tuple[str, ...]


class _VisibleTextParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.title = ""
        self.description = ""
        self._in_title = False
        self._ignored_depth = 0
        self._chunks: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        lower = tag.lower()
        if lower in {"script", "style", "noscript", "svg"}:
            self._ignored_depth += 1
        if lower == "title":
            self._in_title = True
        if lower == "meta":
            values = {key.lower(): (value or "") for key, value in attrs}
            if values.get("name", "").lower() == "description" or values.get("property", "").lower() == "og:description":
                self.description = values.get("content", "")

    def handle_endtag(self, tag: str) -> None:
        lower = tag.lower()
        if lower == "title":
            self._in_title = False
        if lower in {"script", "style", "noscript", "svg"} and self._ignored_depth:
            self._ignored_depth -= 1

    def handle_data(self, data: str) -> None:
        if self._ignored_depth:
            return
        cleaned = _clean_text(data)
        if not cleaned:
            return
        if self._in_title:
            self.title = f"{self.title} {cleaned}".strip()
        elif len(" ".join(self._chunks)) < 4000:
            self._chunks.append(cleaned)

    @property
    def visible_text(self) -> str:
        return _clean_text(" ".join(self._chunks))


def _clean_text(value: Any, limit: int = 700) -> str:
    text = html.unescape(str(value or ""))
    text = re.sub(r"\s+", " ", text).strip()
    return text[:limit]


def _host_allowed(url: str, allowed_domains: tuple[str, ...]) -> bool:
    try:
        parsed = urlparse(url)
        if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password:
            return False
        host = parsed.hostname.lower().rstrip(".")
        try:
            ipaddress.ip_address(host)
            return False
        except ValueError:
            pass
        return any(host == domain or host.endswith(f".{domain}") for domain in allowed_domains)
    except ValueError:
        return False


class PublicSourceClient:
    """공식 API/URL만 조회하는 결정론적 수집기."""

    _LAW_NAMES = (
        "개인정보 보호법",
        "전자상거래 등에서의 소비자보호에 관한 법률",
        "전자금융거래법",
    )

    def __init__(
        self,
        *,
        kosis_api_key: str = "",
        law_open_api_oc: str = "",
        seed_urls: tuple[str, ...] = (),
        allowed_domains: tuple[str, ...] = ("kosis.kr", "law.go.kr", "data.go.kr", "go.kr", "or.kr", "ac.kr"),
        timeout_seconds: float = 8.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._kosis_api_key = kosis_api_key.strip()
        self._law_open_api_oc = law_open_api_oc.strip()
        self._seed_urls = tuple(url.strip() for url in seed_urls if url.strip())
        self._allowed_domains = tuple(d.lower().strip().lstrip(".") for d in allowed_domains if d.strip())
        self._timeout = timeout_seconds
        self._client = client

    @classmethod
    def from_settings(cls, settings: Any) -> "PublicSourceClient":
        return cls(
            kosis_api_key=getattr(settings, "kosis_api_key", ""),
            law_open_api_oc=getattr(settings, "law_open_api_oc", ""),
            seed_urls=tuple(getattr(settings, "get_public_source_urls")()),
            allowed_domains=tuple(getattr(settings, "get_public_source_domains")()),
            timeout_seconds=float(getattr(settings, "public_source_timeout_seconds", 8.0)),
        )

    async def collect(self, query: str, domain: str) -> CollectionReport:
        notices: list[str] = ["외부 AI 검색 API를 사용하지 않았습니다."]
        owns_client = self._client is None
        client = self._client or httpx.AsyncClient(
            timeout=httpx.Timeout(self._timeout),
            follow_redirects=True,
            headers={"User-Agent": "TimiroomPublicSourceCollector/1.0"},
        )
        try:
            tasks: list[asyncio.Future | asyncio.Task | Any] = []
            labels: list[str] = []
            if self._kosis_api_key:
                tasks.append(self._collect_kosis(client, domain or query))
                labels.append("KOSIS")
            else:
                notices.append("KOSIS_API_KEY가 없어 KOSIS 통합검색을 건너뛰었습니다.")

            law_names = self._law_names_for_query(query)
            if self._law_open_api_oc:
                tasks.append(self._collect_laws_via_api(client, law_names))
                labels.append("국가법령정보 API")
            else:
                tasks.append(self._collect_public_law_pages(client, law_names))
                labels.append("국가법령정보 원문")
                notices.append("LAW_OPEN_API_OC가 없어 공개 법령 원문 페이지 확인만 시도했습니다.")

            if self._seed_urls:
                tasks.append(self._collect_seed_urls(client))
                labels.append("공식 URL")
            else:
                notices.append("PUBLIC_SOURCE_URLS가 비어 있어 프로젝트별 공식 자료 수집은 생략했습니다.")

            results = await asyncio.gather(*tasks, return_exceptions=True)
            sources: list[SourceEvidence] = []
            for label, result in zip(labels, results):
                if isinstance(result, Exception):
                    logger.warning("공식 자료 수집 실패 (%s): %s", label, result)
                    notices.append(f"{label} 수집 실패: {type(result).__name__}")
                else:
                    sources.extend(result)
            return CollectionReport(tuple(_deduplicate(sources)), tuple(notices))
        finally:
            if owns_client:
                await client.aclose()

    async def _collect_kosis(self, client: httpx.AsyncClient, keyword: str) -> list[SourceEvidence]:
        response = await client.get(
            "https://kosis.kr/openapi/statisticsSearch.do",
            params={
                "method": "getList",
                "apiKey": self._kosis_api_key,
                "searchNm": _clean_text(keyword, 100),
                "sort": "RANK",
                "startCount": "1",
                "resultCount": "5",
                "format": "json",
                "jsonVD": "Y",
            },
        )
        response.raise_for_status()
        payload = response.json()
        rows = payload if isinstance(payload, list) else payload.get("result", payload.get("data", []))
        catalog_rows: list[tuple[dict, str, str, str, str, str]] = []
        keyword_tokens = [token for token in re.findall(r"[가-힣A-Za-z0-9]+", keyword) if len(token) >= 2]
        for row in rows if isinstance(rows, list) else []:
            if not isinstance(row, dict):
                continue
            title = _clean_text(row.get("TBL_NM") or row.get("title") or row.get("TITLE"))
            org_id = _clean_text(row.get("ORG_ID"), 80)
            table_id = _clean_text(row.get("TBL_ID"), 80)
            if not title:
                continue
            # 통합검색은 본문에서 우연히 질의어가 걸린 비관련 표도 반환한다. 최종 근거에는
            # 통계표 제목이 검색 개념을 직접 포함하는 결과만 보존한다.
            if keyword_tokens and not any(token.lower() in title.lower() for token in keyword_tokens):
                continue
            url = str(row.get("LINK_URL") or row.get("URL") or "").strip()
            if not url and org_id and table_id:
                url = f"https://kosis.kr/statHtml/statHtml.do?orgId={quote(org_id)}&tblId={quote(table_id)}"
            # KOSIS 통합검색은 2026년에도 일부 LINK_URL을 http로 반환하지만 실제 페이지는
            # HTTPS를 지원한다. 다른 호스트는 건드리지 않고 KOSIS 공식 호스트만 승격한다.
            parsed_url = urlparse(url)
            if parsed_url.scheme == "http" and parsed_url.hostname in {"kosis.kr", "www.kosis.kr"}:
                url = parsed_url._replace(scheme="https").geturl()
            if not _host_allowed(url, self._allowed_domains):
                continue
            updated = _clean_text(row.get("SEND_DE") or row.get("sendDe"), 30)
            catalog_rows.append((row, title, org_id, table_id, url, updated))

        results = await asyncio.gather(*(
            self._fetch_kosis_values(client, *item) for item in catalog_rows[:3]
        ), return_exceptions=True)
        records: list[SourceEvidence] = []
        for item, result in zip(catalog_rows[:3], results):
            row, title, org_id, table_id, url, updated = item
            if isinstance(result, SourceEvidence):
                records.append(result)
            else:
                if isinstance(result, Exception):
                    logger.warning("KOSIS 실제 수치 조회 실패 (%s/%s): %s", org_id, table_id, result)
                evidence = f"KOSIS 통합검색에서 확인된 통계표(실제 수치 조회 실패): {title}"
                records.append(SourceEvidence("KOSIS", title, url, evidence, f"{org_id}/{table_id}".strip("/"), updated))
        return records

    async def _fetch_kosis_values(
        self, client: httpx.AsyncClient, row: dict, title: str, org_id: str,
        table_id: str, url: str, updated: str,
    ) -> SourceEvidence | None:
        """Fetch recent numeric cells for a catalog result using KOSIS's official table API."""
        if not org_id or not table_id:
            return None
        base_params = {
            "method": "getList", "apiKey": self._kosis_api_key,
            "orgId": org_id, "tblId": table_id,
            "itmId": "all", "objL1": "all",
            "prdSe": _clean_text(row.get("PRD_SE") or row.get("prdSe") or "Y", 3),
            "newEstPrdCnt": "1", "format": "json", "jsonVD": "Y", "smblChk": "Y",
        }
        rows = []
        # Some tables have two or three classification levels. KOSIS requires every
        # existing level to be selected, while rejecting levels the table does not have.
        for depth in range(1, 4):
            params = dict(base_params)
            for level in range(2, depth + 1):
                params[f"objL{level}"] = "all"
            response = await client.get(
                "https://kosis.kr/openapi/Param/statisticsParameterData.do", params=params,
            )
            response.raise_for_status()
            payload = response.json()
            candidate_rows = payload if isinstance(payload, list) else payload.get("result", payload.get("data", [])) if isinstance(payload, dict) else []
            if isinstance(candidate_rows, list) and candidate_rows:
                rows = candidate_rows
                break
        values = []
        for value_row in rows if isinstance(rows, list) else []:
            if not isinstance(value_row, dict):
                continue
            value = _clean_text(value_row.get("DT"), 80)
            if not value or not re.search(r"\d", value):
                continue
            period = _clean_text(value_row.get("PRD_DE"), 30)
            item_name = _clean_text(value_row.get("ITM_NM"), 120)
            dimensions = [
                _clean_text(value_row.get(key), 100)
                for key in ("C1_NM", "C2_NM", "C3_NM") if value_row.get(key)
            ]
            unit = _clean_text(value_row.get("UNIT_NM"), 40)
            label = " / ".join([part for part in [item_name, *dimensions] if part]) or title
            values.append(f"{period} {label}: {value}{unit}")
            if len(values) >= 3:
                break
        if not values:
            return None
        evidence = "KOSIS 실제 통계값 — " + "; ".join(values)
        return SourceEvidence("KOSIS", title, url, evidence, f"{org_id}/{table_id}", updated)

    @classmethod
    def _law_names_for_query(cls, query: str) -> tuple[str, ...]:
        names = ["개인정보 보호법"]
        if any(token in query for token in ("결제", "구매", "판매", "쇼핑몰", "전자상거래", "구독료")):
            names.append("전자상거래 등에서의 소비자보호에 관한 법률")
        if any(token in query for token in ("전자금융", "송금", "은행", "계좌", "간편결제", "금융거래")):
            names.append("전자금융거래법")
        return tuple(name for name in cls._LAW_NAMES if name in names)

    async def _collect_laws_via_api(
        self, client: httpx.AsyncClient, law_names: tuple[str, ...]
    ) -> list[SourceEvidence]:
        results = await asyncio.gather(*(self._search_one_law(client, name) for name in law_names))
        return [item for group in results for item in group]

    async def _search_one_law(self, client: httpx.AsyncClient, law_name: str) -> list[SourceEvidence]:
        response = await client.get(
            "https://www.law.go.kr/DRF/lawSearch.do",
            params={
                "OC": self._law_open_api_oc,
                "target": "law",
                "type": "JSON",
                "search": "1",
                "query": law_name,
                "display": "3",
            },
        )
        response.raise_for_status()
        payload = response.json()
        root = payload.get("LawSearch", payload) if isinstance(payload, dict) else {}
        rows = root.get("law", []) if isinstance(root, dict) else []
        if isinstance(rows, dict):
            rows = [rows]
        records: list[SourceEvidence] = []
        for row in rows if isinstance(rows, list) else []:
            title = _clean_text(row.get("법령명한글") or row.get("법령명_한글") or row.get("법령명"))
            if not title:
                continue
            law_id = _clean_text(row.get("법령ID") or row.get("법령일련번호"), 80)
            url = f"https://www.law.go.kr/법령/{quote(title)}"
            date = _clean_text(row.get("시행일자") or row.get("공포일자"), 30)
            articles = await self._fetch_law_articles(client, law_id, law_name)
            evidence = f"국가법령정보 공동활용 API 현행 본문"
            if articles:
                evidence += " — " + "; ".join(articles)
            else:
                evidence += f"에서 법령만 확인: {title} (관련 조항 조회 실패)"
            if date:
                evidence += f" (기준일 {date})"
            records.append(SourceEvidence("국가법령정보", title, url, evidence, law_id, date))
        return records

    async def _fetch_law_articles(
        self, client: httpx.AsyncClient, law_id: str, law_name: str,
    ) -> list[str]:
        if not law_id:
            return []
        response = await client.get(
            "https://www.law.go.kr/DRF/lawService.do",
            params={"OC": self._law_open_api_oc, "target": "law", "type": "JSON", "ID": law_id},
        )
        response.raise_for_status()
        payload = response.json()
        candidates: list[tuple[str, str, str]] = []

        def visit(node: Any) -> None:
            if isinstance(node, dict):
                content = _clean_text(node.get("조문내용"), 500)
                heading = _clean_text(node.get("조문제목"), 100)
                if (content and len(content) > len(heading) + 10
                        and not re.match(r"^제\s*\d+\s*장\b", content)):
                    candidates.append((
                        _clean_text(node.get("조문번호"), 30),
                        heading,
                        content,
                    ))
                for value in node.values():
                    visit(value)
            elif isinstance(node, list):
                for value in node:
                    visit(value)

        visit(payload)
        common_tokens = ("개인정보", "동의", "처리", "보호", "안전", "파기", "소비자", "거래", "금융")
        preferred = [item for item in candidates if any(token in f"{item[1]} {item[2]}" for token in common_tokens)]
        selected = preferred[:3] or candidates[:2]
        result = []
        seen = set()
        for number, heading, content in selected:
            key = (number, heading, content)
            if key in seen:
                continue
            seen.add(key)
            article_label = f"제{number}조" if number else "관련 조항"
            if heading:
                article_label += f"({heading})"
            result.append(f"{article_label}: {content}")
        return result

    async def _collect_public_law_pages(
        self, client: httpx.AsyncClient, law_names: tuple[str, ...]
    ) -> list[SourceEvidence]:
        urls = tuple(f"https://www.law.go.kr/법령/{quote(name)}" for name in law_names)
        return await self._fetch_official_pages(client, urls, "국가법령정보")

    async def _collect_seed_urls(self, client: httpx.AsyncClient) -> list[SourceEvidence]:
        safe_urls = tuple(url for url in self._seed_urls if _host_allowed(url, self._allowed_domains))
        rejected = len(self._seed_urls) - len(safe_urls)
        if rejected:
            logger.warning("허용 도메인 밖의 PUBLIC_SOURCE_URLS %d개를 거부했습니다.", rejected)
        return await self._fetch_official_pages(client, safe_urls[:10], "공식 웹 자료")

    async def _fetch_official_pages(
        self, client: httpx.AsyncClient, urls: tuple[str, ...], provider: str
    ) -> list[SourceEvidence]:
        results = await asyncio.gather(*(self._fetch_page(client, url, provider) for url in urls), return_exceptions=True)
        records: list[SourceEvidence] = []
        for result in results:
            if isinstance(result, SourceEvidence):
                records.append(result)
            elif isinstance(result, Exception):
                logger.warning("공식 페이지 확인 실패: %s", result)
        return records

    async def _fetch_page(self, client: httpx.AsyncClient, url: str, provider: str) -> SourceEvidence | None:
        if not _host_allowed(url, self._allowed_domains):
            return None
        response = await client.get(url)
        response.raise_for_status()
        final_url = str(response.url)
        if not _host_allowed(final_url, self._allowed_domains):
            raise ValueError("허용되지 않은 도메인으로 리디렉션됨")
        content_type = response.headers.get("content-type", "").lower()
        if "html" not in content_type:
            return None
        parser = _VisibleTextParser()
        parser.feed(response.text[:500_000])
        title = _clean_text(parser.title, 200) or urlparse(final_url).path.rsplit("/", 1)[-1]
        evidence = _clean_text(parser.description or parser.visible_text)
        if len(evidence) < 20:
            return None
        return SourceEvidence(provider, title, final_url, evidence)


def _deduplicate(records: list[SourceEvidence]) -> list[SourceEvidence]:
    result: list[SourceEvidence] = []
    seen: set[str] = set()
    for record in records:
        key = record.url.lower().rstrip("/")
        if key in seen:
            continue
        seen.add(key)
        result.append(record)
    return result


def format_collection_report(report: CollectionReport) -> str:
    lines = [
        "[수집 정책]",
        "외부 AI 검색 API를 사용하지 않고 공식 OpenAPI와 허용된 공식 웹페이지만 조회했습니다.",
    ]
    if report.sources:
        lines.append("\n[검증된 공식 출처]")
        for index, source in enumerate(report.sources, 1):
            lines.extend(
                [
                    f"SOURCE {index}",
                    f"provider: {source.provider}",
                    f"title: {source.title}",
                    f"url: {source.url}",
                    f"identifier: {source.identifier or '없음'}",
                    f"publishedDate: {source.published_date or '확인되지 않음'}",
                    f"evidence: {source.evidence}",
                ]
            )
    else:
        lines.extend(["\n[검증된 공식 출처]", "수집된 공식 원문이 없습니다. 외부 통계나 법적 사실을 추정하지 않습니다."])
    lines.append("\n[수집 상태]")
    lines.extend(f"- {notice}" for notice in report.notices)
    return "\n".join(lines)
