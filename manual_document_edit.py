"""문서 편집(routers/document.py) 수동 검증 하네스 — 실제 K-EXAONE을 호출한다.

파이프라인 에이전트로 PRD / ERD / 기능 명세서 / API 명세서를 실제로 뽑은 뒤,
그 산출물을 편집 API에 그대로 물려서 EXAONE이 제대로 고치는지 사람이 눈으로 본다.

사용법:
  python manual_document_edit.py generate              # 4종 문서 생성 (수 분 소요, 토큰 씀)
  python manual_document_edit.py generate --only erd   # 일부만 다시 생성
  python manual_document_edit.py edit                  # 저장된 문서로 시나리오 전부 실행
  python manual_document_edit.py edit --doc api        # 한 종류만
  python manual_document_edit.py ask erd "users 테이블에 전화번호 컬럼 추가해줘"

생성물과 리포트는 samples/ 아래에 쌓인다.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

from openai import AsyncOpenAI

from config.settings import settings
from phase2.json_utils import try_parse_json
from phase2.state import PipelineState

SAMPLE_DIR = Path(__file__).parent / "samples"
DOC_TYPES = ("prd", "features", "erd", "api")

logger = logging.getLogger("manual_document_edit")


# ── 앱 부팅 없이 라우터만 쓰기 ────────────────────────────────────────
# routers/document.py는 `from main import exaone_client`를 호출 시점에 한다.
# main을 진짜로 import하면 DB·Kafka·리랭커까지 다 뜨므로, 필요한 두 개만 담은
# 가짜 main 모듈을 sys.modules에 미리 꽂아 그 지연 import를 가로챈다.

def _install_fake_main() -> AsyncOpenAI:
    client = AsyncOpenAI(
        api_key=settings.exaone_api_key,
        base_url="https://api.friendli.ai/dedicated/v1",
    )
    sys.modules.setdefault("main", SimpleNamespace(settings=settings, exaone_client=client))
    return client


def _setup_logging() -> list[logging.LogRecord]:
    """라우터가 남기는 판단 로그(어느 경로로 갔는지)를 리포트에도 담는다."""
    # Windows 콘솔 기본 코드페이지(cp949)로는 ✓ 같은 기호와 일부 한글 기호가
    # 인코딩 에러를 내며 로그 줄이 통째로 사라진다 — 출력만 UTF-8로 돌린다.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    collected: list[logging.LogRecord] = []

    class Collector(logging.Handler):
        def emit(self, record):
            if record.name.startswith("routers.document"):
                collected.append(record)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger().addHandler(Collector())
    return collected


# ── 1단계: 파이프라인 산출물 생성 ────────────────────────────────────
# PM 에이전트는 임베딩·DB가 필요해서 건너뛰고, 그 산출물(feature_list와
# dba/api instruction)만 사람이 쓴 값으로 채운 뒤 나머지 에이전트를 실제로 돌린다.

PROJECT = PipelineState(
    session_id="manual-check",
    project_name="타이미룸",
    platform="web",
    tech_stack=["Spring Boot", "React", "PostgreSQL"],
    problem_definition="스터디룸·공유오피스를 시간 단위로 빌리려면 업체마다 전화로 문의해야 하고, 실시간 빈자리를 알 수 없다.",
    target_users=["시험 기간에 스터디룸이 필요한 대학생", "회의 공간이 필요한 소규모 팀"],
    user_query=(
        "시간 단위로 스터디룸과 회의실을 예약하는 공간 대여 플랫폼을 만들고 싶습니다. "
        "이용자는 위치와 시간대로 빈 공간을 검색해 즉시 예약하고, 공간 사업자는 자기 공간의 "
        "예약 현황과 정산을 관리합니다."
    ),
    market_research=(
        "국내 공간 대여 시장은 2024년 약 1.2조 원 규모로 추정되며 연 15% 성장 중이다(출처: 한국공간산업협회, 2024년). "
        "이용자의 68%가 '실시간 예약 가능 여부 확인'을 가장 큰 불편으로 꼽았다(출처: 오픈서베이, 2024년). "
        "경쟁 서비스는 스페이스클라우드, 아워플레이스가 있으며 대부분 예약 확정까지 평균 4시간이 걸린다."
    ),
    feature_list=[
        "위치·시간대 기반 공간 검색",
        "실시간 예약 가능 여부 확인 및 즉시 예약",
        "예약 취소 및 환불 처리",
        "온라인 결제 (카드/간편결제)",
        "이용 후기 작성 및 평점",
        "공간 사업자용 예약 현황 대시보드",
        "공간 등록 및 요금표 관리",
        "정산 내역 조회",
    ],
    dba_instruction="공간 대여 예약 플랫폼의 DB 스키마를 설계하세요. 이용자, 공간, 예약, 결제, 후기, 정산을 다룹니다.",
    api_instruction="공간 대여 예약 플랫폼의 REST API를 설계하세요. 검색·예약·결제·후기·사업자 관리가 필요합니다.",
    context_prompt="공간 대여 예약 플랫폼. 이용자와 공간 사업자 두 역할이 있으며, 예약은 시간 단위로 이루어진다.",
)


async def _generate(only: tuple[str, ...]) -> None:
    from phase2.agents.api_agent import ApiAgent
    from phase2.agents.dba_agent import DbaAgent
    from phase2.agents.prd_agent import PrdAgent

    client = _install_fake_main()
    model = settings.exaone_endpoint_id
    SAMPLE_DIR.mkdir(exist_ok=True)

    # 기능 명세서는 PM 에이전트의 feature_list가 그대로 문서가 된다
    # (FeaturesPanel이 문자열 배열을 지원 — _FEATURES_PROFILE 주석 참고)
    if "features" in only:
        _save("features", {"features": list(PROJECT.feature_list)})

    async def run(name: str, agent, attr: str, wrap):
        started = time.monotonic()
        logger.info("▶ %s 생성 시작", name)
        state = await agent.execute(PROJECT)
        raw = getattr(state, attr)
        parsed = try_parse_json(raw)
        if not isinstance(parsed, dict):
            logger.error("✗ %s 생성 실패 — 파싱 불가: %.300s", name, raw)
            return
        _save(name, wrap(parsed))
        logger.info("✓ %s 생성 완료 (%.1f초)", name, time.monotonic() - started)

    jobs = []
    if "prd" in only:
        jobs.append(run("prd", PrdAgent(client, model), "prd_document", lambda d: d))
    if "erd" in only:
        jobs.append(run("erd", DbaAgent(client, model), "db_schema", lambda d: d))
    if "api" in only:
        jobs.append(run("api", ApiAgent(client, model), "api_spec", lambda d: d))

    # 세 에이전트는 서로 의존하지 않으므로 같이 돌린다
    results = await asyncio.gather(*jobs, return_exceptions=True)
    for r in results:
        if isinstance(r, Exception):
            logger.error("생성 중 오류: %s", r, exc_info=r)


def _save(name: str, doc: dict) -> None:
    path = SAMPLE_DIR / f"{name}.json"
    path.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("  저장: %s (%d자, 최상위 키 %s)",
                path.name, len(json.dumps(doc, ensure_ascii=False)), list(doc))


def _load(name: str) -> dict | None:
    path = SAMPLE_DIR / f"{name}.json"
    if not path.exists():
        logger.error("%s 없음 — 먼저 `python manual_document_edit.py generate` 를 돌리세요", path)
        return None
    return json.loads(path.read_text(encoding="utf-8"))


# ── 2단계: 편집 시나리오 ─────────────────────────────────────────────
# 각 시나리오는 여러 턴이 될 수 있다. 2턴짜리는 직전 대화가 재작성 단계까지
# 전달되는지(후속 요청이 무엇을 가리키는지 아는지) 확인하는 용도다.

SCENARIOS: dict[str, list[tuple[str, list[str]]]] = {
    "prd": [
        ("항목 하나 수정", ["KPI의 '실시간 예약 성공률' 목표 90%는 너무 공격적이에요. 현실적인 수치로 낮춰주세요."]),
        ("항목 추가", ["핵심 기능에 '노쇼 방지를 위한 예약금 제도'를 추가해줘"]),
        ("항목 삭제", ["사용자 페르소나에서 박상훈(중소기업 관리자)은 빼주세요"]),
        ("객체 섹션 수정", ["기술 스택에서 캐시를 Redis 말고 다른 걸로 바꾸고 이유도 다시 써줘"]),
        ("문자열 섹션 + 후속 대화", [
            "배경 문단이 너무 길어요. 줄여주세요.",
            "아니요, 그거 말고 수치랑 출처는 그대로 남기고 앞부분만 줄여주세요",
        ]),
        ("질문 (수정 아님)", ["이 PRD에서 가장 약한 부분이 어디인 것 같아요?"]),
    ],
    "features": [
        ("항목 추가", ["기능 목록에 '예약 알림 푸시' 기능을 추가해줘"]),
        ("항목 삭제", ["정산 내역 조회 기능은 MVP에서 빼주세요"]),
        ("항목 수정", ["세 번째 기능 이름을 더 구체적으로 바꿔줘"]),
    ],
    "erd": [
        ("컬럼 추가", ["users 테이블에 전화번호 컬럼을 추가해주세요"]),
        ("테이블 추가", ["예약 알림 발송 이력을 저장할 테이블을 하나 추가해줘"]),
        ("테이블 삭제", ["settlement_records 테이블은 지금 단계에선 빼주세요"]),
        ("관계 추가", ["관계 목록에 payments와 refunds 사이 1:N 관계를 추가해줘"]),
    ],
    "api": [
        # 생성물에 실제로 있는 결함들 — 20번은 자리표시자, 19번은 15번과 중복,
        # 13번은 description 자리에 파라미터 설명이 들어가 있다.
        ("망가진 항목 수정", ["20번 엔드포인트가 GET /api/v1/resource-19인데 이게 뭔지 모르겠어요. 제대로 된 API로 바꿔주세요."]),
        ("중복 항목 정리", ["POST /api/v1/spaces-2 가 POST /api/v1/spaces 랑 중복이에요. 정리해주세요."]),
        ("설명 오염 수정", ["GET /api/v1/reviews 의 description이 '조회할 사용자의 ID'로 잘못 들어가 있어요. 고쳐주세요."]),
        ("엔드포인트 추가", ["공간 즐겨찾기(찜하기) API가 없네요. 추가해주세요."]),
        ("여러 개 삭제", ["정산 조회 API 두 개는 빼주세요"]),
        ("후속 대화", [
            "공간 검색 API에 필터 파라미터를 더 넣어주세요",
            "가격대 필터도 같이 넣어주세요",
        ]),
    ],
}


def _summarize(before, after) -> str:
    if isinstance(before, list) and isinstance(after, list):
        kept = sum(1 for item in after if any(item == b for b in before))
        return (f"항목 {len(before)} → {len(after)}개 "
                f"(원본 그대로 {kept}개, 새로 쓰였거나 추가된 것 {len(after) - kept}개)")
    if isinstance(before, dict) and isinstance(after, dict):
        changed = [k for k in set(before) | set(after) if before.get(k) != after.get(k)]
        return f"키 {len(after)}개, 값이 바뀐 키: {changed or '없음'}"
    b, a = len(str(before or "")), len(str(after or ""))
    return f"문자열 {b} → {a}자"


_PLACEHOLDERS = ("$name", "{value}", "기능명", "테이블명", "TODO", "placeholder")


def _placeholder_hits(value) -> list[str]:
    text = json.dumps(value, ensure_ascii=False)
    return [p for p in _PLACEHOLDERS if p in text]


# 값·키 안에 JSON 문법 조각이 남아 있으면 파서가 깨진 응답을 '복구'해 넣은 것이다
# — 예: "DEFAULT_FALSE'}, {: : NOT NULL"
#
# 라우터보다는 넓게 잡되, `"key": "value"` 패턴은 뺐다: 모델이 requestBody에
# `{ "spaceId": "integer" }` 같은 정상 JSON 예시를 적는 일이 실제로 있어서
# 리포트가 오탐으로 덮인다(실측).
_JSON_LEAK = __import__("re").compile(r"[{}\[\]]\s*[,:]|[,:]\s*[{}\[\]]")


def _corrupted_strings(value, path: str = "") -> list[str]:
    """복구 파싱의 흔적이 남은 문자열·키를 찾아 경로와 함께 돌려준다.

    라우터의 탐지기와 일부러 따로 둔다 — 같은 코드를 쓰면 라우터가 놓치는 걸
    하네스도 똑같이 놓친다. 실제로 '값이 아니라 키가 깨진' 유형은 이 독립 검사로 잡혔다.
    """
    hits: list[str] = []
    if isinstance(value, str):
        if _JSON_LEAK.search(value):
            hits.append(f"{path}: {value[:80]!r}")
    elif isinstance(value, dict):
        for k, v in value.items():
            if isinstance(k, str) and _JSON_LEAK.search(k):
                hits.append(f"{path} 의 키: {k[:80]!r}")
            hits.extend(_corrupted_strings(v, f"{path}.{k}" if path else str(k)))
    elif isinstance(value, list):
        for i, v in enumerate(value):
            hits.extend(_corrupted_strings(v, f"{path}[{i}]"))
    return hits


def _nested_loss(before, after) -> list[str]:
    """리스트 항목 안의 중첩 리스트(컬럼·파라미터 등)가 조용히 줄었는지 본다."""
    losses: list[str] = []
    if not (isinstance(before, list) and isinstance(after, list)):
        return losses
    def label(item: dict) -> str:
        # method+path를 함께 써야 한다 — POST /reservations 와 GET /reservations 처럼
        # path가 같은 엔드포인트가 실제로 있어서, path만 쓰면 서로 다른 항목을 비교하게 된다
        parts = [str(item[k]) for k in ("method", "path", "name", "metric", "milestone") if item.get(k)]
        return " ".join(parts)

    by_label = {}
    for item in before:
        if isinstance(item, dict):
            by_label[label(item)] = item
    for item in after:
        if not isinstance(item, dict):
            continue
        origin = by_label.get(label(item))
        if not isinstance(origin, dict):
            continue
        for k, v in origin.items():
            new = item.get(k)
            if isinstance(v, list) and isinstance(new, list) and len(new) < len(v):
                losses.append(f"{label(item)}.{k}:{len(v)} → {len(new)}개")
            elif k not in item:
                losses.append(f"{label(item)}.{k}:키 자체가 사라짐")
    return losses


async def _run_scenario(doc_type: str, document: dict, label: str, turns: list[str], out: list[str]) -> None:
    from routers.document import DocumentEditRequest, edit_document

    out.append(f"\n### {label}\n")
    history: list[dict] = []

    for turn_no, instruction in enumerate(turns, start=1):
        prefix = f"턴 {turn_no}" if len(turns) > 1 else "요청"
        logger.info("── [%s] %s — %s: %s", doc_type, label, prefix, instruction)
        out.append(f"**{prefix}**: {instruction}\n")

        started = time.monotonic()
        try:
            resp = await edit_document(doc_type, DocumentEditRequest(
                document=document,
                instruction=instruction,
                history=history,
            ))
        except Exception as e:
            logger.error("호출 실패: %s", e, exc_info=True)
            out.append(f"> ❌ 호출 실패: `{e}`\n")
            return

        data = resp["data"]
        elapsed = time.monotonic() - started
        out.append(f"- 판정: `{data['intent']}` · 소요 {elapsed:.1f}초 · 수정 섹션 {len(data['edits'])}개\n")
        out.append(f"- 응답: {data['reply']}\n")

        for edit in data["edits"]:
            out.append(f"\n#### 섹션 `{edit['section']}` ({edit['label']})\n")
            out.append(f"- {_summarize(edit['before'], edit['after'])}\n")

            # 사람이 diff를 다 못 읽으므로 조용한 손실은 기계가 먼저 짚는다
            hits = _placeholder_hits(edit["after"])
            if hits:
                out.append(f"- ⚠️ placeholder 의심 문자열: {hits}\n")
            corrupted = _corrupted_strings(edit["after"])
            if corrupted:
                out.append(f"- 🔴 **깨진 값 {len(corrupted)}개** (복구 파싱 흔적):\n")
                out.extend(f"  - `{c}`\n" for c in corrupted[:5])
            losses = _nested_loss(edit["before"], edit["after"])
            if losses:
                out.append(f"- 🔴 **중첩 내용 소실 {len(losses)}건**:\n")
                out.extend(f"  - {c}\n" for c in losses[:8])

            raw_dir = SAMPLE_DIR / "raw"
            raw_dir.mkdir(exist_ok=True)
            safe = f"{doc_type}_{label}_{turn_no}_{edit['section']}".replace(" ", "_").replace("/", "_")
            (raw_dir / f"{safe}.json").write_text(
                json.dumps({"instruction": instruction, "before": edit["before"], "after": edit["after"]},
                           ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            changed = [d for d in edit["diff"] if d["type"] != "same"]
            out.append(f"- 변경 줄 {len(changed)}개 / 전체 {len(edit['diff'])}줄\n")
            out.append("\n```diff\n")
            for d in edit["diff"]:
                mark = {"same": " ", "add": "+", "del": "-"}[d["type"]]
                out.append(f"{mark} {d['text']}\n")
            out.append("```\n")

        # 다음 턴은 이번 대화를 이어받는다 (문서는 승인 전이므로 바뀌지 않는다)
        history.append({"role": "user", "content": instruction})
        history.append({"role": "assistant", "content": data["reply"]})


async def _edit(doc_types: tuple[str, ...], collected: list[logging.LogRecord]) -> None:
    _install_fake_main()
    out: list[str] = [f"# 문서 편집 검증 — {datetime.now():%Y-%m-%d %H:%M}\n"]

    for doc_type in doc_types:
        document = _load(doc_type)
        if document is None:
            continue
        out.append(f"\n## {doc_type} ({len(json.dumps(document, ensure_ascii=False)):,}자)\n")
        for label, turns in SCENARIOS.get(doc_type, []):
            mark = len(collected)
            await _run_scenario(doc_type, document, label, turns, out)
            path_logs = [r.getMessage() for r in collected[mark:] if r.levelno >= logging.INFO]
            if path_logs:
                out.append("\n<details><summary>라우터 판단 로그</summary>\n\n```\n")
                out.extend(line + "\n" for line in path_logs)
                out.append("```\n\n</details>\n")

    report = SAMPLE_DIR / f"report_{datetime.now():%Y%m%d_%H%M%S}.md"
    report.write_text("".join(out), encoding="utf-8")
    logger.info("리포트 저장: %s", report)


async def _ask(doc_type: str, instruction: str, collected: list[logging.LogRecord]) -> None:
    _install_fake_main()
    document = _load(doc_type)
    if document is None:
        return
    out: list[str] = [f"# 단발 요청 — {doc_type}\n"]
    await _run_scenario(doc_type, document, "단발 요청", [instruction], out)
    print("".join(out))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    gen = sub.add_parser("generate", help="파이프라인 에이전트로 문서 생성")
    gen.add_argument("--only", nargs="+", choices=DOC_TYPES, default=list(DOC_TYPES))

    ed = sub.add_parser("edit", help="저장된 문서로 편집 시나리오 실행")
    ed.add_argument("--doc", nargs="+", choices=DOC_TYPES, default=list(DOC_TYPES))

    ask = sub.add_parser("ask", help="문서 하나에 임의 요청 한 번")
    ask.add_argument("doc", choices=DOC_TYPES)
    ask.add_argument("instruction")

    args = parser.parse_args()
    collected = _setup_logging()

    if not settings.exaone_api_key or not settings.exaone_endpoint_id:
        logger.error("EXAONE 설정이 비어 있습니다 — .env의 EXAONE_API_KEY / EXAONE_ENDPOINT_ID 확인")
        sys.exit(1)

    if args.cmd == "generate":
        asyncio.run(_generate(tuple(args.only)))
    elif args.cmd == "edit":
        asyncio.run(_edit(tuple(args.doc), collected))
    else:
        asyncio.run(_ask(args.doc, args.instruction, collected))


if __name__ == "__main__":
    main()
