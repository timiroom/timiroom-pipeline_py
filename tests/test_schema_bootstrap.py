"""docker/init.sql과 migrate_db.py가 같은 테이블을 만드는지 확인한다.

compose 스택은 init.sql만 실행하고(`/docker-entrypoint-initdb.d/`) migrate_db.py는
돌리지 않는다. 그래서 migrate_db.py에만 추가된 테이블은 로컬/compose 환경에서
런타임 UndefinedTable로 터진다. 실제로 pipeline_storage_receipts가 이렇게 누락돼
Phase 4 publish()가 저장 단계에서 실패했다.
"""

import io
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# migrate_db.py는 RAG 문서 테이블 이름을 설정에서 받아 sql.Identifier로 끼워 넣는다.
# 이 테이블만 이름이 환경에 따라 달라지므로 비교에서 제외한다.
_TEMPLATED = {"{}"}


def _created_tables(text: str) -> set[str]:
    return {
        name for name in re.findall(
            r"CREATE TABLE IF NOT EXISTS\s+([A-Za-z_{}]+)", text, re.IGNORECASE,
        )
    } - _TEMPLATED


def _read(relative: str) -> str:
    return io.open(ROOT / relative, encoding="utf-8").read()


def test_init_sql_creates_every_table_migrate_db_creates():
    init_tables = _created_tables(_read("docker/init.sql"))
    migrate_tables = _created_tables(_read("migrate_db.py"))

    missing = migrate_tables - init_tables
    assert not missing, (
        "docker/init.sql에 누락된 테이블: "
        f"{sorted(missing)} — compose 스택에서 런타임에 UndefinedTable로 실패한다"
    )


def test_phase4_storage_tables_are_bootstrapped():
    """Phase 4 저장 경로가 쓰는 두 테이블은 양쪽 스키마에 모두 있어야 한다."""
    init_tables = _created_tables(_read("docker/init.sql"))
    migrate_tables = _created_tables(_read("migrate_db.py"))

    for table in ("pipeline_outbox", "pipeline_storage_receipts"):
        assert table in init_tables, f"docker/init.sql에 {table} 없음"
        assert table in migrate_tables, f"migrate_db.py에 {table} 없음"


def test_document_chunks_has_the_unique_source_key_upsert_depends_on():
    """ingest_event는 ON CONFLICT (source_key)로 중복을 건너뛴다.

    source_key에 UNIQUE가 없으면 이 upsert가 InvalidColumnReference로 실패해
    consumer가 DLQ로만 빠지고 저장 확인이 영원히 pending에 머문다.
    """
    init_sql = _read("docker/init.sql")
    chunks = re.search(
        r"CREATE TABLE IF NOT EXISTS document_chunks\s*\((.*?)\n\);",
        init_sql, re.IGNORECASE | re.DOTALL,
    )
    assert chunks, "docker/init.sql에서 document_chunks 정의를 찾을 수 없음"
    source_key = next(
        line for line in chunks.group(1).splitlines() if "source_key" in line
    )
    assert "UNIQUE" in source_key.upper(), source_key.strip()
    assert "NOT NULL" in source_key.upper(), source_key.strip()
