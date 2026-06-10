"""
한국어 키워드 검색 통합 테스트.
OpenAI 없이 PostgreSQL + kiwipiepy만 사용.
embedding 컬럼은 NULL로 삽입하여 순수 키워드 검색만 검증.
"""
import io
import uuid
import sys

if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "buffer"):
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import psycopg2
import psycopg2.extras
from kiwipiepy import Kiwi

DB_URL = "postgresql://user:password@localhost:5432/timiroom"

TEST_DOCS = [
    "사용자 인증 기능을 구현하기 위해 JWT 토큰 방식을 사용합니다.",
    "회원 가입 시 이메일 인증을 통해 계정을 활성화합니다.",
    "결제 시스템은 카카오페이와 토스페이먼츠를 연동하여 구현합니다.",
    "실시간 채팅 기능은 WebSocket을 이용하여 구현합니다.",
    "파일 업로드 기능은 AWS S3를 사용하여 처리합니다.",
    "관리자 대시보드에서 사용자 목록 및 통계를 확인할 수 있습니다.",
    "푸시 알림은 Firebase Cloud Messaging을 통해 발송합니다.",
    "검색 기능은 Elasticsearch를 활용한 전문 검색을 지원합니다.",
]

kiwi = Kiwi()
SEARCH_TAGS = {"NNG", "NNP", "NNB", "SL", "SH"}


def get_conn():
    return psycopg2.connect(DB_URL)


def setup(conn):
    with conn.cursor() as cur:
        cur.execute("DELETE FROM document_chunks WHERE metadata->>'source' = 'test'")
        for doc in TEST_DOCS:
            tokens_text = " ".join(t.form for t in kiwi.tokenize(doc))
            cur.execute(
                """
                INSERT INTO document_chunks (id, content, metadata, tokens)
                VALUES (%s, %s, %s::jsonb, to_tsvector('simple', %s))
                """,
                (str(uuid.uuid4()), doc, '{"source": "test"}', tokens_text),
            )
    conn.commit()
    print(f"[setup] {len(TEST_DOCS)}개 테스트 문서 삽입 완료\n")


def search(conn, query: str):
    tokens = [t.form for t in kiwi.tokenize(query) if t.tag in SEARCH_TAGS]
    ts_query = " & ".join(tokens)

    print(f"쿼리    : '{query}'")
    print(f"토큰    : {tokens}")
    print(f"tsquery : {ts_query}")

    if not ts_query:
        print("-> 검색어 없음\n")
        return

    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            """
            SELECT content,
                   ts_rank(tokens, to_tsquery('simple', %s)) AS rank
            FROM document_chunks
            WHERE tokens IS NOT NULL
              AND tokens @@ to_tsquery('simple', %s)
            ORDER BY rank DESC
            LIMIT 5
            """,
            (ts_query, ts_query),
        )
        rows = cur.fetchall()

    if rows:
        print(f"결과 ({len(rows)}개):")
        for i, row in enumerate(rows, 1):
            print(f"  {i}. [score={row['rank']:.4f}] {row['content']}")
    else:
        print("  결과 없음")
    print()


QUERIES = [
    "사용자 인증",
    "결제 카카오",
    "채팅 기능",
    "파일 업로드 S3",
    "알림 Firebase",
]

if __name__ == "__main__":
    try:
        conn = get_conn()
    except Exception as e:
        print(f"DB 연결 실패: {e}")
        sys.exit(1)

    try:
        setup(conn)
        print("=" * 60)
        for q in QUERIES:
            search(conn, q)
    finally:
        conn.close()
