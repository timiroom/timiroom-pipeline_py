import psycopg2
from config.settings import settings

conn = psycopg2.connect(settings.db_url)
cur = conn.cursor()
cur.execute(
    "SELECT column_name, data_type FROM information_schema.columns "
    "WHERE table_name = 'document_chunks' ORDER BY ordinal_position"
)
rows = cur.fetchall()
print("document_chunks columns:")
for r in rows:
    print(f"  {r[0]}: {r[1]}")

cur.execute(
    "SELECT indexname, indexdef FROM pg_indexes WHERE tablename = 'document_chunks'"
)
print("\nindexes:")
for r in cur.fetchall():
    print(f"  {r[0]}: {r[1]}")

conn.close()
