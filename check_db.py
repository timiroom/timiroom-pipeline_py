import psycopg2
from config.settings import settings

table_name = settings.get_rag_document_table()
conn = psycopg2.connect(settings.db_url)
cur = conn.cursor()
cur.execute(
    "SELECT column_name, data_type FROM information_schema.columns "
    "WHERE table_name = %s ORDER BY ordinal_position",
    (table_name,),
)
rows = cur.fetchall()
print(f"{table_name} columns:")
for r in rows:
    print(f"  {r[0]}: {r[1]}")

cur.execute(
    "SELECT indexname, indexdef FROM pg_indexes WHERE tablename = %s",
    (table_name,),
)
print("\nindexes:")
for r in cur.fetchall():
    print(f"  {r[0]}: {r[1]}")

conn.close()
