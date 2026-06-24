"""Check exact column names in tickets_7 and tickets_8."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from dotenv import load_dotenv; load_dotenv()
except ImportError:
    pass

import psycopg2
from urllib.parse import urlparse, unquote

DB_URL = os.getenv("DB_URL") or (
    "postgresql://postgres.trssafmvdbeagsdnivcl:"
    "digitalshield%4023070802"
    "@aws-1-ap-south-1.pooler.supabase.com:6543/postgres"
)
p = urlparse(DB_URL)
conn = psycopg2.connect(
    host=p.hostname, port=p.port or 5432,
    dbname=p.path.lstrip("/"),
    user=p.username, password=unquote(p.password or ""),
    sslmode="require", options="-c search_path=public"
)
cur = conn.cursor()

for tbl in ["tickets_7", "tickets_8"]:
    cur.execute("""
        SELECT column_name, data_type
        FROM information_schema.columns
        WHERE table_schema='public' AND table_name=%s
        ORDER BY ordinal_position
    """, (tbl,))
    cols = cur.fetchall()
    print(f"\n=== {tbl} columns ===")
    for c in cols:
        print(f"  {c[0]} ({c[1]})")
    
    # Sample 2 rows
    cur.execute(f"SELECT * FROM {tbl} LIMIT 2")
    rows = cur.fetchall()
    colnames = [desc[0] for desc in cur.description]
    print(f"  Sample row keys: {colnames}")

conn.close()
