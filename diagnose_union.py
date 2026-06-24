"""Test exact UNION ALL SQL against the real DB to diagnose the column error."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from dotenv import load_dotenv; load_dotenv()
except ImportError:
    pass

# Use the ticket_model's own DB_URL (Supabase pooler)
DB_URL = (
    "postgresql://postgres.trssafmvdbeagsdnivcl:"
    "digitalshield%4023070802"
    "@aws-1-ap-south-1.pooler.supabase.com:6543/postgres"
)

import psycopg2, urllib.parse

# Parse URL to psycopg2 args
from urllib.parse import urlparse, unquote
p = urlparse(DB_URL)
conn = psycopg2.connect(
    host=p.hostname, port=p.port,
    dbname=p.path.lstrip("/"),
    user=p.username,
    password=unquote(p.password),
    sslmode="require",
    options="-c search_path=public"
)

cur = conn.cursor()

# Check search_path
cur.execute("SHOW search_path")
print("search_path:", cur.fetchone())

# Check columns of tickets_1 directly
cur.execute("SELECT column_name FROM information_schema.columns WHERE table_name='tickets_1' AND table_schema='public' ORDER BY ordinal_position")
cols = [r[0] for r in cur.fetchall()]
print("tickets_1 columns:", cols)

# Try a simple single-table select
print("\nTesting single table SELECT...")
cur.execute("SELECT user_id, psgn_name FROM tickets_1 LIMIT 2")
rows = cur.fetchall()
print("tickets_1 rows:", rows)

# Try UNION ALL for just 2 tables
print("\nTesting 2-table UNION ALL...")
sql = """
SELECT user_id, psgn_name, train_number FROM tickets_1
UNION ALL
SELECT user_id, psgn_name, train_number FROM tickets_2
LIMIT 3
"""
cur.execute(sql)
rows = cur.fetchall()
print("UNION ALL result:", rows)

# Now test with source_table literal
print("\nTesting UNION ALL with source_table literal...")
sql2 = """
SELECT user_id, psgn_name, 'tkt1' AS source_table FROM tickets_1
UNION ALL
SELECT user_id, psgn_name, 'tkt2' AS source_table FROM tickets_2
LIMIT 3
"""
cur.execute(sql2)
rows = cur.fetchall()
print("With literal:", rows)

conn.close()
print("\nAll SQL tests passed!")
