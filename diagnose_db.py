"""Quick DB schema diagnostic — lists all tables and ticket_1 columns."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from backend.database import db_service
from sqlalchemy import text

with db_service.engine.connect() as conn:
    tables = conn.execute(text(
        "SELECT table_name FROM information_schema.tables "
        "WHERE table_schema='public' ORDER BY table_name"
    )).fetchall()
    print("=== TABLES IN SUPABASE DB ===")
    for t in tables:
        print(" -", t[0])

    # Check columns of tickets_1 if it exists
    ticket_tables = [t[0] for t in tables if t[0].startswith("tickets_")]
    if ticket_tables:
        tbl = ticket_tables[0]
        cols = conn.execute(text(
            "SELECT column_name, data_type FROM information_schema.columns "
            "WHERE table_schema='public' AND table_name=:t ORDER BY ordinal_position"
        ), {"t": tbl}).fetchall()
        print(f"\n=== COLUMNS IN {tbl} ===")
        for c in cols:
            print(f"  {c[0]} ({c[1]})")
    else:
        print("\nNo tickets_* tables found.")

    # Check trains table
    trains_exists = any(t[0] == "trains" for t in tables)
    print(f"\n=== trains table exists: {trains_exists} ===")
    if trains_exists:
        cols = conn.execute(text(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_schema='public' AND table_name='trains' ORDER BY ordinal_position"
        )).fetchall()
        print("  trains columns:", [c[0] for c in cols])
        count = conn.execute(text("SELECT COUNT(*) FROM trains")).scalar()
        print(f"  trains row count: {count}")
