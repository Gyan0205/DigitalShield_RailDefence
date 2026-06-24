"""Probe tickets_3 table schema and sample data."""
import pandas as pd
from sqlalchemy import create_engine, text

DB_URL = "postgresql://postgres.trssafmvdbeagsdnivcl:digitalshield%4023070802@aws-1-ap-south-1.pooler.supabase.com:6543/postgres"
engine = create_engine(DB_URL)

with engine.connect() as conn:
    df = pd.read_sql(text("SELECT * FROM tickets_3 LIMIT 5"), conn)
    print("=== COLUMNS ===")
    for i, col in enumerate(df.columns):
        print(f"  {i}: '{col}' -> dtype: {df[col].dtype}")
    print(f"\n=== SHAPE: {df.shape} ===")
    print("\n=== SAMPLE DATA ===")
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 300)
    print(df.head())
    
    count = pd.read_sql(text("SELECT COUNT(*) as cnt FROM tickets_3"), conn)
    print(f"\n=== TOTAL ROWS: {count['cnt'].iloc[0]} ===")

engine.dispose()
