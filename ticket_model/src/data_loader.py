"""
data_loader.py — Efficient data loading from ALL 8 ticket tables
================================================================
Queries tickets_1 through tickets_8 via UNION ALL using a direct
psycopg2 connection (bypasses SQLAlchemy to avoid URL parsing issues).

Only tables that actually exist in the DB are included (graceful discovery).
Each row is tagged with its source_table for full traceability.
"""

import time
import logging
import pandas as pd
import psycopg2
from urllib.parse import urlparse, unquote

from src.config import DB_URL, REQUIRED_COLUMNS
from src.logger import get_logger

log = get_logger()
logger = logging.getLogger("data_loader")

# All 8 ticket tables — queried via UNION ALL
ALL_TICKET_TABLES = [
    "tickets_1", "tickets_2", "tickets_3", "tickets_4",
    "tickets_5", "tickets_6", "tickets_7", "tickets_8",
]


def _make_psycopg2_conn():
    """
    Build a direct psycopg2 connection from DB_URL.
    Handles both plain and URL-encoded passwords robustly.
    """
    p = urlparse(DB_URL)
    password = unquote(p.password or "")
    return psycopg2.connect(
        host=p.hostname,
        port=p.port or 5432,
        dbname=p.path.lstrip("/"),
        user=p.username,
        password=password,
        sslmode="require",
        options="-c search_path=public",
        connect_timeout=15,
    )


def _discover_existing_tables(conn) -> list:
    """
    Query information_schema to find which ticket tables actually exist.
    Returns only tables present in the database — skips missing ones gracefully.
    """
    try:
        cur = conn.cursor()
        placeholders = ",".join([f"'{t}'" for t in ALL_TICKET_TABLES])
        cur.execute(
            f"SELECT table_name FROM information_schema.tables "
            f"WHERE table_schema = 'public' AND table_name IN ({placeholders})"
        )
        found = {row[0] for row in cur.fetchall()}
        cur.close()

        existing = [t for t in ALL_TICKET_TABLES if t in found]
        missing  = [t for t in ALL_TICKET_TABLES if t not in found]

        if missing:
            log.warning(f"  Ticket tables NOT found in DB (skipped): {missing}")
        log.info(f"  Ticket tables discovered: {existing}")
        return existing

    except Exception as e:
        log.warning(f"  Table discovery failed ({e}). Falling back to all 8 tables.")
        return list(ALL_TICKET_TABLES)


def load_data() -> pd.DataFrame:
    """
    Load and merge data from ALL available ticket tables (tickets_1..tickets_8)
    using UNION ALL via direct psycopg2 connection.

    Only columns defined in REQUIRED_COLUMNS are selected.
    Each row gets a 'source_table' column for traceability.

    Returns:
        pd.DataFrame with all records across all ticket tables, memory-optimised.
    """
    t0 = time.perf_counter()
    log.info("STEP 1: Loading data from ALL ticket tables (tickets_1 to tickets_8)")

    conn = _make_psycopg2_conn()

    try:
        # ── Discover which tables actually exist ──────────────────────────
        existing_tables = _discover_existing_tables(conn)

        if not existing_tables:
            raise RuntimeError(
                "No ticket tables found in database. "
                "Expected at least one of: " + ", ".join(ALL_TICKET_TABLES)
            )

        # ── Load each table individually then concat ──────────────────────
        # tickets_7 and tickets_8 use UPPERCASE column names (USER_ID, TRAIN_NUMBER, etc.)
        # tickets_1..6 use lowercase. Fix: SELECT *, then normalize columns to lowercase.
        # This makes ALL 8 tables compatible with the same feature pipeline.
        frames = []
        _required_set = set(REQUIRED_COLUMNS)

        for tbl in existing_tables:
            log.info(f"    Querying {tbl}...")
            try:
                # SELECT * — works regardless of column name casing
                df_tbl = pd.read_sql(f"SELECT * FROM {tbl}", conn)

                # ── NORMALIZE: lowercase + replace spaces with underscores ──
                # tickets_1..6: already lowercase with underscores (user_id, train_number)
                # tickets_7..8: spaced UPPERCASE ("USER ID", "TRAIN NUMBER")
                # After: lower() + replace(" ", "_") → all tables match the same schema
                df_tbl.columns = [c.lower().replace(" ", "_") for c in df_tbl.columns]

                # ── Filter to only required columns (ignore extra cols) ────
                missing_cols = _required_set - set(df_tbl.columns)
                if missing_cols:
                    log.warning(
                        f"    {tbl} missing required columns after normalisation: "
                        f"{missing_cols} — skipping table"
                    )
                    continue

                df_tbl = df_tbl[REQUIRED_COLUMNS].copy()
                df_tbl["source_table"] = tbl
                frames.append(df_tbl)
                log.info(f"    {tbl}: {len(df_tbl):,} rows")

            except Exception as tbl_err:
                log.warning(f"    {tbl} skipped due to error: {tbl_err}")

        if not frames:
            raise RuntimeError("All ticket table queries failed.")

        df = pd.concat(frames, ignore_index=True)
        log.info(
            f"  UNION ALL complete: {len(df):,} total rows "
            f"from {len(frames)} tables"
        )


    finally:
        conn.close()

    # ── Dtype optimisation ────────────────────────────────────────────────
    if "age" in df.columns:
        df["age"] = pd.to_numeric(df["age"], errors="coerce").astype("Int16")
    if "train_number" in df.columns:
        df["train_number"] = pd.to_numeric(df["train_number"], errors="coerce").astype("Int32")
    if "pnrno" in df.columns:
        df["pnrno"] = pd.to_numeric(df["pnrno"], errors="coerce").astype("Int64")

    # ── Summary ───────────────────────────────────────────────────────────
    elapsed = time.perf_counter() - t0
    mem_mb  = df.memory_usage(deep=True).sum() / 1e6

    log.info(
        f"  Loaded {len(df):,} rows x {len(df.columns)} cols "
        f"from {len(existing_tables)} tables in {elapsed:.2f}s"
    )
    log.info(f"  Memory usage: {mem_mb:.1f} MB")

    if "source_table" in df.columns:
        table_counts = df["source_table"].value_counts().to_dict()
        log.info(f"  Rows per table: {table_counts}")

    return df
