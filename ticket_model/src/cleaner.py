"""
cleaner.py — Vectorised data cleaning and datetime conversion
No loops. All operations are Pandas-native.
"""

import time
import pandas as pd
import numpy as np

from src.logger import get_logger

log = get_logger()


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Vectorised cleaning pipeline:
      1. Strip whitespace from string columns
      2. Combine txn_date + txn_time -> booking_datetime
      3. Convert jrny_date -> datetime
      4. Sanitise age (clip 0-120, fill median)
      5. Drop rows missing critical fields (age, pnrno)

    Returns:
        Cleaned DataFrame (copy — does not mutate input).
    """
    t0 = time.perf_counter()
    log.info("STEP 2: Cleaning data")

    df = df.copy()
    initial_rows = len(df)

    # ── 1. Strip whitespace from string columns ──────────────────────────
    str_cols = df.select_dtypes(include="object").columns
    for col in str_cols:
        df[col] = df[col].astype(str).str.strip()
        df[col] = df[col].replace({"nan": np.nan, "None": np.nan, "": np.nan})
    log.debug("  String columns stripped")

    # ── 2. Combine txn_date + txn_time -> booking_datetime ───────────────
    df["booking_datetime"] = pd.to_datetime(
        df["txn_date"].astype(str) + " " + df["txn_time"].astype(str),
        errors="coerce",
    )
    log.debug("  booking_datetime created")

    # ── 3. Convert jrny_date -> datetime ─────────────────────────────────
    df["jrny_date_dt"] = pd.to_datetime(df["jrny_date"], errors="coerce")
    log.debug("  jrny_date_dt created")

    # ── 4. Sanitise age ──────────────────────────────────────────────────
    df["age"] = pd.to_numeric(df["age"], errors="coerce")
    median_age = df["age"].median()
    if pd.isna(median_age):
        median_age = 25  # sensible fallback
    df["age"] = df["age"].fillna(median_age).clip(lower=0, upper=120).astype(int)
    log.debug("  age sanitised")

    # ── 5. Drop rows with critical missing values ────────────────────────
    df = df.dropna(subset=["pnrno"]).copy()
    df = df[df["age"] > 0].copy()

    dropped = initial_rows - len(df)
    elapsed = time.perf_counter() - t0

    log.info(f"  Dropped {dropped:,} rows with critical missing values")
    log.info(f"  Remaining: {len(df):,} rows ({elapsed:.2f}s)")

    return df
