"""
scoring.py — Weighted multiplicative hybrid risk scoring (PDF-defined signals only)
Positive signals raise score; negative signals (family indicators) lower it.
"""

import time
import numpy as np
import pandas as pd

from src.config import BOOST_WEIGHTS, REDUCER_WEIGHTS
from src.logger import get_logger

log = get_logger()


def apply_hybrid_scoring(
    df: pd.DataFrame,
    base_scores: np.ndarray,
    lof_flags: np.ndarray | None = None,
) -> pd.DataFrame:
    """
    Apply weighted multiplicative boost/reduce to ML base scores.

    Formula: risk_score = base_ml_score * (1.0 + boost - reducer)
    Capped at 10.0, floored at 0.0.

    Args:
        df: DataFrame with all feature columns.
        base_scores: Calibrated 0-10 ML risk scores.
        lof_flags: Optional LOF anomaly flags (True = anomaly).

    Returns:
        DataFrame with 'risk_score' column added.
    """
    t0 = time.perf_counter()
    log.info("STEP 6: Applying weighted hybrid risk scoring")

    df["risk_score"] = base_scores

    # ── Positive boost signals ───────────────────────────────────────────
    boost = pd.Series(0.0, index=df.index, dtype=np.float64)

    # Minor alone — vulnerability signal (minor with no adult in PNR)
    if "minor_alone_flag" in df.columns:
        boost += BOOST_WEIGHTS["minor_alone"] * df["minor_alone_flag"].astype(float)

    # Repeated escort — same user books many different passengers
    if "high_variance_user" in df.columns:
        boost += BOOST_WEIGHTS["repeated_escort"] * df["high_variance_user"].astype(float)

    # Controller-dominated booking — 1 user books >= 3 unrelated pax
    if "controller_flag" in df.columns:
        boost += BOOST_WEIGHTS["controller_dominated"] * df["controller_flag"].astype(float)

    # Evasive behaviour — same user uses >= 3 distinct IPs
    if "evasive_ip_flag" in df.columns:
        boost += BOOST_WEIGHTS["evasive_behavior"] * df["evasive_ip_flag"].astype(float)

    # No family pattern — group of >= 2 with no shared name tokens
    if "no_family_pattern_flag" in df.columns:
        boost += BOOST_WEIGHTS["no_family_pattern"] * df["no_family_pattern_flag"].astype(float)

    # LOF cross-validation boost (model-level signal)
    if lof_flags is not None:
        boost += BOOST_WEIGHTS["lof_cross_validated"] * pd.Series(
            lof_flags.astype(float), index=df.index
        )

    # ── Negative signals (risk reducers — family/normal travel) ─────────
    reducer = pd.Series(0.0, index=df.index, dtype=np.float64)

    # Family age pattern — adult + child with realistic age gap
    if "family_age_gap_flag" in df.columns:
        reducer += REDUCER_WEIGHTS["family_age_pattern"] * df["family_age_gap_flag"].astype(float)

    # Consistent passengers — user repeatedly books same people
    if "consistent_passenger_flag" in df.columns:
        reducer += REDUCER_WEIGHTS["consistent_passengers"] * df["consistent_passenger_flag"].astype(float)

    # Early booking — planned travel >= 48h in advance
    if "is_early_booking" in df.columns:
        reducer += REDUCER_WEIGHTS["early_booking"] * df["is_early_booking"].astype(float)

    # ── Apply multiplicative boost with reducer ──────────────────────────
    # Clamp so multiplier never drops below 0.5 (prevents score zeroing)
    net_factor = (boost - reducer).clip(lower=-0.5)
    multiplier = 1.0 + net_factor
    df["risk_score"] = (df["risk_score"] * multiplier).clip(lower=0.0, upper=10.0)

    boosted_count = (boost > 0).sum()
    reduced_count = (reducer > 0).sum()
    elapsed = time.perf_counter() - t0

    log.info(f"  Boosted  {boosted_count:,} records (positive signals)")
    log.info(f"  Reduced  {reduced_count:,} records (family/normal indicators)")
    log.info(
        f"  Final scores: min={df['risk_score'].min():.4f}, "
        f"mean={df['risk_score'].mean():.4f}, max={df['risk_score'].max():.4f}"
    )
    log.info(f"  Scoring complete ({elapsed:.2f}s)")

    return df
