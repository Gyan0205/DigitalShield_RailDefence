"""
model.py — Isolation Forest + optional LOF cross-validation
RobustScaler for outlier resistance, MinMax calibration for 0-10 scores.
"""

import time
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import RobustScaler, MinMaxScaler

from src.config import ISOLATION_FOREST, LOF_ENABLED, LOF_PARAMS, FEATURE_COLS
from src.logger import get_logger

log = get_logger()


def prepare_features(df: pd.DataFrame) -> np.ndarray:
    """
    Select model features, clip outliers, and scale with RobustScaler.

    Returns:
        Scaled feature matrix (numpy ndarray).
    """
    t0 = time.perf_counter()
    log.info("STEP 4: Preparing features for model")

    # Select only available feature columns (graceful degradation)
    available = [c for c in FEATURE_COLS if c in df.columns]
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        log.warning(f"  Missing features (skipped): {missing}")

    X = df[available].copy()

    # Clip extreme outliers (1st-99th percentile)
    for col in X.columns:
        lo = X[col].quantile(0.01)
        hi = X[col].quantile(0.99)
        if lo < hi:
            X[col] = X[col].clip(lo, hi)

    # Fill remaining NaN with 0
    X = X.fillna(0)

    # RobustScaler (IQR-based, better with outliers than StandardScaler)
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)

    elapsed = time.perf_counter() - t0
    log.info(f"  Feature matrix: {X_scaled.shape} ({elapsed:.2f}s)")

    return X_scaled


def compute_risk_scores(X_scaled: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
    """
    Fit Isolation Forest (and optionally LOF), return calibrated 0-1 scores.

    Returns:
        Tuple of (risk_scores, lof_flags):
          - risk_scores: ndarray of float [0, 10]
          - lof_flags: ndarray of bool (True = anomaly) or None if LOF disabled
    """
    t0 = time.perf_counter()
    log.info("STEP 5: Training Isolation Forest")

    # ── Isolation Forest ─────────────────────────────────────────────────
    iforest = IsolationForest(**ISOLATION_FOREST)
    iforest.fit(X_scaled)

    # decision_function: lower = more anomalous
    raw_scores = iforest.decision_function(X_scaled)

    # Invert + MinMax -> [0, 10]
    inverted = -raw_scores
    scaler = MinMaxScaler(feature_range=(0, 10))
    risk_scores = scaler.fit_transform(inverted.reshape(-1, 1)).flatten()

    log.info(
        f"  IF scores: min={risk_scores.min():.4f}, "
        f"mean={risk_scores.mean():.4f}, max={risk_scores.max():.4f}"
    )

    # ── Optional LOF cross-validation ────────────────────────────────────
    lof_flags = None
    if LOF_ENABLED:
        log.info("  Training Local Outlier Factor (cross-validation)")
        try:
            lof = LocalOutlierFactor(**LOF_PARAMS)
            lof_predictions = lof.fit_predict(X_scaled)
            lof_flags = lof_predictions == -1  # True = anomaly
            lof_count = lof_flags.sum()
            log.info(f"  LOF flagged {lof_count:,} records as anomalies")
        except Exception as e:
            log.warning(f"  LOF failed (non-critical): {e}")
            lof_flags = None

    elapsed = time.perf_counter() - t0
    log.info(f"  Model training complete ({elapsed:.2f}s)")

    return risk_scores, lof_flags
