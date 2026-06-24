"""
explainer.py — Dynamic, data-driven explainability engine (PDF-defined signals)
Generates 1-3 human-readable risk reasons per record.
Fully vectorised using np.where + vectorised string join.
"""

import time
import numpy as np
import pandas as pd

from src.logger import get_logger

log = get_logger()

# Reason templates ordered by severity (most severe first)
REASON_REGISTRY = [
    {
        "id": "minor_alone",
        "priority": 1,
        "template": "Vulnerability: minor (age {age}) travelling with no adult in group",
    },
    {
        "id": "repeated_escort",
        "priority": 2,
        "template": "Repeated escort pattern ({unique_pax} different passengers booked by same user)",
    },
    {
        "id": "controller",
        "priority": 2,
        "template": "Controller-dominated booking (user books {group_size} unrelated passengers)",
    },
    {
        "id": "evasive",
        "priority": 3,
        "template": "Evasive behavior ({ip_count} different IPs used by same user)",
    },
    {
        "id": "no_family_pattern",
        "priority": 4,
        "template": "No family relationship pattern in group (unrelated passengers)",
    },
    {
        "id": "family_indicator",
        "priority": 5,
        "template": "[Family indicator] Adult-child age gap detected (likely family group)",
    },
    {
        "id": "early_booking",
        "priority": 6,
        "template": "[Normal] Early booking (planned travel, low urgency)",
    },
]


def generate_reasons(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate 1-3 human-readable risk reasons for each record.

    Uses vectorised np.where to build reason arrays, then joins
    the top 3 most relevant reasons per record.

    Args:
        df: DataFrame with all feature columns and risk_score.

    Returns:
        DataFrame with 'reason' column added.
    """
    t0 = time.perf_counter()
    log.info("STEP 7: Generating explainability reasons")

    reasons = []

    # 1. Minor alone — vulnerability signal
    if "minor_alone_flag" in df.columns:
        r = np.where(
            df["minor_alone_flag"] == 1,
            "Vulnerability: minor (age " + df["age"].astype(str) + ") with no adult in group",
            "",
        )
        reasons.append(r)

    # 2. Controller-dominated booking
    if "controller_flag" in df.columns:
        r = np.where(
            df["controller_flag"] == 1,
            "Controller-dominated booking (1 user books " + df["group_size"].astype(str) + " unrelated passengers)",
            "",
        )
        reasons.append(r)

    # 3. Evasive behaviour (multiple IPs)
    if "evasive_ip_flag" in df.columns and "unique_ips_per_user" in df.columns:
        r = np.where(
            df["evasive_ip_flag"] == 1,
            "Evasive behavior (" + df["unique_ips_per_user"].astype(str) + " different IPs used by same user)",
            "",
        )
        reasons.append(r)

    # 4. Repeated escort pattern (high variance user)
    if "high_variance_user" in df.columns and "unique_passengers_per_user" in df.columns:
        r = np.where(
            df["high_variance_user"] == 1,
            "Repeated escort: user booked " + df["unique_passengers_per_user"].astype(str) + " different passengers",
            "",
        )
        reasons.append(r)

    # 5. No family relationship pattern in group
    if "no_family_pattern_flag" in df.columns:
        r = np.where(
            df["no_family_pattern_flag"] == 1,
            "No family relationship pattern detected in group (unrelated passengers)",
            "",
        )
        reasons.append(r)

    # 6. Family age gap (risk-reducing context note)
    if "family_age_gap_flag" in df.columns:
        r = np.where(
            df["family_age_gap_flag"] == 1,
            "[Family indicator] Adult-child age gap detected (likely family group)",
            "",
        )
        reasons.append(r)

    # 7. Early booking (risk-reducing context note)
    if "is_early_booking" in df.columns:
        r = np.where(
            df["is_early_booking"] == 1,
            "[Normal] Early booking (planned travel, low urgency)",
            "",
        )
        reasons.append(r)

    # ── Combine reasons (max 3 per record) ───────────────────────────────
    if reasons:
        all_reasons = np.column_stack(reasons)
        df["reason"] = _join_top_reasons(all_reasons, max_reasons=3)
    else:
        df["reason"] = "Anomalous statistical pattern"

    elapsed = time.perf_counter() - t0
    log.info(f"  Reasons generated ({elapsed:.2f}s)")

    return df


def _join_top_reasons(reason_matrix: np.ndarray, max_reasons: int = 3) -> list[str]:
    """
    Join non-empty reasons per row, keeping at most max_reasons.
    Optimised: iterates through numpy rows (faster than df.apply).
    """
    result = []
    for row in reason_matrix:
        filtered = [r for r in row if r]
        if filtered:
            result.append("; ".join(filtered[:max_reasons]))
        else:
            result.append("Anomalous statistical pattern")
    return result
