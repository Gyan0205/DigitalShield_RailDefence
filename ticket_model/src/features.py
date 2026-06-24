"""
features.py — Feature engineering for PDF-defined anomaly detection
10 feature categories. Zero Python loops except for name-similarity groupby.
"""

import time
import pandas as pd
import numpy as np

from src.logger import get_logger

log = get_logger()


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build all risk-relevant features using vectorised Pandas operations.

    Feature categories:
      1. Age features          (is_minor)
      2. Time features         (lead_time_hours, is_early_booking)
      3. Group features        (group_size, minors_per_pnr, adults_per_pnr)
      4. User pattern          (bookings_per_user, unique_passengers_per_user,
                                user_behavioral_variance)
      5. Behavioral signals    (high_variance_user)
      6. Evasive behaviour     (unique_ips_per_user, evasive_ip_flag)
      7. Group name similarity (group_name_similarity, no_family_pattern_flag)
      8. Family age gap        (family_age_gap_flag)
      9. Minor alone           (minor_alone_flag)
     10. Controller + signals  (controller_flag, consistent_passenger_flag)

    Returns:
        DataFrame with all feature columns added.
    """
    t0 = time.perf_counter()
    log.info("STEP 3: Engineering features (PDF-defined anomaly detection)")

    # ══════════════════════════════════════════════════════════════════════
    # 1. AGE FEATURES
    # ══════════════════════════════════════════════════════════════════════
    df["is_minor"] = (df["age"] < 18).astype(np.int8)
    log.debug("  [1/10] Age features")

    # ══════════════════════════════════════════════════════════════════════
    # 2. TIME FEATURES
    # ══════════════════════════════════════════════════════════════════════
    df["lead_time_hours"] = (
        (df["jrny_date_dt"] - df["booking_datetime"]).dt.total_seconds() / 3600
    )
    median_lead = df["lead_time_hours"].median()
    if pd.isna(median_lead):
        median_lead = 48.0
    df["lead_time_hours"] = df["lead_time_hours"].fillna(median_lead).clip(lower=0)

    # Early booking flag (negative / risk-reducing signal)
    df["is_early_booking"] = (df["lead_time_hours"] >= 48).astype(np.int8)
    log.debug("  [2/10] Time features (lead_time_hours, is_early_booking)")

    # ══════════════════════════════════════════════════════════════════════
    # 3. GROUP FEATURES (per PNR)
    # ══════════════════════════════════════════════════════════════════════
    df["group_size"] = df.groupby("pnrno")["pnrno"].transform("count")
    df["_is_minor_int"] = df["is_minor"].astype(int)
    df["minors_per_pnr"] = df.groupby("pnrno")["_is_minor_int"].transform("sum")
    df["adults_per_pnr"] = df["group_size"] - df["minors_per_pnr"]
    df.drop(columns=["_is_minor_int"], inplace=True)
    log.debug("  [3/10] Group features (group_size, minors_per_pnr, adults_per_pnr)")

    # ══════════════════════════════════════════════════════════════════════
    # 4. USER PATTERN FEATURES
    # ══════════════════════════════════════════════════════════════════════
    df["bookings_per_user"] = df.groupby("user_id")["user_id"].transform("count")
    df["unique_passengers_per_user"] = df.groupby("user_id")["psgn_name"].transform("nunique")

    # Ratio: high = booking many different people = suspicious
    df["user_behavioral_variance"] = (
        df["unique_passengers_per_user"] / df["bookings_per_user"].clip(lower=1)
    ).astype(np.float32)
    log.debug("  [4/10] User pattern features")

    # ══════════════════════════════════════════════════════════════════════
    # 5. BEHAVIORAL SIGNALS
    # ══════════════════════════════════════════════════════════════════════
    # Top 10% of users by unique passengers = repeated escort pattern
    high_var_threshold = df["unique_passengers_per_user"].quantile(0.90)
    df["high_variance_user"] = (
        df["unique_passengers_per_user"] >= high_var_threshold
    ).astype(np.int8)
    log.debug("  [5/10] Behavioral signals (high_variance_user)")

    # ══════════════════════════════════════════════════════════════════════
    # 6. EVASIVE BEHAVIOUR — Multiple IPs per user
    # ══════════════════════════════════════════════════════════════════════
    df["unique_ips_per_user"] = df.groupby("user_id")["ip_addrs"].transform("nunique")
    df["evasive_ip_flag"] = (df["unique_ips_per_user"] >= 3).astype(np.int8)
    log.debug("  [6/10] Evasive behaviour (unique_ips_per_user, evasive_ip_flag)")

    # ══════════════════════════════════════════════════════════════════════
    # 7. GROUP NAME SIMILARITY — Family relationship proxy
    # Fraction of PNR passengers sharing a common surname token.
    # Low similarity = unrelated strangers = suspicious group.
    # ══════════════════════════════════════════════════════════════════════
    def _name_similarity_per_pnr(names: pd.Series) -> pd.Series:
        """Fraction of names in a PNR group sharing a token with at least one other."""
        if len(names) <= 1:
            return pd.Series(1.0, index=names.index)  # Solo traveller — neutral
        name_list = names.str.lower().str.split().tolist()
        shared = []
        for i, tokens_i in enumerate(name_list):
            set_i = set(tokens_i) if tokens_i else set()
            has_match = any(
                set_i & (set(tokens_j) if tokens_j else set())
                for j, tokens_j in enumerate(name_list) if j != i
            )
            shared.append(1 if has_match else 0)
        total = len(shared)
        ratio = sum(shared) / total if total > 0 else 0.0
        return pd.Series(ratio, index=names.index)

    df["group_name_similarity"] = (
        df.groupby("pnrno")["psgn_name"]
        .transform(_name_similarity_per_pnr)
        .astype(np.float32)
    )
    # No family pattern: group of >= 2 with < 30% shared surname tokens
    df["no_family_pattern_flag"] = (
        (df["group_size"] >= 2) & (df["group_name_similarity"] < 0.30)
    ).astype(np.int8)
    log.debug("  [7/10] Group name similarity (group_name_similarity, no_family_pattern_flag)")

    # ══════════════════════════════════════════════════════════════════════
    # 8. FAMILY AGE GAP — Adult + child with realistic gap (>= 15 yrs)
    # Positive indicator: likely genuine family group.
    # ══════════════════════════════════════════════════════════════════════
    df["_is_adult_int"] = (df["age"] >= 18).astype(int)
    pnr_has_minor = df.groupby("pnrno")["is_minor"].transform("max")
    pnr_has_adult = df.groupby("pnrno")["_is_adult_int"].transform("max")
    pnr_min_adult_age = df.groupby("pnrno")["age"].transform(
        lambda x: x[x >= 18].min() if (x >= 18).any() else np.nan
    )
    pnr_max_child_age = df.groupby("pnrno")["age"].transform(
        lambda x: x[x < 18].max() if (x < 18).any() else np.nan
    )
    age_gap = (pnr_min_adult_age - pnr_max_child_age).fillna(0)
    df["family_age_gap_flag"] = (
        (pnr_has_minor == 1) & (pnr_has_adult == 1) & (age_gap >= 15)
    ).astype(np.int8)
    df.drop(columns=["_is_adult_int"], inplace=True)
    log.debug("  [8/10] Family age gap flag")

    # ══════════════════════════════════════════════════════════════════════
    # 9. MINOR ALONE — Vulnerability signal
    # Minor (age < 18) travelling in a PNR with ZERO adults.
    # ══════════════════════════════════════════════════════════════════════
    df["minor_alone_flag"] = (
        (df["is_minor"] == 1) & (df["adults_per_pnr"] == 0)
    ).astype(np.int8)
    log.debug("  [9/10] Minor alone flag (vulnerability signal)")

    # ══════════════════════════════════════════════════════════════════════
    # 10. CONTROLLER FLAG + NEGATIVE SIGNALS
    # ══════════════════════════════════════════════════════════════════════
    # Controller: 1 user books >= 3 unrelated passengers (no family pattern)
    df["controller_flag"] = (
        (df["group_size"] >= 3)
        & (df["no_family_pattern_flag"] == 1)
        & (df["bookings_per_user"] >= 3)
    ).astype(np.int8)

    # Consistent passenger flag (risk reducer — loyal/repeat traveller)
    df["consistent_passenger_flag"] = (
        df["user_behavioral_variance"] <= 0.40
    ).astype(np.int8)
    log.debug("  [10/10] Controller flag + consistent_passenger_flag")

    # ══════════════════════════════════════════════════════════════════════
    # SUMMARY
    # ══════════════════════════════════════════════════════════════════════
    elapsed = time.perf_counter() - t0
    log.info(f"  Feature engineering complete in {elapsed:.2f}s")

    return df
