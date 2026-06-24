"""
config.py — Centralised configuration for Digital Shield 2
All constants, thresholds, model hyperparameters, and DB credentials.
"""

# ═══════════════════════════════════════════════════════════════════════════════
# DATABASE
# ═══════════════════════════════════════════════════════════════════════════════

DB_URL = (
    "postgresql://postgres.trssafmvdbeagsdnivcl:"
    "digitalshield%4023070802"
    "@aws-1-ap-south-1.pooler.supabase.com:6543/postgres"
)

TABLE_NAME = "tickets_3"

# Only load the columns we actually need (skip coach_no_seat_no, txn_no, etc.)
REQUIRED_COLUMNS = [
    "user_id", "psgn_name", "train_number", "cls",
    "txn_date", "txn_time", "ip_addrs", "jrny_date",
    "pnrno", "from_stn", "to_stn", "age", "sex",
    "quota", "txntype", "bank_name",
]

# ═══════════════════════════════════════════════════════════════════════════════
# MODEL HYPERPARAMETERS
# ═══════════════════════════════════════════════════════════════════════════════

ISOLATION_FOREST = {
    "n_estimators": 150,
    "contamination": 0.03,
    "max_features": 0.8,
    "random_state": 42,
    "n_jobs": -1,
}

LOF_ENABLED = True
LOF_PARAMS = {
    "n_neighbors": 20,
    "contamination": 0.03,
    "novelty": False,
    "n_jobs": -1,
}

# ═══════════════════════════════════════════════════════════════════════════════
# FEATURE COLUMNS (fed to the model)
# ═══════════════════════════════════════════════════════════════════════════════

FEATURE_COLS = [
    "age",
    "lead_time_hours",
    "group_size",
    "bookings_per_user",
    "unique_passengers_per_user",
    "user_behavioral_variance",
    # PDF-defined anomaly features
    "unique_ips_per_user",       # Evasive behaviour: distinct IPs per user
    "group_name_similarity",    # Family pattern: shared surname token ratio
    "family_age_gap_flag",      # Family indicator: adult + child, age diff >= 15
    "minor_alone_flag",         # Vulnerability: minor with no adult in PNR
    "controller_flag",          # Controller: 1 user books >= 3 unrelated pax
]

# ═══════════════════════════════════════════════════════════════════════════════
# RISK SCORING
# ═══════════════════════════════════════════════════════════════════════════════

HIGH_RISK_THRESHOLD = 8.0

# Weighted boost signals (multiplicative approach)
# Formula: risk_score = base_ml_score * (1.0 + boost - reducer)
# Capped at 10.0, floored at 0.0.
BOOST_WEIGHTS = {
    # PDF positive signals (raise risk)
    "minor_alone":            0.45,  # Vulnerability: minor with no adult in PNR
    "repeated_escort":        0.35,  # Same user books many different passengers
    "controller_dominated":   0.25,  # 1 user books >= 3 unrelated passengers
    "evasive_behavior":       0.25,  # Same user uses >= 3 different IP addresses
    "no_family_pattern":      0.12,  # Group with no shared surname tokens
    "lof_cross_validated":    0.10,  # LOF model also flagged this record
}

# Negative signals (reduce risk — family/normal travel indicators)
REDUCER_WEIGHTS = {
    "family_age_pattern":     0.20,  # Adult + child with realistic age gap (>= 15yr)
    "consistent_passengers":  0.20,  # User consistently books same passengers
    "early_booking":          0.10,  # Booking made >= 48h before journey
}

# Thresholds for PDF signals
EVASIVE_IP_COUNT_THRESHOLD = 3    # Distinct IPs per user >= this = evasive
CONTROLLER_GROUP_MIN = 3          # Min group size for controller flag
NAME_SIMILARITY_THRESHOLD = 0.30  # Below this = no family pattern
FAMILY_AGE_GAP_MIN = 15          # Minimum age gap (years) for family pattern
CONSISTENT_PAX_RATIO = 0.40      # unique_pax / bookings <= this = consistent

# ═══════════════════════════════════════════════════════════════════════════════
# OUTPUT
# ═══════════════════════════════════════════════════════════════════════════════

OUTPUT_COLS = [
    "pnrno", "user_id", "psgn_name", "train_number", "jrny_date", "age", "sex",
    "from_stn", "to_stn", "risk_score", "reason",
]

OUTPUT_DIR = "outputs"

