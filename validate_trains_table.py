"""
Digital Shield Rail Defense — Trains Table Validation Script
=============================================================
Validates the `trains` PostgreSQL table has complete and correct data
required by the CCTV → Train → Ticket pipeline.

The CCTV metadata pipeline provides:
    platform_number, day (e.g. "Tue"), time (HH:MM)

These are used to query the trains table. If the table has incomplete
data, the fusion engine will fail to resolve a train number and the
ticket correlation step will be skipped.

Usage:
    python validate_trains_table.py
    python validate_trains_table.py --min-trains 100
"""

import sys
import os
import argparse
import logging
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("trains_validator")

# Add project root to path so backend imports work
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# ============================================================================
# VALIDATION CONSTANTS
# ============================================================================

REQUIRED_COLUMNS = [
    "train_number",
    "train_name",
    "platform_number",
    "departure_time",
    "departure_days",
    "station_code",
]

EXPECTED_DAY_ABBRS = {"Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"}
TIME_FORMAT_EXAMPLES = []  # Populated during validation


# ============================================================================
# HELPERS
# ============================================================================

def _parse_time(t_str: str) -> bool:
    """Return True if t_str is parseable as HH:MM or HH:MM:SS."""
    try:
        parts = str(t_str).split(":")
        if len(parts) < 2:
            return False
        int(parts[0])
        int(parts[1])
        if len(parts) == 3:
            int(parts[2])
        return True
    except Exception:
        return False


# ============================================================================
# VALIDATION RUNNER
# ============================================================================

def validate_trains_table(min_trains: int = 10) -> bool:
    """
    Run all validation checks against the trains table.

    Returns:
        True if all critical checks pass (warnings are allowed).
        False if any critical check fails.
    """
    print()
    print("=" * 60)
    print("  DIGITAL SHIELD — TRAINS TABLE VALIDATION REPORT")
    print(f"  Run at: {datetime.utcnow().isoformat()}Z")
    print("=" * 60)

    warnings = []
    errors = []
    passed = []

    # ── Connect to DB ─────────────────────────────────────────────────────
    try:
        from backend.database import db_service
        from sqlalchemy import text

        engine = db_service.engine
        passed.append("Database connection established")
    except Exception as e:
        errors.append(f"Cannot connect to database: {e}")
        _print_report(passed, warnings, errors)
        return False

    # ── Check table exists ────────────────────────────────────────────────
    try:
        with engine.connect() as conn:
            result = conn.execute(text(
                "SELECT COUNT(*) FROM information_schema.tables "
                "WHERE table_name = 'trains'"
            )).scalar()
        if result and result > 0:
            passed.append("Table 'trains' exists")
        else:
            errors.append("Table 'trains' does NOT exist in the database")
            _print_report(passed, warnings, errors)
            return False
    except Exception as e:
        errors.append(f"Error checking table existence: {e}")
        _print_report(passed, warnings, errors)
        return False

    # ── Check required columns ────────────────────────────────────────────
    try:
        with engine.connect() as conn:
            result = conn.execute(text(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_name = 'trains'"
            )).fetchall()
        existing_cols = {row[0].lower() for row in result}
        missing_cols = [c for c in REQUIRED_COLUMNS if c.lower() not in existing_cols]

        if not missing_cols:
            passed.append(f"All required columns present: {', '.join(REQUIRED_COLUMNS)}")
        else:
            errors.append(f"Missing required columns: {missing_cols}")
    except Exception as e:
        errors.append(f"Error checking columns: {e}")

    if errors:
        _print_report(passed, warnings, errors)
        return False

    # ── Fetch all rows ────────────────────────────────────────────────────
    try:
        with engine.connect() as conn:
            rows = conn.execute(text(
                "SELECT train_number, train_name, platform_number, "
                "departure_time, departure_days, station_code FROM trains"
            )).fetchall()
        rows = [dict(r._mapping) for r in rows]
    except Exception as e:
        errors.append(f"Error fetching trains data: {e}")
        _print_report(passed, warnings, errors)
        return False

    total = len(rows)
    if total == 0:
        errors.append("trains table is EMPTY — no data found")
        _print_report(passed, warnings, errors)
        return False

    if total >= min_trains:
        passed.append(f"{total:,} train records found (minimum required: {min_trains})")
    else:
        warnings.append(
            f"Only {total} train records found — expected at least {min_trains}. "
            "Pipeline may have limited train resolution."
        )

    # ── NULL checks ───────────────────────────────────────────────────────
    null_train_number = [r for r in rows if not r.get("train_number")]
    null_platform     = [r for r in rows if r.get("platform_number") is None]
    null_dep_time     = [r for r in rows if not r.get("departure_time")]
    null_dep_days     = [r for r in rows if not r.get("departure_days")]

    if not null_train_number:
        passed.append("No NULL train_number values")
    else:
        errors.append(f"{len(null_train_number)} rows have NULL train_number")

    if not null_platform:
        passed.append("No NULL platform_number values")
    else:
        warnings.append(f"{len(null_platform)} rows have NULL platform_number — these trains cannot be resolved from CCTV metadata")

    if not null_dep_time:
        passed.append("No NULL departure_time values")
    else:
        warnings.append(
            f"{len(null_dep_time)} rows have NULL departure_time — "
            f"affected train_numbers: {[r['train_number'] for r in null_dep_time[:5]]}"
        )

    if not null_dep_days:
        passed.append("No NULL departure_days values")
    else:
        warnings.append(
            f"{len(null_dep_days)} rows have NULL departure_days — "
            f"affected train_numbers: {[r['train_number'] for r in null_dep_days[:5]]}"
        )

    # ── Weekday coverage ──────────────────────────────────────────────────
    all_days_found: set = set()
    for row in rows:
        dep_days = row.get("departure_days", "") or ""
        for d in dep_days.split(","):
            d = d.strip()
            if d:
                all_days_found.add(d)

    covered = all_days_found & EXPECTED_DAY_ABBRS
    missing_days = EXPECTED_DAY_ABBRS - all_days_found

    if not missing_days:
        passed.append(f"All 7 weekdays covered in departure_days: {', '.join(sorted(covered))}")
    else:
        warnings.append(
            f"Missing weekday coverage in departure_days: {missing_days}. "
            "Trains on these days cannot be resolved from CCTV metadata."
        )

    # ── Platform coverage ─────────────────────────────────────────────────
    platform_counts: dict = {}
    for row in rows:
        p = row.get("platform_number")
        if p is not None:
            platform_counts[p] = platform_counts.get(p, 0) + 1

    if platform_counts:
        passed.append(
            f"Platforms covered: {sorted(platform_counts.keys())} "
            f"(train counts: {dict(sorted(platform_counts.items()))})"
        )
        low_coverage = [p for p, c in platform_counts.items() if c == 1]
        if low_coverage:
            warnings.append(
                f"Platforms with only 1 train: {low_coverage} — "
                "low coverage may cause missed train resolutions."
            )
    else:
        warnings.append("No platform_number data could be read")

    # ── Time format check ─────────────────────────────────────────────────
    bad_times = [
        r["train_number"]
        for r in rows
        if r.get("departure_time") and not _parse_time(r["departure_time"])
    ]
    if not bad_times:
        passed.append("All departure_time values are parseable (HH:MM or HH:MM:SS)")
    else:
        warnings.append(
            f"{len(bad_times)} rows have unparseable departure_time — "
            f"affected: {bad_times[:5]}"
        )

    # ── Sample data preview ───────────────────────────────────────────────
    print()
    print("  SAMPLE DATA (first 3 rows):")
    for row in rows[:3]:
        print(
            f"    Train {row.get('train_number')} | {row.get('train_name')} | "
            f"Platform {row.get('platform_number')} | "
            f"{row.get('departure_time')} | {row.get('departure_days')}"
        )

    _print_report(passed, warnings, errors)
    return len(errors) == 0


def _print_report(passed: list, warnings: list, errors: list):
    print()
    print("  CHECKS:")
    for msg in passed:
        print(f"  ✓ {msg}")
    for msg in warnings:
        print(f"  ⚠  WARNING: {msg}")
    for msg in errors:
        print(f"  ✗  ERROR:   {msg}")
    print()
    if errors:
        print("  RESULT: ❌ FAILED — Fix errors before running the pipeline")
    elif warnings:
        print("  RESULT: ⚠  PASSED WITH WARNINGS — Review warnings above")
    else:
        print("  RESULT: ✅ FULLY PASSED — trains table is pipeline-ready")
    print("=" * 60)
    print()


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Validate the trains table for CCTV → Train → Ticket pipeline readiness"
    )
    parser.add_argument(
        "--min-trains", type=int, default=10,
        help="Minimum number of train records expected (default: 10)"
    )
    args = parser.parse_args()

    success = validate_trains_table(min_trains=args.min_trains)
    sys.exit(0 if success else 1)
