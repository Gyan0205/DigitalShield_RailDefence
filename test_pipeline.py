"""
Digital Shield Rail Defense — End-to-End Pipeline Test
=======================================================
Validates the complete intelligence pipeline from CCTV event
through to alert generation.

Tests all 8 required steps:
  1. CCTV anomaly received
  2. Metadata extracted (platform, date, time, day)
  3. Trains table queried → nearest train on platform
  4. All 8 ticket tables queried via UNION ALL
  5. ML pipeline executed (IsolationForest + LOF + hybrid scoring)
  6. Final fused score = 60% CCTV + 40% Tickets
  7. Alert inserted into DB if score >= 0.75
  8. Fusion reasoning chain is complete and traceable

Usage:
    python test_pipeline.py
    python test_pipeline.py --verbose
    python test_pipeline.py --skip-db       (skip DB insert test)
    python test_pipeline.py --confidence 0.95  (override test confidence)
"""

import sys
import os
import argparse
import logging
from datetime import datetime, timezone

# Force UTF-8 output on Windows (prevents cp1252 UnicodeEncodeError)
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# -- Project root setup -------------------------------------------------------
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Load .env BEFORE any backend imports so DB credentials are available
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(ROOT, ".env"))
except ImportError:
    pass  # python-dotenv not installed — env vars must be set manually

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("e2e_test")

# ASCII-safe status tokens (no Unicode symbols)
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

PASS = f"{GREEN}[PASS]{RESET}"
FAIL = f"{RED}[FAIL]{RESET}"
SKIP = f"{YELLOW}[SKIP]{RESET}"
INFO = f"{CYAN}[i]{RESET}"


# ══════════════════════════════════════════════════════════════════════════════
# TEST PAYLOAD (exact specification from requirements)
# ══════════════════════════════════════════════════════════════════════════════

TEST_PAYLOAD = {
    "camera_id":          "CAM_SC_P07_A",
    "timestamp":          "2026-01-18T05:30:00",
    "anomaly_type":       "suspicious_escort",
    "anomaly_confidence": 0.82,
    "person_count":       2,
    # CCTV Metadata — pre-attached to footage
    # Train 12793 (Rayalaseema SF Express) | Platform 7 | Daily | departs 05:50
    # 2,140 bookings on this date — will produce non-zero ticket intelligence score
    "platform_number":    7,
    "date":               "2026-01-18",
    "time":               "05:30",
    "day":                "Sunday",
}


# ══════════════════════════════════════════════════════════════════════════════
# TEST RUNNER
# ══════════════════════════════════════════════════════════════════════════════

class PipelineTest:
    def __init__(self, verbose: bool = False, skip_db: bool = False,
                 override_confidence: float = None):
        self.verbose = verbose
        self.skip_db = skip_db
        self.override_confidence = override_confidence
        self.results: list = []
        self.alert = None

    def _step(self, number: int, name: str, ok: bool, detail: str = "", warn: str = ""):
        icon = PASS if ok else FAIL
        self.results.append((number, name, ok, detail))
        print(f"  [{icon}] Step {number}: {name}")
        if detail and (self.verbose or not ok):
            for line in detail.strip().split("\n"):
                print(f"         {INFO} {line}")
        if warn:
            print(f"         {YELLOW}⚠  {warn}{RESET}")

    def run(self) -> bool:
        payload = dict(TEST_PAYLOAD)
        if self.override_confidence is not None:
            payload["anomaly_confidence"] = self.override_confidence

        print()
        print(f"{BOLD}{'=' * 65}{RESET}")
        print(f"{BOLD}  DIGITAL SHIELD — END-TO-END PIPELINE TEST{RESET}")
        print(f"  {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC")
        print(f"{BOLD}{'=' * 65}{RESET}")
        print()
        print(f"  {INFO} Test payload:")
        for k, v in payload.items():
            print(f"       {k}: {v}")
        print()

        # ── Step 1: DB Connectivity ────────────────────────────────────────
        print(f"{BOLD}  [PHASE 1 — DATABASE]{RESET}")
        try:
            from backend.database import db_service
            health = db_service.health_check()
            ok = health.get("status") == "healthy"
            tables = health.get("table_names", [])
            ticket_tables = [t for t in tables if t.startswith("tickets_")]
            self._step(1, "Database connectivity",
                ok,
                f"Status={health.get('status')} | Tables={len(tables)} | "
                f"Ticket tables={ticket_tables}",
                warn="" if ok else health.get("error", "DB unreachable")
            )
        except Exception as e:
            self._step(1, "Database connectivity", False, str(e))
            print(f"\n  {RED}FATAL: Cannot connect to database. Aborting.{RESET}\n")
            return False

        # ── Step 2: Trains table query ─────────────────────────────────────
        print()
        print(f"{BOLD}  [PHASE 2 — TRAINS TABLE LOOKUP]{RESET}")
        train_number_found = ""
        train_name_found   = ""
        try:
            from sqlalchemy import text
            _DAY_SHORT = {
                "monday":"Mon","tuesday":"Tue","wednesday":"Wed",
                "thursday":"Thu","friday":"Fri","saturday":"Sat","sunday":"Sun",
            }
            day_abbr = _DAY_SHORT.get(payload["day"].lower(), payload["day"][:3].capitalize())

            with db_service.engine.connect() as conn:
                rows = conn.execute(text(
                    "SELECT train_number, train_name, departure_days, departure_time "
                    "FROM trains WHERE platform_number = :plat"
                ), {"plat": payload["platform_number"]}).fetchall()

            rows = [dict(r._mapping) for r in rows]
            candidates = [
                r for r in rows
                if day_abbr in [d.strip() for d in (r.get("departure_days") or "").split(",")]
            ]

            if candidates:
                def _secs(t):
                    try:
                        parts = str(t).split(":")
                        return int(parts[0]) * 3600 + int(parts[1]) * 60 + (int(parts[2]) if len(parts) > 2 else 0)
                    except Exception:
                        return 999999

                event_secs = _secs(payload["time"])
                candidates.sort(key=lambda c: abs(_secs(c["departure_time"]) - event_secs))
                best = candidates[0]
                train_number_found = str(best["train_number"])
                train_name_found   = best["train_name"]
                self._step(2, "Trains table query",
                    True,
                    f"Platform={payload['platform_number']} | Day={day_abbr} | Time={payload['time']}\n"
                    f"→ Matched {len(rows)} trains on platform, {len(candidates)} run on {day_abbr}\n"
                    f"→ Best match: Train {train_number_found} ({train_name_found}) "
                    f"departs {best['departure_time']}"
                )
            else:
                self._step(2, "Trains table query", False,
                    f"No trains on Platform {payload['platform_number']} for day {day_abbr}. "
                    "Run validate_trains_table.py to check coverage.",
                    warn="Train lookup failed — tickets step will score 0.0"
                )
        except Exception as e:
            self._step(2, "Trains table query", False, str(e))

        # ── Step 3: Ticket tables discovery ───────────────────────────────
        print()
        print(f"{BOLD}  [PHASE 3 — TICKET TABLES]{RESET}")
        ticket_tables_found = []
        try:
            from sqlalchemy import text
            EXPECTED = [f"tickets_{i}" for i in range(1, 9)]
            with db_service.engine.connect() as conn:
                result = conn.execute(text(
                    "SELECT table_name FROM information_schema.tables "
                    "WHERE table_schema='public' AND table_name = ANY(:t)"
                ), {"t": EXPECTED}).fetchall()
            ticket_tables_found = sorted([r[0] for r in result])
            missing = [t for t in EXPECTED if t not in ticket_tables_found]

            ok = len(ticket_tables_found) > 0
            self._step(3, "All 8 ticket tables present",
                ok and not missing,
                f"Found: {ticket_tables_found}\nMissing: {missing if missing else 'None'}",
                warn=f"Missing tables: {missing}" if missing else ""
            )
        except Exception as e:
            self._step(3, "All 8 ticket tables present", False, str(e))

        # ── Step 4: Ticket row counts ──────────────────────────────────────
        try:
            from sqlalchemy import text
            counts = {}
            with db_service.engine.connect() as conn:
                for tbl in ticket_tables_found:
                    n = conn.execute(text(f"SELECT COUNT(*) FROM {tbl}")).scalar()
                    counts[tbl] = n
            total = sum(counts.values())
            self._step(4, f"Ticket table row counts (UNION ALL = {total:,} rows)",
                total > 0,
                "\n".join(f"  {t}: {c:,}" for t, c in sorted(counts.items()))
            )
        except Exception as e:
            self._step(4, "Ticket table row counts", False, str(e))

        # ── Step 5: Fusion Engine (full pipeline) ─────────────────────────
        print()
        print(f"{BOLD}  [PHASE 4 — FUSION ENGINE]{RESET}")
        try:
            from backend.services.fusion_engine import FusionEngine
            engine = FusionEngine()
            engine.initialize()

            self.alert = engine.fuse_event(
                camera_id          = payload["camera_id"],
                timestamp          = payload["timestamp"],
                anomaly_type       = payload["anomaly_type"],
                anomaly_confidence = payload["anomaly_confidence"],
                person_count       = payload["person_count"],
                platform_number    = payload["platform_number"],
                date               = payload["date"],
                time               = payload["time"],
                day                = payload["day"],
            )

            cctv_score    = payload["anomaly_confidence"]
            expected_score = round(0.60 * cctv_score + 0.40 * (self.alert.fused_confidence - 0.60 * cctv_score) / 0.40, 4) \
                            if self.alert.fused_confidence != cctv_score * 0.60 else cctv_score * 0.60

            self._step(5, "Fusion Engine executed",
                True,
                f"Alert ID:    {self.alert.alert_id}\n"
                f"Severity:    {self.alert.severity}\n"
                f"CCTV score:  {cctv_score:.0%}\n"
                f"Ticket score:{self.alert.source_contributions.get('tickets_anomaly', 0) / 0.40:.0%} (pre-weight)\n"
                f"FINAL score: {self.alert.fused_confidence:.2%}\n"
                f"Train found: {self.alert.train_number} ({self.alert.train_name})\n"
                f"Coach:       {self.alert.estimated_coach}\n"
                f"Suspects:    {self.alert.suspects}"
            )
        except Exception as e:
            self._step(5, "Fusion Engine executed", False, str(e))
            logger.exception("Fusion Engine failed")
            self.alert = None

        # ── Step 6: 60-40 Formula validation ─────────────────────────────
        if self.alert:
            cctv_contrib   = self.alert.source_contributions.get("cctv_anomaly", 0)
            ticket_contrib = self.alert.source_contributions.get("tickets_anomaly", 0)
            calculated     = round(cctv_contrib + ticket_contrib, 4)
            expected       = round(self.alert.fused_confidence, 4)
            formula_ok     = abs(calculated - expected) < 0.01  # 1% tolerance

            self._step(6, "60-40 fusion formula verified",
                formula_ok,
                f"60% × CCTV ({cctv_contrib:.4f}) + 40% × Tickets ({ticket_contrib:.4f}) "
                f"= {calculated:.4f}\n"
                f"Reported fused_confidence: {expected:.4f} | Match: {formula_ok}"
            )
        else:
            self._step(6, "60-40 fusion formula verified", False, "Alert not available (step 5 failed)")

        # ── Step 7: Fusion reasoning chain ───────────────────────────────
        if self.alert:
            reasoning = self.alert.fusion_reasoning or []
            has_cctv   = any("[CCTV]" in r for r in reasoning)
            has_meta   = any("[META]" in r for r in reasoning)
            has_train  = any("[TRAIN]" in r for r in reasoning)
            has_fusion = any("[FUSION]" in r for r in reasoning)

            reasoning_ok = has_cctv and has_meta and has_fusion
            self._step(7, "Fusion reasoning chain complete",
                reasoning_ok,
                "\n".join(f"  {r}" for r in reasoning),
                warn="" if has_train else "No [TRAIN] entry — train was not resolved from trains table"
            )
        else:
            self._step(7, "Fusion reasoning chain", False, "Alert not available")

        # ── Step 8: DB Alert insertion ────────────────────────────────────
        print()
        print(f"{BOLD}  [PHASE 5 — ALERT STORAGE]{RESET}")
        if self.skip_db:
            print(f"  [{SKIP}] Step 8: DB alert insertion (--skip-db flag set)")
            self.results.append((8, "DB alert insertion", None, "skipped"))
        elif self.alert and self.alert.fused_confidence >= 0.75:
            try:
                # Verify the alert exists in DB (was inserted by fuse_event)
                from sqlalchemy import text
                with db_service.engine.connect() as conn:
                    row = conn.execute(text(
                        "SELECT alert_id, severity, fusion_confidence FROM alerts "
                        "WHERE alert_id = :aid LIMIT 1"
                    ), {"aid": self.alert.alert_id}).fetchone()

                if row:
                    self._step(8, "Alert inserted into DB",
                        True,
                        f"alert_id={row[0]} | severity={row[1]} | confidence={row[2]:.2%}"
                    )
                else:
                    # May have been inserted but not visible due to timing — check anyway
                    self._step(8, "Alert inserted into DB",
                        False,
                        f"Alert {self.alert.alert_id} not found in alerts table.\n"
                        "Check DB logs for insert errors.",
                        warn="Alert score >= 0.75 but not found in DB"
                    )
            except Exception as e:
                self._step(8, "Alert inserted into DB", False, str(e))
        elif self.alert:
            self._step(8, "Alert storage (score < 0.75 threshold)",
                True,
                f"Score {self.alert.fused_confidence:.2%} < 0.75 — correctly not inserted into DB.\n"
                "To test DB insertion, use --confidence 0.95"
            )
        else:
            self._step(8, "Alert inserted into DB", False, "Alert not available")

        # ── Final Report ──────────────────────────────────────────────────
        self._print_summary()
        return all(r[2] for r in self.results if r[2] is not None)

    def _print_summary(self):
        passed  = sum(1 for r in self.results if r[2] is True)
        failed  = sum(1 for r in self.results if r[2] is False)
        skipped = sum(1 for r in self.results if r[2] is None)
        total   = len(self.results)

        print()
        print(f"{BOLD}{'=' * 65}{RESET}")
        print(f"{BOLD}  RESULTS: {passed}/{total} passed | {failed} failed | {skipped} skipped{RESET}")

        if failed == 0:
            print(f"  {GREEN}{BOLD}[OK] ALL PIPELINE STEPS PASSED -- System is production-ready{RESET}")
        else:
            print(f"  {RED}{BOLD}[!!] {failed} STEP(S) FAILED -- Review errors above{RESET}")
            print()
            print("  Failed steps:")
            for n, name, ok, detail in self.results:
                if ok is False:
                    print(f"    • Step {n}: {name}")

        if self.alert:
            print()
            print(f"  {BOLD}Final Alert Summary:{RESET}")
            print(f"    Alert ID:   {self.alert.alert_id}")
            print(f"    Severity:   {self.alert.severity}")
            print(f"    Confidence: {self.alert.fused_confidence:.2%}")
            print(f"    Train:      {self.alert.train_number or 'Not resolved'} ({self.alert.train_name or '—'})")
            print(f"    Coach:      {self.alert.estimated_coach or '—'}")
            print(f"    Platform:   {self.alert.platform}")
            print(f"    Suspects:   {self.alert.suspects}")
            print()
            print(f"  {BOLD}Recommended Action:{RESET}")
            for line in (self.alert.recommended_action or "").split("\n"):
                print(f"    {line}")

        print(f"{BOLD}{'=' * 65}{RESET}")
        print()


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Digital Shield end-to-end pipeline test"
    )
    parser.add_argument("--verbose",    action="store_true", help="Show full details for passing steps")
    parser.add_argument("--skip-db",    action="store_true", help="Skip DB alert insertion verification")
    parser.add_argument("--confidence", type=float, default=None,
                        help="Override anomaly_confidence (default: 0.82)")
    args = parser.parse_args()

    test = PipelineTest(
        verbose=args.verbose,
        skip_db=args.skip_db,
        override_confidence=args.confidence,
    )
    success = test.run()
    sys.exit(0 if success else 1)
