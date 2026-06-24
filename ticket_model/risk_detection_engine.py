"""
risk_detection_engine.py — V3 Pipeline Orchestrator
Digital Shield 2: ML Risk Detection Engine

Slim orchestrator that delegates all logic to src/ modules:
  src/data_loader.py  — Database loading
  src/cleaner.py      — Data cleaning
  src/features.py     — Feature engineering
  src/model.py        — Isolation Forest + LOF
  src/scoring.py      — Hybrid risk scoring  
  src/explainer.py    — Reason generation
"""

import sys
import os
import time
from pathlib import Path

# Add the directory containing this file to sys.path to enable src imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Force UTF-8 output on Windows consoles
if sys.platform == "win32":
    os.environ["PYTHONIOENCODING"] = "utf-8"
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import warnings
warnings.filterwarnings("ignore")

import pandas as pd

from src.config import OUTPUT_COLS, HIGH_RISK_THRESHOLD, OUTPUT_DIR
from src.logger import get_logger
from src.data_loader import load_data
from src.cleaner import clean_data
from src.features import engineer_features
from src.model import prepare_features, compute_risk_scores
from src.scoring import apply_hybrid_scoring
from src.explainer import generate_reasons

log = get_logger()


def extract_high_risk(df: pd.DataFrame, threshold: float = HIGH_RISK_THRESHOLD) -> pd.DataFrame:
    """Return only records above the risk threshold, sorted descending."""
    log.info(f"STEP 8: Extracting high-risk records (threshold > {threshold})")

    available_cols = [c for c in OUTPUT_COLS if c in df.columns]
    high_risk = (
        df.loc[df["risk_score"] > threshold, available_cols]
        .sort_values("risk_score", ascending=False)
        .reset_index(drop=True)
    )

    log.info(f"  {len(high_risk):,} high-risk records identified")
    return high_risk


def main():
    """Full ML risk detection pipeline."""
    log.info("=" * 64)
    log.info("  DIGITAL SHIELD 2 v3 — ML Risk Detection Engine")
    log.info("  Isolation Forest + LOF + Weighted Hybrid Boosting")
    log.info("=" * 64)

    pipeline_start = time.perf_counter()

    try:
        # Step 1 — Load data
        df = load_data()

        # Step 2 — Clean
        df = clean_data(df)

        # Step 3 — Feature engineering (14 features)
        df = engineer_features(df)

        # Step 4 — Prepare features for model
        X_scaled = prepare_features(df)

        # Step 5 — Train model & compute base scores
        risk_scores, lof_flags = compute_risk_scores(X_scaled)

        # Step 6 — Weighted hybrid scoring
        df = apply_hybrid_scoring(df, risk_scores, lof_flags)

        # Step 7 — Explainability reasons
        df = generate_reasons(df)

        # Step 8 — Extract high-risk records
        high_risk_df = extract_high_risk(df)

    except Exception as e:
        log.error(f"Pipeline failed: {e}", exc_info=True)
        sys.exit(1)

    # ── Summary statistics ────────────────────────────────────────────────
    total_time = time.perf_counter() - pipeline_start

    log.info("")
    log.info("=" * 64)
    log.info("  PIPELINE COMPLETE")
    log.info("=" * 64)
    log.info(f"  Total records processed : {len(df):,}")
    log.info(f"  High-risk records       : {len(high_risk_df):,}")
    log.info(f"  Total pipeline time     : {total_time:.2f}s")
    log.info(f"  Avg score               : {df['risk_score'].mean():.2f}")
    log.info(f"  Median score            : {df['risk_score'].median():.2f}")
    log.info(f"  90th percentile         : {df['risk_score'].quantile(0.90):.2f}")
    log.info(f"  95th percentile         : {df['risk_score'].quantile(0.95):.2f}")
    log.info(f"  99th percentile         : {df['risk_score'].quantile(0.99):.2f}")
    log.info("=" * 64)

    # ── Print top 20 ─────────────────────────────────────────────────────
    log.info("")
    log.info("TOP 20 HIGHEST-RISK RECORDS:")
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 300)
    pd.set_option("display.max_colwidth", 80)
    print(high_risk_df.head(20).to_string(index=False))

    # ── Save outputs ─────────────────────────────────────────────────────
    output_path = Path(OUTPUT_DIR)
    output_path.mkdir(parents=True, exist_ok=True)

    high_risk_path = output_path / "high_risk_records.csv"
    all_scores_path = output_path / "all_risk_scores.csv"
    json_path = output_path / "high_risk_records.json"

    high_risk_df.to_csv(high_risk_path, index=False)
    high_risk_df.to_json(json_path, orient="records", indent=2)

    available_out = [c for c in OUTPUT_COLS if c in df.columns]
    df[available_out].to_csv(all_scores_path, index=False)

    log.info(f"  Saved {high_risk_path} ({len(high_risk_df):,} records)")
    log.info(f"  Saved {json_path}")
    log.info(f"  Saved {all_scores_path} ({len(df):,} records)")

    return df, high_risk_df


if __name__ == "__main__":
    df, high_risk_df = main()
