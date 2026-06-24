"""
Digital Shield Rail Defense — ML Pipeline Orchestrator
=========================================================
Master script that runs the complete ML pipeline:

  1. Dataset preprocessing
  2. YOLOv8 person detector training
  3. Anomaly classifier training
  4. Evaluation (precision/recall/F1/confusion matrix)
  5. Inference benchmarking
  6. Model versioning & experiment tracking

Usage:
    python ml/training/run_pipeline.py
    python ml/training/run_pipeline.py --skip-yolo --epochs 30
"""

import sys
import json
import time
import logging
import numpy as np
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from ml.config import METADATA_DIR, WEIGHTS_DIR, LOG_FORMAT, LOG_DATE_FORMAT

logger = logging.getLogger("ml_pipeline")
logger.setLevel(logging.INFO)
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter(LOG_FORMAT, LOG_DATE_FORMAT))
    logger.addHandler(_h)


def run_full_pipeline(
    epochs_anomaly: int = 20,
    epochs_yolo: int = 20,
    skip_yolo: bool = False,
    skip_preprocess: bool = False,
    num_synthetic: int = 10000,
    benchmark: bool = True,
):
    """Run the complete ML training and evaluation pipeline."""
    pipeline_start = time.time()

    logger.info("=" * 70)
    logger.info("  DIGITAL SHIELD — COMPLETE ML PIPELINE")
    logger.info("=" * 70)
    logger.info(f"  Started: {datetime.now().isoformat()}")
    logger.info(f"  Anomaly epochs: {epochs_anomaly}")
    logger.info(f"  YOLO epochs: {epochs_yolo}")
    logger.info("=" * 70)

    results = {}

    # ── Initialize Experiment Tracker ───────────────────────
    from ml.training.experiment_tracker import ExperimentTracker, ModelRegistry, ModelVersion
    tracker = ExperimentTracker()
    registry = ModelRegistry()
    exp_id = tracker.start_experiment(
        "full_pipeline",
        {
            "epochs_anomaly": epochs_anomaly,
            "epochs_yolo": epochs_yolo,
            "num_synthetic": num_synthetic,
        },
        tags=["full_pipeline", "automated"],
    )

    # ── Stage 1: Dataset Preprocessing ──────────────────────
    if not skip_preprocess:
        logger.info("\n" + "─" * 50)
        logger.info("STAGE 1: Dataset Preprocessing")
        logger.info("─" * 50)

        from ml.training.preprocess_dataset import DatasetPreprocessor
        preprocessor = DatasetPreprocessor()
        prep_result = preprocessor.run_full_pipeline(num_synthetic=num_synthetic)
        results["preprocessing"] = prep_result
        tracker.log_metric(exp_id, "preprocessing_sequences",
                          prep_result.get("stats", {}).get("train", {}).get("total", 0))
    else:
        logger.info("\n[SKIP] Preprocessing")

    # ── Stage 2: YOLOv8 Training ────────────────────────────
    if not skip_yolo:
        logger.info("\n" + "─" * 50)
        logger.info("STAGE 2: YOLOv8 Person Detector Training")
        logger.info("─" * 50)

        from ml.training.train_yolo import train_yolo, prepare_yolo_dataset
        prepare_yolo_dataset()
        yolo_result = train_yolo(epochs=epochs_yolo, model_size="n")
        results["yolo_training"] = yolo_result
        tracker.log_metrics(exp_id, {
            "yolo_mAP50": yolo_result.get("best_mAP50", 0),
            "yolo_precision": yolo_result.get("precision", 0),
            "yolo_recall": yolo_result.get("recall", 0),
        })

        # Register model
        registry.register(ModelVersion(
            model_name="yolov8_person_detect",
            version=f"1.{epochs_yolo}.0",
            weights_path=yolo_result.get("best_weights", "simulated"),
            metrics={
                "mAP50": yolo_result.get("best_mAP50", 0),
                "precision": yolo_result.get("precision", 0),
            },
            hyperparams={"epochs": epochs_yolo, "model_size": "n"},
            tags=["yolo", "person_detection"],
        ))
    else:
        logger.info("\n[SKIP] YOLOv8 Training")

    # ── Stage 3: Anomaly Classifier Training ────────────────
    logger.info("\n" + "─" * 50)
    logger.info("STAGE 3: Anomaly Classifier Training")
    logger.info("─" * 50)

    from ml.training.train_anomaly_detector import train_model
    anomaly_result = train_model(
        epochs=epochs_anomaly, batch_size=32,
        use_synthetic=True, model_type="lstm",
    )
    results["anomaly_training"] = anomaly_result

    if anomaly_result.get("status") == "complete":
        tracker.log_metrics(exp_id, {
            "anomaly_accuracy": anomaly_result.get("best_val_acc", 0),
            "anomaly_loss": anomaly_result.get("best_val_loss", 0),
        })
        registry.register(ModelVersion(
            model_name="anomaly_lstm",
            version=f"1.{epochs_anomaly}.0",
            weights_path=anomaly_result.get("weights_path", ""),
            metrics={"accuracy": anomaly_result.get("best_val_acc", 0)},
            hyperparams={"epochs": epochs_anomaly, "hidden_size": 256},
            tags=["anomaly", "lstm"],
        ))

    # ── Stage 4: Evaluation ─────────────────────────────────
    logger.info("\n" + "─" * 50)
    logger.info("STAGE 4: Model Evaluation")
    logger.info("─" * 50)

    from ml.evaluation.evaluate_anomaly import evaluate_model
    eval_result = evaluate_model(use_synthetic=True)
    results["evaluation"] = eval_result

    if eval_result.get("overall"):
        tracker.log_metrics(exp_id, {
            "eval_accuracy": eval_result["overall"].get("accuracy", 0),
            "eval_macro_f1": eval_result["overall"].get("macro_f1", 0),
            "eval_weighted_f1": eval_result["overall"].get("weighted_f1", 0),
            "eval_anomaly_f1": eval_result["overall"].get("anomaly_detection_f1", 0),
            "eval_false_alarm": eval_result["overall"].get("false_alarm_rate", 0),
        })

    # ── Stage 5: Inference Benchmarking ─────────────────────
    if benchmark:
        logger.info("\n" + "─" * 50)
        logger.info("STAGE 5: Inference Benchmarking")
        logger.info("─" * 50)

        from ml.evaluation.benchmark import InferenceBenchmark
        bench = InferenceBenchmark()
        bench_result = bench.run_all()
        results["benchmark"] = bench_result
    else:
        logger.info("\n[SKIP] Benchmarking")

    # ── Finalize ────────────────────────────────────────────
    elapsed = time.time() - pipeline_start

    tracker.end_experiment(exp_id, notes=f"Full pipeline in {elapsed:.1f}s")

    # Save master report
    master_report = {
        "timestamp": datetime.now().isoformat(),
        "duration_seconds": round(elapsed, 2),
        "stages": results,
        "experiment_id": exp_id,
        "model_registry": registry.get_stats(),
        "experiment_stats": tracker.get_stats(),
    }

    report_path = METADATA_DIR / "pipeline_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(master_report, f, indent=2, default=str)

    logger.info("\n" + "=" * 70)
    logger.info("  PIPELINE COMPLETE")
    logger.info("=" * 70)
    logger.info(f"  Duration: {elapsed:.1f}s")
    logger.info(f"  Experiment: {exp_id}")
    logger.info(f"  Models registered: {registry.get_stats()['total_versions']}")
    logger.info(f"  Report: {report_path}")
    logger.info("=" * 70)

    return master_report


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Digital Shield ML Pipeline")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--yolo-epochs", type=int, default=20)
    parser.add_argument("--skip-yolo", action="store_true")
    parser.add_argument("--skip-preprocess", action="store_true")
    parser.add_argument("--no-benchmark", action="store_true")
    parser.add_argument("--samples", type=int, default=10000)
    args = parser.parse_args()

    run_full_pipeline(
        epochs_anomaly=args.epochs,
        epochs_yolo=args.yolo_epochs,
        skip_yolo=args.skip_yolo,
        skip_preprocess=args.skip_preprocess,
        num_synthetic=args.samples,
        benchmark=not args.no_benchmark,
    )
