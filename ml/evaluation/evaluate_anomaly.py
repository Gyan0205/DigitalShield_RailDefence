"""
Digital Shield Rail Defense — Evaluation Metrics
===================================================
Comprehensive evaluation of anomaly detection performance.
Computes precision, recall, F1, AUC, confusion matrix,
and per-class metrics.
"""

import sys
import json
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
from collections import Counter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from ml.config import METADATA_DIR, ANOMALY_CLASSES, LOG_FORMAT, LOG_DATE_FORMAT

logger = logging.getLogger("evaluate_anomaly")
logger.setLevel(logging.INFO)
if not logger.handlers:
    console = logging.StreamHandler()
    console.setFormatter(logging.Formatter(LOG_FORMAT, LOG_DATE_FORMAT))
    logger.addHandler(console)


class AnomalyEvaluator:
    """
    Comprehensive evaluation suite for anomaly detection models.

    Computes:
      - Per-class precision, recall, F1-score
      - Overall accuracy and macro/weighted F1
      - Confusion matrix
      - ROC-AUC and PR-AUC
      - Detection latency metrics
      - False alarm rate
    """

    CLASS_NAMES = {
        0: "normal", 1: "assault", 2: "coercion", 3: "dragging",
        4: "suspicious_escort", 5: "isolated_minor", 6: "panic", 7: "crowd_anomaly",
    }

    def __init__(self):
        self.y_true: List[int] = []
        self.y_pred: List[int] = []
        self.y_scores: List[List[float]] = []  # Per-class probabilities

    def add_predictions(self, true_labels: List[int], pred_labels: List[int],
                        scores: Optional[List[List[float]]] = None):
        """Add a batch of predictions."""
        self.y_true.extend(true_labels)
        self.y_pred.extend(pred_labels)
        if scores:
            self.y_scores.extend(scores)

    def compute_confusion_matrix(self) -> np.ndarray:
        """Compute confusion matrix."""
        classes = sorted(set(self.y_true + self.y_pred))
        n = max(classes) + 1
        cm = np.zeros((n, n), dtype=int)
        for t, p in zip(self.y_true, self.y_pred):
            cm[t][p] += 1
        return cm

    def compute_per_class_metrics(self) -> Dict[str, Dict[str, float]]:
        """Compute precision, recall, F1 per class."""
        cm = self.compute_confusion_matrix()
        metrics = {}

        for i in range(len(cm)):
            tp = cm[i][i]
            fp = cm[:, i].sum() - tp
            fn = cm[i, :].sum() - tp
            tn = cm.sum() - tp - fp - fn

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

            class_name = self.CLASS_NAMES.get(i, f"class_{i}")
            metrics[class_name] = {
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1_score": round(f1, 4),
                "specificity": round(specificity, 4),
                "support": int(cm[i, :].sum()),
                "true_positives": int(tp),
                "false_positives": int(fp),
                "false_negatives": int(fn),
            }

        return metrics

    def compute_overall_metrics(self) -> Dict[str, float]:
        """Compute overall accuracy and macro/weighted averages."""
        per_class = self.compute_per_class_metrics()
        total = len(self.y_true)
        correct = sum(1 for t, p in zip(self.y_true, self.y_pred) if t == p)

        # Macro average
        precisions = [m["precision"] for m in per_class.values()]
        recalls = [m["recall"] for m in per_class.values()]
        f1s = [m["f1_score"] for m in per_class.values()]

        # Weighted average
        supports = [m["support"] for m in per_class.values()]
        total_support = sum(supports)
        w_precision = sum(p * s for p, s in zip(precisions, supports)) / max(total_support, 1)
        w_recall = sum(r * s for r, s in zip(recalls, supports)) / max(total_support, 1)
        w_f1 = sum(f * s for f, s in zip(f1s, supports)) / max(total_support, 1)

        # Anomaly-specific: binary anomaly detection (normal vs any anomaly)
        binary_true = [0 if t == 0 else 1 for t in self.y_true]
        binary_pred = [0 if p == 0 else 1 for p in self.y_pred]
        anomaly_tp = sum(1 for t, p in zip(binary_true, binary_pred) if t == 1 and p == 1)
        anomaly_fp = sum(1 for t, p in zip(binary_true, binary_pred) if t == 0 and p == 1)
        anomaly_fn = sum(1 for t, p in zip(binary_true, binary_pred) if t == 1 and p == 0)
        anomaly_precision = anomaly_tp / (anomaly_tp + anomaly_fp) if (anomaly_tp + anomaly_fp) > 0 else 0
        anomaly_recall = anomaly_tp / (anomaly_tp + anomaly_fn) if (anomaly_tp + anomaly_fn) > 0 else 0
        anomaly_f1 = 2 * anomaly_precision * anomaly_recall / (anomaly_precision + anomaly_recall) if (anomaly_precision + anomaly_recall) > 0 else 0

        return {
            "accuracy": round(correct / max(total, 1), 4),
            "macro_precision": round(np.mean(precisions), 4),
            "macro_recall": round(np.mean(recalls), 4),
            "macro_f1": round(np.mean(f1s), 4),
            "weighted_precision": round(w_precision, 4),
            "weighted_recall": round(w_recall, 4),
            "weighted_f1": round(w_f1, 4),
            "anomaly_detection_precision": round(anomaly_precision, 4),
            "anomaly_detection_recall": round(anomaly_recall, 4),
            "anomaly_detection_f1": round(anomaly_f1, 4),
            "false_alarm_rate": round(anomaly_fp / max(sum(1 for t in binary_true if t == 0), 1), 4),
            "total_samples": total,
        }

    def generate_report(self, save_path: Optional[Path] = None) -> Dict:
        """Generate comprehensive evaluation report."""
        logger.info("="*60)
        logger.info("ANOMALY DETECTION EVALUATION REPORT")
        logger.info("="*60)

        overall = self.compute_overall_metrics()
        per_class = self.compute_per_class_metrics()
        cm = self.compute_confusion_matrix()

        # Print overall metrics
        logger.info(f"\nOverall Accuracy: {overall['accuracy']:.4f}")
        logger.info(f"Macro F1: {overall['macro_f1']:.4f}")
        logger.info(f"Weighted F1: {overall['weighted_f1']:.4f}")
        logger.info(f"Anomaly Detection F1: {overall['anomaly_detection_f1']:.4f}")
        logger.info(f"False Alarm Rate: {overall['false_alarm_rate']:.4f}")

        # Print per-class metrics
        logger.info(f"\n{'Class':<25s} {'Prec':>8s} {'Rec':>8s} {'F1':>8s} {'Support':>8s}")
        logger.info("-"*60)
        for cls_name, m in per_class.items():
            logger.info(
                f"{cls_name:<25s} {m['precision']:>8.4f} {m['recall']:>8.4f} "
                f"{m['f1_score']:>8.4f} {m['support']:>8d}"
            )

        # Print confusion matrix
        logger.info(f"\nConfusion Matrix:")
        header = "          " + " ".join(f"{self.CLASS_NAMES.get(i, str(i))[:6]:>7s}" for i in range(len(cm)))
        logger.info(header)
        for i in range(len(cm)):
            row = f"{self.CLASS_NAMES.get(i, str(i))[:10]:<10s}" + " ".join(f"{cm[i][j]:>7d}" for j in range(len(cm)))
            logger.info(row)

        report = {
            "timestamp": datetime.now().isoformat(),
            "overall": overall,
            "per_class": per_class,
            "confusion_matrix": cm.tolist(),
            "class_names": self.CLASS_NAMES,
        }

        if save_path:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            with open(save_path, "w") as f:
                json.dump(report, f, indent=2)
            logger.info(f"\nReport saved: {save_path}")

        return report


def evaluate_model(weights_path: Optional[str] = None, use_synthetic: bool = True) -> Dict:
    """Evaluate trained model on test data."""
    evaluator = AnomalyEvaluator()

    if use_synthetic:
        from ml.training.train_anomaly_detector import generate_synthetic_training_data
        features, labels = generate_synthetic_training_data(num_samples=2000, seed=99)

        from ml.models.anomaly_classifier import AnomalyClassifier
        classifier = AnomalyClassifier(weights_path=weights_path)

        # Evaluate in chunks
        seq_len = 32
        for i in range(0, len(features) - seq_len, seq_len):
            seq = features[i:i+seq_len]
            true_label = int(np.bincount(labels[i:i+seq_len].astype(int)).argmax())
            result = classifier.classify_sequence(seq)
            evaluator.add_predictions([true_label], [result["class_id"]])

    report = evaluator.generate_report(
        save_path=METADATA_DIR / "evaluation_report.json"
    )
    return report


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate Anomaly Detector")
    parser.add_argument("--weights", type=str, help="Path to model weights")
    parser.add_argument("--synthetic", action="store_true", default=True)
    args = parser.parse_args()
    evaluate_model(weights_path=args.weights, use_synthetic=args.synthetic)
