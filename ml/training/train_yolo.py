"""
Digital Shield Rail Defense — YOLOv8 Training Pipeline
=========================================================
Complete training workflow for person detection on railway
CCTV footage using YOLOv8.

Supports:
  - Custom dataset preparation (YOLO format)
  - Transfer learning from pre-trained weights
  - Hyperparameter tuning
  - Multi-GPU training
  - Export to ONNX/TensorRT
"""

import sys
import json
import time
import shutil
import logging
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from ml.config import (
    DATASET_ROOT, PROCESSED_DIR, WEIGHTS_DIR,
    METADATA_DIR, LOG_FORMAT, LOG_DATE_FORMAT,
)

logger = logging.getLogger("train_yolo")
logger.setLevel(logging.INFO)
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter(LOG_FORMAT, LOG_DATE_FORMAT))
    logger.addHandler(_h)


# ============================================================================
# DATASET PREPARATION (YOLO FORMAT)
# ============================================================================

def prepare_yolo_dataset(
    source_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    train_ratio: float = 0.7,
    val_ratio: float = 0.2,
    max_images: int = 10000,
) -> Dict:
    """
    Prepare dataset in YOLO format.

    Creates:
      dataset/yolo/
        train/images/  train/labels/
        val/images/    val/labels/
        test/images/   test/labels/
        data.yaml
    """
    out = output_dir or PROCESSED_DIR / "yolo"
    out.mkdir(parents=True, exist_ok=True)

    for split in ["train", "val", "test"]:
        (out / split / "images").mkdir(parents=True, exist_ok=True)
        (out / split / "labels").mkdir(parents=True, exist_ok=True)

    # Generate synthetic annotations if no source
    if not source_dir or not source_dir.exists():
        logger.info("No source dataset found. Generating synthetic YOLO annotations...")
        stats = _generate_synthetic_yolo(out, max_images)
    else:
        stats = _convert_to_yolo(source_dir, out, train_ratio, val_ratio, max_images)

    # Create data.yaml
    data_yaml = {
        "path": str(out.resolve()),
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "names": {
            0: "person",
            1: "suspicious_person",
            2: "minor",
            3: "group",
        },
        "nc": 4,
    }

    yaml_path = out / "data.yaml"
    try:
        import yaml
        with open(yaml_path, "w") as f:
            yaml.dump(data_yaml, f, default_flow_style=False)
    except ImportError:
        with open(yaml_path, "w") as f:
            f.write(f"path: {out.resolve()}\n")
            f.write("train: train/images\nval: val/images\ntest: test/images\n")
            f.write("nc: 4\nnames:\n  0: person\n  1: suspicious_person\n  2: minor\n  3: group\n")

    logger.info(f"YOLO dataset prepared: {out}")
    stats["data_yaml"] = str(yaml_path)
    return stats


def _generate_synthetic_yolo(out: Path, max_images: int) -> Dict:
    """Generate synthetic YOLO labels for testing."""
    np.random.seed(42)
    splits = {"train": int(max_images * 0.7), "val": int(max_images * 0.2), "test": int(max_images * 0.1)}
    total = 0

    for split, count in splits.items():
        for i in range(count):
            # Create dummy image placeholder
            img_name = f"frame_{split}_{i:05d}.txt"
            label_path = out / split / "labels" / img_name

            # Generate random bounding boxes
            n_objects = np.random.randint(1, 8)
            lines = []
            for _ in range(n_objects):
                cls = np.random.choice([0, 1, 2, 3], p=[0.55, 0.20, 0.15, 0.10])
                cx = np.random.uniform(0.1, 0.9)
                cy = np.random.uniform(0.2, 0.8)
                w = np.random.uniform(0.03, 0.15)
                h = np.random.uniform(0.1, 0.4)
                lines.append(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

            with open(label_path, "w") as f:
                f.write("\n".join(lines))
            total += 1

    return {"total_images": total, "splits": splits}


def _convert_to_yolo(source: Path, out: Path, train_r: float, val_r: float, max_img: int) -> Dict:
    """Convert existing annotated dataset to YOLO format."""
    images = list(source.glob("**/*.jpg")) + list(source.glob("**/*.png"))
    images = images[:max_img]
    np.random.shuffle(images)

    n = len(images)
    train_end = int(n * train_r)
    val_end = int(n * (train_r + val_r))

    splits = {"train": 0, "val": 0, "test": 0}
    for i, img in enumerate(images):
        if i < train_end:
            split = "train"
        elif i < val_end:
            split = "val"
        else:
            split = "test"

        dst = out / split / "images" / img.name
        shutil.copy2(img, dst)
        splits[split] += 1

    return {"total_images": n, "splits": splits}


# ============================================================================
# YOLO TRAINING
# ============================================================================

def train_yolo(
    data_yaml: Optional[str] = None,
    model_size: str = "n",
    epochs: int = 100,
    batch_size: int = 16,
    img_size: int = 640,
    lr0: float = 0.01,
    patience: int = 20,
    device: str = "auto",
    project: Optional[str] = None,
    name: str = "ds_person_detect",
    resume: bool = False,
) -> Dict:
    """
    Train YOLOv8 for person detection on railway CCTV.

    Args:
        model_size: n/s/m/l/x (nano to extra-large)
        epochs: Training epochs
        batch_size: Batch size
        img_size: Input image size
        lr0: Initial learning rate
        patience: Early stopping patience
        device: cuda/cpu/auto
        project: Output project directory
        name: Experiment name
        resume: Resume from last checkpoint

    Returns:
        Training report with metrics
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        logger.error("ultralytics not installed. Run: pip install ultralytics")
        return _simulate_yolo_training(epochs, model_size, name)

    model_name = f"yolov8{model_size}.pt"
    model = YOLO(model_name)

    proj = project or str(WEIGHTS_DIR / "yolo_runs")
    data = data_yaml or str(PROCESSED_DIR / "yolo" / "data.yaml")

    logger.info("=" * 60)
    logger.info("YOLOv8 PERSON DETECTION TRAINING")
    logger.info("=" * 60)
    logger.info(f"Model: {model_name} | Epochs: {epochs} | Batch: {batch_size}")
    logger.info(f"Image size: {img_size} | LR: {lr0} | Patience: {patience}")

    results = model.train(
        data=data,
        epochs=epochs,
        batch=batch_size,
        imgsz=img_size,
        lr0=lr0,
        patience=patience,
        device=device if device != "auto" else None,
        project=proj,
        name=name,
        resume=resume,
        verbose=True,
        save=True,
        save_period=10,
        plots=True,
    )

    # Extract metrics
    report = {
        "status": "complete",
        "model": model_name,
        "epochs": epochs,
        "best_mAP50": round(results.results_dict.get("metrics/mAP50(B)", 0), 4),
        "best_mAP50_95": round(results.results_dict.get("metrics/mAP50-95(B)", 0), 4),
        "precision": round(results.results_dict.get("metrics/precision(B)", 0), 4),
        "recall": round(results.results_dict.get("metrics/recall(B)", 0), 4),
        "save_dir": str(results.save_dir),
        "best_weights": str(results.save_dir / "weights" / "best.pt"),
    }

    # Save report
    report_path = METADATA_DIR / f"yolo_training_{name}.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    return report


def _simulate_yolo_training(epochs: int, model_size: str, name: str) -> Dict:
    """Simulate YOLO training when ultralytics is not installed."""
    logger.info("Simulating YOLOv8 training (ultralytics not available)...")

    np.random.seed(42)
    history = []
    best_map = 0

    for epoch in range(1, min(epochs, 20) + 1):
        progress = epoch / min(epochs, 20)
        noise = np.random.uniform(-0.02, 0.02)

        map50 = min(0.3 + 0.55 * progress + noise, 0.92)
        map50_95 = min(0.15 + 0.40 * progress + noise, 0.72)
        precision = min(0.4 + 0.45 * progress + noise, 0.91)
        recall = min(0.35 + 0.50 * progress + noise, 0.89)
        box_loss = max(0.08 - 0.04 * progress + noise * 0.1, 0.02)

        best_map = max(best_map, map50)

        history.append({
            "epoch": epoch, "mAP50": round(map50, 4),
            "mAP50-95": round(map50_95, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "box_loss": round(box_loss, 4),
        })

        logger.info(
            f"Epoch {epoch:>3d}/{min(epochs,20)} | "
            f"mAP50={map50:.4f} | mAP50-95={map50_95:.4f} | "
            f"P={precision:.4f} | R={recall:.4f} | "
            f"loss={box_loss:.4f}"
        )

    report = {
        "status": "simulated",
        "model": f"yolov8{model_size}.pt",
        "epochs_trained": len(history),
        "best_mAP50": round(best_map, 4),
        "best_mAP50_95": round(history[-1]["mAP50-95"], 4),
        "precision": round(history[-1]["precision"], 4),
        "recall": round(history[-1]["recall"], 4),
        "history": history,
        "timestamp": datetime.now().isoformat(),
    }

    report_path = METADATA_DIR / f"yolo_training_{name}.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"Report saved: {report_path}")

    return report


# ============================================================================
# YOLO EXPORT
# ============================================================================

def export_yolo(
    weights_path: str,
    format: str = "onnx",
    img_size: int = 640,
) -> Dict:
    """Export trained YOLOv8 model to deployment format."""
    try:
        from ultralytics import YOLO
        model = YOLO(weights_path)
        path = model.export(format=format, imgsz=img_size)
        return {"status": "exported", "format": format, "path": str(path)}
    except ImportError:
        return {"status": "simulated", "format": format, "message": "ultralytics not installed"}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train YOLOv8 Person Detector")
    parser.add_argument("--model", default="n", choices=["n", "s", "m", "l", "x"])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--img-size", type=int, default=640)
    parser.add_argument("--data", type=str, help="Path to data.yaml")
    parser.add_argument("--prepare-data", action="store_true", help="Prepare dataset first")
    args = parser.parse_args()

    if args.prepare_data:
        prepare_yolo_dataset()

    train_yolo(
        data_yaml=args.data, model_size=args.model,
        epochs=args.epochs, batch_size=args.batch,
        img_size=args.img_size,
    )
