"""
Digital Shield Rail Defense — ShanghaiTech Dataset Preparation
================================================================
Organizes ShanghaiTech Campus anomaly detection dataset.
"""

import sys
import json
import logging
from pathlib import Path
from typing import Dict
from collections import Counter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from ml.config import (
    SHANGHAITECH_DIR, ANNOTATIONS_DIR, METADATA_DIR,
    LOG_FORMAT, LOG_DATE_FORMAT,
)

logger = logging.getLogger("prepare_shanghaitech")
logger.setLevel(logging.INFO)
if not logger.handlers:
    console = logging.StreamHandler()
    console.setFormatter(logging.Formatter(LOG_FORMAT, LOG_DATE_FORMAT))
    logger.addHandler(console)


def organize_shanghaitech(source_dir: Path = None) -> Dict:
    """
    Organize ShanghaiTech dataset.
    
    ShanghaiTech structure:
      training/
      └── videos/
      testing/
      └── frames/ or videos/
    
    All testing videos contain at least one anomaly.
    """
    src = source_dir or SHANGHAITECH_DIR
    if not src.exists():
        logger.warning(f"ShanghaiTech directory not found: {src}")
        return {"status": "not_found"}

    logger.info(f"Organizing ShanghaiTech dataset from {src}")
    manifest = []

    # Scan for video files recursively
    for video_path in sorted(src.rglob("*.avi")):
        parts = [p.lower() for p in video_path.parts]
        is_training = "training" in parts
        is_anomalous = "testing" in parts  # ShanghaiTech: all test videos have anomalies

        manifest.append({
            "video_path": str(video_path),
            "video_name": video_path.name,
            "split": "train" if is_training else "test",
            "class_id": 0 if is_training else 1,
            "class_name": "normal" if is_training else "anomalous",
            "is_anomalous": is_anomalous,
            "dataset": "shanghaitech",
        })

    manifest_path = ANNOTATIONS_DIR / "shanghaitech_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w") as f:
        json.dump({
            "dataset": "ShanghaiTech Campus",
            "total_videos": len(manifest),
            "train": sum(1 for m in manifest if m["split"] == "train"),
            "test": sum(1 for m in manifest if m["split"] == "test"),
            "videos": manifest,
        }, f, indent=2)

    logger.info(f"ShanghaiTech organized: {len(manifest)} videos")
    return {"status": "complete", "total": len(manifest)}


if __name__ == "__main__":
    organize_shanghaitech()
