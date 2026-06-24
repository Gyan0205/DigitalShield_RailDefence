"""
Digital Shield Rail Defense — UCF Crime Dataset Preparation
=============================================================
Organizes UCF Crime Dataset into the standardized folder structure,
maps crime categories to trafficking-relevant anomaly classes,
and generates split files for training.
"""

import sys
import json
import shutil
import logging
from pathlib import Path
from typing import Dict, List
from collections import Counter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from ml.config import (
    UCF_CRIME_DIR, PROCESSED_DIR, ANNOTATIONS_DIR, METADATA_DIR,
    UCF_CRIME_MAPPING, ANOMALY_CLASSES, LOG_FORMAT, LOG_DATE_FORMAT,
)

logger = logging.getLogger("prepare_ucf_crime")
logger.setLevel(logging.INFO)
if not logger.handlers:
    console = logging.StreamHandler()
    console.setFormatter(logging.Formatter(LOG_FORMAT, LOG_DATE_FORMAT))
    logger.addHandler(console)


def organize_ucf_crime(source_dir: Path = None) -> Dict:
    """
    Organize UCF Crime dataset into standardized structure.
    
    UCF Crime structure:
      UCF_Crime/
      ├── Abuse/
      │   ├── Abuse001.mp4
      │   └── ...
      ├── Arrest/
      ├── Arson/
      ├── Assault/
      ├── Normal/
      └── ...
    
    Maps to our anomaly classes and generates annotation manifest.
    """
    src = source_dir or UCF_CRIME_DIR
    if not src.exists():
        logger.warning(f"UCF Crime directory not found: {src}")
        return {"status": "not_found"}

    logger.info(f"Organizing UCF Crime dataset from {src}")

    # Scan categories
    categories = [d for d in src.iterdir() if d.is_dir()]
    logger.info(f"Found {len(categories)} categories: {[c.name for c in categories]}")

    manifest = []
    class_counts = Counter()

    for category_dir in sorted(categories):
        category_name = category_dir.name
        class_id = UCF_CRIME_MAPPING.get(category_name, 0)
        class_name = ANOMALY_CLASSES.get(class_id, "unknown")

        videos = list(category_dir.glob("*.mp4")) + list(category_dir.glob("*.avi"))
        logger.info(f"  {category_name} → {class_name} (class {class_id}): {len(videos)} videos")

        for video in sorted(videos):
            manifest.append({
                "video_path": str(video),
                "video_name": video.name,
                "ucf_category": category_name,
                "class_id": class_id,
                "class_name": class_name,
                "is_anomalous": class_id != 0,
                "dataset": "ucf_crime",
            })
            class_counts[class_name] += 1

    # Save manifest
    manifest_path = ANNOTATIONS_DIR / "ucf_crime_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w") as f:
        json.dump({
            "dataset": "UCF Crime",
            "total_videos": len(manifest),
            "class_distribution": dict(class_counts),
            "category_mapping": UCF_CRIME_MAPPING,
            "videos": manifest,
        }, f, indent=2)

    logger.info(f"\nUCF Crime organized: {len(manifest)} videos")
    logger.info(f"Class distribution: {dict(class_counts)}")
    logger.info(f"Manifest saved: {manifest_path}")

    return {"status": "complete", "total": len(manifest), "distribution": dict(class_counts)}


def prepare_ucf_temporal_annotations(annotation_file: Path = None) -> Dict:
    """
    Parse UCF Crime temporal annotation file if available.
    UCF provides temporal annotation files indicating anomaly start/end frames.
    """
    ann_file = annotation_file or UCF_CRIME_DIR / "Temporal_Anomaly_Annotation.txt"
    if not ann_file.exists():
        logger.info("No temporal annotation file found — using video-level labels only")
        return {}

    logger.info(f"Parsing temporal annotations from {ann_file}")
    temporal = {}

    with open(ann_file) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3:
                video_name = parts[0]
                # Format varies; typically: VideoName AnomalyStart AnomalyEnd
                try:
                    start_frame = int(parts[1])
                    end_frame = int(parts[2])
                    temporal[video_name] = {
                        "anomaly_start": start_frame,
                        "anomaly_end": end_frame,
                    }
                except ValueError:
                    continue

    output_path = METADATA_DIR / "ucf_temporal_annotations.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(temporal, f, indent=2)

    logger.info(f"Parsed {len(temporal)} temporal annotations")
    return temporal


if __name__ == "__main__":
    organize_ucf_crime()
    prepare_ucf_temporal_annotations()
