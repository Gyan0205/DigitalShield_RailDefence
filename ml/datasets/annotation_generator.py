"""
Digital Shield Rail Defense — Annotation Generator
====================================================
Generates structured annotations for training data by mapping
video sources to anomaly categories, creating YOLO-format labels,
and building train/val/test splits.
"""

import sys
import json
import random
import logging
import hashlib
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from ml.config import (
    RAW_DIR, FRAMES_DIR, ANNOTATIONS_DIR, METADATA_DIR,
    ANOMALY_CLASSES, UCF_CRIME_MAPPING, TRAFFICKING_CLASSES,
    LOG_FORMAT, LOG_DATE_FORMAT,
)

logger = logging.getLogger("annotation_generator")
logger.setLevel(logging.INFO)
if not logger.handlers:
    console = logging.StreamHandler()
    console.setFormatter(logging.Formatter(LOG_FORMAT, LOG_DATE_FORMAT))
    logger.addHandler(console)


class AnnotationGenerator:
    """
    Generates and manages annotations for anomaly detection training.
    
    Supports:
      - UCF Crime category auto-mapping
      - YOLO-format bounding box labels
      - Video-level anomaly labels
      - Frame-level temporal annotations
      - Train/val/test splitting
    """

    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = output_dir or ANNOTATIONS_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.annotations: List[Dict] = []
        self.stats = {"total": 0, "annotated": 0, "normal": 0, "anomalous": 0}

    def _determine_label_from_path(self, video_path: Path) -> Tuple[int, str]:
        """
        Infer anomaly label from directory structure.
        UCF Crime videos are organized by category folder names.
        """
        parts = [p.lower() for p in video_path.parts]

        # Check UCF Crime categories
        for ucf_category, class_id in UCF_CRIME_MAPPING.items():
            if ucf_category.lower() in parts:
                return class_id, ANOMALY_CLASSES.get(class_id, "unknown")

        # Check if in a 'normal' or 'training/testing' directory
        if "normal" in parts or "training" in parts:
            return 0, "normal"

        # Default: mark as normal (to be manually reviewed)
        return 0, "normal"

    def generate_video_annotations(self, dataset_dir: Path, dataset_name: str = "") -> List[Dict]:
        """
        Generate video-level annotations for an entire dataset.
        
        Each annotation contains:
          - video_path, dataset, category, class_id, class_name
          - is_anomalous flag
          - is_trafficking flag (trafficking-specific classes)
        """
        video_files = []
        for ext in (".mp4", ".avi", ".mkv", ".mov", ".wmv"):
            video_files.extend(dataset_dir.rglob(f"*{ext}"))

        logger.info(f"Generating annotations for {len(video_files)} videos in {dataset_dir.name}")
        annotations = []

        for video_path in sorted(video_files):
            class_id, class_name = self._determine_label_from_path(video_path)
            is_anomalous = class_id != 0
            is_trafficking = class_id in TRAFFICKING_CLASSES

            annotation = {
                "video_id": hashlib.md5(str(video_path).encode()).hexdigest()[:12],
                "video_path": str(video_path),
                "video_name": video_path.name,
                "dataset": dataset_name or dataset_dir.name,
                "category": video_path.parent.name,
                "class_id": class_id,
                "class_name": class_name,
                "is_anomalous": is_anomalous,
                "is_trafficking": is_trafficking,
                "split": "",  # Assigned later
                "temporal_annotations": [],  # Frame-level labels if available
            }

            annotations.append(annotation)
            self.stats["total"] += 1
            self.stats["annotated"] += 1
            if is_anomalous:
                self.stats["anomalous"] += 1
            else:
                self.stats["normal"] += 1

        self.annotations.extend(annotations)
        return annotations

    def generate_frame_annotations(self, frames_dir: Path, video_annotation: Dict) -> List[Dict]:
        """
        Generate frame-level annotations for extracted frames.
        Inherits video-level label and adds per-frame metadata.
        """
        frame_annotations = []
        frame_files = sorted(frames_dir.glob("*.jpg")) + sorted(frames_dir.glob("*.png"))

        for i, frame_path in enumerate(frame_files):
            frame_annotations.append({
                "frame_path": str(frame_path),
                "frame_index": i,
                "video_id": video_annotation["video_id"],
                "class_id": video_annotation["class_id"],
                "class_name": video_annotation["class_name"],
                "is_anomalous": video_annotation["is_anomalous"],
                "bounding_boxes": [],  # Populated by detection pipeline
            })

        return frame_annotations

    def generate_yolo_labels(self, detection_results: List[Dict], output_dir: Optional[Path] = None) -> int:
        """
        Convert detection results to YOLO-format label files.
        
        YOLO format: <class_id> <x_center> <y_center> <width> <height>
        All values normalized to [0, 1].
        """
        out = output_dir or (self.output_dir / "yolo_labels")
        out.mkdir(parents=True, exist_ok=True)
        count = 0

        for det in detection_results:
            frame_path = Path(det.get("frame_path", ""))
            label_path = out / f"{frame_path.stem}.txt"
            boxes = det.get("bounding_boxes", [])

            lines = []
            for box in boxes:
                cls_id = box.get("class_id", 0)
                x_c = box.get("x_center", 0)
                y_c = box.get("y_center", 0)
                w = box.get("width", 0)
                h = box.get("height", 0)
                lines.append(f"{cls_id} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}")

            with open(label_path, "w") as f:
                f.write("\n".join(lines))
            count += 1

        logger.info(f"Generated {count} YOLO label files in {out}")
        return count

    def assign_splits(self, train_ratio: float = 0.7, val_ratio: float = 0.15,
                      test_ratio: float = 0.15, seed: int = 42) -> Dict[str, List[Dict]]:
        """
        Assign train/val/test splits with stratification.
        Ensures anomaly class distribution is preserved across splits.
        """
        random.seed(seed)

        # Group by class
        by_class: Dict[int, List[Dict]] = {}
        for ann in self.annotations:
            cls = ann["class_id"]
            by_class.setdefault(cls, []).append(ann)

        splits = {"train": [], "val": [], "test": []}

        for cls_id, items in by_class.items():
            random.shuffle(items)
            n = len(items)
            n_train = int(n * train_ratio)
            n_val = int(n * val_ratio)

            for i, item in enumerate(items):
                if i < n_train:
                    item["split"] = "train"
                    splits["train"].append(item)
                elif i < n_train + n_val:
                    item["split"] = "val"
                    splits["val"].append(item)
                else:
                    item["split"] = "test"
                    splits["test"].append(item)

        logger.info(
            f"Split assigned — Train: {len(splits['train'])}, "
            f"Val: {len(splits['val'])}, Test: {len(splits['test'])}"
        )
        return splits

    def save_annotations(self, filename: str = "annotations.json") -> Path:
        """Save all annotations to JSON."""
        output_path = self.output_dir / filename
        data = {
            "version": "1.0",
            "created_at": datetime.now().isoformat(),
            "stats": self.stats,
            "class_definitions": ANOMALY_CLASSES,
            "trafficking_classes": TRAFFICKING_CLASSES,
            "annotations": self.annotations,
        }
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info(f"Saved {len(self.annotations)} annotations to {output_path}")
        return output_path

    def save_split_files(self, splits: Dict[str, List[Dict]]) -> Dict[str, Path]:
        """Save train/val/test split files."""
        paths = {}
        for split_name, items in splits.items():
            output_path = self.output_dir / f"{split_name}.json"
            with open(output_path, "w") as f:
                json.dump(items, f, indent=2)
            paths[split_name] = output_path

            # Also save as text file (video paths only)
            txt_path = self.output_dir / f"{split_name}.txt"
            with open(txt_path, "w") as f:
                for item in items:
                    f.write(f"{item['video_path']}\t{item['class_id']}\n")
            logger.info(f"Saved {split_name} split: {len(items)} entries")

        return paths

    def generate_class_distribution_report(self) -> Dict:
        """Generate class distribution statistics."""
        distribution = {}
        for ann in self.annotations:
            cls_name = ann["class_name"]
            split = ann.get("split", "unassigned")
            key = f"{cls_name}"
            distribution.setdefault(key, {"total": 0, "train": 0, "val": 0, "test": 0})
            distribution[key]["total"] += 1
            if split in ("train", "val", "test"):
                distribution[key][split] += 1

        report = {"distribution": distribution, "total_samples": len(self.annotations), "stats": self.stats}
        report_path = METADATA_DIR / "class_distribution.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        logger.info(f"Class distribution report saved to {report_path}")
        return report

    def run_full_pipeline(self) -> Dict:
        """Run the complete annotation pipeline for all datasets."""
        # Generate video-level annotations
        for dataset_dir in sorted(d for d in RAW_DIR.iterdir() if d.is_dir()):
            self.generate_video_annotations(dataset_dir)

        # Assign splits
        splits = self.assign_splits()

        # Save everything
        self.save_annotations()
        self.save_split_files(splits)
        report = self.generate_class_distribution_report()

        logger.info(f"\nAnnotation pipeline complete: {self.stats}")
        return {"stats": self.stats, "distribution": report}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Annotation Generator")
    parser.add_argument("--input", type=str, help="Input dataset directory")
    parser.add_argument("--output", type=str, help="Output annotations directory")
    parser.add_argument("--all", action="store_true", help="Annotate all datasets")
    args = parser.parse_args()

    generator = AnnotationGenerator(output_dir=Path(args.output) if args.output else None)
    if args.all:
        generator.run_full_pipeline()
    elif args.input:
        generator.generate_video_annotations(Path(args.input))
        splits = generator.assign_splits()
        generator.save_annotations()
        generator.save_split_files(splits)
