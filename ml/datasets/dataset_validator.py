"""
Digital Shield Rail Defense — Dataset Validator
=================================================
Validates dataset integrity, completeness, and quality
before training. Runs comprehensive checks on all pipeline outputs.
"""

import sys
import json
import logging
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from ml.config import (
    DATASET_ROOT, RAW_DIR, PROCESSED_DIR, FRAMES_DIR, CLIPS_DIR,
    ANNOTATIONS_DIR, METADATA_DIR, MODELS_DIR,
    ANOMALY_CLASSES, DATASET_SOURCES,
    LOG_FORMAT, LOG_DATE_FORMAT,
)

logger = logging.getLogger("dataset_validator")
logger.setLevel(logging.INFO)
if not logger.handlers:
    console = logging.StreamHandler()
    console.setFormatter(logging.Formatter(LOG_FORMAT, LOG_DATE_FORMAT))
    logger.addHandler(console)


class DatasetValidator:
    """
    Comprehensive dataset validation pipeline.
    
    Checks:
      1. Directory structure integrity
      2. Raw dataset presence and file counts
      3. Processed data completeness
      4. Annotation consistency
      5. Metadata completeness
      6. Train/val/test split balance
      7. File format validation
      8. Cross-reference integrity
    """

    def __init__(self):
        self.results: List[Dict] = []
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.passed: int = 0
        self.failed: int = 0

    def _check(self, name: str, condition: bool, error_msg: str = "",
               warning_msg: str = "") -> bool:
        """Run a single validation check."""
        result = {
            "check": name,
            "passed": condition,
            "timestamp": datetime.now().isoformat(),
        }

        if condition:
            self.passed += 1
            logger.info(f"  ✓ {name}")
        else:
            self.failed += 1
            if error_msg:
                result["error"] = error_msg
                self.errors.append(f"[{name}] {error_msg}")
                logger.error(f"  ✗ {name} — {error_msg}")
            elif warning_msg:
                result["warning"] = warning_msg
                self.warnings.append(f"[{name}] {warning_msg}")
                logger.warning(f"  ⚠ {name} — {warning_msg}")

        self.results.append(result)
        return condition

    def validate_directory_structure(self) -> bool:
        """Check that all required directories exist."""
        logger.info("\n[1/8] Validating directory structure...")
        all_good = True
        required_dirs = {
            "dataset/raw": RAW_DIR,
            "dataset/processed": PROCESSED_DIR,
            "dataset/processed/frames": FRAMES_DIR,
            "dataset/processed/clips": CLIPS_DIR,
            "dataset/annotations": ANNOTATIONS_DIR,
            "dataset/metadata": METADATA_DIR,
            "dataset/models": MODELS_DIR,
        }
        for name, path in required_dirs.items():
            ok = self._check(f"Directory exists: {name}", path.exists(),
                           error_msg=f"Missing directory: {path}")
            all_good = all_good and ok
        return all_good

    def validate_raw_datasets(self) -> bool:
        """Check that raw datasets are downloaded."""
        logger.info("\n[2/8] Validating raw datasets...")
        all_good = True
        for ds_name, source in DATASET_SOURCES.items():
            target = source.target_dir
            exists = target.exists()
            file_count = sum(1 for _ in target.rglob("*") if _.is_file()) if exists else 0
            ok = self._check(
                f"Dataset present: {ds_name}",
                exists and file_count > 0,
                warning_msg=f"{ds_name}: {'no files' if exists else 'directory missing'}"
            )
            if ok:
                logger.info(f"    → {file_count} files found")
            all_good = all_good and ok
        return all_good

    def validate_processed_data(self) -> bool:
        """Check processed frames and clips."""
        logger.info("\n[3/8] Validating processed data...")
        all_good = True

        # Check frames
        frame_count = sum(1 for _ in FRAMES_DIR.rglob("*") if _.is_file()) if FRAMES_DIR.exists() else 0
        ok = self._check("Extracted frames exist", frame_count > 0,
                       warning_msg=f"No extracted frames in {FRAMES_DIR}")
        all_good = all_good and ok
        if ok:
            logger.info(f"    → {frame_count} frame files")

        # Check clips
        clip_count = sum(1 for _ in CLIPS_DIR.rglob("*.mp4")) if CLIPS_DIR.exists() else 0
        ok = self._check("Processed clips exist", clip_count > 0,
                       warning_msg=f"No processed clips in {CLIPS_DIR}")
        all_good = all_good and ok
        if ok:
            logger.info(f"    → {clip_count} clip files")

        return all_good

    def validate_annotations(self) -> bool:
        """Check annotation files for completeness and consistency."""
        logger.info("\n[4/8] Validating annotations...")
        all_good = True

        ann_file = ANNOTATIONS_DIR / "annotations.json"
        ok = self._check("annotations.json exists", ann_file.exists(),
                       warning_msg="No annotations.json found")
        all_good = all_good and ok

        if ok:
            with open(ann_file) as f:
                data = json.load(f)

            annotations = data.get("annotations", [])
            ok = self._check("Annotations not empty", len(annotations) > 0,
                           error_msg="annotations.json contains no entries")
            all_good = all_good and ok

            if annotations:
                # Check required fields
                required_fields = ["video_id", "video_path", "class_id", "class_name", "is_anomalous"]
                sample = annotations[0]
                for field in required_fields:
                    ok = self._check(f"Field present: {field}", field in sample,
                                   error_msg=f"Missing field '{field}' in annotations")
                    all_good = all_good and ok

                # Check class distribution
                classes_found = set(a["class_id"] for a in annotations)
                ok = self._check("Multiple classes present", len(classes_found) > 1,
                               warning_msg=f"Only {len(classes_found)} class(es) found")
                all_good = all_good and ok

                # Check for anomalous samples
                anomalous = sum(1 for a in annotations if a["is_anomalous"])
                ok = self._check("Anomalous samples present", anomalous > 0,
                               warning_msg="No anomalous samples in annotations")
                all_good = all_good and ok
                logger.info(f"    → {len(annotations)} annotations, {anomalous} anomalous")

        # Check split files
        for split in ["train", "val", "test"]:
            split_file = ANNOTATIONS_DIR / f"{split}.json"
            self._check(f"{split}.json exists", split_file.exists(),
                       warning_msg=f"No {split} split file")

        return all_good

    def validate_metadata(self) -> bool:
        """Check metadata files."""
        logger.info("\n[5/8] Validating metadata...")
        all_good = True

        meta_file = METADATA_DIR / "video_metadata.json"
        ok = self._check("video_metadata.json exists", meta_file.exists(),
                       warning_msg="No video metadata found")
        all_good = all_good and ok

        if ok:
            with open(meta_file) as f:
                data = json.load(f)
            records = data.get("records", [])
            ok = self._check("Metadata records not empty", len(records) > 0,
                           error_msg="video_metadata.json has no records")
            all_good = all_good and ok

            if records:
                sample = records[0]
                for field in ["station", "platform_number", "camera_id", "train", "timestamp"]:
                    ok = self._check(f"Metadata field: {field}", field in sample,
                                   error_msg=f"Missing '{field}' in metadata")
                    all_good = all_good and ok

        # Camera registry
        cam_file = METADATA_DIR / "camera_registry.json"
        ok = self._check("camera_registry.json exists", cam_file.exists(),
                       warning_msg="No camera registry")
        all_good = all_good and ok

        # Train schedules
        sched_file = METADATA_DIR / "train_schedules.json"
        ok = self._check("train_schedules.json exists", sched_file.exists(),
                       warning_msg="No train schedules")
        all_good = all_good and ok

        return all_good

    def validate_split_balance(self) -> bool:
        """Check train/val/test split ratios."""
        logger.info("\n[6/8] Validating split balance...")
        all_good = True
        counts = {}

        for split in ["train", "val", "test"]:
            split_file = ANNOTATIONS_DIR / f"{split}.json"
            if split_file.exists():
                with open(split_file) as f:
                    counts[split] = len(json.load(f))
            else:
                counts[split] = 0

        total = sum(counts.values())
        if total > 0:
            for split, count in counts.items():
                ratio = count / total
                logger.info(f"    → {split}: {count} ({ratio:.1%})")

            # Check reasonable ratios
            if counts["train"] > 0:
                train_ratio = counts["train"] / total
                ok = self._check("Train ratio reasonable", 0.5 <= train_ratio <= 0.85,
                               warning_msg=f"Train ratio: {train_ratio:.1%}")
                all_good = all_good and ok
        else:
            self._check("Split files have data", False, warning_msg="No split data found")
            all_good = False

        return all_good

    def validate_file_formats(self) -> bool:
        """Validate file formats and check for corruption."""
        logger.info("\n[7/8] Validating file formats...")
        all_good = True

        # Check JSON files are valid
        json_files = list(METADATA_DIR.glob("*.json")) + list(ANNOTATIONS_DIR.glob("*.json"))
        for jf in json_files:
            try:
                with open(jf) as f:
                    json.load(f)
                self._check(f"Valid JSON: {jf.name}", True)
            except json.JSONDecodeError as e:
                self._check(f"Valid JSON: {jf.name}", False, error_msg=str(e))
                all_good = False

        return all_good

    def validate_cross_references(self) -> bool:
        """Check cross-referential integrity between annotations and metadata."""
        logger.info("\n[8/8] Validating cross-references...")
        all_good = True

        ann_file = ANNOTATIONS_DIR / "annotations.json"
        meta_file = METADATA_DIR / "video_metadata.json"

        if ann_file.exists() and meta_file.exists():
            with open(ann_file) as f:
                ann_data = json.load(f)
            with open(meta_file) as f:
                meta_data = json.load(f)

            ann_ids = set(a["video_id"] for a in ann_data.get("annotations", []))
            meta_ids = set(r["video_id"] for r in meta_data.get("records", []))

            overlap = ann_ids & meta_ids
            ok = self._check(
                "Annotation-metadata cross-reference",
                len(overlap) > 0 or (len(ann_ids) == 0 and len(meta_ids) == 0),
                warning_msg=f"Only {len(overlap)}/{len(ann_ids)} annotations have metadata"
            )
            all_good = all_good and ok
        else:
            self._check("Cross-reference files present", False,
                       warning_msg="Missing annotations or metadata for cross-check")
            all_good = False

        return all_good

    def run_full_validation(self) -> Dict:
        """Run all validation checks and generate report."""
        logger.info("="*60)
        logger.info("DIGITAL SHIELD — DATASET VALIDATION PIPELINE")
        logger.info("="*60)

        checks = [
            self.validate_directory_structure,
            self.validate_raw_datasets,
            self.validate_processed_data,
            self.validate_annotations,
            self.validate_metadata,
            self.validate_split_balance,
            self.validate_file_formats,
            self.validate_cross_references,
        ]

        section_results = {}
        for check_fn in checks:
            section_results[check_fn.__name__] = check_fn()

        # Generate report
        report = {
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total_checks": self.passed + self.failed,
                "passed": self.passed,
                "failed": self.failed,
                "pass_rate": f"{self.passed / (self.passed + self.failed) * 100:.1f}%" if (self.passed + self.failed) > 0 else "N/A",
                "errors": len(self.errors),
                "warnings": len(self.warnings),
            },
            "sections": section_results,
            "errors": self.errors,
            "warnings": self.warnings,
            "checks": self.results,
        }

        # Save report
        report_path = METADATA_DIR / "validation_report.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)

        # Print summary
        logger.info("\n" + "="*60)
        logger.info("VALIDATION SUMMARY")
        logger.info("="*60)
        logger.info(f"  Passed:   {self.passed}")
        logger.info(f"  Failed:   {self.failed}")
        logger.info(f"  Errors:   {len(self.errors)}")
        logger.info(f"  Warnings: {len(self.warnings)}")
        logger.info(f"  Report:   {report_path}")
        logger.info("="*60)

        if self.errors:
            logger.error("\nERRORS:")
            for e in self.errors:
                logger.error(f"  • {e}")

        if self.warnings:
            logger.warning("\nWARNINGS:")
            for w in self.warnings:
                logger.warning(f"  • {w}")

        return report


if __name__ == "__main__":
    validator = DatasetValidator()
    report = validator.run_full_validation()
    sys.exit(0 if validator.failed == 0 else 1)
