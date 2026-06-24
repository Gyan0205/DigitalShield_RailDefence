"""
Digital Shield Rail Defense — Pipeline Orchestrator
=====================================================
Master orchestrator that runs the entire dataset management
pipeline end-to-end in the correct dependency order.

Pipeline stages:
  1. Download datasets (or verify existing)
  2. Generate synthetic railway videos
  3. Preprocess all videos (normalize, clip)
  4. Extract frames
  5. Generate annotations
  6. Generate railway metadata
  7. Validate entire dataset
"""

import sys
import json
import time
import logging
from pathlib import Path
from typing import Optional, Dict, List
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from ml.config import (
    DATASET_ROOT, RAW_DIR, PROCESSED_DIR, FRAMES_DIR,
    ANNOTATIONS_DIR, METADATA_DIR,
    LOG_FORMAT, LOG_DATE_FORMAT,
)
from ml.datasets.download_datasets import DatasetDownloader
from ml.datasets.preprocess_videos import VideoPreprocessor
from ml.datasets.frame_extractor import FrameExtractor
from ml.datasets.annotation_generator import AnnotationGenerator
from ml.datasets.metadata_generator import MetadataGenerator
from ml.datasets.synthetic_generator import SyntheticRailwayGenerator
from ml.datasets.dataset_validator import DatasetValidator

logger = logging.getLogger("pipeline_orchestrator")
logger.setLevel(logging.INFO)
if not logger.handlers:
    console = logging.StreamHandler()
    console.setFormatter(logging.Formatter(LOG_FORMAT, LOG_DATE_FORMAT))
    logger.addHandler(console)


class PipelineOrchestrator:
    """
    Master orchestrator for the complete dataset management pipeline.
    
    Runs all stages in dependency order with progress tracking,
    error handling, and comprehensive reporting.
    """

    def __init__(self, skip_download: bool = False, skip_large: bool = True,
                 synthetic_samples: int = 20):
        self.skip_download = skip_download
        self.skip_large = skip_large
        self.synthetic_samples = synthetic_samples
        self.stage_results: Dict[str, Dict] = {}
        self.start_time: Optional[float] = None

    def _run_stage(self, stage_num: int, stage_name: str, func, **kwargs) -> bool:
        """Run a single pipeline stage with timing and error handling."""
        logger.info(f"\n{'='*70}")
        logger.info(f"STAGE {stage_num}: {stage_name}")
        logger.info(f"{'='*70}")

        stage_start = time.time()
        try:
            result = func(**kwargs)
            elapsed = time.time() - stage_start
            self.stage_results[stage_name] = {
                "status": "complete",
                "duration_seconds": round(elapsed, 2),
                "timestamp": datetime.now().isoformat(),
            }
            logger.info(f"Stage {stage_num} complete ({elapsed:.1f}s)")
            return True

        except Exception as e:
            elapsed = time.time() - stage_start
            self.stage_results[stage_name] = {
                "status": "failed",
                "error": str(e),
                "duration_seconds": round(elapsed, 2),
                "timestamp": datetime.now().isoformat(),
            }
            logger.error(f"Stage {stage_num} FAILED: {e}")
            import traceback
            traceback.print_exc()
            return False

    def stage_1_download(self) -> Dict:
        """Stage 1: Download all required datasets."""
        downloader = DatasetDownloader()
        if self.skip_download:
            logger.info("Skipping downloads (--skip-download)")
            return downloader.get_status()
        return downloader.download_all(skip_large=self.skip_large)

    def stage_2_synthetic(self) -> List[Dict]:
        """Stage 2: Generate synthetic railway surveillance videos."""
        generator = SyntheticRailwayGenerator()
        return generator.generate_dataset(samples_per_behavior=self.synthetic_samples)

    def stage_3_preprocess(self) -> Dict:
        """Stage 3: Preprocess all raw videos."""
        processor = VideoPreprocessor()
        return processor.process_all_datasets()

    def stage_4_extract_frames(self) -> Dict:
        """Stage 4: Extract frames from all videos."""
        extractor = FrameExtractor()
        return extractor.extract_all_datasets()

    def stage_5_annotate(self) -> Dict:
        """Stage 5: Generate annotations for all data."""
        annotator = AnnotationGenerator()
        return annotator.run_full_pipeline()

    def stage_6_metadata(self) -> None:
        """Stage 6: Generate synthetic railway metadata."""
        generator = MetadataGenerator()
        generator.generate_station_camera_registry()
        generator.generate_train_schedule_db()
        ann_file = ANNOTATIONS_DIR / "annotations.json"
        generator.generate_all(annotations_file=ann_file)

    def stage_7_validate(self) -> Dict:
        """Stage 7: Validate the entire dataset."""
        validator = DatasetValidator()
        return validator.run_full_validation()

    def run_full_pipeline(self) -> Dict:
        """
        Execute the complete dataset management pipeline.
        
        Returns:
            Pipeline execution report
        """
        self.start_time = time.time()

        logger.info("╔" + "═"*68 + "╗")
        logger.info("║  DIGITAL SHIELD RAIL DEFENSE — DATASET MANAGEMENT PIPELINE       ║")
        logger.info("║  Autonomous download, preprocess, annotate, validate              ║")
        logger.info("╚" + "═"*68 + "╝")

        stages = [
            (1, "Download Datasets", self.stage_1_download),
            (2, "Generate Synthetic Videos", self.stage_2_synthetic),
            (3, "Preprocess Videos", self.stage_3_preprocess),
            (4, "Extract Frames", self.stage_4_extract_frames),
            (5, "Generate Annotations", self.stage_5_annotate),
            (6, "Generate Railway Metadata", self.stage_6_metadata),
            (7, "Validate Dataset", self.stage_7_validate),
        ]

        for num, name, func in stages:
            success = self._run_stage(num, name, func)
            if not success and num <= 2:
                logger.warning(f"Non-critical stage {num} failed, continuing...")
            elif not success:
                logger.error(f"Critical stage {num} failed.")
                # Continue to attempt remaining stages

        total_time = time.time() - self.start_time

        # Generate final report
        report = {
            "pipeline": "Digital Shield Dataset Management",
            "version": "1.0",
            "timestamp": datetime.now().isoformat(),
            "total_duration_seconds": round(total_time, 2),
            "total_duration_human": f"{int(total_time // 60)}m {int(total_time % 60)}s",
            "stages": self.stage_results,
            "config": {
                "skip_download": self.skip_download,
                "skip_large": self.skip_large,
                "synthetic_samples": self.synthetic_samples,
            },
        }

        # Save report
        report_path = METADATA_DIR / "pipeline_report.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)

        # Print summary
        logger.info("\n" + "╔" + "═"*68 + "╗")
        logger.info("║  PIPELINE EXECUTION SUMMARY                                      ║")
        logger.info("╚" + "═"*68 + "╝")
        for name, result in self.stage_results.items():
            status = "✓" if result["status"] == "complete" else "✗"
            duration = result.get("duration_seconds", 0)
            logger.info(f"  {status} {name:<35s} {duration:>8.1f}s")
        logger.info(f"\n  Total: {report['total_duration_human']}")
        logger.info(f"  Report: {report_path}")

        return report

    def run_stage_only(self, stage_number: int) -> bool:
        """Run a specific stage only."""
        stage_map = {
            1: ("Download Datasets", self.stage_1_download),
            2: ("Generate Synthetic Videos", self.stage_2_synthetic),
            3: ("Preprocess Videos", self.stage_3_preprocess),
            4: ("Extract Frames", self.stage_4_extract_frames),
            5: ("Generate Annotations", self.stage_5_annotate),
            6: ("Generate Railway Metadata", self.stage_6_metadata),
            7: ("Validate Dataset", self.stage_7_validate),
        }
        if stage_number not in stage_map:
            logger.error(f"Invalid stage: {stage_number}. Valid: 1-7")
            return False
        name, func = stage_map[stage_number]
        return self._run_stage(stage_number, name, func)


# ============================================================================
# CLI ENTRY POINT
# ============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Digital Shield — Dataset Management Pipeline Orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Stages:
  1  Download datasets
  2  Generate synthetic railway videos
  3  Preprocess all videos
  4  Extract frames
  5  Generate annotations
  6  Generate railway metadata
  7  Validate dataset

Examples:
  python pipeline_orchestrator.py                    # Run full pipeline
  python pipeline_orchestrator.py --stage 2          # Run stage 2 only
  python pipeline_orchestrator.py --skip-download    # Skip downloads
  python pipeline_orchestrator.py --synthetic 50     # 50 samples per behavior
        """,
    )
    parser.add_argument("--stage", type=int, help="Run specific stage only (1-7)")
    parser.add_argument("--skip-download", action="store_true", help="Skip dataset downloads")
    parser.add_argument("--skip-large", action="store_true", default=True, help="Skip large datasets (>5GB)")
    parser.add_argument("--synthetic", type=int, default=20, help="Synthetic samples per behavior")

    args = parser.parse_args()

    orchestrator = PipelineOrchestrator(
        skip_download=args.skip_download,
        skip_large=args.skip_large,
        synthetic_samples=args.synthetic,
    )

    if args.stage:
        orchestrator.run_stage_only(args.stage)
    else:
        orchestrator.run_full_pipeline()


if __name__ == "__main__":
    main()
