"""
Digital Shield Rail Defense — Video Preprocessor
==================================================
Normalizes raw surveillance footage for ML training.
Features: resolution normalization, FPS standardization,
CLAHE brightness correction, clip segmentation.
"""

import cv2
import sys
import json
import logging
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict, Tuple
from datetime import datetime
from dataclasses import asdict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from ml.config import (
    RAW_DIR, PROCESSED_DIR, CLIPS_DIR, METADATA_DIR,
    PreprocessConfig, DEFAULT_PREPROCESS, LOG_FORMAT, LOG_DATE_FORMAT,
)

logger = logging.getLogger("video_preprocessor")
logger.setLevel(logging.INFO)
if not logger.handlers:
    console = logging.StreamHandler()
    console.setFormatter(logging.Formatter(LOG_FORMAT, LOG_DATE_FORMAT))
    logger.addHandler(console)


class VideoPreprocessor:
    """Preprocesses raw surveillance videos into standardized clips."""

    def __init__(self, config: Optional[PreprocessConfig] = None):
        self.config = config or DEFAULT_PREPROCESS
        self.stats = {"total_videos": 0, "processed": 0, "skipped": 0, "failed": 0, "total_clips": 0, "total_frames": 0}
        self._clahe = None
        if self.config.apply_clahe:
            self._clahe = cv2.createCLAHE(clipLimit=self.config.clahe_clip_limit, tileGridSize=self.config.clahe_tile_grid)

    def _validate_video(self, video_path: Path) -> Tuple[bool, dict]:
        """Validate a video file and extract metadata."""
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return False, {"error": "Cannot open video"}
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = frame_count / fps if fps > 0 else 0
        cap.release()
        metadata = {"original_fps": fps, "frame_count": frame_count, "width": width, "height": height, "duration_seconds": round(duration, 2)}
        if frame_count < self.config.min_frames_per_video:
            return False, {**metadata, "error": f"Too few frames: {frame_count}"}
        if fps <= 0:
            return False, {**metadata, "error": "Invalid FPS"}
        return True, metadata

    def _normalize_frame(self, frame: np.ndarray) -> np.ndarray:
        """Apply resolution and brightness normalization."""
        frame = cv2.resize(frame, (self.config.frame_width, self.config.frame_height), interpolation=cv2.INTER_LINEAR)
        if self._clahe is not None:
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            l_channel, a, b = cv2.split(lab)
            l_channel = self._clahe.apply(l_channel)
            frame = cv2.cvtColor(cv2.merge([l_channel, a, b]), cv2.COLOR_LAB2BGR)
        return frame

    def process_video(self, video_path: Path, output_dir: Optional[Path] = None) -> Dict:
        """Process a single video: validate, normalize, segment into clips."""
        self.stats["total_videos"] += 1
        result = {"source": str(video_path), "status": "pending", "clips": [], "metadata": {}}
        is_valid, metadata = self._validate_video(video_path)
        result["metadata"] = metadata
        if not is_valid:
            self.stats["failed"] += 1
            result["status"] = "invalid"
            return result

        out_dir = output_dir or CLIPS_DIR
        video_out_dir = out_dir / video_path.stem
        video_out_dir.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(str(video_path))
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = max(1, int(original_fps / self.config.target_fps))
        frames, frame_idx, processed_count = [], 0, 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % frame_interval == 0:
                frames.append(self._normalize_frame(frame))
                processed_count += 1
                if processed_count >= self.config.max_frames_per_video:
                    break
            frame_idx += 1
        cap.release()

        if len(frames) < self.config.min_frames_per_video:
            self.stats["failed"] += 1
            result["status"] = "too_few_frames"
            return result

        # Generate clips
        clip_frames = self.config.clip_duration_seconds * self.config.target_fps
        stride_frames = self.config.clip_stride_seconds * self.config.target_fps
        clip_idx = 0
        for start in range(0, len(frames) - clip_frames + 1, stride_frames):
            clip = frames[start:start + clip_frames]
            clip_path = video_out_dir / f"clip_{clip_idx:04d}.mp4"
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(clip_path), fourcc, self.config.target_fps, (self.config.frame_width, self.config.frame_height))
            for f in clip:
                writer.write(f)
            writer.release()
            result["clips"].append({"path": str(clip_path), "start_frame": start, "end_frame": start + clip_frames, "num_frames": len(clip)})
            clip_idx += 1
            self.stats["total_clips"] += 1

        self.stats["processed"] += 1
        self.stats["total_frames"] += len(frames)
        result["status"] = "complete"
        result["metadata"]["processed_frames"] = len(frames)
        result["metadata"]["clips_generated"] = len(result["clips"])
        logger.info(f"Processed: {video_path.name} → {len(frames)} frames, {len(result['clips'])} clips")
        return result

    def process_dataset(self, dataset_dir: Path, output_dir: Optional[Path] = None) -> List[Dict]:
        """Process all videos in a dataset directory."""
        video_files = []
        for ext in (".mp4", ".avi", ".mkv", ".mov", ".wmv"):
            video_files.extend(dataset_dir.rglob(f"*{ext}"))
        logger.info(f"Found {len(video_files)} videos in {dataset_dir}")
        results = []
        for i, vp in enumerate(sorted(video_files)):
            logger.info(f"[{i+1}/{len(video_files)}] {vp.name}")
            results.append(self.process_video(vp, output_dir))
        return results

    def process_all_datasets(self) -> Dict[str, List[Dict]]:
        """Process all raw datasets."""
        all_results = {}
        for dataset_dir in sorted(d for d in RAW_DIR.iterdir() if d.is_dir()):
            logger.info(f"\n{'='*60}\nProcessing: {dataset_dir.name}\n{'='*60}")
            all_results[dataset_dir.name] = self.process_dataset(dataset_dir)
        report_path = METADATA_DIR / "preprocessing_report.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w") as f:
            json.dump({"stats": self.stats, "config": asdict(self.config), "timestamp": datetime.now().isoformat()}, f, indent=2)
        logger.info(f"\nPreprocessing complete. Report: {report_path}")
        return all_results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Video Preprocessor")
    parser.add_argument("--input", type=str, help="Input video or directory")
    parser.add_argument("--output", type=str, help="Output directory")
    parser.add_argument("--all", action="store_true", help="Process all raw datasets")
    args = parser.parse_args()
    processor = VideoPreprocessor()
    if args.all:
        processor.process_all_datasets()
    elif args.input:
        p = Path(args.input)
        out = Path(args.output) if args.output else None
        if p.is_dir():
            processor.process_dataset(p, out)
        else:
            processor.process_video(p, out)
