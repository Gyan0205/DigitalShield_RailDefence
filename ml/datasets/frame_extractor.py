"""
Digital Shield Rail Defense — Frame Extractor
===============================================
Extracts individual frames from videos at configurable FPS,
applies normalization, and organizes output by dataset/category.
"""

import cv2
import sys
import json
import logging
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from ml.config import (
    RAW_DIR, FRAMES_DIR, METADATA_DIR,
    DEFAULT_PREPROCESS, LOG_FORMAT, LOG_DATE_FORMAT,
)

logger = logging.getLogger("frame_extractor")
logger.setLevel(logging.INFO)
if not logger.handlers:
    console = logging.StreamHandler()
    console.setFormatter(logging.Formatter(LOG_FORMAT, LOG_DATE_FORMAT))
    logger.addHandler(console)


class FrameExtractor:
    """Extracts and saves individual frames from video files."""

    def __init__(self, target_fps: int = 2, frame_size: Tuple[int, int] = (640, 480),
                 output_format: str = "jpg", quality: int = 95):
        self.target_fps = target_fps
        self.frame_size = frame_size
        self.output_format = output_format
        self.quality = quality
        self.stats = {"videos_processed": 0, "frames_extracted": 0, "errors": 0}

    def extract_from_video(self, video_path: Path, output_dir: Optional[Path] = None,
                           max_frames: int = 300) -> Dict:
        """
        Extract frames from a single video.
        
        Returns:
            Dict with extraction results and frame paths.
        """
        result = {"source": str(video_path), "frames": [], "status": "pending"}
        out_dir = output_dir or (FRAMES_DIR / video_path.stem)
        out_dir.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            result["status"] = "error"
            result["error"] = "Cannot open video"
            self.stats["errors"] += 1
            return result

        original_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_interval = max(1, int(original_fps / self.target_fps))

        frame_idx = 0
        extracted = 0

        while cap.isOpened() and extracted < max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_interval == 0:
                # Resize
                frame = cv2.resize(frame, self.frame_size, interpolation=cv2.INTER_LINEAR)

                # Save frame
                frame_filename = f"frame_{extracted:06d}.{self.output_format}"
                frame_path = out_dir / frame_filename

                if self.output_format == "jpg":
                    cv2.imwrite(str(frame_path), frame, [cv2.IMWRITE_JPEG_QUALITY, self.quality])
                else:
                    cv2.imwrite(str(frame_path), frame)

                result["frames"].append({
                    "path": str(frame_path),
                    "original_frame_idx": frame_idx,
                    "timestamp_sec": round(frame_idx / original_fps, 3) if original_fps > 0 else 0,
                })
                extracted += 1

            frame_idx += 1

        cap.release()
        self.stats["videos_processed"] += 1
        self.stats["frames_extracted"] += extracted
        result["status"] = "complete"
        result["metadata"] = {
            "original_fps": original_fps,
            "total_original_frames": total_frames,
            "extracted_frames": extracted,
            "frame_interval": frame_interval,
            "output_dir": str(out_dir),
        }
        logger.info(f"Extracted {extracted} frames from {video_path.name}")
        return result

    def extract_from_directory(self, input_dir: Path, output_dir: Optional[Path] = None) -> List[Dict]:
        """Extract frames from all videos in a directory."""
        video_files = []
        for ext in (".mp4", ".avi", ".mkv", ".mov", ".wmv"):
            video_files.extend(input_dir.rglob(f"*{ext}"))

        logger.info(f"Found {len(video_files)} videos in {input_dir}")
        results = []

        for i, vp in enumerate(sorted(video_files)):
            logger.info(f"[{i+1}/{len(video_files)}] Extracting: {vp.name}")
            # Preserve directory structure
            relative = vp.relative_to(input_dir)
            frame_out = (output_dir or FRAMES_DIR) / relative.parent / vp.stem
            results.append(self.extract_from_video(vp, frame_out))

        return results

    def extract_all_datasets(self) -> Dict[str, List[Dict]]:
        """Extract frames from all raw datasets."""
        all_results = {}
        for dataset_dir in sorted(d for d in RAW_DIR.iterdir() if d.is_dir()):
            logger.info(f"\n{'='*60}\nExtracting frames: {dataset_dir.name}\n{'='*60}")
            out = FRAMES_DIR / dataset_dir.name
            all_results[dataset_dir.name] = self.extract_from_directory(dataset_dir, out)

        # Save extraction report
        report = {
            "stats": self.stats,
            "timestamp": datetime.now().isoformat(),
            "datasets": {k: len(v) for k, v in all_results.items()},
        }
        report_path = METADATA_DIR / "frame_extraction_report.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)

        logger.info(f"\nExtraction complete: {self.stats['frames_extracted']} total frames")
        return all_results

    def extract_keyframes(self, video_path: Path, output_dir: Optional[Path] = None,
                          threshold: float = 30.0) -> Dict:
        """
        Extract keyframes based on scene change detection.
        Uses frame differencing to detect significant visual changes.
        """
        out_dir = output_dir or (FRAMES_DIR / f"{video_path.stem}_keyframes")
        out_dir.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return {"status": "error", "error": "Cannot open video"}

        prev_gray = None
        keyframes = []
        frame_idx = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(cv2.resize(frame, self.frame_size), cv2.COLOR_BGR2GRAY)

            if prev_gray is not None:
                diff = cv2.absdiff(gray, prev_gray)
                mean_diff = np.mean(diff)

                if mean_diff > threshold:
                    frame_path = out_dir / f"keyframe_{len(keyframes):04d}.{self.output_format}"
                    resized = cv2.resize(frame, self.frame_size)
                    cv2.imwrite(str(frame_path), resized)
                    keyframes.append({
                        "path": str(frame_path),
                        "frame_idx": frame_idx,
                        "scene_change_score": round(float(mean_diff), 2),
                    })

            prev_gray = gray
            frame_idx += 1

        cap.release()
        logger.info(f"Extracted {len(keyframes)} keyframes from {video_path.name}")
        return {"status": "complete", "keyframes": keyframes, "total_frames": frame_idx}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Frame Extractor")
    parser.add_argument("--input", type=str, help="Input video or directory")
    parser.add_argument("--output", type=str, help="Output directory")
    parser.add_argument("--fps", type=int, default=2, help="Target FPS")
    parser.add_argument("--keyframes", action="store_true", help="Extract keyframes only")
    parser.add_argument("--all", action="store_true", help="Extract from all datasets")
    args = parser.parse_args()

    extractor = FrameExtractor(target_fps=args.fps)
    if args.all:
        extractor.extract_all_datasets()
    elif args.input:
        p = Path(args.input)
        out = Path(args.output) if args.output else None
        if args.keyframes:
            extractor.extract_keyframes(p, out)
        elif p.is_dir():
            extractor.extract_from_directory(p, out)
        else:
            extractor.extract_from_video(p, out)
