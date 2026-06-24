"""
Digital Shield Rail Defense — Synthetic Railway Video Generator
================================================================
Generates simulated railway platform surveillance scenarios
with synthetic overlays and annotations for training data.
"""

import cv2
import sys
import json
import random
import logging
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from ml.config import (
    SIMULATED_DIR, ANNOTATIONS_DIR, METADATA_DIR,
    ANOMALY_CLASSES, INDIAN_RAILWAY_STATIONS,
    LOG_FORMAT, LOG_DATE_FORMAT,
)

logger = logging.getLogger("synthetic_generator")
logger.setLevel(logging.INFO)
if not logger.handlers:
    console = logging.StreamHandler()
    console.setFormatter(logging.Formatter(LOG_FORMAT, LOG_DATE_FORMAT))
    logger.addHandler(console)


class SyntheticRailwayGenerator:
    """
    Generates synthetic railway platform surveillance videos
    with annotated anomaly scenarios.

    Creates simple simulated scenes with:
      - Moving person rectangles on platform background
      - Normal vs anomalous behavior patterns
      - Auto-generated bounding box annotations
      - Railway metadata per generated video
    """

    def __init__(self, output_dir: Optional[Path] = None, seed: int = 42):
        self.output_dir = output_dir or SIMULATED_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)
        self.generated_annotations: List[Dict] = []

    def _create_platform_background(self, width: int = 640, height: int = 480) -> np.ndarray:
        """Create a synthetic railway platform background."""
        bg = np.full((height, width, 3), (180, 180, 180), dtype=np.uint8)

        # Platform edge (yellow line)
        cv2.rectangle(bg, (0, height - 60), (width, height - 50), (0, 200, 255), -1)

        # Track area (dark gray)
        cv2.rectangle(bg, (0, height - 50), (width, height), (80, 80, 80), -1)

        # Rails
        cv2.line(bg, (0, height - 35), (width, height - 35), (120, 120, 120), 2)
        cv2.line(bg, (0, height - 20), (width, height - 20), (120, 120, 120), 2)

        # Platform pillars
        for x in range(80, width, 160):
            cv2.rectangle(bg, (x - 5, 50), (x + 5, height - 60), (150, 150, 150), -1)

        # Roof overhang
        cv2.rectangle(bg, (0, 0), (width, 50), (140, 140, 140), -1)

        return bg

    def _generate_person_trajectory(self, width: int, height: int,
                                     behavior: str, num_frames: int) -> List[Dict]:
        """
        Generate trajectory for a simulated person based on behavior type.
        
        Returns list of frame-level bounding box positions.
        """
        trajectory = []
        platform_y_range = (100, height - 70)

        if behavior == "normal":
            # Normal walking: smooth horizontal movement
            x = random.randint(20, 50)
            y = random.randint(*platform_y_range)
            speed_x = random.uniform(2, 5)
            person_w, person_h = random.randint(25, 40), random.randint(60, 90)

            for f in range(num_frames):
                x += speed_x + random.gauss(0, 0.5)
                y += random.gauss(0, 0.3)
                x = max(0, min(width - person_w, x))
                y = max(platform_y_range[0], min(platform_y_range[1] - person_h, y))
                trajectory.append({"x": int(x), "y": int(y), "w": person_w, "h": person_h})

        elif behavior == "suspicious_escort":
            # Two people moving together, one close behind
            x1 = random.randint(20, 50)
            y1 = random.randint(*platform_y_range)
            person_w, person_h = 30, 75

            for f in range(num_frames):
                x1 += random.uniform(1, 3)
                y1 += random.gauss(0, 0.5)
                x1 = max(0, min(width - person_w * 2, x1))
                y1 = max(platform_y_range[0], min(platform_y_range[1] - person_h, y1))
                trajectory.append({"x": int(x1), "y": int(y1), "w": person_w, "h": person_h,
                                  "x2": int(x1 + person_w + 5), "y2": int(y1 + 5), "w2": person_w - 5, "h2": person_h - 15})

        elif behavior == "dragging":
            # One person being pulled, erratic movement
            x = random.randint(50, 200)
            y = random.randint(*platform_y_range)
            person_w, person_h = 35, 70

            for f in range(num_frames):
                x += random.uniform(3, 7)
                y += random.gauss(0, 2)  # More vertical wobble
                x = max(0, min(width - person_w, x))
                y = max(platform_y_range[0], min(platform_y_range[1] - person_h, y))
                trajectory.append({"x": int(x), "y": int(y), "w": person_w, "h": person_h})

        elif behavior == "loitering":
            # Person moving in small area
            cx = random.randint(100, width - 100)
            cy = random.randint(*platform_y_range)
            person_w, person_h = 30, 75

            for f in range(num_frames):
                x = cx + random.gauss(0, 15)
                y = cy + random.gauss(0, 8)
                x = max(0, min(width - person_w, x))
                y = max(platform_y_range[0], min(platform_y_range[1] - person_h, y))
                trajectory.append({"x": int(x), "y": int(y), "w": person_w, "h": person_h})

        elif behavior == "panic":
            # Fast, erratic movement
            x = random.randint(width // 2 - 50, width // 2 + 50)
            y = random.randint(*platform_y_range)
            person_w, person_h = 30, 75

            for f in range(num_frames):
                x += random.gauss(0, 10)
                y += random.gauss(0, 5)
                x = max(0, min(width - person_w, x))
                y = max(platform_y_range[0], min(platform_y_range[1] - person_h, y))
                trajectory.append({"x": int(x), "y": int(y), "w": person_w, "h": person_h})

        else:  # Default normal
            return self._generate_person_trajectory(width, height, "normal", num_frames)

        return trajectory

    def generate_video(self, behavior: str = "normal", num_frames: int = 64,
                       width: int = 640, height: int = 480, fps: int = 2) -> Dict:
        """
        Generate a single synthetic surveillance video.

        Args:
            behavior: Behavior type to simulate
            num_frames: Number of frames
            width, height: Video dimensions
            fps: Output FPS

        Returns:
            Generation result with video path and annotations
        """
        video_id = f"sim_{behavior}_{random.randint(10000, 99999)}"
        video_dir = self.output_dir / behavior
        video_dir.mkdir(parents=True, exist_ok=True)
        video_path = video_dir / f"{video_id}.mp4"

        # Generate trajectory
        trajectory = self._generate_person_trajectory(width, height, behavior, num_frames)

        # Create video
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))
        bg = self._create_platform_background(width, height)

        frame_annotations = []
        for i, pos in enumerate(trajectory):
            frame = bg.copy()

            # Draw person(s)
            color = (0, 100, 200)  # Person color
            cv2.rectangle(frame, (pos["x"], pos["y"]),
                         (pos["x"] + pos["w"], pos["y"] + pos["h"]), color, -1)

            # Head circle
            cv2.circle(frame, (pos["x"] + pos["w"] // 2, pos["y"] - 5), 8, (200, 170, 140), -1)

            # Second person if present
            if "x2" in pos:
                cv2.rectangle(frame, (pos["x2"], pos["y2"]),
                             (pos["x2"] + pos["w2"], pos["y2"] + pos["h2"]), (0, 80, 160), -1)
                cv2.circle(frame, (pos["x2"] + pos["w2"] // 2, pos["y2"] - 5), 7, (200, 170, 140), -1)

            # Add noise for realism
            noise = np.random.randint(0, 10, frame.shape, dtype=np.uint8)
            frame = cv2.add(frame, noise)

            # Timestamp overlay
            ts = f"CAM-SIM | {datetime.now().strftime('%H:%M:%S')} | F{i:04d}"
            cv2.putText(frame, ts, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

            writer.write(frame)

            # Store bounding box annotation (YOLO format: normalized)
            x_center = (pos["x"] + pos["w"] / 2) / width
            y_center = (pos["y"] + pos["h"] / 2) / height
            w_norm = pos["w"] / width
            h_norm = pos["h"] / height
            frame_annotations.append({
                "frame_idx": i,
                "boxes": [{"class_id": 0, "x_center": round(x_center, 6),
                           "y_center": round(y_center, 6), "width": round(w_norm, 6),
                           "height": round(h_norm, 6)}]
            })

        writer.release()

        # Map behavior to anomaly class
        behavior_to_class = {
            "normal": 0, "suspicious_escort": 4, "dragging": 3,
            "loitering": 9, "panic": 6, "assault": 1, "coercion": 2,
        }
        class_id = behavior_to_class.get(behavior, 0)

        annotation = {
            "video_id": video_id,
            "video_path": str(video_path),
            "behavior": behavior,
            "class_id": class_id,
            "class_name": ANOMALY_CLASSES.get(class_id, "unknown"),
            "is_anomalous": class_id != 0,
            "num_frames": num_frames,
            "fps": fps,
            "resolution": f"{width}x{height}",
            "frame_annotations": frame_annotations,
        }
        self.generated_annotations.append(annotation)

        logger.info(f"Generated: {video_path.name} ({behavior}, {num_frames} frames)")
        return annotation

    def generate_dataset(self, samples_per_behavior: int = 20) -> List[Dict]:
        """
        Generate a complete synthetic dataset with all behavior types.
        """
        behaviors = ["normal", "suspicious_escort", "dragging", "loitering", "panic"]
        # More normal samples to match real-world distribution
        behavior_counts = {
            "normal": samples_per_behavior * 3,
            "suspicious_escort": samples_per_behavior,
            "dragging": samples_per_behavior,
            "loitering": samples_per_behavior,
            "panic": samples_per_behavior,
        }

        logger.info(f"\nGenerating synthetic railway surveillance dataset...")
        logger.info(f"Behaviors: {list(behavior_counts.keys())}")
        logger.info(f"Total videos: {sum(behavior_counts.values())}")

        all_annotations = []
        for behavior, count in behavior_counts.items():
            logger.info(f"\n  Generating {count} '{behavior}' videos...")
            for i in range(count):
                num_frames = random.randint(32, 96)
                ann = self.generate_video(behavior=behavior, num_frames=num_frames)
                all_annotations.append(ann)

        # Save annotations
        output_path = ANNOTATIONS_DIR / "synthetic_annotations.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump({
                "version": "1.0",
                "created_at": datetime.now().isoformat(),
                "total": len(all_annotations),
                "annotations": all_annotations,
            }, f, indent=2)

        logger.info(f"\nSynthetic dataset complete: {len(all_annotations)} videos")
        logger.info(f"Annotations saved: {output_path}")
        return all_annotations


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Synthetic Railway Video Generator")
    parser.add_argument("--samples", type=int, default=20, help="Samples per behavior type")
    parser.add_argument("--behavior", type=str, help="Generate specific behavior only")
    parser.add_argument("--count", type=int, default=1, help="Number of videos (single behavior)")
    args = parser.parse_args()

    generator = SyntheticRailwayGenerator()
    if args.behavior:
        for _ in range(args.count):
            generator.generate_video(behavior=args.behavior)
    else:
        generator.generate_dataset(samples_per_behavior=args.samples)
