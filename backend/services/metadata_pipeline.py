"""
Digital Shield Rail Defense — Metadata Pipeline
==================================================
Production-grade metadata pipeline that generates, validates,
and exports synthetic railway metadata for every video in the
training dataset.

Pipeline stages:
  1. Initialize railway simulator (trains, schedules, cameras)
  2. Scan all dataset videos
  3. Assign realistic metadata per video
  4. Export to JSON, CSV, and per-video sidecar files
  5. Generate statistical summaries
"""

import json
import random
import hashlib
import logging
from pathlib import Path
from typing import Optional, Dict, List
from datetime import datetime, timedelta, date, time as dt_time
from collections import Counter, defaultdict

from backend.services.railway_schema import VideoMetadata, CameraSchema
from backend.services.railway_simulator import RailwaySimulator
from backend.services.railway_stations import STATIONS_DB, STATION_LOOKUP

logger = logging.getLogger("metadata_pipeline")
logger.setLevel(logging.INFO)
if not logger.handlers:
    console = logging.StreamHandler()
    console.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)-25s | %(message)s",
                          "%Y-%m-%d %H:%M:%S")
    )
    logger.addHandler(console)


# ============================================================================
# ANOMALY CLASS MAPPING
# ============================================================================

ANOMALY_CLASSES = {
    0: "normal", 1: "assault", 2: "coercion", 3: "dragging",
    4: "suspicious_escort", 5: "isolated_minor", 6: "panic_behavior",
    7: "theft", 8: "vandalism", 9: "loitering", 10: "fighting",
}

# Indian holidays for realistic temporal simulation
INDIAN_HOLIDAYS_2026 = [
    "2026-01-26",  # Republic Day
    "2026-03-17",  # Holi
    "2026-04-14",  # Ambedkar Jayanti
    "2026-08-15",  # Independence Day
    "2026-10-02",  # Gandhi Jayanti
    "2026-10-20",  # Dussehra
    "2026-11-08",  # Diwali
    "2026-12-25",  # Christmas
]

class MetadataPipeline:
    """
    Generates permanent recording context for videos using local SC data.
    """

    def __init__(self, output_dir: Optional[Path] = None, seed: int = 42):
        self.output_dir = output_dir or Path("dataset/metadata")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.seed = seed
        random.seed(seed)
        self.records: List[VideoMetadata] = []
        self.context_pool: List[Dict] = []
        
        # Load local context immediately
        self._load_local_context()

    def _load_local_context(self):
        """Load SC context from local JSON (extracted from DB)."""
        context_path = Path("backend/data/sc_context.json")
        if context_path.exists():
            with open(context_path) as f:
                self.context_pool = json.load(f)
            logger.info(f"Loaded {len(self.context_pool)} SC context patterns from local storage.")
        else:
            logger.warning("Local context file not found. Metadata will be fallback-based.")

    def generate_for_video(self, video_path: Path, dataset_name: str = "") -> VideoMetadata:
        """Assign permanent recording context to a single video."""
        
        if self.context_pool:
            ctx = random.choice(self.context_pool)
            platform = ctx.get("platform") or 1
            time_str = ctx.get("time") or "12:00"
                
            dep_days_str = ctx.get("days", "Sun")
            valid_days = [d.strip() for d in dep_days_str.split(",") if d.strip()]
            chosen_day_short = random.choice(valid_days) if valid_days else "Sun"
            
            day_map = {"Mon": 0, "Tue": 1, "Wed": 2, "Thu": 3, "Fri": 4, "Sat": 5, "Sun": 6}
            target_weekday = day_map.get(chosen_day_short, 6)
            
            range_start = date(2026, 1, 1)
            range_end   = date(2026, 3, 27)
            
            matching_dates = []
            curr = range_start
            while curr <= range_end:
                if curr.weekday() == target_weekday:
                    matching_dates.append(curr)
                curr += timedelta(days=1)
            
            final_date = random.choice(matching_dates) if matching_dates else range_start
            date_str = final_date.strftime("%Y-%m-%d")
            full_day_name = final_date.strftime("%A")
        else:
            platform, date_str, time_str, full_day_name = 1, "2026-01-01", "12:00", "Thursday"

        video_id = hashlib.md5(f"{video_path}_{self.seed}".encode()).hexdigest()[:12]

        metadata = VideoMetadata(
            video_id=video_id,
            video_file=str(video_path),
            video_name=video_path.name,
            dataset=dataset_name,
            platform_number=platform,
            date=date_str,
            time=time_str,
            day=full_day_name
        )

        self.records.append(metadata)
        
        # PERMANENT ASSIGNMENT: Save sidecar JSON next to the metadata records
        self._save_sidecar(metadata)
        
        return metadata

    def _save_sidecar(self, metadata: VideoMetadata):
        """Save a permanent metadata sidecar file."""
        sidecar_name = f"{Path(metadata.video_name).stem}.json"
        sidecar_path = self.output_dir / "sidecars" / sidecar_name
        sidecar_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(sidecar_path, "w") as f:
            json.dump(metadata.to_dict(), f, indent=2)

    def run(self, video_root: Path) -> Dict:
        """Scan videos and attach permanent context."""
        video_extensions = {".mp4", ".avi", ".mkv", ".mov", ".wmv"}
        all_videos = []
        for ext in video_extensions:
            all_videos.extend(video_root.rglob(f"*{ext}"))
        
        logger.info(f"Found {len(all_videos)} videos. Assigning PERMANENT context...")
        
        for i, video_path in enumerate(sorted(all_videos)):
            try:
                rel = video_path.relative_to(video_root)
                dataset_name = rel.parts[0] if len(rel.parts) > 1 else "unknown"
            except ValueError:
                dataset_name = "unknown"

            self.generate_for_video(video_path, dataset_name)

        self._export_json()
        
        return {
            "videos_processed": len(self.records),
            "sidecars_generated": len(self.records),
            "output_dir": str(self.output_dir)
        }

    def _export_json(self):
        """Export context records to master JSON."""
        output = {
            "version": "3.0",
            "station": "Secunderabad Junction (SC)",
            "total_records": len(self.records),
            "records": [r.to_dict() for r in self.records],
        }
        path = self.output_dir / "video_metadata.json"
        with open(path, "w") as f:
            json.dump(output, f, indent=2)
        logger.info(f"Saved metadata context to {path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="SC CCTV Metadata Pipeline")
    parser.add_argument("--video-root", type=str, default="dataset/raw", help="Video root")
    parser.add_argument("--output", type=str, default="dataset/metadata", help="Output dir")
    args = parser.parse_args()

    pipeline = MetadataPipeline(output_dir=Path(args.output))
    pipeline.run(video_root=Path(args.video_root))
