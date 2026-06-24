"""
Digital Shield Rail Defense — Metadata Generator
==================================================
Generates synthetic railway metadata for every training video/frame,
mapping anomaly detection data to realistic Indian railway context.
"""

import sys
import json
import random
import logging
import hashlib
from pathlib import Path
from typing import Optional, Dict, List
from datetime import datetime, timedelta

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from ml.config import (
    RAW_DIR, METADATA_DIR, ANNOTATIONS_DIR,
    INDIAN_RAILWAY_STATIONS, COACH_DESIGNATIONS, TRAIN_TYPES,
    ANOMALY_CLASSES, LOG_FORMAT, LOG_DATE_FORMAT,
)

logger = logging.getLogger("metadata_generator")
logger.setLevel(logging.INFO)
if not logger.handlers:
    console = logging.StreamHandler()
    console.setFormatter(logging.Formatter(LOG_FORMAT, LOG_DATE_FORMAT))
    logger.addHandler(console)


# ============================================================================
# SYNTHETIC TRAIN DATABASE
# ============================================================================

def _generate_train_database(num_trains: int = 200, seed: int = 42) -> List[Dict]:
    """Generate a pool of synthetic Indian railway trains."""
    random.seed(seed)
    trains = []
    prefixes = ["Rajdhani", "Shatabdi", "Duronto", "Garib Rath", "Superfast", "Express", "Jan Shatabdi", "Mail"]
    city_names = [s["city"] for s in INDIAN_RAILWAY_STATIONS]

    for i in range(num_trains):
        train_number = random.randint(10001, 99999)
        origin = random.choice(city_names)
        dest = random.choice([c for c in city_names if c != origin])
        train_type = random.choice(TRAIN_TYPES)
        prefix = random.choice(prefixes)
        train_name = f"{origin}-{dest} {prefix}"

        num_coaches = random.randint(12, 24)
        coaches = random.sample(COACH_DESIGNATIONS, min(num_coaches, len(COACH_DESIGNATIONS)))

        num_stops = random.randint(3, 12)
        route_stations = random.sample(
            [s["code"] for s in INDIAN_RAILWAY_STATIONS],
            min(num_stops, len(INDIAN_RAILWAY_STATIONS))
        )

        trains.append({
            "train_number": str(train_number),
            "train_name": train_name,
            "train_type": train_type,
            "origin": origin,
            "destination": dest,
            "coaches": coaches,
            "total_coaches": num_coaches,
            "route_stations": route_stations,
        })

    return trains


TRAIN_DATABASE = _generate_train_database()


class MetadataGenerator:
    """
    Generates synthetic railway metadata for training videos,
    mapping each video to a realistic railway station/platform/train context.
    """

    def __init__(self, output_dir: Optional[Path] = None, seed: int = 42):
        self.output_dir = output_dir or METADATA_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.seed = seed
        random.seed(seed)
        self.metadata_records: List[Dict] = []

    def _generate_camera_id(self, station: Dict, platform: int) -> str:
        """Generate a realistic camera ID."""
        zone_codes = {"entry": "A", "mid": "B", "exit": "C"}
        zone = random.choice(list(zone_codes.keys()))
        return f"CAM_{station['code']}_P{platform:02d}_{zone_codes[zone]}"

    def _generate_timestamp(self, base_date: Optional[datetime] = None) -> datetime:
        """Generate a realistic railway platform timestamp."""
        base = base_date or datetime(2026, 5, 1)
        # Railway platforms are active roughly 4 AM to midnight
        hour = random.choices(
            range(24),
            weights=[0,0,0,0,2,4,6,8,10,10,8,8,8,8,8,8,10,10,10,8,6,4,2,1],
            k=1
        )[0]
        minute = random.randint(0, 59)
        day_offset = random.randint(0, 30)
        return base + timedelta(days=day_offset, hours=hour, minutes=minute)

    def generate_for_video(self, video_path: Path, annotation: Optional[Dict] = None) -> Dict:
        """
        Generate synthetic railway metadata for a single video.
        
        Returns a metadata record containing station, platform, train,
        coach, camera, and timestamp information.
        """
        station = random.choice(INDIAN_RAILWAY_STATIONS)
        platform = random.randint(1, station["platforms"])
        train = random.choice(TRAIN_DATABASE)
        coach = random.choice(train["coaches"])
        camera_id = self._generate_camera_id(station, platform)
        timestamp = self._generate_timestamp()

        metadata = {
            "video_id": hashlib.md5(str(video_path).encode()).hexdigest()[:12],
            "video_file": str(video_path),
            "video_name": video_path.name,
            # Station context
            "station": {
                "code": station["code"],
                "name": station["name"],
                "city": station["city"],
                "state": station["state"],
                "zone": station["zone"],
                "latitude": station["lat"],
                "longitude": station["lng"],
            },
            "platform_number": platform,
            # Camera context
            "camera_id": camera_id,
            "camera_zone": camera_id.split("_")[-1],
            # Train context
            "train": {
                "number": train["train_number"],
                "name": train["train_name"],
                "type": train["train_type"],
                "coaches": train["coaches"],
            },
            "estimated_coach": coach,
            # Temporal context
            "timestamp": timestamp.isoformat(),
            "date": timestamp.strftime("%Y-%m-%d"),
            "time": timestamp.strftime("%H:%M:%S"),
            "day_of_week": timestamp.strftime("%A"),
            "is_night": timestamp.hour < 6 or timestamp.hour >= 22,
            "is_peak_hour": timestamp.hour in (7, 8, 9, 17, 18, 19),
            # Anomaly context (if annotation provided)
            "anomaly_type": annotation.get("class_name", "unknown") if annotation else "unknown",
            "anomaly_class_id": annotation.get("class_id", -1) if annotation else -1,
            "is_anomalous": annotation.get("is_anomalous", False) if annotation else False,
        }

        self.metadata_records.append(metadata)
        return metadata

    def generate_for_dataset(self, dataset_dir: Path, annotations: Optional[List[Dict]] = None) -> List[Dict]:
        """Generate metadata for all videos in a dataset."""
        video_files = []
        for ext in (".mp4", ".avi", ".mkv", ".mov", ".wmv"):
            video_files.extend(dataset_dir.rglob(f"*{ext}"))

        # Build annotation lookup
        ann_lookup = {}
        if annotations:
            for ann in annotations:
                ann_lookup[ann.get("video_path", "")] = ann

        logger.info(f"Generating metadata for {len(video_files)} videos in {dataset_dir.name}")
        records = []
        for vp in sorted(video_files):
            ann = ann_lookup.get(str(vp))
            records.append(self.generate_for_video(vp, ann))

        return records

    def generate_all(self, annotations_file: Optional[Path] = None) -> List[Dict]:
        """Generate metadata for all datasets."""
        # Load annotations if available
        annotations = None
        if annotations_file and annotations_file.exists():
            with open(annotations_file) as f:
                data = json.load(f)
                annotations = data.get("annotations", [])

        for dataset_dir in sorted(d for d in RAW_DIR.iterdir() if d.is_dir()):
            logger.info(f"\n{'='*60}\nGenerating metadata: {dataset_dir.name}\n{'='*60}")
            dataset_anns = None
            if annotations:
                dataset_anns = [a for a in annotations if a.get("dataset") == dataset_dir.name]
            self.generate_for_dataset(dataset_dir, dataset_anns)

        self.save()
        return self.metadata_records

    def save(self, filename: str = "video_metadata.json") -> Path:
        """Save all metadata to JSON."""
        output_path = self.output_dir / filename
        data = {
            "version": "1.0",
            "created_at": datetime.now().isoformat(),
            "total_records": len(self.metadata_records),
            "stations_used": len(set(r["station"]["code"] for r in self.metadata_records)) if self.metadata_records else 0,
            "trains_used": len(set(r["train"]["number"] for r in self.metadata_records)) if self.metadata_records else 0,
            "records": self.metadata_records,
        }
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info(f"Saved {len(self.metadata_records)} metadata records to {output_path}")
        return output_path

    def generate_station_camera_registry(self) -> Path:
        """Generate a complete camera-to-station registry."""
        registry = []
        for station in INDIAN_RAILWAY_STATIONS:
            for platform in range(1, station["platforms"] + 1):
                for zone in ["A", "B", "C"]:
                    camera_id = f"CAM_{station['code']}_P{platform:02d}_{zone}"
                    registry.append({
                        "camera_id": camera_id,
                        "station_code": station["code"],
                        "station_name": station["name"],
                        "platform_number": platform,
                        "zone": {"A": "entry", "B": "mid", "C": "exit"}[zone],
                        "is_active": random.random() > 0.05,
                    })

        output_path = self.output_dir / "camera_registry.json"
        with open(output_path, "w") as f:
            json.dump({"cameras": registry, "total": len(registry)}, f, indent=2)
        logger.info(f"Generated camera registry: {len(registry)} cameras")
        return output_path

    def generate_train_schedule_db(self) -> Path:
        """Generate synthetic train schedule database."""
        schedules = []
        for train in TRAIN_DATABASE:
            base_departure = datetime(2026, 5, 1, random.randint(0, 23), random.choice([0, 15, 30, 45]))
            for i, station_code in enumerate(train["route_stations"]):
                arrival = base_departure + timedelta(hours=i * random.uniform(0.5, 3))
                departure = arrival + timedelta(minutes=random.randint(2, 15))
                platform = random.randint(1, 10)
                schedules.append({
                    "train_number": train["train_number"],
                    "train_name": train["train_name"],
                    "station_code": station_code,
                    "stop_sequence": i + 1,
                    "arrival_time": arrival.strftime("%H:%M"),
                    "departure_time": departure.strftime("%H:%M"),
                    "platform_number": platform,
                    "halt_minutes": (departure - arrival).seconds // 60,
                })

        output_path = self.output_dir / "train_schedules.json"
        with open(output_path, "w") as f:
            json.dump({"schedules": schedules, "total_trains": len(TRAIN_DATABASE)}, f, indent=2)
        logger.info(f"Generated {len(schedules)} schedule entries for {len(TRAIN_DATABASE)} trains")
        return output_path


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Metadata Generator")
    parser.add_argument("--all", action="store_true", help="Generate for all datasets")
    parser.add_argument("--registry", action="store_true", help="Generate camera registry")
    parser.add_argument("--schedules", action="store_true", help="Generate train schedules")
    parser.add_argument("--annotations", type=str, help="Path to annotations.json")
    args = parser.parse_args()

    generator = MetadataGenerator()
    if args.registry:
        generator.generate_station_camera_registry()
    if args.schedules:
        generator.generate_train_schedule_db()
    if args.all:
        ann_path = Path(args.annotations) if args.annotations else ANNOTATIONS_DIR / "annotations.json"
        generator.generate_all(ann_path)
    if not any([args.all, args.registry, args.schedules]):
        generator.generate_station_camera_registry()
        generator.generate_train_schedule_db()
        generator.generate_all()
