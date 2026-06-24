"""
Digital Shield Rail Defense — Railway Metadata Schema
=======================================================
Pydantic v2 schemas defining the metadata data model for
synthetic railway intelligence. Every field maps to a real
Indian Railways operational concept.
"""

from datetime import datetime, time
from typing import Optional, List, Dict
from dataclasses import dataclass, field


# ============================================================================
# STATION SCHEMA
# ============================================================================

@dataclass
class StationSchema:
    """Indian railway station operational data."""
    code: str                    # e.g., "NDLS"
    name: str                    # e.g., "New Delhi"
    city: str
    state: str
    zone: str                    # e.g., "Northern Railway"
    division: str = ""           # e.g., "Delhi Division"
    latitude: float = 0.0
    longitude: float = 0.0
    platforms: int = 1
    is_junction: bool = False
    is_terminus: bool = False
    daily_footfall: int = 0      # Estimated daily passengers
    risk_tier: str = "STANDARD"  # STANDARD, ELEVATED, HIGH — trafficking risk


# ============================================================================
# TRAIN SCHEMA
# ============================================================================

@dataclass
class TrainSchema:
    """Indian railway train operational data."""
    number: str                  # e.g., "12727"
    name: str                    # e.g., "Godavari Express"
    train_type: str              # Rajdhani, Shatabdi, Express, etc.
    origin_code: str
    origin_name: str
    destination_code: str
    destination_name: str
    coaches: List[str] = field(default_factory=list)
    total_coaches: int = 0
    runs_on: List[str] = field(default_factory=lambda: ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"])
    avg_speed_kmph: int = 55
    total_distance_km: int = 0
    pantry_car: bool = False


# ============================================================================
# SCHEDULE SCHEMA
# ============================================================================

@dataclass
class ScheduleEntry:
    """Single stop in a train's schedule."""
    train_number: str
    train_name: str
    station_code: str
    station_name: str
    stop_sequence: int           # 1-indexed stop number
    arrival_time: str            # "HH:MM" (24h)
    departure_time: str
    platform_number: int
    halt_minutes: int
    distance_from_origin_km: int = 0
    day: int = 1                 # Day of journey (1, 2, 3...)


# ============================================================================
# CAMERA SCHEMA
# ============================================================================

@dataclass
class CameraSchema:
    """CCTV camera installation record."""
    camera_id: str               # e.g., "CAM_NDLS_P05_B"
    station_code: str
    station_name: str
    platform_number: int
    zone: str                    # "entry", "mid", "exit"
    camera_type: str = "PTZ"     # PTZ, Fixed, Dome
    resolution: str = "1080p"
    feed_url: str = ""
    is_active: bool = True
    coverage_area_sqm: float = 200.0
    # Coach visibility: which coaches are visible from this camera
    visible_coaches_start: int = 0  # Coach position from engine
    visible_coaches_end: int = 4


# ============================================================================
# VIDEO METADATA SCHEMA (Environmental Context Only)
# ============================================================================

@dataclass
class VideoMetadata:
    """Surveillance video metadata record defining the recording context."""
    # Video identity
    video_id: str
    video_file: str
    video_name: str
    dataset: str

    # Station context (SC only)
    station_code: str = "SC"
    station_name: str = "Secunderabad Junction"

    # Platform (Where)
    platform_number: int = 1

    # Temporal context (When)
    date: str = ""          # YYYY-MM-DD
    time: str = ""          # HH:MM
    day: str = ""           # Day of week (Monday, Tuesday, etc.)

    def to_dict(self) -> Dict:
        """Serialize to JSON-compatible dict."""
        return {
            "video_id": self.video_id,
            "video_file": self.video_file,
            "video_name": self.video_name,
            "dataset": self.dataset,
            "station": {
                "code": self.station_code,
                "name": self.station_name
            },
            "platform_number": self.platform_number,
            "date": self.date,
            "time": self.time,
            "day": self.day
        }
