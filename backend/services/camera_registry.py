"""
Digital Shield Rail Defense — Camera Intelligence Registry
=============================================================
Core registry system for CCTV camera installations across the
Indian railway network. Provides CRUD operations, camera-to-platform
mapping, coach visibility computation, and bulk management.

Camera ID Convention:
    CAM_{STATION_CODE}_P{PLATFORM:02d}_{ZONE}
    Example: CAM_NDLS_P05_B → New Delhi, Platform 5, Mid-zone

Zone mapping:
    A = Entry zone  (covers coaches 1-4 from engine)
    B = Mid zone    (covers coaches 5-12)
    C = Exit zone   (covers coaches 13-24)
"""

import json
import logging
import re
from pathlib import Path
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime
from collections import defaultdict

logger = logging.getLogger("camera_registry")
logger.setLevel(logging.INFO)
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)-25s | %(message)s", "%Y-%m-%d %H:%M:%S"
    ))
    logger.addHandler(_h)


# ============================================================================
# CAMERA RECORD (extended from base schema)
# ============================================================================

@dataclass
class CameraRecord:
    """Complete CCTV camera installation record with operational fields."""
    camera_id: str
    station_code: str
    station_name: str
    platform_number: int
    zone: str                        # "entry", "mid", "exit"
    camera_type: str = "PTZ"         # PTZ, Fixed, Dome, Bullet
    resolution: str = "1080p"
    feed_url: str = ""               # RTSP URL
    is_active: bool = True
    is_online: bool = True
    coverage_area_sqm: float = 200.0
    visible_coaches_start: int = 0
    visible_coaches_end: int = 4
    # Installation metadata
    installed_date: str = ""
    last_maintenance: str = ""
    firmware_version: str = "v4.2.1"
    night_vision: bool = True
    audio_enabled: bool = False
    # Operational flags
    tampering_alert: bool = False
    recording_status: str = "active" # active, paused, error
    storage_days: int = 30

    def to_dict(self) -> Dict:
        return asdict(self)

    @staticmethod
    def parse_camera_id(camera_id: str) -> Optional[Dict]:
        """
        Parse a camera ID into its components.

        CAM_NDLS_P05_B → {"station": "NDLS", "platform": 5, "zone_code": "B", "zone": "mid"}
        CAM_05         → {"station": None, "platform": 5, "zone_code": None, "zone": None}
        """
        zone_map = {"A": "entry", "B": "mid", "C": "exit"}

        # Full format: CAM_XXXX_P##_Z
        full_match = re.match(r'^CAM_([A-Z]{2,5})_P(\d{1,2})_([A-C])$', camera_id)
        if full_match:
            return {
                "station_code": full_match.group(1),
                "platform": int(full_match.group(2)),
                "zone_code": full_match.group(3),
                "zone": zone_map.get(full_match.group(3), "unknown"),
            }

        # Short format: CAM_## or CAM##
        short_match = re.match(r'^CAM[_-]?(\d{1,2})$', camera_id)
        if short_match:
            return {
                "station_code": None,
                "platform": int(short_match.group(1)),
                "zone_code": None,
                "zone": None,
            }

        return None


# ============================================================================
# CAMERA REGISTRY SERVICE
# ============================================================================

class CameraRegistry:
    """
    Central registry for all CCTV cameras in the railway network.
    Supports CRUD operations, lookup by various keys, and persistence.

    Usage:
        registry = CameraRegistry()
        registry.load("dataset/metadata/camera_registry.json")
        cam = registry.get("CAM_NDLS_P05_B")
        cams = registry.get_by_station("NDLS")
        cams = registry.get_by_platform("NDLS", 5)
        mapped = registry.resolve_camera("CAM_05", station_code="NDLS")
    """

    def __init__(self):
        self._cameras: Dict[str, CameraRecord] = {}
        self._station_index: Dict[str, List[str]] = defaultdict(list)
        self._platform_index: Dict[str, List[str]] = defaultdict(list)

    @property
    def total_cameras(self) -> int:
        return len(self._cameras)

    @property
    def stations(self) -> List[str]:
        return sorted(self._station_index.keys())

    # ------------------------------------------------------------------
    # CRUD OPERATIONS
    # ------------------------------------------------------------------

    def register(self, camera: CameraRecord) -> CameraRecord:
        """Register a new camera or update existing."""
        self._cameras[camera.camera_id] = camera
        self._station_index[camera.station_code].append(camera.camera_id)
        key = f"{camera.station_code}_P{camera.platform_number}"
        self._platform_index[key].append(camera.camera_id)
        return camera

    def bulk_register(self, cameras: List[CameraRecord]):
        """Register multiple cameras at once."""
        for cam in cameras:
            self.register(cam)
        logger.info(f"Bulk registered {len(cameras)} cameras")

    def get(self, camera_id: str) -> Optional[CameraRecord]:
        """Get a camera by its ID."""
        return self._cameras.get(camera_id)

    def remove(self, camera_id: str) -> bool:
        """Remove a camera from the registry."""
        cam = self._cameras.pop(camera_id, None)
        if cam:
            self._station_index[cam.station_code] = [
                c for c in self._station_index[cam.station_code] if c != camera_id
            ]
            key = f"{cam.station_code}_P{cam.platform_number}"
            self._platform_index[key] = [
                c for c in self._platform_index[key] if c != camera_id
            ]
            return True
        return False

    def update_status(self, camera_id: str, is_active: bool = None,
                      is_online: bool = None, recording_status: str = None) -> Optional[CameraRecord]:
        """Update operational status of a camera."""
        cam = self._cameras.get(camera_id)
        if not cam:
            return None
        if is_active is not None:
            cam.is_active = is_active
        if is_online is not None:
            cam.is_online = is_online
        if recording_status is not None:
            cam.recording_status = recording_status
        return cam

    # ------------------------------------------------------------------
    # QUERY OPERATIONS
    # ------------------------------------------------------------------

    def get_by_station(self, station_code: str, active_only: bool = False) -> List[CameraRecord]:
        """Get all cameras at a station."""
        cam_ids = self._station_index.get(station_code, [])
        cameras = [self._cameras[cid] for cid in cam_ids if cid in self._cameras]
        if active_only:
            cameras = [c for c in cameras if c.is_active and c.is_online]
        return cameras

    def get_by_platform(self, station_code: str, platform: int,
                        active_only: bool = False) -> List[CameraRecord]:
        """Get all cameras on a specific platform."""
        key = f"{station_code}_P{platform}"
        cam_ids = self._platform_index.get(key, [])
        cameras = [self._cameras[cid] for cid in cam_ids if cid in self._cameras]
        if active_only:
            cameras = [c for c in cameras if c.is_active and c.is_online]
        return cameras

    def get_by_zone(self, station_code: str, zone: str) -> List[CameraRecord]:
        """Get all cameras in a specific zone at a station."""
        return [
            c for c in self.get_by_station(station_code)
            if c.zone == zone
        ]

    def find_cameras_for_coach(self, station_code: str, platform: int,
                                coach_position: int) -> List[CameraRecord]:
        """Find cameras that can see a specific coach position."""
        return [
            c for c in self.get_by_platform(station_code, platform, active_only=True)
            if c.visible_coaches_start <= coach_position <= c.visible_coaches_end
        ]

    def get_camera_coverage(self, station_code: str) -> Dict[int, List[Dict]]:
        """Get complete camera coverage map for a station."""
        coverage = defaultdict(list)
        for cam in self.get_by_station(station_code):
            coverage[cam.platform_number].append({
                "camera_id": cam.camera_id,
                "zone": cam.zone,
                "coaches_visible": f"{cam.visible_coaches_start}-{cam.visible_coaches_end}",
                "type": cam.camera_type,
                "active": cam.is_active and cam.is_online,
            })
        return dict(coverage)

    # ------------------------------------------------------------------
    # CAMERA ID RESOLUTION (the core mapping engine)
    # ------------------------------------------------------------------

    def resolve_camera(self, camera_id: str, station_code: str = None,
                       context: Dict = None) -> Dict:
        """
        Resolve a camera ID to its full station/platform context.

        Handles both full IDs (CAM_NDLS_P05_B) and short IDs (CAM_05).
        When given a short ID, uses station_code context to disambiguate.

        Args:
            camera_id: Camera identifier (full or short)
            station_code: Station context for short IDs
            context: Additional context (timestamp, train, etc.)

        Returns:
            Resolved mapping with station, platform, zone, and camera details
        """
        # Try direct lookup first
        cam = self.get(camera_id)
        if cam:
            return self._build_resolution(cam, confidence=1.0)

        # Parse camera ID
        parsed = CameraRecord.parse_camera_id(camera_id)
        if not parsed:
            return {"resolved": False, "camera_id": camera_id, "error": "Cannot parse camera ID"}

        # Full ID with station code embedded
        if parsed["station_code"]:
            full_id = f"CAM_{parsed['station_code']}_P{parsed['platform']:02d}_{parsed['zone_code']}"
            cam = self.get(full_id)
            if cam:
                return self._build_resolution(cam, confidence=1.0)

            # Camera not in registry — return parsed info
            return {
                "resolved": True,
                "camera_id": camera_id,
                "station_code": parsed["station_code"],
                "platform": parsed["platform"],
                "zone": parsed["zone"],
                "in_registry": False,
                "confidence": 0.8,
            }

        # Short ID — need station context
        if station_code:
            # Try all zones for this platform
            for zone_code in ["A", "B", "C"]:
                full_id = f"CAM_{station_code}_P{parsed['platform']:02d}_{zone_code}"
                cam = self.get(full_id)
                if cam:
                    return self._build_resolution(cam, confidence=0.9,
                                                    note="Resolved from short ID with station context")

            return {
                "resolved": True,
                "camera_id": camera_id,
                "station_code": station_code,
                "platform": parsed["platform"],
                "zone": None,
                "in_registry": False,
                "confidence": 0.6,
                "note": "Platform inferred from short ID",
            }

        return {
            "resolved": False,
            "camera_id": camera_id,
            "platform": parsed.get("platform"),
            "error": "Station context required for short camera IDs",
        }

    def _build_resolution(self, cam: CameraRecord, confidence: float = 1.0,
                          note: str = "") -> Dict:
        """Build a full resolution result from a camera record."""
        return {
            "resolved": True,
            "camera_id": cam.camera_id,
            "station_code": cam.station_code,
            "station_name": cam.station_name,
            "platform": cam.platform_number,
            "zone": cam.zone,
            "camera_type": cam.camera_type,
            "resolution": cam.resolution,
            "is_active": cam.is_active,
            "is_online": cam.is_online,
            "coaches_visible": f"{cam.visible_coaches_start}-{cam.visible_coaches_end}",
            "coverage_sqm": cam.coverage_area_sqm,
            "night_vision": cam.night_vision,
            "in_registry": True,
            "confidence": confidence,
            "note": note,
        }

    # ------------------------------------------------------------------
    # STATISTICS
    # ------------------------------------------------------------------

    def get_network_stats(self) -> Dict:
        """Get statistics for the entire camera network."""
        total = len(self._cameras)
        active = sum(1 for c in self._cameras.values() if c.is_active)
        online = sum(1 for c in self._cameras.values() if c.is_online)

        zone_counts = defaultdict(int)
        type_counts = defaultdict(int)
        resolution_counts = defaultdict(int)
        for c in self._cameras.values():
            zone_counts[c.zone] += 1
            type_counts[c.camera_type] += 1
            resolution_counts[c.resolution] += 1

        return {
            "total_cameras": total,
            "active": active,
            "inactive": total - active,
            "online": online,
            "offline": total - online,
            "stations_covered": len(self._station_index),
            "platforms_covered": len(self._platform_index),
            "by_zone": dict(zone_counts),
            "by_type": dict(type_counts),
            "by_resolution": dict(resolution_counts),
        }

    # ------------------------------------------------------------------
    # PERSISTENCE
    # ------------------------------------------------------------------

    def save(self, filepath: Path):
        """Save registry to JSON file."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "version": "2.0",
            "created_at": datetime.now().isoformat(),
            "total": len(self._cameras),
            "cameras": [c.to_dict() for c in self._cameras.values()],
        }
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)
        logger.info(f"Registry saved: {filepath} ({len(self._cameras)} cameras)")

    def load(self, filepath: Path) -> int:
        """Load registry from JSON file."""
        filepath = Path(filepath)
        if not filepath.exists():
            logger.warning(f"Registry file not found: {filepath}")
            return 0

        with open(filepath) as f:
            data = json.load(f)

        cameras_data = data.get("cameras", [])
        count = 0
        for c in cameras_data:
            cam = CameraRecord(
                camera_id=c.get("camera_id", ""),
                station_code=c.get("station_code", c.get("station", "")),
                station_name=c.get("station_name", ""),
                platform_number=c.get("platform_number", c.get("platform", 0)),
                zone=c.get("zone", ""),
                camera_type=c.get("camera_type", c.get("type", "PTZ")),
                resolution=c.get("resolution", "1080p"),
                feed_url=c.get("feed_url", ""),
                is_active=c.get("is_active", c.get("active", True)),
                is_online=c.get("is_online", True),
                coverage_area_sqm=c.get("coverage_area_sqm", 200.0),
                visible_coaches_start=c.get("visible_coaches_start", 0),
                visible_coaches_end=c.get("visible_coaches_end", 4),
                night_vision=c.get("night_vision", True),
            )
            self.register(cam)
            count += 1

        logger.info(f"Loaded {count} cameras from {filepath}")
        return count

    def generate_from_stations(self, stations: List = None) -> int:
        """
        Generate a complete camera network from station data.
        Creates 3 cameras per platform (entry/mid/exit).
        """
        import random
        from backend.services.railway_stations import STATIONS_DB
        station_list = stations or STATIONS_DB

        zone_config = {
            "A": {"name": "entry", "coach_start": 1, "coach_end": 4, "type": "PTZ"},
            "B": {"name": "mid", "coach_start": 5, "coach_end": 12, "type": "PTZ"},
            "C": {"name": "exit", "coach_start": 13, "coach_end": 24, "type": "Fixed"},
        }
        count = 0
        for station in station_list:
            for platform in range(1, station.platforms + 1):
                for zone_code, zinfo in zone_config.items():
                    cam = CameraRecord(
                        camera_id=f"CAM_{station.code}_P{platform:02d}_{zone_code}",
                        station_code=station.code,
                        station_name=station.name,
                        platform_number=platform,
                        zone=zinfo["name"],
                        camera_type=zinfo["type"],
                        resolution="1080p" if station.daily_footfall > 100000 else "720p",
                        is_active=random.random() > 0.03,
                        is_online=random.random() > 0.05,
                        coverage_area_sqm=random.uniform(150, 300),
                        visible_coaches_start=zinfo["coach_start"],
                        visible_coaches_end=zinfo["coach_end"],
                        installed_date="2024-01-15",
                        night_vision=True,
                    )
                    self.register(cam)
                    count += 1

        logger.info(f"Generated {count} cameras for {len(station_list)} stations")
        return count
