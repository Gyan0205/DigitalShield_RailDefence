"""
Digital Shield Rail Defense — Platform Mapping Engine
=======================================================
Resolves camera observations to platform-level intelligence.

Given a camera ID and timestamp, determines:
  - Which platform the camera covers
  - Which train is at that platform
  - Which coaches are visible
  - Which passengers could be on those coaches
  - The operational status of the platform

This is the core intelligence layer that bridges
CCTV anomaly detection → railway operations → passenger narrowing.
"""

import logging
from typing import Optional, List, Dict
from datetime import datetime
from collections import defaultdict

from backend.services.camera_registry import CameraRegistry, CameraRecord
from backend.services.railway_simulator import RailwaySimulator

logger = logging.getLogger("platform_mapper")
logger.setLevel(logging.INFO)
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)-25s | %(message)s", "%Y-%m-%d %H:%M:%S"
    ))
    logger.addHandler(_h)


class PlatformMapper:
    """
    Maps camera IDs to platform-level railway intelligence.

    The core workflow:
        camera_id → resolve camera → get platform → find train at platform
        → identify visible coaches → narrow passengers → return context

    Usage:
        mapper = PlatformMapper()
        mapper.initialize()
        result = mapper.resolve("CAM_NDLS_P05_B", timestamp="2026-05-12T08:30:00")
        result = mapper.resolve("CAM_05", station_code="NDLS")
    """

    def __init__(self):
        self.camera_registry = CameraRegistry()
        self.simulator = RailwaySimulator()
        self._initialized = False

    def initialize(self):
        """Initialize all subsystems."""
        if self._initialized:
            return

        logger.info("Initializing Platform Mapping Engine...")

        # Generate camera network
        self.camera_registry.generate_from_stations()
        logger.info(f"  Camera registry: {self.camera_registry.total_cameras} cameras")

        # Generate train schedules
        self.simulator.generate_trains(count=300)
        self.simulator.generate_schedules()
        logger.info(f"  Train database: {len(self.simulator.trains)} trains, "
                     f"{len(self.simulator.schedules)} schedules")

        self._initialized = True
        logger.info("Platform Mapping Engine initialized")

    def resolve(self, camera_id: str, station_code: str = None,
                timestamp: str = None, tolerance_minutes: int = 15) -> Dict:
        """
        Resolve a camera ID to complete platform intelligence.

        Args:
            camera_id: Camera identifier (e.g., "CAM_NDLS_P05_B" or "CAM_05")
            station_code: Station context for short IDs
            timestamp: ISO timestamp or "HH:MM" for train lookup
            tolerance_minutes: Schedule matching tolerance

        Returns:
            Complete platform intelligence context
        """
        if not self._initialized:
            self.initialize()

        result = {
            "camera_id": camera_id,
            "resolved": False,
            "platform": None,
            "station": None,
            "trains_at_platform": [],
            "visible_coaches": None,
            "operational_context": {},
        }

        # Step 1: Resolve camera to station/platform
        cam_result = self.camera_registry.resolve_camera(camera_id, station_code)

        if not cam_result.get("resolved"):
            result["error"] = cam_result.get("error", "Cannot resolve camera")
            return result

        result["resolved"] = True
        result["camera"] = cam_result
        result["platform"] = cam_result.get("platform")
        result["station"] = {
            "code": cam_result.get("station_code"),
            "name": cam_result.get("station_name", ""),
        }

        stn_code = cam_result.get("station_code")
        platform = cam_result.get("platform")

        # Enrich station info
        station_obj = self.simulator.get_station_by_code(stn_code)
        if station_obj:
            result["station"] = {
                "code": station_obj.code,
                "name": station_obj.name,
                "city": station_obj.city,
                "state": station_obj.state,
                "zone": station_obj.zone,
                "platforms": station_obj.platforms,
                "risk_tier": station_obj.risk_tier,
            }

        # Step 2: Find trains at this platform
        if timestamp and stn_code:
            time_str = self._extract_time(timestamp)
            if time_str:
                schedule_matches = self.simulator.find_train_at_station(
                    stn_code, time_str, tolerance_minutes
                )

                # Filter to this platform
                platform_matches = [
                    m for m in schedule_matches
                    if m.platform_number == platform
                ]

                # If no exact platform match, include all matches
                if not platform_matches:
                    platform_matches = schedule_matches

                for match in platform_matches:
                    train_info = self._build_train_context(match, cam_result)
                    result["trains_at_platform"].append(train_info)

        # Step 3: Compute visible coaches from this camera
        if cam_result.get("in_registry") and cam_result.get("coaches_visible"):
            coaches_range = cam_result["coaches_visible"]
            result["visible_coaches"] = {
                "range": coaches_range,
                "note": f"Coach positions {coaches_range} visible from {cam_result.get('zone', 'unknown')} zone camera",
            }

            # If we know the train, map positions to actual coach names
            if result["trains_at_platform"]:
                train = result["trains_at_platform"][0]
                coaches = train.get("coaches", [])
                try:
                    start, end = map(int, coaches_range.split("-"))
                    visible = coaches[start-1:end] if coaches else []
                    result["visible_coaches"]["coach_names"] = visible
                except (ValueError, IndexError):
                    pass

        # Step 4: Platform operational context
        if stn_code and platform:
            result["operational_context"] = self._get_platform_context(
                stn_code, platform, timestamp
            )

        # Step 5: Adjacent cameras
        if stn_code and platform:
            adjacent = self.camera_registry.get_by_platform(stn_code, platform, active_only=True)
            result["adjacent_cameras"] = [
                {"camera_id": c.camera_id, "zone": c.zone, "active": c.is_active}
                for c in adjacent
                if c.camera_id != camera_id
            ]

        return result

    def resolve_batch(self, camera_ids: List[str], station_code: str = None,
                      timestamp: str = None) -> List[Dict]:
        """Resolve multiple camera IDs at once."""
        return [self.resolve(cid, station_code, timestamp) for cid in camera_ids]

    def get_station_coverage_map(self, station_code: str) -> Dict:
        """
        Get complete camera coverage map for a station.

        Returns platform-by-platform camera layout with
        zone coverage and active/inactive status.
        """
        if not self._initialized:
            self.initialize()

        station = self.simulator.get_station_by_code(station_code)
        if not station:
            return {"error": f"Station not found: {station_code}"}

        coverage = self.camera_registry.get_camera_coverage(station_code)

        # Compute coverage completeness
        total_platforms = station.platforms
        covered_platforms = len(coverage)
        full_coverage = sum(
            1 for p, cams in coverage.items()
            if len(cams) >= 3 and all(c["active"] for c in cams)
        )

        return {
            "station": {
                "code": station.code,
                "name": station.name,
                "city": station.city,
                "total_platforms": total_platforms,
            },
            "coverage": {
                "platforms_covered": covered_platforms,
                "full_coverage_platforms": full_coverage,
                "coverage_pct": round(covered_platforms / total_platforms * 100, 1),
            },
            "platforms": coverage,
        }

    def get_all_stations_status(self) -> List[Dict]:
        """Get camera status summary for all stations."""
        if not self._initialized:
            self.initialize()

        status = []
        for station in self.simulator.stations:
            cameras = self.camera_registry.get_by_station(station.code)
            active = sum(1 for c in cameras if c.is_active and c.is_online)
            status.append({
                "station_code": station.code,
                "station_name": station.name,
                "city": station.city,
                "platforms": station.platforms,
                "total_cameras": len(cameras),
                "active_cameras": active,
                "offline_cameras": len(cameras) - active,
                "health_pct": round(active / len(cameras) * 100, 1) if cameras else 0,
                "risk_tier": station.risk_tier,
            })
        return sorted(status, key=lambda x: x["health_pct"])

    # ------------------------------------------------------------------
    # INTERNAL HELPERS
    # ------------------------------------------------------------------

    def _extract_time(self, timestamp: str) -> Optional[str]:
        """Extract HH:MM from various timestamp formats."""
        if not timestamp:
            return None
        # Try ISO format
        try:
            dt = datetime.fromisoformat(timestamp)
            return dt.strftime("%H:%M")
        except (ValueError, TypeError):
            pass
        # Already HH:MM
        if len(timestamp) == 5 and ":" in timestamp:
            return timestamp
        return None

    def _build_train_context(self, schedule_entry, cam_result: Dict) -> Dict:
        """Build train context from a schedule match."""
        train = next(
            (t for t in self.simulator.trains if t.number == schedule_entry.train_number),
            None
        )

        ctx = {
            "train_number": schedule_entry.train_number,
            "train_name": schedule_entry.train_name,
            "arrival": schedule_entry.arrival_time,
            "departure": schedule_entry.departure_time,
            "platform": schedule_entry.platform_number,
            "halt_minutes": schedule_entry.halt_minutes,
            "stop_sequence": schedule_entry.stop_sequence,
        }

        if train:
            ctx["train_type"] = train.train_type
            ctx["origin"] = train.origin_name
            ctx["destination"] = train.destination_name
            ctx["total_coaches"] = train.total_coaches
            ctx["coaches"] = train.coaches

        return ctx

    def _get_platform_context(self, station_code: str, platform: int,
                               timestamp: str = None) -> Dict:
        """Get operational context for a platform."""
        cameras = self.camera_registry.get_by_platform(station_code, platform)
        active = [c for c in cameras if c.is_active and c.is_online]

        hour = None
        if timestamp:
            time_str = self._extract_time(timestamp)
            if time_str:
                hour = int(time_str.split(":")[0])

        ctx = {
            "cameras_total": len(cameras),
            "cameras_active": len(active),
            "zones_covered": sorted(set(c.zone for c in active)),
            "full_coverage": len(set(c.zone for c in active)) >= 3,
        }

        if hour is not None:
            activity = self.simulator.get_platform_activity(station_code, platform, hour)
            ctx["activity_level"] = activity.get("activity_level", "unknown")
            ctx["trains_this_hour"] = activity.get("train_count", 0)

            # Risk assessment
            is_night = hour < 6 or hour >= 22
            is_peak = hour in (7, 8, 9, 17, 18, 19)
            ctx["is_night"] = is_night
            ctx["is_peak_hour"] = is_peak
            ctx["risk_modifier"] = "elevated" if is_night else ("normal" if is_peak else "standard")

        return ctx
