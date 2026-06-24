"""
Digital Shield Rail Defense — Train Intelligence Engine
=========================================================
Core intelligence layer that infers active trains from
platform observations. Bridges CCTV timestamps to railway
operational context.

Inference workflow:
    CCTV timestamp + camera/platform
    → ScheduleDB query
    → Active train identification
    → Coach estimation
    → Passenger narrowing context
    → Confidence scoring

Supports:
  - Exact time matching
  - Fuzzy time-window matching
  - Platform-specific queries
  - Train status inference (arriving/halted/departing)
  - Multi-train disambiguation
  - Occupancy-based confidence scoring
"""

import logging
from typing import Optional, List, Dict, Tuple
from datetime import datetime
from collections import defaultdict

from backend.services.schedule_db import ScheduleDB, ScheduleRecord
from backend.services.railway_simulator import RailwaySimulator
from backend.services.railway_stations import STATIONS_DB, STATION_LOOKUP

logger = logging.getLogger("train_intelligence")
logger.setLevel(logging.INFO)
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)-25s | %(message)s", "%Y-%m-%d %H:%M:%S"
    ))
    logger.addHandler(_h)


class TrainIntelligence:
    """
    Infers active trains from platform + timestamp observations.

    This is the core engine for the anomaly → train → passenger
    correlation workflow in Digital Shield.

    Usage:
        engine = TrainIntelligence()
        engine.initialize()

        # Infer train from observation
        result = engine.infer_train(
            station_code="NDLS",
            platform=5,
            timestamp="2026-05-12T08:30:00",
        )

        # Get complete platform context
        context = engine.get_platform_intelligence(
            station_code="NDLS",
            platform=5,
            timestamp="08:30",
        )
    """

    def __init__(self):
        self.schedule_db = ScheduleDB()
        self.simulator = RailwaySimulator()
        self._initialized = False

    def initialize(self, train_count: int = 300, apply_delays: bool = True):
        """Initialize simulator and build schedule database."""
        if self._initialized:
            return

        logger.info("Initializing Train Intelligence Engine...")
        self.simulator.generate_trains(count=train_count)
        self.simulator.generate_schedules()
        self.schedule_db.build_from_simulator(self.simulator, apply_delays=apply_delays)

        logger.info(
            f"  Schedule DB: {self.schedule_db.total_records} records, "
            f"{self.schedule_db.total_trains} trains, "
            f"{self.schedule_db.total_stations} stations"
        )
        self._initialized = True

    # ======================================================================
    # CORE: INFER TRAIN FROM OBSERVATION
    # ======================================================================

    def infer_train(
        self,
        station_code: str,
        platform: int = None,
        timestamp: str = None,
        tolerance_minutes: int = 15,
        day_of_week: str = None,
    ) -> Dict:
        """
        Infer the active train at a platform from a timestamp.

        Args:
            station_code: Station code (e.g., "NDLS")
            platform: Platform number
            timestamp: ISO datetime or "HH:MM"
            tolerance_minutes: Time window for matching
            day_of_week: "Mon", "Tue", etc. for filtering

        Returns:
            Inference result with matched trains, confidence, and context
        """
        if not self._initialized:
            self.initialize()

        time_str = self._extract_time(timestamp)
        if not time_str:
            return {"status": "error", "error": "Invalid timestamp format"}

        result = {
            "query": {
                "station_code": station_code,
                "platform": platform,
                "timestamp": timestamp,
                "time": time_str,
                "tolerance_minutes": tolerance_minutes,
            },
            "status": "success",
            "trains_found": 0,
            "primary_train": None,
            "all_matches": [],
            "confidence": 0.0,
            "inference_method": "",
        }

        # Step 1: Query by platform if available
        if platform:
            matches = self.schedule_db.query_by_platform(
                station_code, platform, time_str, tolerance_minutes
            )
            result["inference_method"] = "platform_time"
        else:
            matches = self.schedule_db.query_by_station_time(
                station_code, time_str, tolerance_minutes
            )
            result["inference_method"] = "station_time"

        # Step 2: Filter by running day
        if day_of_week and matches:
            day_abbr = day_of_week[:3]
            filtered = [m for m in matches if day_abbr in m.runs_on]
            if filtered:
                matches = filtered

        # Step 3: Build results and compute confidence
        if not matches:
            result["status"] = "no_match"
            result["confidence"] = 0.0
            return result

        result["trains_found"] = len(matches)

        # Score and rank matches
        scored = []
        for m in matches:
            score = self._compute_match_score(m, time_str, platform)
            train_result = self._build_train_result(m, score)
            scored.append((score, train_result))

        scored.sort(key=lambda x: x[0], reverse=True)

        # Primary train = highest scored
        result["primary_train"] = scored[0][1]
        result["confidence"] = scored[0][0]
        result["all_matches"] = [s[1] for s in scored]

        # Add train status inference
        if result["primary_train"]:
            result["primary_train"]["inferred_status"] = self._infer_status(
                scored[0][1], time_str
            )

        return result

    # ======================================================================
    # PLATFORM INTELLIGENCE (ENRICHED CONTEXT)
    # ======================================================================

    def get_platform_intelligence(
        self,
        station_code: str,
        platform: int,
        timestamp: str = None,
    ) -> Dict:
        """
        Get complete intelligence context for a platform observation.

        Returns everything needed to narrow passengers and
        generate explainable alerts.
        """
        if not self._initialized:
            self.initialize()

        time_str = self._extract_time(timestamp) or "12:00"
        hour = int(time_str.split(":")[0])

        station = STATION_LOOKUP.get(station_code)
        if not station:
            return {"error": f"Station not found: {station_code}"}

        # Train inference
        train_result = self.infer_train(station_code, platform, timestamp)

        # Platform occupancy
        occupancy = self.schedule_db.get_platform_occupancy(station_code, platform)

        # Platform conflicts
        conflicts = self.schedule_db.detect_platform_conflicts(station_code, platform)

        # Delay stats for station
        delay_stats = self.schedule_db.get_delay_statistics(station_code)

        # Adjacent trains (previous and next)
        timetable = self.schedule_db.get_station_timetable(station_code)
        platform_tt = [t for t in timetable if t.get("platform") == platform]

        prev_train = None
        next_train = None
        query_min = int(time_str.split(":")[0]) * 60 + int(time_str.split(":")[1])

        for t in platform_tt:
            dep = t.get("actual_departure") or t.get("scheduled_departure")
            if dep and dep != "--":
                try:
                    h, m = map(int, dep.split(":"))
                    t_min = h * 60 + m
                    if t_min < query_min:
                        prev_train = t
                    elif t_min > query_min and next_train is None:
                        next_train = t
                except ValueError:
                    pass

        return {
            "station": {
                "code": station.code,
                "name": station.name,
                "city": station.city,
                "state": station.state,
                "zone": station.zone,
                "risk_tier": station.risk_tier,
            },
            "platform": platform,
            "query_time": time_str,
            "train_inference": train_result,
            "occupancy": {
                "total_trains_today": occupancy["total_trains"],
                "current_hour_trains": occupancy["hourly_distribution"].get(hour, 0),
                "peak_hour": occupancy["peak_hour"],
                "peak_trains": occupancy["peak_trains"],
            },
            "delay_context": {
                "on_time_pct": delay_stats.get("on_time_pct", 0),
                "avg_delay": delay_stats.get("avg_delay_minutes", 0),
                "severe_delays": delay_stats.get("severe_delays", 0),
            },
            "conflicts": len(conflicts),
            "conflict_details": conflicts[:3],
            "adjacent_trains": {
                "previous": {
                    "train": prev_train.get("train_number") if prev_train else None,
                    "name": prev_train.get("train_name") if prev_train else None,
                    "departed": prev_train.get("actual_departure") if prev_train else None,
                } if prev_train else None,
                "next": {
                    "train": next_train.get("train_number") if next_train else None,
                    "name": next_train.get("train_name") if next_train else None,
                    "arriving": next_train.get("actual_arrival") if next_train else None,
                } if next_train else None,
            },
            "temporal_context": {
                "is_night": hour < 6 or hour >= 22,
                "is_peak_hour": hour in (7, 8, 9, 17, 18, 19),
                "activity_level": self._activity_level(occupancy["hourly_distribution"].get(hour, 0)),
            },
        }

    # ======================================================================
    # TIMESTAMP CORRELATION
    # ======================================================================

    def correlate_timestamp(
        self,
        station_code: str,
        iso_timestamp: str,
    ) -> Dict:
        """
        Correlate an ISO timestamp with the railway schedule.

        Returns all trains that could have been at the station
        around that time, with their arrival/departure status.
        """
        if not self._initialized:
            self.initialize()

        try:
            dt = datetime.fromisoformat(iso_timestamp)
        except (ValueError, TypeError):
            return {"error": "Invalid ISO timestamp"}

        time_str = dt.strftime("%H:%M")
        day_name = dt.strftime("%a")

        # Wide search: ±30 minutes
        matches = self.schedule_db.query_by_station_time(
            station_code, time_str, tolerance_minutes=30
        )

        # Filter by running day
        day_matches = [m for m in matches if day_name in m.runs_on]

        results = []
        for m in day_matches:
            status = self._infer_status(self._build_train_result(m, 0.5), time_str)
            results.append({
                "train_number": m.train_number,
                "train_name": m.train_name,
                "train_type": m.train_type,
                "platform": m.platform_number,
                "scheduled_arrival": m.scheduled_arrival,
                "scheduled_departure": m.scheduled_departure,
                "actual_arrival": m.actual_arrival,
                "actual_departure": m.actual_departure,
                "delay_minutes": m.delay_minutes,
                "inferred_status": status,
                "coaches": m.coaches,
                "total_coaches": m.total_coaches,
                "origin": m.origin_station,
                "destination": m.destination_station,
            })

        return {
            "timestamp": iso_timestamp,
            "station_code": station_code,
            "query_time": time_str,
            "day": day_name,
            "total_matches": len(results),
            "trains": results,
        }

    # ======================================================================
    # STATION-WIDE ANALYTICS
    # ======================================================================

    def get_station_schedule(self, station_code: str,
                              hour_start: int = 0, hour_end: int = 24) -> Dict:
        """Get full station timetable for a time window."""
        if not self._initialized:
            self.initialize()

        timetable = self.schedule_db.get_station_timetable(
            station_code, hour_start, hour_end
        )
        station = STATION_LOOKUP.get(station_code)

        return {
            "station": {
                "code": station_code,
                "name": station.name if station else "",
            },
            "time_window": f"{hour_start:02d}:00 - {hour_end:02d}:00",
            "total_services": len(timetable),
            "timetable": timetable,
        }

    def get_train_full_route(self, train_number: str) -> Dict:
        """Get complete route and schedule for a specific train."""
        if not self._initialized:
            self.initialize()

        info = self.schedule_db.get_train_info(train_number)
        if not info:
            return {"error": f"Train not found: {train_number}"}
        return info

    def search(self, query: str) -> Dict:
        """Search trains by number or name."""
        if not self._initialized:
            self.initialize()

        results = self.schedule_db.search_trains(query)
        return {
            "query": query,
            "results": len(results),
            "trains": results,
        }

    def get_network_delay_report(self) -> Dict:
        """Get delay statistics across the entire network."""
        if not self._initialized:
            self.initialize()

        overall = self.schedule_db.get_delay_statistics()

        # Per-station breakdown (top 10 most delayed)
        station_delays = []
        for station in STATIONS_DB:
            stats = self.schedule_db.get_delay_statistics(station.code)
            if stats.get("total_services", 0) > 0:
                station_delays.append({
                    "station_code": station.code,
                    "station_name": station.name,
                    **stats,
                })

        station_delays.sort(key=lambda x: x.get("avg_delay_minutes", 0), reverse=True)

        return {
            "overall": overall,
            "most_delayed_stations": station_delays[:10],
            "most_punctual_stations": station_delays[-5:] if len(station_delays) > 5 else [],
        }

    # ======================================================================
    # INTERNAL HELPERS
    # ======================================================================

    def _compute_match_score(self, record: ScheduleRecord,
                              time_str: str, platform: int = None) -> float:
        """Score a schedule match from 0.0 to 1.0."""
        score = 0.5  # Base score

        try:
            qh, qm = map(int, time_str.split(":"))
            query_min = qh * 60 + qm
        except ValueError:
            return score

        # Time proximity bonus (closer = higher)
        arr_min = record.arrival_minutes
        dep_min = record.departure_minutes

        best_diff = 999
        for t in [arr_min, dep_min]:
            if t is not None:
                diff = min(abs(t - query_min), 1440 - abs(t - query_min))
                best_diff = min(best_diff, diff)

        if best_diff <= 2:
            score += 0.4
        elif best_diff <= 5:
            score += 0.3
        elif best_diff <= 10:
            score += 0.2
        elif best_diff <= 15:
            score += 0.1

        # Within dwell window bonus
        if arr_min is not None and dep_min is not None:
            if arr_min <= query_min <= dep_min:
                score += 0.1  # Currently at platform

        # Platform match bonus
        if platform and record.platform_number == platform:
            score += 0.05

        # Train type reliability bonus
        reliability = {
            "Rajdhani Express": 0.03, "Shatabdi Express": 0.03,
            "Vande Bharat Express": 0.04, "Duronto Express": 0.02,
        }
        score += reliability.get(record.train_type, 0)

        # On-time bonus
        if record.delay_minutes == 0:
            score += 0.02

        return min(score, 1.0)

    def _build_train_result(self, record: ScheduleRecord, score: float) -> Dict:
        """Build a structured train result from a schedule record."""
        return {
            "train_number": record.train_number,
            "train_name": record.train_name,
            "train_type": record.train_type,
            "platform": record.platform_number,
            "scheduled_arrival": record.scheduled_arrival,
            "scheduled_departure": record.scheduled_departure,
            "actual_arrival": record.actual_arrival,
            "actual_departure": record.actual_departure,
            "delay_minutes": record.delay_minutes,
            "halt_minutes": record.halt_minutes,
            "dwell_time": record.dwell_time_minutes,
            "stop_sequence": record.stop_sequence,
            "origin": record.origin_station,
            "destination": record.destination_station,
            "coaches": record.coaches,
            "total_coaches": record.total_coaches,
            "status": record.status,
            "match_score": round(score, 4),
        }

    def _infer_status(self, train_result: Dict, time_str: str) -> str:
        """Infer train status relative to query time."""
        try:
            qh, qm = map(int, time_str.split(":"))
            query_min = qh * 60 + qm
        except ValueError:
            return "unknown"

        arr_str = train_result.get("actual_arrival") or train_result.get("scheduled_arrival", "--")
        dep_str = train_result.get("actual_departure") or train_result.get("scheduled_departure", "--")

        arr_min = None
        dep_min = None

        if arr_str and arr_str != "--":
            try:
                h, m = map(int, arr_str.split(":"))
                arr_min = h * 60 + m
            except ValueError:
                pass
        if dep_str and dep_str != "--":
            try:
                h, m = map(int, dep_str.split(":"))
                dep_min = h * 60 + m
            except ValueError:
                pass

        if arr_min is None and dep_min is not None:
            # Origin station
            if query_min < dep_min:
                return "boarding"
            elif query_min == dep_min:
                return "departing"
            else:
                return "departed"
        elif dep_min is None and arr_min is not None:
            # Destination station
            if query_min < arr_min:
                return "approaching"
            elif query_min == arr_min:
                return "arriving"
            else:
                return "arrived_terminus"
        elif arr_min is not None and dep_min is not None:
            if query_min < arr_min - 5:
                return "approaching"
            elif arr_min - 5 <= query_min < arr_min:
                return "arriving"
            elif arr_min <= query_min <= dep_min:
                return "halted"
            elif dep_min < query_min <= dep_min + 3:
                return "departing"
            else:
                return "departed"

        return "unknown"

    @staticmethod
    def _activity_level(train_count: int) -> str:
        if train_count >= 8:
            return "very_high"
        elif train_count >= 5:
            return "high"
        elif train_count >= 3:
            return "moderate"
        elif train_count >= 1:
            return "low"
        return "idle"

    @staticmethod
    def _extract_time(timestamp: str) -> Optional[str]:
        if not timestamp:
            return None
        try:
            dt = datetime.fromisoformat(timestamp)
            return dt.strftime("%H:%M")
        except (ValueError, TypeError):
            pass
        if len(timestamp) == 5 and ":" in timestamp:
            return timestamp
        return None
