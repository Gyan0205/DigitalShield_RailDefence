"""
Digital Shield Rail Defense — Schedule Database
==================================================
High-performance in-memory schedule database with multi-key
indexing for O(1) lookups by station, platform, train, and
time-window. Supports import/export, delay simulation,
and schedule conflict detection.

Index structure:
  _by_station[station_code] → [ScheduleEntry...]
  _by_train[train_number]   → [ScheduleEntry...]
  _by_platform[(station,pf)] → [ScheduleEntry...]
  _by_hour[(station,hour)]   → [ScheduleEntry...]
"""

import json
import logging
import random
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Set
from datetime import datetime, timedelta
from collections import defaultdict
from dataclasses import dataclass, field, asdict

logger = logging.getLogger("schedule_db")
logger.setLevel(logging.INFO)
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)-25s | %(message)s", "%Y-%m-%d %H:%M:%S"
    ))
    logger.addHandler(_h)


# ============================================================================
# ENHANCED SCHEDULE RECORD
# ============================================================================

@dataclass
class ScheduleRecord:
    """Extended schedule entry with delay and status modeling."""
    # Core identity
    train_number: str
    train_name: str
    train_type: str
    station_code: str
    station_name: str
    stop_sequence: int

    # Times
    scheduled_arrival: str       # "HH:MM" or "--" for origin
    scheduled_departure: str     # "HH:MM" or "--" for destination
    actual_arrival: str = ""     # With delay applied
    actual_departure: str = ""

    # Platform
    platform_number: int = 0
    halt_minutes: int = 0
    distance_from_origin_km: int = 0
    day: int = 1

    # Operational
    delay_minutes: int = 0
    status: str = "on_time"      # on_time, delayed, arrived, departed, cancelled
    is_origin: bool = False
    is_destination: bool = False

    # Route context
    origin_station: str = ""
    destination_station: str = ""
    coaches: List[str] = field(default_factory=list)
    total_coaches: int = 0
    runs_on: List[str] = field(default_factory=list)

    def __post_init__(self):
        """Compute actual times from scheduled + delay."""
        if not self.actual_arrival:
            self.actual_arrival = self._apply_delay(self.scheduled_arrival, self.delay_minutes)
        if not self.actual_departure:
            self.actual_departure = self._apply_delay(self.scheduled_departure, self.delay_minutes)

    @staticmethod
    def _apply_delay(time_str: str, delay_min: int) -> str:
        if time_str == "--" or not time_str:
            return time_str
        try:
            h, m = map(int, time_str.split(":"))
            total = h * 60 + m + delay_min
            return f"{(total // 60) % 24:02d}:{total % 60:02d}"
        except ValueError:
            return time_str

    @property
    def arrival_minutes(self) -> Optional[int]:
        """Convert actual arrival to minutes since midnight."""
        return self._to_minutes(self.actual_arrival or self.scheduled_arrival)

    @property
    def departure_minutes(self) -> Optional[int]:
        """Convert actual departure to minutes since midnight."""
        return self._to_minutes(self.actual_departure or self.scheduled_departure)

    @property
    def scheduled_arrival_minutes(self) -> Optional[int]:
        return self._to_minutes(self.scheduled_arrival)

    @property
    def scheduled_departure_minutes(self) -> Optional[int]:
        return self._to_minutes(self.scheduled_departure)

    @staticmethod
    def _to_minutes(time_str: str) -> Optional[int]:
        if not time_str or time_str == "--":
            return None
        try:
            h, m = map(int, time_str.split(":"))
            return h * 60 + m
        except ValueError:
            return None

    @property
    def dwell_time_minutes(self) -> int:
        """Compute actual dwell time at this station."""
        arr = self.arrival_minutes
        dep = self.departure_minutes
        if arr is not None and dep is not None:
            diff = dep - arr
            return diff if diff >= 0 else diff + 1440
        return self.halt_minutes

    def to_dict(self) -> Dict:
        return {
            "train_number": self.train_number,
            "train_name": self.train_name,
            "train_type": self.train_type,
            "station_code": self.station_code,
            "station_name": self.station_name,
            "stop_sequence": self.stop_sequence,
            "scheduled_arrival": self.scheduled_arrival,
            "scheduled_departure": self.scheduled_departure,
            "actual_arrival": self.actual_arrival,
            "actual_departure": self.actual_departure,
            "platform": self.platform_number,
            "halt_minutes": self.halt_minutes,
            "dwell_time": self.dwell_time_minutes,
            "delay_minutes": self.delay_minutes,
            "status": self.status,
            "distance_km": self.distance_from_origin_km,
            "day": self.day,
            "is_origin": self.is_origin,
            "is_destination": self.is_destination,
            "origin": self.origin_station,
            "destination": self.destination_station,
            "coaches": self.coaches,
            "total_coaches": self.total_coaches,
        }


# ============================================================================
# SCHEDULE DATABASE
# ============================================================================

class ScheduleDB:
    """
    High-performance in-memory schedule database.

    Multi-key indexing for fast lookups:
      - By station code
      - By train number
      - By (station, platform) pair
      - By (station, hour) pair

    Usage:
        db = ScheduleDB()
        db.build_from_simulator(simulator)
        trains = db.query_by_station_time("NDLS", "08:30", tolerance=15)
        trains = db.query_by_platform("NDLS", 5, "08:30")
        full_route = db.get_train_route("12727")
    """

    def __init__(self):
        self._records: List[ScheduleRecord] = []
        self._by_station: Dict[str, List[int]] = defaultdict(list)
        self._by_train: Dict[str, List[int]] = defaultdict(list)
        self._by_platform: Dict[Tuple[str, int], List[int]] = defaultdict(list)
        self._by_hour: Dict[Tuple[str, int], List[int]] = defaultdict(list)
        self._train_lookup: Dict[str, Dict] = {}  # train_number → train info

    @property
    def total_records(self) -> int:
        return len(self._records)

    @property
    def total_trains(self) -> int:
        return len(self._by_train)

    @property
    def total_stations(self) -> int:
        return len(self._by_station)

    # ------------------------------------------------------------------
    # BUILD / INSERT
    # ------------------------------------------------------------------

    def insert(self, record: ScheduleRecord):
        """Insert a schedule record and update all indexes."""
        idx = len(self._records)
        self._records.append(record)

        self._by_station[record.station_code].append(idx)
        self._by_train[record.train_number].append(idx)
        self._by_platform[(record.station_code, record.platform_number)].append(idx)

        # Index by hour for arrival and departure
        for time_str in [record.scheduled_arrival, record.scheduled_departure]:
            if time_str and time_str != "--":
                try:
                    h = int(time_str.split(":")[0])
                    self._by_hour[(record.station_code, h)].append(idx)
                except ValueError:
                    pass

    def build_from_simulator(self, simulator, apply_delays: bool = True):
        """
        Populate from RailwaySimulator data.

        Enriches each ScheduleEntry with train details and
        optionally applies realistic delay simulation.
        """
        if not simulator.schedules:
            simulator.generate_schedules()
        if not simulator.trains:
            simulator.generate_trains()

        # Build train lookup
        for train in simulator.trains:
            self._train_lookup[train.number] = {
                "name": train.name, "type": train.train_type,
                "origin_code": train.origin_code, "origin_name": train.origin_name,
                "dest_code": train.destination_code, "dest_name": train.destination_name,
                "coaches": train.coaches, "total_coaches": train.total_coaches,
                "runs_on": train.runs_on, "avg_speed": train.avg_speed_kmph,
            }

        # Group schedule entries by train to identify origin/dest
        train_entries = defaultdict(list)
        for entry in simulator.schedules:
            train_entries[entry.train_number].append(entry)

        for train_number, entries in train_entries.items():
            train_info = self._train_lookup.get(train_number, {})
            entries_sorted = sorted(entries, key=lambda e: e.stop_sequence)

            # Simulate delay (Indian railways delay distribution)
            delay = 0
            if apply_delays:
                delay = self._simulate_delay(train_info.get("type", "Express"))

            for i, entry in enumerate(entries_sorted):
                is_origin = i == 0
                is_dest = i == len(entries_sorted) - 1

                record = ScheduleRecord(
                    train_number=entry.train_number,
                    train_name=entry.train_name,
                    train_type=train_info.get("type", "Express"),
                    station_code=entry.station_code,
                    station_name=entry.station_name,
                    stop_sequence=entry.stop_sequence,
                    scheduled_arrival=entry.arrival_time,
                    scheduled_departure=entry.departure_time,
                    platform_number=entry.platform_number,
                    halt_minutes=entry.halt_minutes,
                    distance_from_origin_km=entry.distance_from_origin_km,
                    day=entry.day,
                    delay_minutes=delay,
                    status="on_time" if delay == 0 else "delayed",
                    is_origin=is_origin,
                    is_destination=is_dest,
                    origin_station=train_info.get("origin_name", ""),
                    destination_station=train_info.get("dest_name", ""),
                    coaches=train_info.get("coaches", []),
                    total_coaches=train_info.get("total_coaches", 0),
                    runs_on=train_info.get("runs_on", []),
                )
                self.insert(record)

            # Accumulate delay along route (later stops are more delayed)
            if apply_delays and delay > 0:
                pass  # Already applied uniformly; could cascade here

        logger.info(
            f"ScheduleDB built: {self.total_records} records, "
            f"{self.total_trains} trains, {self.total_stations} stations"
        )

    def _simulate_delay(self, train_type: str) -> int:
        """Simulate realistic Indian railway delays."""
        # Delay probability by train type
        delay_profiles = {
            "Rajdhani Express": (0.15, 5, 20),    # 15% chance, 5-20 min
            "Shatabdi Express": (0.12, 5, 15),
            "Vande Bharat Express": (0.08, 3, 10),
            "Duronto Express": (0.20, 5, 30),
            "Superfast Express": (0.30, 5, 45),
            "Express": (0.45, 10, 90),
            "Mail": (0.50, 10, 120),
            "Passenger": (0.60, 15, 180),
        }
        prob, min_delay, max_delay = delay_profiles.get(train_type, (0.35, 5, 60))

        if random.random() < prob:
            # Heavy-tailed delay distribution
            base = random.randint(min_delay, max_delay)
            if random.random() < 0.1:  # 10% chance of severe delay
                base = int(base * random.uniform(1.5, 3.0))
            return base
        return 0

    # ------------------------------------------------------------------
    # QUERY: BY STATION + TIME
    # ------------------------------------------------------------------

    def query_by_station_time(self, station_code: str, time_str: str,
                               tolerance_minutes: int = 15,
                               use_actual: bool = True) -> List[ScheduleRecord]:
        """
        Find all trains at a station around a given time.

        Args:
            station_code: Station code (e.g., "NDLS")
            time_str: "HH:MM" format
            tolerance_minutes: Search window ±
            use_actual: Use actual (delayed) times vs scheduled

        Returns:
            Sorted list of matching ScheduleRecords
        """
        try:
            qh, qm = map(int, time_str.split(":"))
            query_min = qh * 60 + qm
        except ValueError:
            return []

        indices = set(self._by_station.get(station_code, []))
        matches = []

        for idx in indices:
            record = self._records[idx]

            if use_actual:
                arr_min = record.arrival_minutes
                dep_min = record.departure_minutes
            else:
                arr_min = record.scheduled_arrival_minutes
                dep_min = record.scheduled_departure_minutes

            # Check if query time falls within arrival-departure window
            for t_min in [arr_min, dep_min]:
                if t_min is None:
                    continue
                diff = abs(t_min - query_min)
                # Handle midnight wraparound
                diff = min(diff, 1440 - diff)
                if diff <= tolerance_minutes:
                    matches.append(record)
                    break

        # Sort by relevance (closest to query time first)
        matches.sort(key=lambda r: self._time_distance(
            r.arrival_minutes or r.departure_minutes, query_min
        ))
        return matches

    # ------------------------------------------------------------------
    # QUERY: BY PLATFORM
    # ------------------------------------------------------------------

    def query_by_platform(self, station_code: str, platform: int,
                          time_str: str = None,
                          tolerance_minutes: int = 15) -> List[ScheduleRecord]:
        """Find all trains at a specific platform, optionally filtered by time."""
        key = (station_code, platform)
        indices = self._by_platform.get(key, [])
        records = [self._records[i] for i in indices]

        if time_str:
            try:
                qh, qm = map(int, time_str.split(":"))
                query_min = qh * 60 + qm
            except ValueError:
                return records

            filtered = []
            for r in records:
                for t_min in [r.arrival_minutes, r.departure_minutes]:
                    if t_min is not None:
                        diff = min(abs(t_min - query_min), 1440 - abs(t_min - query_min))
                        if diff <= tolerance_minutes:
                            filtered.append(r)
                            break
            return filtered

        return records

    # ------------------------------------------------------------------
    # QUERY: BY TRAIN
    # ------------------------------------------------------------------

    def get_train_route(self, train_number: str) -> List[ScheduleRecord]:
        """Get complete route/schedule for a train."""
        indices = self._by_train.get(train_number, [])
        records = [self._records[i] for i in indices]
        return sorted(records, key=lambda r: r.stop_sequence)

    def get_train_info(self, train_number: str) -> Optional[Dict]:
        """Get enriched train information."""
        info = self._train_lookup.get(train_number)
        if not info:
            return None

        route = self.get_train_route(train_number)
        stops = [
            {
                "station_code": r.station_code, "station_name": r.station_name,
                "arrival": r.scheduled_arrival, "departure": r.scheduled_departure,
                "platform": r.platform_number, "halt_min": r.halt_minutes,
                "delay": r.delay_minutes, "distance_km": r.distance_from_origin_km,
            }
            for r in route
        ]

        return {
            "train_number": train_number,
            **info,
            "total_stops": len(stops),
            "route": stops,
            "total_delay": route[0].delay_minutes if route else 0,
            "status": route[0].status if route else "unknown",
        }

    def search_trains(self, query: str) -> List[Dict]:
        """Search trains by number or name substring."""
        query_lower = query.lower()
        results = []
        for number, info in self._train_lookup.items():
            if query_lower in number.lower() or query_lower in info.get("name", "").lower():
                results.append({"train_number": number, **info})
        return results[:20]

    # ------------------------------------------------------------------
    # QUERY: STATION SCHEDULE
    # ------------------------------------------------------------------

    def get_station_timetable(self, station_code: str,
                               hour_start: int = 0, hour_end: int = 24) -> List[Dict]:
        """
        Get full timetable for a station within an hour range.
        Returns chronologically sorted schedule.
        """
        indices = self._by_station.get(station_code, [])
        entries = []

        for idx in indices:
            r = self._records[idx]
            # Check if any time falls in hour range
            for t_min in [r.arrival_minutes, r.departure_minutes]:
                if t_min is not None:
                    h = t_min // 60
                    if hour_start <= h < hour_end:
                        entries.append(r)
                        break

        entries.sort(key=lambda r: r.arrival_minutes or r.departure_minutes or 0)
        return [r.to_dict() for r in entries]

    # ------------------------------------------------------------------
    # ANALYTICS
    # ------------------------------------------------------------------

    def get_platform_occupancy(self, station_code: str, platform: int) -> Dict:
        """Compute hourly platform occupancy for a day."""
        key = (station_code, platform)
        indices = self._by_platform.get(key, [])

        hourly = {h: 0 for h in range(24)}
        for idx in indices:
            r = self._records[idx]
            arr = r.arrival_minutes
            dep = r.departure_minutes

            if arr is not None:
                hourly[arr // 60] += 1
            elif dep is not None:
                hourly[dep // 60] += 1

        peak_hour = max(hourly, key=hourly.get) if any(hourly.values()) else 0
        return {
            "station_code": station_code,
            "platform": platform,
            "total_trains": len(indices),
            "hourly_distribution": hourly,
            "peak_hour": peak_hour,
            "peak_trains": hourly[peak_hour],
        }

    def detect_platform_conflicts(self, station_code: str, platform: int,
                                   overlap_threshold_min: int = 5) -> List[Dict]:
        """Detect trains that overlap on the same platform."""
        key = (station_code, platform)
        indices = self._by_platform.get(key, [])
        records = [(i, self._records[i]) for i in indices]

        # Build time windows
        windows = []
        for idx, r in records:
            arr = r.arrival_minutes or r.departure_minutes
            dep = r.departure_minutes or r.arrival_minutes
            if arr is not None and dep is not None:
                windows.append((arr, dep, r))

        windows.sort(key=lambda w: w[0])

        conflicts = []
        for i in range(len(windows)):
            for j in range(i + 1, len(windows)):
                a_start, a_end, a_rec = windows[i]
                b_start, b_end, b_rec = windows[j]

                overlap = min(a_end, b_end) - max(a_start, b_start)
                if overlap >= overlap_threshold_min:
                    conflicts.append({
                        "train_a": {"number": a_rec.train_number, "name": a_rec.train_name,
                                    "arrival": a_rec.actual_arrival, "departure": a_rec.actual_departure},
                        "train_b": {"number": b_rec.train_number, "name": b_rec.train_name,
                                    "arrival": b_rec.actual_arrival, "departure": b_rec.actual_departure},
                        "overlap_minutes": overlap,
                        "platform": platform,
                    })

        return conflicts

    def get_delay_statistics(self, station_code: str = None) -> Dict:
        """Compute delay statistics, optionally filtered by station."""
        if station_code:
            indices = self._by_station.get(station_code, [])
            records = [self._records[i] for i in indices]
        else:
            records = self._records

        delays = [r.delay_minutes for r in records]
        delayed = [d for d in delays if d > 0]

        if not delays:
            return {"total_services": 0}

        return {
            "total_services": len(delays),
            "on_time_count": len(delays) - len(delayed),
            "delayed_count": len(delayed),
            "on_time_pct": round((len(delays) - len(delayed)) / len(delays) * 100, 1),
            "avg_delay_minutes": round(sum(delayed) / len(delayed), 1) if delayed else 0,
            "max_delay_minutes": max(delayed) if delayed else 0,
            "median_delay_minutes": sorted(delayed)[len(delayed)//2] if delayed else 0,
            "severe_delays": sum(1 for d in delayed if d > 60),
        }

    # ------------------------------------------------------------------
    # PERSISTENCE
    # ------------------------------------------------------------------

    def save(self, filepath: Path):
        """Export schedule database to JSON."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "version": "2.0",
            "created_at": datetime.now().isoformat(),
            "total_records": self.total_records,
            "total_trains": self.total_trains,
            "total_stations": self.total_stations,
            "records": [r.to_dict() for r in self._records],
        }
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)
        logger.info(f"ScheduleDB saved: {filepath} ({self.total_records} records)")

    def load(self, filepath: Path) -> int:
        """Import schedule database from JSON."""
        filepath = Path(filepath)
        if not filepath.exists():
            return 0

        with open(filepath) as f:
            data = json.load(f)

        for r in data.get("records", []):
            record = ScheduleRecord(
                train_number=r.get("train_number", ""),
                train_name=r.get("train_name", ""),
                train_type=r.get("train_type", ""),
                station_code=r.get("station_code", ""),
                station_name=r.get("station_name", ""),
                stop_sequence=r.get("stop_sequence", 0),
                scheduled_arrival=r.get("scheduled_arrival", "--"),
                scheduled_departure=r.get("scheduled_departure", "--"),
                actual_arrival=r.get("actual_arrival", ""),
                actual_departure=r.get("actual_departure", ""),
                platform_number=r.get("platform", 0),
                halt_minutes=r.get("halt_minutes", 0),
                delay_minutes=r.get("delay_minutes", 0),
                status=r.get("status", "on_time"),
                distance_from_origin_km=r.get("distance_km", 0),
                day=r.get("day", 1),
                is_origin=r.get("is_origin", False),
                is_destination=r.get("is_destination", False),
                coaches=r.get("coaches", []),
                total_coaches=r.get("total_coaches", 0),
            )
            self.insert(record)

        logger.info(f"ScheduleDB loaded: {self.total_records} records from {filepath}")
        return self.total_records

    # ------------------------------------------------------------------
    # HELPERS
    # ------------------------------------------------------------------

    @staticmethod
    def _time_distance(t_min: Optional[int], query_min: int) -> int:
        if t_min is None:
            return 9999
        diff = abs(t_min - query_min)
        return min(diff, 1440 - diff)
