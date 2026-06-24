"""
Digital Shield Rail Defense — Synthetic Railway Simulator
==========================================================
Simulates the complete Indian railway operational ecosystem:
  - Train generation with realistic numbering schemes
  - Schedule generation with proper arrival/departure times
  - Coach composition following IRCTC standards
  - Passenger flow simulation
  - Camera placement and visibility mapping

This simulator creates the "ground truth" railway context
that public anomaly datasets lack, enabling the ML pipeline
to correlate detected anomalies with operational intelligence.
"""

import random
import hashlib
import logging
from typing import List, Dict, Tuple, Optional
from datetime import datetime, timedelta

from backend.services.railway_schema import (
    StationSchema, TrainSchema, ScheduleEntry, CameraSchema,
)
from backend.services.railway_stations import STATIONS_DB, STATION_LOOKUP

logger = logging.getLogger("railway_simulator")

# ============================================================================
# CONSTANTS
# ============================================================================

TRAIN_TYPES = [
    "Rajdhani Express", "Shatabdi Express", "Duronto Express",
    "Garib Rath Express", "Superfast Express", "Express", "Mail",
    "Jan Shatabdi Express", "Humsafar Express", "Tejas Express",
    "Vande Bharat Express", "Sampark Kranti Express", "Passenger",
]

# Coach composition templates by train type
COACH_TEMPLATES = {
    "Rajdhani Express": {"1A": 1, "2A": 2, "3A": 5, "PC": 1, "SLR": 2, "EOG": 2},
    "Shatabdi Express": {"CC": 8, "EC": 2, "PC": 1, "EOG": 2},
    "Duronto Express": {"1A": 1, "2A": 2, "3A": 4, "SL": 4, "PC": 1, "SLR": 2, "EOG": 2},
    "Vande Bharat Express": {"EC": 4, "CC": 12},
    "Tejas Express": {"EC": 4, "CC": 8, "EOG": 2},
    "Garib Rath Express": {"3A": 10, "SLR": 2, "EOG": 2},
    "Superfast Express": {"1A": 1, "2A": 2, "3A": 4, "SL": 8, "GN": 2, "SLR": 2},
    "Express": {"2A": 1, "3A": 3, "SL": 10, "GN": 3, "SLR": 2},
    "Mail": {"2A": 1, "3A": 2, "SL": 8, "GN": 4, "SLR": 2},
    "Jan Shatabdi Express": {"CC": 4, "2S": 6, "GN": 2, "SLR": 2},
    "Humsafar Express": {"3A": 12, "SLR": 2, "EOG": 2},
    "Sampark Kranti Express": {"2A": 1, "3A": 3, "SL": 8, "GN": 2, "SLR": 2},
    "Passenger": {"GN": 8, "SLR": 2},
}

# Famous Indian train name patterns
ROUTE_NAMES = [
    "Rajdhani", "Shatabdi", "Duronto", "Garib Rath", "Jan Shatabdi",
    "Sampark Kranti", "Humsafar", "Tejas", "Vande Bharat",
    "Superfast", "Express", "Mail", "Special",
    "Intercity", "Link", "Premium", "Antyodaya",
]

# Time distribution weights: more trains during peak hours
HOUR_WEIGHTS = [
    1, 0, 0, 0, 2, 5, 8, 12, 15, 12, 8, 6,  # 00-11
    6, 5, 6, 8, 12, 15, 12, 10, 8, 5, 3, 2,  # 12-23
]


class RailwaySimulator:
    """
    Simulates the complete Indian railway operational ecosystem.

    Generates realistic trains, schedules, cameras, and operational
    metadata that maps to CCTV surveillance footage timestamps.

    Usage:
        sim = RailwaySimulator(seed=42)
        trains = sim.generate_trains(count=300)
        schedules = sim.generate_schedules()
        cameras = sim.generate_camera_network()
        metadata = sim.assign_video_metadata(video_path, timestamp)
    """

    def __init__(self, seed: int = 42):
        self.seed = seed
        random.seed(seed)
        self.stations = STATIONS_DB
        self.trains: List[TrainSchema] = []
        self.schedules: List[ScheduleEntry] = []
        self.cameras: List[CameraSchema] = []

    # ========================================================================
    # TRAIN GENERATION
    # ========================================================================

    def generate_trains(self, count: int = 300) -> List[TrainSchema]:
        """
        Generate realistic Indian railway trains.

        Uses proper numbering:
          - 10000-19999: Superfast/Mail/Express
          - 20000-29999: Rajdhani/Shatabdi/Duronto
          - 30000-49999: Regular Express
          - 50000-59999: Passenger
          - 60000-69999: MEMU/DEMU
        """
        self.trains = []
        used_numbers = set()
        city_names = [s.city for s in self.stations]

        for i in range(count):
            train_type = random.choice(TRAIN_TYPES)

            # Generate number based on type
            if "Rajdhani" in train_type or "Shatabdi" in train_type or "Duronto" in train_type:
                base = random.randint(20001, 22999)
            elif "Vande Bharat" in train_type or "Tejas" in train_type:
                base = random.randint(22001, 22999)
            elif "Superfast" in train_type:
                base = random.randint(12001, 12999)
            elif "Passenger" in train_type:
                base = random.randint(51001, 59999)
            else:
                base = random.randint(10001, 19999)

            # Ensure even/odd pair (down/up)
            train_number = base if base % 2 == 1 else base + 1
            while train_number in used_numbers:
                train_number += 2
            used_numbers.add(train_number)

            # Select origin and destination
            origin_station = random.choice(self.stations)
            dest_candidates = [s for s in self.stations if s.code != origin_station.code]
            dest_station = random.choice(dest_candidates)

            # Generate coaches
            template = COACH_TEMPLATES.get(train_type, COACH_TEMPLATES["Express"])
            coaches = []
            for coach_type, qty in template.items():
                if coach_type in ("EOG", "SLR", "PC"):
                    coaches.extend([coach_type] * qty)
                else:
                    for j in range(1, qty + 1):
                        coaches.append(f"{coach_type}{j}" if qty > 1 else coach_type)

            # Build train name
            prefix = random.choice(ROUTE_NAMES) if random.random() > 0.3 else ""
            train_name = f"{origin_station.city}-{dest_station.city} {prefix} {train_type}".strip()

            # Speed and distance
            speed_map = {"Rajdhani Express": 90, "Shatabdi Express": 85, "Vande Bharat Express": 100,
                         "Duronto Express": 80, "Superfast Express": 70, "Express": 55, "Passenger": 35}
            avg_speed = speed_map.get(train_type, 55) + random.randint(-5, 10)
            distance = random.randint(200, 2500)

            train = TrainSchema(
                number=str(train_number),
                name=train_name,
                train_type=train_type,
                origin_code=origin_station.code,
                origin_name=origin_station.city,
                destination_code=dest_station.code,
                destination_name=dest_station.city,
                coaches=coaches,
                total_coaches=len(coaches),
                runs_on=self._generate_running_days(train_type),
                avg_speed_kmph=avg_speed,
                total_distance_km=distance,
                pantry_car="PC" in coaches,
            )
            self.trains.append(train)

        logger.info(f"Generated {len(self.trains)} trains")
        return self.trains

    def _generate_running_days(self, train_type: str) -> List[str]:
        """Generate running days based on train type."""
        all_days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        if train_type in ("Rajdhani Express", "Shatabdi Express", "Vande Bharat Express"):
            return all_days  # Daily
        elif train_type == "Passenger":
            return all_days  # Daily
        elif random.random() > 0.6:
            return all_days  # 60% trains run daily
        else:
            return sorted(random.sample(all_days, random.randint(3, 5)))

    # ========================================================================
    # SCHEDULE GENERATION
    # ========================================================================

    def generate_schedules(self) -> List[ScheduleEntry]:
        """
        Generate realistic train schedules with proper
        arrival/departure times at intermediate stations.
        """
        if not self.trains:
            self.generate_trains()

        self.schedules = []

        for train in self.trains:
            # Determine number of stops
            min_stops = 2 if "Duronto" in train.train_type else 3
            max_stops = 5 if "Rajdhani" in train.train_type else min(15, len(self.stations))
            num_stops = random.randint(min_stops, max_stops)

            # Select route stations
            origin = STATION_LOOKUP.get(train.origin_code)
            destination = STATION_LOOKUP.get(train.destination_code)
            if not origin or not destination:
                continue

            intermediate = [s for s in self.stations if s.code not in (train.origin_code, train.destination_code)]
            random.shuffle(intermediate)
            route = [origin] + intermediate[:num_stops - 2] + [destination]

            # Generate departure time from origin
            hour = random.choices(range(24), weights=HOUR_WEIGHTS, k=1)[0]
            minute = random.choice([0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55])
            current_time = datetime(2026, 5, 1, hour, minute)
            cumulative_distance = 0
            day = 1

            for stop_idx, station in enumerate(route):
                is_origin = stop_idx == 0
                is_destination = stop_idx == len(route) - 1

                # Travel time to this station
                if stop_idx > 0:
                    leg_distance = random.randint(50, 350)
                    cumulative_distance += leg_distance
                    travel_hours = leg_distance / train.avg_speed_kmph
                    current_time += timedelta(hours=travel_hours)

                # Arrival and departure
                arrival = current_time
                if is_origin:
                    halt = 0  # No halt at origin (departure only)
                elif is_destination:
                    halt = 0  # No halt at destination (arrival only)
                elif station.is_junction:
                    halt = random.randint(5, 20)
                else:
                    halt = random.randint(2, 10)

                departure = arrival + timedelta(minutes=halt)

                # Day calculation
                if arrival.hour < hour and stop_idx > 0:
                    day = 2

                # Platform assignment (higher traffic stations get better platforms)
                if station.is_terminus or station.is_junction:
                    platform = random.randint(1, min(5, station.platforms))
                else:
                    platform = random.randint(1, station.platforms)

                entry = ScheduleEntry(
                    train_number=train.number,
                    train_name=train.name,
                    station_code=station.code,
                    station_name=station.name,
                    stop_sequence=stop_idx + 1,
                    arrival_time="--" if is_origin else arrival.strftime("%H:%M"),
                    departure_time="--" if is_destination else departure.strftime("%H:%M"),
                    platform_number=platform,
                    halt_minutes=halt,
                    distance_from_origin_km=cumulative_distance,
                    day=day,
                )
                self.schedules.append(entry)

                current_time = departure

        logger.info(f"Generated {len(self.schedules)} schedule entries for {len(self.trains)} trains")
        return self.schedules

    # ========================================================================
    # CAMERA NETWORK
    # ========================================================================

    def generate_camera_network(self) -> List[CameraSchema]:
        """
        Generate complete CCTV camera network across all stations.

        Each platform gets 3 cameras:
          - Zone A (entry): covers coach positions 1-4
          - Zone B (mid): covers coach positions 5-12
          - Zone C (exit): covers coach positions 13-24
        """
        self.cameras = []
        zone_config = {
            "A": {"name": "entry", "coach_start": 1, "coach_end": 4, "type": "PTZ"},
            "B": {"name": "mid", "coach_start": 5, "coach_end": 12, "type": "PTZ"},
            "C": {"name": "exit", "coach_start": 13, "coach_end": 24, "type": "Fixed"},
        }

        for station in self.stations:
            for platform in range(1, station.platforms + 1):
                for zone_code, zone_info in zone_config.items():
                    camera_id = f"CAM_{station.code}_P{platform:02d}_{zone_code}"
                    camera = CameraSchema(
                        camera_id=camera_id,
                        station_code=station.code,
                        station_name=station.name,
                        platform_number=platform,
                        zone=zone_info["name"],
                        camera_type=zone_info["type"],
                        resolution="1080p" if station.daily_footfall > 100000 else "720p",
                        is_active=random.random() > 0.03,  # 97% uptime
                        coverage_area_sqm=random.uniform(150, 300),
                        visible_coaches_start=zone_info["coach_start"],
                        visible_coaches_end=zone_info["coach_end"],
                    )
                    self.cameras.append(camera)

        logger.info(f"Generated {len(self.cameras)} cameras across {len(self.stations)} stations")
        return self.cameras

    # ========================================================================
    # OPERATIONAL QUERIES
    # ========================================================================

    def find_train_at_station(self, station_code: str, time_str: str,
                               tolerance_minutes: int = 15) -> List[ScheduleEntry]:
        """
        Find all trains at a station around a given time.

        Args:
            station_code: Station code (e.g., "NDLS")
            time_str: Time in "HH:MM" format
            tolerance_minutes: +/- minutes to search

        Returns:
            List of matching schedule entries
        """
        try:
            query_hour, query_min = map(int, time_str.split(":"))
            query_minutes = query_hour * 60 + query_min
        except ValueError:
            return []

        matches = []
        for entry in self.schedules:
            if entry.station_code != station_code:
                continue

            for time_val in [entry.arrival_time, entry.departure_time]:
                if time_val == "--":
                    continue
                try:
                    h, m = map(int, time_val.split(":"))
                    entry_minutes = h * 60 + m
                    if abs(entry_minutes - query_minutes) <= tolerance_minutes:
                        matches.append(entry)
                        break
                except ValueError:
                    continue

        return matches

    def find_cameras_for_coach(self, station_code: str, platform: int,
                                coach_position: int) -> List[CameraSchema]:
        """Find cameras that can see a specific coach position."""
        return [
            c for c in self.cameras
            if c.station_code == station_code
            and c.platform_number == platform
            and c.visible_coaches_start <= coach_position <= c.visible_coaches_end
            and c.is_active
        ]

    def get_station_by_code(self, code: str) -> Optional[StationSchema]:
        """Lookup a station by its code."""
        return STATION_LOOKUP.get(code)

    def get_trains_on_route(self, origin_code: str, dest_code: str) -> List[TrainSchema]:
        """Find all trains running between two stations."""
        return [
            t for t in self.trains
            if t.origin_code == origin_code and t.destination_code == dest_code
        ]

    def get_platform_activity(self, station_code: str, platform: int,
                               hour: int) -> Dict:
        """Get activity level for a specific platform at a given hour."""
        trains_at_platform = [
            e for e in self.schedules
            if e.station_code == station_code and e.platform_number == platform
        ]

        # Count trains in the hour
        count = 0
        for e in trains_at_platform:
            for t in [e.arrival_time, e.departure_time]:
                if t == "--":
                    continue
                try:
                    h = int(t.split(":")[0])
                    if h == hour:
                        count += 1
                        break
                except ValueError:
                    continue

        return {
            "station_code": station_code,
            "platform": platform,
            "hour": hour,
            "train_count": count,
            "activity_level": "high" if count > 4 else ("medium" if count > 2 else "low"),
        }

    # ========================================================================
    # SERIALIZATION
    # ========================================================================

    def to_dict(self) -> Dict:
        """Serialize entire simulator state."""
        return {
            "version": "2.0",
            "created_at": datetime.now().isoformat(),
            "stations": {
                "count": len(self.stations),
                "data": [
                    {
                        "code": s.code, "name": s.name, "city": s.city,
                        "state": s.state, "zone": s.zone, "division": s.division,
                        "latitude": s.latitude, "longitude": s.longitude,
                        "platforms": s.platforms, "is_junction": s.is_junction,
                        "daily_footfall": s.daily_footfall, "risk_tier": s.risk_tier,
                    }
                    for s in self.stations
                ],
            },
            "trains": {
                "count": len(self.trains),
                "data": [
                    {
                        "number": t.number, "name": t.name, "type": t.train_type,
                        "origin": t.origin_name, "destination": t.destination_name,
                        "coaches": t.coaches, "total_coaches": t.total_coaches,
                        "runs_on": t.runs_on, "avg_speed": t.avg_speed_kmph,
                        "distance_km": t.total_distance_km,
                    }
                    for t in self.trains
                ],
            },
            "schedules": {
                "count": len(self.schedules),
                "data": [
                    {
                        "train_number": e.train_number, "train_name": e.train_name,
                        "station_code": e.station_code, "station_name": e.station_name,
                        "stop": e.stop_sequence, "arrival": e.arrival_time,
                        "departure": e.departure_time, "platform": e.platform_number,
                        "halt_min": e.halt_minutes, "distance_km": e.distance_from_origin_km,
                        "day": e.day,
                    }
                    for e in self.schedules
                ],
            },
            "cameras": {
                "count": len(self.cameras),
                "data": [
                    {
                        "camera_id": c.camera_id, "station": c.station_code,
                        "platform": c.platform_number, "zone": c.zone,
                        "type": c.camera_type, "resolution": c.resolution,
                        "active": c.is_active,
                        "coaches_visible": f"{c.visible_coaches_start}-{c.visible_coaches_end}",
                    }
                    for c in self.cameras
                ],
            },
        }
