"""
Digital Shield Rail Defense — Train Schedule Intelligence API
===============================================================
FastAPI service for train schedule queries, active train
inference, timestamp correlation, and delay analytics.

Endpoints:
    GET  /api/schedule/infer          — Infer train from platform+timestamp
    GET  /api/schedule/correlate      — Correlate ISO timestamp with schedules
    GET  /api/schedule/platform       — Platform intelligence context
    GET  /api/schedule/station        — Station timetable
    GET  /api/schedule/train/{number} — Train full route
    GET  /api/schedule/search         — Search trains by name/number
    GET  /api/schedule/delays         — Network delay report
    GET  /api/schedule/occupancy      — Platform occupancy analysis
    GET  /api/schedule/conflicts      — Platform conflict detection
    GET  /api/schedule/stats          — Schedule database statistics
"""

import sys
import logging
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    from fastapi import APIRouter, HTTPException, Query
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False

from backend.services.train_intelligence import TrainIntelligence

logger = logging.getLogger("train_api")

# ============================================================================
# SINGLETON ENGINE
# ============================================================================

_engine: Optional[TrainIntelligence] = None


def get_engine() -> TrainIntelligence:
    global _engine
    if _engine is None:
        _engine = TrainIntelligence()
        _engine.initialize()
    return _engine


# ============================================================================
# ROUTER
# ============================================================================

if FASTAPI_AVAILABLE:
    router = APIRouter(prefix="/api/schedule", tags=["Train Schedule Intelligence"])

    # ==================================================================
    # CORE: INFER TRAIN
    # ==================================================================

    @router.get("/infer")
    async def infer_train(
        station: str = Query(..., description="Station code (e.g., NDLS)"),
        platform: Optional[int] = Query(None, description="Platform number"),
        timestamp: str = Query(..., description="ISO timestamp or HH:MM"),
        tolerance: int = Query(15, description="Time tolerance in minutes"),
        day: Optional[str] = Query(None, description="Day of week (Mon, Tue, etc.)"),
    ):
        """
        Infer the active train at a platform from a timestamp.

        This is the core intelligence endpoint:
          platform + time → active train → coaches → passengers

        Examples:
        - `/api/schedule/infer?station=NDLS&platform=5&timestamp=08:30`
        - `/api/schedule/infer?station=SC&timestamp=2026-05-12T14:30:00`
        """
        engine = get_engine()
        result = engine.infer_train(
            station_code=station.upper(),
            platform=platform,
            timestamp=timestamp,
            tolerance_minutes=tolerance,
            day_of_week=day,
        )
        if result.get("status") == "error":
            raise HTTPException(400, detail=result.get("error"))
        return result

    # ==================================================================
    # TIMESTAMP CORRELATION
    # ==================================================================

    @router.get("/correlate")
    async def correlate_timestamp(
        station: str = Query(..., description="Station code"),
        timestamp: str = Query(..., description="Full ISO timestamp"),
    ):
        """
        Correlate an ISO timestamp with the railway schedule.

        Returns all trains that could have been at the station
        around that time, accounting for delays and running days.

        Example:
        - `/api/schedule/correlate?station=NDLS&timestamp=2026-05-12T08:30:00`
        """
        engine = get_engine()
        result = engine.correlate_timestamp(station.upper(), timestamp)
        if "error" in result:
            raise HTTPException(400, detail=result["error"])
        return result

    # ==================================================================
    # PLATFORM INTELLIGENCE
    # ==================================================================

    @router.get("/platform")
    async def platform_intelligence(
        station: str = Query(..., description="Station code"),
        platform: int = Query(..., description="Platform number"),
        timestamp: Optional[str] = Query(None, description="Time (HH:MM or ISO)"),
    ):
        """
        Get complete intelligence context for a platform.

        Returns train inference, occupancy, delays, conflicts,
        adjacent trains, and temporal context.
        """
        engine = get_engine()
        result = engine.get_platform_intelligence(
            station.upper(), platform, timestamp
        )
        if "error" in result:
            raise HTTPException(404, detail=result["error"])
        return result

    # ==================================================================
    # STATION TIMETABLE
    # ==================================================================

    @router.get("/station")
    async def station_timetable(
        station: str = Query(..., description="Station code"),
        hour_start: int = Query(0, ge=0, le=23, description="Start hour (0-23)"),
        hour_end: int = Query(24, ge=1, le=24, description="End hour (1-24)"),
    ):
        """
        Get full timetable for a station within a time window.

        Example:
        - `/api/schedule/station?station=NDLS&hour_start=6&hour_end=12`
        """
        engine = get_engine()
        return engine.get_station_schedule(station.upper(), hour_start, hour_end)

    # ==================================================================
    # TRAIN ROUTE
    # ==================================================================

    @router.get("/train/{train_number}")
    async def train_route(train_number: str):
        """
        Get complete route and schedule for a specific train.

        Example: `/api/schedule/train/12727`
        """
        engine = get_engine()
        result = engine.get_train_full_route(train_number)
        if "error" in result:
            raise HTTPException(404, detail=result["error"])
        return result

    # ==================================================================
    # SEARCH
    # ==================================================================

    @router.get("/search")
    async def search_trains(
        q: str = Query(..., description="Search query (train number or name)"),
    ):
        """
        Search trains by number or name.

        Example: `/api/schedule/search?q=Rajdhani`
        """
        engine = get_engine()
        return engine.search(q)

    # ==================================================================
    # DELAY ANALYTICS
    # ==================================================================

    @router.get("/delays")
    async def delay_report():
        """
        Get network-wide delay statistics.

        Returns overall stats and per-station breakdown
        sorted by average delay.
        """
        engine = get_engine()
        return engine.get_network_delay_report()

    # ==================================================================
    # PLATFORM OCCUPANCY
    # ==================================================================

    @router.get("/occupancy")
    async def platform_occupancy(
        station: str = Query(..., description="Station code"),
        platform: int = Query(..., description="Platform number"),
    ):
        """
        Get hourly platform occupancy for a full day.

        Example: `/api/schedule/occupancy?station=NDLS&platform=5`
        """
        engine = get_engine()
        return engine.schedule_db.get_platform_occupancy(station.upper(), platform)

    # ==================================================================
    # CONFLICT DETECTION
    # ==================================================================

    @router.get("/conflicts")
    async def platform_conflicts(
        station: str = Query(..., description="Station code"),
        platform: int = Query(..., description="Platform number"),
        threshold: int = Query(5, description="Overlap threshold in minutes"),
    ):
        """
        Detect trains that overlap on the same platform.

        Example: `/api/schedule/conflicts?station=NDLS&platform=5`
        """
        engine = get_engine()
        conflicts = engine.schedule_db.detect_platform_conflicts(
            station.upper(), platform, threshold
        )
        return {
            "station": station.upper(),
            "platform": platform,
            "threshold_minutes": threshold,
            "conflicts_found": len(conflicts),
            "conflicts": conflicts,
        }

    # ==================================================================
    # DATABASE STATS
    # ==================================================================

    @router.get("/stats")
    async def schedule_stats():
        """Get schedule database statistics."""
        engine = get_engine()
        delay_stats = engine.schedule_db.get_delay_statistics()

        return {
            "total_records": engine.schedule_db.total_records,
            "total_trains": engine.schedule_db.total_trains,
            "total_stations": engine.schedule_db.total_stations,
            "delay_statistics": delay_stats,
        }
