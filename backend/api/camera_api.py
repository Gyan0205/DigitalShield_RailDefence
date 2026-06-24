"""
Digital Shield Rail Defense — Camera Intelligence FastAPI Service
==================================================================
RESTful API for the camera-to-platform intelligence mapping engine.

Endpoints:
    GET  /api/cameras/{camera_id}/resolve      — Resolve camera to platform
    GET  /api/cameras/{camera_id}               — Get camera details
    GET  /api/cameras                           — List all cameras (filtered)
    GET  /api/stations                          — List all stations
    GET  /api/stations/{code}                   — Get station details
    GET  /api/stations/{code}/coverage          — Camera coverage map
    GET  /api/stations/{code}/platforms/{num}   — Platform details
    GET  /api/platforms/map                     — Resolve camera → platform
    GET  /api/network/stats                     — Network-wide statistics
    GET  /api/network/health                    — All stations health
    GET  /api/trains/at-station                 — Trains at station/time
    POST /api/cameras/resolve-batch             — Batch camera resolution
    GET  /health                                — Service health check
"""

import sys
import logging
from pathlib import Path
from typing import Optional, List
from datetime import datetime

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    from fastapi import FastAPI, HTTPException, Query
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False

from backend.config import Settings
from backend.services.camera_registry import CameraRegistry
from backend.services.platform_mapper import PlatformMapper
from backend.services.railway_stations import STATIONS_DB, STATION_LOOKUP

logger = logging.getLogger("camera_api")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)-25s | %(message)s",
)

# ============================================================================
# APPLICATION FACTORY
# ============================================================================

settings = Settings()

# Initialize core services (singleton)
_mapper: Optional[PlatformMapper] = None


def get_mapper() -> PlatformMapper:
    global _mapper
    if _mapper is None:
        _mapper = PlatformMapper()
        _mapper.initialize()
    return _mapper


def create_app() -> "FastAPI":
    """Create and configure FastAPI application."""
    if not FASTAPI_AVAILABLE:
        raise ImportError("FastAPI not installed. Run: pip install fastapi uvicorn")

    app = FastAPI(
        title="Digital Shield — Camera Intelligence API",
        description="Railway CCTV camera-to-platform mapping and intelligence service",
        version=settings.app.app_version,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.server.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Mount train schedule router
    try:
        from backend.api.train_api import router as train_router, get_engine
        app.include_router(train_router)
        logger.info("Train Schedule Intelligence router mounted")
    except Exception as e:
        logger.warning(f"Train API router not loaded: {e}")

    # Mount coach intelligence router
    try:
        from backend.api.coach_api import router as coach_router
        app.include_router(coach_router)
        logger.info("Coach Intelligence router mounted")
    except Exception as e:
        logger.warning(f"Coach API router not loaded: {e}")



    # Mount explainable intelligence router
    try:
        from backend.api.xai_api import router as xai_router
        app.include_router(xai_router)
        logger.info("Explainable Intelligence router mounted")
    except Exception as e:
        logger.warning(f"XAI API router not loaded: {e}")

    # Mount multi-source fusion router
    try:
        from backend.api.fusion_api import router as fusion_router
        app.include_router(fusion_router)
        logger.info("Multi-Source Fusion router mounted")
    except Exception as e:
        logger.warning(f"Fusion API router not loaded: {e}")

    # Startup event
    @app.on_event("startup")
    async def startup():
        logger.info("Starting Digital Shield Intelligence API...")
        get_mapper()
        # Initialize train intelligence engine
        try:
            from backend.api.train_api import get_engine
            get_engine()
            logger.info("Train Intelligence Engine initialized")
        except Exception as e:
            logger.warning(f"Train engine init deferred: {e}")
        logger.info("API ready")

    # ======================================================================
    # HEALTH
    # ======================================================================

    @app.get("/health", tags=["System"])
    async def health_check():
        """Service health check."""
        mapper = get_mapper()
        health = {
            "status": "healthy",
            "service": "Digital Shield Intelligence API",
            "version": settings.app.app_version,
            "cameras_loaded": mapper.camera_registry.total_cameras,
            "stations": len(mapper.camera_registry.stations),
            "trains": len(mapper.simulator.trains),
            "timestamp": datetime.now().isoformat(),
        }
        try:
            from backend.api.train_api import get_engine
            engine = get_engine()
            health["schedule_records"] = engine.schedule_db.total_records
            health["schedule_trains"] = engine.schedule_db.total_trains
        except Exception:
            pass
        return health

    # ======================================================================
    # CAMERA RESOLUTION (core endpoint)
    # ======================================================================

    @app.get("/api/cameras/{camera_id}/resolve", tags=["Camera Intelligence"])
    async def resolve_camera(
        camera_id: str,
        station: Optional[str] = Query(None, description="Station code for short IDs"),
        timestamp: Optional[str] = Query(None, description="ISO timestamp or HH:MM"),
        tolerance: int = Query(15, description="Schedule tolerance in minutes"),
    ):
        """
        Resolve a camera ID to full platform intelligence.

        Examples:
        - `/api/cameras/CAM_NDLS_P05_B/resolve` → Full resolution
        - `/api/cameras/CAM_05/resolve?station=NDLS` → Short ID with context
        - `/api/cameras/CAM_NDLS_P05_B/resolve?timestamp=08:30` → With train lookup
        """
        mapper = get_mapper()
        result = mapper.resolve(camera_id, station_code=station,
                                timestamp=timestamp, tolerance_minutes=tolerance)
        if not result.get("resolved"):
            raise HTTPException(404, detail=result.get("error", "Camera not found"))
        return result

    @app.post("/api/cameras/resolve-batch", tags=["Camera Intelligence"])
    async def resolve_cameras_batch(
        camera_ids: List[str],
        station: Optional[str] = Query(None),
        timestamp: Optional[str] = Query(None),
    ):
        """Resolve multiple camera IDs in a single request."""
        mapper = get_mapper()
        results = mapper.resolve_batch(camera_ids, station_code=station, timestamp=timestamp)
        return {
            "total": len(results),
            "resolved": sum(1 for r in results if r.get("resolved")),
            "results": results,
        }

    # ======================================================================
    # CAMERA CRUD
    # ======================================================================

    @app.get("/api/cameras/{camera_id}", tags=["Cameras"])
    async def get_camera(camera_id: str):
        """Get camera details by ID."""
        mapper = get_mapper()
        cam = mapper.camera_registry.get(camera_id)
        if not cam:
            raise HTTPException(404, detail=f"Camera not found: {camera_id}")
        return cam.to_dict()

    @app.get("/api/cameras", tags=["Cameras"])
    async def list_cameras(
        station: Optional[str] = Query(None, description="Filter by station code"),
        platform: Optional[int] = Query(None, description="Filter by platform number"),
        zone: Optional[str] = Query(None, description="Filter by zone (entry/mid/exit)"),
        active_only: bool = Query(False, description="Only active cameras"),
        limit: int = Query(100, le=1000),
        offset: int = Query(0, ge=0),
    ):
        """List cameras with optional filters."""
        mapper = get_mapper()
        registry = mapper.camera_registry

        if station and platform:
            cameras = registry.get_by_platform(station, platform, active_only)
        elif station:
            cameras = registry.get_by_station(station, active_only)
        elif zone and station:
            cameras = registry.get_by_zone(station, zone)
        else:
            cameras = list(registry._cameras.values())
            if active_only:
                cameras = [c for c in cameras if c.is_active and c.is_online]

        if zone and not station:
            cameras = [c for c in cameras if c.zone == zone]

        total = len(cameras)
        cameras = cameras[offset:offset + limit]

        return {
            "total": total,
            "limit": limit,
            "offset": offset,
            "cameras": [c.to_dict() for c in cameras],
        }

    # ======================================================================
    # STATION ENDPOINTS
    # ======================================================================

    @app.get("/api/stations", tags=["Stations"])
    async def list_stations(
        zone: Optional[str] = Query(None, description="Filter by railway zone"),
        risk_tier: Optional[str] = Query(None, description="Filter by risk tier"),
    ):
        """List all railway stations."""
        stations = STATIONS_DB
        if zone:
            stations = [s for s in stations if zone.lower() in s.zone.lower()]
        if risk_tier:
            stations = [s for s in stations if s.risk_tier == risk_tier.upper()]

        return {
            "total": len(stations),
            "stations": [
                {
                    "code": s.code, "name": s.name, "city": s.city,
                    "state": s.state, "zone": s.zone, "platforms": s.platforms,
                    "daily_footfall": s.daily_footfall, "risk_tier": s.risk_tier,
                    "is_junction": s.is_junction,
                    "coordinates": {"lat": s.latitude, "lng": s.longitude},
                }
                for s in stations
            ],
        }

    @app.get("/api/stations/{code}", tags=["Stations"])
    async def get_station(code: str):
        """Get station details by code."""
        station = STATION_LOOKUP.get(code.upper())
        if not station:
            raise HTTPException(404, detail=f"Station not found: {code}")

        mapper = get_mapper()
        cameras = mapper.camera_registry.get_by_station(code.upper())
        active_cams = [c for c in cameras if c.is_active and c.is_online]

        return {
            "code": station.code, "name": station.name,
            "city": station.city, "state": station.state,
            "zone": station.zone, "division": station.division,
            "platforms": station.platforms,
            "is_junction": station.is_junction,
            "is_terminus": station.is_terminus,
            "daily_footfall": station.daily_footfall,
            "risk_tier": station.risk_tier,
            "coordinates": {"lat": station.latitude, "lng": station.longitude},
            "camera_count": len(cameras),
            "active_cameras": len(active_cams),
            "camera_health_pct": round(len(active_cams) / len(cameras) * 100, 1) if cameras else 0,
        }

    @app.get("/api/stations/{code}/coverage", tags=["Stations"])
    async def get_station_coverage(code: str):
        """Get complete camera coverage map for a station."""
        mapper = get_mapper()
        result = mapper.get_station_coverage_map(code.upper())
        if "error" in result:
            raise HTTPException(404, detail=result["error"])
        return result

    @app.get("/api/stations/{code}/platforms/{platform}", tags=["Stations"])
    async def get_platform_detail(
        code: str, platform: int,
        timestamp: Optional[str] = Query(None),
    ):
        """Get detailed platform information including cameras and trains."""
        mapper = get_mapper()
        station = STATION_LOOKUP.get(code.upper())
        if not station:
            raise HTTPException(404, detail=f"Station not found: {code}")
        if platform < 1 or platform > station.platforms:
            raise HTTPException(400, detail=f"Platform {platform} does not exist (max: {station.platforms})")

        cameras = mapper.camera_registry.get_by_platform(code.upper(), platform)
        active = [c for c in cameras if c.is_active and c.is_online]

        result = {
            "station_code": code.upper(),
            "station_name": station.name,
            "platform": platform,
            "cameras": [
                {"camera_id": c.camera_id, "zone": c.zone, "type": c.camera_type,
                 "active": c.is_active, "online": c.is_online,
                 "coaches_visible": f"{c.visible_coaches_start}-{c.visible_coaches_end}"}
                for c in cameras
            ],
            "active_cameras": len(active),
            "full_coverage": len(set(c.zone for c in active)) >= 3,
            "zones_covered": sorted(set(c.zone for c in active)),
        }

        # Add train info if timestamp provided
        if timestamp:
            time_str = mapper._extract_time(timestamp)
            if time_str:
                schedules = mapper.simulator.find_train_at_station(code.upper(), time_str)
                platform_trains = [s for s in schedules if s.platform_number == platform]
                result["trains"] = [
                    {
                        "train_number": s.train_number, "train_name": s.train_name,
                        "arrival": s.arrival_time, "departure": s.departure_time,
                        "halt_min": s.halt_minutes,
                    }
                    for s in platform_trains
                ]

        return result

    # ======================================================================
    # NETWORK INTELLIGENCE
    # ======================================================================

    @app.get("/api/network/stats", tags=["Network"])
    async def network_statistics():
        """Get network-wide camera statistics."""
        mapper = get_mapper()
        return mapper.camera_registry.get_network_stats()

    @app.get("/api/network/health", tags=["Network"])
    async def network_health():
        """Get health status for all stations."""
        mapper = get_mapper()
        statuses = mapper.get_all_stations_status()
        healthy = sum(1 for s in statuses if s["health_pct"] >= 90)
        degraded = sum(1 for s in statuses if 50 <= s["health_pct"] < 90)
        critical = sum(1 for s in statuses if s["health_pct"] < 50)

        return {
            "overall_health": "healthy" if critical == 0 else ("degraded" if critical < 3 else "critical"),
            "stations_healthy": healthy,
            "stations_degraded": degraded,
            "stations_critical": critical,
            "total_stations": len(statuses),
            "stations": statuses,
        }

    # ======================================================================
    # TRAIN ENDPOINTS
    # ======================================================================

    @app.get("/api/trains/at-station", tags=["Trains"])
    async def trains_at_station(
        station: str = Query(..., description="Station code"),
        time: str = Query(..., description="Time in HH:MM format"),
        tolerance: int = Query(15, description="Tolerance in minutes"),
    ):
        """Find trains at a station around a given time."""
        mapper = get_mapper()
        matches = mapper.simulator.find_train_at_station(
            station.upper(), time, tolerance
        )
        return {
            "station": station.upper(),
            "query_time": time,
            "tolerance_minutes": tolerance,
            "total_matches": len(matches),
            "trains": [
                {
                    "train_number": m.train_number, "train_name": m.train_name,
                    "arrival": m.arrival_time, "departure": m.departure_time,
                    "platform": m.platform_number, "halt_min": m.halt_minutes,
                    "stop_sequence": m.stop_sequence,
                }
                for m in matches
            ],
        }

    # ======================================================================
    # PLATFORM MAPPING (shorthand)
    # ======================================================================

    @app.get("/api/platforms/map", tags=["Camera Intelligence"])
    async def map_camera_to_platform(
        camera_id: str = Query(..., description="Camera ID (full or short)"),
        station: Optional[str] = Query(None, description="Station code"),
    ):
        """
        Quick camera → platform mapping.

        Example: `/api/platforms/map?camera_id=CAM_05&station=SC`
        Returns: `{ "platform": 5, "station": "Secunderabad Junction" }`
        """
        mapper = get_mapper()
        result = mapper.resolve(camera_id, station_code=station.upper() if station else None)

        if not result.get("resolved"):
            raise HTTPException(404, detail=result.get("error", "Cannot resolve camera"))

        return {
            "camera_id": camera_id,
            "platform": result.get("platform"),
            "station_code": result.get("station", {}).get("code"),
            "station_name": result.get("station", {}).get("name"),
            "zone": result.get("camera", {}).get("zone"),
            "resolved": True,
        }

    @app.get("/api/config", tags=["System"])
    async def get_config():
        """Get current service configuration."""
        return settings.to_dict()

    return app


# ============================================================================
# APPLICATION INSTANCE
# ============================================================================

if FASTAPI_AVAILABLE:
    app = create_app()


# ============================================================================
# CLI ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    if not FASTAPI_AVAILABLE:
        print("FastAPI not installed. Run: pip install fastapi uvicorn")
        sys.exit(1)

    import uvicorn
    uvicorn.run(
        "backend.api.camera_api:app",
        host=settings.server.host,
        port=settings.server.port,
        reload=settings.server.reload,
    )
