"""
Digital Shield Rail Defense -- Fusion Intelligence API
========================================================
FastAPI endpoints for multi-source fusion intelligence.

Endpoints:
    POST /api/fusion/fuse-event      - Fuse a single event from all 5 sources
    POST /api/fusion/fuse-batch      - Fuse multiple events
    GET  /api/fusion/history         - Alert history
    GET  /api/fusion/stats           - Fusion statistics
    GET  /api/fusion/event-bus       - Event bus diagnostics
    GET  /api/fusion/architecture    - Fusion architecture info
"""

import sys, logging
from pathlib import Path
from typing import Optional, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    from fastapi import APIRouter, Query
    from pydantic import BaseModel
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False

from backend.services.fusion_engine import FusionEngine

logger = logging.getLogger("fusion_api")
_engine: Optional[FusionEngine] = None

def get_engine() -> FusionEngine:
    global _engine
    if _engine is None:
        _engine = FusionEngine()
        _engine.initialize()
    return _engine


if FASTAPI_AVAILABLE:

    class FuseRequest(BaseModel):
        camera_id: str = "CAM_SC_P03_B"
        timestamp: str = "2026-05-12T14:30:00"
        anomaly_type: str = "suspicious_escort"
        anomaly_confidence: float = 0.82
        person_count: int = 2
        station_code: Optional[str] = None
        platform: Optional[int] = None
        train_number: Optional[str] = None
        coach: Optional[str] = None

    class BatchRequest(BaseModel):
        events: List[FuseRequest]

    router = APIRouter(prefix="/api/fusion", tags=["Multi-Source Fusion"])

    @router.post("/fuse-event")
    async def fuse_event(request: FuseRequest):
        """Fuse a single event from all 5 intelligence sources."""
        engine = get_engine()
        alert = engine.fuse_event(**request.model_dump())
        return alert.to_dict()

    @router.post("/fuse-batch")
    async def fuse_batch(request: BatchRequest):
        """Fuse multiple events and return alerts sorted by confidence."""
        engine = get_engine()
        events = [e.model_dump() for e in request.events]
        alerts = engine.fuse_batch(events)
        return {
            "total": len(alerts),
            "alerts": [a.to_dict() for a in alerts],
        }

    @router.get("/history")
    async def alert_history(limit: int = Query(50, le=200)):
        """Get fusion alert history."""
        engine = get_engine()
        return {"alerts": engine.get_alert_history(limit)}

    @router.get("/stats")
    async def fusion_stats():
        """Get fusion engine statistics."""
        engine = get_engine()
        return engine.get_fusion_stats()

    @router.get("/event-bus")
    async def event_bus_info():
        """Get event bus diagnostics."""
        engine = get_engine()
        return engine.get_event_bus_stats()

    @router.get("/architecture")
    async def architecture():
        """Fusion architecture documentation."""
        return {
            "engine": "Multi-Source Fusion Intelligence",
            "version": "2.0.0",
            "sources": [
                {"id": 1, "name": "CCTV Anomaly", "weight": 0.46, "module": "YOLOv8 + DeepSORT"},
                {"id": 2, "name": "Railway Metadata", "weight": 0.15, "module": "CameraRegistry + StationDB"},
                {"id": 3, "name": "Train Intelligence", "weight": 0.23, "module": "TrainIntelligence + ScheduleDB"},
                {"id": 4, "name": "Coach Estimation", "weight": 0.15, "module": "BogieMapper + CoachOCR"},
            ],
            "fusion_method": "Dempster-Shafer belief combination + weighted linear blend",
            "formula": "fused = 0.6 * weighted_linear + 0.4 * DS_agreement",
            "event_architecture": "Async EventBus with topic-based pub/sub",
            "temporal_correlation": "Sliding 5-minute window incident clustering",
            "severity_thresholds": {
                "CRITICAL": ">= 0.55",
                "HIGH": ">= 0.35",
                "MEDIUM": ">= 0.20",
                "LOW": "< 0.20",
            },
        }
