"""
Digital Shield Rail Defense -- Explainable Intelligence API
=============================================================
FastAPI endpoints for explainable alert generation, audit
logging, and intelligence queries.

Endpoints:
    POST /api/xai/generate-alert    - Generate explainable alert from CCTV event
    GET  /api/xai/audit-log         - Get audit trail
    GET  /api/xai/audit-stats       - Get audit statistics
    POST /api/xai/explain-scenario  - Explain a hypothetical scenario
    GET  /api/xai/pipeline-info     - Pipeline architecture info
"""

import sys
import logging
from pathlib import Path
from typing import Optional, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    from fastapi import APIRouter, HTTPException, Query
    from pydantic import BaseModel
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False

from backend.services.xai_engine import ExplainableIntelligenceEngine

logger = logging.getLogger("xai_api")

_engine: Optional[ExplainableIntelligenceEngine] = None

def get_engine() -> ExplainableIntelligenceEngine:
    global _engine
    if _engine is None:
        _engine = ExplainableIntelligenceEngine()
        _engine.initialize()
    return _engine


if FASTAPI_AVAILABLE:

    class AlertRequest(BaseModel):
        camera_id: str = "CAM_SC_P03_B"
        timestamp: str = "2026-05-12T14:30:00"
        anomaly_type: str = "suspicious_escort"
        anomaly_confidence: float = 0.82
        person_count: int = 2
        track_ids: Optional[List[int]] = None
        frame_number: int = 0
        video_source: str = ""
        station_code: Optional[str] = None
        platform: Optional[int] = None
        train_number: Optional[str] = None
        coach: Optional[str] = None

    class ScenarioRequest(BaseModel):
        anomaly_type: str = "suspicious_escort"
        anomaly_confidence: float = 0.75
        station_code: str = "SC"
        platform: int = 3
        timestamp: str = "2026-05-12T23:30:00"
        train_number: Optional[str] = None
        coach: Optional[str] = None

    router = APIRouter(prefix="/api/xai", tags=["Explainable Intelligence"])

    @router.post("/generate-alert")
    async def generate_alert(request: AlertRequest):
        """
        Generate a fully explainable intelligence alert.

        Runs the complete pipeline: CCTV -> camera -> train -> coach
        -> passengers -> anomaly scoring -> explainable alert.

        Returns alert with reasoning chain, confidence breakdown,
        triggered rules, and RPF action recommendation.
        """
        engine = get_engine()
        alert = engine.generate_alert(
            camera_id=request.camera_id,
            timestamp=request.timestamp,
            anomaly_type=request.anomaly_type,
            anomaly_confidence=request.anomaly_confidence,
            person_count=request.person_count,
            track_ids=request.track_ids,
            frame_number=request.frame_number,
            video_source=request.video_source,
            station_code=request.station_code,
            platform=request.platform,
            train_number=request.train_number,
            coach=request.coach,
        )
        return alert.to_dict()

    @router.post("/explain-scenario")
    async def explain_scenario(request: ScenarioRequest):
        """
        Explain a hypothetical scenario without specific CCTV data.

        Useful for training RPF officers and testing rule behavior.
        """
        engine = get_engine()
        camera_id = f"CAM_{request.station_code}_P{request.platform:02d}_B"
        alert = engine.generate_alert(
            camera_id=camera_id,
            timestamp=request.timestamp,
            anomaly_type=request.anomaly_type,
            anomaly_confidence=request.anomaly_confidence,
            station_code=request.station_code,
            platform=request.platform,
            train_number=request.train_number,
            coach=request.coach,
        )
        return {
            "scenario": {
                "anomaly": request.anomaly_type,
                "station": request.station_code,
                "platform": request.platform,
                "time": request.timestamp,
            },
            "alert": alert.to_dict(),
            "explanation": alert.explanation_summary,
        }

    @router.get("/audit-log")
    async def audit_log(limit: int = Query(50, le=200)):
        """Get the audit trail of generated alerts."""
        engine = get_engine()
        return {"entries": engine.get_audit_log(limit)}

    @router.get("/audit-stats")
    async def audit_stats():
        """Get aggregate audit statistics."""
        engine = get_engine()
        return engine.get_audit_stats()

    @router.get("/pipeline-info")
    async def pipeline_info():
        """Get pipeline architecture and subsystem info."""
        return {
            "pipeline_name": "Digital Shield Explainable Intelligence",
            "version": "1.0.0",
            "stages": [
                {"id": 1, "name": "CCTV Detection", "module": "YOLOv8 + DeepSORT", "output": "anomaly_type + confidence"},
                {"id": 2, "name": "Camera Resolution", "module": "CameraRegistry", "output": "station + platform + zone"},
                {"id": 3, "name": "Train Inference", "module": "TrainIntelligence", "output": "train_number + status"},
                {"id": 4, "name": "Coach Estimation", "module": "BogieMapper", "output": "coach + class"},
                {"id": 5, "name": "Passenger Narrowing", "module": "PassengerIntelligence", "output": "matching passengers"},
                {"id": 6, "name": "Booking Anomaly", "module": "BookingAnomalyScorer", "output": "risk scores + rules"},
                {"id": 7, "name": "Temporal Analysis", "module": "XAI Engine", "output": "night/peak modifier"},
                {"id": 8, "name": "Composite Confidence", "module": "XAI Engine", "output": "weighted score"},
                {"id": 9, "name": "Severity Classification", "module": "XAI Engine", "output": "LOW/MEDIUM/HIGH/CRITICAL"},
                {"id": 10, "name": "Explainable Report", "module": "XAI Engine", "output": "reasoning + action"},
            ],
            "confidence_weights": {
                "cctv_anomaly": "35%",
                "train_match": "15%",
                "coach_estimation": "10%",
                "booking_anomaly": "40%",
            },
            "severity_thresholds": {
                "CRITICAL": ">= 0.55 OR critical rule + 2 suspects",
                "HIGH": ">= 0.35 OR 2 suspects OR critical rule",
                "MEDIUM": ">= 0.20 OR 1 suspect",
                "LOW": "< 0.20",
            },
            "detection_rules": 7,
            "mitigating_rules": 4,
            "total_api_endpoints": 51,
        }
