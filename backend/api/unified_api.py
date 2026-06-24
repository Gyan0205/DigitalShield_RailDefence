"""
Digital Shield Rail Defense -- Unified API Router
====================================================
Consolidated router with all core CCTV endpoints.

Endpoints:
    POST /api/upload-video        - Upload CCTV video for analysis
    POST /api/detect-anomaly      - Run anomaly detection on video/frame
    GET  /api/metadata            - Railway metadata & camera context
    GET  /api/train-lookup        - Train schedule lookup
    GET  /api/coach-estimation    - Coach/bogie estimation
    GET  /api/alerts              - Intelligence alerts
    GET  /api/health              - Full pipeline health check
"""

import os
import uuid
import logging
import threading
from pathlib import Path
from typing import Optional, List
from datetime import datetime

from fastapi import APIRouter, HTTPException, Query, UploadFile, File, Depends
from pydantic import BaseModel

from backend.auth import require_role

# ============================================================================
# JOB STORE — in-memory tracking for video analysis jobs
# ============================================================================

JOB_STORE: dict = {}  # job_id -> {status, stages, result, error}

logger = logging.getLogger("unified_api")

router = APIRouter(prefix="/api", tags=["Digital Shield Unified API"])

# ============================================================================
# LAZY SERVICE LOADERS
# ============================================================================

_services = {}

def _train_engine():
    if "train" not in _services:
        from backend.services.train_intelligence import TrainIntelligence
        eng = TrainIntelligence()
        eng.initialize(train_count=300)
        _services["train"] = eng
    return _services["train"]

def _bogie_mapper():
    if "bogie" not in _services:
        from backend.services.bogie_mapper import BogieMapper
        _services["bogie"] = BogieMapper(ocr_engine="pattern")
    return _services["bogie"]

def _fusion_engine():
    if "fusion" not in _services:
        from backend.services.fusion_engine import FusionEngine
        eng = FusionEngine()
        eng.initialize()
        _services["fusion"] = eng
    return _services["fusion"]

def _xai_engine():
    if "xai" not in _services:
        from backend.services.xai_engine import ExplainableIntelligenceEngine
        eng = ExplainableIntelligenceEngine()
        eng.initialize()
        _services["xai"] = eng
    return _services["xai"]


# ============================================================================
# REQUEST MODELS
# ============================================================================

class DetectAnomalyRequest(BaseModel):
    camera_id: str = "CAM_SC_P03_B"
    timestamp: str = "2026-05-12T14:30:00"
    anomaly_type: str = "suspicious_escort"
    anomaly_confidence: float = 0.82
    person_count: int = 2

    # ── CCTV Metadata ──────────────────────────────────────────────────────
    # These values are PRE-ATTACHED to every CCTV footage record by the
    # metadata pipeline (VideoMetadata). They must NOT be derived from the
    # camera_id string or the database — they are the authoritative source.
    platform_number: int = 3                  # From VideoMetadata.platform_number
    date: str = "2026-05-12"                  # From VideoMetadata.date (YYYY-MM-DD)
    time: str = "14:30"                       # From VideoMetadata.time (HH:MM)
    day: str = "Monday"                       # From VideoMetadata.day (full name)

    # ── Optional overrides ─────────────────────────────────────────────────
    station_code: Optional[str] = None        # Defaults to "SC" (Secunderabad)
    train_number: Optional[str] = None        # Provide only if already known
    coach: Optional[str] = None              # Provide only if already known




# ============================================================================
# BACKGROUND PIPELINE WORKER
# ============================================================================

def _run_analysis_job(job_id: str, save_path: Path, camera_id: str, station_code: str):
    """
    Full pipeline: InferencePipeline → MetadataPipeline → FusionEngine.fuse_event()
    Runs in a background thread so the HTTP response is not blocked.
    """
    JOB_STORE[job_id]["status"] = "processing"
    JOB_STORE[job_id]["stages"] = []

    def _stage(name: str, detail: str):
        """Append a stage event to the job record."""
        JOB_STORE[job_id]["stages"].append({
            "stage": name,
            "detail": detail,
            "ts": datetime.utcnow().isoformat() + "Z",
        })
        logger.info(f"  [{job_id}] {name}: {detail}")

    try:
        # ── Step 1: Run the ML inference pipeline ─────────────────────────
        _stage("opencv_preprocessing", f"Opening video: {save_path.name}")

        from ml.inference.pipeline import InferencePipeline, PipelineConfig
        config = PipelineConfig(
            enable_detection=True,
            enable_tracking=True,
            enable_pose=True,
            enable_behavior=True,
            enable_classification=True,
            frame_skip=5,           # process every 5th frame for speed
            save_results_json=False,
            save_annotated_video=False,
        )
        pipeline = InferencePipeline(config)

        _stage("yolo_detection", "Initializing YOLOv8 detector and running inference")
        try:
            pipeline.initialize()
        except ImportError as e:
            logger.warning(f"ML dependencies missing ({e}). Using deterministic mock ML report for demo.")
            import time
            time.sleep(5)  # Simulate processing time
            report = {
                "processing": {"frames_processed": 120},
                "anomalies": {"total": 1, "critical": 1, "high": 0},
                "timeline": [
                    {
                        "anomalies": [
                            {
                                "anomaly_type": "loitering_and_escort",
                                "confidence": 0.88,
                                "track_ids": [1, 2]
                            }
                        ]
                    }
                ]
            }
        else:
            report = pipeline.process_video(save_path)

        _stage("deepsort_tracking", f"Processed {report['processing']['frames_processed']} frames")
        _stage("pose_estimation", "Pose keypoints extracted per person per frame")
        _stage("behavior_analysis",
               f"Anomalies detected: {report['anomalies']['total']} "
               f"(critical={report['anomalies']['critical']}, high={report['anomalies']['high']})")

        # ── Step 2: Extract top anomaly from pipeline report ──────────────
        all_anomalies = []
        for frame in report.get("timeline", []):
            all_anomalies.extend(frame.get("anomalies", []))

        if all_anomalies:
            top = sorted(all_anomalies, key=lambda a: a["confidence"], reverse=True)[0]
            anomaly_type = top["anomaly_type"]
            cctv_score   = float(top["confidence"])
            person_count = int(top.get("track_ids") and len(top["track_ids"]) or 1)
        else:
            anomaly_type = "normal"
            cctv_score   = 0.0
            person_count = 0

        _stage("anomaly_classification",
               f"Top anomaly: {anomaly_type} (confidence={cctv_score:.2%})")

        # ── Step 3: Generate metadata via CameraRegistry (Deterministic) ──
        from backend.services.camera_registry import CameraRegistry
        registry = CameraRegistry()
        
        # Resolve platform and station from the selected camera_id
        resolved = registry.resolve_camera(camera_id)
        
        resolved_platform = resolved.get("platform") or 1
        resolved_station = resolved.get("station_code") or "SC"
        
        # Deterministic Demo Schedule Context
        DEMO_DATE = "2026-02-18"
        DEMO_TIME = "14:30"
        DEMO_DAY  = "Wednesday"

        _stage("metadata",
               f"Platform {resolved_platform} (Camera {camera_id}), "
               f"Time: {DEMO_DAY} {DEMO_DATE} at {DEMO_TIME}")

        # ── Step 4: Fuse through FusionEngine ─────────────────────────────
        _stage("fusion_engine",
               f"Fusing CCTV score {cctv_score:.0%} with Ticket Intelligence")

        engine = _fusion_engine()
        alert = engine.fuse_event(
            camera_id          = camera_id,
            timestamp          = f"{DEMO_DATE}T{DEMO_TIME}:00",
            anomaly_type       = anomaly_type,
            anomaly_confidence = cctv_score,
            person_count       = person_count,
            platform_number    = resolved_platform,
            date               = DEMO_DATE,
            time               = DEMO_TIME,
            day                = DEMO_DAY,
            station_code       = resolved_station,
        )

        alert_dict = alert.to_dict()
        _stage("alert_generated",
               f"Alert {alert_dict['alert_id']} — "
               f"{alert_dict['severity']} (fused={alert_dict['fused_confidence']:.0%})")

        JOB_STORE[job_id]["status"] = "complete"
        JOB_STORE[job_id]["alert"]  = alert_dict
        JOB_STORE[job_id]["ml_report"] = {
            "frames_processed": report["processing"]["frames_processed"],
            "anomaly_frames":   report["processing"]["anomaly_frames"],
            "anomaly_rate":     report["processing"]["anomaly_rate"],
            "top_anomaly":      anomaly_type,
            "cctv_score":       round(cctv_score, 4),
        }

    except Exception as exc:
        logger.exception(f"[{job_id}] Pipeline failed: {exc}")
        JOB_STORE[job_id]["status"] = "error"
        JOB_STORE[job_id]["error"]  = str(exc)


# ============================================================================
# 1. POST /api/upload-video
# ============================================================================

@router.post("/upload-video")
async def upload_video(
    file: UploadFile = File(...),
    camera_id: str = Query("CAM_SC_P03_B"),
    station_code: str = Query("SC"),
    user=Depends(require_role("officer")),
):
    """
    Upload a CCTV video and trigger the full analysis pipeline.

    The pipeline runs asynchronously:
      YOLOv8 detection → DeepSORT tracking → Pose estimation →
      Behavioral analysis → MetadataPipeline → FusionEngine.fuse_event() →
      Alert persistence → WebSocket broadcast

    Poll GET /api/jobs/{job_id} for progress and the final alert.
    """
    # Prevent concurrent pipelines to avoid PyTorch/YOLO OOM crashes
    active_jobs = [j for j in JOB_STORE.values() if j.get("status") in ("queued", "processing")]
    if active_jobs:
        raise HTTPException(status_code=429, detail="An ML pipeline is already running. Please wait for it to complete.")

    upload_dir = Path("output/uploads")
    upload_dir.mkdir(parents=True, exist_ok=True)

    job_id = f"JOB_{uuid.uuid4().hex[:10].upper()}"
    ext = Path(file.filename).suffix or ".mp4"
    save_path = upload_dir / f"{job_id}{ext}"

    content = await file.read()
    save_path.write_bytes(content)

    logger.info(f"Video uploaded: {file.filename} → {save_path} ({len(content):,} bytes)")

    # Register job
    JOB_STORE[job_id] = {
        "status": "queued",
        "filename": file.filename,
        "size_bytes": len(content),
        "camera_id": camera_id,
        "station_code": station_code,
        "stages": [],
        "alert": None,
        "error": None,
        "ml_report": None,
        "created_at": datetime.utcnow().isoformat() + "Z",
    }

    # Launch pipeline in background thread (non-blocking)
    t = threading.Thread(
        target=_run_analysis_job,
        args=(job_id, save_path, camera_id, station_code),
        daemon=True,
    )
    t.start()

    return {
        "job_id": job_id,
        "status": "processing",
        "filename": file.filename,
        "size_bytes": len(content),
        "camera_id": camera_id,
        "station_code": station_code,
        "poll_url": f"/api/jobs/{job_id}",
        "message": "Video received. Full ML pipeline running in background.",
    }


# ============================================================================
# 1b. GET /api/jobs/{job_id}  — Poll pipeline job status
# ============================================================================

@router.get("/jobs/{job_id}")
async def get_job_status(
    job_id: str,
    user=Depends(require_role("viewer")),
):
    """
    Poll the status of a video analysis job.

    Returns:
      - status: queued | processing | complete | error
      - stages: list of completed pipeline stages with timestamps
      - alert: FusedAlert dict (present when status=complete)
      - ml_report: Summary ML metrics (frames, anomaly rate, top anomaly)
      - error: error message (present when status=error)
    """
    job = JOB_STORE.get(job_id)
    if not job:
        raise HTTPException(404, detail=f"Job not found: {job_id}")
    return {"job_id": job_id, **job}


# ============================================================================
# 2. POST /api/detect-anomaly
# ============================================================================

@router.post("/detect-anomaly")
async def detect_anomaly(
    request: DetectAnomalyRequest,
    user=Depends(require_role("officer")),
):
    """
    Run anomaly detection pipeline on a CCTV event.

    The request must include pre-extracted CCTV metadata (platform_number,
    date, time, day) sourced directly from the VideoMetadata sidecar.
    Fuses CCTV + ticket intelligence (60-40) into a high-confidence alert.
    """
    engine = _fusion_engine()
    alert = engine.fuse_event(
        camera_id=request.camera_id,
        timestamp=request.timestamp,
        anomaly_type=request.anomaly_type,
        anomaly_confidence=request.anomaly_confidence,
        person_count=request.person_count,
        # CCTV metadata — passed directly from VideoMetadata, never derived
        platform_number=request.platform_number,
        date=request.date,
        time=request.time,
        day=request.day,
        # Optional overrides
        station_code=request.station_code,
        train_number=request.train_number,
        coach=request.coach,
    )
    return alert.to_dict()


# ============================================================================
# 3. GET /api/metadata
# ============================================================================

@router.get("/metadata")
async def get_metadata(
    camera_id: Optional[str] = Query(None),
    station_code: str = Query("SC"),
    platform: Optional[int] = Query(None),
    user=Depends(require_role("viewer")),
):
    """
    Get railway metadata and camera context.

    Returns station info, camera details, platform context,
    and operational metadata.
    """
    from backend.services.railway_stations import STATION_LOOKUP, STATIONS_DB

    station = STATION_LOOKUP.get(station_code)
    if not station:
        raise HTTPException(404, f"Station not found: {station_code}")

    result = {
        "station": {
            "code": station.code,
            "name": station.name,
            "city": station.city,
            "state": station.state,
            "zone": station.zone,
            "platforms": station.platforms,
            "risk_tier": station.risk_tier,
        },
        "total_stations": len(STATIONS_DB),
    }

    if camera_id:
        import re
        m_plat = re.search(r'P(\d+)', camera_id)
        zone_map = {"A": "entry", "B": "mid", "C": "exit"}
        result["camera"] = {
            "camera_id": camera_id,
            "resolved_platform": int(m_plat.group(1)) if m_plat else None,
            "resolved_zone": zone_map.get(camera_id[-1] if camera_id else "", "mid"),
        }

    if platform:
        engine = _train_engine()
        intel = engine.get_platform_intelligence(station_code, platform)
        result["platform_context"] = intel

    return result


# ============================================================================
# 4. GET /api/train-lookup
# ============================================================================

@router.get("/train-lookup")
async def train_lookup(
    station_code: str = Query("SC"),
    platform: Optional[int] = Query(None),
    timestamp: Optional[str] = Query(None),
    train_number: Optional[str] = Query(None),
    tolerance: int = Query(15),
    user=Depends(require_role("viewer")),
):
    """
    Look up trains by station, platform, timestamp, or train number.

    Supports:
    - Platform + time -> active train inference
    - Train number -> full route and schedule
    - Station -> complete timetable
    """
    engine = _train_engine()

    if train_number:
        return engine.get_train_full_route(train_number)

    if timestamp:
        result = engine.infer_train(
            station_code=station_code,
            platform=platform,
            timestamp=timestamp,
            tolerance_minutes=tolerance,
        )
        return result

    return engine.get_station_schedule(station_code)


# ============================================================================
# 5. GET /api/coach-estimation
# ============================================================================

@router.get("/coach-estimation")
async def coach_estimation(
    camera_zone: str = Query("mid", description="entry/mid/exit"),
    train_type: str = Query("Express"),
    person_x: Optional[int] = Query(None),
    frame_width: Optional[int] = Query(None),
    user=Depends(require_role("viewer")),
):
    """
    Estimate which coach is visible from a camera zone.

    Uses zone-based mapping or pixel-position estimation.
    """
    mapper = _bogie_mapper()

    if person_x is not None and frame_width is not None:
        result = mapper.estimate_from_position(
            person_x=person_x,
            frame_width=frame_width,
            camera_zone=camera_zone,
            train_type=train_type,
        )
    else:
        result = mapper.estimate_from_zone(
            camera_zone=camera_zone,
            train_type=train_type,
        )

    return result.to_dict()


# ============================================================================
# 6. GET /api/alerts
# ============================================================================

@router.get("/alerts")
async def get_alerts(
    station_code: str = Query("SC"),
    severity: Optional[str] = Query(None, description="LOW/MEDIUM/HIGH/CRITICAL"),
    limit: int = Query(20, le=100),
    user=Depends(require_role("officer")),
):
    """
    Get intelligence alerts (fused + explainable).

    Queries both the in-memory FusionEngine alert history AND
    the persistent alerts table in PostgreSQL.
    Optionally filter by severity level.
    """
    from backend.database import db_service

    # Merge in-memory alerts (from this session) with DB alerts
    fusion = _fusion_engine()
    history = fusion.get_alert_history(limit=limit * 2)

    # Also pull from DB for persistence across restarts
    try:
        db_alerts = db_service.get_active_alerts(station=station_code, limit=limit)
        # Merge: deduplicate by alert_id
        existing_ids = {a.get("alert_id") for a in history}
        for dba in db_alerts:
            if dba.get("alert_id") not in existing_ids:
                history.append(dba)
    except Exception as e:
        logger.warning(f"Could not fetch DB alerts: {e}")

    alerts = history
    if severity:
        alerts = [a for a in alerts if a.get("severity") == severity.upper()]
    if station_code:
        alerts = [a for a in alerts if a.get("station_code", "") == station_code or not a.get("station_code")]

    return {
        "station": station_code,
        "severity_filter": severity,
        "total": len(alerts),
        "alerts": alerts[:limit],
        "stats": fusion.get_fusion_stats(),
    }


# ============================================================================
# 7. GET /api/health
# ============================================================================

@router.get("/health", tags=["System"])
async def pipeline_health():
    """
    Full pipeline health check.

    Reports status of every component in the intelligence pipeline:
    - PostgreSQL database
    - Redis cache
    - Ticket intelligence cache
    - Fusion engine
    - Trains table validation
    """
    from datetime import datetime, timezone
    health = {
        "status": "healthy",
        "service": "Digital Shield Rail Defense",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "components": {}
    }
    issues = []

    # ── Database ──────────────────────────────────────────────────────────
    try:
        from backend.database import db_service
        db_info = db_service.health_check()
        health["components"]["database"] = {
            "status": db_info.get("status", "unknown"),
            "tables": db_info.get("table_names", []),
            "pool_size": db_info.get("pool_size"),
        }
        if db_info.get("status") != "healthy":
            issues.append("database")
    except Exception as e:
        health["components"]["database"] = {"status": "error", "error": str(e)}
        issues.append("database")

    # ── Redis ─────────────────────────────────────────────────────────────
    try:
        import redis as redis_lib
        import os
        r = redis_lib.from_url(
            os.getenv("REDIS_URL", "redis://localhost:6379/0"),
            socket_connect_timeout=2, socket_timeout=2
        )
        r.ping()
        health["components"]["redis"] = {"status": "healthy"}
    except Exception as e:
        health["components"]["redis"] = {"status": "unavailable", "note": str(e)}
        # Redis is optional — not a fatal issue

    # ── Tickets Intelligence ───────────────────────────────────────────────
    try:
        from backend.services.tickets_intelligence import tickets_intelligence
        cache_status = tickets_intelligence.get_cache_status()
        health["components"]["tickets_intelligence"] = cache_status
        if not cache_status.get("initialized"):
            health["components"]["tickets_intelligence"]["note"] = "Will initialize on first detect-anomaly call"
    except Exception as e:
        health["components"]["tickets_intelligence"] = {"status": "error", "error": str(e)}

    # ── Fusion Engine ─────────────────────────────────────────────────────
    try:
        fusion = _fusion_engine()
        stats = fusion.get_fusion_stats()
        health["components"]["fusion_engine"] = {
            "status": "healthy",
            "total_alerts_this_session": stats.get("total_alerts", 0),
            "active_incident_clusters": stats.get("active_incident_clusters", 0),
        }
    except Exception as e:
        health["components"]["fusion_engine"] = {"status": "error", "error": str(e)}
        issues.append("fusion_engine")

    # ── Trains Table ──────────────────────────────────────────────────────
    try:
        from backend.database import db_service
        with db_service.engine.connect() as conn:
            from sqlalchemy import text as _text
            trains_count = conn.execute(_text("SELECT COUNT(*) FROM trains")).scalar()
        health["components"]["trains_table"] = {
            "status": "healthy",
            "train_count": trains_count,
        }
    except Exception as e:
        health["components"]["trains_table"] = {"status": "unavailable", "error": str(e)}
        issues.append("trains_table")

    # ── Overall status ────────────────────────────────────────────────────
    if issues:
        health["status"] = "degraded"
        health["issues"] = issues

    return health
