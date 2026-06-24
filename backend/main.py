"""
Digital Shield Rail Defense -- Production FastAPI Application
================================================================
Main application entry point for the complete Digital Shield
backend. Integrates all services, middleware, authentication,
and Docker-ready configuration.

Run:
    uvicorn backend.main:app --reload --port 8000

Docker:
    docker build -t digital-shield .
    docker run -p 8000:8000 digital-shield
"""

import sys
import os
import logging
from pathlib import Path
from contextlib import asynccontextmanager

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from backend.config import Settings
from backend.middleware import (
    RequestLoggingMiddleware,
    SecurityHeadersMiddleware,
    RateLimitMiddleware,
)

# ============================================================================
# LOGGING
# ============================================================================

LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)-25s | %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger("digital_shield")

# File handler for production
log_dir = PROJECT_ROOT / "logs"
log_dir.mkdir(exist_ok=True)
file_handler = logging.FileHandler(log_dir / "app.log", encoding="utf-8")
file_handler.setFormatter(logging.Formatter(LOG_FORMAT))
logging.getLogger().addHandler(file_handler)

# ============================================================================
# LIFESPAN (Startup / Shutdown)
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle management."""
    logger.info("=" * 60)
    logger.info("  DIGITAL SHIELD RAIL DEFENSE -- STARTING")
    logger.info("=" * 60)
    settings = Settings()
    logger.info(f"  Environment: {settings.app.environment}")
    logger.info(f"  Version: {settings.app.app_version}")
    logger.info(f"  Debug: {settings.app.debug}")
    logger.info("  All services will lazy-load on first request")

    # Database health check
    try:
        from backend.database import db_service
        db_health = db_service.health_check()
        if db_health.get("status") == "healthy":
            logger.info(f"  Database: CONNECTED ({db_health.get('tables', 0)} tables)")
        else:
            logger.warning(f"  Database: OFFLINE ({db_health.get('error', 'unknown')})")
    except Exception as e:
        logger.warning(f"  Database: NOT CONFIGURED ({e})")

    logger.info("=" * 60)

    # ── Wire WebSocket broadcast to FusionEngine on startup ──────────────
    # Use asyncio.create_task so this runs after the first request
    # initializes the fusion engine (lazy-loaded singleton).
    import asyncio as _asyncio

    async def _wire_ws_broadcast():
        """Wire WS broadcast callback once fusion engine is initialized."""
        # Retry for up to 60 seconds in case the fusion engine hasn't been
        # initialized yet (it lazy-loads on first /api/detect-anomaly call).
        for attempt in range(30):
            await _asyncio.sleep(2)
            try:
                from backend.api.unified_api import _services
                fusion = _services.get("fusion")
                if fusion and hasattr(app.state, "broadcast_alert"):
                    def _sync_broadcast(alert_dict: dict):
                        """Sync wrapper — schedules async broadcast on the event loop."""
                        try:
                            loop = _asyncio.get_event_loop()
                            if loop.is_running():
                                loop.create_task(app.state.broadcast_alert(alert_dict))
                        except Exception as exc:
                            logger.warning(f"WebSocket broadcast failed: {exc}")

                    fusion._ws_broadcast_callback = _sync_broadcast
                    logger.info("  ✓ WebSocket broadcast wired to FusionEngine")
                    return
            except Exception as e:
                logger.debug(f"  WS wiring attempt {attempt+1}: {e}")
        logger.warning("  WebSocket wiring: FusionEngine never initialized")

    _asyncio.create_task(_wire_ws_broadcast())

    yield  # Application runs here

    logger.info("Digital Shield shutting down...")



# ============================================================================
# APP FACTORY
# ============================================================================

def create_app() -> FastAPI:
    """Create and configure the production FastAPI application."""
    settings = Settings()

    app = FastAPI(
        title="Digital Shield Rail Defense",
        description=(
            "AI-powered railway anti-trafficking surveillance platform.\n\n"
            "Combines CCTV anomaly detection and railway intelligence "
            "into a unified explainable intelligence system.\n\n"
            "**Core APIs:**\n"
            "- `/api/upload-video` — Upload CCTV footage\n"
            "- `/api/detect-anomaly` — Run anomaly detection\n"
            "- `/api/metadata` — Railway metadata\n"
            "- `/api/train-lookup` — Train schedule queries\n"
            "- `/api/coach-estimation` — Coach/bogie estimation\n"
            "- `/api/alerts` — Intelligence alerts\n\n"
            "**Authentication:** Pass `X-API-Key` header. "
            "In development mode, no key required."
        ),
        version=settings.app.app_version,
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # ── CORS ────────────────────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.server.cors_origins + ["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Security Headers ────────────────────────────────────
    app.add_middleware(SecurityHeadersMiddleware)

    # ── Request Logging ─────────────────────────────────────
    app.add_middleware(RequestLoggingMiddleware)

    # ── Rate Limiting ───────────────────────────────────────
    env = os.getenv("ENVIRONMENT", "development")
    if env != "development":
        app.add_middleware(RateLimitMiddleware, max_requests=200, window_seconds=60)

    # ── Mount Unified API (8 required endpoints) ────────────
    from backend.api.unified_api import router as unified_router
    app.include_router(unified_router)
    logger.info("Unified API router mounted (8 core endpoints)")

    # ── Mount Extended Service Routers ──────────────────────
    _mount_optional(app, "backend.api.camera_api", "camera_router", "Camera Intelligence")
    _mount_optional(app, "backend.api.train_api", "router", "Train Schedule")
    _mount_optional(app, "backend.api.coach_api", "router", "Coach Intelligence")
    _mount_optional(app, "backend.api.xai_api", "router", "Explainable AI")
    _mount_optional(app, "backend.api.fusion_api", "router", "Multi-Source Fusion")

    # ── Serve Frontend (production) ─────────────────────────
    frontend_dist = PROJECT_ROOT / "frontend" / "dist"
    if frontend_dist.exists():
        app.mount("/assets", StaticFiles(directory=str(frontend_dist / "assets")), name="static")
        logger.info(f"  + Frontend static assets mounted")

    # ── Health Check ────────────────────────────────────────
    @app.get("/health", tags=["System"])
    async def health():
        db_status = "unknown"
        try:
            from backend.database import db_service
            db_health = db_service.health_check()
            db_status = db_health.get("status", "unknown")
        except Exception:
            db_status = "not_configured"
        return {
            "status": "healthy",
            "service": "Digital Shield Rail Defense",
            "version": settings.app.app_version,
            "environment": settings.app.environment,
            "database": db_status,
        }

    # ── Metrics (Prometheus) ────────────────────────────────
    @app.get("/metrics", tags=["System"])
    async def prometheus_metrics():
        from fastapi.responses import PlainTextResponse
        try:
            from backend.monitoring import metrics
            return PlainTextResponse(metrics.get_prometheus_text(), media_type="text/plain")
        except Exception:
            return PlainTextResponse("# no metrics\n", media_type="text/plain")

    @app.get("/api/system/metrics", tags=["System"])
    async def system_metrics():
        try:
            from backend.monitoring import metrics
            return metrics.get_summary()
        except Exception:
            return {"error": "Metrics not available"}

    # ── System Info ─────────────────────────────────────────
    @app.get("/api/system/info", tags=["System"])
    async def system_info():
        routes = sorted([
            r.path for r in app.routes
            if hasattr(r, "path") and r.path.startswith("/api")
        ])
        return {
            "service": "Digital Shield Rail Defense",
            "version": settings.app.app_version,
            "environment": settings.app.environment,
            "total_api_routes": len(routes),
            "core_endpoints": [
                "/api/upload-video",
                "/api/detect-anomaly",
                "/api/metadata",
                "/api/train-lookup",
                "/api/coach-estimation",
                "/api/alerts",
            ],
            "all_routes": routes,
        }

    # ── Global Exception Handler ────────────────────────────
    @app.exception_handler(Exception)
    async def global_exception_handler(request, exc):
        logger.error(f"Unhandled exception: {exc}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error", "type": str(type(exc).__name__)},
        )

    # ── WebSocket — Real-time Alert Stream ────────────────────
    from fastapi import WebSocket, WebSocketDisconnect
    import asyncio, json as _json

    _ws_clients: list = []

    @app.websocket("/ws/alerts")
    async def ws_alerts(websocket: WebSocket):
        """WebSocket endpoint for real-time alert push."""
        await websocket.accept()
        _ws_clients.append(websocket)
        logger.info(f"WebSocket client connected ({len(_ws_clients)} total)")
        try:
            while True:
                # Keep-alive: wait for client messages or disconnect
                data = await websocket.receive_text()
                # Echo back for ping/pong
                if data == "ping":
                    await websocket.send_text("pong")
        except WebSocketDisconnect:
            _ws_clients.remove(websocket)
            logger.info(f"WebSocket client disconnected ({len(_ws_clients)} total)")

    # Expose broadcast function for alert pushing
    async def _broadcast_alert(alert_data: dict):
        msg = _json.dumps(alert_data, default=str)
        dead = []
        for ws in _ws_clients:
            try:
                await ws.send_text(msg)
            except Exception:
                dead.append(ws)
        for ws in dead:
            _ws_clients.remove(ws)

    app.state.broadcast_alert = _broadcast_alert
    app.state.ws_clients = _ws_clients

    return app


def _mount_optional(app: FastAPI, module_path: str, router_name: str, label: str):
    """Safely mount an optional router."""
    try:
        import importlib
        mod = importlib.import_module(module_path)
        r = getattr(mod, router_name, None)
        if r:
            app.include_router(r)
            logger.info(f"  + {label} router mounted")
    except Exception as e:
        logger.warning(f"  - {label} router skipped: {e}")


# ============================================================================
# APPLICATION INSTANCE
# ============================================================================

app = create_app()

# ============================================================================
# CLI ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    settings = Settings()
    uvicorn.run(
        "backend.main:app",
        host=settings.server.host,
        port=settings.server.port,
        reload=settings.server.reload,
        log_level="info",
    )
