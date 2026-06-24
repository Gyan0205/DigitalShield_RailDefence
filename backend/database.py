"""
Digital Shield Rail Defense — PostgreSQL Database Integration
================================================================
Production database layer using SQLAlchemy async with connection
pooling, session management, and health checks.

Tables:
  - detections    : Raw CCTV anomaly detection events
  - alerts        : Fused intelligence alerts for RPF
  - audit_log     : Action audit trail
  - train_schedules: Train timetable cache
"""

import os
import logging
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Optional, AsyncGenerator

# Load .env so Supabase credentials are available in all execution contexts
try:
    from dotenv import load_dotenv as _load_dotenv
    _env_path = Path(__file__).resolve().parent.parent / ".env"
    _load_dotenv(dotenv_path=_env_path, override=False)
except ImportError:
    pass

from sqlalchemy import create_engine, text, Column, String, Integer, Float, DateTime, Text, Boolean, JSON
from sqlalchemy.orm import sessionmaker, DeclarativeBase, Session
from sqlalchemy.pool import QueuePool

logger = logging.getLogger("database")

# ============================================================================
# Configuration
# ============================================================================

def get_database_url() -> str:
    """Build database URL from environment variables."""
    import urllib.parse
    host = os.getenv("DB_HOST", "localhost")
    port = os.getenv("DB_PORT", "5432")
    name = os.getenv("DB_NAME", "digital_shield")
    user = os.getenv("DB_USER", "ds_admin")
    password = urllib.parse.quote_plus(os.getenv("DB_PASSWORD", "ds_secure_2026"))
    return f"postgresql://{user}:{password}@{host}:{port}/{name}"


# ============================================================================
# Engine & Session
# ============================================================================

_engine = None
_SessionLocal = None


def get_engine():
    """Get or create SQLAlchemy engine with connection pooling."""
    global _engine
    if _engine is None:
        url = get_database_url()
        pool_min = int(os.getenv("DB_POOL_MIN", "5"))
        pool_max = int(os.getenv("DB_POOL_MAX", "20"))

        _engine = create_engine(
            url,
            poolclass=QueuePool,
            pool_size=pool_min,
            max_overflow=pool_max - pool_min,
            pool_timeout=30,
            pool_recycle=1800,
            pool_pre_ping=True,
            echo=os.getenv("DB_ECHO", "false").lower() == "true",
        )
        logger.info(f"Database engine created: {_engine.url.host}:{_engine.url.port}/{_engine.url.database}")
    return _engine


def get_session_factory():
    """Get or create session factory."""
    global _SessionLocal
    if _SessionLocal is None:
        _SessionLocal = sessionmaker(bind=get_engine(), autocommit=False, autoflush=False)
    return _SessionLocal


def get_db() -> Session:
    """Dependency injector for FastAPI routes."""
    SessionLocal = get_session_factory()
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ============================================================================
# Base Model
# ============================================================================

class Base(DeclarativeBase):
    pass


# ============================================================================
# ORM Models
# ============================================================================


class DetectionModel(Base):
    __tablename__ = "detections"

    id = Column(String(36), primary_key=True)
    camera_id = Column(String(30), nullable=False)
    station_code = Column(String(10), nullable=False)
    platform = Column(Integer)
    timestamp = Column(DateTime, nullable=False)
    anomaly_type = Column(String(50), nullable=False)
    anomaly_confidence = Column(Float)
    bounding_box = Column(JSON)
    track_id = Column(Integer)
    frame_number = Column(Integer)
    video_path = Column(String(500))
    created_at = Column(DateTime)


class AlertModel(Base):
    __tablename__ = "alerts"

    id = Column(String(36), primary_key=True)
    alert_id = Column(String(30), unique=True, nullable=False)
    severity = Column(String(15), nullable=False)
    alert_type = Column(String(50), nullable=False)
    station_code = Column(String(10), nullable=False)
    platform = Column(Integer)
    train_number = Column(String(10))
    coach = Column(String(10))
    suspect_description = Column(Text)
    fusion_confidence = Column(Float)
    source_scores = Column(JSON)
    triggered_rules = Column(JSON)
    xai_explanation = Column(Text)
    intervention_protocol = Column(Text)
    status = Column(String(20), default="active")
    assigned_to = Column(String(100))
    resolved_at = Column(DateTime)
    resolution_notes = Column(Text)
    created_at = Column(DateTime)
    updated_at = Column(DateTime)


class AuditLogModel(Base):
    __tablename__ = "audit_log"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime)
    user_role = Column(String(20))
    action = Column(String(100), nullable=False)
    resource = Column(String(200))
    details = Column(JSON)
    ip_address = Column(String(45))
    request_id = Column(String(50))


class TrainScheduleModel(Base):
    __tablename__ = "train_schedules"

    id = Column(Integer, primary_key=True, autoincrement=True)
    train_number = Column(String(10), nullable=False)
    train_name = Column(String(100))
    station_code = Column(String(10), nullable=False)
    platform = Column(Integer)
    arrival_time = Column(String(10))
    departure_time = Column(String(10))
    days_of_week = Column(String(20))
    is_active = Column(Boolean, default=True)
    updated_at = Column(DateTime)


# ============================================================================
# Database Services
# ============================================================================

class DatabaseService:
    """High-level database operations for Digital Shield."""

    def __init__(self):
        self._engine = None

    @property
    def engine(self):
        if self._engine is None:
            self._engine = get_engine()
        return self._engine

    def health_check(self) -> dict:
        """Check database connectivity."""
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text("SELECT 1"))
                result.fetchone()

                # Get table counts
                tables = conn.execute(text(
                    "SELECT table_name FROM information_schema.tables "
                    "WHERE table_schema='public'"
                )).fetchall()

                return {
                    "status": "healthy",
                    "host": self.engine.url.host,
                    "database": self.engine.url.database,
                    "tables": len(tables),
                    "table_names": [t[0] for t in tables],
                    "pool_size": self.engine.pool.size(),
                    "pool_checked_out": self.engine.pool.checkedout(),
                }
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}

    def get_active_alerts(self, station: str = "SC", limit: int = 20) -> list:
        """Get active alerts for a station."""
        try:
            with self.engine.connect() as conn:
                rows = conn.execute(text(
                    "SELECT * FROM alerts "
                    "WHERE station_code = :st AND status = 'active' "
                    "ORDER BY created_at DESC LIMIT :lim"
                ), {"st": station, "lim": limit}).fetchall()
                return [dict(r._mapping) for r in rows]
        except Exception:
            return []

    def insert_detection(self, detection: dict) -> bool:
        """Insert a CCTV detection record."""
        try:
            with self.engine.connect() as conn:
                conn.execute(text(
                    "INSERT INTO detections (id, camera_id, station_code, platform, "
                    "timestamp, anomaly_type, anomaly_confidence) "
                    "VALUES (gen_random_uuid(), :cam, :st, :pl, :ts, :at, :ac)"
                ), {
                    "cam": detection["camera_id"],
                    "st": detection.get("station_code", "SC"),
                    "pl": detection.get("platform"),
                    "ts": detection["timestamp"],
                    "at": detection["anomaly_type"],
                    "ac": detection.get("anomaly_confidence", 0),
                })
                conn.commit()
                return True
        except Exception as e:
            logger.error(f"Insert detection failed: {e}")
            return False

    def insert_alert(self, alert: dict) -> bool:
        """Insert a fused alert record."""
        import json
        try:
            with self.engine.connect() as conn:
                conn.execute(text(
                    "INSERT INTO alerts (id, alert_id, severity, alert_type, station_code, platform, "
                    "train_number, coach, suspect_description, fusion_confidence, source_scores, "
                    "triggered_rules, xai_explanation, intervention_protocol, status, created_at, updated_at) "
                    "VALUES (gen_random_uuid(), :aid, :sev, :at, :st, :pl, :tn, :ch, :sd, :fc, :ss, :tr, :xe, :ip, :status, NOW(), NOW())"
                ), {
                    "aid": alert["alert_id"],
                    "sev": alert["severity"],
                    "at": alert["alert_type"],
                    "st": alert.get("station_code", "SC"),
                    "pl": alert.get("platform"),
                    "tn": alert.get("train_number"),
                    "ch": alert.get("coach"),
                    "sd": alert.get("suspect_description", ""),
                    "fc": alert.get("fusion_confidence", 0.0),
                    "ss": json.dumps(alert.get("source_scores", {})),
                    "tr": json.dumps(alert.get("triggered_rules", [])),
                    "xe": alert.get("xai_explanation", ""),
                    "ip": alert.get("intervention_protocol", ""),
                    "status": alert.get("status", "active")
                })
                conn.commit()
                return True
        except Exception as e:
            logger.error(f"Insert alert failed: {e}")
            return False

    def log_audit(self, action: str, resource: str = None,
                  details: dict = None, role: str = None, ip: str = None):
        """Write an audit log entry."""
        try:
            with self.engine.connect() as conn:
                conn.execute(text(
                    "INSERT INTO audit_log (action, resource, details, user_role, ip_address) "
                    "VALUES (:act, :res, :det, :role, :ip)"
                ), {
                    "act": action, "res": resource,
                    "det": str(details) if details else None,
                    "role": role, "ip": ip,
                })
                conn.commit()
        except Exception as e:
            logger.warning(f"Audit log failed: {e}")


# Singleton
db_service = DatabaseService()
