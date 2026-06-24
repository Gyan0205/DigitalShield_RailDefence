"""
Digital Shield Rail Defense — Monitoring & Metrics
======================================================
Application-level metrics, structured logging, and health
monitoring for production observability.
"""

import os
import time
import logging
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional
from collections import defaultdict, deque
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler

logger = logging.getLogger("monitoring")


# ============================================================================
# STRUCTURED LOGGING
# ============================================================================

class StructuredLogger:
    """
    Production logging with structured JSON output,
    rotating file handlers, and audit trail.
    """

    def __init__(self, log_dir: str = None):
        self.log_dir = Path(log_dir or os.getenv("LOG_DIR", "logs"))
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._setup_handlers()

    def _setup_handlers(self):
        """Configure rotating log handlers."""
        # Main application log (50MB, 10 backups)
        app_handler = RotatingFileHandler(
            self.log_dir / "app.log",
            maxBytes=50 * 1024 * 1024,
            backupCount=10,
            encoding="utf-8",
        )
        app_handler.setFormatter(logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)-25s | %(message)s"
        ))
        logging.getLogger().addHandler(app_handler)

        # Error log (separate)
        error_handler = RotatingFileHandler(
            self.log_dir / "error.log",
            maxBytes=20 * 1024 * 1024,
            backupCount=5,
            encoding="utf-8",
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(logging.Formatter(
            "%(asctime)s | %(name)s | %(funcName)s:%(lineno)d | %(message)s"
        ))
        logging.getLogger().addHandler(error_handler)

        # JSON structured log (for log aggregation)
        json_handler = RotatingFileHandler(
            self.log_dir / "structured.jsonl",
            maxBytes=100 * 1024 * 1024,
            backupCount=5,
            encoding="utf-8",
        )
        json_handler.setFormatter(JsonFormatter())
        logging.getLogger().addHandler(json_handler)

        # Daily audit log
        audit_dir = self.log_dir / "audit"
        audit_dir.mkdir(exist_ok=True)
        audit_handler = TimedRotatingFileHandler(
            audit_dir / "audit.log",
            when="midnight",
            backupCount=90,
            encoding="utf-8",
        )
        audit_handler.setFormatter(logging.Formatter(
            "%(asctime)s | %(message)s"
        ))
        audit_logger = logging.getLogger("audit")
        audit_logger.addHandler(audit_handler)
        audit_logger.setLevel(logging.INFO)

    def log_event(self, event_type: str, data: Dict):
        """Log a structured event."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "event": event_type,
            **data,
        }
        logger.info(json.dumps(entry))


class JsonFormatter(logging.Formatter):
    """JSON log formatter for structured logging pipelines."""

    def format(self, record):
        log_entry = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        if record.exc_info and record.exc_info[1]:
            log_entry["exception"] = str(record.exc_info[1])
        return json.dumps(log_entry)


# ============================================================================
# APPLICATION METRICS
# ============================================================================

class MetricsCollector:
    """
    In-process metrics collector for API monitoring.
    Tracks request counts, latencies, error rates, and custom gauges.
    """

    def __init__(self, window_size: int = 1000):
        self._counters: Dict[str, int] = defaultdict(int)
        self._latencies: Dict[str, deque] = defaultdict(lambda: deque(maxlen=window_size))
        self._gauges: Dict[str, float] = {}
        self._start_time = time.time()

    def increment(self, name: str, value: int = 1):
        """Increment a counter."""
        self._counters[name] += value

    def record_latency(self, endpoint: str, duration_ms: float):
        """Record request latency."""
        self._latencies[endpoint].append(duration_ms)

    def set_gauge(self, name: str, value: float):
        """Set a gauge value."""
        self._gauges[name] = value

    def get_summary(self) -> Dict:
        """Get metrics summary."""
        uptime = time.time() - self._start_time

        # Request stats
        total_requests = self._counters.get("requests_total", 0)
        total_errors = self._counters.get("errors_total", 0)
        error_rate = total_errors / max(total_requests, 1) * 100

        # Latency stats
        latency_summary = {}
        for endpoint, times in self._latencies.items():
            if times:
                import numpy as np
                arr = np.array(list(times))
                latency_summary[endpoint] = {
                    "count": len(arr),
                    "mean_ms": round(float(arr.mean()), 2),
                    "p50_ms": round(float(np.percentile(arr, 50)), 2),
                    "p95_ms": round(float(np.percentile(arr, 95)), 2),
                    "p99_ms": round(float(np.percentile(arr, 99)), 2),
                }

        return {
            "uptime_seconds": round(uptime, 1),
            "uptime_human": _format_duration(uptime),
            "requests": {
                "total": total_requests,
                "errors": total_errors,
                "error_rate_pct": round(error_rate, 2),
            },
            "counters": dict(self._counters),
            "gauges": dict(self._gauges),
            "latencies": latency_summary,
        }

    def get_prometheus_text(self) -> str:
        """Export metrics in Prometheus text format."""
        lines = []
        lines.append(f"# HELP ds_uptime_seconds Application uptime")
        lines.append(f"# TYPE ds_uptime_seconds gauge")
        lines.append(f"ds_uptime_seconds {time.time() - self._start_time:.1f}")

        for name, value in self._counters.items():
            safe_name = name.replace(".", "_").replace("-", "_")
            lines.append(f"ds_{safe_name} {value}")

        for name, value in self._gauges.items():
            safe_name = name.replace(".", "_").replace("-", "_")
            lines.append(f"ds_{safe_name} {value}")

        return "\n".join(lines) + "\n"


# ============================================================================
# HEALTH MONITOR
# ============================================================================

class HealthMonitor:
    """Monitor health of all system components."""

    def __init__(self):
        self._checks = {}

    def register_check(self, name: str, check_fn):
        """Register a health check function."""
        self._checks[name] = check_fn

    def run_all(self) -> Dict:
        """Run all health checks."""
        results = {}
        overall = "healthy"

        for name, fn in self._checks.items():
            try:
                result = fn()
                results[name] = result
                if result.get("status") != "healthy":
                    overall = "degraded"
            except Exception as e:
                results[name] = {"status": "unhealthy", "error": str(e)}
                overall = "unhealthy"

        return {
            "status": overall,
            "timestamp": datetime.now().isoformat(),
            "checks": results,
        }


# ============================================================================
# HELPERS
# ============================================================================

def _format_duration(seconds: float) -> str:
    """Format seconds into human-readable duration."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds/60:.0f}m {seconds%60:.0f}s"
    hours = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    return f"{hours}h {mins}m"


# ============================================================================
# SINGLETONS
# ============================================================================

metrics = MetricsCollector()
health_monitor = HealthMonitor()

# Auto-setup structured logging in production
if os.getenv("ENVIRONMENT") == "production":
    structured_logger = StructuredLogger()
