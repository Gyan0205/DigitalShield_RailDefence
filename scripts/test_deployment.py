"""
Digital Shield Rail Defense — Deployment Infrastructure Verification
=====================================================================
Validates all deployment artifacts exist and are correct.
"""
import sys
import json
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

print("=" * 70)
print("  DIGITAL SHIELD — DEPLOYMENT INFRASTRUCTURE VERIFICATION")
print("=" * 70)

passed = 0
failed = 0

def check(label, condition):
    global passed, failed
    if condition:
        print(f"  [OK] {label}")
        passed += 1
    else:
        print(f"  [FAIL] {label}")
        failed += 1

# ── Docker ──────────────────────────────────────────────────
print("\n[1] Docker Infrastructure")
check("Dockerfile exists", (PROJECT / "Dockerfile").exists())
check("docker-compose.yml exists", (PROJECT / "docker-compose.yml").exists())
check(".dockerignore exists", (PROJECT / ".dockerignore").exists())

# Verify multi-stage build
df = (PROJECT / "Dockerfile").read_text()
check("Multi-stage build (frontend)", "frontend-build" in df)
check("Non-root user", "useradd" in df)
check("HEALTHCHECK directive", "HEALTHCHECK" in df)
check("Multi-worker uvicorn", "--workers" in df)

# Verify compose services
dc = (PROJECT / "docker-compose.yml").read_text()
check("Service: api", "ds-api" in dc)
check("Service: db (PostgreSQL)", "postgres:15" in dc)
check("Service: redis", "redis:7" in dc)
check("Service: nginx", "nginx:alpine" in dc)
check("Service: prometheus", "prom/prometheus" in dc)
check("Service: grafana", "grafana/grafana" in dc)
check("Health checks configured", "service_healthy" in dc)
check("Volume persistence", "pgdata" in dc)
check("Network isolation", "ds-network" in dc)

# ── Deploy Config ───────────────────────────────────────────
print("\n[2] Deployment Configuration")
check(".env.example exists", (PROJECT / ".env.example").exists())
check("Nginx config", (PROJECT / "deploy" / "nginx.conf").exists())
check("PostgreSQL init.sql", (PROJECT / "deploy" / "init.sql").exists())
check("Prometheus config", (PROJECT / "deploy" / "prometheus.yml").exists())
check("Deploy script (bash)", (PROJECT / "deploy" / "deploy.sh").exists())
check("Deploy script (PowerShell)", (PROJECT / "deploy" / "deploy.ps1").exists())

# Verify nginx
nginx = (PROJECT / "deploy" / "nginx.conf").read_text()
check("Nginx: API proxy", "proxy_pass http://api_backend" in nginx)
check("Nginx: SPA routing", "try_files" in nginx)
check("Nginx: Rate limiting", "limit_req_zone" in nginx)
check("Nginx: Security headers", "X-Frame-Options" in nginx)
check("Nginx: Gzip compression", "gzip on" in nginx)
check("Nginx: Upload limit 500M", "500M" in nginx)

# Verify SQL
sql = (PROJECT / "deploy" / "init.sql").read_text()
check("SQL: detections table", "CREATE TABLE IF NOT EXISTS detections" in sql)
check("SQL: alerts table", "CREATE TABLE IF NOT EXISTS alerts" in sql)
check("SQL: audit_log table", "CREATE TABLE IF NOT EXISTS audit_log" in sql)
check("SQL: train_schedules table", "CREATE TABLE IF NOT EXISTS train_schedules" in sql)
check("SQL: model_metrics table", "CREATE TABLE IF NOT EXISTS model_metrics" in sql)
check("SQL: Indexes created", "CREATE INDEX" in sql)
check("SQL: SC train seed data", "'AP Express'" in sql)
check("SQL: Trigram search", "pg_trgm" in sql)

# ── Backend Integration ────────────────────────────────────
print("\n[3] Backend Integration")
check("database.py exists", (PROJECT / "backend" / "database.py").exists())
check("monitoring.py exists", (PROJECT / "backend" / "monitoring.py").exists())
check("main.py exists", (PROJECT / "backend" / "main.py").exists())
check("middleware.py exists", (PROJECT / "backend" / "middleware.py").exists())
check("auth.py exists", (PROJECT / "backend" / "auth.py").exists())
check("requirements.txt", (PROJECT / "requirements.txt").exists())

# Verify database module
db_mod = (PROJECT / "backend" / "database.py").read_text()
check("DB: SQLAlchemy ORM models", "DeclarativeBase" in db_mod)
check("DB: Connection pooling", "QueuePool" in db_mod)
check("DB: DetectionModel", "class DetectionModel" in db_mod)
check("DB: AlertModel", "class AlertModel" in db_mod)
check("DB: Health check", "def health_check" in db_mod)
check("DB: Audit logging", "def log_audit" in db_mod)

# Verify monitoring module
mon_mod = (PROJECT / "backend" / "monitoring.py").read_text()
check("Monitor: StructuredLogger", "class StructuredLogger" in mon_mod)
check("Monitor: MetricsCollector", "class MetricsCollector" in mon_mod)
check("Monitor: HealthMonitor", "class HealthMonitor" in mon_mod)
check("Monitor: Prometheus export", "get_prometheus_text" in mon_mod)
check("Monitor: JSON formatter", "class JsonFormatter" in mon_mod)
check("Monitor: Rotating logs", "RotatingFileHandler" in mon_mod)
check("Monitor: Audit trail", "TimedRotatingFileHandler" in mon_mod)

# Verify main.py integration
main = (PROJECT / "backend" / "main.py").read_text()
check("Main: DB health at startup", "db_service.health_check" in main)
check("Main: Prometheus endpoint", "/metrics" in main)
check("Main: System metrics API", "/api/system/metrics" in main)
check("Main: Frontend static mount", "StaticFiles" in main)

# ── Frontend ────────────────────────────────────────────────
print("\n[4] Frontend Integration")
check("Frontend package.json", (PROJECT / "frontend" / "package.json").exists())
check("Frontend index.html", (PROJECT / "frontend" / "index.html").exists())
check("Vite config with proxy", "proxy" in (PROJECT / "frontend" / "vite.config.js").read_text())

# ── Summary ─────────────────────────────────────────────────
print("\n" + "=" * 70)
total = passed + failed
print(f"  RESULTS: {passed}/{total} checks passed")
if failed == 0:
    print("  STATUS: ALL INFRASTRUCTURE VERIFIED [PASS]")
else:
    print(f"  STATUS: {failed} checks FAILED")
print("=" * 70)
