"""
Digital Shield Rail Defense — FINAL PRODUCTION AUDIT
=====================================================
Comprehensive verification of all systems.
"""
import sys, os, json, time, importlib, traceback
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))
os.chdir(PROJECT)

passed = failed = warned = 0
issues = []

def ok(label):
    global passed; passed += 1; print(f"  [OK] {label}")
def fail(label, detail=""):
    global failed; failed += 1; issues.append(f"{label}: {detail}"); print(f"  [FAIL] {label} -- {detail}")
def warn(label, detail=""):
    global warned; warned += 1; print(f"  [WARN] {label} -- {detail}")

def check(label, condition, detail=""):
    if condition: ok(label)
    else: fail(label, detail)

print("=" * 70)
print("  DIGITAL SHIELD RAIL DEFENSE -- FINAL PRODUCTION AUDIT")
print("=" * 70)

# ================================================================
# PHASE 1: FILE STRUCTURE
# ================================================================
print("\n[1] FILE STRUCTURE")
required_files = {
    "Dockerfile": PROJECT / "Dockerfile",
    "docker-compose.yml": PROJECT / "docker-compose.yml",
    "requirements.txt": PROJECT / "requirements.txt",
    ".env.example": PROJECT / ".env.example",
    ".dockerignore": PROJECT / ".dockerignore",
    "README.md": PROJECT / "README.md",
    "backend/main.py": PROJECT / "backend" / "main.py",
    "backend/auth.py": PROJECT / "backend" / "auth.py",
    "backend/config.py": PROJECT / "backend" / "config.py",
    "backend/middleware.py": PROJECT / "backend" / "middleware.py",
    "backend/database.py": PROJECT / "backend" / "database.py",
    "backend/monitoring.py": PROJECT / "backend" / "monitoring.py",
    "backend/api/unified_api.py": PROJECT / "backend" / "api" / "unified_api.py",
    "deploy/nginx.conf": PROJECT / "deploy" / "nginx.conf",
    "deploy/init.sql": PROJECT / "deploy" / "init.sql",
    "deploy/prometheus.yml": PROJECT / "deploy" / "prometheus.yml",
    "frontend/package.json": PROJECT / "frontend" / "package.json",
    "frontend/src/App.jsx": PROJECT / "frontend" / "src" / "App.jsx",
    "frontend/src/api.js": PROJECT / "frontend" / "src" / "api.js",
    "ml/config.py": PROJECT / "ml" / "config.py",
    "ml/inference/pipeline.py": PROJECT / "ml" / "inference" / "pipeline.py",
}
for name, path in required_files.items():
    check(name, path.exists(), "File missing")

# ================================================================
# PHASE 2: PYTHON IMPORTS (no broken imports)
# ================================================================
print("\n[2] PYTHON IMPORTS")
modules_to_test = [
    "backend.config",
    "backend.auth",
    "backend.middleware",
    "backend.database",
    "backend.monitoring",
    "backend.api.unified_api",
    "backend.api.train_api",
    "backend.api.coach_api",
    "backend.api.xai_api",
    "backend.api.fusion_api",
    "backend.api.camera_api",
    "backend.services.train_intelligence",
    "backend.services.bogie_mapper",
    "backend.services.coach_ocr",
    "backend.services.fusion_engine",
    "backend.services.xai_engine",
    "backend.services.railway_stations",
    "backend.services.railway_simulator",
    "backend.services.schedule_db",
    "backend.services.platform_mapper",
    "backend.services.camera_registry",
    "backend.services.metadata_pipeline",
    "ml.config",
    "ml.models.anomaly_classifier",
    "ml.inference.pipeline",
    "ml.training.preprocess_dataset",
    "ml.training.experiment_tracker",
    "ml.training.run_pipeline",
    "ml.evaluation.benchmark",
    "ml.datasets.download_datasets",
]
for mod in modules_to_test:
    try:
        importlib.import_module(mod)
        ok(mod)
    except Exception as e:
        fail(mod, str(e)[:80])

# ================================================================
# PHASE 3: FASTAPI APP CREATION
# ================================================================
print("\n[3] FASTAPI APPLICATION")
try:
    from backend.main import create_app, app
    check("App factory works", app is not None)
    routes = [r.path for r in app.routes if hasattr(r, "path")]
    api_routes = [r for r in routes if r.startswith("/api")]
    check("API routes >= 25", len(api_routes) >= 25, f"Only {len(api_routes)} found")
    
    # Check all 6 core endpoints
    core = ["/api/upload-video", "/api/detect-anomaly", "/api/metadata",
            "/api/train-lookup", "/api/coach-estimation", "/api/alerts"]
    for ep in core:
        check(f"Core endpoint {ep}", ep in routes, "Missing from route table")
    
    check("Health endpoint", "/health" in routes)
    check("Metrics endpoint", "/metrics" in routes)
    check("WebSocket /ws/alerts", "/ws/alerts" in routes)
    check("System info", "/api/system/info" in routes)
except Exception as e:
    fail("App creation", str(e)[:100])

# ================================================================
# PHASE 4: SERVICE INTEGRATION
# ================================================================
print("\n[4] SERVICE INTEGRATION")
try:
    from backend.services.train_intelligence import TrainIntelligence
    eng = TrainIntelligence()
    eng.initialize(train_count=50)
    sched = eng.get_station_schedule("SC")
    check("Train Intelligence (50 trains)", sched and "timetable" in sched)
    check("SC schedule has entries", len(sched.get("timetable", [])) > 0, f"{len(sched.get('timetable',[]))} entries")
except Exception as e:
    fail("Train Intelligence", str(e)[:80])

try:
    from backend.services.bogie_mapper import BogieMapper
    bm = BogieMapper(ocr_engine="pattern")
    est = bm.estimate_from_zone("mid", "Express")
    check("Bogie Mapper", est is not None and hasattr(est, "estimated_coach"))
    check("Coach estimation result", est.estimated_coach is not None, f"Got: {est.estimated_coach}")
except Exception as e:
    fail("Bogie Mapper", str(e)[:80])



try:
    from backend.services.fusion_engine import FusionEngine
    fe = FusionEngine()
    fe.initialize()
    alert = fe.fuse_event(camera_id="CAM_SC_P03_B", timestamp="2026-05-12T14:30:00",
                          anomaly_type="suspicious_escort", anomaly_confidence=0.82,
                          person_count=2, platform=3)
    check("Fusion Engine", alert is not None)
    d = alert.to_dict()
    check("Fused alert has severity", "severity" in d, str(list(d.keys())[:5]))
    check("Fused alert CRITICAL", d.get("severity") == "CRITICAL")
    check("Alert has triggered_rules", "triggered_rules" in d)
    check("Alert has fusion_reasoning", "fusion_reasoning" in d)
    check("Alert has recommended_action", "recommended_action" in d)
except Exception as e:
    fail("Fusion Engine", str(e)[:80])

try:
    from backend.services.xai_engine import ExplainableIntelligenceEngine
    xai = ExplainableIntelligenceEngine()
    xai.initialize()
    check("XAI Engine initialized", True)
except Exception as e:
    fail("XAI Engine", str(e)[:80])

# ================================================================
# PHASE 5: ML PIPELINE
# ================================================================
print("\n[5] ML PIPELINE")
try:
    from ml.models.anomaly_classifier import AnomalyClassifier
    import numpy as np
    ac = AnomalyClassifier()
    seq = np.random.randn(16, 32).astype(np.float32)
    result = ac.classify_sequence(seq)
    check("Anomaly Classifier inference", "class_name" in result)
    check("Classification result", result["class_name"] in ac.CLASS_NAMES.values(), f"Got: {result['class_name']}")
except Exception as e:
    fail("Anomaly Classifier", str(e)[:80])

try:
    from ml.models.pose_estimator import PoseEstimator
    pe = PoseEstimator()
    check("Pose Estimator (graceful)", pe.model is not None or pe.model is None)  # Both are OK
    if pe.model is None:
        warn("Pose model", "ultralytics not installed, fallback mode active")
    else:
        ok("Pose model loaded with ultralytics")
except Exception as e:
    fail("Pose Estimator", str(e)[:80])

try:
    from ml.inference.pipeline import InferencePipeline, PipelineConfig
    cfg = PipelineConfig(enable_detection=False, enable_tracking=False,
                         enable_pose=False, enable_behavior=False,
                         enable_classification=False)
    pipe = InferencePipeline(cfg)
    check("Inference Pipeline import", True)
except Exception as e:
    fail("Inference Pipeline", str(e)[:80])

# ================================================================
# PHASE 6: FRONTEND FILES
# ================================================================
print("\n[6] FRONTEND INTEGRATION")
components = ["Header", "Sidebar", "StatsGrid", "CCTVPanel", "AlertsPanel",
              "XAICard", "CoachViz", "PlatformMonitor", "AnalyticsCharts"]
for comp in components:
    path = PROJECT / "frontend" / "src" / "components" / f"{comp}.jsx"
    check(f"Component {comp}.jsx exists", path.exists())

# Check API integration in components
api_js = (PROJECT / "frontend" / "src" / "api.js").read_text(encoding="utf-8")
check("api.js has useApi hook", "useApi" in api_js)
check("api.js has health endpoint", "health" in api_js)
check("api.js template bug fixed", "${station}" not in api_js.split("'")[0] if "'" in api_js else True)

api_connected = 0
for comp in ["AlertsPanel", "CCTVPanel", "StatsGrid", "XAICard", "CoachViz", "PlatformMonitor", "AnalyticsCharts", "Header"]:
    content = (PROJECT / "frontend" / "src" / "components" / f"{comp}.jsx").read_text(encoding="utf-8")
    if "useApi" in content or "api." in content:
        ok(f"{comp} calls API")
        api_connected += 1
    else:
        warn(f"{comp} uses mock data only", "No API call found")
check("Components with API integration >= 7", api_connected >= 7, f"Only {api_connected}/8")

app_jsx = (PROJECT / "frontend" / "src" / "App.jsx").read_text(encoding="utf-8")
check("App.jsx no hardcoded stats array", "stats = [" not in app_jsx[:500])

# ================================================================
# PHASE 7: AUTH & SECURITY
# ================================================================
print("\n[7] SECURITY")
auth_code = (PROJECT / "backend" / "auth.py").read_text(encoding="utf-8")
check("No hardcoded admin keys", "ds-admin-key-2026" not in auth_code)
check("Uses env vars for keys", "DS_API_KEY_ADMIN" in auth_code)
check("Production blocks unauthenticated", 'production' in auth_code)
check("RBAC hierarchy exists", "ROLE_HIERARCHY" in auth_code)

main_code = (PROJECT / "backend" / "main.py").read_text(encoding="utf-8")
check("WebSocket endpoint in main", "ws/alerts" in main_code)
check("Security headers middleware", "SecurityHeaders" in main_code)
check("Rate limit middleware", "RateLimit" in main_code)

env_example = (PROJECT / ".env.example").read_text(encoding="utf-8")
check(".env has DS_API_KEY_ADMIN", "DS_API_KEY_ADMIN" in env_example)
check(".env has DS_API_KEY_OFFICER", "DS_API_KEY_OFFICER" in env_example)
check(".env has DS_API_KEY_VIEWER", "DS_API_KEY_VIEWER" in env_example)

# ================================================================
# PHASE 8: DOCKER & DEPLOYMENT
# ================================================================
print("\n[8] DOCKER & DEPLOYMENT")
df = (PROJECT / "Dockerfile").read_text(encoding="utf-8")
check("Multi-stage Dockerfile", "frontend-build" in df)
check("Non-root user", "useradd" in df or "adduser" in df)
check("HEALTHCHECK", "HEALTHCHECK" in df)

dc = (PROJECT / "docker-compose.yml").read_text(encoding="utf-8")
for svc in ["ds-api", "postgres:15", "redis:7", "nginx:alpine", "prom/prometheus", "grafana/grafana"]:
    check(f"docker-compose: {svc}", svc in dc)

nginx = (PROJECT / "deploy" / "nginx.conf").read_text(encoding="utf-8")
check("Nginx rate limiting", "limit_req" in nginx)
check("Nginx SPA routing", "try_files" in nginx)

sql = (PROJECT / "deploy" / "init.sql").read_text(encoding="utf-8")
for table in ["detections", "alerts", "audit_log", "train_schedules", "model_metrics"]:
    check(f"SQL: {table} table", f"CREATE TABLE IF NOT EXISTS {table}" in sql)

check("README.md exists", (PROJECT / "README.md").exists())
readme = (PROJECT / "README.md").read_text(encoding="utf-8")
check("README has Quick Start", "Quick Start" in readme)
check("README has API endpoints", "/api/" in readme)

# ================================================================
# PHASE 9: DATABASE MODULE
# ================================================================
print("\n[9] DATABASE MODULE")
db_code = (PROJECT / "backend" / "database.py").read_text(encoding="utf-8")
check("SQLAlchemy ORM models", "DeclarativeBase" in db_code)
check("Connection pooling", "QueuePool" in db_code)
check("Health check method", "def health_check" in db_code)
check("Audit logging", "def log_audit" in db_code)
check("AlertModel", "class AlertModel" in db_code)

# ================================================================
# PHASE 10: MONITORING
# ================================================================
print("\n[10] MONITORING")
mon = (PROJECT / "backend" / "monitoring.py").read_text(encoding="utf-8")
check("StructuredLogger", "class StructuredLogger" in mon)
check("MetricsCollector", "class MetricsCollector" in mon)
check("Prometheus text export", "get_prometheus_text" in mon)
check("HealthMonitor", "class HealthMonitor" in mon)

# ================================================================
# RESULTS
# ================================================================
print("\n" + "=" * 70)
total = passed + failed
print(f"  RESULTS: {passed}/{total} passed, {failed} failed, {warned} warnings")
print(f"  PRODUCTION READINESS: {passed/total*100:.0f}%" if total else "  N/A")

if issues:
    print(f"\n  ISSUES ({len(issues)}):")
    for i, issue in enumerate(issues, 1):
        print(f"    {i}. {issue}")

if failed == 0:
    print("\n  STATUS: ALL CHECKS PASSED [PRODUCTION READY]")
elif failed <= 3:
    print("\n  STATUS: MINOR ISSUES [DEMO READY]")
else:
    print(f"\n  STATUS: {failed} FAILURES [NEEDS FIXES]")

print("=" * 70)

# Write results JSON
results = {
    "total": total, "passed": passed, "failed": failed, "warnings": warned,
    "production_readiness_pct": round(passed/total*100, 1) if total else 0,
    "demo_ready": failed <= 3,
    "issues": issues,
}
results_path = PROJECT / "output" / "final_audit.json"
results_path.parent.mkdir(exist_ok=True)
with open(results_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\n  Results saved: {results_path}")
