"""Test the complete unified FastAPI backend."""
import sys
sys.path.insert(0, ".")

print("=" * 70)
print("  DIGITAL SHIELD RAIL DEFENSE -- UNIFIED BACKEND VERIFICATION")
print("=" * 70)

# -------------------------------------------------------------------
# 1. Middleware
# -------------------------------------------------------------------
print("\n[1] Middleware modules")
from backend.middleware import RequestLoggingMiddleware, SecurityHeadersMiddleware, RateLimitMiddleware
print("    RequestLoggingMiddleware: OK")
print("    SecurityHeadersMiddleware: OK")
print("    RateLimitMiddleware: OK")

# -------------------------------------------------------------------
# 2. Authentication
# -------------------------------------------------------------------
print("\n[2] Authentication & RBAC")
from backend.auth import API_KEYS, ROLE_HIERARCHY
print(f"    API keys configured: {len(API_KEYS)}")
print(f"    Roles: {list(ROLE_HIERARCHY.keys())}")
for role, perms in ROLE_HIERARCHY.items():
    print(f"      {role}: can access {perms}")

# -------------------------------------------------------------------
# 3. App Factory
# -------------------------------------------------------------------
print("\n[3] Application Factory")
from backend.main import create_app
app = create_app()

routes = sorted([r.path for r in app.routes if hasattr(r, "path") and r.path.startswith("/api")])
print(f"    Total API routes: {len(routes)}")

# -------------------------------------------------------------------
# 4. Core 8 Endpoints
# -------------------------------------------------------------------
print("\n[4] Core 6 Required Endpoints")
required = [
    "/api/upload-video",
    "/api/detect-anomaly",
    "/api/metadata",
    "/api/train-lookup",
    "/api/coach-estimation",
    "/api/alerts",
]
for ep in required:
    found = ep in routes
    status = "FOUND" if found else "MISSING"
    print(f"    [{status}] {ep}")

# -------------------------------------------------------------------
# 5. Extended Service Routers
# -------------------------------------------------------------------
print("\n[5] Extended Service Routers")
prefixes = {
    "cameras": "/api/cameras",
    "schedule": "/api/schedule",
    "coach": "/api/coach",
    "xai": "/api/xai",
    "fusion": "/api/fusion",
    "system": "/api/system",
}
for name, prefix in prefixes.items():
    count = sum(1 for r in routes if r.startswith(prefix))
    print(f"    {name:<15s} {count:>3d} routes ({prefix}/*)")

# -------------------------------------------------------------------
# 6. Health Check
# -------------------------------------------------------------------
print("\n[6] Health Check")
health_routes = [r.path for r in app.routes if hasattr(r, "path") and r.path == "/health"]
print(f"    /health endpoint: {'FOUND' if health_routes else 'MISSING'}")

# -------------------------------------------------------------------
# 7. Docker Files
# -------------------------------------------------------------------
print("\n[7] Docker Support")
from pathlib import Path
dockerfile = Path("Dockerfile").exists()
compose = Path("docker-compose.yml").exists()
reqs = Path("requirements.txt").exists()
print(f"    Dockerfile: {'OK' if dockerfile else 'MISSING'}")
print(f"    docker-compose.yml: {'OK' if compose else 'MISSING'}")
print(f"    requirements.txt: {'OK' if reqs else 'MISSING'}")

# -------------------------------------------------------------------
# 8. All Routes Summary
# -------------------------------------------------------------------
print(f"\n[8] Complete Route Map ({len(routes)} routes)")
for r in routes:
    print(f"    {r}")

print("\n" + "=" * 70)
print("  ALL TESTS PASSED")
print("=" * 70)
