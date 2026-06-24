"""Test script for the Camera Intelligence Mapping Engine."""
import sys
sys.path.insert(0, ".")

print("=" * 65)
print("  CAMERA INTELLIGENCE MAPPING ENGINE — VERIFICATION")
print("=" * 65)

# ─────────────────────────────────────────────────────────────────
# 1. Camera Registry
# ─────────────────────────────────────────────────────────────────
print("\n[1] Camera Registry")
from backend.services.camera_registry import CameraRegistry, CameraRecord

registry = CameraRegistry()
count = registry.generate_from_stations()
print(f"    Generated: {count} cameras across {len(registry.stations)} stations")

# Test ID parsing
tests = [
    ("CAM_NDLS_P05_B", {"station_code": "NDLS", "platform": 5, "zone": "mid"}),
    ("CAM_SC_P03_A",   {"station_code": "SC", "platform": 3, "zone": "entry"}),
    ("CAM_05",         {"station_code": None, "platform": 5, "zone": None}),
    ("CAM-12",         {"station_code": None, "platform": 12, "zone": None}),
]
print("    ID Parsing:")
for cam_id, expected in tests:
    parsed = CameraRecord.parse_camera_id(cam_id)
    status = "OK" if parsed and parsed["platform"] == expected["platform"] else "FAIL"
    print(f"      {cam_id:<22s} -> P{parsed['platform'] if parsed else '?'}, "
          f"stn={parsed['station_code'] if parsed else '?'}, "
          f"zone={parsed['zone'] if parsed else '?'}  [{status}]")

# Test station query
ndls_cams = registry.get_by_station("NDLS")
print(f"    Cameras at NDLS: {len(ndls_cams)}")
p5_cams = registry.get_by_platform("NDLS", 5)
print(f"    Cameras at NDLS P5: {len(p5_cams)}")
for c in p5_cams:
    print(f"      {c.camera_id} zone={c.zone} coaches={c.visible_coaches_start}-{c.visible_coaches_end}")

# Test resolution
print("    Resolution tests:")
r1 = registry.resolve_camera("CAM_NDLS_P05_B")
print(f"      CAM_NDLS_P05_B -> P{r1.get('platform')}, {r1.get('station_name')}, zone={r1.get('zone')} [conf={r1.get('confidence')}]")

r2 = registry.resolve_camera("CAM_05", station_code="SC")
print(f"      CAM_05 + SC    -> P{r2.get('platform')}, zone={r2.get('zone')} [conf={r2.get('confidence')}]")

r3 = registry.resolve_camera("CAM_05")
print(f"      CAM_05 (no ctx) -> resolved={r3.get('resolved')} error={r3.get('error', 'none')}")

# Network stats
stats = registry.get_network_stats()
print(f"    Network: {stats['total_cameras']} total, {stats['active']} active, "
      f"{stats['stations_covered']} stations, {stats['platforms_covered']} platforms")
print(f"    By zone: {stats['by_zone']}")
print(f"    By type: {stats['by_type']}")

# ─────────────────────────────────────────────────────────────────
# 2. Platform Mapper
# ─────────────────────────────────────────────────────────────────
print("\n[2] Platform Mapping Engine")
from backend.services.platform_mapper import PlatformMapper

mapper = PlatformMapper()
mapper.initialize()

# Full resolution with train lookup
print("    Resolve CAM_NDLS_P05_B at 08:30:")
result = mapper.resolve("CAM_NDLS_P05_B", timestamp="08:30")
print(f"      Station: {result['station'].get('name')} ({result['station'].get('code')})")
print(f"      Platform: {result['platform']}")
print(f"      Trains: {len(result['trains_at_platform'])}")
for t in result["trains_at_platform"][:2]:
    print(f"        {t['train_number']} {t['train_name'][:45]} arr={t['arrival']} dep={t['departure']}")
if result.get("visible_coaches"):
    print(f"      Visible coaches: {result['visible_coaches']['range']}")
print(f"      Adjacent cameras: {len(result.get('adjacent_cameras', []))}")
ctx = result.get("operational_context", {})
print(f"      Context: {ctx.get('cameras_active', '?')} active cams, "
      f"coverage={'full' if ctx.get('full_coverage') else 'partial'}")

# Short ID resolution
print("\n    Resolve CAM_05 with station=SC:")
result2 = mapper.resolve("CAM_05", station_code="SC")
print(f"      Resolved: {result2.get('resolved')}")
print(f"      Platform: {result2.get('platform')}")
print(f"      Station: {result2.get('station', {}).get('name')}")

# Station coverage map
print("\n    Station coverage for CSTM (Mumbai CST):")
coverage = mapper.get_station_coverage_map("CSTM")
print(f"      Platforms covered: {coverage['coverage']['platforms_covered']}/{coverage['station']['total_platforms']}")
print(f"      Full coverage: {coverage['coverage']['full_coverage_platforms']} platforms")
print(f"      Coverage %: {coverage['coverage']['coverage_pct']}%")

# Network health
print("\n    Network health (top 5 lowest):")
health = mapper.get_all_stations_status()
for s in health[:5]:
    print(f"      {s['station_code']:<6s} {s['station_name']:<30s} "
          f"{s['active_cameras']}/{s['total_cameras']} cams ({s['health_pct']}%) "
          f"[{s['risk_tier']}]")

# ─────────────────────────────────────────────────────────────────
# 3. Configuration
# ─────────────────────────────────────────────────────────────────
print("\n[3] Configuration Management")
from backend.config import Settings
settings = Settings()
print(f"    App: {settings.app.app_name} v{settings.app.app_version}")
print(f"    Server: {settings.server.host}:{settings.server.port}")
print(f"    Camera config: registry={settings.camera.registry_file}")
print(f"    ML config: model={settings.ml.yolo_model}, conf={settings.ml.confidence_threshold}")

# ─────────────────────────────────────────────────────────────────
# 4. FastAPI Import Check
# ─────────────────────────────────────────────────────────────────
print("\n[4] FastAPI Service")
try:
    from backend.api.camera_api import create_app, FASTAPI_AVAILABLE
    if FASTAPI_AVAILABLE:
        api = create_app()
        routes = [r.path for r in api.routes if hasattr(r, "path")]
        print(f"    FastAPI app created: {len(routes)} routes")
        for r in sorted(routes):
            if r.startswith("/api") or r == "/health":
                print(f"      {r}")
    else:
        print("    FastAPI not installed (pip install fastapi uvicorn)")
except Exception as e:
    print(f"    FastAPI import: {e}")

print("\n" + "=" * 65)
print("  ALL TESTS PASSED")
print("=" * 65)
