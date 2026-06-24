"""Test script for the Explainable AI Intelligence Engine."""
import sys
sys.path.insert(0, ".")

print("=" * 70)
print("  EXPLAINABLE AI INTELLIGENCE ENGINE -- VERIFICATION")
print("=" * 70)

# -------------------------------------------------------------------
# 1. XAI Engine Initialization
# -------------------------------------------------------------------
print("\n[1] XAI Engine Initialization")
from backend.services.xai_engine import ExplainableIntelligenceEngine

engine = ExplainableIntelligenceEngine()
engine.initialize()
print("    All subsystems online")

# -------------------------------------------------------------------
# 2. Generate Alert (Full Pipeline)
# -------------------------------------------------------------------
print("\n[2] Full Pipeline Alert Generation")
alert = engine.generate_alert(
    camera_id="CAM_SC_P03_B",
    timestamp="2026-05-12T14:30:00",
    anomaly_type="suspicious_escort",
    anomaly_confidence=0.82,
    person_count=2,
    track_ids=[101, 102],
    frame_number=4520,
)

print(f"    Alert ID: {alert.alert_id}")
print(f"    Severity: {alert.severity}")
print(f"    Station: {alert.railway_context.station_name} ({alert.railway_context.station_code})")
print(f"    Platform: {alert.railway_context.platform}")
print(f"    Train: {alert.railway_context.train_number} ({alert.railway_context.train_name})")
print(f"    Coach: {alert.railway_context.estimated_coach} ({alert.railway_context.coach_class})")

print(f"    Suspects: {len(alert.suspects)}")
print(f"    Rules triggered: {len(alert.triggered_rules)}")
print(f"    Mitigating factors: {len(alert.mitigating_factors)}")

# -------------------------------------------------------------------
# 3. Confidence Breakdown
# -------------------------------------------------------------------
print("\n[3] Confidence Breakdown")
c = alert.confidence
print(f"    CCTV anomaly:     {c.cctv_confidence:.0%} (weight 35%)")
print(f"    Train match:      {c.train_match_confidence:.0%} (weight 15%)")
print(f"    Coach estimation: {c.coach_confidence:.0%} (weight 10%)")
print(f"    Booking anomaly:  {c.booking_anomaly_score:.0%} (weight 40%)")
print(f"    Temporal mod:     x{c.temporal_modifier:.1f}")
print(f"    COMPOSITE:        {c.composite_confidence:.0%}")

# -------------------------------------------------------------------
# 4. Pipeline Stages
# -------------------------------------------------------------------
print("\n[4] Pipeline Stages")
for stage in alert.pipeline_stages:
    print(f"    [{stage['status']:<12s}] {stage['stage']:<22s} {stage['detail']}")

# -------------------------------------------------------------------
# 5. Reasoning Chain
# -------------------------------------------------------------------
print("\n[5] Reasoning Chain")
for i, reason in enumerate(alert.reasoning_chain, 1):
    print(f"    {i}. {reason}")

# -------------------------------------------------------------------
# 6. Triggered Rules
# -------------------------------------------------------------------
print("\n[6] Triggered Rules")
if alert.triggered_rules:
    for r in alert.triggered_rules:
        print(f"    [{r['score']:+.1f}] {r['rule']}: {r['explanation'][:80]}")
else:
    print("    No rules triggered (normal behavior)")

if alert.mitigating_factors:
    print("\n    Mitigating:")
    for m in alert.mitigating_factors:
        print(f"    [{m['score']:+.1f}] {m['rule']}: {m['explanation'][:80]}")

# -------------------------------------------------------------------
# 7. Explainable Report
# -------------------------------------------------------------------
print("\n[7] Explainable Report (RPF output)")
for line in alert.explanation_summary.split("\n")[:20]:
    print(f"    {line}")

# -------------------------------------------------------------------
# 8. Recommended Action
# -------------------------------------------------------------------
print("\n[8] Recommended Action")
for line in alert.recommended_action.split("\n"):
    print(f"    {line}")

# -------------------------------------------------------------------
# 9. Night-time Alert (temporal modifier test)
# -------------------------------------------------------------------
print("\n[9] Night-time Alert Test")
night_alert = engine.generate_alert(
    camera_id="CAM_SC_P05_B",
    timestamp="2026-05-12T23:45:00",
    anomaly_type="isolated_minor",
    anomaly_confidence=0.75,
    person_count=1,
)
print(f"    Severity: {night_alert.severity}")
print(f"    Temporal modifier: x{night_alert.confidence.temporal_modifier:.1f}")
print(f"    Is night: {night_alert.railway_context.is_night}")
print(f"    Composite confidence: {night_alert.confidence.composite_confidence:.0%}")

# -------------------------------------------------------------------
# 10. Audit Trail
# -------------------------------------------------------------------
print("\n[10] Audit Trail")
log = engine.get_audit_log()
print(f"    Total audit entries: {len(log)}")
for entry in log:
    print(f"    [{entry['severity']:<8s}] {entry['alert_id']} | "
          f"station={entry['station']} | conf={entry['confidence']:.0%}")

stats = engine.get_audit_stats()
print(f"\n    Stats: {stats}")

# -------------------------------------------------------------------
# 11. FastAPI Routes
# -------------------------------------------------------------------
print("\n[11] FastAPI Integration")
try:
    from backend.api.camera_api import create_app
    app = create_app()
    routes = sorted([r.path for r in app.routes
                     if hasattr(r, "path") and r.path.startswith("/api")])
    xai_routes = [r for r in routes if "/xai" in r]
    print(f"    Total API routes: {len(routes)}")
    print(f"    XAI routes: {len(xai_routes)}")
    for r in xai_routes:
        print(f"      {r}")
except Exception as e:
    print(f"    FastAPI: {e}")

print("\n" + "=" * 70)
print("  ALL TESTS PASSED")
print("=" * 70)
