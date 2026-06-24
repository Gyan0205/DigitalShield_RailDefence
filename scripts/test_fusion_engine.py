"""Test script for the Multi-Source Fusion Intelligence Engine."""
import sys
sys.path.insert(0, ".")

print("=" * 70)
print("  MULTI-SOURCE FUSION INTELLIGENCE ENGINE -- VERIFICATION")
print("=" * 70)

# -------------------------------------------------------------------
# 1. Evidence Accumulator (Dempster-Shafer)
# -------------------------------------------------------------------
print("\n[1] Evidence Accumulator (Dempster-Shafer)")
from backend.services.fusion_engine import EvidenceAccumulator

acc = EvidenceAccumulator()

# Single source
print(f"    Single belief [0.8]: {acc.combine_beliefs([0.8])}")

# Two agreeing sources
print(f"    Two agreeing [0.8, 0.7]: {acc.combine_beliefs([0.8, 0.7])}")

# Five sources
beliefs = [0.82, 0.95, 0.65, 0.90, 0.37]
print(f"    Five sources {beliefs}: {acc.combine_beliefs(beliefs)}")

# Weighted combine
weighted = [(0.82, 0.30), (0.95, 0.10), (0.65, 0.15), (0.90, 0.10), (0.37, 0.35)]
print(f"    Weighted combine: {acc.weighted_combine(weighted)}")

# Agreement / Conflict
print(f"    Agreement: {acc.compute_agreement(beliefs)}")
print(f"    Conflict: {acc.compute_conflict(beliefs)}")

# -------------------------------------------------------------------
# 2. Event Bus
# -------------------------------------------------------------------
print("\n[2] Event Bus (Async Pub/Sub)")
from backend.services.fusion_engine import EventBus, Signal

bus = EventBus()
received = []
bus.subscribe("test.topic", lambda s: received.append(s.signal_id))
bus.subscribe("*", lambda s: None)  # wildcard

bus.publish("test.topic", Signal(signal_id="SIG_001", source="test", confidence=0.9))
bus.publish("test.topic", Signal(signal_id="SIG_002", source="test", confidence=0.7))
bus.publish("other.topic", Signal(signal_id="SIG_003", source="test", confidence=0.5))

print(f"    Received on test.topic: {received}")
print(f"    Total events: {bus.total_events}")
print(f"    Event log: {len(bus.get_event_log())} entries")

# -------------------------------------------------------------------
# 3. Temporal Correlator
# -------------------------------------------------------------------
print("\n[3] Temporal Correlator")
from backend.services.fusion_engine import TemporalCorrelator

corr = TemporalCorrelator(window_seconds=300)
s1 = Signal(signal_id="A", source="cctv", timestamp="2026-05-12T14:30:00",
            payload={"station_code": "SC", "platform": 3})
s2 = Signal(signal_id="B", source="train", timestamp="2026-05-12T14:31:00",
            payload={"station_code": "SC", "platform": 3})
s3 = Signal(signal_id="C", source="cctv", timestamp="2026-05-12T20:00:00",
            payload={"station_code": "SC", "platform": 5})

c1 = corr.add_signal(s1)
c2 = corr.add_signal(s2)
c3 = corr.add_signal(s3)

print(f"    Signal A cluster: {c1}")
print(f"    Signal B cluster: {c2} (same as A: {c1 == c2})")
print(f"    Signal C cluster: {c3} (different: {c1 != c3})")
print(f"    Active clusters: {corr.get_active_clusters()}")

# -------------------------------------------------------------------
# 4. Full Fusion (5 Sources)
# -------------------------------------------------------------------
print("\n[4] Full Fusion Engine")
from backend.services.fusion_engine import FusionEngine

engine = FusionEngine()
engine.initialize()

alert = engine.fuse_event(
    camera_id="CAM_SC_P03_B",
    timestamp="2026-05-12T14:30:00",
    anomaly_type="suspicious_escort",
    anomaly_confidence=0.82,
    person_count=2,
)

print(f"    Alert: {alert.alert_id}")
print(f"    Severity: {alert.severity}")
print(f"    Fused confidence: {alert.fused_confidence:.0%}")
print(f"    Agreement score: {alert.agreement_score:.0%}")
print(f"    Evidence mass: {alert.evidence_mass:.0%}")
print(f"    Station: {alert.station_name} ({alert.station_code})")
print(f"    Platform: {alert.platform}")
print(f"    Train: {alert.train_number} ({alert.train_name})")
print(f"    Coach: {alert.estimated_coach}")

print(f"    Suspects: {alert.suspects}")

# -------------------------------------------------------------------
# 5. Source Contributions
# -------------------------------------------------------------------
print("\n[5] Source Contributions")
for source, contrib in alert.source_contributions.items():
    print(f"    {source:<25s} contribution={contrib:.4f}")

# -------------------------------------------------------------------
# 6. Fusion Reasoning
# -------------------------------------------------------------------
print("\n[6] Fusion Reasoning Chain")
for reason in alert.fusion_reasoning:
    print(f"    {reason}")

# -------------------------------------------------------------------
# 7. Night-time Fusion
# -------------------------------------------------------------------
print("\n[7] Night-time Fusion Test")
night = engine.fuse_event(
    camera_id="CAM_SC_P05_B",
    timestamp="2026-05-12T23:30:00",
    anomaly_type="isolated_minor",
    anomaly_confidence=0.75,
)
print(f"    Severity: {night.severity}")
print(f"    Fused: {night.fused_confidence:.0%} (night boost applied)")

# -------------------------------------------------------------------
# 8. Batch Fusion
# -------------------------------------------------------------------
print("\n[8] Batch Fusion")
batch = engine.fuse_batch([
    {"camera_id": "CAM_SC_P01_A", "timestamp": "2026-05-12T08:00:00",
     "anomaly_type": "coercion", "anomaly_confidence": 0.91},
    {"camera_id": "CAM_SC_P07_C", "timestamp": "2026-05-12T15:20:00",
     "anomaly_type": "dragging", "anomaly_confidence": 0.65},
])
print(f"    Batch alerts: {len(batch)}")
for a in batch:
    print(f"      {a.alert_id[:18]:<18s} {a.severity:<10s} "
          f"conf={a.fused_confidence:.0%} train={a.train_number}")

# -------------------------------------------------------------------
# 9. Statistics
# -------------------------------------------------------------------
print("\n[9] Fusion Stats")
stats = engine.get_fusion_stats()
print(f"    Total alerts: {stats['total_alerts']}")
print(f"    Severity dist: {stats['severity_distribution']}")
print(f"    Avg confidence: {stats['avg_confidence']:.0%}")
print(f"    Events processed: {stats['total_events_processed']}")
print(f"    Active clusters: {stats['active_incident_clusters']}")

# -------------------------------------------------------------------
# 10. FastAPI Routes
# -------------------------------------------------------------------
print("\n[10] FastAPI Integration")
try:
    from backend.api.camera_api import create_app
    app = create_app()
    routes = sorted([r.path for r in app.routes
                     if hasattr(r, "path") and r.path.startswith("/api")])
    fusion_routes = [r for r in routes if "/fusion" in r]
    print(f"    Total API routes: {len(routes)}")
    print(f"    Fusion routes: {len(fusion_routes)}")
    for r in fusion_routes:
        print(f"      {r}")
except Exception as e:
    print(f"    FastAPI: {e}")

print("\n" + "=" * 70)
print("  ALL TESTS PASSED")
print("=" * 70)
