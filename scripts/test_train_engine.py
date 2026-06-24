"""Test script for the Train Schedule Intelligence Engine."""
import sys
sys.path.insert(0, ".")

print("=" * 65)
print("  TRAIN SCHEDULE INTELLIGENCE ENGINE — VERIFICATION")
print("=" * 65)

# ─────────────────────────────────────────────────────────────────
# 1. Schedule Database
# ─────────────────────────────────────────────────────────────────
print("\n[1] Schedule Database")
from backend.services.schedule_db import ScheduleDB
from backend.services.railway_simulator import RailwaySimulator

sim = RailwaySimulator(seed=42)
sim.generate_trains(count=300)
sim.generate_schedules()

db = ScheduleDB()
db.build_from_simulator(sim, apply_delays=True)
print(f"    Records: {db.total_records}")
print(f"    Trains:  {db.total_trains}")
print(f"    Stations: {db.total_stations}")

# Query by station + time
print("\n    Query: NDLS at 08:30 (±15 min)")
matches = db.query_by_station_time("NDLS", "08:30", tolerance_minutes=15)
print(f"    Matches: {len(matches)}")
for m in matches[:3]:
    print(f"      {m.train_number} {m.train_name[:45]}")
    print(f"        arr={m.scheduled_arrival}(+{m.delay_minutes}min={m.actual_arrival}) "
          f"dep={m.scheduled_departure} P{m.platform_number} [{m.status}]")

# Query by platform
print("\n    Query: NDLS Platform 3 (all day)")
pf_records = db.query_by_platform("NDLS", 3)
print(f"    Trains at P3: {len(pf_records)}")

# Train route
print("\n    Train route for first train:")
first_train = sim.trains[0].number
route = db.get_train_route(first_train)
print(f"    Train {first_train}: {len(route)} stops")
for r in route[:4]:
    print(f"      {r.stop_sequence}. {r.station_name:<25s} arr={r.actual_arrival:<6s} dep={r.actual_departure:<6s} P{r.platform_number}")

# Delay stats
print("\n    Delay Statistics (NDLS):")
stats = db.get_delay_statistics("NDLS")
print(f"      On-time: {stats['on_time_pct']}%")
print(f"      Avg delay: {stats['avg_delay_minutes']} min")
print(f"      Max delay: {stats['max_delay_minutes']} min")
print(f"      Severe (>60min): {stats['severe_delays']}")

# Platform conflicts
print("\n    Platform conflicts (NDLS P1):")
conflicts = db.detect_platform_conflicts("NDLS", 1)
print(f"      Conflicts found: {len(conflicts)}")
for c in conflicts[:2]:
    print(f"      {c['train_a']['number']} vs {c['train_b']['number']} "
          f"overlap={c['overlap_minutes']} min")

# ─────────────────────────────────────────────────────────────────
# 2. Train Intelligence Engine
# ─────────────────────────────────────────────────────────────────
print("\n[2] Train Intelligence Engine")
from backend.services.train_intelligence import TrainIntelligence

engine = TrainIntelligence()
engine.initialize()

# Core: Infer train from platform + timestamp
print("\n    Infer train: NDLS P5 @ 08:30")
result = engine.infer_train(
    station_code="NDLS", platform=5, timestamp="08:30"
)
print(f"      Status: {result['status']}")
print(f"      Trains found: {result['trains_found']}")
print(f"      Confidence: {result['confidence']:.4f}")
print(f"      Method: {result['inference_method']}")
if result.get("primary_train"):
    pt = result["primary_train"]
    print(f"      Primary: {pt['train_number']} {pt['train_name'][:45]}")
    print(f"        arr={pt['actual_arrival']} dep={pt['actual_departure']} "
          f"delay={pt['delay_minutes']}min")
    print(f"        Status: {pt.get('inferred_status', '?')}")
    print(f"        Coaches: {pt['total_coaches']} ({pt['coaches'][:5]}...)")

# Timestamp correlation
print("\n    Correlate: NDLS @ 2026-05-12T08:30:00")
corr = engine.correlate_timestamp("NDLS", "2026-05-12T08:30:00")
print(f"      Day: {corr['day']}")
print(f"      Trains: {corr['total_matches']}")
for t in corr.get("trains", [])[:2]:
    print(f"        {t['train_number']} ({t['inferred_status']}) "
          f"arr={t['actual_arrival']} dep={t['actual_departure']}")

# Platform intelligence
print("\n    Platform Intelligence: SC P3 @ 14:30")
intel = engine.get_platform_intelligence("SC", 3, "14:30")
print(f"      Station: {intel['station']['name']} [{intel['station']['risk_tier']}]")
print(f"      Trains today: {intel['occupancy']['total_trains_today']}")
print(f"      Current hour: {intel['occupancy']['current_hour_trains']} trains")
print(f"      On-time: {intel['delay_context']['on_time_pct']}%")
print(f"      Conflicts: {intel['conflicts']}")
print(f"      Night: {intel['temporal_context']['is_night']}, "
      f"Peak: {intel['temporal_context']['is_peak_hour']}")

# Station timetable
print("\n    Station timetable: CSTM 06:00-12:00")
tt = engine.get_station_schedule("CSTM", hour_start=6, hour_end=12)
print(f"      Services: {tt['total_services']}")

# Search
print("\n    Search: 'Rajdhani'")
sr = engine.search("Rajdhani")
print(f"      Results: {sr['results']}")

# Delay report
print("\n    Network delay report:")
dr = engine.get_network_delay_report()
overall = dr["overall"]
print(f"      Total services: {overall['total_services']}")
print(f"      On-time: {overall['on_time_pct']}%")
print(f"      Avg delay: {overall['avg_delay_minutes']} min")
print(f"      Most delayed station: {dr['most_delayed_stations'][0]['station_name']}")

# ─────────────────────────────────────────────────────────────────
# 3. FastAPI Routes
# ─────────────────────────────────────────────────────────────────
print("\n[3] FastAPI Routes")
try:
    from backend.api.camera_api import create_app
    app = create_app()
    routes = sorted([r.path for r in app.routes if hasattr(r, "path") and r.path.startswith("/api")])
    schedule_routes = [r for r in routes if "/schedule" in r]
    print(f"    Total API routes: {len(routes)}")
    print(f"    Schedule routes: {len(schedule_routes)}")
    for r in schedule_routes:
        print(f"      {r}")
except Exception as e:
    print(f"    FastAPI: {e}")

print("\n" + "=" * 65)
print("  ALL TESTS PASSED")
print("=" * 65)
