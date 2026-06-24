"""Test script for the railway metadata engine."""
import sys
sys.path.insert(0, ".")

print("=" * 60)
print("TESTING RAILWAY METADATA ENGINE")
print("=" * 60)

# 1. Railway Simulator
print("\n[1] Initializing Railway Simulator...")
from backend.services.railway_simulator import RailwaySimulator
sim = RailwaySimulator(seed=42)
trains = sim.generate_trains(count=300)
schedules = sim.generate_schedules()
cameras = sim.generate_camera_network()
print(f"    Trains: {len(trains)}")
print(f"    Schedules: {len(schedules)}")
print(f"    Cameras: {len(cameras)}")

# 2. Sample Train
print("\n[2] Sample Train:")
t = trains[0]
print(f"    {t.number} - {t.name}")
print(f"    Type: {t.train_type}")
print(f"    Coaches: {t.total_coaches} -> {t.coaches[:8]}...")
print(f"    Route: {t.origin_name} -> {t.destination_name}")

# 3. Schedule Query
print("\n[3] Trains at NDLS around 08:00:")
matches = sim.find_train_at_station("NDLS", "08:00", tolerance_minutes=30)
print(f"    Found {len(matches)} trains")
for m in matches[:3]:
    print(f"    {m.train_number} ({m.train_name[:40]}) arr={m.arrival_time} dep={m.departure_time} P{m.platform_number}")

# 4. Camera Query
print("\n[4] Cameras at NDLS Platform 3:")
cams = sim.find_cameras_for_coach("NDLS", 3, 5)
for c in cams:
    print(f"    {c.camera_id} zone={c.zone} type={c.camera_type} coaches={c.visible_coaches_start}-{c.visible_coaches_end}")

# 5. Full Metadata Pipeline
print("\n[5] Running Metadata Pipeline...")
from pathlib import Path
from backend.services.metadata_pipeline import MetadataPipeline
pipeline = MetadataPipeline(output_dir=Path("dataset/metadata"), seed=42)
report = pipeline.run(
    video_root=Path("dataset/raw"),
    annotations_file=Path("dataset/annotations/annotations.json"),
)
print(f"    Videos processed: {report['videos_processed']}")
print(f"    Stations used: {report['stations_used']}")
print(f"    Trains used: {report['trains_used']}")
print(f"    Cameras used: {report['cameras_used']}")

# 6. Sample Metadata Record
if pipeline.records:
    print("\n[6] Sample Metadata Record:")
    r = pipeline.records[0]
    d = r.to_dict()
    print(f"    Station: {d['station']['name']} ({d['station']['code']})")
    print(f"    Platform: {d['platform_number']}")
    print(f"    Camera: {d['camera_id']}")
    print(f"    Train: {d['train']['number']} - {d['train']['name'][:50]}")
    print(f"    Coach: {d['estimated_coach']} (position #{d['coach_position']})")
    print(f"    Arrival: {d['arrival_time']}  Departure: {d['departure_time']}")
    print(f"    Time: {d['time']} {d['day_of_week']}")
    print(f"    Train status: {d['train_status']}")
    print(f"    Night: {d['is_night']}  Peak: {d['is_peak_hour']}")
    print(f"    Anomaly: {d['anomaly_type']} (class {d['anomaly_class_id']})")

print("\n" + "=" * 60)
print("ALL TESTS PASSED")
print("=" * 60)
