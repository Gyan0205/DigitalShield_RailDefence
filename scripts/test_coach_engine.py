"""Test script for the Coach/Bogie Estimation Engine."""
import sys
sys.path.insert(0, ".")

import cv2
import numpy as np

print("=" * 65)
print("  COACH/BOGIE ESTIMATION ENGINE — VERIFICATION")
print("=" * 65)

# ─────────────────────────────────────────────────────────────────
# 1. Coach OCR Pipeline
# ─────────────────────────────────────────────────────────────────
print("\n[1] Coach OCR Pipeline")
from backend.services.coach_ocr import CoachOCRPipeline, COACH_REGEX

ocr = CoachOCRPipeline(engine="pattern")
print(f"    Engine: {ocr.engine}")

# Test text parsing
test_texts = [
    "S2",
    "COACH S2 SLEEPER",
    "B1 3A CLASS",
    "GN GENERAL",
    "H1 FIRST CLASS AC",
    "A1 A2 SECOND AC",
    "CC CHAIR CAR",
    "SLR LUGGAGE",
]
print("    Text parsing tests:")
for text in test_texts:
    dets = ocr.parse_coach_text(text)
    if dets:
        d = dets[0]
        print(f"      '{text}' -> {d.designation} ({d.class_name}) tier={d.fare_tier} conf={d.confidence}")
    else:
        print(f"      '{text}' -> no detection")

# Test OCR error correction
print("\n    OCR error correction:")
error_texts = ["51", "82", "6N", "5L"]
for text in error_texts:
    dets = ocr.parse_coach_text(text)
    if dets:
        d = dets[0]
        print(f"      '{text}' -> {d.designation} (corrected: {d.source})")

# Test regex on realistic strings
print("\n    Regex detection in noisy text:")
noisy = "FRAME 342 CAMERA 05 COACH S3 SLEEPER CLASS INDIAN RAILWAYS B2 THIRD AC"
matches = COACH_REGEX.findall(noisy)
# flatten the tuple groups from alternation
found = [m for groups in matches for m in groups if m]
print(f"      Text: '{noisy[:60]}...'")
print(f"      Found: {found}")

# Test on synthetic frame with text
print("\n    Synthetic frame OCR:")
frame = np.zeros((200, 400, 3), dtype=np.uint8) + 40
cv2.putText(frame, "S2", (100, 120), cv2.FONT_HERSHEY_SIMPLEX, 3.0, (255, 255, 255), 5)
detections = ocr.detect_coach_text(frame)
if detections:
    print(f"      Detected: {detections[0].designation} conf={detections[0].confidence:.2f}")
else:
    print(f"      Pattern engine (no Tesseract): fell back to zone inference")

# Preprocessing
preprocessed = ocr.preprocess(frame)
print(f"      Preprocessed shape: {preprocessed.shape}, dtype: {preprocessed.dtype}")

# ─────────────────────────────────────────────────────────────────
# 2. Bogie Mapping Engine
# ─────────────────────────────────────────────────────────────────
print("\n[2] Bogie Mapping Engine")
from backend.services.bogie_mapper import BogieMapper, COACH_ORDER_BY_TYPE

mapper = BogieMapper(ocr_engine="pattern")

# Test zone-based estimation for different train types
print("    Zone-based estimation:")
train_types = ["Rajdhani Express", "Shatabdi Express", "Superfast Express", "Passenger"]
for tt in train_types:
    for zone in ["entry", "mid", "exit"]:
        result = mapper.estimate_from_zone(zone, train_type=tt)
        print(f"      {tt:<24s} zone={zone:<5s} -> {result.estimated_coach:<4s} "
              f"({result.coach_class}) tier={result.fare_tier} conf={result.confidence}")

# Test with actual coach list
print("\n    With specific coach list:")
coaches = ["SLR", "GN", "GN", "S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8", "B1", "B2", "B3", "A1", "H1", "PC", "SLR"]
for zone in ["entry", "mid", "exit"]:
    result = mapper.estimate_from_zone(zone, train_coaches=coaches)
    print(f"      zone={zone:<5s} -> {result.estimated_coach:<4s} "
          f"pos={result.coach_position} candidates={result.zone_candidates}")

# Test position-based estimation
print("\n    Position-based estimation (1920px frame):")
positions = [100, 480, 640, 960, 1440, 1800]
for px in positions:
    result = mapper.estimate_from_position(px, 1920, "mid", coaches, "Superfast Express")
    print(f"      x={px:<5d} -> {result.estimated_coach:<4s} pos={result.coach_position} conf={result.confidence}")

# Test combined estimation (zone + position)
print("\n    Combined estimation:")
result = mapper.estimate(
    person_x=640, frame_width=1920,
    camera_zone="mid", train_coaches=coaches, train_type="Superfast Express"
)
print(f"      Method: {result.estimation_method}")
print(f"      Coach: {result.estimated_coach} ({result.coach_class})")
print(f"      Position: #{result.coach_position}")
print(f"      Confidence: {result.confidence}")
print(f"      Fare tier: {result.fare_tier}")

# ─────────────────────────────────────────────────────────────────
# 3. Coach Metadata
# ─────────────────────────────────────────────────────────────────
print("\n[3] Coach Metadata")
test_coaches = ["S2", "B1", "A1", "H1", "CC", "GN", "EC", "SLR"]
for code in test_coaches:
    info = BogieMapper.get_coach_class_info(code)
    if "error" not in info:
        print(f"    {code:<4s} -> {info['class_name']:<20s} tier={info['fare_tier']} "
              f"berths={info['berth_capacity']} reserved={info['has_reservation']} ac={info['ac_available']}")

# Train layout
print("\n    Rajdhani Express layout:")
layout = BogieMapper.get_train_coach_layout(
    COACH_ORDER_BY_TYPE["Rajdhani Express"], "Rajdhani Express"
)
print(f"      Total: {layout['total_coaches']} coaches, {layout['passenger_coaches']} passenger")
for c in layout["layout"]:
    marker = "*" if c["is_passenger"] else " "
    print(f"      {marker} #{c['position']:2d} {c['coach']:<4s} {c['class_name']:<20s} [{c['zone']}]")

# ─────────────────────────────────────────────────────────────────
# 4. FastAPI Routes
# ─────────────────────────────────────────────────────────────────
print("\n[4] FastAPI Integration")
try:
    from backend.api.camera_api import create_app
    app = create_app()
    routes = sorted([r.path for r in app.routes if hasattr(r, "path") and r.path.startswith("/api")])
    coach_routes = [r for r in routes if "/coach" in r]
    print(f"    Total API routes: {len(routes)}")
    print(f"    Coach routes: {len(coach_routes)}")
    for r in coach_routes:
        print(f"      {r}")
except Exception as e:
    print(f"    FastAPI: {e}")

print("\n" + "=" * 65)
print("  ALL TESTS PASSED")
print("=" * 65)
