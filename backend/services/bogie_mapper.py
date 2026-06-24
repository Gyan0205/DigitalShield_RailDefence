"""
Digital Shield Rail Defense — Bogie Mapping Engine
=====================================================
Maps CCTV observations to specific coaches/bogies.

Three estimation strategies:
  1. OCR-based: Read coach text from frame (highest confidence)
  2. Zone-based: Infer from camera zone + train composition
  3. Position-based: Estimate from person's pixel position + train layout

The engine integrates:
  - CoachOCRPipeline (text detection)
  - Camera zone coach visibility ranges
  - Train coach composition from RailwaySimulator
  - Person tracking positions from DeepSORT

Coach Position Convention (Indian Railways):
  Engine ← [EOG][SLR][GN][S1][S2]...[B1][A1][H1][PC][SLR][EOG] → Tail
  Position 1 is nearest to the engine/entry side of the platform.
"""

import re
import logging
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass, field

try:
    import cv2
except ImportError:
    import sys
    from unittest.mock import MagicMock
    class MockCv2(MagicMock):
        pass
    cv2 = MockCv2()
    sys.modules['cv2'] = cv2
import numpy as np

from backend.services.coach_ocr import (
    CoachOCRPipeline, CoachDetection, COACH_CLASS_NAMES, COACH_FARE_TIER,
)

logger = logging.getLogger("bogie_mapper")
logger.setLevel(logging.INFO)
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)-25s | %(message)s", "%Y-%m-%d %H:%M:%S"
    ))
    logger.addHandler(_h)


# ============================================================================
# ZONE-TO-COACH MAPPING TABLES
# ============================================================================

# Standard stopping zones at Indian railway platforms
# Zone A = entry/engine side, Zone B = middle, Zone C = exit/tail side
ZONE_COACH_RANGES = {
    "entry": {"start": 1, "end": 4, "description": "Engine side: EOG, SLR, GN, first sleeper coaches"},
    "mid":   {"start": 5, "end": 12, "description": "Middle: Sleeper & AC coaches"},
    "exit":  {"start": 13, "end": 24, "description": "Tail side: Premium AC, PC, rear SLR/EOG"},
}

# Train-type specific coach class order (engine → tail)
COACH_ORDER_BY_TYPE = {
    "Rajdhani Express": ["EOG", "SLR", "3A", "3A", "3A", "3A", "3A", "2A", "2A", "1A", "PC", "SLR", "EOG"],
    "Shatabdi Express": ["EOG", "EC", "EC", "CC", "CC", "CC", "CC", "CC", "CC", "CC", "CC", "PC", "EOG"],
    "Vande Bharat Express": ["EC", "EC", "EC", "EC", "CC", "CC", "CC", "CC", "CC", "CC", "CC", "CC", "CC", "CC", "CC", "CC"],
    "Duronto Express": ["EOG", "SLR", "SL", "SL", "SL", "SL", "3A", "3A", "3A", "3A", "2A", "2A", "1A", "PC", "SLR", "EOG"],
    "Superfast Express": ["SLR", "GN", "GN", "S", "S", "S", "S", "S", "S", "S", "S", "B", "B", "B", "B", "A", "A", "1A", "SLR"],
    "Express": ["SLR", "GN", "GN", "GN", "S", "S", "S", "S", "S", "S", "S", "S", "S", "S", "B", "B", "B", "A", "SLR"],
    "Mail": ["SLR", "GN", "GN", "GN", "GN", "S", "S", "S", "S", "S", "S", "S", "S", "B", "B", "A", "SLR"],
    "Garib Rath Express": ["EOG", "SLR", "3A", "3A", "3A", "3A", "3A", "3A", "3A", "3A", "3A", "3A", "SLR", "EOG"],
    "Jan Shatabdi Express": ["SLR", "GN", "GN", "2S", "2S", "2S", "2S", "2S", "2S", "CC", "CC", "CC", "CC", "SLR"],
    "Humsafar Express": ["EOG", "SLR", "3A", "3A", "3A", "3A", "3A", "3A", "3A", "3A", "3A", "3A", "3A", "3A", "SLR", "EOG"],
    "Passenger": ["SLR", "GN", "GN", "GN", "GN", "GN", "GN", "GN", "GN", "SLR"],
}


@dataclass
class CoachEstimation:
    """Complete coach/bogie estimation result."""
    # Primary estimate
    estimated_coach: str             # e.g., "S2"
    coach_class: str                 # e.g., "Sleeper"
    coach_position: int              # Position from engine (1-indexed)
    fare_tier: int                   # 1-5

    # Confidence and method
    confidence: float                # 0.0-1.0
    estimation_method: str           # "ocr", "zone", "position", "combined"

    # Supporting evidence
    ocr_detections: List[CoachDetection] = field(default_factory=list)
    zone_candidates: List[str] = field(default_factory=list)
    all_coaches: List[str] = field(default_factory=list)

    # Context
    train_type: str = ""
    total_coaches: int = 0
    camera_zone: str = ""

    def to_dict(self) -> Dict:
        return {
            "estimated_coach": self.estimated_coach,
            "coach_class": self.coach_class,
            "coach_position": self.coach_position,
            "fare_tier": self.fare_tier,
            "confidence": round(self.confidence, 4),
            "estimation_method": self.estimation_method,
            "ocr_detections": [d.to_dict() for d in self.ocr_detections],
            "zone_candidates": self.zone_candidates,
            "train_type": self.train_type,
            "total_coaches": self.total_coaches,
            "camera_zone": self.camera_zone,
        }


# ============================================================================
# BOGIE MAPPING ENGINE
# ============================================================================

class BogieMapper:
    """
    Maps CCTV observations to specific coaches/bogies.

    Combines OCR text detection, camera zone analysis, and
    person position estimation to determine which coach a
    suspect most likely entered.

    Usage:
        mapper = BogieMapper()

        # From frame (OCR-based)
        result = mapper.estimate_from_frame(frame, camera_zone="mid", train_coaches=["S1","S2",...])

        # From camera zone (no frame needed)
        result = mapper.estimate_from_zone("mid", train_coaches=["S1","S2",...])

        # From pixel position
        result = mapper.estimate_from_position(
            person_x=640, frame_width=1920,
            camera_zone="mid", train_coaches=["S1","S2",...]
        )
    """

    def __init__(self, ocr_engine: str = "auto", min_ocr_confidence: float = 0.5):
        self.ocr = CoachOCRPipeline(engine=ocr_engine, min_confidence=min_ocr_confidence)

    # ------------------------------------------------------------------
    # STRATEGY 1: OCR-BASED (Highest confidence)
    # ------------------------------------------------------------------

    def estimate_from_frame(
        self,
        frame: np.ndarray,
        camera_zone: str = "mid",
        train_coaches: List[str] = None,
        train_type: str = "Express",
    ) -> CoachEstimation:
        """
        Estimate coach from CCTV frame using OCR.

        Args:
            frame: BGR CCTV frame
            camera_zone: Camera zone (entry/mid/exit)
            train_coaches: Actual coach list from train data
            train_type: Train type for zone mapping fallback

        Returns:
            CoachEstimation with OCR results
        """
        detections = self.ocr.detect_coach_text(frame)

        if detections:
            # Use highest-confidence detection
            best = detections[0]

            # Cross-validate with train's actual coaches
            if train_coaches:
                validated = [d for d in detections if d.designation in train_coaches]
                if validated:
                    best = validated[0]
                    best.confidence = min(best.confidence + 0.1, 1.0)

            # Find position in coach list
            position = 0
            if train_coaches and best.designation in train_coaches:
                position = train_coaches.index(best.designation) + 1

            return CoachEstimation(
                estimated_coach=best.designation,
                coach_class=best.class_name,
                coach_position=position,
                fare_tier=best.fare_tier,
                confidence=best.confidence,
                estimation_method="ocr",
                ocr_detections=detections,
                all_coaches=train_coaches or [],
                train_type=train_type,
                total_coaches=len(train_coaches) if train_coaches else 0,
                camera_zone=camera_zone,
            )

        # Fallback to zone-based estimation
        return self.estimate_from_zone(camera_zone, train_coaches, train_type)

    # ------------------------------------------------------------------
    # STRATEGY 2: ZONE-BASED (Medium confidence)
    # ------------------------------------------------------------------

    def estimate_from_zone(
        self,
        camera_zone: str,
        train_coaches: List[str] = None,
        train_type: str = "Express",
    ) -> CoachEstimation:
        """
        Estimate coach from camera zone and train composition.

        Uses the zone's coach visibility range to narrow candidates,
        then picks the most likely coach based on train type patterns.
        """
        zone_info = ZONE_COACH_RANGES.get(camera_zone, ZONE_COACH_RANGES["mid"])
        coach_start = zone_info["start"]
        coach_end = zone_info["end"]

        # Get coaches in this zone
        if train_coaches:
            zone_coaches = train_coaches[coach_start - 1:coach_end]
        else:
            # Use template coaches for the train type
            template = COACH_ORDER_BY_TYPE.get(train_type, COACH_ORDER_BY_TYPE.get("Express", []))
            zone_coaches = template[coach_start - 1:coach_end]

        if not zone_coaches:
            zone_coaches = ["GN"]

        # Filter out non-passenger coaches
        passenger_coaches = [
            c for c in zone_coaches
            if not any(c.startswith(x) for x in ["EOG", "SLR", "RMS", "PC"])
        ]

        if not passenger_coaches:
            passenger_coaches = zone_coaches

        # Pick the most representative coach (middle of the zone)
        mid_idx = len(passenger_coaches) // 2
        estimated = passenger_coaches[mid_idx]

        # Find class info
        class_code = re.match(r'^([A-Z]+)', estimated)
        class_code_str = class_code.group(1) if class_code else estimated
        class_name = COACH_CLASS_NAMES.get(class_code_str, "Unknown")
        fare_tier = COACH_FARE_TIER.get(class_code_str, 0)

        # Position in full coach list
        position = 0
        all_coaches = train_coaches or COACH_ORDER_BY_TYPE.get(train_type, [])
        if estimated in all_coaches:
            position = all_coaches.index(estimated) + 1

        return CoachEstimation(
            estimated_coach=estimated,
            coach_class=class_name,
            coach_position=position or coach_start + mid_idx,
            fare_tier=fare_tier,
            confidence=0.65,
            estimation_method="zone",
            zone_candidates=passenger_coaches,
            all_coaches=all_coaches,
            train_type=train_type,
            total_coaches=len(all_coaches),
            camera_zone=camera_zone,
        )

    # ------------------------------------------------------------------
    # STRATEGY 3: POSITION-BASED (Lower confidence)
    # ------------------------------------------------------------------

    def estimate_from_position(
        self,
        person_x: int,
        frame_width: int,
        camera_zone: str = "mid",
        train_coaches: List[str] = None,
        train_type: str = "Express",
    ) -> CoachEstimation:
        """
        Estimate coach from person's pixel X-position in the frame.

        Divides the frame into coach segments and maps the person's
        horizontal position to the nearest coach.
        """
        zone_info = ZONE_COACH_RANGES.get(camera_zone, ZONE_COACH_RANGES["mid"])
        coach_start = zone_info["start"]
        coach_end = zone_info["end"]

        all_coaches = train_coaches or COACH_ORDER_BY_TYPE.get(train_type, [])
        zone_coaches = all_coaches[coach_start - 1:coach_end]

        if not zone_coaches:
            return self.estimate_from_zone(camera_zone, train_coaches, train_type)

        # Map pixel position to coach index
        normalized_x = person_x / max(frame_width, 1)
        coach_idx = int(normalized_x * len(zone_coaches))
        coach_idx = min(coach_idx, len(zone_coaches) - 1)
        coach_idx = max(coach_idx, 0)

        estimated = zone_coaches[coach_idx]

        class_code = re.match(r'^([A-Z]+)', estimated)
        class_code_str = class_code.group(1) if class_code else estimated
        class_name = COACH_CLASS_NAMES.get(class_code_str, "Unknown")
        fare_tier = COACH_FARE_TIER.get(class_code_str, 0)

        position = coach_start + coach_idx

        return CoachEstimation(
            estimated_coach=estimated,
            coach_class=class_name,
            coach_position=position,
            fare_tier=fare_tier,
            confidence=0.45,
            estimation_method="position",
            zone_candidates=zone_coaches,
            all_coaches=all_coaches,
            train_type=train_type,
            total_coaches=len(all_coaches),
            camera_zone=camera_zone,
        )

    # ------------------------------------------------------------------
    # COMBINED ESTIMATION (all strategies)
    # ------------------------------------------------------------------

    def estimate(
        self,
        frame: np.ndarray = None,
        person_x: int = None,
        frame_width: int = None,
        camera_zone: str = "mid",
        train_coaches: List[str] = None,
        train_type: str = "Express",
    ) -> CoachEstimation:
        """
        Combined estimation using all available strategies.
        Automatically selects the best result based on confidence.
        """
        candidates = []

        # OCR (best)
        if frame is not None:
            ocr_result = self.estimate_from_frame(
                frame, camera_zone, train_coaches, train_type
            )
            candidates.append(ocr_result)

        # Position (medium)
        if person_x is not None and frame_width is not None:
            pos_result = self.estimate_from_position(
                person_x, frame_width, camera_zone, train_coaches, train_type
            )
            candidates.append(pos_result)

        # Zone (fallback)
        zone_result = self.estimate_from_zone(camera_zone, train_coaches, train_type)
        candidates.append(zone_result)

        # Select highest confidence
        candidates.sort(key=lambda c: c.confidence, reverse=True)
        best = candidates[0]

        # If multiple methods agree, boost confidence
        if len(candidates) >= 2:
            agreements = sum(
                1 for c in candidates
                if c.estimated_coach == best.estimated_coach
            )
            if agreements >= 2:
                best.confidence = min(best.confidence + 0.1, 1.0)
                best.estimation_method = "combined"

        return best

    # ------------------------------------------------------------------
    # COACH METADATA QUERIES
    # ------------------------------------------------------------------

    @staticmethod
    def get_coach_class_info(designation: str) -> Dict:
        """Get metadata for a coach designation."""
        match = re.match(r'^([A-Z]+)(\d*)$', designation.upper())
        if not match:
            return {"error": f"Invalid designation: {designation}"}

        class_code = match.group(1)
        number = int(match.group(2)) if match.group(2) else 0

        class_name = COACH_CLASS_NAMES.get(class_code)
        if not class_name:
            return {"error": f"Unknown class code: {class_code}"}

        fare_tier = COACH_FARE_TIER.get(class_code, 0)

        berth_count = {
            "SL": 72, "S": 72, "3A": 64, "B": 64,
            "2A": 46, "A": 46, "1A": 24, "H": 24, "HA": 24,
            "CC": 78, "C": 78, "EC": 56,
            "2S": 108, "D": 108,
            "GN": 90, "GS": 90,
        }.get(class_code, 0)

        return {
            "designation": designation.upper(),
            "class_code": class_code,
            "class_name": class_name,
            "coach_number": number,
            "fare_tier": fare_tier,
            "fare_tier_name": {0: "Non-passenger", 1: "General", 2: "Economy",
                               3: "Standard", 4: "Premium", 5: "Luxury"}.get(fare_tier, "Unknown"),
            "berth_capacity": berth_count,
            "has_reservation": class_code not in ("GN", "GS"),
            "ac_available": class_code in ("1A", "H", "HA", "2A", "A", "3A", "B", "EC", "CC", "C"),
        }

    @staticmethod
    def get_train_coach_layout(coaches: List[str], train_type: str = "") -> Dict:
        """Build a visual coach layout from coach list."""
        layout = []
        for i, coach in enumerate(coaches, 1):
            match = re.match(r'^([A-Z]+)', coach)
            class_code = match.group(1) if match else coach
            class_name = COACH_CLASS_NAMES.get(class_code, "Unknown")
            fare_tier = COACH_FARE_TIER.get(class_code, 0)

            is_passenger = class_code not in ("EOG", "SLR", "RMS", "PC")

            layout.append({
                "position": i,
                "coach": coach,
                "class_code": class_code,
                "class_name": class_name,
                "fare_tier": fare_tier,
                "is_passenger": is_passenger,
                "zone": "entry" if i <= 4 else ("mid" if i <= 12 else "exit"),
            })

        return {
            "train_type": train_type,
            "total_coaches": len(coaches),
            "passenger_coaches": sum(1 for l in layout if l["is_passenger"]),
            "layout": layout,
        }
