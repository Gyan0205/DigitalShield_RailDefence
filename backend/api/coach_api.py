"""
Digital Shield Rail Defense — Coach Intelligence API
======================================================
FastAPI endpoints for coach/bogie estimation, metadata,
and train layout queries.

Endpoints:
    POST /api/coach/estimate-frame     — OCR-based estimation from frame
    GET  /api/coach/estimate-zone      — Zone-based estimation
    GET  /api/coach/estimate-position  — Position-based estimation
    GET  /api/coach/info/{designation} — Coach class metadata
    GET  /api/coach/layout             — Train coach layout
    GET  /api/coach/zone-map           — Zone-to-coach mapping table
    POST /api/coach/parse-text         — Parse raw text for designations
    GET  /api/coach/fare-tiers         — All fare tier definitions
"""

import sys
import logging
import base64
from pathlib import Path
from typing import Optional, List

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    from fastapi import APIRouter, HTTPException, Query, UploadFile, File
    from pydantic import BaseModel
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False

from backend.services.bogie_mapper import (
    BogieMapper, ZONE_COACH_RANGES, COACH_ORDER_BY_TYPE,
)
from backend.services.coach_ocr import COACH_CLASS_NAMES, COACH_FARE_TIER

logger = logging.getLogger("coach_api")

# ============================================================================
# SINGLETON MAPPER
# ============================================================================

_mapper: Optional[BogieMapper] = None


def get_mapper() -> BogieMapper:
    global _mapper
    if _mapper is None:
        _mapper = BogieMapper(ocr_engine="auto")
    return _mapper


# ============================================================================
# REQUEST/RESPONSE MODELS
# ============================================================================

if FASTAPI_AVAILABLE:

    class TextParseRequest(BaseModel):
        text: str
        train_coaches: Optional[List[str]] = None

    class FrameEstimateRequest(BaseModel):
        """Base64-encoded frame for OCR estimation."""
        frame_base64: str
        camera_zone: str = "mid"
        train_coaches: Optional[List[str]] = None
        train_type: str = "Express"

    # ======================================================================
    # ROUTER
    # ======================================================================

    router = APIRouter(prefix="/api/coach", tags=["Coach Intelligence"])

    # ==================================================================
    # ZONE-BASED ESTIMATION
    # ==================================================================

    @router.get("/estimate-zone")
    async def estimate_from_zone(
        zone: str = Query(..., description="Camera zone (entry/mid/exit)"),
        train_type: str = Query("Express", description="Train type"),
        coaches: Optional[str] = Query(None, description="Comma-separated coach list"),
    ):
        """
        Estimate likely coach from camera zone.

        Example:
        - `/api/coach/estimate-zone?zone=mid&train_type=Rajdhani Express`
        """
        mapper = get_mapper()
        coach_list = coaches.split(",") if coaches else None
        result = mapper.estimate_from_zone(zone, coach_list, train_type)
        return result.to_dict()

    # ==================================================================
    # POSITION-BASED ESTIMATION
    # ==================================================================

    @router.get("/estimate-position")
    async def estimate_from_position(
        person_x: int = Query(..., description="Person X position in pixels"),
        frame_width: int = Query(1920, description="Frame width in pixels"),
        zone: str = Query("mid", description="Camera zone"),
        train_type: str = Query("Express", description="Train type"),
        coaches: Optional[str] = Query(None, description="Comma-separated coach list"),
    ):
        """
        Estimate coach from person's X-position in frame.

        Example:
        - `/api/coach/estimate-position?person_x=640&frame_width=1920&zone=mid`
        """
        mapper = get_mapper()
        coach_list = coaches.split(",") if coaches else None
        result = mapper.estimate_from_position(
            person_x, frame_width, zone, coach_list, train_type
        )
        return result.to_dict()

    # ==================================================================
    # FRAME-BASED ESTIMATION (OCR)
    # ==================================================================

    @router.post("/estimate-frame")
    async def estimate_from_frame(request: FrameEstimateRequest):
        """
        Estimate coach from a CCTV frame using OCR.

        Accepts a base64-encoded frame image and runs the
        full OCR pipeline to detect coach text.
        """
        mapper = get_mapper()

        try:
            img_bytes = base64.b64decode(request.frame_base64)
            nparr = np.frombuffer(img_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if frame is None:
                raise ValueError("Cannot decode image")
        except Exception as e:
            raise HTTPException(400, detail=f"Invalid image: {e}")

        result = mapper.estimate_from_frame(
            frame, request.camera_zone,
            request.train_coaches, request.train_type,
        )
        return result.to_dict()

    # ==================================================================
    # COACH METADATA
    # ==================================================================

    @router.get("/info/{designation}")
    async def coach_info(designation: str):
        """
        Get metadata for a coach designation.

        Example: `/api/coach/info/S2`
        """
        info = BogieMapper.get_coach_class_info(designation)
        if "error" in info:
            raise HTTPException(404, detail=info["error"])
        return info

    @router.get("/layout")
    async def train_layout(
        train_type: str = Query("Express", description="Train type"),
        coaches: Optional[str] = Query(None, description="Comma-separated coach list"),
    ):
        """
        Get visual coach layout for a train type.

        Example: `/api/coach/layout?train_type=Rajdhani Express`
        """
        if coaches:
            coach_list = coaches.split(",")
        else:
            coach_list = COACH_ORDER_BY_TYPE.get(train_type, COACH_ORDER_BY_TYPE.get("Express", []))

        layout = BogieMapper.get_train_coach_layout(coach_list, train_type)
        return layout

    @router.get("/zone-map")
    async def zone_mapping():
        """Get the zone-to-coach mapping table."""
        return {
            "zones": {
                zone: {**info, "coaches_example": COACH_ORDER_BY_TYPE.get("Express", [])[info["start"]-1:info["end"]]}
                for zone, info in ZONE_COACH_RANGES.items()
            },
            "train_types": list(COACH_ORDER_BY_TYPE.keys()),
        }

    @router.get("/fare-tiers")
    async def fare_tiers():
        """Get all fare tier definitions."""
        tiers = {
            0: {"name": "Non-passenger", "coaches": []},
            1: {"name": "General", "coaches": []},
            2: {"name": "Economy", "coaches": []},
            3: {"name": "Standard", "coaches": []},
            4: {"name": "Premium", "coaches": []},
            5: {"name": "Luxury", "coaches": []},
        }
        for code, tier in COACH_FARE_TIER.items():
            name = COACH_CLASS_NAMES.get(code, code)
            tiers[tier]["coaches"].append({"code": code, "name": name})
        return {"fare_tiers": tiers}

    @router.post("/parse-text")
    async def parse_text(request: TextParseRequest):
        """
        Parse raw text for coach designations.

        Handles OCR output with errors and noise.

        Example body: `{"text": "COACH S2 SLEEPER CLASS"}`
        """
        mapper = get_mapper()
        detections = mapper.ocr.parse_coach_text(request.text)

        # Cross-validate with train coaches if provided
        if request.train_coaches:
            for d in detections:
                if d.designation in request.train_coaches:
                    d.confidence = min(d.confidence + 0.1, 1.0)

        return {
            "input_text": request.text,
            "detections": [d.to_dict() for d in detections],
            "total_found": len(detections),
        }
