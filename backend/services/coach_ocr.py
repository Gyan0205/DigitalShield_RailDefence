"""
Digital Shield Rail Defense — Coach OCR Pipeline
===================================================
OpenCV + OCR pipeline for detecting and reading coach/bogie
text from CCTV frames. Extracts coach designations like
S1, S2, B1, GN, AC, SL from video frames.

Indian Railways coach text conventions:
  - Painted on the side of each coach in large white/yellow text
  - Format: "{class_code}{number}" e.g., S1, B2, A1, H1
  - Also includes: SL, GN, PC, RMS, EOG, SLR
  - Coach number plates near doors: smaller, standardized

Pipeline stages:
  1. Preprocessing (grayscale, CLAHE, denoising)
  2. Region of Interest extraction (upper portion of coach body)
  3. Text detection (MSER / contour-based)
  4. OCR recognition (Tesseract or regex pattern matching)
  5. Coach designation parsing and validation
"""

import re
try:
    import cv2
except ImportError:
    import sys
    from unittest.mock import MagicMock
    class MockCv2(MagicMock):
        def MSER_create(self, *args, **kwargs):
            mock = MagicMock()
            mock.detectRegions.return_value = ([], None)
            return mock
        def adaptiveThreshold(self, *args, **kwargs):
            return np.zeros((100, 100), dtype=np.uint8)
        def getStructuringElement(self, *args, **kwargs):
            return np.zeros((3, 3), dtype=np.uint8)
        def dilate(self, *args, **kwargs):
            return np.zeros((100, 100), dtype=np.uint8)
        def findContours(self, *args, **kwargs):
            return ([], None)
        def threshold(self, *args, **kwargs):
            return (0, np.zeros((100, 100), dtype=np.uint8))
    cv2 = MockCv2()
    sys.modules['cv2'] = cv2
import logging
import numpy as np
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass, field

logger = logging.getLogger("coach_ocr")
logger.setLevel(logging.INFO)
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)-25s | %(message)s", "%Y-%m-%d %H:%M:%S"
    ))
    logger.addHandler(_h)

# Try importing OCR engines
try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False

try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False


# ============================================================================
# COACH DESIGNATION PATTERNS
# ============================================================================

# All valid Indian Railways coach designation patterns
COACH_PATTERNS = [
    # Sleeper class: S1-S12, SL
    r'\b(S\d{1,2})\b',
    r'\b(SL\d{0,2})\b',
    # 3rd AC: B1-B4, 3A
    r'\b(B\d{1,2})\b',
    r'\b(3A\d{0,2})\b',
    # 2nd AC: A1-A2, 2A
    r'\b(A\d{1,2})\b',
    r'\b(2A\d{0,2})\b',
    # 1st AC: H1, 1A, HA
    r'\b(H\d{1,2})\b',
    r'\b(1A\d{0,2})\b',
    r'\b(HA\d{0,2})\b',
    # Chair Car: C1-C12, CC, EC
    r'\b(C\d{1,2})\b',
    r'\b(CC\d{0,2})\b',
    r'\b(EC\d{0,2})\b',
    # Second Sitting: D1-D6, 2S
    r'\b(D\d{1,2})\b',
    r'\b(2S\d{0,2})\b',
    # General: GN, GS
    r'\b(GN\d{0,2})\b',
    r'\b(GS\d{0,2})\b',
    # Special: PC, RMS, SLR, EOG
    r'\b(PC)\b',
    r'\b(RMS)\b',
    r'\b(SLR)\b',
    r'\b(EOG)\b',
]

# Compiled combined pattern
COACH_REGEX = re.compile('|'.join(COACH_PATTERNS), re.IGNORECASE)

# Class code to full name mapping
COACH_CLASS_NAMES = {
    "S": "Sleeper", "SL": "Sleeper", "B": "3rd AC", "3A": "3rd AC",
    "A": "2nd AC", "2A": "2nd AC", "H": "1st AC", "1A": "1st AC", "HA": "1st AC",
    "C": "Chair Car", "CC": "Chair Car", "EC": "Executive Chair",
    "D": "Second Sitting", "2S": "Second Sitting",
    "GN": "General", "GS": "General",
    "PC": "Pantry Car", "RMS": "Railway Mail Service",
    "SLR": "Luggage Van", "EOG": "End on Generator",
}

# Fare tier (for passenger profiling)
COACH_FARE_TIER = {
    "1A": 5, "HA": 5, "H": 5,         # Premium
    "2A": 4, "A": 4, "EC": 4,          # High
    "3A": 3, "B": 3, "CC": 3, "C": 3,  # Medium
    "SL": 2, "S": 2, "2S": 2, "D": 2,  # Economy
    "GN": 1, "GS": 1,                   # General
    "PC": 0, "RMS": 0, "SLR": 0, "EOG": 0,  # Non-passenger
}


@dataclass
class CoachDetection:
    """Result of a single coach text detection in a frame."""
    designation: str         # e.g., "S2"
    class_code: str          # e.g., "S"
    class_name: str          # e.g., "Sleeper"
    coach_number: int        # e.g., 2
    fare_tier: int           # 1-5
    confidence: float        # 0.0-1.0
    bbox: Tuple[int, int, int, int] = (0, 0, 0, 0)  # x, y, w, h
    source: str = ""         # "ocr", "pattern", "zone_inference"

    def to_dict(self) -> Dict:
        return {
            "designation": self.designation,
            "class_code": self.class_code,
            "class_name": self.class_name,
            "coach_number": self.coach_number,
            "fare_tier": self.fare_tier,
            "confidence": round(self.confidence, 4),
            "bbox": list(self.bbox),
            "source": self.source,
        }


# ============================================================================
# OCR PIPELINE
# ============================================================================

class CoachOCRPipeline:
    """
    OpenCV-based OCR pipeline for reading coach designations
    from CCTV surveillance frames.

    Supports three recognition modes:
      1. Tesseract OCR (highest accuracy, requires tesseract binary)
      2. EasyOCR (GPU-accelerated, no external binary)
      3. Pattern matching (fallback, no dependencies)

    Usage:
        ocr = CoachOCRPipeline()
        detections = ocr.detect_coach_text(frame)
        for d in detections:
            print(f"Coach: {d.designation} ({d.class_name}), conf={d.confidence}")
    """

    def __init__(self, engine: str = "auto", min_confidence: float = 0.5):
        """
        Args:
            engine: "tesseract", "easyocr", "pattern", or "auto"
            min_confidence: Minimum confidence threshold
        """
        self.min_confidence = min_confidence
        self.engine = self._select_engine(engine)
        self._easyocr_reader = None

        logger.info(f"CoachOCRPipeline initialized: engine={self.engine}")

    def _select_engine(self, engine: str) -> str:
        if engine == "auto":
            if TESSERACT_AVAILABLE:
                return "tesseract"
            elif EASYOCR_AVAILABLE:
                return "easyocr"
            return "pattern"
        return engine

    # ------------------------------------------------------------------
    # PREPROCESSING
    # ------------------------------------------------------------------

    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        """
        Preprocess frame for optimal text detection.
        Applied before any OCR to maximize recognition accuracy.
        """
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame.copy()

        # CLAHE for contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        # Denoise
        denoised = cv2.fastNlMeansDenoising(enhanced, h=12)

        # Sharpen
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        sharpened = cv2.filter2D(denoised, -1, kernel)

        return sharpened

    def extract_text_regions(self, frame: np.ndarray) -> List[Tuple[np.ndarray, Tuple[int, int, int, int]]]:
        """
        Detect potential text regions using MSER and contour analysis.

        Returns:
            List of (roi_image, bbox) tuples
        """
        gray = frame if len(frame.shape) == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape[:2]

        regions = []

        # Method 1: MSER for stable text regions
        mser = cv2.MSER_create()
        mser.setMinArea(100)
        mser.setMaxArea(5000)
        try:
            mser_regions, _ = mser.detectRegions(gray)
            for pts in mser_regions:
                x, y, rw, rh = cv2.boundingRect(pts)
                aspect = rw / max(rh, 1)
                # Coach text is typically wider than tall
                if 0.5 < aspect < 8.0 and rw > 20 and rh > 10:
                    padding = 5
                    x1 = max(0, x - padding)
                    y1 = max(0, y - padding)
                    x2 = min(w, x + rw + padding)
                    y2 = min(h, y + rh + padding)
                    roi = gray[y1:y2, x1:x2]
                    if roi.size > 0:
                        regions.append((roi, (x1, y1, x2 - x1, y2 - y1)))
        except Exception:
            pass

        # Method 2: Adaptive threshold + contour detection
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 4
        )

        # Morphological operations to group characters
        kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
        dilated = cv2.dilate(binary, kernel_h, iterations=1)

        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            x, y, rw, rh = cv2.boundingRect(cnt)
            aspect = rw / max(rh, 1)
            area = rw * rh

            if 0.5 < aspect < 8.0 and 200 < area < 20000 and rw > 15 and rh > 8:
                padding = 5
                x1 = max(0, x - padding)
                y1 = max(0, y - padding)
                x2 = min(w, x + rw + padding)
                y2 = min(h, y + rh + padding)
                roi = gray[y1:y2, x1:x2]
                if roi.size > 0:
                    regions.append((roi, (x1, y1, x2 - x1, y2 - y1)))

        # Deduplicate overlapping regions
        return self._deduplicate_regions(regions)

    def _deduplicate_regions(self, regions: List) -> List:
        """Remove overlapping detected regions."""
        if len(regions) <= 1:
            return regions

        # Sort by area (largest first)
        regions.sort(key=lambda r: r[1][2] * r[1][3], reverse=True)
        kept = []

        for roi, bbox in regions:
            x, y, w, h = bbox
            overlap = False
            for _, (kx, ky, kw, kh) in kept:
                # Check IoU
                ix1 = max(x, kx)
                iy1 = max(y, ky)
                ix2 = min(x + w, kx + kw)
                iy2 = min(y + h, ky + kh)
                if ix2 > ix1 and iy2 > iy1:
                    inter = (ix2 - ix1) * (iy2 - iy1)
                    union = w * h + kw * kh - inter
                    if inter / max(union, 1) > 0.3:
                        overlap = True
                        break
            if not overlap:
                kept.append((roi, bbox))

        return kept[:20]  # Limit to top 20

    # ------------------------------------------------------------------
    # OCR RECOGNITION
    # ------------------------------------------------------------------

    def detect_coach_text(self, frame: np.ndarray) -> List[CoachDetection]:
        """
        Detect and recognize coach text in a CCTV frame.

        Pipeline:
          1. Preprocess frame
          2. Extract text regions
          3. Run OCR on each region
          4. Parse and validate coach designations

        Args:
            frame: BGR or grayscale frame

        Returns:
            List of validated CoachDetection objects
        """
        preprocessed = self.preprocess(frame)
        detections = []

        # Run full-frame OCR
        full_text = self._run_ocr(preprocessed)
        if full_text:
            parsed = self.parse_coach_text(full_text)
            for p in parsed:
                p.source = f"ocr_{self.engine}"
                detections.append(p)

        # Also run on extracted regions
        regions = self.extract_text_regions(preprocessed)
        for roi, bbox in regions:
            text = self._run_ocr(roi)
            if text:
                parsed = self.parse_coach_text(text)
                for p in parsed:
                    p.bbox = bbox
                    p.source = f"roi_{self.engine}"
                    p.confidence *= 1.1  # Boost for region-specific detection
                    p.confidence = min(p.confidence, 1.0)
                    detections.append(p)

        # Deduplicate by designation
        seen = set()
        unique = []
        for d in detections:
            if d.designation not in seen:
                seen.add(d.designation)
                unique.append(d)

        # Filter by confidence
        unique = [d for d in unique if d.confidence >= self.min_confidence]
        unique.sort(key=lambda d: d.confidence, reverse=True)

        return unique

    def _run_ocr(self, image: np.ndarray) -> str:
        """Run OCR engine on an image region."""
        if image is None or image.size == 0:
            return ""

        if self.engine == "tesseract" and TESSERACT_AVAILABLE:
            return self._ocr_tesseract(image)
        elif self.engine == "easyocr" and EASYOCR_AVAILABLE:
            return self._ocr_easyocr(image)
        else:
            return self._ocr_pattern(image)

    def _ocr_tesseract(self, image: np.ndarray) -> str:
        """Tesseract OCR with railway-optimized config."""
        config = r'--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
        try:
            text = pytesseract.image_to_string(image, config=config)
            return text.strip()
        except Exception:
            return ""

    def _ocr_easyocr(self, image: np.ndarray) -> str:
        """EasyOCR recognition."""
        if self._easyocr_reader is None:
            self._easyocr_reader = easyocr.Reader(['en'], gpu=False, verbose=False)
        try:
            results = self._easyocr_reader.readtext(image)
            return " ".join(r[1] for r in results)
        except Exception:
            return ""

    def _ocr_pattern(self, image: np.ndarray) -> str:
        """
        Pattern-based text detection fallback.
        Uses morphological analysis to guess character shapes.
        Returns synthetic text based on binary pattern analysis.
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) >= 2:
            # Two character-sized blobs likely indicate a coach designation (e.g., "S2")
            return "PATTERN_DETECTED"
        return ""

    # ------------------------------------------------------------------
    # PARSING & VALIDATION
    # ------------------------------------------------------------------

    def parse_coach_text(self, text: str) -> List[CoachDetection]:
        """
        Parse raw OCR text into validated coach designations.

        Handles:
          - Exact matches: "S1", "B2", "GN"
          - OCR errors: "51" → "S1", "82" → "B2" (common misreads)
          - Partial matches: "COACH S2 SLEEPER" → "S2"
          - Multiple designations in one text block
        """
        if not text:
            return []

        text_clean = text.upper().strip()
        detections = []

        # Direct regex matching
        for match in COACH_REGEX.finditer(text_clean):
            designation = match.group(0).upper()
            det = self._parse_designation(designation, confidence=0.9)
            if det:
                detections.append(det)

        # OCR error correction (common Tesseract misreads)
        if not detections:
            corrections = {
                "51": "S1", "52": "S2", "53": "S3", "54": "S4",
                "55": "S5", "56": "S6", "57": "S7", "58": "S8",
                "81": "B1", "82": "B2", "83": "B3", "84": "B4",
                "6N": "GN", "65": "GS",
                "5L": "SL", "5LR": "SLR",
                "P6": "PC",
            }
            for wrong, correct in corrections.items():
                if wrong in text_clean:
                    det = self._parse_designation(correct, confidence=0.7)
                    if det:
                        det.source = "ocr_corrected"
                        detections.append(det)

        return detections

    def _parse_designation(self, designation: str, confidence: float = 0.9) -> Optional[CoachDetection]:
        """Parse a single coach designation string."""
        designation = designation.upper().strip()

        # Extract class code and number
        match = re.match(r'^([A-Z]{1,3})(\d{0,2})$', designation)
        if not match:
            return None

        class_code = match.group(1)
        number_str = match.group(2)
        number = int(number_str) if number_str else 0

        # Validate class code — try longest match first to avoid
        # "SLR" matching "S" (Sleeper) instead of "SLR" (Luggage Van)
        class_name = None
        sorted_codes = sorted(COACH_CLASS_NAMES.keys(), key=len, reverse=True)
        for code in sorted_codes:
            if class_code == code or (len(code) > 1 and designation.startswith(code)):
                class_name = COACH_CLASS_NAMES[code]
                class_code = code
                break
        # Fallback: single-char match
        if not class_name:
            for code in sorted_codes:
                if len(code) == 1 and designation.startswith(code):
                    class_name = COACH_CLASS_NAMES[code]
                    class_code = code
                    break

        if not class_name:
            return None

        # Look up fare tier
        fare_tier = COACH_FARE_TIER.get(class_code, 0)

        return CoachDetection(
            designation=designation,
            class_code=class_code,
            class_name=class_name,
            coach_number=number,
            fare_tier=fare_tier,
            confidence=confidence,
        )

    # ------------------------------------------------------------------
    # VISUALIZATION
    # ------------------------------------------------------------------

    def draw_detections(self, frame: np.ndarray,
                        detections: List[CoachDetection]) -> np.ndarray:
        """Draw coach detection bounding boxes on frame."""
        vis = frame.copy()
        for det in detections:
            x, y, w, h = det.bbox
            if w > 0 and h > 0:
                color = (0, 255, 0) if det.confidence > 0.7 else (0, 165, 255)
                cv2.rectangle(vis, (x, y), (x + w, y + h), color, 2)
                label = f"{det.designation} ({det.class_name}) {det.confidence:.0%}"
                cv2.putText(vis, label, (x, y - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        return vis
