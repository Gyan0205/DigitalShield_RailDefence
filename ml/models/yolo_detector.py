"""
Digital Shield Rail Defense — YOLOv8 Person Detector
======================================================
High-performance person detection engine using YOLOv8 for
railway CCTV surveillance. Detects and localizes all persons
in each frame with confidence scoring.

Features:
  - YOLOv8n/s/m variant support (configurable)
  - Person-only filtering (COCO class 0)
  - Batch and single-frame inference
  - RTSP/webcam stream support
  - Configurable confidence and IoU thresholds
  - GPU/CPU auto-detection
  - Detection result serialization
"""

import cv2
import logging
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Union
from dataclasses import dataclass, field

logger = logging.getLogger("yolo_detector")


@dataclass
class Detection:
    """Single person detection result."""
    bbox_xyxy: Tuple[float, float, float, float]  # (x1, y1, x2, y2) absolute
    bbox_xywh: Tuple[float, float, float, float]  # (x_center, y_center, w, h) absolute
    confidence: float
    class_id: int = 0
    class_name: str = "person"
    track_id: Optional[int] = None

    @property
    def area(self) -> float:
        x1, y1, x2, y2 = self.bbox_xyxy
        return max(0, x2 - x1) * max(0, y2 - y1)

    @property
    def center(self) -> Tuple[float, float]:
        return self.bbox_xywh[0], self.bbox_xywh[1]

    def to_dict(self) -> Dict:
        return {
            "bbox_xyxy": list(self.bbox_xyxy),
            "bbox_xywh": list(self.bbox_xywh),
            "confidence": round(self.confidence, 4),
            "class_id": self.class_id,
            "class_name": self.class_name,
            "track_id": self.track_id,
            "area": round(self.area, 2),
            "center": list(self.center),
        }


@dataclass
class FrameDetections:
    """All detections for a single frame."""
    frame_idx: int
    timestamp: float
    detections: List[Detection] = field(default_factory=list)
    frame_shape: Tuple[int, int, int] = (0, 0, 0)

    @property
    def person_count(self) -> int:
        return len(self.detections)

    def to_dict(self) -> Dict:
        return {
            "frame_idx": self.frame_idx,
            "timestamp": round(self.timestamp, 3),
            "person_count": self.person_count,
            "frame_shape": list(self.frame_shape),
            "detections": [d.to_dict() for d in self.detections],
        }


class YOLODetector:
    """
    YOLOv8-based person detection engine for CCTV surveillance.

    Usage:
        detector = YOLODetector(model_variant="yolov8n.pt")
        detections = detector.detect(frame)
        all_dets = detector.detect_video("surveillance.mp4")
    """

    def __init__(
        self,
        model_variant: str = "yolov8n.pt",
        confidence_threshold: float = 0.5,
        iou_threshold: float = 0.45,
        max_detections: int = 50,
        device: str = "auto",
        input_size: int = 640,
    ):
        self.model_variant = model_variant
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.max_detections = max_detections
        self.input_size = input_size
        self.device = device
        self.model = None
        self._load_model()

    def _load_model(self):
        """Load YOLOv8 model."""
        try:
            from ultralytics import YOLO
            self.model = YOLO(self.model_variant)

            # Determine device
            if self.device == "auto":
                import torch
                self.device = "cuda" if torch.cuda.is_available() else "cpu"

            logger.info(
                f"YOLOv8 loaded: {self.model_variant} on {self.device} | "
                f"conf={self.confidence_threshold} iou={self.iou_threshold}"
            )
        except ImportError:
            logger.error("ultralytics not installed. Run: pip install ultralytics")
            raise
        except Exception as e:
            logger.error(f"Failed to load YOLOv8 model: {e}")
            raise

    def detect(self, frame: np.ndarray, return_annotated: bool = False) -> Union[FrameDetections, Tuple[FrameDetections, np.ndarray]]:
        """
        Detect persons in a single frame.

        Args:
            frame: BGR image (numpy array)
            return_annotated: If True, also return frame with drawn boxes

        Returns:
            FrameDetections object (and annotated frame if requested)
        """
        results = self.model.predict(
            source=frame,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            classes=[0],  # Person class only
            max_det=self.max_detections,
            imgsz=self.input_size,
            device=self.device,
            verbose=False,
        )

        detections = []
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu().numpy())
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                w, h = x2 - x1, y2 - y1

                detections.append(Detection(
                    bbox_xyxy=(float(x1), float(y1), float(x2), float(y2)),
                    bbox_xywh=(float(cx), float(cy), float(w), float(h)),
                    confidence=conf,
                ))

        frame_dets = FrameDetections(
            frame_idx=0,
            timestamp=0.0,
            detections=detections,
            frame_shape=frame.shape,
        )

        if return_annotated:
            annotated = self._draw_detections(frame.copy(), detections)
            return frame_dets, annotated

        return frame_dets

    def detect_batch(self, frames: List[np.ndarray]) -> List[FrameDetections]:
        """Detect persons in a batch of frames."""
        all_detections = []
        results = self.model.predict(
            source=frames,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            classes=[0],
            max_det=self.max_detections,
            imgsz=self.input_size,
            device=self.device,
            verbose=False,
        )

        for idx, result in enumerate(results):
            detections = []
            if result.boxes is not None:
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0].cpu().numpy())
                    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                    w, h = x2 - x1, y2 - y1
                    detections.append(Detection(
                        bbox_xyxy=(float(x1), float(y1), float(x2), float(y2)),
                        bbox_xywh=(float(cx), float(cy), float(w), float(h)),
                        confidence=conf,
                    ))

            all_detections.append(FrameDetections(
                frame_idx=idx,
                timestamp=0.0,
                detections=detections,
                frame_shape=frames[idx].shape if idx < len(frames) else (0, 0, 0),
            ))

        return all_detections

    def detect_video(
        self,
        video_path: Union[str, Path],
        frame_skip: int = 1,
        max_frames: Optional[int] = None,
        callback=None,
    ) -> List[FrameDetections]:
        """
        Run detection on an entire video file.

        Args:
            video_path: Path to video
            frame_skip: Process every Nth frame
            max_frames: Maximum frames to process
            callback: Optional callback(frame_idx, frame_detections) per frame

        Returns:
            List of FrameDetections for all processed frames
        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        all_detections = []
        frame_idx = 0
        processed = 0

        logger.info(f"Processing video: {video_path} ({total_frames} frames, {fps:.1f} FPS)")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if max_frames and processed >= max_frames:
                break

            if frame_idx % frame_skip == 0:
                frame_dets = self.detect(frame)
                frame_dets.frame_idx = frame_idx
                frame_dets.timestamp = frame_idx / fps if fps > 0 else 0
                all_detections.append(frame_dets)
                processed += 1

                if callback:
                    callback(frame_idx, frame_dets)

                if processed % 100 == 0:
                    logger.info(f"  Frame {frame_idx}/{total_frames}: {frame_dets.person_count} persons")

            frame_idx += 1

        cap.release()
        logger.info(f"Detection complete: {processed} frames, {sum(d.person_count for d in all_detections)} total detections")
        return all_detections

    def _draw_detections(self, frame: np.ndarray, detections: List[Detection]) -> np.ndarray:
        """Draw bounding boxes on frame."""
        for det in detections:
            x1, y1, x2, y2 = [int(v) for v in det.bbox_xyxy]
            color = (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            label = f"Person {det.confidence:.2f}"
            if det.track_id is not None:
                label = f"ID:{det.track_id} {det.confidence:.2f}"

            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw, y1), color, -1)
            cv2.putText(frame, label, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        # Person count overlay
        cv2.putText(frame, f"Persons: {len(detections)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        return frame

    def warmup(self, num_iterations: int = 3):
        """Warm up model for consistent inference times."""
        dummy = np.zeros((self.input_size, self.input_size, 3), dtype=np.uint8)
        for _ in range(num_iterations):
            self.detect(dummy)
        logger.info("YOLOv8 warmup complete")
