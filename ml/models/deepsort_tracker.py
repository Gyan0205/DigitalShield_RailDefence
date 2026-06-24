"""
Digital Shield Rail Defense — DeepSORT Multi-Person Tracker
=============================================================
Persistent identity tracking across video frames using DeepSORT.
Maintains track IDs for each person through occlusions,
enabling behavioral timeline analysis.

Features:
  - Persistent track ID assignment
  - Re-identification through occlusion
  - Track lifecycle management (tentative → confirmed → lost)
  - Per-track trajectory history
  - Velocity and direction estimation
  - Integrates with YOLOv8 detector output
"""

import cv2
import logging
import numpy as np
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass, field
from collections import defaultdict

logger = logging.getLogger("deepsort_tracker")


@dataclass
class TrackState:
    """Persistent state for a tracked person."""
    track_id: int
    bbox_xyxy: Tuple[float, float, float, float]
    bbox_ltwh: Tuple[float, float, float, float]
    confidence: float
    is_confirmed: bool
    time_since_update: int
    # Trajectory history
    trajectory: List[Tuple[float, float]] = field(default_factory=list)
    velocities: List[Tuple[float, float]] = field(default_factory=list)
    frame_indices: List[int] = field(default_factory=list)
    # Behavioral features
    total_distance: float = 0.0
    avg_speed: float = 0.0
    direction_changes: int = 0
    stationary_frames: int = 0
    is_stationary: bool = False

    @property
    def center(self) -> Tuple[float, float]:
        x1, y1, x2, y2 = self.bbox_xyxy
        return (x1 + x2) / 2, (y1 + y2) / 2

    @property
    def track_length(self) -> int:
        return len(self.trajectory)

    def to_dict(self) -> Dict:
        return {
            "track_id": self.track_id,
            "bbox_xyxy": list(self.bbox_xyxy),
            "confidence": round(self.confidence, 4),
            "is_confirmed": self.is_confirmed,
            "center": list(self.center),
            "track_length": self.track_length,
            "total_distance": round(self.total_distance, 2),
            "avg_speed": round(self.avg_speed, 2),
            "direction_changes": self.direction_changes,
            "is_stationary": self.is_stationary,
        }


class DeepSORTTracker:
    """
    DeepSORT-based multi-person tracker for railway CCTV surveillance.

    Usage:
        tracker = DeepSORTTracker()
        tracks = tracker.update(detections, frame)
        history = tracker.get_track_history(track_id)
    """

    def __init__(
        self,
        max_age: int = 30,
        n_init: int = 3,
        max_iou_distance: float = 0.7,
        max_cosine_distance: float = 0.3,
        nn_budget: int = 100,
        embedder: str = "mobilenet",
        stationary_threshold: float = 5.0,
    ):
        self.max_age = max_age
        self.n_init = n_init
        self.max_iou_distance = max_iou_distance
        self.max_cosine_distance = max_cosine_distance
        self.nn_budget = nn_budget
        self.embedder = embedder
        self.stationary_threshold = stationary_threshold

        self.tracker = None
        self._track_histories: Dict[int, TrackState] = {}
        self._frame_count = 0
        self._load_tracker()

    def _load_tracker(self):
        """Initialize DeepSORT tracker."""
        try:
            from deep_sort_realtime.deepsort_tracker import DeepSort
            self.tracker = DeepSort(
                max_age=self.max_age,
                n_init=self.n_init,
                max_iou_distance=self.max_iou_distance,
                max_cosine_distance=self.max_cosine_distance,
                nn_budget=self.nn_budget,
                embedder=self.embedder,
            )
            logger.info(
                f"DeepSORT initialized: max_age={self.max_age}, "
                f"embedder={self.embedder}"
            )
        except ImportError:
            logger.error("deep-sort-realtime not installed. Run: pip install deep-sort-realtime")
            raise

    def update(self, detections, frame: np.ndarray, frame_idx: int = 0) -> List[TrackState]:
        """
        Update tracker with new detections.

        Args:
            detections: List of Detection objects or FrameDetections
            frame: Current BGR frame
            frame_idx: Current frame index

        Returns:
            List of active TrackState objects
        """
        self._frame_count += 1

        # Convert detections to DeepSORT format: ([left, top, w, h], confidence, class)
        ds_detections = []
        if hasattr(detections, 'detections'):
            det_list = detections.detections
        else:
            det_list = detections

        for det in det_list:
            x1, y1, x2, y2 = det.bbox_xyxy
            w, h = x2 - x1, y2 - y1
            ds_detections.append(([x1, y1, w, h], det.confidence, 0))

        # Update DeepSORT
        tracks = self.tracker.update_tracks(ds_detections, frame=frame)

        # Build track states
        active_tracks = []
        for track in tracks:
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            ltrb = track.to_ltrb()
            x1, y1, x2, y2 = ltrb
            ltwh = track.to_ltwh()

            # Get or create track history
            if track_id not in self._track_histories:
                self._track_histories[track_id] = TrackState(
                    track_id=track_id,
                    bbox_xyxy=(float(x1), float(y1), float(x2), float(y2)),
                    bbox_ltwh=(float(ltwh[0]), float(ltwh[1]), float(ltwh[2]), float(ltwh[3])),
                    confidence=track.det_conf if track.det_conf else 0.0,
                    is_confirmed=True,
                    time_since_update=0,
                )
            else:
                state = self._track_histories[track_id]
                state.bbox_xyxy = (float(x1), float(y1), float(x2), float(y2))
                state.bbox_ltwh = (float(ltwh[0]), float(ltwh[1]), float(ltwh[2]), float(ltwh[3]))
                state.confidence = track.det_conf if track.det_conf else state.confidence
                state.is_confirmed = True
                state.time_since_update = 0

            state = self._track_histories[track_id]
            center = state.center
            state.trajectory.append(center)
            state.frame_indices.append(frame_idx)

            # Compute velocity and behavioral features
            if len(state.trajectory) >= 2:
                prev = state.trajectory[-2]
                dx = center[0] - prev[0]
                dy = center[1] - prev[1]
                speed = np.sqrt(dx**2 + dy**2)
                state.velocities.append((dx, dy))
                state.total_distance += speed

                # Check stationary
                if speed < self.stationary_threshold:
                    state.stationary_frames += 1
                    state.is_stationary = state.stationary_frames > 10
                else:
                    state.stationary_frames = 0
                    state.is_stationary = False

                # Direction change detection
                if len(state.velocities) >= 3:
                    v1 = state.velocities[-2]
                    v2 = state.velocities[-1]
                    dot = v1[0] * v2[0] + v1[1] * v2[1]
                    mag1 = np.sqrt(v1[0]**2 + v1[1]**2) + 1e-8
                    mag2 = np.sqrt(v2[0]**2 + v2[1]**2) + 1e-8
                    cos_angle = dot / (mag1 * mag2)
                    if cos_angle < -0.3:  # Sharp direction change
                        state.direction_changes += 1

                # Average speed
                state.avg_speed = state.total_distance / len(state.trajectory)

            active_tracks.append(state)

        return active_tracks

    def get_track_history(self, track_id: int) -> Optional[TrackState]:
        """Get complete history for a track ID."""
        return self._track_histories.get(track_id)

    def get_all_tracks(self) -> Dict[int, TrackState]:
        """Get all tracked persons (including lost tracks)."""
        return self._track_histories.copy()

    def get_active_track_ids(self) -> List[int]:
        """Get IDs of currently active (confirmed) tracks."""
        if self.tracker is None:
            return []
        return [
            t.track_id for t in self.tracker.tracker.tracks
            if t.is_confirmed()
        ]

    def get_pairwise_distances(self, tracks: List[TrackState]) -> Dict[Tuple[int, int], float]:
        """
        Compute pairwise distances between all active tracks.
        Critical for detecting escort behavior and group dynamics.
        """
        distances = {}
        for i, t1 in enumerate(tracks):
            for j, t2 in enumerate(tracks):
                if i >= j:
                    continue
                c1 = t1.center
                c2 = t2.center
                dist = np.sqrt((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2)
                distances[(t1.track_id, t2.track_id)] = dist
        return distances

    def draw_tracks(self, frame: np.ndarray, tracks: List[TrackState],
                    draw_trajectory: bool = True, trail_length: int = 30) -> np.ndarray:
        """Draw tracking visualization on frame."""
        colors = {}
        for track in tracks:
            tid = track.track_id
            if tid not in colors:
                # Deterministic color per track ID
                np.random.seed(tid * 37)
                colors[tid] = tuple(int(c) for c in np.random.randint(50, 255, 3))

            color = colors[tid]
            x1, y1, x2, y2 = [int(v) for v in track.bbox_xyxy]
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Label
            label = f"ID:{tid} spd:{track.avg_speed:.1f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
            cv2.putText(frame, label, (x1 + 2, y1 - 4),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

            # Trajectory trail
            if draw_trajectory and len(track.trajectory) > 1:
                pts = track.trajectory[-trail_length:]
                for k in range(1, len(pts)):
                    p1 = (int(pts[k-1][0]), int(pts[k-1][1]))
                    p2 = (int(pts[k][0]), int(pts[k][1]))
                    alpha = k / len(pts)
                    thickness = max(1, int(alpha * 3))
                    cv2.line(frame, p1, p2, color, thickness)

        # Track count
        cv2.putText(frame, f"Tracks: {len(tracks)}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        return frame

    def reset(self):
        """Reset all tracking state."""
        self._load_tracker()
        self._track_histories.clear()
        self._frame_count = 0
        logger.info("Tracker reset")
