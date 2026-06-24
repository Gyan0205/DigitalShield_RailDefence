"""
Digital Shield Rail Defense — Behavioral Anomaly Analyzer
===========================================================
Analyzes tracked person behaviors to detect trafficking-related
anomalies. Combines trajectory, pose, and interaction features
into anomaly scores with explainable reasoning.

Detects:
  1. Assault — sudden fast movement toward another person
  2. Coercion — person being constrained/restrained
  3. Dragging — person being forcefully moved
  4. Suspicious escort — adult closely escorting minor/reluctant person
  5. Isolated minor — child alone on platform for extended period
  6. Panic movement — erratic high-speed directional changes
  7. Crowd anomalies — unusual density or dispersal patterns
"""

import logging
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict

logger = logging.getLogger("behavior_analyzer")


@dataclass
class AnomalyScore:
    """Anomaly detection result for a person or interaction."""
    anomaly_type: str
    confidence: float  # 0.0 to 1.0
    class_id: int
    track_ids: List[int]
    reasoning: List[str]  # Human-readable explanations
    features: Dict[str, float] = field(default_factory=dict)
    frame_idx: int = 0
    timestamp: float = 0.0

    @property
    def is_anomalous(self) -> bool:
        return self.confidence >= 0.5

    @property
    def risk_level(self) -> str:
        if self.confidence >= 0.85:
            return "CRITICAL"
        elif self.confidence >= 0.7:
            return "HIGH"
        elif self.confidence >= 0.5:
            return "MEDIUM"
        elif self.confidence >= 0.3:
            return "LOW"
        return "NORMAL"

    def to_dict(self) -> Dict:
        return {
            "anomaly_type": self.anomaly_type,
            "confidence": round(self.confidence, 4),
            "risk_level": self.risk_level,
            "class_id": self.class_id,
            "track_ids": self.track_ids,
            "reasoning": self.reasoning,
            "features": {k: round(v, 4) for k, v in self.features.items()},
            "frame_idx": self.frame_idx,
            "timestamp": round(self.timestamp, 3),
        }


ANOMALY_CLASS_MAP = {
    "normal": 0, "assault": 1, "coercion": 2, "dragging": 3,
    "suspicious_escort": 4, "isolated_minor": 5, "panic": 6,
    "crowd_anomaly": 7,
}


class BehaviorAnalyzer:
    """
    Behavioral anomaly analyzer for railway CCTV surveillance.

    Combines movement patterns, pose features, and inter-person
    interactions to score anomalies in real-time.

    Usage:
        analyzer = BehaviorAnalyzer()
        scores = analyzer.analyze_frame(tracks, poses, frame_idx)
    """

    def __init__(
        self,
        assault_speed_threshold: float = 40.0,
        escort_distance_threshold: float = 80.0,
        minor_size_ratio: float = 0.55,
        panic_speed_threshold: float = 30.0,
        panic_direction_changes: int = 5,
        loiter_duration_frames: int = 60,
        crowd_density_threshold: float = 0.3,
    ):
        self.assault_speed_threshold = assault_speed_threshold
        self.escort_distance_threshold = escort_distance_threshold
        self.minor_size_ratio = minor_size_ratio
        self.panic_speed_threshold = panic_speed_threshold
        self.panic_direction_changes = panic_direction_changes
        self.loiter_duration_frames = loiter_duration_frames
        self.crowd_density_threshold = crowd_density_threshold

        # State tracking
        self._track_anomaly_history: Dict[int, List[AnomalyScore]] = defaultdict(list)
        self._isolation_counters: Dict[int, int] = defaultdict(int)

    def analyze_frame(
        self,
        tracks: List,  # List of TrackState
        poses: Optional[List] = None,  # List of PersonPose
        pairwise_distances: Optional[Dict] = None,
        frame_idx: int = 0,
        timestamp: float = 0.0,
        frame_shape: Tuple[int, int] = (480, 640),
    ) -> List[AnomalyScore]:
        """
        Analyze all tracked persons in a frame for anomalies.

        Returns list of AnomalyScore objects for detected anomalies.
        """
        anomalies = []

        # Per-person analysis
        for track in tracks:
            # Panic detection
            panic_score = self._detect_panic(track, frame_idx, timestamp)
            if panic_score:
                anomalies.append(panic_score)

            # Isolated minor detection
            isolation_score = self._detect_isolated_minor(
                track, tracks, pairwise_distances, frame_idx, timestamp, frame_shape
            )
            if isolation_score:
                anomalies.append(isolation_score)

        # Pairwise interaction analysis
        if len(tracks) >= 2:
            for i, t1 in enumerate(tracks):
                for j, t2 in enumerate(tracks):
                    if i >= j:
                        continue

                    # Assault detection
                    assault_score = self._detect_assault(t1, t2, frame_idx, timestamp)
                    if assault_score:
                        anomalies.append(assault_score)

                    # Suspicious escort detection
                    escort_score = self._detect_suspicious_escort(
                        t1, t2, poses, frame_idx, timestamp
                    )
                    if escort_score:
                        anomalies.append(escort_score)

                    # Dragging detection
                    drag_score = self._detect_dragging(t1, t2, poses, frame_idx, timestamp)
                    if drag_score:
                        anomalies.append(drag_score)

                    # Coercion detection
                    coercion_score = self._detect_coercion(t1, t2, poses, frame_idx, timestamp)
                    if coercion_score:
                        anomalies.append(coercion_score)

        # Crowd anomaly detection
        crowd_score = self._detect_crowd_anomaly(tracks, frame_idx, timestamp, frame_shape)
        if crowd_score:
            anomalies.append(crowd_score)

        # Store anomaly history
        for score in anomalies:
            for tid in score.track_ids:
                self._track_anomaly_history[tid].append(score)

        return anomalies

    def _detect_panic(self, track, frame_idx: int, timestamp: float) -> Optional[AnomalyScore]:
        """Detect panic movement: high speed + frequent direction changes."""
        if track.track_length < 10:
            return None

        reasoning = []
        score = 0.0

        # Speed analysis
        if track.avg_speed > self.panic_speed_threshold:
            speed_factor = min(1.0, track.avg_speed / (self.panic_speed_threshold * 2))
            score += speed_factor * 0.4
            reasoning.append(f"High speed: {track.avg_speed:.1f}px/frame (threshold: {self.panic_speed_threshold})")

        # Direction change analysis
        if track.direction_changes > self.panic_direction_changes:
            dir_factor = min(1.0, track.direction_changes / (self.panic_direction_changes * 3))
            score += dir_factor * 0.4
            reasoning.append(f"Frequent direction changes: {track.direction_changes} (threshold: {self.panic_direction_changes})")

        # Erratic velocity variance
        if len(track.velocities) > 5:
            vel_magnitudes = [np.sqrt(v[0]**2 + v[1]**2) for v in track.velocities[-20:]]
            vel_std = np.std(vel_magnitudes)
            if vel_std > 10:
                score += min(0.2, vel_std / 100)
                reasoning.append(f"Erratic speed variance: {vel_std:.1f}")

        if score < 0.3:
            return None

        return AnomalyScore(
            anomaly_type="panic",
            confidence=min(1.0, score),
            class_id=ANOMALY_CLASS_MAP["panic"],
            track_ids=[track.track_id],
            reasoning=reasoning,
            features={"avg_speed": track.avg_speed, "direction_changes": track.direction_changes},
            frame_idx=frame_idx,
            timestamp=timestamp,
        )

    def _detect_assault(self, t1, t2, frame_idx: int, timestamp: float) -> Optional[AnomalyScore]:
        """Detect assault: sudden rapid approach between two persons."""
        if t1.track_length < 5 or t2.track_length < 5:
            return None

        # Current distance
        c1, c2 = t1.center, t2.center
        current_dist = np.sqrt((c1[0]-c2[0])**2 + (c1[1]-c2[1])**2)

        # Check if approaching rapidly
        if len(t1.trajectory) < 5 or len(t2.trajectory) < 5:
            return None

        prev_c1 = t1.trajectory[-5]
        prev_c2 = t2.trajectory[-5]
        prev_dist = np.sqrt((prev_c1[0]-prev_c2[0])**2 + (prev_c1[1]-prev_c2[1])**2)

        closing_speed = (prev_dist - current_dist) / 5  # px/frame closing rate
        approach_speed = max(t1.avg_speed, t2.avg_speed)

        reasoning = []
        score = 0.0

        if closing_speed > 15 and current_dist < 100:
            score += 0.4
            reasoning.append(f"Rapid approach: closing at {closing_speed:.1f}px/frame")

        if approach_speed > self.assault_speed_threshold and current_dist < 60:
            score += 0.4
            reasoning.append(f"High-speed close proximity: {approach_speed:.1f}px/frame at {current_dist:.0f}px")

        if current_dist < 30:
            score += 0.2
            reasoning.append(f"Very close contact: {current_dist:.0f}px")

        if score < 0.4:
            return None

        return AnomalyScore(
            anomaly_type="assault",
            confidence=min(1.0, score),
            class_id=ANOMALY_CLASS_MAP["assault"],
            track_ids=[t1.track_id, t2.track_id],
            reasoning=reasoning,
            features={"closing_speed": closing_speed, "distance": current_dist, "approach_speed": approach_speed},
            frame_idx=frame_idx,
            timestamp=timestamp,
        )

    def _detect_suspicious_escort(self, t1, t2, poses, frame_idx, timestamp) -> Optional[AnomalyScore]:
        """Detect suspicious escort: adult closely following/leading a smaller person."""
        if t1.track_length < 15 or t2.track_length < 15:
            return None

        c1, c2 = t1.center, t2.center
        dist = np.sqrt((c1[0]-c2[0])**2 + (c1[1]-c2[1])**2)

        if dist > self.escort_distance_threshold:
            return None

        reasoning = []
        score = 0.0

        # Consistent close proximity over time
        if t1.track_length >= 20 and t2.track_length >= 20:
            recent_t1 = t1.trajectory[-20:]
            recent_t2 = t2.trajectory[-20:]
            min_len = min(len(recent_t1), len(recent_t2))
            close_frames = sum(
                1 for i in range(min_len)
                if np.sqrt((recent_t1[i][0]-recent_t2[i][0])**2 + (recent_t1[i][1]-recent_t2[i][1])**2) < self.escort_distance_threshold
            )
            proximity_ratio = close_frames / min_len
            if proximity_ratio > 0.7:
                score += 0.3
                reasoning.append(f"Sustained close proximity: {proximity_ratio:.0%} of last {min_len} frames")

        # Size difference (adult-minor indicator)
        area1 = (t1.bbox_xyxy[2]-t1.bbox_xyxy[0]) * (t1.bbox_xyxy[3]-t1.bbox_xyxy[1])
        area2 = (t2.bbox_xyxy[2]-t2.bbox_xyxy[0]) * (t2.bbox_xyxy[3]-t2.bbox_xyxy[1])
        size_ratio = min(area1, area2) / (max(area1, area2) + 1e-8)

        if size_ratio < self.minor_size_ratio:
            score += 0.3
            reasoning.append(f"Significant size difference (possible minor): ratio={size_ratio:.2f}")

        # Same direction of travel
        if len(t1.velocities) >= 5 and len(t2.velocities) >= 5:
            v1 = np.mean(t1.velocities[-5:], axis=0)
            v2 = np.mean(t2.velocities[-5:], axis=0)
            mag1 = np.sqrt(v1[0]**2 + v1[1]**2) + 1e-8
            mag2 = np.sqrt(v2[0]**2 + v2[1]**2) + 1e-8
            cos_sim = (v1[0]*v2[0] + v1[1]*v2[1]) / (mag1 * mag2)
            if cos_sim > 0.7:
                score += 0.2
                reasoning.append(f"Co-directional movement: cos_sim={cos_sim:.2f}")

        # One person leading (ahead in direction of travel)
        if len(t1.velocities) >= 3:
            v_avg = np.mean(t1.velocities[-3:], axis=0)
            if np.sqrt(v_avg[0]**2 + v_avg[1]**2) > 2:
                # Project positions onto direction of travel
                v_norm = v_avg / (np.sqrt(v_avg[0]**2 + v_avg[1]**2) + 1e-8)
                proj_diff = (c1[0]-c2[0]) * v_norm[0] + (c1[1]-c2[1]) * v_norm[1]
                if abs(proj_diff) > 20:
                    score += 0.2
                    leader = t1.track_id if proj_diff > 0 else t2.track_id
                    reasoning.append(f"Leader-follower dynamic: Track {leader} is leading")

        if score < 0.4:
            return None

        return AnomalyScore(
            anomaly_type="suspicious_escort",
            confidence=min(1.0, score),
            class_id=ANOMALY_CLASS_MAP["suspicious_escort"],
            track_ids=[t1.track_id, t2.track_id],
            reasoning=reasoning,
            features={"distance": dist, "size_ratio": size_ratio},
            frame_idx=frame_idx,
            timestamp=timestamp,
        )

    def _detect_dragging(self, t1, t2, poses, frame_idx, timestamp) -> Optional[AnomalyScore]:
        """Detect dragging: one person forcefully moving another."""
        if t1.track_length < 10 or t2.track_length < 10:
            return None

        c1, c2 = t1.center, t2.center
        dist = np.sqrt((c1[0]-c2[0])**2 + (c1[1]-c2[1])**2)
        if dist > 100:
            return None

        reasoning = []
        score = 0.0

        # One moving, other resisting (speed difference + same direction)
        if t1.avg_speed > 5 and t2.avg_speed > 5:
            speed_diff = abs(t1.avg_speed - t2.avg_speed)
            faster = t1 if t1.avg_speed > t2.avg_speed else t2
            slower = t2 if t1.avg_speed > t2.avg_speed else t1

            if speed_diff > 5 and slower.direction_changes > faster.direction_changes * 1.5:
                score += 0.4
                reasoning.append(
                    f"Speed mismatch with resistance: faster={faster.avg_speed:.1f}, "
                    f"slower={slower.avg_speed:.1f}, slower has {slower.direction_changes} direction changes"
                )

        # Pose indicators (if available)
        if poses and len(poses) >= 2:
            for p1 in poses:
                for p2 in poses:
                    if p1 is p2:
                        continue
                    # Check torso angle (being pulled = leaning)
                    if abs(p2.torso_angle) > 20:
                        score += 0.2
                        reasoning.append(f"Person leaning at {p2.torso_angle:.1f}° (possible resistance)")

                    # Arm extension toward other person
                    ext = p1.arm_extension
                    if max(ext) > 0.85:
                        score += 0.2
                        reasoning.append(f"Extended arm reaching toward other person (extension={max(ext):.2f})")

        if score < 0.4:
            return None

        return AnomalyScore(
            anomaly_type="dragging",
            confidence=min(1.0, score),
            class_id=ANOMALY_CLASS_MAP["dragging"],
            track_ids=[t1.track_id, t2.track_id],
            reasoning=reasoning,
            features={"distance": dist, "speed_diff": abs(t1.avg_speed - t2.avg_speed)},
            frame_idx=frame_idx,
            timestamp=timestamp,
        )

    def _detect_coercion(self, t1, t2, poses, frame_idx, timestamp) -> Optional[AnomalyScore]:
        """Detect coercion: one person restraining/controlling another."""
        c1, c2 = t1.center, t2.center
        dist = np.sqrt((c1[0]-c2[0])**2 + (c1[1]-c2[1])**2)
        if dist > 80:
            return None

        reasoning = []
        score = 0.0

        # Very close + one stationary
        if dist < 40 and (t1.is_stationary or t2.is_stationary):
            moving = t1 if not t1.is_stationary else t2
            still = t2 if not t1.is_stationary else t1
            score += 0.3
            reasoning.append(f"Close contact with stationary person: dist={dist:.0f}px")

        # Pose-based restraint detection
        if poses and len(poses) >= 2:
            for p in poses:
                wrist_hip = p.to_dict().get("min_wrist_hip_dist", None)
                if p.arm_extension[0] > 0.8 or p.arm_extension[1] > 0.8:
                    score += 0.2
                    reasoning.append("Extended arm (possible restraint)")
                if abs(p.torso_angle) > 15:
                    score += 0.1
                    reasoning.append(f"Abnormal posture: torso angle {p.torso_angle:.1f}°")

        # Smaller person stationary while larger moves
        area1 = (t1.bbox_xyxy[2]-t1.bbox_xyxy[0]) * (t1.bbox_xyxy[3]-t1.bbox_xyxy[1])
        area2 = (t2.bbox_xyxy[2]-t2.bbox_xyxy[0]) * (t2.bbox_xyxy[3]-t2.bbox_xyxy[1])
        if area1 > area2 * 1.5 and t2.is_stationary and not t1.is_stationary:
            score += 0.3
            reasoning.append("Larger person active near stationary smaller person")

        if score < 0.4:
            return None

        return AnomalyScore(
            anomaly_type="coercion",
            confidence=min(1.0, score),
            class_id=ANOMALY_CLASS_MAP["coercion"],
            track_ids=[t1.track_id, t2.track_id],
            reasoning=reasoning,
            features={"distance": dist},
            frame_idx=frame_idx,
            timestamp=timestamp,
        )

    def _detect_isolated_minor(self, track, all_tracks, pairwise_distances,
                                frame_idx, timestamp, frame_shape) -> Optional[AnomalyScore]:
        """Detect isolated minor: small person alone for extended period."""
        # Estimate if person could be a minor (smaller bounding box)
        h = track.bbox_xyxy[3] - track.bbox_xyxy[1]
        frame_h = frame_shape[0] if frame_shape[0] > 0 else 480
        relative_height = h / frame_h

        if relative_height > 0.35:  # Likely adult
            self._isolation_counters[track.track_id] = 0
            return None

        # Check if isolated (no one nearby)
        min_dist_to_others = float('inf')
        c = track.center
        for other in all_tracks:
            if other.track_id == track.track_id:
                continue
            oc = other.center
            d = np.sqrt((c[0]-oc[0])**2 + (c[1]-oc[1])**2)
            min_dist_to_others = min(min_dist_to_others, d)

        if min_dist_to_others < 150:
            self._isolation_counters[track.track_id] = 0
            return None

        self._isolation_counters[track.track_id] += 1
        isolation_duration = self._isolation_counters[track.track_id]

        if isolation_duration < self.loiter_duration_frames:
            return None

        reasoning = [
            f"Small person (height ratio: {relative_height:.2f}) alone for {isolation_duration} frames",
            f"Nearest person: {min_dist_to_others:.0f}px away",
        ]

        if track.is_stationary:
            reasoning.append("Person is stationary (possible distress)")

        confidence = min(1.0, 0.4 + (isolation_duration - self.loiter_duration_frames) * 0.01)

        return AnomalyScore(
            anomaly_type="isolated_minor",
            confidence=confidence,
            class_id=ANOMALY_CLASS_MAP["isolated_minor"],
            track_ids=[track.track_id],
            reasoning=reasoning,
            features={"relative_height": relative_height, "isolation_duration": isolation_duration,
                      "nearest_person_dist": min_dist_to_others},
            frame_idx=frame_idx,
            timestamp=timestamp,
        )

    def _detect_crowd_anomaly(self, tracks, frame_idx, timestamp, frame_shape) -> Optional[AnomalyScore]:
        """Detect crowd anomalies: unusual density or sudden dispersal."""
        if len(tracks) < 5:
            return None

        reasoning = []
        score = 0.0

        # Compute crowd density
        frame_area = frame_shape[0] * frame_shape[1] if frame_shape[0] > 0 else 640 * 480
        person_area = sum(
            (t.bbox_xyxy[2]-t.bbox_xyxy[0]) * (t.bbox_xyxy[3]-t.bbox_xyxy[1])
            for t in tracks
        )
        density = person_area / frame_area

        if density > self.crowd_density_threshold:
            score += 0.3
            reasoning.append(f"High crowd density: {density:.2%} of frame occupied")

        # Check for simultaneous high-speed movement (stampede/dispersal)
        fast_movers = [t for t in tracks if t.avg_speed > 20]
        if len(fast_movers) > len(tracks) * 0.5:
            score += 0.4
            reasoning.append(f"Mass rapid movement: {len(fast_movers)}/{len(tracks)} persons moving fast")

            # Check if dispersing (moving away from center)
            center_x = np.mean([t.center[0] for t in tracks])
            center_y = np.mean([t.center[1] for t in tracks])
            dispersing = 0
            for t in fast_movers:
                if len(t.velocities) >= 2:
                    v = t.velocities[-1]
                    dx = t.center[0] - center_x
                    dy = t.center[1] - center_y
                    dot = v[0] * dx + v[1] * dy
                    if dot > 0:
                        dispersing += 1
            if dispersing > len(fast_movers) * 0.6:
                score += 0.3
                reasoning.append(f"Crowd dispersal detected: {dispersing}/{len(fast_movers)} moving outward")

        if score < 0.3:
            return None

        return AnomalyScore(
            anomaly_type="crowd_anomaly",
            confidence=min(1.0, score),
            class_id=ANOMALY_CLASS_MAP["crowd_anomaly"],
            track_ids=[t.track_id for t in tracks],
            reasoning=reasoning,
            features={"density": density, "fast_movers": len(fast_movers), "total_persons": len(tracks)},
            frame_idx=frame_idx,
            timestamp=timestamp,
        )

    def get_anomaly_summary(self) -> Dict:
        """Get summary of all detected anomalies."""
        all_anomalies = []
        for tid, scores in self._track_anomaly_history.items():
            all_anomalies.extend(scores)

        by_type = defaultdict(int)
        for a in all_anomalies:
            by_type[a.anomaly_type] += 1

        return {
            "total_anomalies": len(all_anomalies),
            "by_type": dict(by_type),
            "tracks_with_anomalies": len(self._track_anomaly_history),
            "critical_count": sum(1 for a in all_anomalies if a.risk_level == "CRITICAL"),
            "high_count": sum(1 for a in all_anomalies if a.risk_level == "HIGH"),
        }

    def reset(self):
        """Reset analyzer state."""
        self._track_anomaly_history.clear()
        self._isolation_counters.clear()
