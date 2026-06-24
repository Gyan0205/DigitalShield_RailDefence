"""
Digital Shield Rail Defense — YOLOv8 Pose Estimator
=====================================================
Skeleton keypoint extraction for behavioral analysis.
Detects 17 COCO-format body keypoints per person for
posture and interaction analysis.

Keypoints (COCO): nose, L/R eye, L/R ear, L/R shoulder,
  L/R elbow, L/R wrist, L/R hip, L/R knee, L/R ankle
"""

import cv2
import logging
import numpy as np
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass, field

logger = logging.getLogger("pose_estimator")

# COCO keypoint definitions
KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]

SKELETON_CONNECTIONS = [
    (0, 1), (0, 2), (1, 3), (2, 4),  # Head
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # Arms
    (5, 11), (6, 12), (11, 12),  # Torso
    (11, 13), (13, 15), (12, 14), (14, 16),  # Legs
]


@dataclass
class PersonPose:
    """Pose estimation result for a single person."""
    keypoints: np.ndarray  # (17, 3) — x, y, confidence per keypoint
    bbox_xyxy: Tuple[float, float, float, float]
    person_confidence: float
    track_id: Optional[int] = None

    @property
    def keypoint_dict(self) -> Dict[str, Tuple[float, float, float]]:
        return {
            name: (float(self.keypoints[i][0]), float(self.keypoints[i][1]), float(self.keypoints[i][2]))
            for i, name in enumerate(KEYPOINT_NAMES)
            if i < len(self.keypoints)
        }

    def get_keypoint(self, name: str) -> Optional[Tuple[float, float, float]]:
        if name in KEYPOINT_NAMES:
            idx = KEYPOINT_NAMES.index(name)
            if idx < len(self.keypoints):
                return tuple(self.keypoints[idx])
        return None

    @property
    def shoulder_width(self) -> float:
        ls = self.keypoints[5]
        rs = self.keypoints[6]
        if ls[2] > 0.3 and rs[2] > 0.3:
            return float(np.sqrt((ls[0]-rs[0])**2 + (ls[1]-rs[1])**2))
        return 0.0

    @property
    def torso_angle(self) -> float:
        """Torso lean angle (deviation from vertical). Useful for detecting falls/dragging."""
        ls, rs = self.keypoints[5], self.keypoints[6]
        lh, rh = self.keypoints[11], self.keypoints[12]
        if all(k[2] > 0.3 for k in [ls, rs, lh, rh]):
            shoulder_mid = ((ls[0]+rs[0])/2, (ls[1]+rs[1])/2)
            hip_mid = ((lh[0]+rh[0])/2, (lh[1]+rh[1])/2)
            dx = shoulder_mid[0] - hip_mid[0]
            dy = shoulder_mid[1] - hip_mid[1]
            angle = float(np.degrees(np.arctan2(dx, -dy)))
            return angle
        return 0.0

    @property
    def arm_extension(self) -> Tuple[float, float]:
        """Left and right arm extension ratio. High values indicate reaching/grabbing."""
        left_ext = right_ext = 0.0
        ls, le, lw = self.keypoints[5], self.keypoints[7], self.keypoints[9]
        rs, re, rw = self.keypoints[6], self.keypoints[8], self.keypoints[10]
        if all(k[2] > 0.3 for k in [ls, le, lw]):
            arm_len = np.sqrt((ls[0]-le[0])**2 + (ls[1]-le[1])**2) + np.sqrt((le[0]-lw[0])**2 + (le[1]-lw[1])**2)
            reach = np.sqrt((ls[0]-lw[0])**2 + (ls[1]-lw[1])**2)
            left_ext = reach / (arm_len + 1e-8)
        if all(k[2] > 0.3 for k in [rs, re, rw]):
            arm_len = np.sqrt((rs[0]-re[0])**2 + (rs[1]-re[1])**2) + np.sqrt((re[0]-rw[0])**2 + (re[1]-rw[1])**2)
            reach = np.sqrt((rs[0]-rw[0])**2 + (rs[1]-rw[1])**2)
            right_ext = reach / (arm_len + 1e-8)
        return (float(left_ext), float(right_ext))

    def to_dict(self) -> Dict:
        return {
            "bbox_xyxy": list(self.bbox_xyxy),
            "confidence": round(self.person_confidence, 4),
            "track_id": self.track_id,
            "shoulder_width": round(self.shoulder_width, 2),
            "torso_angle": round(self.torso_angle, 2),
            "arm_extension": [round(v, 3) for v in self.arm_extension],
            "keypoints": self.keypoint_dict,
        }


class PoseEstimator:
    """
    YOLOv8-pose based skeleton keypoint estimator.

    Usage:
        estimator = PoseEstimator()
        poses = estimator.estimate(frame)
        features = estimator.extract_behavioral_features(pose)
    """

    def __init__(self, model_variant: str = "yolov8n-pose.pt",
                 confidence_threshold: float = 0.5, device: str = "auto"):
        self.model_variant = model_variant
        self.confidence_threshold = confidence_threshold
        self.device = device
        self.model = None
        self._load_model()

    def _load_model(self):
        try:
            from ultralytics import YOLO
            self.model = YOLO(self.model_variant)
            if self.device == "auto":
                import torch
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"YOLOv8-pose loaded: {self.model_variant} on {self.device}")
        except ImportError:
            logger.warning("ultralytics not installed. Pose estimator will use fallback mode.")
            self.model = None

    def estimate(self, frame: np.ndarray) -> List[PersonPose]:
        """Estimate poses for all persons in frame."""
        if self.model is None:
            logger.warning("Pose model not loaded. Returning empty poses.")
            return []
        results = self.model.predict(
            source=frame, conf=self.confidence_threshold,
            device=self.device, verbose=False,
        )
        poses = []
        for result in results:
            if result.keypoints is None or result.boxes is None:
                continue
            for i, (kps, box) in enumerate(zip(result.keypoints, result.boxes)):
                keypoints = kps.data[0].cpu().numpy()  # (17, 3)
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu().numpy())
                poses.append(PersonPose(
                    keypoints=keypoints,
                    bbox_xyxy=(float(x1), float(y1), float(x2), float(y2)),
                    person_confidence=conf,
                ))
        return poses

    def extract_behavioral_features(self, pose: PersonPose) -> Dict[str, float]:
        """
        Extract behavioral features from a single pose.
        These features are used for anomaly classification.
        """
        features = {
            "torso_angle": pose.torso_angle,
            "shoulder_width": pose.shoulder_width,
            "left_arm_extension": pose.arm_extension[0],
            "right_arm_extension": pose.arm_extension[1],
        }

        # Head orientation (nose relative to shoulders)
        nose = pose.keypoints[0]
        ls, rs = pose.keypoints[5], pose.keypoints[6]
        if all(k[2] > 0.3 for k in [nose, ls, rs]):
            shoulder_mid_x = (ls[0] + rs[0]) / 2
            features["head_offset_x"] = float(nose[0] - shoulder_mid_x)
            features["head_offset_y"] = float(nose[1] - (ls[1] + rs[1]) / 2)
        else:
            features["head_offset_x"] = 0.0
            features["head_offset_y"] = 0.0

        # Leg spread (stance width)
        la, ra = pose.keypoints[15], pose.keypoints[16]
        if la[2] > 0.3 and ra[2] > 0.3:
            features["stance_width"] = float(abs(la[0] - ra[0]))
        else:
            features["stance_width"] = 0.0

        # Body compactness (bbox area vs keypoint spread)
        valid_kps = pose.keypoints[pose.keypoints[:, 2] > 0.3]
        if len(valid_kps) > 2:
            kp_spread = float(np.std(valid_kps[:, :2]))
            features["body_compactness"] = kp_spread
        else:
            features["body_compactness"] = 0.0

        # Wrist-hip proximity (restraint indicator)
        lw, rw = pose.keypoints[9], pose.keypoints[10]
        lh, rh = pose.keypoints[11], pose.keypoints[12]
        min_wrist_hip = float('inf')
        for w in [lw, rw]:
            for h in [lh, rh]:
                if w[2] > 0.3 and h[2] > 0.3:
                    d = np.sqrt((w[0]-h[0])**2 + (w[1]-h[1])**2)
                    min_wrist_hip = min(min_wrist_hip, d)
        features["min_wrist_hip_dist"] = float(min_wrist_hip) if min_wrist_hip != float('inf') else 0.0

        return features

    def extract_interaction_features(self, pose1: PersonPose, pose2: PersonPose) -> Dict[str, float]:
        """
        Extract interaction features between two persons.
        Critical for detecting coercion, escort, and dragging behaviors.
        """
        c1 = ((pose1.bbox_xyxy[0]+pose1.bbox_xyxy[2])/2, (pose1.bbox_xyxy[1]+pose1.bbox_xyxy[3])/2)
        c2 = ((pose2.bbox_xyxy[0]+pose2.bbox_xyxy[2])/2, (pose2.bbox_xyxy[1]+pose2.bbox_xyxy[3])/2)

        features = {
            "interpersonal_distance": float(np.sqrt((c1[0]-c2[0])**2 + (c1[1]-c2[1])**2)),
            "height_ratio": float(min(pose1.bbox_xyxy[3]-pose1.bbox_xyxy[1], pose2.bbox_xyxy[3]-pose2.bbox_xyxy[1]) /
                                  (max(pose1.bbox_xyxy[3]-pose1.bbox_xyxy[1], pose2.bbox_xyxy[3]-pose2.bbox_xyxy[1]) + 1e-8)),
        }

        # Check if wrists of person1 are near person2's body (grabbing indicator)
        for name, idx in [("left_wrist", 9), ("right_wrist", 10)]:
            w = pose1.keypoints[idx]
            if w[2] > 0.3:
                x2_1, y2_1, x2_2, y2_2 = pose2.bbox_xyxy
                inside = x2_1 <= w[0] <= x2_2 and y2_1 <= w[1] <= y2_2
                features[f"p1_{name}_in_p2_bbox"] = 1.0 if inside else 0.0
            else:
                features[f"p1_{name}_in_p2_bbox"] = 0.0

        # Size difference (adult vs minor indicator)
        area1 = (pose1.bbox_xyxy[2]-pose1.bbox_xyxy[0]) * (pose1.bbox_xyxy[3]-pose1.bbox_xyxy[1])
        area2 = (pose2.bbox_xyxy[2]-pose2.bbox_xyxy[0]) * (pose2.bbox_xyxy[3]-pose2.bbox_xyxy[1])
        features["size_ratio"] = float(min(area1, area2) / (max(area1, area2) + 1e-8))
        features["potential_adult_minor"] = 1.0 if features["size_ratio"] < 0.5 else 0.0

        return features

    def draw_poses(self, frame: np.ndarray, poses: List[PersonPose]) -> np.ndarray:
        """Draw skeleton overlays on frame."""
        for pose in poses:
            kps = pose.keypoints
            # Draw skeleton connections
            for i, j in SKELETON_CONNECTIONS:
                if kps[i][2] > 0.3 and kps[j][2] > 0.3:
                    p1 = (int(kps[i][0]), int(kps[i][1]))
                    p2 = (int(kps[j][0]), int(kps[j][1]))
                    cv2.line(frame, p1, p2, (0, 255, 0), 2)
            # Draw keypoints
            for kp in kps:
                if kp[2] > 0.3:
                    cv2.circle(frame, (int(kp[0]), int(kp[1])), 4, (0, 0, 255), -1)
        return frame
