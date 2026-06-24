"""
Digital Shield Rail Defense — LSTM Anomaly Classifier
=======================================================
Temporal anomaly classifier using LSTM/GRU networks.
Processes sequences of behavioral features to classify
anomaly types over time windows.
"""

import logging
import numpy as np
from typing import Optional, Dict, List, Tuple
from pathlib import Path

logger = logging.getLogger("anomaly_classifier")

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available. Anomaly classifier will use rule-based fallback.")


if TORCH_AVAILABLE:
    class AnomalyLSTM(nn.Module):
        """LSTM-based temporal anomaly classifier."""

        def __init__(self, input_size: int = 32, hidden_size: int = 256,
                     num_layers: int = 2, num_classes: int = 8, dropout: float = 0.3):
            super().__init__()
            self.hidden_size = hidden_size
            self.num_layers = num_layers

            self.lstm = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0,
                bidirectional=True,
            )

            self.attention = nn.Sequential(
                nn.Linear(hidden_size * 2, hidden_size),
                nn.Tanh(),
                nn.Linear(hidden_size, 1),
            )

            self.classifier = nn.Sequential(
                nn.Linear(hidden_size * 2, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size // 2, num_classes),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """
            Args:
                x: (batch, seq_len, input_size)
            Returns:
                logits: (batch, num_classes)
            """
            lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden*2)

            # Attention mechanism
            attn_weights = self.attention(lstm_out)  # (batch, seq_len, 1)
            attn_weights = torch.softmax(attn_weights, dim=1)
            context = torch.sum(lstm_out * attn_weights, dim=1)  # (batch, hidden*2)

            logits = self.classifier(context)
            return logits

    class AnomalyGRU(nn.Module):
        """GRU-based temporal anomaly classifier (lighter alternative)."""

        def __init__(self, input_size: int = 32, hidden_size: int = 128,
                     num_layers: int = 2, num_classes: int = 8, dropout: float = 0.3):
            super().__init__()
            self.gru = nn.GRU(
                input_size=input_size, hidden_size=hidden_size,
                num_layers=num_layers, batch_first=True,
                dropout=dropout if num_layers > 1 else 0,
            )
            self.classifier = nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size // 2, num_classes),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            _, hidden = self.gru(x)
            logits = self.classifier(hidden[-1])
            return logits


class AnomalyClassifier:
    """
    Temporal anomaly classifier that processes sequences
    of per-frame features to classify behavior over time.

    Supports both LSTM (with attention) and GRU architectures,
    plus a rule-based fallback when PyTorch is unavailable.

    Usage:
        classifier = AnomalyClassifier()
        result = classifier.classify_sequence(feature_sequence)
    """

    CLASS_NAMES = {
        0: "normal", 1: "assault", 2: "coercion", 3: "dragging",
        4: "suspicious_escort", 5: "isolated_minor", 6: "panic", 7: "crowd_anomaly",
    }

    def __init__(self, model_type: str = "lstm", weights_path: Optional[str] = None,
                 input_size: int = 32, hidden_size: int = 256,
                 num_classes: int = 8, device: str = "auto"):
        self.model_type = model_type
        self.input_size = input_size
        self.num_classes = num_classes
        self.device = device
        self.model = None

        if TORCH_AVAILABLE:
            if device == "auto":
                self.device = "cuda" if torch.cuda.is_available() else "cpu"

            if model_type == "lstm":
                self.model = AnomalyLSTM(input_size, hidden_size, num_classes=num_classes)
            else:
                self.model = AnomalyGRU(input_size, hidden_size, num_classes=num_classes)

            if weights_path and Path(weights_path).exists():
                state_dict = torch.load(weights_path, map_location=self.device)
                self.model.load_state_dict(state_dict)
                logger.info(f"Loaded weights from {weights_path}")

            self.model.to(self.device)
            self.model.eval()
            logger.info(f"AnomalyClassifier initialized: {model_type} on {self.device}")
        else:
            logger.info("AnomalyClassifier: using rule-based fallback (no PyTorch)")

    def features_to_vector(self, behavioral_features: Dict[str, float],
                           track_features: Dict[str, float] = None) -> np.ndarray:
        """
        Convert behavioral feature dict to fixed-size vector.
        Ensures consistent feature ordering for model input.
        """
        feature_keys = [
            "torso_angle", "shoulder_width", "left_arm_extension", "right_arm_extension",
            "head_offset_x", "head_offset_y", "stance_width", "body_compactness",
            "min_wrist_hip_dist",
            # Track features
            "avg_speed", "total_distance", "direction_changes", "stationary_frames",
            "is_stationary",
            # Interaction features
            "interpersonal_distance", "height_ratio", "size_ratio",
            "p1_left_wrist_in_p2_bbox", "p1_right_wrist_in_p2_bbox",
            "potential_adult_minor",
            # Crowd features
            "density", "fast_movers", "total_persons",
        ]

        combined = {}
        if behavioral_features:
            combined.update(behavioral_features)
        if track_features:
            combined.update(track_features)

        vector = np.zeros(self.input_size, dtype=np.float32)
        for i, key in enumerate(feature_keys[:self.input_size]):
            vector[i] = combined.get(key, 0.0)

        return vector

    def classify_sequence(self, feature_sequence: np.ndarray) -> Dict:
        """
        Classify a temporal sequence of feature vectors.

        Args:
            feature_sequence: (seq_len, input_size) array

        Returns:
            Dict with class_id, class_name, confidence, all_probabilities
        """
        if self.model is not None and TORCH_AVAILABLE:
            return self._classify_neural(feature_sequence)
        return self._classify_rule_based(feature_sequence)

    def _classify_neural(self, feature_sequence: np.ndarray) -> Dict:
        """Neural network classification."""
        with torch.no_grad():
            x = torch.FloatTensor(feature_sequence).unsqueeze(0).to(self.device)
            logits = self.model(x)
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
            class_id = int(np.argmax(probs))

        return {
            "class_id": class_id,
            "class_name": self.CLASS_NAMES.get(class_id, "unknown"),
            "confidence": float(probs[class_id]),
            "probabilities": {self.CLASS_NAMES[i]: float(p) for i, p in enumerate(probs)},
        }

    def _classify_rule_based(self, feature_sequence: np.ndarray) -> Dict:
        """Rule-based fallback classification."""
        if len(feature_sequence) == 0:
            return {"class_id": 0, "class_name": "normal", "confidence": 1.0, "probabilities": {}}

        # Aggregate features over sequence
        mean_features = np.mean(feature_sequence, axis=0)
        max_features = np.max(feature_sequence, axis=0)

        scores = {name: 0.0 for name in self.CLASS_NAMES.values()}
        scores["normal"] = 0.5  # Default baseline

        # Simple rule-based scoring using feature indices
        # Indices: 0=torso_angle, 9=avg_speed, 11=dir_changes, 14=interp_dist
        if self.input_size > 9 and max_features[9] > 30:
            scores["panic"] = max(scores["panic"], min(1.0, max_features[9] / 50))
        if self.input_size > 11 and max_features[11] > 5:
            scores["panic"] = max(scores["panic"], min(1.0, max_features[11] / 15))
        if self.input_size > 14 and mean_features[14] > 0 and mean_features[14] < 50:
            scores["assault"] = max(scores["assault"], 0.5)

        best_class = max(scores, key=scores.get)
        class_id = {v: k for k, v in self.CLASS_NAMES.items()}.get(best_class, 0)

        return {
            "class_id": class_id,
            "class_name": best_class,
            "confidence": scores[best_class],
            "probabilities": scores,
        }

    def save_weights(self, path: str):
        """Save model weights."""
        if self.model and TORCH_AVAILABLE:
            torch.save(self.model.state_dict(), path)
            logger.info(f"Weights saved: {path}")
