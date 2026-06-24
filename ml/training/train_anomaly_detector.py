"""
Digital Shield Rail Defense — Anomaly Detection Training
==========================================================
End-to-end training workflow for the LSTM/GRU anomaly classifier.
Handles data loading, feature extraction, model training,
validation, and checkpoint management.
"""

import sys
import json
import time
import logging
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from ml.config import (
    ANNOTATIONS_DIR, METADATA_DIR, WEIGHTS_DIR, CHECKPOINTS_DIR,
    DEFAULT_ANOMALY_CLF, LOG_FORMAT, LOG_DATE_FORMAT,
)

logger = logging.getLogger("train_anomaly")
logger.setLevel(logging.INFO)
if not logger.handlers:
    console = logging.StreamHandler()
    console.setFormatter(logging.Formatter(LOG_FORMAT, LOG_DATE_FORMAT))
    logger.addHandler(console)

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class AnomalyFeatureDataset:
    """
    Dataset that loads pre-extracted feature sequences for training.
    Each sample is a (sequence, label) pair where sequence is
    a temporal window of behavioral feature vectors.
    """

    def __init__(self, features: np.ndarray, labels: np.ndarray,
                 sequence_length: int = 32):
        """
        Args:
            features: (N, feature_dim) array of frame-level features
            labels: (N,) array of per-frame class labels
            sequence_length: temporal window size
        """
        self.features = features
        self.labels = labels
        self.sequence_length = sequence_length
        self.samples = self._create_sequences()

    def _create_sequences(self) -> List[Tuple[np.ndarray, int]]:
        """Create sliding window sequences."""
        samples = []
        for i in range(len(self.features) - self.sequence_length + 1):
            seq = self.features[i:i + self.sequence_length]
            # Majority vote label for the sequence
            seq_labels = self.labels[i:i + self.sequence_length]
            label = int(np.bincount(seq_labels.astype(int)).argmax())
            samples.append((seq, label))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        seq, label = self.samples[idx]
        if TORCH_AVAILABLE:
            return torch.FloatTensor(seq), torch.LongTensor([label])[0]
        return seq, label


def generate_synthetic_training_data(
    num_samples: int = 5000,
    feature_dim: int = 32,
    num_classes: int = 8,
    sequence_length: int = 32,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic training data for initial model development.
    Creates feature sequences with class-specific patterns.
    """
    np.random.seed(seed)
    features = []
    labels = []

    samples_per_class = num_samples // num_classes

    for class_id in range(num_classes):
        for _ in range(samples_per_class):
            seq = np.random.randn(sequence_length, feature_dim).astype(np.float32) * 0.3

            if class_id == 0:  # Normal
                seq[:, 9] = np.random.uniform(1, 5, sequence_length)  # Low speed
                seq[:, 11] = 0  # No direction changes
            elif class_id == 1:  # Assault
                seq[:, 9] = np.random.uniform(30, 50, sequence_length)  # High speed
                seq[:, 14] = np.random.uniform(10, 40, sequence_length)  # Close distance
            elif class_id == 2:  # Coercion
                seq[:, 0] = np.random.uniform(10, 30, sequence_length)  # Abnormal torso
                seq[:, 14] = np.random.uniform(20, 50, sequence_length)  # Close
                seq[:, 3] = np.random.uniform(0.7, 1.0, sequence_length)  # Arm extension
            elif class_id == 3:  # Dragging
                seq[:, 9] = np.random.uniform(10, 25, sequence_length)  # Moderate speed
                seq[:, 11] = np.random.uniform(3, 8, sequence_length)  # Some direction changes
                seq[:, 0] = np.random.uniform(15, 40, sequence_length)  # Leaning torso
            elif class_id == 4:  # Suspicious escort
                seq[:, 14] = np.random.uniform(30, 70, sequence_length)  # Consistent proximity
                seq[:, 16] = np.random.uniform(0.3, 0.5, sequence_length)  # Size difference
                seq[:, 9] = np.random.uniform(3, 8, sequence_length)  # Walking speed
            elif class_id == 5:  # Isolated minor
                seq[:, 12] = np.random.uniform(50, 100, sequence_length)  # Stationary
                seq[:, 14] = np.random.uniform(150, 300, sequence_length)  # Far from others
            elif class_id == 6:  # Panic
                seq[:, 9] = np.random.uniform(25, 60, sequence_length)  # High speed
                seq[:, 11] = np.random.uniform(5, 15, sequence_length)  # Many direction changes
            elif class_id == 7:  # Crowd anomaly
                seq[:, 20] = np.random.uniform(0.3, 0.6, sequence_length)  # High density
                seq[:, 21] = np.random.uniform(5, 15, sequence_length)  # Many fast movers

            features.append(seq)
            labels.extend([class_id] * sequence_length)

    features = np.concatenate(features, axis=0)
    labels = np.array(labels)

    # Shuffle
    indices = np.random.permutation(len(features))
    return features[indices], labels[indices]


def train_model(
    epochs: int = 50,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    sequence_length: int = 32,
    feature_dim: int = 32,
    hidden_size: int = 256,
    num_classes: int = 8,
    model_type: str = "lstm",
    patience: int = 10,
    use_synthetic: bool = True,
    output_dir: Optional[Path] = None,
) -> Dict:
    """
    Train the anomaly classification model.

    Returns:
        Training report with metrics
    """
    if not TORCH_AVAILABLE:
        logger.error("PyTorch required for training. Install: pip install torch")
        return {"status": "error", "message": "PyTorch not available"}

    out_dir = output_dir or WEIGHTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("="*60)
    logger.info("ANOMALY CLASSIFIER TRAINING")
    logger.info("="*60)
    logger.info(f"Model: {model_type} | Hidden: {hidden_size} | Classes: {num_classes}")
    logger.info(f"Epochs: {epochs} | Batch: {batch_size} | LR: {learning_rate}")

    # Generate or load data
    if use_synthetic:
        logger.info("Generating synthetic training data...")
        features, labels = generate_synthetic_training_data(
            num_samples=5000, feature_dim=feature_dim,
            num_classes=num_classes, sequence_length=sequence_length,
        )
    else:
        # Load from processed dataset
        features_path = METADATA_DIR / "training_features.npy"
        labels_path = METADATA_DIR / "training_labels.npy"
        if features_path.exists() and labels_path.exists():
            features = np.load(features_path)
            labels = np.load(labels_path)
        else:
            logger.warning("No training data found. Using synthetic data.")
            features, labels = generate_synthetic_training_data()

    # Split data
    n = len(features)
    train_end = int(n * 0.7)
    val_end = int(n * 0.85)

    train_dataset = AnomalyFeatureDataset(features[:train_end], labels[:train_end], sequence_length)
    val_dataset = AnomalyFeatureDataset(features[train_end:val_end], labels[train_end:val_end], sequence_length)
    test_dataset = AnomalyFeatureDataset(features[val_end:], labels[val_end:], sequence_length)

    logger.info(f"Data: train={len(train_dataset)}, val={len(val_dataset)}, test={len(test_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # Build model
    from ml.models.anomaly_classifier import AnomalyLSTM, AnomalyGRU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model_type == "lstm":
        model = AnomalyLSTM(feature_dim, hidden_size, num_classes=num_classes)
    else:
        model = AnomalyGRU(feature_dim, hidden_size, num_classes=num_classes)

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    # Training loop
    best_val_loss = float('inf')
    best_val_acc = 0
    patience_counter = 0
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    start_time = time.time()

    for epoch in range(epochs):
        # Train
        model.train()
        train_loss, train_correct, train_total = 0, 0, 0

        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item() * len(batch_y)
            train_correct += (logits.argmax(1) == batch_y).sum().item()
            train_total += len(batch_y)

        train_loss /= train_total
        train_acc = train_correct / train_total

        # Validate
        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                logits = model(batch_x)
                loss = criterion(logits, batch_y)
                val_loss += loss.item() * len(batch_y)
                val_correct += (logits.argmax(1) == batch_y).sum().item()
                val_total += len(batch_y)

        val_loss /= max(val_total, 1)
        val_acc = val_correct / max(val_total, 1)
        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        logger.info(
            f"Epoch {epoch+1}/{epochs} | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}"
        )

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            patience_counter = 0
            # Save best checkpoint
            torch.save(model.state_dict(), out_dir / "anomaly_classifier_best.pth")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break

    elapsed = time.time() - start_time

    # Final save
    torch.save(model.state_dict(), out_dir / "anomaly_classifier_final.pth")

    report = {
        "status": "complete",
        "model_type": model_type,
        "epochs_trained": epoch + 1,
        "best_val_loss": round(best_val_loss, 6),
        "best_val_acc": round(best_val_acc, 4),
        "final_train_acc": round(train_acc, 4),
        "training_time_seconds": round(elapsed, 2),
        "device": str(device),
        "weights_path": str(out_dir / "anomaly_classifier_best.pth"),
    }

    report_path = METADATA_DIR / "training_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    logger.info(f"\nTraining complete in {elapsed:.1f}s")
    logger.info(f"Best val accuracy: {best_val_acc:.4f}")
    logger.info(f"Weights saved: {out_dir / 'anomaly_classifier_best.pth'}")

    return report


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train Anomaly Classifier")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--model", choices=["lstm", "gru"], default="lstm")
    parser.add_argument("--synthetic", action="store_true", default=True)
    args = parser.parse_args()

    train_model(
        epochs=args.epochs, batch_size=args.batch_size,
        learning_rate=args.lr, model_type=args.model,
        use_synthetic=args.synthetic,
    )
