"""
Digital Shield Rail Defense — Model Versioning & Experiment Tracking
=======================================================================
Tracks model versions, training experiments, hyperparameters,
and performance metrics across training runs.
"""

import sys
import json
import hashlib
import shutil
import logging
from pathlib import Path
from typing import Optional, Dict, List, Any
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from ml.config import WEIGHTS_DIR, METADATA_DIR, LOG_FORMAT, LOG_DATE_FORMAT

logger = logging.getLogger("experiment_tracker")
logger.setLevel(logging.INFO)
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter(LOG_FORMAT, LOG_DATE_FORMAT))
    logger.addHandler(_h)


# ============================================================================
# MODEL VERSION REGISTRY
# ============================================================================

class ModelVersion:
    """A single model version with metadata."""

    def __init__(self, model_name: str, version: str, weights_path: str,
                 metrics: Dict = None, hyperparams: Dict = None,
                 description: str = "", tags: List[str] = None):
        self.model_name = model_name
        self.version = version
        self.weights_path = weights_path
        self.metrics = metrics or {}
        self.hyperparams = hyperparams or {}
        self.description = description
        self.tags = tags or []
        self.created_at = datetime.now().isoformat()
        self.checksum = self._compute_checksum()

    def _compute_checksum(self) -> str:
        p = Path(self.weights_path)
        if p.exists():
            return hashlib.md5(p.read_bytes()).hexdigest()[:12]
        return "no_file"

    def to_dict(self) -> Dict:
        return {
            "model_name": self.model_name, "version": self.version,
            "weights_path": self.weights_path, "metrics": self.metrics,
            "hyperparams": self.hyperparams, "description": self.description,
            "tags": self.tags, "created_at": self.created_at,
            "checksum": self.checksum,
        }


class ModelRegistry:
    """Central registry for model versions."""

    def __init__(self, registry_dir: Optional[Path] = None):
        self.dir = registry_dir or METADATA_DIR / "model_registry"
        self.dir.mkdir(parents=True, exist_ok=True)
        self._registry_file = self.dir / "registry.json"
        self._versions: List[Dict] = self._load()

    def _load(self) -> List[Dict]:
        if self._registry_file.exists():
            with open(self._registry_file) as f:
                return json.load(f)
        return []

    def _save(self):
        with open(self._registry_file, "w") as f:
            json.dump(self._versions, f, indent=2)

    def register(self, version: ModelVersion) -> str:
        """Register a new model version."""
        self._versions.append(version.to_dict())
        self._save()
        logger.info(f"Registered: {version.model_name} v{version.version}")
        return version.version

    def get_latest(self, model_name: str) -> Optional[Dict]:
        matches = [v for v in self._versions if v["model_name"] == model_name]
        return matches[-1] if matches else None

    def get_best(self, model_name: str, metric: str = "accuracy") -> Optional[Dict]:
        matches = [v for v in self._versions if v["model_name"] == model_name]
        if not matches:
            return None
        return max(matches, key=lambda v: v.get("metrics", {}).get(metric, 0))

    def list_versions(self, model_name: Optional[str] = None) -> List[Dict]:
        if model_name:
            return [v for v in self._versions if v["model_name"] == model_name]
        return self._versions

    def get_stats(self) -> Dict:
        models = {}
        for v in self._versions:
            name = v["model_name"]
            if name not in models:
                models[name] = {"count": 0, "versions": []}
            models[name]["count"] += 1
            models[name]["versions"].append(v["version"])
        return {"total_versions": len(self._versions), "models": models}


# ============================================================================
# EXPERIMENT TRACKER
# ============================================================================

class ExperimentTracker:
    """
    Track training experiments with hyperparameters and metrics.

    Usage:
        tracker = ExperimentTracker()
        exp = tracker.start_experiment("yolo_person_detect", {"epochs": 100, "lr": 0.01})
        # ... training ...
        tracker.log_metric(exp, "mAP50", 0.85)
        tracker.end_experiment(exp)
    """

    def __init__(self, experiments_dir: Optional[Path] = None):
        self.dir = experiments_dir or METADATA_DIR / "experiments"
        self.dir.mkdir(parents=True, exist_ok=True)
        self._experiments: List[Dict] = self._load_all()

    def _load_all(self) -> List[Dict]:
        index_path = self.dir / "index.json"
        if index_path.exists():
            with open(index_path) as f:
                return json.load(f)
        return []

    def _save_index(self):
        with open(self.dir / "index.json", "w") as f:
            json.dump(self._experiments, f, indent=2)

    def start_experiment(self, name: str, hyperparams: Dict, tags: List[str] = None) -> str:
        """Start a new experiment. Returns experiment ID."""
        exp_id = f"EXP_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{name[:20]}"
        exp = {
            "experiment_id": exp_id,
            "name": name,
            "status": "running",
            "hyperparams": hyperparams,
            "metrics": {},
            "metric_history": {},
            "tags": tags or [],
            "started_at": datetime.now().isoformat(),
            "ended_at": None,
            "duration_seconds": None,
            "notes": "",
        }
        self._experiments.append(exp)
        self._save_index()

        # Save individual experiment file
        with open(self.dir / f"{exp_id}.json", "w") as f:
            json.dump(exp, f, indent=2)

        logger.info(f"Started experiment: {exp_id}")
        return exp_id

    def log_metric(self, exp_id: str, metric_name: str, value: float, step: int = None):
        """Log a metric value for an experiment."""
        for exp in self._experiments:
            if exp["experiment_id"] == exp_id:
                exp["metrics"][metric_name] = value
                if metric_name not in exp["metric_history"]:
                    exp["metric_history"][metric_name] = []
                exp["metric_history"][metric_name].append({
                    "value": value, "step": step, "timestamp": datetime.now().isoformat(),
                })
                break
        self._save_index()

    def log_metrics(self, exp_id: str, metrics: Dict[str, float], step: int = None):
        """Log multiple metrics at once."""
        for name, value in metrics.items():
            self.log_metric(exp_id, name, value, step)

    def end_experiment(self, exp_id: str, status: str = "completed", notes: str = ""):
        """End an experiment."""
        for exp in self._experiments:
            if exp["experiment_id"] == exp_id:
                exp["status"] = status
                exp["ended_at"] = datetime.now().isoformat()
                exp["notes"] = notes
                start = datetime.fromisoformat(exp["started_at"])
                exp["duration_seconds"] = round((datetime.now() - start).total_seconds(), 2)
                break

        self._save_index()
        logger.info(f"Ended experiment: {exp_id} ({status})")

    def get_experiment(self, exp_id: str) -> Optional[Dict]:
        for exp in self._experiments:
            if exp["experiment_id"] == exp_id:
                return exp
        return None

    def compare_experiments(self, exp_ids: List[str], metric: str = "accuracy") -> List[Dict]:
        """Compare experiments by a specific metric."""
        results = []
        for exp_id in exp_ids:
            exp = self.get_experiment(exp_id)
            if exp:
                results.append({
                    "experiment_id": exp_id,
                    "name": exp["name"],
                    metric: exp.get("metrics", {}).get(metric, None),
                    "hyperparams": exp.get("hyperparams", {}),
                    "status": exp["status"],
                })
        return sorted(results, key=lambda x: x.get(metric, 0) or 0, reverse=True)

    def get_best_experiment(self, name: str = None, metric: str = "accuracy") -> Optional[Dict]:
        """Get the best experiment by metric."""
        candidates = self._experiments
        if name:
            candidates = [e for e in candidates if e["name"] == name]
        if not candidates:
            return None
        return max(candidates, key=lambda e: e.get("metrics", {}).get(metric, 0))

    def list_experiments(self, name: str = None, status: str = None) -> List[Dict]:
        results = self._experiments
        if name:
            results = [e for e in results if e["name"] == name]
        if status:
            results = [e for e in results if e["status"] == status]
        return results

    def get_stats(self) -> Dict:
        statuses = {}
        for e in self._experiments:
            s = e["status"]
            statuses[s] = statuses.get(s, 0) + 1
        return {
            "total_experiments": len(self._experiments),
            "by_status": statuses,
        }


if __name__ == "__main__":
    # Demo
    registry = ModelRegistry()
    tracker = ExperimentTracker()

    # Register a model
    v = ModelVersion(
        model_name="anomaly_lstm", version="1.0.0",
        weights_path="dataset/models/weights/anomaly_classifier_best.pth",
        metrics={"accuracy": 0.92, "macro_f1": 0.89},
        hyperparams={"hidden_size": 256, "lr": 1e-3, "epochs": 50},
        description="Initial LSTM anomaly classifier",
        tags=["baseline", "lstm"],
    )
    registry.register(v)

    # Run experiment
    exp_id = tracker.start_experiment(
        "anomaly_lstm_v2",
        {"hidden_size": 512, "lr": 5e-4, "epochs": 100},
        tags=["improved", "lstm"],
    )
    tracker.log_metrics(exp_id, {"accuracy": 0.94, "macro_f1": 0.91}, step=100)
    tracker.end_experiment(exp_id, notes="Larger hidden size improves F1")

    print(f"Registry: {registry.get_stats()}")
    print(f"Experiments: {tracker.get_stats()}")
