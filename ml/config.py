"""
Digital Shield Rail Defense — ML Configuration
Central configuration for all ML pipeline parameters, dataset paths,
model hyperparameters, and processing constants.
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List


# ============================================================================
# PATH CONFIGURATION
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_ROOT = PROJECT_ROOT / "dataset"
ML_ROOT = PROJECT_ROOT / "ml"

# Dataset directory structure
RAW_DIR = DATASET_ROOT / "raw"
PROCESSED_DIR = DATASET_ROOT / "processed"
ANNOTATIONS_DIR = DATASET_ROOT / "annotations"
METADATA_DIR = DATASET_ROOT / "metadata"
MODELS_DIR = DATASET_ROOT / "models"
WEIGHTS_DIR = MODELS_DIR / "weights"
CHECKPOINTS_DIR = MODELS_DIR / "checkpoints"

# Processed subdirectories
FRAMES_DIR = PROCESSED_DIR / "frames"
CLIPS_DIR = PROCESSED_DIR / "clips"
FEATURES_DIR = PROCESSED_DIR / "features"

# Raw dataset subdirectories
UCF_CRIME_DIR = RAW_DIR / "ucf_crime"
UCSD_PED_DIR = RAW_DIR / "ucsd_pedestrian"
SHANGHAITECH_DIR = RAW_DIR / "shanghaitech"
RAILWAY_CCTV_DIR = RAW_DIR / "railway_cctv"
SIMULATED_DIR = RAW_DIR / "simulated"


# ============================================================================
# DATASET SOURCES
# ============================================================================

@dataclass
class DatasetSource:
    """Configuration for a single dataset source."""
    name: str
    url: str
    description: str
    expected_size_gb: float
    file_type: str  # 'zip', 'tar.gz', 'rar', 'direct'
    target_dir: Path
    mirror_urls: List[str] = field(default_factory=list)
    requires_auth: bool = False
    sha256: str = ""


DATASET_SOURCES: Dict[str, DatasetSource] = {
    "ucf_crime": DatasetSource(
        name="UCF Crime Dataset",
        url="https://www.kaggle.com/api/v1/datasets/download/odins0n/ucf-crime-dataset",
        description="Large-scale anomaly detection dataset with 13 crime categories (1900 videos)",
        expected_size_gb=13.0,
        file_type="zip",
        target_dir=UCF_CRIME_DIR,
        mirror_urls=[
            "https://www.crcv.ucf.edu/data/UCF_Crimes.zip",
        ],
        requires_auth=True,
    ),
    "ucsd_pedestrian": DatasetSource(
        name="UCSD Pedestrian Anomaly Dataset",
        url="http://www.svcl.ucsd.edu/projects/anomaly/UCSD_Anomaly_Dataset.tar.gz",
        description="Pedestrian anomaly detection (Ped1: 70 clips, Ped2: 28 clips)",
        expected_size_gb=0.5,
        file_type="tar.gz",
        target_dir=UCSD_PED_DIR,
        mirror_urls=[
            "https://drive.google.com/uc?id=1dvnMJsGXqMAGE5pY_cGILiEPqLjLYeq_",
        ],
    ),
    "shanghaitech": DatasetSource(
        name="ShanghaiTech Campus Dataset",
        url="https://svip-lab.github.io/dataset/campus_dataset.html",
        description="Campus surveillance anomaly dataset (437 videos, 130 abnormal events)",
        expected_size_gb=2.0,
        file_type="zip",
        target_dir=SHANGHAITECH_DIR,
        mirror_urls=[
            "https://drive.google.com/uc?id=1rB1dWLKSGIjsNA6subNfiPjLiFbMXnCm",
        ],
    ),
    "railway_cctv": DatasetSource(
        name="Railway CCTV Compilation",
        url="",  # Curated from public domain sources
        description="Public domain railway station CCTV footage for domain adaptation",
        expected_size_gb=1.0,
        file_type="direct",
        target_dir=RAILWAY_CCTV_DIR,
    ),
    "simulated": DatasetSource(
        name="Simulated Railway Scenarios",
        url="",  # Self-generated
        description="Synthetically generated railway platform scenarios with annotations",
        expected_size_gb=0.5,
        file_type="direct",
        target_dir=SIMULATED_DIR,
    ),
}


# ============================================================================
# ANOMALY CATEGORIES
# ============================================================================

ANOMALY_CLASSES = {
    0: "normal",
    1: "assault",
    2: "coercion",
    3: "dragging",
    4: "suspicious_escort",
    5: "isolated_minor",
    6: "panic_behavior",
    7: "theft",
    8: "vandalism",
    9: "loitering",
    10: "fighting",
    11: "robbery",
    12: "shooting",
    13: "arson",
}

# Trafficking-specific anomaly types (subset)
TRAFFICKING_CLASSES = {
    1: "assault",
    2: "coercion",
    3: "dragging",
    4: "suspicious_escort",
    5: "isolated_minor",
    6: "panic_behavior",
}

# UCF Crime category mapping to our anomaly classes
UCF_CRIME_MAPPING = {
    "Abuse": 1,         # → assault
    "Arrest": 0,        # → normal (law enforcement)
    "Arson": 13,        # → arson
    "Assault": 1,       # → assault
    "Burglary": 0,      # → normal (property)
    "Explosion": 0,     # → normal
    "Fighting": 10,     # → fighting
    "Normal": 0,        # → normal
    "RoadAccidents": 0, # → normal
    "Robbery": 11,      # → robbery
    "Shooting": 12,     # → shooting
    "Shoplifting": 7,   # → theft
    "Stealing": 7,      # → theft
    "Vandalism": 8,     # → vandalism
}


# ============================================================================
# PREPROCESSING PARAMETERS
# ============================================================================

@dataclass
class PreprocessConfig:
    """Video preprocessing configuration."""
    target_fps: int = 2
    frame_width: int = 640
    frame_height: int = 480
    clip_duration_seconds: int = 16
    clip_stride_seconds: int = 8
    max_frames_per_video: int = 300
    min_frames_per_video: int = 10
    image_format: str = "jpg"
    image_quality: int = 95
    normalize_brightness: bool = True
    apply_clahe: bool = True
    clahe_clip_limit: float = 2.0
    clahe_tile_grid: tuple = (8, 8)


@dataclass
class AugmentationConfig:
    """Data augmentation parameters."""
    horizontal_flip: bool = True
    random_crop_scale: tuple = (0.8, 1.0)
    color_jitter_brightness: float = 0.2
    color_jitter_contrast: float = 0.2
    color_jitter_saturation: float = 0.1
    gaussian_noise_std: float = 0.01
    rotation_degrees: int = 5


# ============================================================================
# MODEL CONFIGURATION
# ============================================================================

@dataclass
class YOLOConfig:
    """YOLOv8 detection model configuration."""
    model_variant: str = "yolov8n.pt"
    confidence_threshold: float = 0.5
    iou_threshold: float = 0.45
    max_detections: int = 50
    person_class_id: int = 0
    input_size: int = 640
    device: str = "auto"  # 'auto', 'cuda', 'cpu'


@dataclass
class DeepSORTConfig:
    """DeepSORT tracker configuration."""
    max_age: int = 30
    n_init: int = 3
    max_iou_distance: float = 0.7
    max_cosine_distance: float = 0.3
    nn_budget: int = 100
    embedder: str = "mobilenet"
    half: bool = True


@dataclass
class AnomalyClassifierConfig:
    """LSTM-based anomaly classifier configuration."""
    sequence_length: int = 32
    hidden_size: int = 256
    num_layers: int = 2
    dropout: float = 0.3
    num_classes: int = 14
    learning_rate: float = 1e-3
    batch_size: int = 32
    epochs: int = 50
    patience: int = 10


@dataclass
class XGBoostConfig:
    """XGBoost ticket anomaly scorer configuration."""
    n_estimators: int = 500
    max_depth: int = 8
    learning_rate: float = 0.05
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    min_child_weight: int = 3
    gamma: float = 0.1
    reg_alpha: float = 0.1
    reg_lambda: float = 1.0
    scale_pos_weight: float = 10.0
    eval_metric: str = "logloss"
    early_stopping_rounds: int = 20


# ============================================================================
# RAILWAY METADATA CONFIGURATION
# ============================================================================

INDIAN_RAILWAY_STATIONS = [
    {"code": "NDLS", "name": "New Delhi", "city": "New Delhi", "state": "Delhi", "zone": "Northern", "lat": 28.6425, "lng": 77.2196, "platforms": 16},
    {"code": "CSTM", "name": "Chhatrapati Shivaji Terminus", "city": "Mumbai", "state": "Maharashtra", "zone": "Central", "lat": 18.9398, "lng": 72.8355, "platforms": 18},
    {"code": "HWH", "name": "Howrah Junction", "city": "Kolkata", "state": "West Bengal", "zone": "Eastern", "lat": 22.5836, "lng": 88.3422, "platforms": 23},
    {"code": "MAS", "name": "Chennai Central", "city": "Chennai", "state": "Tamil Nadu", "zone": "Southern", "lat": 13.0827, "lng": 80.2755, "platforms": 17},
    {"code": "SBC", "name": "KSR Bengaluru", "city": "Bengaluru", "state": "Karnataka", "zone": "South Western", "lat": 12.9779, "lng": 77.5661, "platforms": 10},
    {"code": "SC", "name": "Secunderabad Junction", "city": "Hyderabad", "state": "Telangana", "zone": "South Central", "lat": 17.4337, "lng": 78.5016, "platforms": 10},
    {"code": "JP", "name": "Jaipur Junction", "city": "Jaipur", "state": "Rajasthan", "zone": "North Western", "lat": 26.9196, "lng": 75.7878, "platforms": 7},
    {"code": "LKO", "name": "Lucknow Charbagh", "city": "Lucknow", "state": "Uttar Pradesh", "zone": "Northern", "lat": 26.8314, "lng": 80.9204, "platforms": 9},
    {"code": "ADI", "name": "Ahmedabad Junction", "city": "Ahmedabad", "state": "Gujarat", "zone": "Western", "lat": 23.0258, "lng": 72.6004, "platforms": 12},
    {"code": "PUNE", "name": "Pune Junction", "city": "Pune", "state": "Maharashtra", "zone": "Central", "lat": 18.5287, "lng": 73.8745, "platforms": 6},
    {"code": "BPL", "name": "Bhopal Junction", "city": "Bhopal", "state": "Madhya Pradesh", "zone": "West Central", "lat": 23.2688, "lng": 77.4134, "platforms": 6},
    {"code": "CNB", "name": "Kanpur Central", "city": "Kanpur", "state": "Uttar Pradesh", "zone": "North Central", "lat": 26.4534, "lng": 80.3515, "platforms": 10},
    {"code": "PNBE", "name": "Patna Junction", "city": "Patna", "state": "Bihar", "zone": "East Central", "lat": 25.6079, "lng": 85.1001, "platforms": 10},
    {"code": "GHY", "name": "Guwahati", "city": "Guwahati", "state": "Assam", "zone": "Northeast Frontier", "lat": 26.1831, "lng": 91.7504, "platforms": 5},
    {"code": "TVC", "name": "Thiruvananthapuram Central", "city": "Thiruvananthapuram", "state": "Kerala", "zone": "Southern", "lat": 8.4894, "lng": 76.9507, "platforms": 5},
    {"code": "VSKP", "name": "Visakhapatnam", "city": "Visakhapatnam", "state": "Andhra Pradesh", "zone": "East Coast", "lat": 17.7215, "lng": 83.2889, "platforms": 8},
    {"code": "NGP", "name": "Nagpur Junction", "city": "Nagpur", "state": "Maharashtra", "zone": "Central", "lat": 21.1472, "lng": 79.0845, "platforms": 8},
    {"code": "RNC", "name": "Ranchi Junction", "city": "Ranchi", "state": "Jharkhand", "zone": "South Eastern", "lat": 23.3143, "lng": 85.3214, "platforms": 6},
    {"code": "CDG", "name": "Chandigarh Junction", "city": "Chandigarh", "state": "Chandigarh", "zone": "Northern", "lat": 30.6767, "lng": 76.8092, "platforms": 6},
    {"code": "BBS", "name": "Bhubaneswar", "city": "Bhubaneswar", "state": "Odisha", "zone": "East Coast", "lat": 20.2713, "lng": 85.8388, "platforms": 6},
]

COACH_DESIGNATIONS = [
    "SL", "S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9", "S10", "S11", "S12",
    "3A", "B1", "B2", "B3", "B4",
    "2A", "A1", "A2",
    "1A", "H1",
    "GN", "GS",
    "PC", "RMS",
]

TRAIN_TYPES = ["Superfast Express", "Express", "Mail", "Rajdhani", "Shatabdi", "Duronto", "Garib Rath", "Jan Shatabdi", "Passenger"]


# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

LOG_DIR = PROJECT_ROOT / "logs"
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)-25s | %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")


# ============================================================================
# RUNTIME DEFAULTS
# ============================================================================

DEFAULT_PREPROCESS = PreprocessConfig()
DEFAULT_AUGMENTATION = AugmentationConfig()
DEFAULT_YOLO = YOLOConfig()
DEFAULT_DEEPSORT = DeepSORTConfig()
DEFAULT_ANOMALY_CLF = AnomalyClassifierConfig()
DEFAULT_XGBOOST = XGBoostConfig()
