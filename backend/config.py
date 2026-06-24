"""
Digital Shield Rail Defense — Backend Configuration
=====================================================
Centralized configuration management for all backend services.
Supports environment variables, defaults, and runtime overrides.
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class AppConfig:
    """Application-level configuration."""
    app_name: str = "Digital Shield Rail Defense"
    app_version: str = "1.0.0"
    debug: bool = False
    environment: str = "development"  # development, staging, production


@dataclass
class ServerConfig:
    """FastAPI server configuration."""
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 4
    reload: bool = True
    cors_origins: list = field(default_factory=lambda: ["http://localhost:3000", "http://localhost:5173"])


@dataclass
class DatabaseConfig:
    """PostgreSQL database configuration."""
    host: str = os.getenv("DB_HOST", "localhost")
    port: int = int(os.getenv("DB_PORT", "5432"))
    name: str = os.getenv("DB_NAME", "digital_shield")
    user: str = os.getenv("DB_USER", "postgres")
    password: str = os.getenv("DB_PASSWORD", "")
    pool_min: int = 2
    pool_max: int = 10

    @property
    def url(self) -> str:
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.name}"


@dataclass
class CameraConfig:
    """Camera intelligence configuration."""
    registry_file: str = "dataset/metadata/camera_registry.json"
    default_resolution: str = "1080p"
    schedule_tolerance_minutes: int = 15
    cameras_per_platform_zones: int = 3
    train_count: int = 300
    auto_generate_on_startup: bool = True


@dataclass
class PathConfig:
    """File path configuration."""
    project_root: Path = Path(__file__).resolve().parent.parent
    dataset_root: Path = field(default=None)
    metadata_dir: Path = field(default=None)
    annotations_dir: Path = field(default=None)
    output_dir: Path = field(default=None)
    logs_dir: Path = field(default=None)

    def __post_init__(self):
        self.dataset_root = self.dataset_root or self.project_root / "dataset"
        self.metadata_dir = self.metadata_dir or self.dataset_root / "metadata"
        self.annotations_dir = self.annotations_dir or self.dataset_root / "annotations"
        self.output_dir = self.output_dir or self.project_root / "output"
        self.logs_dir = self.logs_dir or self.project_root / "logs"

        # Ensure directories exist
        for d in [self.metadata_dir, self.annotations_dir, self.output_dir, self.logs_dir]:
            d.mkdir(parents=True, exist_ok=True)


@dataclass
class MLConfig:
    """ML pipeline configuration."""
    yolo_model: str = "yolov8n.pt"
    confidence_threshold: float = 0.5
    iou_threshold: float = 0.45
    deepsort_max_age: int = 30
    anomaly_risk_threshold: float = 0.5
    device: str = "auto"


class Settings:
    """
    Singleton settings manager.

    Usage:
        settings = Settings()
        settings.camera.registry_file
        settings.server.port
        settings.paths.metadata_dir
    """

    _instance: Optional["Settings"] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._init_settings()
        return cls._instance

    def _init_settings(self):
        self.app = AppConfig(
            debug=os.getenv("DEBUG", "false").lower() == "true",
            environment=os.getenv("ENVIRONMENT", "development"),
        )
        self.server = ServerConfig(
            host=os.getenv("HOST", "0.0.0.0"),
            port=int(os.getenv("PORT", "8000")),
        )
        self.database = DatabaseConfig()
        self.camera = CameraConfig()
        self.paths = PathConfig()
        self.ml = MLConfig()

    def to_dict(self) -> dict:
        from dataclasses import asdict
        return {
            "app": asdict(self.app),
            "server": asdict(self.server),
            "camera": asdict(self.camera),
            "ml": asdict(self.ml),
        }
