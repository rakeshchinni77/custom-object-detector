"""Centralized configuration loader for the custom object detection project.

This module reads YAML configuration values, validates them, and exposes a
strongly typed configuration object for downstream phases.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

try:
    import yaml
except ImportError as exc:  # pragma: no cover - handled at runtime when PyYAML is missing
    yaml = None

logger = logging.getLogger(__name__)


class ConfigError(Exception):
    """Raised when configuration loading or validation fails."""


@dataclass(frozen=True)
class ProjectConfig:
    """Project metadata configuration."""

    name: str
    version: str


@dataclass(frozen=True)
class DatasetConfig:
    """Dataset paths and related configuration."""

    root: str
    train: str
    valid: str
    test: str
    yaml_file: str


@dataclass(frozen=True)
class TrainingConfig:
    """Training hyperparameters."""

    batch_size: int
    epochs: int
    image_size: int
    optimizer: str
    learning_rate: float
    weight_decay: float
    workers: int


@dataclass(frozen=True)
class ModelConfig:
    """Model runtime configuration."""

    pretrained_model: str
    save_directory: str


@dataclass(frozen=True)
class InferenceConfig:
    """Inference thresholds."""

    confidence_threshold: float
    iou_threshold: float


@dataclass(frozen=True)
class SystemConfig:
    """System-level runtime settings."""

    device: str
    random_seed: int


@dataclass(frozen=True)
class LoggingConfig:
    """Logging configuration."""

    level: str
    save_logs: bool


@dataclass(frozen=True)
class AppConfig:
    """Master application configuration."""

    project: ProjectConfig
    dataset: DatasetConfig
    training: TrainingConfig
    model: ModelConfig
    inference: InferenceConfig
    system: SystemConfig
    logging: LoggingConfig

    def get_device(self) -> str:
        """Return the configured training/inference device."""
        return self.system.device

    def get_dataset_yaml(self) -> Path:
        """Return the dataset YAML configuration path as a Path."""
        return Path(self.dataset.yaml_file)

    def get_model_directory(self) -> Path:
        """Return the model save directory as a Path."""
        return Path(self.model.save_directory)

    def get_output_directory(self) -> Path:
        """Return the default output directory for artifacts."""
        return Path(self.dataset.root) / ".." / "outputs"


def _require_mapping(value: Any, context: str) -> Dict[str, Any]:
    """Ensure a configuration value is a mapping and return it."""
    if not isinstance(value, dict):
        raise ConfigError(f"{context} must be a mapping")
    return value


def _require_string(value: Any, context: str) -> str:
    """Ensure a configuration value is a string."""
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{context} must be a non-empty string")
    return value


def _require_int(value: Any, context: str) -> int:
    """Ensure a configuration value is an integer."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"{context} must be an integer")
    return value


def _require_float(value: Any, context: str) -> float:
    """Ensure a configuration value is a number."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{context} must be a number")
    return float(value)


def _require_bool(value: Any, context: str) -> bool:
    """Ensure a configuration value is a boolean."""
    if not isinstance(value, bool):
        raise ConfigError(f"{context} must be a boolean")
    return value


def _load_yaml_config(config_path: Path) -> Dict[str, Any]:
    """Load and validate a YAML configuration file."""
    if yaml is None:
        raise ConfigError("PyYAML is required to load configuration files")

    if not config_path.exists():
        raise ConfigError(f"Configuration file not found: {config_path}")

    try:
        with config_path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
    except yaml.YAMLError as exc:  # type: ignore[attr-defined]
        raise ConfigError(f"Invalid YAML syntax in {config_path}: {exc}") from exc

    if not isinstance(data, dict):
        raise ConfigError("Configuration root must be a mapping")

    return data


def load_config(config_path: Optional[str | Path] = None) -> AppConfig:
    """Load and validate the application configuration.

    Args:
        config_path: Optional path to a YAML file. Defaults to the repository
            configuration file at config/config.yaml.

    Returns:
        A populated AppConfig instance.

    Raises:
        ConfigError: If the config file is missing, malformed, or contains
            invalid values.
    """
    if config_path is None:
        config_path = Path(__file__).resolve().parent.parent / "config" / "config.yaml"
    else:
        config_path = Path(config_path)

    logger.info("Loading configuration from %s", config_path)
    raw_config = _load_yaml_config(config_path)

    project_cfg = _require_mapping(raw_config.get("project"), "project")
    dataset_cfg = _require_mapping(raw_config.get("dataset"), "dataset")
    training_cfg = _require_mapping(raw_config.get("training"), "training")
    model_cfg = _require_mapping(raw_config.get("model"), "model")
    inference_cfg = _require_mapping(raw_config.get("inference"), "inference")
    system_cfg = _require_mapping(raw_config.get("system"), "system")
    logging_cfg = _require_mapping(raw_config.get("logging"), "logging")

    app_config = AppConfig(
        project=ProjectConfig(
            name=_require_string(project_cfg.get("name"), "project.name"),
            version=_require_string(project_cfg.get("version"), "project.version"),
        ),
        dataset=DatasetConfig(
            root=_require_string(dataset_cfg.get("root"), "dataset.root"),
            train=_require_string(dataset_cfg.get("train"), "dataset.train"),
            valid=_require_string(dataset_cfg.get("valid"), "dataset.valid"),
            test=_require_string(dataset_cfg.get("test"), "dataset.test"),
            yaml_file=_require_string(dataset_cfg.get("yaml_file"), "dataset.yaml_file"),
        ),
        training=TrainingConfig(
            batch_size=_require_int(training_cfg.get("batch_size"), "training.batch_size"),
            epochs=_require_int(training_cfg.get("epochs"), "training.epochs"),
            image_size=_require_int(training_cfg.get("image_size"), "training.image_size"),
            optimizer=_require_string(training_cfg.get("optimizer"), "training.optimizer"),
            learning_rate=_require_float(training_cfg.get("learning_rate"), "training.learning_rate"),
            weight_decay=_require_float(training_cfg.get("weight_decay"), "training.weight_decay"),
            workers=_require_int(training_cfg.get("workers"), "training.workers"),
        ),
        model=ModelConfig(
            pretrained_model=_require_string(model_cfg.get("pretrained_model"), "model.pretrained_model"),
            save_directory=_require_string(model_cfg.get("save_directory"), "model.save_directory"),
        ),
        inference=InferenceConfig(
            confidence_threshold=_require_float(inference_cfg.get("confidence_threshold"), "inference.confidence_threshold"),
            iou_threshold=_require_float(inference_cfg.get("iou_threshold"), "inference.iou_threshold"),
        ),
        system=SystemConfig(
            device=_require_string(system_cfg.get("device"), "system.device"),
            random_seed=_require_int(system_cfg.get("random_seed"), "system.random_seed"),
        ),
        logging=LoggingConfig(
            level=_require_string(logging_cfg.get("level"), "logging.level"),
            save_logs=_require_bool(logging_cfg.get("save_logs"), "logging.save_logs"),
        ),
    )

    logger.info("Configuration loaded successfully.")
    return app_config


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    config = load_config()
    print("Loaded configuration:")
    print(f"Project: {config.project.name} v{config.project.version}")
    print(f"Dataset root: {config.dataset.root}")
    print(f"Training batch size: {config.training.batch_size}")
    print(f"Model save directory: {config.model.save_directory}")
    print(f"Device: {config.get_device()}")
    print(f"Dataset YAML: {config.get_dataset_yaml()}")
    print(f"Model directory: {config.get_model_directory()}")
    print(f"Output directory: {config.get_output_directory()}")
