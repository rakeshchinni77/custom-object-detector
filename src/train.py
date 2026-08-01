"""Training pipeline for the custom object detection system.

This module loads configuration from the YAML file, resolves the pretrained
YOLOv8 weights, and starts model training using the configured dataset and
hyperparameters.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

import torch
from ultralytics import YOLO

from src.config import AppConfig, ConfigError, load_config

logger = logging.getLogger(__name__)

REQUIRED_OUTPUT_FILES = (
    "best.pt",
    "last.pt",
    "results.csv",
    "results.png",
    "confusion_matrix.png",
)

CURVE_OUTPUT_FILES = (
    ("PR_curve.png", "BoxPR_curve.png"),
    ("F1_curve.png", "BoxF1_curve.png"),
    ("P_curve.png", "BoxP_curve.png"),
    ("R_curve.png", "BoxR_curve.png"),
)


def _resolve_repo_root() -> Path:
    """Return the repository root directory."""
    return Path(__file__).resolve().parent.parent


def _resolve_path(path_value: str | Path, *, repo_root: Path) -> Path:
    """Resolve a config path relative to the repository root when needed."""
    path = Path(path_value)
    if path.is_absolute():
        return path
    return (repo_root / path).resolve()


def _resolve_device(config: AppConfig) -> str:
    """Resolve the runtime device from the configuration."""
    configured_device = config.system.device.lower()
    if configured_device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if configured_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested in the configuration but is not available")
    return configured_device


def load_model(config: Optional[AppConfig] = None) -> YOLO:
    """Load the pretrained YOLO model from the configured path.

    Args:
        config: Optional application configuration instance. When omitted, the
            configuration is loaded from the default YAML file.

    Returns:
        A YOLO model instance.

    Raises:
        ConfigError: If the configuration is missing or invalid.
        FileNotFoundError: If the requested model path cannot be resolved.
        RuntimeError: If the model cannot be loaded.
    """
    if config is None:
        config = load_config()

    logger.info("Loading model")
    repo_root = _resolve_repo_root()
    model_path = _resolve_path(config.model.pretrained_model, repo_root=repo_root)

    if not model_path.exists():
        logger.warning(
            "Pretrained weights not found locally at %s; attempting to resolve them via Ultralytics.",
            model_path,
        )

    try:
        model = YOLO(str(config.model.pretrained_model))
    except Exception as exc:  # pragma: no cover - depends on installed weights/network
        raise RuntimeError(
            f"Unable to load pretrained model '{config.model.pretrained_model}': {exc}"
        ) from exc

    logger.info("Model loaded successfully from %s", config.model.pretrained_model)
    return model


def train_model(
    config: Optional[AppConfig] = None,
    model: Optional[YOLO] = None,
    epochs: Optional[int] = None,
) -> tuple[Path, float]:
    """Train the YOLO model using values from the configuration.

    Args:
        config: Optional application configuration instance.
        model: Optional pretrained YOLO model instance.
        epochs: Optional override for the number of epochs. When omitted, the
            value from the configuration is used.

    Returns:
        A tuple containing the training output directory and elapsed seconds.

    Raises:
        ConfigError: If the configuration is missing or invalid.
        FileNotFoundError: If the dataset YAML file is missing.
        RuntimeError: If training cannot be completed.
    """
    if config is None:
        config = load_config()
    if model is None:
        model = load_model(config)

    repo_root = _resolve_repo_root()
    dataset_yaml = _resolve_path(config.dataset.yaml_file, repo_root=repo_root)
    if not dataset_yaml.exists():
        raise FileNotFoundError(f"Dataset configuration not found: {dataset_yaml}")

    output_root = (repo_root / config.model.save_directory).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    train_name = "train"
    train_dir = output_root / train_name
    train_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(config.system.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.system.random_seed)

    logger.info("Training started")
    start_time = time.perf_counter()

    try:
        model.train(
            data=str(dataset_yaml),
            epochs=epochs if epochs is not None else config.training.epochs,
            batch=config.training.batch_size,
            imgsz=config.training.image_size,
            device=_resolve_device(config),
            optimizer=config.training.optimizer,
            lr0=config.training.learning_rate,
            workers=config.training.workers,
            project=str(output_root),
            name=train_name,
            exist_ok=True,
            seed=config.system.random_seed,
            verbose=True,
        )
    except KeyboardInterrupt as exc:
        raise RuntimeError("Training interrupted by the user") from exc
    except Exception as exc:  # pragma: no cover - depends on runtime environment
        raise RuntimeError(f"Training failed: {exc}") from exc

    duration = time.perf_counter() - start_time
    logger.info("Training finished")
    logger.info("Training duration: %.2f seconds", duration)

    weights_dir = train_dir / "weights"
    if not weights_dir.exists():
        raise RuntimeError(f"Training output directory was not created: {weights_dir}")

    missing_files = [
        file_name for file_name in REQUIRED_OUTPUT_FILES if not (train_dir / file_name).exists() and not (weights_dir / file_name).exists()
    ]
    if missing_files:
        raise RuntimeError(f"Training completed but expected outputs were not generated: {missing_files}")

    for curve_name, alt_name in CURVE_OUTPUT_FILES:
        if (train_dir / curve_name).exists() or (train_dir / alt_name).exists():
            continue
        missing_files.append(curve_name)

    if missing_files:
        raise RuntimeError(f"Training completed but expected outputs were not generated: {missing_files}")

    return train_dir, duration


def print_training_summary(config: AppConfig, train_dir: Path, duration_seconds: float) -> None:
    """Print a concise training summary to stdout."""
    weights_dir = train_dir / "weights"
    best_model_path = weights_dir / "best.pt"
    last_model_path = weights_dir / "last.pt"

    print("\nTraining summary")
    print("=" * 40)
    print(f"Project: {config.project.name}")
    print(f"Dataset: {config.dataset.yaml_file}")
    print(f"Epochs: {config.training.epochs}")
    print(f"Batch size: {config.training.batch_size}")
    print(f"Image size: {config.training.image_size}")
    print(f"Device: {_resolve_device(config)}")
    print(f"Training duration: {duration_seconds:.2f}s")
    print(f"Model location: {best_model_path}")
    print(f"Last weights: {last_model_path}")
    print(f"Results directory: {train_dir}")


def main() -> None:
    """Run the complete training workflow."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    logger.info("Loading configuration")
    try:
        config = load_config()
    except ConfigError as exc:
        raise RuntimeError(f"Configuration loading failed: {exc}") from exc

    repo_root = _resolve_repo_root()
    pretrained_model_path = (repo_root / "models" / "train" / "weights" / "best.pt").resolve()
    if pretrained_model_path.exists():
        print("Pretrained model already exists. Skipping training.")
        train_dir = (repo_root / config.model.save_directory / "train").resolve()
        print_training_summary(config, train_dir, duration_seconds=0.0)
        return

    try:
        model = load_model(config)
    except (FileNotFoundError, RuntimeError) as exc:
        raise RuntimeError(f"Model loading failed: {exc}") from exc

    try:
        training_dir, duration_seconds = train_model(config=config, model=model)
    except RuntimeError as exc:
        raise RuntimeError(f"Training failed: {exc}") from exc

    print_training_summary(config, training_dir, duration_seconds=duration_seconds)


if __name__ == "__main__":
    main()
