"""Evaluation pipeline for the trained YOLO object detector.

This module loads the configured model weights, runs validation on the
configured dataset, prints the main metrics, and saves prediction images
for a small set of validation samples.
"""

from __future__ import annotations

import logging
import random
import sys
import time
from pathlib import Path
from typing import Optional

import cv2
import torch
from ultralytics import YOLO

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import AppConfig, ConfigError, load_config
from src.metrics import print_metrics, save_metrics_json, save_metrics_txt
from src.visualization import draw_bounding_boxes, save_visualization

logger = logging.getLogger(__name__)


def _resolve_repo_root() -> Path:
    """Return the repository root directory."""
    return Path(__file__).resolve().parent.parent


def _resolve_path(path_value: str | Path, *, repo_root: Path) -> Path:
    """Resolve a config path relative to the repository root when needed."""
    path = Path(path_value)
    if path.is_absolute():
        return path
    return (repo_root / path).resolve()


def load_model(config: Optional[AppConfig] = None) -> YOLO:
    """Load the trained YOLO model from the configured artifact path."""
    if config is None:
        config = load_config()

    repo_root = _resolve_repo_root()
    model_path = repo_root / config.model.save_directory / "train" / "weights" / "best.pt"
    if not model_path.exists():
        raise FileNotFoundError(f"Trained model weights not found: {model_path}")

    try:
        model = YOLO(str(model_path))
    except Exception as exc:  # pragma: no cover - depends on runtime environment
        raise RuntimeError(f"Unable to load model weights '{model_path}': {exc}") from exc

    logger.info("Loaded model weights from %s", model_path)
    return model


def run_evaluation(
    config: Optional[AppConfig] = None,
    model: Optional[YOLO] = None,
) -> tuple[dict[str, float], Path, Path, Path]:
    """Run YOLO validation and save evaluation artifacts.

    Returns:
        A tuple containing the metrics dictionary, metrics JSON path, metrics
        text path, and the predictions directory.
    """
    if config is None:
        config = load_config()
    if model is None:
        model = load_model(config)

    repo_root = _resolve_repo_root()
    dataset_yaml = _resolve_path(config.dataset.yaml_file, repo_root=repo_root)
    if not dataset_yaml.exists():
        raise FileNotFoundError(f"Dataset configuration was not found: {dataset_yaml}")

    output_root = (repo_root / "outputs").resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    predictions_dir = output_root / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.perf_counter()
    results = model.val(
        data=str(dataset_yaml),
        imgsz=config.training.image_size,
        conf=config.inference.confidence_threshold,
        iou=config.inference.iou_threshold,
        device="cuda" if torch.cuda.is_available() else "cpu",
        project=str(output_root),
        name="evaluation",
        exist_ok=True,
    )
    evaluation_time = time.perf_counter() - start_time

    summaries = getattr(results, "results_dict", {}) or {}
    metrics = {
        "precision": float(summaries.get("metrics/precision(B)", summaries.get("precision", 0.0))),
        "recall": float(summaries.get("metrics/recall(B)", summaries.get("recall", 0.0))),
        "map50": float(summaries.get("metrics/mAP50(B)", summaries.get("mAP50", 0.0))),
        "map50_95": float(summaries.get("metrics/mAP50-95(B)", summaries.get("mAP50_95", 0.0))),
        "num_images": len(list((repo_root / config.dataset.valid).glob("images/*"))) if (repo_root / config.dataset.valid).exists() else 0,
        "evaluation_time_seconds": float(evaluation_time),
    }

    metrics_json_path = save_metrics_json(output_root / "metrics.json", metrics)
    metrics_txt_path = save_metrics_txt(output_root / "metrics.txt", metrics)

    valid_images_dir = repo_root / config.dataset.valid / "images"
    image_paths = sorted(valid_images_dir.glob("*"))
    image_paths = [path for path in image_paths if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}]

    if len(image_paths) >= 5:
        sampled_paths = random.sample(image_paths, 5)
    else:
        sampled_paths = image_paths

    for image_path in sampled_paths:
        try:
            predicted = model(image_path, conf=config.inference.confidence_threshold, imgsz=config.training.image_size)
            result = predicted[0]
            image = cv2.imread(str(image_path))
            if image is None:
                raise FileNotFoundError(f"Unable to read image: {image_path}")
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            boxes = result.boxes.xyxy.tolist() if hasattr(result.boxes, "xyxy") else []
            labels = [result.names[int(class_id)] for class_id in result.boxes.cls.tolist()] if hasattr(result.boxes, "cls") else []
            confidences = result.boxes.conf.tolist() if hasattr(result.boxes, "conf") else []
            annotated = draw_bounding_boxes(rgb_image, boxes, labels, confidences=confidences)
            output_image_path = predictions_dir / f"{image_path.stem}_predicted.jpg"
            save_visualization(annotated, output_image_path)
        except Exception as exc:  # pragma: no cover - depends on runtime environment
            logger.warning("Unable to generate prediction for %s: %s", image_path, exc)

    return metrics, metrics_json_path, metrics_txt_path, predictions_dir


def main() -> None:
    """Run the full evaluation workflow."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    logger.info("Loading configuration")
    try:
        config = load_config()
    except ConfigError as exc:
        raise RuntimeError(f"Configuration loading failed: {exc}") from exc

    try:
        model = load_model(config)
    except (FileNotFoundError, RuntimeError) as exc:
        raise RuntimeError(f"Model loading failed: {exc}") from exc

    try:
        metrics, metrics_json_path, metrics_txt_path, predictions_dir = run_evaluation(config=config, model=model)
    except RuntimeError as exc:
        raise RuntimeError(f"Evaluation failed: {exc}") from exc

    print_metrics(metrics)
    print(f"Metrics JSON: {metrics_json_path}")
    print(f"Metrics text report: {metrics_txt_path}")
    print(f"Predictions directory: {predictions_dir}")


if __name__ == "__main__":
    main()
