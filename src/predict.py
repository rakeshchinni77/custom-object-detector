"""Prediction utilities for the custom object detector.

This module provides a small, reusable inference engine that other
components (including an API) can import. It loads configuration using
``load_config()``, loads the trained weights from the model output
directory, runs YOLOv8 inference, formats predictions, and saves an
annotated image using the project's visualization helpers.

Only Phase 7 functionality belongs here — no training or evaluation logic
is implemented.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import torch
from ultralytics import YOLO

# Ensure repository root is importable when running as a script.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import AppConfig, ConfigError, load_config
from src.visualization import draw_bounding_boxes, save_visualization

logger = logging.getLogger(__name__)


def _resolve_device(config: AppConfig) -> str:
    """Resolve the runtime device from the configuration.

    Args:
        config: Application configuration.

    Returns:
        A device string usable by Ultralytics/PyTorch ("cpu" or "cuda").
    """
    configured = config.system.device.lower()
    if configured == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if configured == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested in config but is not available")
    return configured


def load_model(config: Optional[AppConfig] = None) -> YOLO:
    """Load the trained YOLO model (best.pt) using Ultralytics.

    Args:
        config: Optional AppConfig. When omitted, configuration is loaded.

    Returns:
        A loaded ``YOLO`` model instance.

    Raises:
        ConfigError: If configuration cannot be loaded.
        FileNotFoundError: If the expected weights file does not exist.
        RuntimeError: If Ultralytics fails to load the model.
    """
    if config is None:
        config = load_config()

    model_path = (REPO_ROOT / config.model.save_directory / "train" / "weights" / "best.pt").resolve()
    if not model_path.exists():
        raise FileNotFoundError(f"Trained weights not found: {model_path}")

    logger.info("Loading model from %s", model_path)
    try:
        model = YOLO(str(model_path))
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(f"Unable to load model weights '{model_path}': {exc}") from exc

    logger.info("Model loaded")
    return model


def preprocess_image(image_path: str | Path) -> Tuple[Path, Any]:
    """Validate and load an image from disk and convert to RGB.

    Args:
        image_path: Local path to an image file.

    Returns:
        A tuple with the resolved Path and an RGB NumPy image array.

    Raises:
        FileNotFoundError: If the path does not exist.
        ValueError: If the image cannot be read or has unsupported format.
    """
    path = Path(image_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    if path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}:
        raise ValueError(f"Unsupported image format: {path.suffix}")

    image_bgr = cv2.imread(str(path))
    if image_bgr is None:
        raise ValueError(f"Unable to read image (corrupt or unreadable): {path}")

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    logger.info("Loaded image %s (shape=%s)", path.name, image_rgb.shape)
    return path, image_rgb


def predict_image(
    image_path: str | Path,
    config: Optional[AppConfig] = None,
    model: Optional[YOLO] = None,
) -> List[Dict[str, Any]]:
    """Run inference on a single image and return formatted predictions.

    This function is intended to be imported by the FastAPI server later.

    Args:
        image_path: Path to the image file.
        config: Optional application config. Loaded if omitted.
        model: Optional pre-loaded YOLO model. If omitted, `load_model` is used.

    Returns:
        A list of prediction dictionaries (see ``format_predictions``).

    Raises:
        RuntimeError: On inference or configuration errors.
    """
    if config is None:
        try:
            config = load_config()
        except ConfigError as exc:
            raise RuntimeError(f"Configuration load failed: {exc}") from exc

    if model is None:
        model = load_model(config)

    path, _ = preprocess_image(image_path)

    device = _resolve_device(config)
    logger.info("Running inference on %s using device=%s", path.name, device)

    try:
        results = model(
            str(path),
            conf=config.inference.confidence_threshold,
            iou=config.inference.iou_threshold,
            imgsz=config.training.image_size,
            device=device,
        )
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(f"Inference failed for {path}: {exc}") from exc

    if not results:
        return []

    return format_predictions(results[0], model=model)


def format_predictions(result: Any, model: Optional[YOLO] = None) -> List[Dict[str, Any]]:
    """Convert Ultralytics prediction result into a list of dicts.

    Each dict contains: class_id, class_name, confidence, bbox (x1,y1,x2,y2).
    """
    if not hasattr(result, "boxes") or result.boxes is None:
        return []

    xyxy = getattr(result.boxes, "xyxy", None)
    confidences = getattr(result.boxes, "conf", None)
    class_ids = getattr(result.boxes, "cls", None)

    if xyxy is None or confidences is None or class_ids is None:
        return []

    names = getattr(result, "names", None)
    if names is None and model is not None:
        names = getattr(model, "names", None)

    preds: List[Dict[str, Any]] = []
    for box, conf, cls in zip(xyxy.tolist(), confidences.tolist(), class_ids.tolist()):
        cls_idx = int(cls)
        if names is None:
            cls_name = str(cls_idx)
        elif hasattr(names, "get"):
            cls_name = names.get(cls_idx, str(cls_idx))
        else:
            cls_name = names[cls_idx] if cls_idx < len(names) else str(cls_idx)

        preds.append(
            {
                "class_id": cls_idx,
                "class_name": cls_name,
                "confidence": float(conf),
                "bbox": [float(box[0]), float(box[1]), float(box[2]), float(box[3])],
            }
        )

    return preds


def save_prediction_image(
    image_rgb: Any,
    result: Any,
    output_path: Optional[str | Path] = None,
    config: Optional[AppConfig] = None,
) -> Path:
    """Annotate an RGB image with predicted boxes and save to outputs/predictions.

    Args:
        image_rgb: RGB image array (NumPy).
        result: YOLO result object for the same image.
        output_path: Optional destination path; default placed under outputs/predictions.
        config: Optional AppConfig to determine repository layout.

    Returns:
        The saved Path.
    """
    if config is None:
        config = load_config()

    out_dir = (REPO_ROOT / "outputs" / "predictions").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if output_path is None:
        output_path = out_dir / "prediction_sample.jpg"
    else:
        output_path = Path(output_path)
        if not output_path.is_absolute():
            output_path = (REPO_ROOT / output_path).resolve()

    boxes = result.boxes.xyxy.tolist() if hasattr(result.boxes, "xyxy") else []
    confidences = result.boxes.conf.tolist() if hasattr(result.boxes, "conf") else []
    class_ids = result.boxes.cls.tolist() if hasattr(result.boxes, "cls") else []

    names = getattr(result, "names", {})
    labels: List[str] = []
    for cls in class_ids:
        cls_idx = int(cls)
        if hasattr(names, "get"):
            labels.append(names.get(cls_idx, str(cls_idx)))
        else:
            labels.append(names[cls_idx] if cls_idx < len(names) else str(cls_idx))

    annotated = draw_bounding_boxes(image_rgb, boxes, labels, confidences=confidences)
    saved = save_visualization(annotated, output_path)
    logger.info("Saved prediction visualization: %s", saved)
    return saved


def main() -> None:
    """CLI entry point: run prediction on a provided image path.

    Usage: python src/predict.py /path/to/image.jpg
    If no argument is provided the user will be prompted.
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    try:
        config = load_config()
    except ConfigError as exc:
        raise RuntimeError(f"Configuration loading failed: {exc}") from exc

    image_arg = sys.argv[1] if len(sys.argv) > 1 else input("Image path: ").strip()
    if not image_arg:
        raise RuntimeError("An image path must be provided")

    model = load_model(config)
    path, image_rgb = preprocess_image(image_arg)

    # Run inference and save both structured results and visualization
    preds = predict_image(path, config=config, model=model)

    # Run model again to access the raw result object for visualization
    # (Ultralytics returns results when called with a path)
    raw_results = model(str(path), conf=config.inference.confidence_threshold, iou=config.inference.iou_threshold, imgsz=config.training.image_size, device=_resolve_device(config))
    if not raw_results:
        logger.info("No detections returned for %s", path)
    else:
        saved_path = save_prediction_image(image_rgb, raw_results[0], config=config)
        logger.info("Prediction completed. Annotated image: %s", saved_path)

    print("Predictions:")
    for p in preds:
        print(p)


if __name__ == "__main__":
    main()
