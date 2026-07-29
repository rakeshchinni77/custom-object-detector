"""Reusable helpers for YOLO-style object detection datasets.

This module centralizes image loading, label parsing, coordinate conversion,
and file discovery so the dataset pipeline remains modular and reusable.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def read_image(image_path: Path, *, convert_bgr_to_rgb: bool = True) -> np.ndarray:
    """Read an image from disk and return it as a NumPy array.

    Args:
        image_path: Path to the image file.
        convert_bgr_to_rgb: Whether to convert from OpenCV BGR ordering to RGB.

    Returns:
        The decoded image array.

    Raises:
        FileNotFoundError: If the image file does not exist.
        ValueError: If the image cannot be decoded.
    """
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    logger.debug("Loading image: %s", image_path)
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Unable to decode image: {image_path}")

    if convert_bgr_to_rgb:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def read_yolo_label(label_path: Path) -> List[Tuple[int, float, float, float, float]]:
    """Parse a YOLO-style annotation file into a list of label tuples.

    Args:
        label_path: Path to the label file.

    Returns:
        A list of tuples of the form ``(class_id, x_center, y_center, width, height)``.

    Raises:
        ValueError: If the content is malformed or contains invalid values.
    """
    if not label_path.exists():
        logger.warning("Label file missing; returning empty annotations: %s", label_path)
        return []

    logger.debug("Reading labels: %s", label_path)
    annotations: List[Tuple[int, float, float, float, float]] = []
    with label_path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 5:
                raise ValueError(f"Invalid annotation format in {label_path} at line {line_number}")

            try:
                class_id = int(parts[0])
                x_center = float(parts[1])
                y_center = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])
            except ValueError as exc:
                raise ValueError(f"Invalid numeric values in {label_path} at line {line_number}") from exc

            validate_yolo_bbox((class_id, x_center, y_center, width, height))
            annotations.append((class_id, x_center, y_center, width, height))

    return annotations


def validate_yolo_bbox(annotation: Tuple[int, float, float, float, float]) -> None:
    """Validate a YOLO annotation tuple.

    Args:
        annotation: Tuple of ``(class_id, x_center, y_center, width, height)``.

    Raises:
        ValueError: If the annotation values are invalid.
    """
    if len(annotation) != 5:
        raise ValueError("YOLO annotation must contain 5 values")

    class_id, x_center, y_center, width, height = annotation
    if not isinstance(class_id, int):
        raise ValueError("class_id must be an integer")
    if not (0.0 <= x_center <= 1.0 and 0.0 <= y_center <= 1.0):
        raise ValueError("x_center and y_center must lie in [0, 1]")
    if not (0.0 < width <= 1.0 and 0.0 < height <= 1.0):
        raise ValueError("width and height must lie in (0, 1]")


def yolo_to_pascal_voc(annotation: Tuple[int, float, float, float, float], image_shape: Tuple[int, int, int]) -> Tuple[int, float, float, float, float]:
    """Convert a YOLO annotation to Pascal VOC coordinates.

    Args:
        annotation: YOLO annotation tuple.
        image_shape: Image shape as ``(height, width, channels)``.

    Returns:
        Tuple of ``(class_id, x_min, y_min, x_max, y_max)``.
    """
    class_id, x_center, y_center, width, height = annotation
    height_px, width_px, _ = image_shape

    x_min = (x_center - width / 2.0) * width_px
    y_min = (y_center - height / 2.0) * height_px
    x_max = (x_center + width / 2.0) * width_px
    y_max = (y_center + height / 2.0) * height_px
    return class_id, float(x_min), float(y_min), float(x_max), float(y_max)


def pascal_voc_to_yolo(annotation: Tuple[int, float, float, float, float], image_shape: Tuple[int, int, int]) -> Tuple[int, float, float, float, float]:
    """Convert Pascal VOC coordinates to YOLO annotation values.

    Args:
        annotation: Pascal VOC annotation tuple.
        image_shape: Image shape as ``(height, width, channels)``.

    Returns:
        Tuple of ``(class_id, x_center, y_center, width, height)``.
    """
    class_id, x_min, y_min, x_max, y_max = annotation
    height_px, width_px, _ = image_shape

    width = (x_max - x_min) / width_px
    height = (y_max - y_min) / height_px
    x_center = (x_min + x_max) / (2.0 * width_px)
    y_center = (y_min + y_max) / (2.0 * height_px)
    return class_id, float(x_center), float(y_center), float(width), float(height)


def load_class_names(data_yaml_path: Path) -> List[str]:
    """Load class names from a YOLO-style `data.yaml` file.

    Args:
        data_yaml_path: Path to the YAML file.

    Returns:
        A list of class names.

    Raises:
        FileNotFoundError: If the YAML file does not exist.
        ValueError: If class names cannot be parsed.
    """
    if not data_yaml_path.exists():
        raise FileNotFoundError(f"YAML config not found: {data_yaml_path}")

    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - dependency may be missing in some environments
        raise ImportError("PyYAML is required to read data.yaml") from exc

    with data_yaml_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)

    names = data.get("names") if isinstance(data, dict) else None
    if not isinstance(names, list) or not all(isinstance(name, str) for name in names):
        raise ValueError("Invalid class names in YAML config")
    return names


def list_image_files(image_dir: Path) -> List[Path]:
    """List supported image files in an image directory.

    Args:
        image_dir: Directory to scan.

    Returns:
        Sorted list of image paths.
    """
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    return sorted(
        path for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
    )
