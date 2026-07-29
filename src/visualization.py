"""Visualization helpers for object detection datasets.

This module draws bounding boxes and labels onto images and saves them to the
`outputs/visualizations` directory.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np

from src.dataset_utils import pascal_voc_to_yolo

logger = logging.getLogger(__name__)


def draw_bounding_boxes(
    image: np.ndarray,
    boxes: Sequence[Sequence[float]],
    labels: Sequence[str],
    confidences: Optional[Sequence[float]] = None,
    color: Tuple[int, int, int] = (0, 255, 0),
) -> np.ndarray:
    """Draw bounding boxes and labels onto an RGB image.

    Args:
        image: RGB image as a NumPy array.
        boxes: Bounding boxes in Pascal VOC format ``(x_min, y_min, x_max, y_max)``.
        labels: Class labels to draw.
        confidences: Optional confidence values.
        color: Box and text color.

    Returns:
        Annotated image as a NumPy array.
    """
    image_copy = image.copy()
    if image_copy.dtype != np.uint8:
        image_copy = np.asarray(image_copy, dtype=np.uint8)

    for idx, (box, label) in enumerate(zip(boxes, labels)):
        x_min, y_min, x_max, y_max = [int(float(value)) for value in box]
        cv2.rectangle(image_copy, (x_min, y_min), (x_max, y_max), color, 2)

        text = str(label)
        if confidences is not None and idx < len(confidences):
            text = f"{text} {confidences[idx]:.2f}"

        cv2.putText(image_copy, text, (x_min, max(0, y_min - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    return image_copy


def save_visualization(image: np.ndarray, output_path: Path, *, boxes: Optional[Sequence[Sequence[float]]] = None, labels: Optional[Sequence[str]] = None, confidences: Optional[Sequence[float]] = None) -> Path:
    """Save a visualization image to disk.

    Args:
        image: RGB image array.
        output_path: Destination path for the saved visualization.
        boxes: Optional bounding boxes to draw.
        labels: Optional labels to draw.
        confidences: Optional confidence values.

    Returns:
        The path to the written file.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    annotated = image
    if boxes is not None and labels is not None:
        annotated = draw_bounding_boxes(image, boxes, labels, confidences=confidences)

    cv2.imwrite(str(output_path), cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))
    logger.info("Saving visualization: %s", output_path)
    return output_path


def show_image(image: np.ndarray) -> None:
    """Display an RGB image using Matplotlib.

    Args:
        image: RGB image array.
    """
    plt.figure(figsize=(8, 8))
    plt.imshow(image)
    plt.axis("off")
    plt.show()
