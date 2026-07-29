"""PyTorch dataset wrapper for YOLO-style object detection data.

This module exposes a reusable dataset class that loads images, parses labels,
and optionally applies transforms for train/validation/test splits.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from src.dataset_utils import list_image_files, read_image, read_yolo_label, yolo_to_pascal_voc

logger = logging.getLogger(__name__)


class YOLODataset(Dataset):
    """A reusable dataset for YOLO-style object detection annotations."""

    def __init__(
        self,
        image_dir: str | Path,
        label_dir: str | Path,
        transform: Optional[object] = None,
        class_names: Optional[List[str]] = None,
    ) -> None:
        """Initialize the dataset.

        Args:
            image_dir: Directory containing images.
            label_dir: Directory containing label files.
            transform: Optional callable/transform pipeline.
            class_names: Optional list of class names used for metadata.
        """
        self.image_dir = Path(image_dir)
        self.label_dir = Path(label_dir)
        self.transform = transform
        self.class_names = class_names or []
        self.image_files = self._find_image_files()
        self.label_files = self._find_label_files()

    def _find_image_files(self) -> List[Path]:
        """Find supported image files in the image directory."""
        try:
            return list_image_files(self.image_dir)
        except FileNotFoundError as exc:
            logger.warning("Image directory missing: %s", exc)
            return []

    def _find_label_files(self) -> List[Path]:
        """Find label files matching the image directory."""
        if not self.label_dir.exists():
            logger.warning("Label directory missing: %s", self.label_dir)
            return []
        return sorted(path for path in self.label_dir.iterdir() if path.is_file() and path.suffix.lower() == ".txt")

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.image_files)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Load a single sample and return an image tensor and target dictionary.

        Args:
            index: Sample index.

        Returns:
            A tuple ``(image_tensor, target_dict)``.
        """
        if index >= len(self.image_files):
            raise IndexError("Dataset index out of range")

        image_path = self.image_files[index]
        label_path = self.label_dir / f"{image_path.stem}.txt"

        try:
            image = read_image(image_path)
        except (FileNotFoundError, ValueError) as exc:
            logger.error("Failed to load image %s: %s", image_path, exc)
            raise

        if label_path.exists():
            try:
                annotations = read_yolo_label(label_path)
            except (FileNotFoundError, ValueError) as exc:
                logger.error("Failed to read labels %s: %s", label_path, exc)
                raise
        else:
            logger.warning("Label file missing for %s; returning empty target", image_path.name)
            annotations = []

        target: Dict[str, torch.Tensor] = {"boxes": [], "labels": []}
        boxes: List[List[float]] = []
        labels: List[int] = []

        for annotation in annotations:
            class_id, x_center, y_center, width, height = annotation
            class_id = int(class_id)
            converted = yolo_to_pascal_voc((class_id, x_center, y_center, width, height), image.shape)
            _, x_min, y_min, x_max, y_max = converted
            boxes.append([x_min, y_min, x_max, y_max])
            labels.append(class_id)

        target["boxes"] = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4) if boxes else torch.empty((0, 4), dtype=torch.float32)
        target["labels"] = torch.as_tensor(labels, dtype=torch.int64) if labels else torch.empty((0,), dtype=torch.int64)

        if self.transform is not None:
            try:
                transformed = self.transform(image=image, bboxes=target["boxes"].numpy(), class_labels=target["labels"].numpy())
            except Exception as exc:
                logger.error("Augmentation failed for %s: %s", image_path, exc)
                raise

            image = transformed["image"]
            target["boxes"] = torch.as_tensor(transformed["bboxes"], dtype=torch.float32).reshape(-1, 4)
            target["labels"] = torch.as_tensor(transformed["class_labels"], dtype=torch.int64)

        return image, target
