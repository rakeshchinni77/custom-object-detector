"""Tests for dataset utilities and the YOLODataset class.

Creates ephemeral image/label files in a temporary directory so tests do not
modify repository data. Validates loading, annotation parsing, tensor shapes,
and error handling when images are missing.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from src.dataset import YOLODataset
from src.dataset_utils import read_yolo_label


def _write_image(path: Path, shape=(100, 200, 3)) -> None:
    arr = np.zeros(shape, dtype=np.uint8)
    arr[10:40, 20:60] = (255, 255, 255)
    cv2.imwrite(str(path), cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))


def test_dataset_initialization_and_loading():
    with tempfile.TemporaryDirectory() as tmpdir:
        base = Path(tmpdir)
        images = base / "images"
        labels = base / "labels"
        images.mkdir()
        labels.mkdir()

        img_path = images / "000.jpg"
        lbl_path = labels / "000.txt"
        _write_image(img_path)

        # write a single YOLO annotation: class 0 centered
        lbl_path.write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")

        dataset = YOLODataset(image_dir=images, label_dir=labels, class_names=["head", "helmet"])
        assert len(dataset) > 0

        image, target = dataset[0]
        # image is an ndarray with 3 channels
        assert hasattr(image, "shape") and image.shape[2] == 3

        assert "boxes" in target and "labels" in target
        assert isinstance(target["boxes"], torch.Tensor)
        assert isinstance(target["labels"], torch.Tensor)
        assert target["boxes"].ndim == 2 and target["boxes"].shape[1] == 4


def test_missing_image_raises():
    with tempfile.TemporaryDirectory() as tmpdir:
        base = Path(tmpdir)
        images = base / "images"
        labels = base / "labels"
        images.mkdir()
        labels.mkdir()

        img_path = images / "000.jpg"
        lbl_path = labels / "000.txt"
        _write_image(img_path)
        lbl_path.write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")

        dataset = YOLODataset(image_dir=images, label_dir=labels)

        # remove image file to simulate missing image
        img_path.unlink()

        with pytest.raises(FileNotFoundError):
            _ = dataset[0]
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
from torch.utils.data import DataLoader

from src.augmentations import get_valid_transforms
from src.dataset import YOLODataset
from src.dataset_utils import read_yolo_label, SUPPORTED_IMAGE_EXTENSIONS


class YOLODatasetTests(unittest.TestCase):
    def _create_image(self, directory: Path, name: str) -> Path:
        path = directory / name
        image = np.full((100, 120, 3), 128, dtype=np.uint8)
        cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        return path

    def test_dataset_loads_image_and_labels(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            data_dir = Path(tmp_dir)
            image_dir = data_dir / "images"
            label_dir = data_dir / "labels"
            image_dir.mkdir(parents=True)
            label_dir.mkdir(parents=True)

            image_path = self._create_image(image_dir, "sample.jpg")
            label_path = label_dir / "sample.txt"
            label_path.write_text("0 0.5 0.5 0.5 0.5\n", encoding="utf-8")

            dataset = YOLODataset(image_dir=image_dir, label_dir=label_dir, transform=get_valid_transforms(64), class_names=["obj"])
            image, target = dataset[0]

            self.assertEqual(image.shape[0], 3)
            self.assertEqual(target["boxes"].shape, (1, 4))
            self.assertEqual(target["labels"].item(), 0)

    def test_dataset_returns_empty_target_for_missing_label(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            data_dir = Path(tmp_dir)
            image_dir = data_dir / "images"
            label_dir = data_dir / "labels"
            image_dir.mkdir(parents=True)
            label_dir.mkdir(parents=True)

            self._create_image(image_dir, "sample.png")
            dataset = YOLODataset(image_dir=image_dir, label_dir=label_dir, transform=get_valid_transforms(64))
            image, target = dataset[0]

            self.assertEqual(target["boxes"].shape, (0, 4))
            self.assertEqual(target["labels"].shape, (0,))


class DatasetUtilsTests(unittest.TestCase):
    def test_supported_image_extensions_contains_jpeg(self) -> None:
        self.assertIn(".jpg", SUPPORTED_IMAGE_EXTENSIONS)
        self.assertIn(".jpeg", SUPPORTED_IMAGE_EXTENSIONS)


if __name__ == "__main__":
    unittest.main()
