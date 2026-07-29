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
