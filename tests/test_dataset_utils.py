import tempfile
import unittest
from pathlib import Path

from src.dataset_utils import read_yolo_label


class ReadYoloLabelTests(unittest.TestCase):
    def test_missing_label_returns_empty_annotations(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            label_path = Path(tmp_dir) / "missing.txt"
            self.assertEqual(read_yolo_label(label_path), [])


if __name__ == "__main__":
    unittest.main()
