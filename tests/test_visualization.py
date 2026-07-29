import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from src.visualization import draw_bounding_boxes, save_visualization


class VisualizationTests(unittest.TestCase):
    def test_draw_bounding_boxes_converts_float_image(self) -> None:
        image = np.ones((50, 50, 3), dtype=np.float32) * 255.0
        boxes = [[10.5, 10.5, 40.7, 40.7]]
        labels = ["obj"]

        annotated = draw_bounding_boxes(image, boxes, labels)
        self.assertEqual(annotated.dtype, np.uint8)
        self.assertEqual(annotated.shape, image.shape)

    def test_save_visualization_saves_file(self) -> None:
        image = np.ones((50, 50, 3), dtype=np.uint8) * 128
        box = [[5, 5, 30, 30]]
        labels = ["obj"]

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "vis.png"
            result_path = save_visualization(image, output_path, boxes=box, labels=labels)
            self.assertTrue(result_path.exists())
            loaded = cv2.imread(str(result_path))
            self.assertIsNotNone(loaded)
            self.assertEqual(loaded.shape[0], 50)
            self.assertEqual(loaded.shape[1], 50)


if __name__ == "__main__":
    unittest.main()
