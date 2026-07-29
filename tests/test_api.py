"""API tests for the Custom Object Detector service.

Uses FastAPI's TestClient to exercise the endpoints. The heavy model
loading and inference paths are mocked so tests run quickly and do not
depend on the presence of GPU or large model artifacts.
"""

from __future__ import annotations

import io
import tempfile
from unittest.mock import Mock, patch

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.api import app


@pytest.fixture(autouse=True)
def patch_model_load_and_predict():
    """Patch model loading and prediction to avoid heavy dependencies.

    The fixture yields while patches are active for each test that uses
    the TestClient (the startup event will use the patched `load_model`).
    """
    mock_model = Mock()
    with patch("src.api.load_model", return_value=mock_model), patch(
        "src.api.predict_image",
        return_value=[{"class_name": "helmet", "confidence": 0.97, "bbox": [1, 2, 3, 4]}],
    ):
        yield


def make_temp_image() -> str:
    """Create a small temporary JPEG image and return its path.

    Returns:
        The filesystem path to the created JPEG file.
    """
    arr = np.zeros((64, 64, 3), dtype=np.uint8)
    arr[10:30, 10:30] = (255, 0, 0)
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
    cv2.imwrite(tmp.name, cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))
    tmp.close()
    return tmp.name


def test_root_endpoint():
    with TestClient(app) as client:
        resp = client.get("/")
    assert resp.status_code == 200
    body = resp.json()
    assert "message" in body
    assert "version" in body


def test_health_endpoint():
    with TestClient(app) as client:
        resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert "status" in body
    assert "model_loaded" in body
    assert "version" in body


def test_predict_endpoint_success():
    image_path = make_temp_image()
    with open(image_path, "rb") as fh, TestClient(app) as client:
        files = {"file": ("test.jpg", fh, "image/jpeg")}
        resp = client.post("/predict", files=files)

    assert resp.status_code == 200
    body = resp.json()
    assert "predictions" in body
    assert isinstance(body["predictions"], list)
    assert body["predictions"]
    pred = body["predictions"][0]
    assert "class_name" in pred
    assert "confidence" in pred
    assert "bbox" in pred


def test_predict_invalid_upload_returns_400():
    # Upload a non-image file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as tmp:
        tmp.write(b"not an image")
        tmp.flush()
        tmp_path = tmp.name

    with open(tmp_path, "rb") as fh, TestClient(app) as client:
        files = {"file": ("bad.txt", fh, "text/plain")}
        resp = client.post("/predict", files=files)

    assert resp.status_code == 400
