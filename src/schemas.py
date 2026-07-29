"""Pydantic schemas for the Custom Object Detector API.

Defines request/response models used by the FastAPI application.
"""

from __future__ import annotations

from typing import List

from pydantic import BaseModel


class Prediction(BaseModel):
    """A single object detection prediction.

    Attributes:
        class_name: Human-friendly class label.
        confidence: Detection confidence score between 0 and 1.
        bbox: Bounding box in XYXY format: [x1, y1, x2, y2].
    """

    class_name: str
    confidence: float
    bbox: List[float]


class PredictionResponse(BaseModel):
    """Response model for the /predict endpoint.

    Attributes:
        predictions: A list of detected objects.
    """

    predictions: List[Prediction]


class HealthResponse(BaseModel):
    """Health-check response model.

    Attributes:
        status: Service health status ("healthy"/"unhealthy").
        model_loaded: Whether the model is available for inference.
        version: API semantic version string.
    """

    status: str
    model_loaded: bool
    version: str
