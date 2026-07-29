"""FastAPI application for the Custom Object Detector.

Implements a thin API that delegates inference to ``src.predict``. The
API exposes health and prediction endpoints and provides automatic OpenAPI
documentation (Swagger) at ``/docs``.
"""

from __future__ import annotations

import logging
import shutil
import tempfile
from pathlib import Path
from typing import List

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from src.predict import load_model, predict_image, preprocess_image
from src.schemas import HealthResponse, Prediction, PredictionResponse

VERSION = "1.0"

logger = logging.getLogger("custom_object_detector.api")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

app = FastAPI(title="Custom Object Detector API", version=VERSION)


@app.on_event("startup")
def startup_event() -> None:
    """Application startup: attempt to load the trained model into memory.

    We store the loaded model on ``app.state.model`` for reuse by request
    handlers. If model loading fails we still start the server but report
    the model missing on the health endpoint.
    """
    logger.info("API startup: loading model")
    try:
        app.state.model = load_model()
        app.state.model_loaded = True
        logger.info("Model loaded into app.state.model")
    except Exception as exc:  # pragma: no cover - runtime environment
        app.state.model = None
        app.state.model_loaded = False
        logger.error("Model failed to load on startup: %s", exc)


@app.get("/", response_class=JSONResponse)
def read_root() -> JSONResponse:
    """Root endpoint identifying the service.

    Returns a small JSON payload with a version string.
    """
    logger.info("GET / request")
    return JSONResponse({"message": "Custom Object Detector API", "version": VERSION})


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Health check for readiness and model availability.

    Returns whether the service is healthy and whether the model is loaded.
    """
    logger.info("GET /health request")
    status = "healthy" if app.state.model_loaded else "unhealthy"
    return HealthResponse(status=status, model_loaded=bool(app.state.model_loaded), version=VERSION)


try:
    @app.post("/predict", response_model=PredictionResponse)
    async def predict(file: UploadFile = File(...)) -> PredictionResponse:
        """Accept an image upload, run object detection, and return predictions.

        The uploaded file is temporarily saved to disk and validated using the
        shared ``preprocess_image`` function from ``src.predict``. After
        inference the temporary file is removed. Errors are returned with
        appropriate HTTP status codes.
        """
        logger.info("POST /predict request received: filename=%s", file.filename)

        if not app.state.model_loaded or app.state.model is None:
            logger.error("Prediction requested but model is not loaded")
            raise HTTPException(status_code=404, detail="Model not found")

        # Save upload to a temporary file for predict.py to consume
        suffix = Path(file.filename).suffix or ".jpg"
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        try:
            contents = await file.read()
            tmp.write(contents)
            tmp.flush()
            tmp_path = Path(tmp.name)
            tmp.close()

            # Validate image; preprocess_image raises ValueError/FileNotFound
            try:
                preprocess_image(tmp_path)
            except FileNotFoundError as exc:
                logger.exception("Uploaded image not found or missing: %s", exc)
                raise HTTPException(status_code=400, detail=str(exc))
            except ValueError as exc:
                logger.exception("Uploaded image invalid: %s", exc)
                raise HTTPException(status_code=400, detail=str(exc))

            # Run prediction (business logic lives in src.predict)
            logger.info("Starting prediction for %s", tmp_path.name)
            try:
                preds = predict_image(tmp_path, model=app.state.model)
            except Exception as exc:  # pragma: no cover - runtime dependent
                logger.exception("Prediction failed: %s", exc)
                raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}")

            # Convert to Pydantic models and return
            predictions: List[Prediction] = [
                Prediction(class_name=p["class_name"], confidence=p["confidence"], bbox=p["bbox"]) for p in preds
            ]

            logger.info("Prediction completed for %s: %d objects", tmp_path.name, len(predictions))
            return PredictionResponse(predictions=predictions)

        finally:
            try:
                if not tmp.closed:
                    tmp.close()
            except Exception:
                pass
            try:
                Path(tmp.name).unlink(missing_ok=True)
                logger.info("Deleted temporary file %s", tmp.name)
            except Exception:
                logger.warning("Failed to delete temporary file %s", tmp.name)
except RuntimeError as err:  # missing optional multipart dependency
    logger.warning("Failed to register /predict endpoint at import time: %s", err)

    @app.post("/predict", response_model=PredictionResponse)
    async def predict_stub() -> PredictionResponse:
        """Stub endpoint when multipart dependency is missing.

        This returns a 500 with guidance so the operator can install the
        required dependency (`python-multipart`). The real endpoint will
        work when the dependency is available.
        """
        logger.error("Predict endpoint unavailable: %s", err)
        raise HTTPException(
            status_code=500,
            detail=(
                "Server missing required dependency 'python-multipart'. "
                "Install with: pip install python-multipart"
            ),
        )
