# Custom Object Detector API

A production-ready REST API for custom object detection built with YOLOv8, FastAPI, Docker, and automated GitHub Actions CI. The system loads a pretrained model, serves predictions over HTTP, and supports containerized deployment.

## Features

- **Object Detection** using YOLOv8 and OpenCV
- **REST API** served by FastAPI
- **Docker Deployment** for portable production execution
- **GitHub Actions CI** for automated validation
- **Health Endpoint** for readiness checks
- **Prediction Endpoint** for image inference

## Technologies Used

- Python
- YOLOv8 (Ultralytics)
- FastAPI
- Docker
- OpenCV
- PyTorch
- Albumentations
- NumPy
- GitHub Actions

## Architecture Diagram

```text
Inference Pipeline (Docker)
  Client Application
      |
      | HTTP POST /predict (Image)
      v
  FastAPI Server
      |
      | Preprocess / Load
      v
  Inference Engine
      |
      | Postprocess (NMS)
      v
  JSON Output

Training Pipeline
  Raw Dataset (Images & Annotations)
      |
      | Split & Augment
      v
  Data Loader
      |
      | Batches
      v
  Training Loop
   / |  |  \ \
  /  |  |   \ \
Pretrained  Config  Data Loader  Validate
 Weights     File    (Batches)     
             (yaml)              
                |                
                v                
            Training Loop        
                |
                | Calculate Loss & Optimize
                v
         Trained Model Weights
                |
                v
          Evaluation Script (mAP)
```

## Project Structure

```text
.
├── .github/
│   └── workflows/
│       └── submission.yml
├── .dockerignore
├── .env.example
├── .gitattributes
├── .gitignore
├── config/
│   └── config.yaml
├── data/
│   ├── data.yaml
│   ├── raw/
│   ├── test/
│   ├── train/
│   └── valid/
├── docs/
│   └── dataset.md
├── Dockerfile
├── docker-compose.yml
├── models/
│   ├── pretrained/
│   └── train/
│       └── weights/
│           └── best.pt
├── outputs/
├── requirements-docker.txt
├── requirements.txt
├── src/
│   ├── api.py
│   ├── config.py
│   ├── dataset.py
│   ├── dataset_utils.py
│   ├── evaluate.py
│   ├── metrics.py
│   ├── predict.py
│   ├── schemas.py
│   ├── train.py
│   ├── visualization.py
│   ├── split_dataset.py
│   ├── verify_dataset.py
│   └── augmentations.py
├── submission.yml
├── tests/
│   ├── test_api.py
│   └── test_dataset.py
└── README.md
```

## Installation

Clone repository:

```bash
git clone "https://github.com/rakeshchinni77/custom-object-detector"
```

Change into the project directory:

```bash
cd custom-object-detector
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate the environment:

Windows:

```powershell
.venv\Scripts\activate
```

Linux/macOS:

```bash
source .venv/bin/activate
```

Upgrade pip:

```bash
python -m pip install --upgrade pip
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

## Dataset Preparation

The dataset is expected to come from Roboflow and is stored under `data/`. The repository includes dataset metadata and splits for training, validation, and testing.

If the dataset files are tracked with Git LFS, install Git LFS before cloning and pulling the repository.

```text
data/
   train/
   valid/
   test/
   data.yaml
```

## Configuration

Model and training settings are defined in `config/config.yaml`. This file includes parameters such as:

- `batch size`
- `epochs`
- `image size`
- `optimizer`
- `learning rate`

Use `config/config.yaml` to adjust training behavior without changing source code.

## Training

Run the training pipeline with:

```bash
python -m src.train
```

Training will automatically skip retraining if `models/train/weights/best.pt` already exists.

## Evaluation

Run evaluation with:

```bash
python -m src.evaluate
```

The evaluation workflow produces:

```text
outputs/
    predictions/
    metrics.json
    metrics.txt
```

## Docker

Build the Docker image:

```bash
docker build -t object-detector .
```

Run the container:

```bash
docker run -d -p 8000:8000 object-detector
```

Or use Docker Compose:

```bash
docker compose up --build
```

## API Documentation

The FastAPI application exposes these endpoints:

- `GET /` — root endpoint returning service information
- `GET /health` — health check endpoint reporting readiness and model status
- `POST /predict` — prediction endpoint accepting multipart image uploads

Swagger UI is available at `http://localhost:8000/docs` when the service is running.

## cURL Example

Windows:

```powershell
curl -X POST ^
  http://localhost:8000/predict ^
  -F "file=@helmet.jpg"
```

Linux/macOS:

```bash
curl -X POST \
  http://localhost:8000/predict \
  -F "file=@helmet.jpg"
```

## Example JSON Response

```json
{
  "predictions": [
    {
      "class_name": "helmet",
      "confidence": 0.97,
      "bbox": [86.45, 21.68, 197.02, 154.96]
    }
  ]
}
```

> Note: `class_name` is the detected object label returned by the API.

## Running Tests

Run the test suite with:

```bash
pytest
```

## Results

- **mAP@0.5:** 0.6244
- **mAP@0.5:0.95:** 0.4355
- **Precision:** 0.6357
- **Recall:** 0.6116


## License

This project is licensed under the terms in `LICENSE`.

## Author

Custom Object Detector API project author.
