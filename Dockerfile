# TODO: Define a minimal base image and production-ready build steps
FROM python:3.10-slim

WORKDIR /app

# TODO: Copy requirements and install dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# TODO: Copy source code and application files
COPY . /app

EXPOSE 8000

# TODO: Set a production entrypoint for the FastAPI app
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
