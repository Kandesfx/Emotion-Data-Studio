# ============================================================
# Emotion Data Studio — Cloud Run Dockerfile
# ============================================================
# Deploys the FastAPI backend as a REST API on Google Cloud Run.
# Desktop app syncs data through this API.
#
# Build: docker build -t eds-api -f deploy/Dockerfile .
# Run:   docker run -p 8080:8080 eds-api
# ============================================================

FROM python:3.12-slim

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first (cache layer)
COPY requirements-cloud.txt .
RUN pip install --no-cache-dir -r requirements-cloud.txt

# Copy application code
COPY backend/ backend/
COPY app_cloud.py .

# Create data directory
RUN mkdir -p /app/data

# Environment variables
ENV PORT=8080
ENV ENV=production
ENV EDS_DATA_DIR=/app/data
ENV PYTHONUNBUFFERED=1

# Health check
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:${PORT}/health')"

# Run FastAPI with uvicorn
CMD exec uvicorn app_cloud:app --host 0.0.0.0 --port ${PORT} --workers 2
