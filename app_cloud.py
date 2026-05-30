"""
Emotion Data Studio — Cloud Run API Entry Point
=================================================
FastAPI server for cloud deployment.
Desktop app syncs data through this REST API.

This is SEPARATE from the desktop app.py entry point.
The desktop app calls backend services directly (no HTTP).
This server exists only for:
  - Cloud-side data access (by team members)
  - Cloud SQL sync endpoint
  - Health checks

Usage (local): uvicorn app_cloud:app --port 8080
Usage (Cloud Run): Deployed via Dockerfile
"""

import os
import sys
import logging

# Ensure project root is in path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("EDS-Cloud")

# Create FastAPI app
app = FastAPI(
    title="Emotion Data Studio API",
    description="Cloud API for data synchronization and team collaboration",
    version="1.0.0",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize database
from backend.database.local_db import init_database
init_database()
logger.info("Database initialized")


# ============================================================
# Health & Status
# ============================================================

@app.get("/health")
async def health():
    """Health check endpoint for Cloud Run"""
    return {"status": "healthy", "version": "1.0.0"}


@app.get("/api/status")
async def status():
    """Server status with database info"""
    from backend.database.local_db import get_session
    from backend.database.models import Video, Clip

    session = get_session()
    try:
        total_videos = session.query(Video).count()
        total_clips = session.query(Clip).count()
        return {
            "status": "running",
            "videos": total_videos,
            "clips": total_clips,
            "environment": os.getenv("ENV", "development"),
        }
    finally:
        session.close()


# ============================================================
# Include API routers (from existing backend/api/)
# ============================================================

try:
    from backend.api.videos import router as videos_router
    from backend.api.clips import router as clips_router
    from backend.api.labels import router as labels_router

    app.include_router(videos_router, prefix="/api/videos", tags=["videos"])
    app.include_router(clips_router, prefix="/api/clips", tags=["clips"])
    app.include_router(labels_router, prefix="/api/labels", tags=["labels"])
    logger.info("API routers loaded")
except ImportError as e:
    logger.warning(f"Could not load API routers: {e}")


# ============================================================
# Sync Endpoints
# ============================================================

@app.post("/api/sync/upload")
async def sync_upload(data: dict):
    """
    Receive sync data from desktop app.
    Desktop → Cloud SQL.
    """
    # TODO: Implement sync receive endpoint
    return {"status": "received", "records": 0}


@app.get("/api/sync/download")
async def sync_download(since: str = None):
    """
    Send sync data to desktop app.
    Cloud SQL → Desktop.
    """
    # TODO: Implement sync send endpoint
    return {"status": "ok", "records": [], "since": since}


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
