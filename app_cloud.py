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

from backend.config import settings

# Create FastAPI app
app = FastAPI(
    title="Emotion Data Studio API",
    description="Cloud API for data synchronization and team collaboration",
    version=settings.VERSION,
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
    return {"status": "healthy", "version": settings.VERSION}


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
    from backend.api.stats import router as stats_router

    # Routers already have internal prefix (/videos, /clips, etc.)
    # Only add /api at the app level
    app.include_router(videos_router, prefix="/api", tags=["videos"])
    app.include_router(clips_router, prefix="/api", tags=["clips"])
    app.include_router(labels_router, prefix="/api", tags=["labels"])
    app.include_router(stats_router, prefix="/api", tags=["stats"])
    logger.info("API routers loaded")
except ImportError as e:
    logger.warning(f"Could not load API routers: {e}")


# ============================================================
# Sync Endpoints
# ============================================================

@app.post("/api/sync/upload", tags=["sync"])
async def sync_upload(data: dict):
    """
    Receive sync data from desktop app.
    Desktop → Cloud SQL.
    """
    try:
        from backend.cloud.sync_manager import SyncManager
        manager = SyncManager()
        report = manager.sync_metadata(direction="upload")
        return {"status": "success", "report": report}
    except Exception as e:
        logger.error(f"Sync upload failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/sync/download", tags=["sync"])
async def sync_download(since: str = None):
    """
    Send sync data to desktop app.
    Cloud SQL → Desktop.
    """
    try:
        from backend.cloud.sync_manager import SyncManager
        manager = SyncManager()
        report = manager.sync_metadata(direction="download")
        return {"status": "success", "report": report}
    except Exception as e:
        logger.error(f"Sync download failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/sync/files", tags=["sync"])
async def sync_files_upload(sync_videos: bool = False, sync_clips: bool = True, sync_audio: bool = True):
    """
    Upload processed files to Google Cloud Storage.
    """
    try:
        from backend.cloud.sync_manager import SyncManager
        manager = SyncManager()
        report = manager.sync_files(
            sync_videos=sync_videos,
            sync_clips=sync_clips,
            sync_audio=sync_audio,
        )
        return {"status": "success", "report": report}
    except Exception as e:
        logger.error(f"File sync failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/sync/status", tags=["sync"])
async def sync_status():
    """Get current sync status."""
    try:
        from backend.cloud.sync_manager import SyncManager
        manager = SyncManager()
        return {
            "available": manager.is_available,
            "last_sync": manager.get_sync_status(),
        }
    except Exception as e:
        return {"available": False, "error": str(e)}


@app.post("/api/sync/full", tags=["sync"])
async def sync_full(sync_videos: bool = False):
    """
    Run full bidirectional sync (metadata + files).
    """
    try:
        from backend.cloud.sync_manager import SyncManager
        manager = SyncManager()
        report = manager.full_sync(sync_videos=sync_videos)
        return {"status": "success", "report": report}
    except Exception as e:
        logger.error(f"Full sync failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
