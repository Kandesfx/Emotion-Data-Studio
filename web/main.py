"""
Emotion Data Studio — Web Dashboard Backend
==========================================
FastAPI server chạy trên Colab GPU, phục vụ HTML/JS dashboard qua ngrok.

Mỗi endpoint trả JSON cho API calls và HTML cho page renders.

Usage (Colab):
    !python web/main.py
    # Server chạy tại port 8765, truy cập qua ngrok

Usage (local dev):
    python web/main.py --reload
"""

from __future__ import annotations

import os
import sys
import json
import logging
import base64
from pathlib import Path
from datetime import datetime
from typing import Any, Optional

# Ensure project root in path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import uvicorn
from fastapi import FastAPI, HTTPException, Query, Request, Response
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("EDS-Web")

# =============================================================================
# App Setup
# =============================================================================

app = FastAPI(
    title="EDS Web Dashboard",
    description="Emotion Data Studio — Colab GPU Processing Dashboard",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Paths
WEB_DIR = Path(__file__).parent
TEMPLATE_DIR = WEB_DIR / "templates"
STATIC_DIR = WEB_DIR / "static"
COLAB_ROOT = WEB_DIR.parent

# Mount static files
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Import and register routers
sys.path.insert(0, str(COLAB_ROOT))
from backend.api.gemini_api import router as gemini_router
from backend.api.colab_worker import router as colab_worker_router
app.include_router(gemini_router)
app.include_router(colab_worker_router)

# =============================================================================
# Database helpers
# =============================================================================

def _get_session():
    """Get a local SQLite session."""
    from backend.database.local_db import get_session
    return get_session()


def _get_model(key: str, default=None):
    """Safely get a model from model_manager."""
    try:
        from backend.ai_models.model_manager import model_manager
        return model_manager.get_model(key)
    except Exception:
        return default


# =============================================================================
# Dashboard API
# =============================================================================

@app.get("/api/stats")
async def get_stats() -> dict[str, Any]:
    """Return aggregate stats for dashboard display."""
    from backend.database.models import Video, Clip
    from sqlalchemy import func

    session = _get_session()
    try:
        total_videos = session.query(func.count(Video.id)).scalar() or 0
        total_clips = session.query(func.count(Clip.id)).scalar() or 0
        approved = session.query(func.count(Clip.id)).filter(
            Clip.status == "approved"
        ).scalar() or 0
        rejected = session.query(func.count(Clip.id)).filter(
            Clip.status == "rejected"
        ).scalar() or 0
        pending = session.query(func.count(Clip.id)).filter(
            Clip.status == "needs_review"
        ).scalar() or 0

        # Emotion quota targets (from project config)
        EMOTION_TARGETS = {
            "happy": 600, "sad": 450, "angry": 450,
            "fear": 300, "surprise": 300, "disgust": 300,
            "neutral": 600,
        }
        emotion_counts = {}
        for emotion in EMOTION_TARGETS:
            emotion_counts[emotion] = {
                "count": session.query(func.count(Clip.id)).filter(
                    Clip.predicted_emotion == emotion,
                    Clip.status == "approved",
                ).scalar() or 0,
                "target": EMOTION_TARGETS[emotion],
            }

        # Queue stats
        from backend.database.models import ProcessQueue
        queue_pending = session.query(func.count(ProcessQueue.id)).filter(
            ProcessQueue.status == "pending"
        ).scalar() or 0
        queue_running = session.query(func.count(ProcessQueue.id)).filter(
            ProcessQueue.status == "running"
        ).scalar() or 0
        queue_done = session.query(func.count(ProcessQueue.id)).filter(
            ProcessQueue.status == "done"
        ).scalar() or 0

        # Auto vs human
        auto_approved = session.query(func.count(Clip.id)).filter(
            Clip.status == "approved",
            Clip.review_type == "auto",
        ).scalar() or 0
        human_reviewed = session.query(func.count(Clip.id)).filter(
            Clip.status == "approved",
            Clip.review_type == "human",
        ).scalar() or 0

        total_target = sum(EMOTION_TARGETS.values())
        return {
            "total_videos": total_videos,
            "total_clips": total_clips,
            "approved": approved,
            "rejected": rejected,
            "pending_review": pending,
            "emotion_quota": emotion_counts,
            "emotion_target_total": total_target,
            "queue": {
                "pending": queue_pending,
                "running": queue_running,
                "completed": queue_done,
            },
            "auto_approved": auto_approved,
            "human_reviewed": human_reviewed,
            "approval_rate": round(approved / total_clips * 100, 1) if total_clips > 0 else 0,
        }
    finally:
        session.close()


# =============================================================================
# Clips API
# =============================================================================

@app.get("/api/clips")
async def list_clips(
    status: str = Query(None, description="Filter by status"),
    emotion: str = Query(None, description="Filter by predicted emotion"),
    video_id: str = Query(None, description="Filter by video ID"),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
) -> dict[str, Any]:
    """List clips with optional filters."""
    from backend.database.models import Clip
    from sqlalchemy import func

    session = _get_session()
    try:
        q = session.query(Clip)

        if status:
            q = q.filter(Clip.status == status)
        if emotion:
            q = q.filter(Clip.predicted_emotion == emotion)
        if video_id:
            q = q.filter(Clip.video_id == video_id)

        total = q.count()
        clips = q.order_by(Clip.id.desc()).offset(offset).limit(limit).all()

        items = [_clip_dict(c) for c in clips]
        return {"items": items, "total": total, "limit": limit, "offset": offset}
    finally:
        session.close()


@app.get("/api/clips/{clip_id}")
async def get_clip(clip_id: int) -> dict[str, Any]:
    """Get a single clip with all details."""
    from backend.database.models import Clip

    session = _get_session()
    try:
        clip = session.query(Clip).filter(Clip.id == clip_id).first()
        if not clip:
            raise HTTPException(status_code=404, detail="Clip not found")
        return _clip_dict(clip)
    finally:
        session.close()


@app.put("/api/clips/{clip_id}")
async def update_clip(clip_id: int, data: dict) -> dict[str, Any]:
    """Update clip label, status, sentiment, or review info."""
    from backend.database.models import Clip

    session = _get_session()
    try:
        clip = session.query(Clip).filter(Clip.id == clip_id).first()
        if not clip:
            raise HTTPException(status_code=404, detail="Clip not found")

        # Allowed updates
        allowed = {
            "status", "user_emotion", "user_sentiment",
            "review_type", "review_notes",
        }
        for key in allowed:
            if key in data:
                setattr(clip, key, data[key])

        clip.updated_at = datetime.utcnow()
        session.commit()

        return {"status": "ok", "clip": _clip_dict(clip)}
    except HTTPException:
        raise
    except Exception as e:
        session.rollback()
        raise HTTPException(status_code=400, detail=str(e))
    finally:
        session.close()


def _clip_dict(clip) -> dict[str, Any]:
    """Convert Clip ORM object to dict for JSON serialization."""
    from backend.database.models import Clip
    cols = [c.key for c in Clip.__table__.columns]
    result = {c: _safe_getattr(clip, c) for c in cols}
    # Serialize per_model_scores JSON if present
    if hasattr(clip, "per_model_scores") and clip.per_model_scores:
        try:
            result["per_model_scores"] = json.loads(clip.per_model_scores)
        except Exception:
            result["per_model_scores"] = clip.per_model_scores
    return result


def _safe_getattr(obj, name: str) -> Any:
    try:
        val = getattr(obj, name, None)
        if hasattr(val, "isoformat"):
            return val.isoformat()
        return val
    except Exception:
        return None


# =============================================================================
# Videos API
# =============================================================================

@app.get("/api/videos")
async def list_videos(
    status: str = Query(None),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
) -> dict[str, Any]:
    """List videos."""
    from backend.database.models import Video
    from sqlalchemy import func

    session = _get_session()
    try:
        q = session.query(Video)
        if status:
            q = q.filter(Video.status == status)
        total = q.count()
        videos = q.order_by(Video.id.desc()).offset(offset).limit(limit).all()
        items = [_video_dict(v) for v in videos]
        return {"items": items, "total": total, "limit": limit, "offset": offset}
    finally:
        session.close()


@app.get("/api/videos/{video_id}")
async def get_video(video_id: int) -> dict[str, Any]:
    """Get a single video."""
    from backend.database.models import Video
    session = _get_session()
    try:
        video = session.query(Video).filter(Video.id == video_id).first()
        if not video:
            raise HTTPException(status_code=404, detail="Video not found")
        return _video_dict(video)
    finally:
        session.close()


@app.post("/api/videos/{video_id}/process")
async def trigger_video_process(video_id: int) -> dict[str, Any]:
    """Queue a video for processing."""
    from backend.database.models import ProcessQueue
    session = _get_session()
    try:
        video = session.query(Video).filter(Video.id == video_id).first()
        if not video:
            raise HTTPException(status_code=404, detail="Video not found")

        # Check if already queued
        existing = session.query(ProcessQueue).filter(
            ProcessQueue.video_id == video_id,
            ProcessQueue.status.in_(["pending", "running"]),
        ).first()
        if existing:
            return {"status": "already_queued", "queue_id": existing.id}

        queue_item = ProcessQueue(
            video_id=video_id,
            status="pending",
            priority=1,
        )
        session.add(queue_item)
        session.commit()
        return {"status": "queued", "queue_id": queue_item.id}
    finally:
        session.close()


def _video_dict(video) -> dict[str, Any]:
    from backend.database.models import Video
    cols = [c.key for c in Video.__table__.columns]
    result = {}
    for c in cols:
        val = _safe_getattr(video, c)
        result[c] = val
    return result


# =============================================================================
# Harvest API
# =============================================================================

@app.post("/api/harvest")
async def harvest_urls(data: dict) -> dict[str, Any]:
    """Import URLs into the harvest queue and optionally start processing."""
    from backend.database.models import Video, ProcessQueue
    from urllib.parse import urlparse

    urls: list[str] = data.get("urls", [])
    target_emotions: list[str] = data.get("target_emotions", [])
    start_processing: bool = data.get("start_processing", False)

    if not urls:
        raise HTTPException(status_code=400, detail="No URLs provided")

    session = _get_session()
    try:
        def detect_source(url):
            host = urlparse(url).netloc.lower()
            if "youtube" in host or "youtu.be" in host:
                return "youtube"
            if "drive.google" in host:
                return "drive"
            if "tiktok" in host:
                return "tiktok"
            if "facebook" in host or "fb.watch" in host:
                return "facebook"
            return "url"

        added = []
        errors = []
        for url in urls:
            try:
                video = Video(
                    title=url,
                    source_url=url.strip(),
                    source_type=detect_source(url.strip()),
                    status="queued",
                    target_emotion=target_emotions[0] if target_emotions else None,
                )
                session.add(video)
                session.flush()

                q_item = ProcessQueue(
                    video_id=video.id,
                    status="queued",
                    priority=1,
                    target_emotion=target_emotions[0] if target_emotions else None,
                )
                session.add(q_item)
                session.commit()
                session.refresh(video)
                added.append({"url": url, "video_id": video.id})
            except Exception as exc:
                session.rollback()
                errors.append({"url": url, "error": str(exc)})

        return {"added": added, "errors": errors, "total": len(added)}
    finally:
        session.close()


@app.get("/api/queue")
async def get_queue() -> dict[str, Any]:
    """Return current processing queue status."""
    from backend.database.models import ProcessQueue, Video
    from sqlalchemy import func

    session = _get_session()
    try:
        items = (
            session.query(ProcessQueue, Video)
            .join(Video, ProcessQueue.video_id == Video.id)
            .order_by(ProcessQueue.priority.desc(), ProcessQueue.created_at.asc())
            .all()
        )
        return {
            "items": [
                {
                    "queue_id": q.id,
                    "video_id": q.video_id,
                    "title": v.title if v else "Unknown",
                    "status": q.status,
                    "priority": q.priority,
                    "created_at": _safe_getattr(q, "created_at"),
                    "started_at": _safe_getattr(q, "started_at"),
                    "completed_at": _safe_getattr(q, "completed_at"),
                    "error": q.error_msg,
                }
                for q, v in items
            ]
        }
    finally:
        session.close()


@app.post("/api/queue/pause")
async def pause_queue() -> dict[str, Any]:
    """Pause the processing queue."""
    from backend.database.models import ProcessQueue
    session = _get_session()
    try:
        paused = session.query(ProcessQueue).filter(
            ProcessQueue.status == "pending"
        ).update({"priority": 0})
        session.commit()
        return {"status": "paused", "paused_count": paused}
    finally:
        session.close()


@app.post("/api/queue/resume")
async def resume_queue() -> dict[str, Any]:
    """Resume the processing queue."""
    from backend.database.models import ProcessQueue
    session = _get_session()
    try:
        resumed = session.query(ProcessQueue).filter(
            ProcessQueue.status == "pending",
            ProcessQueue.priority == 0,
        ).update({"priority": 1})
        session.commit()
        return {"status": "resumed", "resumed_count": resumed}
    finally:
        session.close()


# =============================================================================
# Export API
# =============================================================================

@app.post("/api/export")
async def export_dataset(data: dict) -> dict[str, Any]:
    """Trigger dataset export to .pkl for MulT training."""
    from backend.services.exporters.mmsa_exporter import MMSAExporter
    from backend.database.models import Clip

    output_path = data.get("output_path", "/tmp/emotions_dataset.pkl")
    require_aligned = data.get("require_aligned", True)

    session = _get_session()
    try:
        exporter = MMSAExporter(session)
        result = exporter.export(
            output_path=output_path,
            feature_dir=str(COLAB_ROOT / "data" / "features"),
            require_aligned=require_aligned,
        )
        return result
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        session.close()


# =============================================================================
# Settings API
# =============================================================================

@app.get("/api/settings")
async def get_settings() -> dict[str, Any]:
    """Return current pipeline settings."""
    from backend.config import settings
    return {
        "scene_threshold": settings.SCENE_THRESHOLD,
        "min_clip_duration": settings.MIN_CLIP_DURATION,
        "max_clip_duration": settings.MAX_CLIP_DURATION,
        "smart_face_confidence": settings.SMART_FACE_CONFIDENCE,
        "smart_target_clip_duration": settings.SMART_TARGET_CLIP_DURATION,
        "emotion_weights": {"visual": 0.4, "audio": 0.3, "text": 0.3},
    }


@app.put("/api/settings")
async def update_settings(data: dict) -> dict[str, Any]:
    """Update pipeline settings (runtime only — saved to user_settings.json)."""
    import json as _json
    from backend.config import settings

    settings_path = settings.user_settings_path
    try:
        current = _json.loads(settings_path.read_text(encoding="utf-8"))
    except Exception:
        current = {}

    current.setdefault("pipeline", {}).update(data.get("pipeline", {}))
    settings_path.parent.mkdir(parents=True, exist_ok=True)
    settings_path.write_text(_json.dumps(current, indent=2, ensure_ascii=False), encoding="utf-8")

    return {"status": "saved", "settings": current}


# =============================================================================
# Stream endpoints (video / audio playback)
# =============================================================================

@app.get("/api/clips/{clip_id}/video")
async def stream_clip_video(clip_id: int) -> Response:
    """Stream clip video file."""
    from backend.database.models import Clip
    session = _get_session()
    try:
        clip = session.query(Clip).filter(Clip.id == clip_id).first()
        if not clip or not clip.clip_path:
            raise HTTPException(status_code=404, detail="Clip video not found")

        video_path = Path(clip.clip_path)
        if not video_path.exists():
            raise HTTPException(status_code=404, detail="Video file not found")

        with open(video_path, "rb") as f:
            data = f.read()
        return Response(
            content=data,
            media_type="video/mp4",
            headers={"Accept-Ranges": "bytes"},
        )
    finally:
        session.close()


@app.get("/api/clips/{clip_id}/audio")
async def stream_clip_audio(clip_id: int) -> Response:
    """Stream clip audio file."""
    from backend.database.models import Clip
    session = _get_session()
    try:
        clip = session.query(Clip).filter(Clip.id == clip_id).first()
        if not clip or not clip.audio_path:
            raise HTTPException(status_code=404, detail="Audio not found")

        audio_path = Path(clip.audio_path)
        if not audio_path.exists():
            raise HTTPException(status_code=404, detail="Audio file not found")

        with open(audio_path, "rb") as f:
            data = f.read()
        return Response(
            content=data,
            media_type="audio/wav",
        )
    finally:
        session.close()


@app.get("/api/clips/{clip_id}/frame")
async def get_clip_thumbnail(clip_id: int, t: float = Query(0.0)) -> Response:
    """Extract a frame from clip at time t (seconds)."""
    from backend.database.models import Clip
    import cv2

    session = _get_session()
    try:
        clip = session.query(Clip).filter(Clip.id == clip_id).first()
        if not clip or not clip.clip_path:
            raise HTTPException(status_code=404, detail="Clip not found")

        video_path = Path(clip.clip_path)
        if not video_path.exists():
            raise HTTPException(status_code=404, detail="Video file not found")

        cap = cv2.VideoCapture(str(video_path))
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ret, frame = cap.read()
        cap.release()

        if not ret or frame is None:
            raise HTTPException(status_code=404, detail="Could not extract frame")

        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return Response(content=buf.tobytes(), media_type="image/jpeg")
    finally:
        session.close()


# =============================================================================
# Pipeline control
# =============================================================================

@app.post("/api/pipeline/start")
async def start_pipeline() -> dict[str, Any]:
    """Start the background pipeline worker."""
    import threading

    def run():
        from backend.services.pipeline_orchestrator import PipelineOrchestrator
        from backend.database.models import ProcessQueue

        while True:
            try:
                session = _get_session()
                try:
                    item = (
                        session.query(ProcessQueue)
                        .filter(ProcessQueue.status == "queued")
                        .order_by(ProcessQueue.priority.desc(), ProcessQueue.created_at.asc())
                        .first()
                    )
                    if not item:
                        import time
                        time.sleep(5)
                        continue

                    item.status = "running"
                    item.started_at = datetime.utcnow()
                    session.commit()

                    orchestrator = PipelineOrchestrator()
                    orchestrator.process_video(video_id=item.video_id)

                    item.status = "done"
                    item.completed_at = datetime.utcnow()
                    session.commit()
                finally:
                    session.close()
            except Exception as exc:
                import time, logging
                logging.getLogger("Pipeline").error(f"Pipeline error: {exc}")
                time.sleep(10)

    threading.Thread(target=run, daemon=True).start()
    return {"status": "started"}


# =============================================================================
# HTML Page Renderers
# =============================================================================

def _render_template(name: str, **context) -> HTMLResponse:
    """Render an HTML template with context."""
    template_path = TEMPLATE_DIR / f"{name}.html"
    if not template_path.exists():
        raise HTTPException(status_code=404, detail=f"Template {name} not found")

    try:
        content = template_path.read_text(encoding="utf-8")
        # Simple {{variable}} replacement
        for key, value in context.items():
            placeholder = f"{{{{{key}}}}}"
            if placeholder in content:
                content = content.replace(placeholder, str(value))
        return HTMLResponse(content=content)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/", response_class=HTMLResponse)
async def dashboard_page(request: Request) -> HTMLResponse:
    return _render_template("dashboard")


@app.get("/review", response_class=HTMLResponse)
async def review_page(request: Request) -> HTMLResponse:
    return _render_template("review")


@app.get("/harvest", response_class=HTMLResponse)
async def harvest_page(request: Request) -> HTMLResponse:
    return _render_template("harvest")


@app.get("/settings", response_class=HTMLResponse)
async def settings_page(request: Request) -> HTMLResponse:
    return _render_template("settings")


@app.get("/export", response_class=HTMLResponse)
async def export_page(request: Request) -> HTMLResponse:
    return _render_template("export")


@app.get("/gemini", response_class=HTMLResponse)
async def gemini_page(request: Request) -> HTMLResponse:
    return _render_template("gemini")


# =============================================================================
# Health
# =============================================================================

@app.get("/health")
async def health() -> dict[str, Any]:
    gpu_name = "Unknown"
    gpu_memory = 0
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory // (1024**3)
    except Exception:
        pass

    return {
        "status": "healthy",
        "gpu": gpu_name,
        "gpu_memory_gb": gpu_memory,
        "colab": os.path.exists("/content") or os.path.exists("/drive"),
    }


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="EDS Web Dashboard")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()

    # Init database
    try:
        from backend.database.local_db import init_database
        init_database()
        logger.info("Database initialized")
    except Exception as exc:
        logger.warning(f"Database init skipped: {exc}")

    uvicorn.run(
        "web.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )
