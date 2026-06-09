from __future__ import annotations

import time
from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy import func
from sqlalchemy.orm import Session

from backend.database.local_db import get_db
from backend.database.models import Clip, ProcessQueue, Video

router = APIRouter(prefix="/stats", tags=["Stats"])

EMOTION_QUOTAS = {
    "happy": 600,
    "sad": 450,
    "angry": 450,
    "neutral": 450,
    "surprise": 450,
    "fear": 300,
    "disgust": 300,
}


@router.get("/")
def dashboard_stats(db: Session = Depends(get_db)):
    total_clips = db.query(Clip).count()
    approved_clips = db.query(Clip).filter(Clip.status.in_(["approved", "auto_approved"])).count()
    pending_clips = db.query(Clip).filter(Clip.status.in_(["pending", "needs_review"])).count()
    rejected_clips = db.query(Clip).filter(Clip.status == "rejected").count()
    total_videos = db.query(Video).count()

    rows = (
        db.query(Clip.predicted_emotion, func.count(Clip.id))
        .filter(Clip.status.in_(["approved", "auto_approved"]), Clip.predicted_emotion.isnot(None))
        .group_by(Clip.predicted_emotion)
        .all()
    )
    counts = {emotion: count for emotion, count in rows}
    quota = {
        emotion: {
            "count": counts.get(emotion, 0),
            "target": target,
            "remaining": max(target - counts.get(emotion, 0), 0),
            "percentage": round(counts.get(emotion, 0) / target * 100, 2) if target else 0.0,
        }
        for emotion, target in EMOTION_QUOTAS.items()
    }

    queue_counts = dict(
        db.query(ProcessQueue.status, func.count(ProcessQueue.id)).group_by(ProcessQueue.status).all()
    )

    return {
        "total_clips": total_clips,
        "approved_clips": approved_clips,
        "pending_clips": pending_clips,
        "rejected_clips": rejected_clips,
        "total_videos": total_videos,
        "approval_progress": round(approved_clips / 3000 * 100, 2),
        "emotion_quota": quota,
        "queue": queue_counts,
        "auto_approved": db.query(Clip).filter(Clip.status == "approved", Clip.decision_by == "auto").count(),
        "human_reviewed": db.query(Clip).filter(Clip.decision_by == "human").count(),
    }


class ModelStatusResponse(BaseModel):
    key: str
    loaded: bool
    error: str | None


class ModelDetail(BaseModel):
    loaded: bool
    error: str | None
    model_type: str
    notes: str


@router.get("/models", response_model=dict[str, ModelDetail])
def model_status():
    """Return current status of all AI models used by the emotion ensemble."""
    try:
        from backend.ai_models.model_manager import model_manager
        statuses = model_manager.status()
        details = {}
        for s in statuses:
            if s.key == "whisper":
                notes = "ASR for Vietnamese transcription. Falls back to faster-whisper."
            elif s.key == "deepface":
                notes = "Facial emotion recognition (7 emotions). TensorFlow backend."
            elif s.key == "mtcnn":
                notes = "Face detection for crop extraction."
            elif s.key == "text_emotion":
                notes = "PhoBERT zero-shot + Vietnamese lexicon. Loads lazily on first text inference."
            elif s.key == "audio_emotion":
                notes = "74-dim COVAREP features + template cosine sim. MLP checkpoint optional."
            else:
                notes = ""

            details[s.key] = ModelDetail(
                loaded=s.loaded,
                error=s.error,
                model_type=s.key,
                notes=notes,
            )
        return details
    except Exception as e:
        return {"_error": ModelDetail(loaded=False, error=str(e), model_type="unknown", notes="")}


@router.get("/models/check")
def model_health_check():
    """Lightweight health check — tests that models can respond without loading heavy weights."""
    try:
        from backend.ai_models.model_manager import model_manager
        statuses = model_manager.status()
        loaded = [s for s in statuses if s.loaded]
        failed = [s for s in statuses if s.error]
        return {
            "status": "healthy" if not failed else "degraded",
            "loaded_count": len(loaded),
            "total_count": len(statuses),
            "models": {s.key: s.loaded for s in statuses},
            "errors": {s.key: s.error for s in failed},
        }
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}


@router.get("/quota")
def emotion_quota(db: Session = Depends(get_db)):
    """Return current emotion quota progress."""
    rows = (
        db.query(Clip.predicted_emotion, func.count(Clip.id))
        .filter(Clip.status.in_(["approved", "auto_approved"]), Clip.predicted_emotion.isnot(None))
        .group_by(Clip.predicted_emotion)
        .all()
    )
    counts = {emotion: count for emotion, count in rows}
    return {
        emotion: {
            "count": counts.get(emotion, 0),
            "target": target,
            "remaining": max(target - counts.get(emotion, 0), 0),
            "percentage": round(counts.get(emotion, 0) / target * 100, 2) if target else 0.0,
        }
        for emotion, target in EMOTION_QUOTAS.items()
    }
