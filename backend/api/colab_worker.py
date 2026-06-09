"""
Emotion Data Studio — Colab GPU Worker API
=========================================
Cho phép Colab đăng ký như một remote GPU worker.
Local backend chỉ định job cho Colab → Colab chạy pipeline → trả kết quả.

Kiến trúc:
  Local Backend (port 8765)  ←→  ngrok tunnel  ←→  Colab GPU Worker

Flow:
  1. Colab khởi động → gọi POST /api/worker/register → đăng ký ngrok URL
  2. User thêm video → backend tạo queue item
  3. Colab gọi GET /api/worker/claim → nhận 1 job
  4. Colab xử lý → gọi POST /api/worker/complete với kết quả
  5. Backend cập nhật DB
"""

from __future__ import annotations

import os
import logging
from datetime import datetime, timedelta
from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from backend.config import settings
from backend.database.local_db import get_db
from backend.database.models import Video, Clip, ProcessQueue
from backend.database.local_db import SessionLocal

logger = logging.getLogger("EDS-ColabWorker")
router = APIRouter(prefix="/api/worker", tags=["Colab GPU Worker"])

# ── In-memory registry (for single-Colab setup) ───────────────────────────────
# Key: worker_id, Value: {url, registered_at, last_heartbeat, gpu_name}
_worker_registry: dict[str, dict] = {}


# ── Request / Response Schemas ───────────────────────────────────────────────

class WorkerRegisterRequest(BaseModel):
    worker_id: str = Field(..., description="Unique worker ID, e.g. 'colab-t4-01'")
    gpu_name: str = Field(default="unknown", description="GPU name, e.g. 'Tesla T4'")
    gpu_memory_gb: float = Field(default=0.0, description="GPU memory in GB")
    worker_url: Optional[str] = Field(default=None, description="Optional: direct URL if already exposed")
    capabilities: list[str] = Field(default_factory=lambda: ["gpu", "pipeline"], description="Capabilities: gpu, pipeline, gemini")


class WorkerRegisterResponse(BaseModel):
    status: str
    worker_id: str
    registered_at: str
    backend_url: str
    gpu_name: str


class ClaimResponse(BaseModel):
    status: str  # "ok" | "no_jobs" | "error"
    video_id: Optional[str] = None
    queue_item_id: Optional[int] = None
    video_title: Optional[str] = None
    video_url: Optional[str] = None
    video_path: Optional[str] = None
    duration_sec: Optional[float] = None
    message: Optional[str] = None


class CompleteRequest(BaseModel):
    worker_id: str
    queue_item_id: int
    video_id: str
    status: str = Field(..., description="'done', 'error', 'cancelled'")
    total_clips: int = 0
    approved_clips: int = 0
    error_msg: Optional[str] = None
    gemini_segments: Optional[list[dict]] = None


class CompleteResponse(BaseModel):
    status: str
    message: str


class HeartbeatRequest(BaseModel):
    worker_id: str
    gpu_utilization: Optional[float] = None
    gpu_memory_used_gb: Optional[float] = None
    processing_video_id: Optional[str] = None


# ── Helpers ────────────────────────────────────────────────────────────────

def _cleanup_stale_workers(max_idle_seconds: int = 300):
    """Remove workers that haven't sent heartbeat in max_idle_seconds."""
    cutoff = datetime.utcnow() - timedelta(seconds=max_idle_seconds)
    stale = [
        wid for wid, info in _worker_registry.items()
        if info.get("last_heartbeat", datetime.min) < cutoff
    ]
    for wid in stale:
        del _worker_registry[wid]
    if stale:
        logger.info(f"Cleaned up {len(stale)} stale worker(s): {stale}")


# ── Endpoints ──────────────────────────────────────────────────────────────

@router.post("/register", response_model=WorkerRegisterResponse)
async def register_worker(req: WorkerRegisterRequest) -> dict[str, Any]:
    """
    Colab gọi khi khởi động để đăng ký worker.
    Backend trả về worker_id để Colab dùng cho các lần gọi sau.
    """
    _cleanup_stale_workers()

    now = datetime.utcnow()
    _worker_registry[req.worker_id] = {
        "gpu_name": req.gpu_name,
        "gpu_memory_gb": req.gpu_memory_gb,
        "registered_at": now,
        "last_heartbeat": now,
        "worker_url": req.worker_url,
        "capabilities": req.capabilities,
        "active": True,
    }

    logger.info(
        f"🤖 Worker registered: {req.worker_id} | "
        f"GPU: {req.gpu_name} ({req.gpu_memory_gb}GB) | "
        f"Capabilities: {req.capabilities}"
    )

    return {
        "status": "registered",
        "worker_id": req.worker_id,
        "registered_at": now.isoformat(),
        "backend_url": "local",
        "gpu_name": req.gpu_name,
    }


@router.post("/heartbeat")
async def worker_heartbeat(req: HeartbeatRequest) -> dict[str, Any]:
    """Colab gửi heartbeat định kỳ (mỗi 30s) để giữ worker alive."""
    if req.worker_id not in _worker_registry:
        raise HTTPException(status_code=404, detail="Worker not registered")

    _worker_registry[req.worker_id]["last_heartbeat"] = datetime.utcnow()
    _worker_registry[req.worker_id]["gpu_utilization"] = req.gpu_utilization
    _worker_registry[req.worker_id]["gpu_memory_used_gb"] = req.gpu_memory_used_gb
    _worker_registry[req.worker_id]["processing_video_id"] = req.processing_video_id

    return {"status": "ok", "worker_id": req.worker_id, "time": datetime.utcnow().isoformat()}


@router.post("/unregister")
async def unregister_worker(worker_id: str) -> dict[str, Any]:
    """Colab gọi khi shutdown để unregister."""
    if worker_id in _worker_registry:
        del _worker_registry[worker_id]
        logger.info(f"Worker unregistered: {worker_id}")
        return {"status": "unregistered", "worker_id": worker_id}
    return {"status": "not_found", "worker_id": worker_id}


@router.get("/status")
async def worker_status() -> dict[str, Any]:
    """Trả về trạng thái tất cả workers + queue."""
    _cleanup_stale_workers()

    workers = []
    for wid, info in _worker_registry.items():
        workers.append({
            "worker_id": wid,
            "gpu_name": info.get("gpu_name"),
            "gpu_memory_gb": info.get("gpu_memory_gb"),
            "registered_at": info.get("registered_at").isoformat() if info.get("registered_at") else None,
            "last_heartbeat": info.get("last_heartbeat").isoformat() if info.get("last_heartbeat") else None,
            "processing_video_id": info.get("processing_video_id"),
            "capabilities": info.get("capabilities", []),
        })

    session = SessionLocal()
    try:
        queued = session.query(ProcessQueue).filter(ProcessQueue.status == "queued").count()
        running = session.query(ProcessQueue).filter(ProcessQueue.status == "running").count()
        done = session.query(ProcessQueue).filter(ProcessQueue.status == "done").count()
        error = session.query(ProcessQueue).filter(ProcessQueue.status == "error").count()
    finally:
        if hasattr(session, "close"):
            session.close()

    return {
        "workers": workers,
        "worker_count": len(workers),
        "queue": {
            "queued": queued,
            "running": running,
            "done": done,
            "error": error,
        },
        "has_idle_worker": any(
            w.get("processing_video_id") is None for w in workers
        ),
    }


@router.get("/claim", response_model=ClaimResponse)
async def claim_job(worker_id: str) -> dict[str, Any]:
    """
    Colab gọi để nhận 1 job từ queue.
    Trả về video info nếu có job, hoặc {'status': 'no_jobs'} nếu không.
    """
    _cleanup_stale_workers()

    if worker_id not in _worker_registry:
        raise HTTPException(status_code=403, detail="Worker not registered. Call /register first.")

    session = SessionLocal()
    try:
        # Pick the highest-priority pending job
        item = (
            session.query(ProcessQueue)
            .filter(ProcessQueue.status == "queued")
            .order_by(ProcessQueue.priority.desc(), ProcessQueue.id.asc())
            .first()
        )

        if not item:
            return {
                "status": "no_jobs",
                "message": "Không có job nào trong queue. Thêm video để bắt đầu.",
            }

        # Mark as running
        item.status = "running"
        item.started_at = datetime.utcnow()
        session.commit()

        # Get video info
        video = session.query(Video).filter(Video.id == item.video_id).first()
        if not video:
            item.status = "error"
            item.error_msg = "Video not found"
            session.commit()
            raise HTTPException(status_code=404, detail="Video not found in DB")

        video.status = "processing"
        session.commit()

        logger.info(f"🤖 Job claimed: {item.video_id} by worker {worker_id}")

        return {
            "status": "ok",
            "video_id": video.id,
            "queue_item_id": item.id,
            "video_title": video.title,
            "video_url": video.source_url,
            "video_path": video.file_path,
            "duration_sec": video.duration_sec,
            "message": f"Job assigned: {video.title}",
        }

    except HTTPException:
        raise
    except Exception as exc:
        session.rollback()
        logger.error(f"Claim error: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        session.close()


@router.post("/complete", response_model=CompleteResponse)
async def complete_job(req: CompleteRequest) -> dict[str, Any]:
    """
    Colab gọi khi xử lý xong để báo kết quả.
    Backend cập nhật ProcessQueue và Video records.
    """
    session = SessionLocal()
    try:
        item = session.query(ProcessQueue).filter(ProcessQueue.id == req.queue_item_id).first()
        if not item:
            raise HTTPException(status_code=404, detail="Queue item not found")

        video = session.query(Video).filter(Video.id == req.video_id).first()

        item.status = req.status
        item.completed_at = datetime.utcnow()
        if req.status == "error":
            item.error_msg = (req.error_msg or "Unknown error")[:1000]

        if video:
            if req.status == "done":
                video.status = "completed"
            else:
                video.status = req.status
            video.total_clips = req.total_clips
            video.approved_clips = req.approved_clips
            video.updated_at = datetime.utcnow()

            # Apply Gemini segments if provided
            if req.gemini_segments and video.id:
                _apply_gemini_segments(session, video.id, req.gemini_segments)

        session.commit()

        logger.info(
            f"✅ Job complete: {req.video_id} | status={req.status} | "
            f"clips={req.total_clips} approved={req.approved_clips}"
        )

        return {
            "status": "ok",
            "message": f"Job {req.video_id} marked as {req.status}",
        }

    except HTTPException:
        raise
    except Exception as exc:
        session.rollback()
        logger.error(f"Complete error: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        session.close()


@router.post("/skip")
async def skip_job(worker_id: str, queue_item_id: int) -> dict[str, Any]:
    """Worker yêu cầu bỏ qua job hiện tại, quay lại queue."""
    session = SessionLocal()
    try:
        item = session.query(ProcessQueue).filter(ProcessQueue.id == queue_item_id).first()
        if not item:
            raise HTTPException(status_code=404, detail="Queue item not found")

        item.status = "queued"
        item.started_at = None
        session.commit()

        return {"status": "ok", "message": f"Job {queue_item_id} returned to queue"}
    except HTTPException:
        raise
    except Exception as exc:
        session.rollback()
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        session.close()


# ── Internal helpers ───────────────────────────────────────────────────────

def _apply_gemini_segments(session: Session, video_id: str, segments: list[dict]):
    """Tạo clips từ Gemini segments đã phân tích."""
    for idx, seg in enumerate(segments):
        start = float(seg.get("start_time", 0))
        end = float(seg.get("end_time", 0))
        emotion = str(seg.get("emotion", "neutral")).lower()
        intensity = float(seg.get("intensity", 0.7))
        reasoning = str(seg.get("reasoning", ""))[:500]

        # Check duplicate
        existing = session.query(Clip).filter(
            Clip.video_id == video_id,
            Clip.start_time == start,
            Clip.end_time == end,
        ).first()

        if existing:
            existing.predicted_emotion = emotion
            existing.confidence = intensity
            existing.review_notes = f"[Gemini auto] {reasoning}"
            existing.decision_by = "gemini"
        else:
            clip = Clip(
                video_id=video_id,
                clip_index=idx,
                start_time=start,
                end_time=end,
                duration=end - start,
                predicted_emotion=emotion,
                confidence=intensity,
                status="needs_review",
                decision_by="gemini",
                review_notes=f"[Gemini auto] {reasoning}",
            )
            session.add(clip)
