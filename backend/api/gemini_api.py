"""
Emotion Data Studio — Gemini Auto-Labeler API
============================================
FastAPI endpoints cho Gemini-powered auto-labeling.

Routes:
  GET  /api/gemini/status          — Kiểm tra cấu hình
  POST /api/gemini/analyze         — Phân tích 1 video
  POST /api/gemini/analyze-clip    — Verify 1 clip
  POST /api/gemini/batch           — Batch analyze nhiều video
  GET  /api/gemini/segments        — List segments đã analyze
  POST /api/gemini/segments/{id}/apply — Apply segment labels lên clip
"""

from __future__ import annotations

import os, json, logging
from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Query, BackgroundTasks
from pydantic import BaseModel, Field

from backend.config import settings
from backend.database.local_db import get_session
from backend.database.models import Clip

logger = logging.getLogger("EDS-Gemini-API")

router = APIRouter(prefix="/api/gemini", tags=["Gemini Auto-Labeler"])


# ── Request/Response Schemas ────────────────────────────────

class AnalyzeVideoRequest(BaseModel):
    video_path: Optional[str] = None
    gcs_uri: Optional[str] = None
    intensity_threshold: float = Field(default=0.6, ge=0.0, le=1.0)
    max_segments: int = Field(default=20, ge=1, le=50)


class AnalyzeClipRequest(BaseModel):
    clip_id: Optional[int] = None
    clip_path: Optional[str] = None
    intensity_threshold: float = Field(default=0.5, ge=0.0, le=1.0)


class BatchAnalyzeRequest(BaseModel):
    video_paths: list[str]
    intensity_threshold: float = Field(default=0.6, ge=0.0, le=1.0)
    max_segments_per_video: int = Field(default=20, ge=1, le=50)


class ApplySegmentRequest(BaseModel):
    segment_index: int
    emotion: str
    intensity: float


# ── Helpers ────────────────────────────────────────────────

def _get_labeler():
    from backend.services.gemini_auto_labeler import GeminiAutoLabeler
    return GeminiAutoLabeler()


# ── Status ────────────────────────────────────────────────

@router.get("/status")
async def gemini_status() -> dict[str, Any]:
    """Kiểm tra trạng thái cấu hình Gemini."""
    try:
        labeler = _get_labeler()
        return labeler.status()
    except ImportError as exc:
        return {
            "configured": False,
            "message": f"google-genai package chưa cài: {exc}",
            "model": "gemini-2.5-flash",
        }
    except Exception as exc:
        return {
            "configured": False,
            "message": str(exc),
            "model": "gemini-2.5-flash",
        }


# ── Analyze Video ─────────────────────────────────────────

@router.post("/analyze")
async def analyze_video(req: AnalyzeVideoRequest) -> dict[str, Any]:
    """
    Phân tích 1 video để tìm các đoạn cảm xúc mạnh.
    Trả về list segments với start_time, end_time, emotion, intensity.
    """
    try:
        labeler = _get_labeler()
        configured, msg = labeler.is_configured()
        if not configured:
            raise HTTPException(status_code=503, detail=f"Gemini chưa cấu hình: {msg}")

        result = labeler.analyze_video(
            video_path=req.video_path,
            gcs_uri=req.gcs_uri,
            intensity_threshold=req.intensity_threshold,
            max_segments=req.max_segments,
        )

        return {
            "status": "ok",
            "segments": result["segments"],
            "segment_count": len(result["segments"]),
            "video_duration": result["video_duration"],
            "estimated_cost_usd": result["total_cost_usd"],
            "model": result["model_used"],
            "cost_estimate": result["cost_estimate"],
        }

    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    except Exception as exc:
        logger.error(f"Gemini analyze failed: {exc}")
        raise HTTPException(status_code=500, detail=f"Lỗi phân tích: {exc}")


# ── Analyze Clip ──────────────────────────────────────────

@router.post("/analyze-clip")
async def analyze_clip(req: AnalyzeClipRequest) -> dict[str, Any]:
    """
    Verify/re-score 1 clip đã cắt bằng Gemini.
    Dùng clip_id (ưu tiên) hoặc clip_path.
    """
    try:
        labeler = _get_labeler()
        configured, msg = labeler.is_configured()
        if not configured:
            raise HTTPException(status_code=503, detail=f"Gemini chưa cấu hình: {msg}")

        clip_path = req.clip_path
        if req.clip_id:
            session = get_session()
            try:
                clip = session.query(Clip).filter(Clip.id == req.clip_id).first()
                if not clip:
                    raise HTTPException(status_code=404, detail="Clip not found")
                clip_path = clip.clip_path
            finally:
                session.close()

        if not clip_path:
            raise HTTPException(status_code=400, detail="clip_id or clip_path required")

        result = labeler.analyze_clip(
            clip_path=clip_path,
            intensity_threshold=req.intensity_threshold,
        )

        return {
            "status": "ok",
            "analysis": result["analysis"],
            "duration": result["duration"],
            "estimated_cost_usd": result["total_cost_usd"],
        }

    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"Gemini clip analyze failed: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))


# ── Batch Analyze ─────────────────────────────────────────

@router.post("/batch")
async def batch_analyze(req: BatchAnalyzeRequest) -> dict[str, Any]:
    """
    Phân tích nhiều video liên tiếp.
    Kết quả lưu tạm vào thư mục cache.
    """
    try:
        labeler = _get_labeler()
        configured, msg = labeler.is_configured()
        if not configured:
            raise HTTPException(status_code=503, detail=f"Gemini chưa cấu hình: {msg}")

        results = labeler.batch_analyze(
            video_paths=req.video_paths,
            intensity_threshold=req.intensity_threshold,
            max_segments_per_video=req.max_segments_per_video,
        )

        # Cache results
        cache_dir = settings.DATA_DIR / "cache" / "gemini_batch"
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = cache_dir / f"batch_{os.getpid()}.json"
        cache_file.write_text(
            json.dumps(results, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        total_segments = sum(len(r.get("segments", [])) for r in results)
        total_cost = sum(r.get("total_cost_usd", 0) for r in results)
        errors = sum(1 for r in results if "error" in r)

        return {
            "status": "ok",
            "total_videos": len(results),
            "total_segments": total_segments,
            "errors": errors,
            "estimated_total_cost_usd": round(total_cost, 4),
            "cache_file": str(cache_file),
            "results": [
                {
                    "video_path": r.get("video_path") or r.get("gcs_uri"),
                    "segments": r.get("segments", []),
                    "error": r.get("error"),
                }
                for r in results
            ],
        }

    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"Gemini batch failed: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))


# ── Cost Estimation ───────────────────────────────────────

@router.get("/estimate-cost")
async def estimate_cost(
    duration_sec: float = Query(..., gt=0, description="Video duration in seconds"),
) -> dict[str, Any]:
    """Ước tính chi phí cho video có thời lượng N giây."""
    labeler = _get_labeler()
    return {
        "duration_sec": duration_sec,
        **labeler._estimate_cost(duration_sec),
        "budget_27m_vnd_usd": 1000,  # ~27M VND ≈ $1000
        "videos_covered": int(1000 / labeler._estimate_cost(duration_sec)["estimated_total_usd"])
        if labeler._estimate_cost(duration_sec)["estimated_total_usd"] > 0 else 0,
    }


# ── Apply segment labels to clips ─────────────────────────

@router.post("/segments/{video_id}/apply")
async def apply_segment_labels(
    video_id: str,
    segments: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Áp dụng các segment labels lên clips của 1 video.
    Tạo hoặc cập nhật Clip records với emotion/intensity từ Gemini.

    Body: [{"start_time": 12.5, "end_time": 28.3, "emotion": "angry", "intensity": 0.87}, ...]
    """
    session = get_session()
    try:
        from backend.database.models import Video, Clip
        from datetime import datetime

        video = session.query(Video).filter(Video.id == video_id).first()
        if not video:
            raise HTTPException(status_code=404, detail="Video not found")

        applied = 0
        skipped = 0

        for seg in segments:
            start = float(seg["start_time"])
            end = float(seg["end_time"])
            emotion = str(seg["emotion"]).lower()
            intensity = float(seg.get("intensity", 0))

            # Check if clip already exists at this time range
            existing = session.query(Clip).filter(
                Clip.video_id == video_id,
                Clip.start_time == start,
                Clip.end_time == end,
            ).first()

            if existing:
                existing.predicted_emotion = emotion
                existing.confidence = intensity
                existing.review_notes = f"[Gemini auto] {seg.get('reasoning', '')}"
                existing.decision_by = "gemini"
                existing.updated_at = datetime.utcnow()
                skipped += 1
            else:
                new_clip = Clip(
                    video_id=video_id,
                    clip_index=0,  # Will be recalculated
                    start_time=start,
                    end_time=end,
                    duration=end - start,
                    predicted_emotion=emotion,
                    confidence=intensity,
                    status="needs_review",
                    decision_by="gemini",
                    review_notes=f"[Gemini auto] {seg.get('reasoning', '')}",
                )
                session.add(new_clip)
                applied += 1

        session.commit()

        return {
            "status": "ok",
            "video_id": video_id,
            "segments_applied": applied,
            "segments_updated": skipped,
            "total": applied + skipped,
        }

    except HTTPException:
        raise
    except Exception as exc:
        session.rollback()
        logger.error(f"Apply segments failed: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        session.close()
