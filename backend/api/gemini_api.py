"""
Emotion Data Studio — Gemini Auto-Labeler API
============================================
FastAPI endpoints cho Gemini-powered auto-labeling.

Routes:
  GET  /api/gemini/status                — Kiểm tra cấu hình
  POST /api/gemini/analyze               — Phân tích 1 video
  POST /api/gemini/analyze-clip          — Verify 1 clip
  POST /api/gemini/batch                 — Batch analyze nhiều video
  GET  /api/gemini/segments              — List segments đã analyze
  POST /api/gemini/segments/{id}/apply   — Apply segment labels lên clip
  POST /api/gemini/cut-and-create        — AI Auto-Cut: quét + cắt + insert Clip
  GET  /api/gemini/estimate-cost         — Ước tính chi phí
"""

from __future__ import annotations

import os, json, logging
from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Query, BackgroundTasks
from pydantic import BaseModel, Field

from backend.config import settings
from backend.database.local_db import get_session
from backend.database.models import Clip, Video

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


class CutAndCreateRequest(BaseModel):
    video_id: str = Field(..., description="ID cua Video record trong DB")
    intensity_threshold: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    max_segments: Optional[int] = Field(default=None, ge=1, le=50)


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
    n_clips: int = Query(default=10, ge=0, le=200,
                         description="So clip du kien (Verify pass tinh theo so clip)"),
    avg_clip_duration: float = Query(default=7.0, gt=0, le=60,
                                     description="Duration trung binh moi clip"),
) -> dict[str, Any]:
    """Ước tính chi phí cho video có thời lượng N giây + n_clips verify."""
    from backend.services.ai_video_segmenter import AIVideoSegmenter

    breakdown = AIVideoSegmenter.estimate_total_cost(
        video_duration=duration_sec,
        n_clips=n_clips,
        avg_clip_duration=avg_clip_duration,
    )
    total_usd = breakdown["grand_total_usd"]
    return {
        "duration_sec": duration_sec,
        "n_clips": n_clips,
        "avg_clip_duration": avg_clip_duration,
        "total_usd": round(total_usd, 6),
        "scan_usd": breakdown["scan"]["total_usd"],
        "verify_usd": breakdown["verify"]["total_usd"],
        "breakdown": breakdown,
        "budget_27m_vnd_usd": 1000,
        "videos_covered": int(1000 / total_usd) if total_usd > 0 else 0,
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


# ── AI Auto-Cut: Vertex AI scan + FFmpeg cut + Insert Clip ──

@router.post("/cut-and-create")
async def cut_and_create(req: CutAndCreateRequest) -> dict[str, Any]:
    """
    AI Auto-Cut: Goi Vertex AI (Gemini) quet video, sau do FFmpeg cat truc tiep
    thanh cac file clip.mp4, va insert Clip records vao DB.

    Buoc 1: Lay Video tu DB theo video_id.
    Buoc 2: AIVideoSegmenter.cut_video() -> Vertex AI scan + FFmpeg cut.
    Buoc 3: persist_clips() -> bulk insert Clip records (status=needs_review).
    """
    session = get_session()
    try:
        video = session.query(Video).filter(Video.id == req.video_id).first()
        if not video:
            raise HTTPException(status_code=404, detail=f"Video not found: {req.video_id}")
        if not video.file_path or not Path(video.file_path).exists():
            raise HTTPException(
                status_code=400,
                detail=f"Video file not found on disk: {video.file_path}",
            )

        from backend.services.ai_video_segmenter import AIVideoSegmenter

        segmenter = AIVideoSegmenter()
        configured, msg = segmenter.is_configured()
        if not configured:
            raise HTTPException(
                status_code=503,
                detail=(
                    f"Vertex AI chua cau hinh: {msg}. "
                    "Hay dat GOOGLE_APPLICATION_CREDENTIALS hoac chay script test."
                ),
            )

        # Override settings neu user truyen
        original_threshold = settings.AI_AUTOCUT_INTENSITY_THRESHOLD
        original_max = settings.AI_AUTOCUT_MAX_SEGMENTS
        try:
            if req.intensity_threshold is not None:
                settings.AI_AUTOCUT_INTENSITY_THRESHOLD = req.intensity_threshold
            if req.max_segments is not None:
                settings.AI_AUTOCUT_MAX_SEGMENTS = req.max_segments

            result = segmenter.cut_video(
                video_path=video.file_path,
                video_id=video.id,
            )

            # ── Sprint 3: Stage 4 Verify pass (neu co clips) ────────────
            # Verify tung clip da cat de xac minh emotion + quality.
            # Chi chay khi AI_AUTOCUT_VERIFY_STRICT hoac khi co nhieu hon 0 clip.
            verify_summary: dict[str, Any] = {}
            if result.clips:
                verify_summary = {
                    "total": len(result.clips),
                    "passed": 0, "rejected": 0,
                    "errors": 0, "by_verdict": {},
                }
                for seg in result.clips:
                    try:
                        v_res = segmenter.verify_clip(
                            clip_path=seg.clip_path,
                            predicted_emotion=seg.emotion,
                            predicted_intensity=seg.intensity,
                            transcript="",
                            audio_features=None,
                            face_stats=None,
                        )
                        merged = segmenter.combine_verdicts(
                            {"emotion": seg.emotion, "intensity": seg.intensity}, v_res
                        )
                        seg.verify_verdict = merged.get("verify_verdict", "")
                        seg.verify_status = merged.get("verify_status", "passed")
                        seg.verify_reasoning = merged.get("verify_reasoning", "")
                        seg.rejected_by_verify = merged.get("rejected_by_verify", False)
                        seg.reject_reason = merged.get("reject_reason", "")
                        verdict = seg.verify_verdict or ""
                        verify_summary["by_verdict"][verdict] = (
                            verify_summary["by_verdict"].get(verdict, 0) + 1
                        )
                        if seg.verify_status == "passed":
                            verify_summary["passed"] += 1
                        elif seg.verify_status == "rejected":
                            verify_summary["rejected"] += 1
                        if v_res.get("error"):
                            verify_summary["errors"] += 1
                    except Exception as verify_exc:
                        logger.warning(f"[Verify] {seg.clip_id} loi: {verify_exc}")
                        verify_summary["errors"] += 1
                        seg.verify_verdict = "stats_mismatch"
                        seg.verify_status = "passed"

                # Strict mode: loai clip bi reject
                from backend.config import settings as _cfg
                if _cfg.AI_AUTOCUT_VERIFY_STRICT:
                    before = len(result.clips)
                    result.clips = [s for s in result.clips if not s.rejected_by_verify]
                    logger.info(
                        f"[Verify] strict: loai {before - len(result.clips)} clip, "
                        f"giu {len(result.clips)}"
                    )
                result.verify_summary = verify_summary
                result.stage4_verified = verify_summary.get("total", 0)

            inserted = segmenter.persist_clips(result, session)
        finally:
            settings.AI_AUTOCUT_INTENSITY_THRESHOLD = original_threshold
            settings.AI_AUTOCUT_MAX_SEGMENTS = original_max

        # Cap nhat Video metadata
        video.total_clips = (video.total_clips or 0) + inserted
        video.processing_mode = "ai_autocut"
        session.commit()

        return {
            "status": "ok",
            "video_id": video.id,
            "video_duration_sec": result.video_duration,
            "total_segments": result.total_segments,
            "clips_inserted": inserted,
            "estimated_cost_usd": result.total_cost_usd,
            # Sprint 3 — chi tiet cost
            "cost_breakdown": result.cost_breakdown,
            # Sprint 3 — stage counters
            "stage1_candidates": result.stage1_candidates,
            "stage2_passed": result.stage2_passed,
            "stage3_cut": result.stage3_cut,
            "stage4_verified": result.stage4_verified,
            # Sprint 2 — verify summary
            "verify_summary": result.verify_summary or {},
            "clips": [
                {
                    "clip_id": c.clip_id,
                    "clip_path": c.clip_path,
                    "start_time": c.start_time,
                    "end_time": c.end_time,
                    "duration": c.end_time - c.start_time,
                    "emotion": c.emotion,
                    "intensity": c.intensity,
                    "face_coverage": c.face_coverage,
                    "subject": c.subject,
                    "verify_verdict": c.verify_verdict or None,
                    "verify_status": c.verify_status or "not_run",
                    "rejected_by_verify": c.rejected_by_verify,
                    "reject_reason": c.reject_reason or None,
                }
                for c in result.clips
            ],
        }
    except HTTPException:
        raise
    except Exception as exc:
        session.rollback()
        logger.error(f"cut-and-create failed: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        session.close()
