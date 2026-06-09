from __future__ import annotations

import os
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import List
from urllib.parse import urlparse

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from sqlalchemy.orm import Session

from backend.api.schemas import DriveHarvestRequest, HarvestRequest, QueueItemResponse, VideoResponse
from backend.config import settings
from backend.database.local_db import SessionLocal, get_db
from backend.database.models import ProcessQueue, Video

router = APIRouter(prefix="/harvest", tags=["Harvest"])

VIDEO_EXTENSIONS = {".mp4", ".mkv", ".avi", ".webm", ".mov", ".m4v"}


def _detect_source_type(url: str) -> str:
    host = urlparse(url).netloc.lower()
    if "youtube" in host or "youtu.be" in host:
        return "youtube"
    if "drive.google" in host:
        return "drive"
    if "tiktok" in host:
        return "tiktok"
    if "facebook" in host or "fb.watch" in host:
        return "facebook"
    if "dailymotion" in host:
        return "dailymotion"
    return "url"


def _enqueue_video(db: Session, video: Video, priority: int = 0, target_emotion: str | None = None) -> ProcessQueue:
    item = ProcessQueue(
        video_id=video.id,
        priority=priority,
        target_emotion=target_emotion,
        status="queued",
    )
    video.status = "queued"
    video.target_emotion = target_emotion
    db.add(item)
    db.commit()
    db.refresh(item)
    return item


def _run_queue_once() -> dict:
    from backend.services.pipeline_orchestrator import PipelineOrchestrator

    db = SessionLocal()
    processed = 0
    errors = 0
    try:
        while True:
            item = (
                db.query(ProcessQueue)
                .filter(ProcessQueue.status == "queued")
                .order_by(ProcessQueue.priority.desc(), ProcessQueue.id.asc())
                .first()
            )
            if not item:
                break

            item.status = "running"
            item.started_at = datetime.utcnow()
            db.commit()

            try:
                PipelineOrchestrator().run_pipeline(item.video_id, db)
                item.status = "done"
                item.completed_at = datetime.utcnow()
                processed += 1
            except Exception as exc:  # keep queue resilient
                item.status = "error"
                item.error_msg = str(exc)[:1000]
                video = db.query(Video).filter(Video.id == item.video_id).first()
                if video:
                    video.status = "error"
                    video.error_msg = str(exc)[:1000]
                errors += 1
            db.commit()
    finally:
        db.close()

    return {"processed": processed, "errors": errors}


@router.post("/", response_model=List[VideoResponse])
def harvest_urls(payload: HarvestRequest, db: Session = Depends(get_db)):
    """Import URL list (YouTube/playlist/channel/TikTok/Facebook/etc.) into the Colab EDS queue."""
    urls = [url.strip() for url in payload.urls if url and url.strip()]
    if not urls:
        raise HTTPException(status_code=400, detail="Danh sách URL rỗng")

    videos: list[Video] = []
    for url in urls:
        video = Video(
            title=url,
            source_url=url,
            source_type=_detect_source_type(url),
            status="queued",
            target_emotion=payload.target_emotion,
        )
        db.add(video)
        db.flush()
        _enqueue_video(db, video, payload.priority, payload.target_emotion)
        videos.append(video)

    if payload.auto_start:
        _run_queue_once()

    return videos


@router.post("/drive", response_model=VideoResponse)
def harvest_drive_link(payload: DriveHarvestRequest, db: Session = Depends(get_db)):
    """Import a Google Drive shared file/folder link. Download is handled later by the pipeline/queue."""
    video = Video(
        title=payload.drive_url,
        source_url=payload.drive_url,
        source_type="drive",
        status="queued",
        target_emotion=payload.target_emotion,
    )
    db.add(video)
    db.flush()
    _enqueue_video(db, video, payload.priority, payload.target_emotion)
    return video


@router.post("/scan-inbox", response_model=List[VideoResponse])
def scan_inbox(target_emotion: str | None = None, priority: int = 0, db: Session = Depends(get_db)):
    """Scan the Drive/local inbox folder and enqueue new video files."""
    inbox = settings.inbox_dir
    videos_dir = settings.DATA_DIR / "videos"
    inbox.mkdir(parents=True, exist_ok=True)
    videos_dir.mkdir(parents=True, exist_ok=True)

    imported: list[Video] = []
    for file_path in sorted(inbox.iterdir()):
        if not file_path.is_file() or file_path.suffix.lower() not in VIDEO_EXTENSIONS:
            continue

        exists = db.query(Video).filter(Video.file_path == str(file_path)).first()
        if exists:
            continue

        safe_name = f"{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{file_path.name}"
        dest = videos_dir / safe_name
        shutil.move(str(file_path), str(dest))

        video = Video(
            title=file_path.stem,
            source_type="inbox",
            file_path=str(dest),
            status="queued",
            target_emotion=target_emotion,
        )
        db.add(video)
        db.flush()
        _enqueue_video(db, video, priority, target_emotion)
        imported.append(video)

    return imported


@router.post("/upload", response_model=VideoResponse)
async def upload_video(
    file: UploadFile = File(...),
    target_emotion: str | None = None,
    priority: int = 0,
    db: Session = Depends(get_db),
):
    """Direct upload fallback for dashboard/ngrok workflows."""
    suffix = Path(file.filename or "video.mp4").suffix.lower()
    if suffix not in VIDEO_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Định dạng video không hỗ trợ: {suffix}")

    uploads_dir = settings.DATA_DIR / "videos"
    uploads_dir.mkdir(parents=True, exist_ok=True)
    filename = f"upload_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{Path(file.filename or 'video').name}"
    dest = uploads_dir / filename

    with dest.open("wb") as out_file:
        shutil.copyfileobj(file.file, out_file)

    video = Video(
        title=Path(file.filename or filename).stem,
        source_type="upload",
        file_path=str(dest),
        status="queued",
        target_emotion=target_emotion,
    )
    db.add(video)
    db.flush()
    _enqueue_video(db, video, priority, target_emotion)
    return video


@router.get("/queue", response_model=List[QueueItemResponse])
def list_queue(status: str | None = None, db: Session = Depends(get_db)):
    query = db.query(ProcessQueue)
    if status:
        query = query.filter(ProcessQueue.status == status)
    return query.order_by(ProcessQueue.priority.desc(), ProcessQueue.id.asc()).all()


@router.post("/queue/run")
def run_queue_now():
    """Run queued items synchronously. In Colab, call this from a background thread/cell."""
    return _run_queue_once()


@router.post("/queue/pause")
def pause_queue(db: Session = Depends(get_db)):
    updated = db.query(ProcessQueue).filter(ProcessQueue.status == "queued").update({"status": "paused"})
    db.commit()
    return {"paused": updated}


@router.post("/queue/resume")
def resume_queue(db: Session = Depends(get_db)):
    updated = db.query(ProcessQueue).filter(ProcessQueue.status == "paused").update({"status": "queued"})
    db.commit()
    return {"resumed": updated}
