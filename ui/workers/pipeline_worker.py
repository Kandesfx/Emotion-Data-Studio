"""
Emotion Data Studio - robust pipeline worker.

Runs backend processing in a QThread and never touches Qt widgets directly.
Adds preflight validation, structured error reporting and safe cancellation flags.
"""

from __future__ import annotations

import os
import shutil
import traceback
from pathlib import Path
from urllib.parse import urlparse

from PySide6.QtCore import QThread, Signal


class PipelineWorker(QThread):
    """Background worker for running the AI pipeline."""

    progress_updated = Signal(str, int, int)
    log_message = Signal(str)
    stage_completed = Signal(str)
    pipeline_finished = Signal(dict)
    error_occurred = Signal(str)

    def __init__(self, video_url: str, movie_name: str = "Unknown"):
        super().__init__()
        self.video_url = (video_url or "").strip()
        self.movie_name = (movie_name or "Unknown").strip() or "Unknown"
        self._is_cancelled = False

    def run(self):
        """Execute the full pipeline in a background thread."""
        db = None
        video = None
        try:
            self._preflight()
            self.log_message.emit(f"[INFO] Source: {self.video_url}")
            self.log_message.emit(f"[INFO] Movie: {self.movie_name}")

            from backend.database.local_db import get_session
            from backend.database.models import Video
            from backend.services.pipeline_orchestrator import PipelineOrchestrator

            db = get_session()
            video = Video(
                title=self._derive_title(),
                movie_name=self.movie_name,
                source_url=None if self._is_local_file() else self.video_url,
                file_path=self.video_url if self._is_local_file() else None,
                status="pending",
                processing_mode="auto",
            )
            db.add(video)
            db.commit()
            db.refresh(video)

            self.log_message.emit(f"[INFO] Created video job: {video.id}")
            self.progress_updated.emit("queued", 0, 100)

            orchestrator = PipelineOrchestrator()
            orchestrator.run_pipeline(video.id, db, progress_callback=self._progress_callback)

            if self._is_cancelled:
                video.status = "cancelled"
                db.commit()
                self.log_message.emit("[CANCELLED] Pipeline cancelled by user")
                return

            db.refresh(video)
            result = {
                "status": video.status,
                "video_id": video.id,
                "title": video.title,
                "total_clips": video.total_clips,
                "approved_clips": video.approved_clips,
            }
            self.log_message.emit("[SUCCESS] Pipeline completed")
            self.pipeline_finished.emit(result)

        except Exception as exc:
            details = traceback.format_exc()
            safe_message = self._format_error(exc)
            self.log_message.emit(f"[ERROR] {safe_message}")
            self.log_message.emit(details)
            if db is not None and video is not None:
                try:
                    video.status = "error"
                    db.commit()
                except Exception:
                    db.rollback()
            self.error_occurred.emit(safe_message)
        finally:
            if db is not None:
                db.close()

    def cancel(self):
        """Request cooperative cancellation."""
        self._is_cancelled = True
        self.log_message.emit("[INFO] Cancellation requested...")

    def _progress_callback(self, stage: str, current: int, total: int, message: str = ""):
        if stage == "check_cancel":
            return not self._is_cancelled
        if message:
            self.log_message.emit(message)
        safe_total = int(total or 100)
        safe_current = int(current or 0)
        self.progress_updated.emit(stage, safe_current, safe_total)
        if safe_total and safe_current >= safe_total:
            self.stage_completed.emit(stage)
        return not self._is_cancelled

    def _preflight(self):
        if not self.video_url:
            raise ValueError("Please enter a YouTube URL or choose a local video file.")

        if self._is_local_file():
            path = Path(self.video_url)
            if not path.exists():
                raise FileNotFoundError(f"Video file was not found: {path}")
            if path.suffix.lower() not in {".mp4", ".mkv", ".avi", ".webm", ".mov"}:
                raise ValueError("Unsupported video format. Use mp4/mkv/avi/webm/mov.")
        else:
            parsed = urlparse(self.video_url)
            if parsed.scheme not in {"http", "https"} or not parsed.netloc:
                raise ValueError("Invalid URL. Enter an http/https URL or select a video file.")

        from backend.config import settings
        settings.DATA_DIR.mkdir(parents=True, exist_ok=True)
        for subdir in ("videos", "clips", "frames", "audio", "exports"):
            (settings.DATA_DIR / subdir).mkdir(parents=True, exist_ok=True)

        if shutil.which(settings.FFMPEG_PATH) is None:
            raise RuntimeError(
                "FFmpeg was not found. Install FFmpeg and add it to PATH, "
                "or configure FFMPEG_PATH before running the pipeline."
            )

    def _is_local_file(self) -> bool:
        parsed = urlparse(self.video_url)
        return parsed.scheme in {"", "file"} or os.path.exists(self.video_url)

    def _derive_title(self) -> str:
        if self._is_local_file():
            return Path(self.video_url).stem or self.movie_name
        return self.movie_name if self.movie_name != "Unknown" else "YouTube Video"

    @staticmethod
    def _format_error(exc: Exception) -> str:
        text = str(exc).strip() or exc.__class__.__name__
        return text[:1200]
