"""Desktop sync worker — runs GCS/cloud sync in a background QThread."""

from __future__ import annotations

from PySide6.QtCore import QThread, Signal


class SyncWorker(QThread):
    """Background worker for cloud sync operations."""

    progress_updated = Signal(str, int, int)
    log_message = Signal(str)
    sync_finished = Signal(dict)
    error_occurred = Signal(str)

    def __init__(self, sync_type: str = "full", sync_videos: bool = False):
        super().__init__()
        self.sync_type = sync_type
        self.sync_videos = sync_videos

    def run(self):
        self.log_message.emit(f"[Sync] Starting {self.sync_type} sync...")
        try:
            from backend.cloud.sync_manager import SyncManager
            manager = SyncManager()

            if not manager.is_available:
                self.error_occurred.emit(
                    "Cloud sync not available. "
                    "Set GOOGLE_APPLICATION_CREDENTIALS and GCS_BUCKET_NAME in Settings."
                )
                return

            def on_progress(current, total, message):
                self.progress_updated.emit(self.sync_type, current, total)
                if message:
                    self.log_message.emit(f"[Sync] {message}")

            if self.sync_type == "full":
                report = manager.full_sync(sync_videos=self.sync_videos, on_progress=on_progress)
            elif self.sync_type == "metadata":
                report = manager.sync_metadata(direction="bidirectional", on_progress=on_progress)
            elif self.sync_type == "upload":
                report = manager.sync_metadata(direction="upload", on_progress=on_progress)
            elif self.sync_type == "download":
                report = manager.sync_metadata(direction="download", on_progress=on_progress)
            elif self.sync_type == "files":
                report = manager.sync_files(
                    sync_videos=self.sync_videos,
                    on_progress=on_progress,
                )
            else:
                self.error_occurred.emit(f"Unknown sync type: {self.sync_type}")
                return

            errors = report.get("errors", [])
            if errors:
                self.log_message.emit(f"[Sync] Completed with {len(errors)} error(s)")
                for err in errors[:5]:
                    self.log_message.emit(f"  ⚠️  {err}")
            else:
                self.log_message.emit("[Sync] Completed successfully")

            self.sync_finished.emit(report)

        except Exception as exc:
            self.error_occurred.emit(str(exc))
