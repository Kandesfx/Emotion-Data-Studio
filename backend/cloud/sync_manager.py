"""
Emotion Data Studio — Smart Sync Manager
==========================================
Bidirectional sync between local SQLite and Cloud SQL + GCS.

Sync flow:
  1. Compare local vs cloud timestamps
  2. Upload newer local records → Cloud SQL
  3. Download newer cloud records → local SQLite
  4. Upload processed files → GCS
  5. Log all sync operations
"""

import os
import logging
from datetime import datetime
from typing import Optional, List, Dict

logger = logging.getLogger(__name__)


class SyncManager:
    """
    Smart bidirectional sync manager.
    Handles metadata sync (SQLite ↔ Cloud SQL) and file sync (local ↔ GCS).
    """

    def __init__(self):
        self._gcs = None
        self._cloud_sql = None
        self._initialized = False

    def _init_clients(self):
        """Lazy-init cloud clients"""
        if self._initialized:
            return

        try:
            from backend.cloud.gcs_client import GCSClient
            from backend.cloud.cloudsql_client import CloudSQLClient

            self._gcs = GCSClient()
            self._cloud_sql = CloudSQLClient()
            self._initialized = True
        except Exception as e:
            logger.error(f"Failed to init cloud clients: {e}")
            raise

    @property
    def is_available(self) -> bool:
        """Check if cloud sync is available"""
        try:
            self._init_clients()
            return self._gcs.is_configured and self._cloud_sql.is_configured
        except Exception:
            return False

    # ================================================================
    # METADATA SYNC (SQLite ↔ Cloud SQL)
    # ================================================================

    def sync_metadata(self, direction: str = "bidirectional",
                      on_progress=None) -> Dict:
        """
        Sync metadata between local SQLite and Cloud SQL.
        
        Args:
            direction: "upload", "download", or "bidirectional"
            on_progress: Callback(current, total, message)
            
        Returns:
            Sync report dict
        """
        self._init_clients()

        report = {
            "uploaded_videos": 0,
            "uploaded_clips": 0,
            "uploaded_labels": 0,
            "downloaded_videos": 0,
            "downloaded_clips": 0,
            "downloaded_labels": 0,
            "errors": [],
            "started_at": datetime.utcnow().isoformat(),
        }

        try:
            if direction in ("upload", "bidirectional"):
                self._sync_upload_metadata(report, on_progress)

            if direction in ("download", "bidirectional"):
                self._sync_download_metadata(report, on_progress)

        except Exception as e:
            report["errors"].append(str(e))
            logger.error(f"Metadata sync error: {e}")

        report["completed_at"] = datetime.utcnow().isoformat()
        self._log_sync("metadata", report)
        return report

    def _sync_upload_metadata(self, report: Dict, on_progress=None):
        """Upload local metadata to cloud"""
        from backend.database.local_db import get_session
        from backend.database.models import Video, Clip, Label

        local_session = get_session()
        try:
            # Sync Videos
            videos = local_session.query(Video).all()
            total = len(videos)

            with self._cloud_sql.get_session() as cloud_session:
                for i, video in enumerate(videos):
                    try:
                        # Check if exists in cloud
                        existing = cloud_session.query(Video).filter(
                            Video.id == video.id
                        ).first()

                        if existing:
                            # Update if local is newer
                            if video.updated_at and existing.updated_at:
                                if video.updated_at > existing.updated_at:
                                    self._copy_record(video, existing)
                                    report["uploaded_videos"] += 1
                        else:
                            # Insert new
                            new_video = Video()
                            self._copy_record(video, new_video)
                            new_video.id = video.id
                            cloud_session.add(new_video)
                            report["uploaded_videos"] += 1

                    except Exception as e:
                        report["errors"].append(f"Video {video.id}: {e}")

                    if on_progress:
                        on_progress(i + 1, total, f"Uploading videos: {i+1}/{total}")

                # Sync Clips
                clips = local_session.query(Clip).all()
                total_clips = len(clips)

                for i, clip in enumerate(clips):
                    try:
                        existing = cloud_session.query(Clip).filter(
                            Clip.id == clip.id
                        ).first()

                        if existing:
                            if clip.updated_at and existing.updated_at:
                                if clip.updated_at > existing.updated_at:
                                    self._copy_record(clip, existing)
                                    report["uploaded_clips"] += 1
                        else:
                            new_clip = Clip()
                            self._copy_record(clip, new_clip)
                            new_clip.id = clip.id
                            cloud_session.add(new_clip)
                            report["uploaded_clips"] += 1

                    except Exception as e:
                        report["errors"].append(f"Clip {clip.id}: {e}")

                    if on_progress:
                        on_progress(i + 1, total_clips, f"Uploading clips: {i+1}/{total_clips}")

                # Sync Labels
                labels = local_session.query(Label).all()
                for label in labels:
                    try:
                        existing = cloud_session.query(Label).filter(
                            Label.clip_id == label.clip_id
                        ).first()

                        if existing:
                            self._copy_record(label, existing)
                        else:
                            new_label = Label()
                            self._copy_record(label, new_label)
                            cloud_session.add(new_label)
                        report["uploaded_labels"] += 1
                    except Exception as e:
                        report["errors"].append(f"Label {label.clip_id}: {e}")

        finally:
            local_session.close()

    def _sync_download_metadata(self, report: Dict, on_progress=None):
        """Download cloud metadata to local"""
        from backend.database.local_db import get_session
        from backend.database.models import Video, Clip, Label

        local_session = get_session()
        try:
            with self._cloud_sql.get_session() as cloud_session:
                # Download Videos
                cloud_videos = cloud_session.query(Video).all()

                for video in cloud_videos:
                    try:
                        existing = local_session.query(Video).filter(
                            Video.id == video.id
                        ).first()

                        if existing:
                            if video.updated_at and existing.updated_at:
                                if video.updated_at > existing.updated_at:
                                    self._copy_record(video, existing)
                                    report["downloaded_videos"] += 1
                        else:
                            new_video = Video()
                            self._copy_record(video, new_video)
                            new_video.id = video.id
                            local_session.add(new_video)
                            report["downloaded_videos"] += 1

                    except Exception as e:
                        report["errors"].append(f"Download Video {video.id}: {e}")

                # Download Clips
                cloud_clips = cloud_session.query(Clip).all()

                for clip in cloud_clips:
                    try:
                        existing = local_session.query(Clip).filter(
                            Clip.id == clip.id
                        ).first()

                        if existing:
                            if clip.updated_at and existing.updated_at:
                                if clip.updated_at > existing.updated_at:
                                    self._copy_record(clip, existing)
                                    report["downloaded_clips"] += 1
                        else:
                            new_clip = Clip()
                            self._copy_record(clip, new_clip)
                            new_clip.id = clip.id
                            local_session.add(new_clip)
                            report["downloaded_clips"] += 1

                    except Exception as e:
                        report["errors"].append(f"Download Clip {clip.id}: {e}")

            local_session.commit()

        except Exception as e:
            local_session.rollback()
            report["errors"].append(f"Download sync failed: {e}")
        finally:
            local_session.close()

    # ================================================================
    # FILE SYNC (Local ↔ GCS)
    # ================================================================

    def sync_files(self, sync_videos: bool = False,
                   sync_clips: bool = True,
                   sync_audio: bool = True,
                   on_progress=None) -> Dict:
        """
        Sync processed files to Google Cloud Storage.
        
        Args:
            sync_videos: Upload raw video files (large!)
            sync_clips: Upload processed clip files
            sync_audio: Upload extracted audio files
            on_progress: Callback(current, total, message)
            
        Returns:
            Sync report dict
        """
        self._init_clients()
        from backend.config import settings

        report = {
            "uploaded_files": 0,
            "total_size_mb": 0,
            "errors": [],
        }

        data_dir = settings.DATA_DIR

        try:
            # Upload clips
            if sync_clips:
                clips_dir = data_dir / "clips"
                if clips_dir.exists():
                    files = list(clips_dir.rglob("*"))
                    files = [f for f in files if f.is_file()]
                    total = len(files)

                    for i, f in enumerate(files):
                        try:
                            gcs_path = f"data/clips/{f.relative_to(clips_dir).as_posix()}"
                            if not self._gcs.file_exists(gcs_path):
                                self._gcs.upload_file(str(f), gcs_path)
                                report["uploaded_files"] += 1
                                report["total_size_mb"] += f.stat().st_size / (1024 * 1024)
                        except Exception as e:
                            report["errors"].append(f"Upload {f.name}: {e}")

                        if on_progress:
                            on_progress(i + 1, total, f"Uploading clips: {i+1}/{total}")

            # Upload audio
            if sync_audio:
                audio_dir = data_dir / "audio"
                if audio_dir.exists():
                    files = list(audio_dir.rglob("*"))
                    files = [f for f in files if f.is_file()]

                    for i, f in enumerate(files):
                        try:
                            gcs_path = f"data/audio/{f.relative_to(audio_dir).as_posix()}"
                            if not self._gcs.file_exists(gcs_path):
                                self._gcs.upload_file(str(f), gcs_path)
                                report["uploaded_files"] += 1
                                report["total_size_mb"] += f.stat().st_size / (1024 * 1024)
                        except Exception as e:
                            report["errors"].append(f"Upload {f.name}: {e}")

            # Upload raw videos (optional — large files!)
            if sync_videos:
                videos_dir = data_dir / "videos"
                if videos_dir.exists():
                    files = list(videos_dir.rglob("*"))
                    files = [f for f in files if f.is_file()]

                    for i, f in enumerate(files):
                        try:
                            gcs_path = f"data/videos/{f.relative_to(videos_dir).as_posix()}"
                            if not self._gcs.file_exists(gcs_path):
                                self._gcs.upload_file(str(f), gcs_path)
                                report["uploaded_files"] += 1
                                report["total_size_mb"] += f.stat().st_size / (1024 * 1024)
                        except Exception as e:
                            report["errors"].append(f"Upload {f.name}: {e}")

        except Exception as e:
            report["errors"].append(f"File sync error: {e}")

        self._log_sync("files", report)
        return report

    # ================================================================
    # FULL SYNC
    # ================================================================

    def full_sync(self, sync_videos: bool = False, on_progress=None) -> Dict:
        """
        Run full bidirectional sync (metadata + files).
        
        Returns:
            Combined sync report
        """
        report = {
            "metadata": {},
            "files": {},
            "status": "success",
        }

        try:
            # Step 1: Metadata sync
            if on_progress:
                on_progress(0, 100, "Syncing metadata...")

            report["metadata"] = self.sync_metadata(
                direction="bidirectional",
                on_progress=on_progress,
            )

            # Step 2: File sync
            if on_progress:
                on_progress(50, 100, "Syncing files...")

            report["files"] = self.sync_files(
                sync_videos=sync_videos,
                on_progress=on_progress,
            )

            if on_progress:
                on_progress(100, 100, "Sync complete!")

        except Exception as e:
            report["status"] = "error"
            report["error"] = str(e)

        return report

    # ================================================================
    # HELPERS
    # ================================================================

    @staticmethod
    def _copy_record(source, target):
        """Copy SQLAlchemy model attributes from source to target"""
        from sqlalchemy import inspect
        mapper = inspect(type(source))
        for column in mapper.columns:
            key = column.key
            if key != "id":  # Don't overwrite primary key
                value = getattr(source, key, None)
                if value is not None:
                    try:
                        setattr(target, key, value)
                    except AttributeError:
                        pass  # Skip properties and read-only attributes

    def _log_sync(self, sync_type: str, report: Dict):
        """Log sync operation to local database"""
        try:
            from backend.database.local_db import get_session
            from backend.database.models import SyncLog

            session = get_session()
            try:
                import json
                log = SyncLog(
                    direction="bidirectional",
                    entity_type=sync_type,
                    entity_id=None,
                    status="success" if not report.get("errors") else "partial",
                    error_message=json.dumps(report.get("errors", [])),
                    synced_at=datetime.utcnow(),
                )
                session.add(log)
                session.commit()
            finally:
                session.close()
        except Exception as e:
            logger.error(f"Failed to log sync: {e}")

    def get_sync_status(self) -> Dict:
        """Get last sync status from database"""
        try:
            from backend.database.local_db import get_session
            from backend.database.models import SyncLog

            session = get_session()
            try:
                last_sync = session.query(SyncLog).order_by(
                    SyncLog.synced_at.desc()
                ).first()

                if last_sync:
                    return {
                        "last_sync": last_sync.synced_at.isoformat(),
                        "status": last_sync.status,
                        "type": last_sync.entity_type,
                    }
                return {"last_sync": None, "status": "never"}
            finally:
                session.close()
        except Exception:
            return {"last_sync": None, "status": "error"}
