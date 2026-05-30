"""
Emotion Data Studio — Pipeline Worker (QThread)
=================================================
Runs the full AI pipeline in a background thread,
emitting signals for real-time UI updates.
"""

import traceback
from PySide6.QtCore import QThread, Signal


class PipelineWorker(QThread):
    """
    Background worker for running the AI pipeline.
    Communicates with UI via Qt Signals.
    """

    # Signals
    progress_updated = Signal(str, int, int)    # stage_name, current, total
    log_message = Signal(str)                    # log text
    stage_completed = Signal(str)                # stage_name
    pipeline_finished = Signal(dict)             # result summary
    error_occurred = Signal(str)                 # error message

    def __init__(self, video_url: str = None, video_path: str = None,
                 movie_name: str = "Unknown", video_id: str = None):
        super().__init__()
        self.video_url = video_url
        self.video_path = video_path
        self.movie_name = movie_name
        self.video_id = video_id
        self._is_cancelled = False

    def run(self):
        """
        Execute the full pipeline in background thread.
        IMPORTANT: Do NOT access any UI widgets here — only emit signals.
        """
        try:
            self.log_message.emit("[START] Pipeline started for processing")

            # Import backend modules
            from backend.database.local_db import get_session
            from backend.database.models import Video
            from backend.services.pipeline_orchestrator import PipelineOrchestrator
            import uuid

            session = get_session()

            try:
                # Step 1: Create or get Video record
                if self.video_id:
                    # Resume existing video
                    video = session.query(Video).filter(Video.id == self.video_id).first()
                    if not video:
                        self.error_occurred.emit(f"Video ID {self.video_id} not found")
                        return
                    self.log_message.emit(f"[INFO] Resuming video: {video.title}")
                else:
                    # Create new video record
                    video_id = str(uuid.uuid4())
                    video = Video(
                        id=video_id,
                        title=self.movie_name or "Untitled",
                        movie_name=self.movie_name,
                        source_url=self.video_url,
                        file_path=self.video_path,
                        status="pending"
                    )
                    session.add(video)
                    session.commit()
                    self.video_id = video_id
                    self.log_message.emit(f"[INFO] Created video record: {video_id}")

                if self._is_cancelled:
                    self.log_message.emit("[CANCELLED] Pipeline cancelled by user")
                    return

                # Step 2: Run the pipeline
                self.log_message.emit("[INFO] Starting AI pipeline orchestrator...")
                orchestrator = PipelineOrchestrator()
                orchestrator.run_pipeline(
                    video_id=self.video_id,
                    db=session
                )

                if self._is_cancelled:
                    self.log_message.emit("[CANCELLED] Pipeline cancelled by user")
                    return

                # Step 3: Report results
                video = session.query(Video).filter(Video.id == self.video_id).first()
                result = {
                    "status": video.status if video else "unknown",
                    "video_id": self.video_id,
                    "total_clips": video.total_clips if video else 0,
                    "approved_clips": video.approved_clips if video else 0,
                }

                if video and video.status == "completed":
                    self.log_message.emit(
                        f"[SUCCESS] Pipeline completed: {result['total_clips']} clips, "
                        f"{result['approved_clips']} auto-approved"
                    )
                    self.pipeline_finished.emit(result)
                else:
                    self.log_message.emit(f"[WARNING] Pipeline ended with status: {video.status if video else 'unknown'}")
                    self.pipeline_finished.emit(result)

            except Exception as e:
                self.log_message.emit(f"[ERROR] Pipeline failed: {str(e)}")
                self.log_message.emit(f"[TRACE] {traceback.format_exc()}")
                self.error_occurred.emit(str(e))
            finally:
                session.close()

        except Exception as e:
            self.log_message.emit(f"[FATAL] Could not start pipeline: {str(e)}")
            self.error_occurred.emit(str(e))

    def cancel(self):
        """Request pipeline cancellation"""
        self._is_cancelled = True
        self.log_message.emit("[INFO] Cancellation requested...")
