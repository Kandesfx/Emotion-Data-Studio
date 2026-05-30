"""
Emotion Data Studio — Pipeline Worker (QThread)
=================================================
Runs the full AI pipeline in a background thread,
emitting signals for real-time UI updates.
"""

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

    def __init__(self, video_url: str, movie_name: str = "Unknown"):
        super().__init__()
        self.video_url = video_url
        self.movie_name = movie_name
        self._is_cancelled = False

    def run(self):
        """
        Execute the full pipeline in background thread.
        IMPORTANT: Do NOT access any UI widgets here — only emit signals.
        """
        try:
            self.log_message.emit(f"[INFO] Starting pipeline for: {self.video_url}")
            self.log_message.emit(f"[INFO] Movie name: {self.movie_name}")

            # Import backend services
            from backend.services.pipeline_orchestrator import PipelineOrchestrator

            orchestrator = PipelineOrchestrator()

            # Process video through the pipeline
            result = orchestrator.process_video(
                url=self.video_url,
                movie_name=self.movie_name
            )

            if self._is_cancelled:
                self.log_message.emit("[CANCELLED] Pipeline cancelled by user")
                return

            self.log_message.emit("[SUCCESS] Pipeline completed successfully")
            self.pipeline_finished.emit(result if isinstance(result, dict) else {"status": "completed"})

        except Exception as e:
            self.log_message.emit(f"[ERROR] Pipeline failed: {str(e)}")
            self.error_occurred.emit(str(e))

    def cancel(self):
        """Request pipeline cancellation"""
        self._is_cancelled = True
        self.log_message.emit("[INFO] Cancellation requested...")
