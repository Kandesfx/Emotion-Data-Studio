"""
Emotion Data Studio — Processing Monitor Page
===============================================
Real-time pipeline monitoring with:
  - Overall progress bar
  - Per-stage progress indicators
  - Live log output
  - System resource monitor (GPU/RAM)
  - Pause/Cancel controls
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFrame, QProgressBar, QPlainTextEdit, QScrollArea,
    QSizePolicy, QSpacerItem
)
from PySide6.QtCore import Qt, Signal, Slot, QTimer
from PySide6.QtGui import QFont, QTextCursor

from ui.styles.theme import Colors, Spacing


class StageProgressWidget(QFrame):
    """Single pipeline stage progress indicator"""

    def __init__(self, stage_name: str, stage_icon: str, parent=None):
        super().__init__(parent)
        self.stage_name = stage_name
        self.setObjectName("card")

        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(12)

        # Icon
        icon_label = QLabel(stage_icon)
        icon_label.setFixedWidth(24)
        layout.addWidget(icon_label)

        # Name
        name_label = QLabel(stage_name)
        name_label.setMinimumWidth(150)
        name_label.setObjectName("statLabel")
        layout.addWidget(name_label)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(100)
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar, stretch=1)

        # Percentage text
        self.pct_label = QLabel("0%")
        self.pct_label.setFixedWidth(48)
        self.pct_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        layout.addWidget(self.pct_label)

        # Status icon
        self.status_label = QLabel("⌛")
        self.status_label.setFixedWidth(24)
        layout.addWidget(self.status_label)

    def set_progress(self, current: int, total: int):
        """Update progress"""
        if total > 0:
            pct = int(current / total * 100)
        else:
            pct = 0
        self.progress_bar.setValue(pct)
        self.pct_label.setText(f"{pct}%")

        if pct >= 100:
            self.status_label.setText("✅")
            self.progress_bar.setObjectName("progressSuccess")
            self.progress_bar.setStyle(self.progress_bar.style())  # Force style refresh
        elif pct > 0:
            self.status_label.setText("⏳")

    def reset(self):
        """Reset stage to initial state"""
        self.progress_bar.setValue(0)
        self.pct_label.setText("0%")
        self.status_label.setText("⌛")


class ProcessingPage(QWidget):
    """Processing Monitor page"""

    processing_completed = Signal(str)  # Emits video_id

    # Pipeline stages
    STAGES = [
        ("download", "📥", "Download Video"),
        ("scene_split", "✂️", "Scene Split"),
        ("face_detect", "👤", "Face Detection"),
        ("audio_extract", "🔊", "Audio Extract"),
        ("transcribe", "📝", "Transcription (Whisper)"),
        ("emotion_label", "🎭", "AI Emotion Labeling"),
        ("quality_score", "⭐", "Quality Scoring"),
    ]

    def __init__(self, parent=None):
        super().__init__(parent)
        self._stage_widgets: dict[str, StageProgressWidget] = {}
        self._is_processing = False
        self._elapsed_seconds = 0
        self._setup_ui()

    def _setup_ui(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)

        scroll_content = QWidget()
        self.main_layout = QVBoxLayout(scroll_content)
        self.main_layout.setContentsMargins(32, 24, 32, 24)
        self.main_layout.setSpacing(20)

        page_layout = QVBoxLayout(self)
        page_layout.setContentsMargins(0, 0, 0, 0)
        page_layout.addWidget(scroll)
        scroll.setWidget(scroll_content)

        # --- Header ---
        header = QVBoxLayout()
        header.setSpacing(4)

        self.title_label = QLabel("⚙️ Processing Monitor")
        self.title_label.setObjectName("pageTitle")
        header.addWidget(self.title_label)

        self.subtitle_label = QLabel("Chưa có video đang xử lý")
        self.subtitle_label.setObjectName("pageSubtitle")
        header.addWidget(self.subtitle_label)

        self.main_layout.addLayout(header)

        # --- Overall Progress ---
        overall_card = QFrame()
        overall_card.setObjectName("cardElevated")
        overall_layout = QVBoxLayout(overall_card)
        overall_layout.setSpacing(8)

        progress_header = QHBoxLayout()
        progress_title = QLabel("Overall Progress")
        progress_title.setObjectName("sectionTitle")
        progress_header.addWidget(progress_title)

        self.elapsed_label = QLabel("⏱️ 00:00")
        self.elapsed_label.setObjectName("mutedText")
        progress_header.addWidget(self.elapsed_label)

        overall_layout.addLayout(progress_header)

        self.overall_progress = QProgressBar()
        self.overall_progress.setObjectName("progressLarge")
        self.overall_progress.setMinimum(0)
        self.overall_progress.setMaximum(100)
        self.overall_progress.setValue(0)
        overall_layout.addWidget(self.overall_progress)

        # Resource usage
        resource_row = QHBoxLayout()
        self.gpu_usage_label = QLabel("🖥️ GPU: 0%")
        self.gpu_usage_label.setObjectName("mutedText")
        resource_row.addWidget(self.gpu_usage_label)

        self.ram_usage_label = QLabel("💾 RAM: 0 / 0 GB")
        self.ram_usage_label.setObjectName("mutedText")
        resource_row.addWidget(self.ram_usage_label)
        resource_row.addStretch()
        overall_layout.addLayout(resource_row)

        self.main_layout.addWidget(overall_card)

        # --- Pipeline Stages ---
        stages_title = QLabel("📋 Pipeline Stages")
        stages_title.setObjectName("sectionTitle")
        self.main_layout.addWidget(stages_title)

        for stage_key, stage_icon, stage_name in self.STAGES:
            widget = StageProgressWidget(stage_name, stage_icon)
            self._stage_widgets[stage_key] = widget
            self.main_layout.addWidget(widget)

        # --- Controls ---
        controls_row = QHBoxLayout()
        controls_row.setSpacing(12)

        self.pause_btn = QPushButton("⏸ Pause")
        self.pause_btn.setEnabled(False)
        self.pause_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        controls_row.addWidget(self.pause_btn)

        self.cancel_btn = QPushButton("⏹ Cancel")
        self.cancel_btn.setObjectName("dangerBtn")
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.cancel_btn.clicked.connect(self._on_cancel)
        controls_row.addWidget(self.cancel_btn)

        controls_row.addStretch()

        self.view_results_btn = QPushButton("📊 View Partial Results")
        self.view_results_btn.setEnabled(False)
        self.view_results_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        controls_row.addWidget(self.view_results_btn)

        self.main_layout.addLayout(controls_row)

        # --- Live Log ---
        log_title = QLabel("📋 Live Log")
        log_title.setObjectName("sectionTitle")
        self.main_layout.addWidget(log_title)

        self.log_viewer = QPlainTextEdit()
        self.log_viewer.setObjectName("logViewer")
        self.log_viewer.setReadOnly(True)
        self.log_viewer.setMinimumHeight(200)
        self.log_viewer.setMaximumHeight(300)
        self.main_layout.addWidget(self.log_viewer)

        self.main_layout.addStretch()

        # --- Timer for elapsed time ---
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._update_elapsed)

    # ================================================================
    # PUBLIC API
    # ================================================================

    def start_monitoring(self, video_id: str):
        """Called when pipeline starts — reset and begin monitoring"""
        self._is_processing = True
        self._elapsed_seconds = 0
        self.subtitle_label.setText(f"Đang xử lý video: {video_id[:16]}...")
        self.overall_progress.setValue(0)
        self.log_viewer.clear()
        self.log_viewer.appendPlainText(f"[START] Pipeline started for {video_id}")

        # Reset all stages
        for widget in self._stage_widgets.values():
            widget.reset()

        # Enable controls
        self.pause_btn.setEnabled(True)
        self.cancel_btn.setEnabled(True)

        # Start timer
        self._timer.start(1000)

    @Slot(str, int, int)
    def on_progress_updated(self, stage_name: str, current: int, total: int):
        """Update a specific stage's progress"""
        if stage_name in self._stage_widgets:
            self._stage_widgets[stage_name].set_progress(current, total)

        # Recalculate overall progress
        self._update_overall_progress()

    @Slot(str)
    def on_log_message(self, message: str):
        """Append log message"""
        self.log_viewer.appendPlainText(message)
        # Auto-scroll to bottom
        cursor = self.log_viewer.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        self.log_viewer.setTextCursor(cursor)

    @Slot(str)
    def on_stage_completed(self, stage_name: str):
        """Mark a stage as completed"""
        if stage_name in self._stage_widgets:
            self._stage_widgets[stage_name].set_progress(100, 100)

    @Slot(dict)
    def on_pipeline_finished(self, result: dict):
        """Handle pipeline completion"""
        self._is_processing = False
        self._timer.stop()

        self.overall_progress.setValue(100)
        self.subtitle_label.setText("✅ Xử lý hoàn tất!")
        self.log_viewer.appendPlainText("[DONE] Pipeline completed successfully")

        self.pause_btn.setEnabled(False)
        self.cancel_btn.setEnabled(False)
        self.view_results_btn.setEnabled(True)

        video_id = result.get("video_id", "")
        self.processing_completed.emit(video_id)

    @Slot(str)
    def on_error(self, error_msg: str):
        """Handle pipeline error"""
        self._is_processing = False
        self._timer.stop()

        self.subtitle_label.setText("❌ Lỗi xử lý!")
        self.subtitle_label.setObjectName("errorText")
        self.log_viewer.appendPlainText(f"[ERROR] {error_msg}")

        self.pause_btn.setEnabled(False)
        self.cancel_btn.setEnabled(False)

    # ================================================================
    # INTERNAL
    # ================================================================

    def _update_elapsed(self):
        """Update elapsed time display"""
        self._elapsed_seconds += 1
        mins = self._elapsed_seconds // 60
        secs = self._elapsed_seconds % 60
        self.elapsed_label.setText(f"⏱️ {mins:02d}:{secs:02d}")

    def _update_overall_progress(self):
        """Calculate and update overall progress from all stages"""
        total_pct = 0
        for widget in self._stage_widgets.values():
            total_pct += widget.progress_bar.value()
        overall = total_pct // len(self._stage_widgets) if self._stage_widgets else 0
        self.overall_progress.setValue(overall)

    def _on_cancel(self):
        """Cancel the current pipeline"""
        self._is_processing = False
        self._timer.stop()
        self.subtitle_label.setText("⏹ Pipeline cancelled")
        self.log_viewer.appendPlainText("[CANCELLED] Pipeline cancelled by user")
        self.pause_btn.setEnabled(False)
        self.cancel_btn.setEnabled(False)

    def refresh_data(self):
        """Refresh page (called when switching to this page)"""
        pass
