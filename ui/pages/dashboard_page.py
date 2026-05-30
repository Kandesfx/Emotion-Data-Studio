"""
Emotion Data Studio — Dashboard Page
======================================
Main dashboard with:
  - Statistics cards (total clips, approved, pending, rejected)
  - Import video form (URL input)
  - Video list table
  - Emotion distribution chart
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QLineEdit, QPushButton, QFrame,
    QTableWidget, QTableWidgetItem, QHeaderView,
    QScrollArea, QSizePolicy, QSpacerItem, QMessageBox,
    QAbstractItemView
)
from PySide6.QtCore import Qt, Signal, Slot, QTimer
from PySide6.QtGui import QFont

from ui.styles.theme import Colors, Spacing, Typography, EMOTION_MAP


class StatsCard(QFrame):
    """Single statistics card widget"""

    def __init__(self, icon: str, label: str, value: str = "0",
                 color: str = Colors.TEXT_PRIMARY, parent=None):
        super().__init__(parent)
        self.setObjectName("statsCard")
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        # Top row: icon + label
        top_row = QHBoxLayout()
        icon_label = QLabel(icon)
        icon_label.setObjectName("statIcon")
        top_row.addWidget(icon_label)

        name_label = QLabel(label)
        name_label.setObjectName("statLabel")
        top_row.addWidget(name_label)
        top_row.addStretch()
        layout.addLayout(top_row)

        # Value
        self.value_label = QLabel(value)
        self.value_label.setObjectName("statValue")
        self.value_label.setStyleSheet(f"color: {color};")
        layout.addWidget(self.value_label)

    def set_value(self, value: str):
        self.value_label.setText(value)


class DashboardPage(QWidget):
    """Dashboard & Import page"""

    processing_started = Signal(str)  # Emits video_id

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()
        # Load data after UI is ready
        QTimer.singleShot(100, self.refresh_data)

    def _setup_ui(self):
        # Main scroll area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)

        scroll_content = QWidget()
        self.main_layout = QVBoxLayout(scroll_content)
        self.main_layout.setContentsMargins(32, 24, 32, 24)
        self.main_layout.setSpacing(24)

        # Page layout
        page_layout = QVBoxLayout(self)
        page_layout.setContentsMargins(0, 0, 0, 0)
        page_layout.addWidget(scroll)
        scroll.setWidget(scroll_content)

        # --- Page Header ---
        self._build_header()

        # --- Stats Cards ---
        self._build_stats_section()

        # --- Import Video Section ---
        self._build_import_section()

        # --- Video List Table ---
        self._build_video_table()

        # --- Spacer ---
        self.main_layout.addStretch()

    def _build_header(self):
        """Page title and subtitle"""
        header_layout = QVBoxLayout()
        header_layout.setSpacing(4)

        title = QLabel("📊 Dashboard")
        title.setObjectName("pageTitle")
        header_layout.addWidget(title)

        subtitle = QLabel("Tổng quan dữ liệu và import video mới")
        subtitle.setObjectName("pageSubtitle")
        header_layout.addWidget(subtitle)

        self.main_layout.addLayout(header_layout)

    def _build_stats_section(self):
        """Statistics cards grid"""
        stats_layout = QHBoxLayout()
        stats_layout.setSpacing(16)

        self.stat_total = StatsCard("📁", "Total Clips", "0", Colors.TEXT_PRIMARY)
        self.stat_approved = StatsCard("✅", "Approved", "0", Colors.SUCCESS)
        self.stat_pending = StatsCard("⏳", "Pending", "0", Colors.WARNING)
        self.stat_rejected = StatsCard("❌", "Rejected", "0", Colors.ERROR)
        self.stat_videos = StatsCard("🎬", "Videos", "0", Colors.ACCENT_LIGHT)

        stats_layout.addWidget(self.stat_total)
        stats_layout.addWidget(self.stat_approved)
        stats_layout.addWidget(self.stat_pending)
        stats_layout.addWidget(self.stat_rejected)
        stats_layout.addWidget(self.stat_videos)

        self.main_layout.addLayout(stats_layout)

    def _build_import_section(self):
        """Video import form"""
        import_card = QFrame()
        import_card.setObjectName("card")

        card_layout = QVBoxLayout(import_card)
        card_layout.setSpacing(12)

        # Section title
        section_title = QLabel("📎 Import Video")
        section_title.setObjectName("sectionTitle")
        card_layout.addWidget(section_title)

        # URL Input row
        url_row = QHBoxLayout()
        url_row.setSpacing(8)

        self.url_input = QLineEdit()
        self.url_input.setPlaceholderText("Nhập URL YouTube (vd: https://youtube.com/watch?v=...)")
        self.url_input.setMinimumHeight(38)
        self.url_input.returnPressed.connect(self._on_process_clicked)
        url_row.addWidget(self.url_input, stretch=1)

        self.process_btn = QPushButton("▶  Xử lý")
        self.process_btn.setObjectName("primaryBtn")
        self.process_btn.setMinimumHeight(38)
        self.process_btn.setMinimumWidth(120)
        self.process_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.process_btn.clicked.connect(self._on_process_clicked)
        url_row.addWidget(self.process_btn)

        card_layout.addLayout(url_row)

        # Movie name row
        name_row = QHBoxLayout()
        name_row.setSpacing(8)

        name_label = QLabel("Tên bộ phim:")
        name_label.setObjectName("statLabel")
        name_label.setFixedWidth(100)
        name_row.addWidget(name_label)

        self.movie_name_input = QLineEdit()
        self.movie_name_input.setPlaceholderText("Ví dụ: Về Nhà Đi Con - Ep01")
        name_row.addWidget(self.movie_name_input, stretch=1)

        # Import from file button
        self.import_file_btn = QPushButton("📁 Import File")
        self.import_file_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.import_file_btn.clicked.connect(self._on_import_file)
        name_row.addWidget(self.import_file_btn)

        card_layout.addLayout(name_row)

        self.main_layout.addWidget(import_card)

    def _build_video_table(self):
        """Video list table"""
        table_card = QFrame()
        table_card.setObjectName("card")

        card_layout = QVBoxLayout(table_card)
        card_layout.setSpacing(12)

        # Section header
        header_row = QHBoxLayout()
        section_title = QLabel("🎬 Danh sách Video đã Import")
        section_title.setObjectName("sectionTitle")
        header_row.addWidget(section_title)
        header_row.addStretch()

        refresh_btn = QPushButton("🔄 Refresh")
        refresh_btn.setObjectName("ghostBtn")
        refresh_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        refresh_btn.clicked.connect(self.refresh_data)
        header_row.addWidget(refresh_btn)

        card_layout.addLayout(header_row)

        # Table
        self.video_table = QTableWidget()
        self.video_table.setColumnCount(6)
        self.video_table.setHorizontalHeaderLabels([
            "#", "Tên Video", "Clips", "Approved", "Pending", "Status"
        ])
        self.video_table.horizontalHeader().setStretchLastSection(True)
        self.video_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.video_table.verticalHeader().setVisible(False)
        self.video_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.video_table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.video_table.setMinimumHeight(250)
        self.video_table.setAlternatingRowColors(False)
        self.video_table.setShowGrid(False)

        card_layout.addWidget(self.video_table)

        self.main_layout.addWidget(table_card)

    # ================================================================
    # ACTIONS
    # ================================================================

    @Slot()
    def _on_process_clicked(self):
        """Handle process button click — start pipeline"""
        url = self.url_input.text().strip()
        if not url:
            QMessageBox.warning(self, "Lỗi", "Vui lòng nhập URL video.")
            return

        movie_name = self.movie_name_input.text().strip() or "Unknown"

        # Disable UI during processing
        self.process_btn.setEnabled(False)
        self.process_btn.setText("⏳ Đang xử lý...")
        self.url_input.setEnabled(False)

        # Start pipeline via backend service (in QThread)
        self._start_pipeline(url, movie_name)

    def _start_pipeline(self, url: str, movie_name: str):
        """Start the AI pipeline in a background thread"""
        from ui.workers.pipeline_worker import PipelineWorker

        self._worker = PipelineWorker(url, movie_name)
        self._worker.pipeline_finished.connect(self._on_pipeline_finished)
        self._worker.error_occurred.connect(self._on_pipeline_error)
        self._worker.start()

        # Emit signal so MainWindow can switch to Processing page
        self.processing_started.emit("processing")

    @Slot(dict)
    def _on_pipeline_finished(self, result: dict):
        """Handle pipeline completion"""
        self.process_btn.setEnabled(True)
        self.process_btn.setText("▶  Xử lý")
        self.url_input.setEnabled(True)
        self.url_input.clear()
        self.movie_name_input.clear()

        # Refresh data
        self.refresh_data()

    @Slot(str)
    def _on_pipeline_error(self, error_msg: str):
        """Handle pipeline error"""
        self.process_btn.setEnabled(True)
        self.process_btn.setText("▶  Xử lý")
        self.url_input.setEnabled(True)

        QMessageBox.critical(self, "Pipeline Error", f"Lỗi xử lý:\n{error_msg}")

    @Slot()
    def _on_import_file(self):
        """Import video from local file"""
        from PySide6.QtWidgets import QFileDialog
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Chọn video file",
            "", "Video Files (*.mp4 *.mkv *.avi *.webm *.mov)"
        )
        if file_path:
            self.url_input.setText(file_path)

    # ================================================================
    # DATA
    # ================================================================

    @Slot()
    def refresh_data(self):
        """Refresh dashboard statistics and video list from database"""
        try:
            from backend.database.local_db import get_session
            from backend.database.models import Video, Clip

            session = get_session()
            try:
                # Stats
                total_clips = session.query(Clip).count()
                approved = session.query(Clip).filter(
                    Clip.status.in_(['approved', 'auto_approved'])
                ).count()
                pending = session.query(Clip).filter(
                    Clip.status.in_(['pending', 'needs_review'])
                ).count()
                rejected = session.query(Clip).filter(Clip.status == 'rejected').count()
                total_videos = session.query(Video).count()

                self.stat_total.set_value(f"{total_clips:,}")
                self.stat_approved.set_value(f"{approved:,}")
                self.stat_pending.set_value(f"{pending:,}")
                self.stat_rejected.set_value(f"{rejected:,}")
                self.stat_videos.set_value(f"{total_videos:,}")

                # Video list
                videos = session.query(Video).order_by(Video.created_at.desc()).all()
                self._populate_video_table(videos, session)

            finally:
                session.close()

        except Exception as e:
            # Database not available yet — show zeros
            self.stat_total.set_value("0")
            self.stat_approved.set_value("0")
            self.stat_pending.set_value("0")
            self.stat_rejected.set_value("0")
            self.stat_videos.set_value("0")

    def _populate_video_table(self, videos, session):
        """Fill video table with data"""
        from backend.database.models import Clip

        self.video_table.setRowCount(len(videos))

        for row, video in enumerate(videos):
            # Count clips per video
            total = session.query(Clip).filter(Clip.video_id == video.id).count()
            done = session.query(Clip).filter(
                Clip.video_id == video.id,
                Clip.status.in_(['approved', 'auto_approved'])
            ).count()
            pend = session.query(Clip).filter(
                Clip.video_id == video.id,
                Clip.status.in_(['pending', 'needs_review'])
            ).count()

            # Calculate progress
            progress = f"✅ {int(done/total*100)}%" if total > 0 else "⏳ {status}".format(
                status=video.status or "pending"
            )

            items = [
                str(row + 1),
                video.title or video.movie_name or "Unknown",
                str(total),
                str(done),
                str(pend),
                progress
            ]

            for col, text in enumerate(items):
                item = QTableWidgetItem(text)
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                if col == 0:
                    item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self.video_table.setItem(row, col, item)
