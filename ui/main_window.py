"""

Emotion Data Studio — Main Window

==================================

Central application window with sidebar navigation + stacked pages.

Manages page routing and application-level state.

"""



import sys

from PySide6.QtWidgets import (

    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,

    QStackedWidget, QStatusBar, QLabel, QSizePolicy

)

from PySide6.QtCore import Qt, QTimer

from PySide6.QtGui import QFont, QIcon



from ui.widgets.sidebar import Sidebar

from ui.pages.dashboard_page import DashboardPage

from ui.pages.video_manager_page import VideoManagerPage

from ui.pages.processing_page import ProcessingPage

from ui.pages.segment_editor_page import SegmentEditorPage

from ui.pages.review_page import ReviewPage

from ui.pages.export_page import ExportPage
from ui.pages.settings_page import SettingsPage

from ui.styles.theme import Colors, Sizes





class MainWindow(QMainWindow):

    """

    Main application window.

    Layout: [Sidebar | Stacked Pages]

    """



    def __init__(self):

        super().__init__()

        self.setWindowTitle("Emotion Data Studio")

        self.setMinimumSize(1200, 750)

        self.resize(1440, 900)



        # Center window on screen

        self._center_on_screen()



        self._setup_ui()

        self._setup_statusbar()

        self._setup_updater()



    def _center_on_screen(self):

        """Center window on the primary screen"""

        from PySide6.QtWidgets import QApplication

        screen = QApplication.primaryScreen()

        if screen:

            screen_geo = screen.availableGeometry()

            x = (screen_geo.width() - self.width()) // 2

            y = (screen_geo.height() - self.height()) // 2

            self.move(x, y)



    def _setup_ui(self):

        """Build the main UI layout"""

        # Central widget

        central = QWidget()

        central.setObjectName("centralWidget")

        self.setCentralWidget(central)



        # Main layout: Sidebar + Content

        main_layout = QHBoxLayout(central)

        main_layout.setContentsMargins(0, 0, 0, 0)

        main_layout.setSpacing(0)



        # --- Sidebar ---

        self.sidebar = Sidebar()

        self.sidebar.page_changed.connect(self._on_page_changed)

        main_layout.addWidget(self.sidebar)



        # --- Content area (Stacked pages) ---

        self.content_stack = QStackedWidget()

        self.content_stack.setObjectName("contentStack")

        main_layout.addWidget(self.content_stack, stretch=1)



        # --- Create pages ---

        self.dashboard_page = DashboardPage()

        self.video_manager_page = VideoManagerPage()

        self.processing_page = ProcessingPage()

        self.segment_editor_page = SegmentEditorPage()

        self.review_page = ReviewPage()

        self.export_page = ExportPage()

        self.settings_page = SettingsPage()



        # Add pages to stack (order matches sidebar NAV_ITEMS)

        self.content_stack.addWidget(self.dashboard_page)   # Index 0

        self.content_stack.addWidget(self.video_manager_page)  # Index 1

        self.content_stack.addWidget(self.processing_page)  # Index 2

        self.content_stack.addWidget(self.segment_editor_page)  # Index 3

        self.content_stack.addWidget(self.review_page)      # Index 4

        self.content_stack.addWidget(self.export_page)      # Index 5

        self.content_stack.addWidget(self.settings_page)    # Index 6



        # Start on Dashboard

        self.content_stack.setCurrentIndex(0)



        # --- Connect cross-page signals ---

        self._connect_signals()

        # --- Load last active video project state on launch ---
        self._load_saved_project()



    def _setup_statusbar(self):

        """Setup the status bar with system info"""

        self.status_bar = QStatusBar()

        self.setStatusBar(self.status_bar)



        # Status message

        self.status_label = QLabel("Sẵn sàng")

        self.status_label.setObjectName("mutedText")

        self.status_bar.addWidget(self.status_label, stretch=1)



        # GPU status

        self.gpu_label = QLabel("GPU: N/A")

        self.gpu_label.setObjectName("mutedText")

        self.status_bar.addPermanentWidget(self.gpu_label)



        # Separator

        sep = QLabel("  |  ")

        sep.setObjectName("mutedText")

        self.status_bar.addPermanentWidget(sep)



        # Database status

        self.db_label = QLabel("ĐB: SQLite ✓")

        self.db_label.setObjectName("mutedText")

        self.status_bar.addPermanentWidget(self.db_label)



        # Update GPU status periodically

        self._gpu_timer = QTimer(self)

        self._gpu_timer.timeout.connect(self._update_gpu_status)

        self._gpu_timer.start(5000)  # Every 5 seconds

        self._update_gpu_status()  # Initial update



    def _on_page_changed(self, index: int):

        """Handle sidebar page navigation"""

        self.content_stack.setCurrentIndex(index)



        # Update status bar context

        page_names = ["Bảng Điều Khiển", "Quản Lý Video", "Xử Lý", "Soạn Đoạn", "Kiểm Duyệt", "Xuất & Đồng Bộ", "Cài Đặt"]

        if 0 <= index < len(page_names):

            self.status_label.setText(f"📍 {page_names[index]}")



        # Refresh page data when switching to it

        current_page = self.content_stack.currentWidget()

        if hasattr(current_page, 'refresh_data'):

            current_page.refresh_data()



    def _connect_signals(self):

        """Connect cross-page communication signals"""

        # Dashboard -> Processing: attach active worker so monitor gets preflight/progress/log signals.

        if hasattr(self.dashboard_page, 'processing_worker_started'):

            self.dashboard_page.processing_worker_started.connect(

                self._on_processing_worker_started

            )

        if hasattr(self.dashboard_page, 'active_video_changed'):

            self.dashboard_page.active_video_changed.connect(self._on_active_video_changed)

        if hasattr(self.video_manager_page, 'active_video_changed'):

            self.video_manager_page.active_video_changed.connect(self._on_active_video_changed)

        if hasattr(self.video_manager_page, 'open_review_requested'):

            self.video_manager_page.open_review_requested.connect(self._on_open_review_requested)

        if hasattr(self.video_manager_page, 'open_segment_requested'):

            self.video_manager_page.open_segment_requested.connect(self._on_open_segment_requested)



        # Dashboard -> Processing: backward-compatible navigation signal.

        if hasattr(self.dashboard_page, 'processing_started'):

            self.dashboard_page.processing_started.connect(

                self._on_processing_started

            )

        if hasattr(self.video_manager_page, 'active_video_changed'):

            self.video_manager_page.active_video_changed.connect(self._on_active_video_changed)

        if hasattr(self.video_manager_page, 'open_review_requested'):

            self.video_manager_page.open_review_requested.connect(self._on_open_review_for_video)

        if hasattr(self.video_manager_page, 'open_segment_requested'):

            self.video_manager_page.open_segment_requested.connect(self._on_open_segment_for_video)



        # Processing → Review: when processing completes

        if hasattr(self.processing_page, 'processing_completed'):

            self.processing_page.processing_completed.connect(

                self._on_processing_completed

            )

        if hasattr(self.segment_editor_page, 'segment_worker_started'):

            self.segment_editor_page.segment_worker_started.connect(

                self._on_processing_worker_started

            )



        if hasattr(self.settings_page, 'settings_saved'):

            self.settings_page.settings_saved.connect(self._on_settings_saved)

        if hasattr(self.settings_page, 'update_check_requested'):

            self.settings_page.update_check_requested.connect(self.check_for_updates_manual)



    def _on_active_video_changed(self, video_id: str):

        """Scope review/export workspace to the selected source video."""

        if hasattr(self.review_page, 'set_active_video'):

            self.review_page.set_active_video(video_id)

        if hasattr(self.export_page, 'set_active_video'):

            self.export_page.set_active_video(video_id)

        if hasattr(self.dashboard_page, 'set_active_video'):

            self.dashboard_page.set_active_video(video_id)

        self.status_label.setText(f"Active video set: {video_id[:8]}...")



    def _on_open_review_for_video(self, video_id: str):

        self._on_active_video_changed(video_id)

        self.sidebar.set_page(4)

        self.content_stack.setCurrentIndex(4)



    def _on_open_segment_for_video(self, video_id: str, video_path: str):

        self._on_active_video_changed(video_id)

        if hasattr(self.segment_editor_page, 'load_video'):

            self.segment_editor_page.load_video(video_id, video_path, 'semi_auto')

        self.sidebar.set_page(3)

        self.content_stack.setCurrentIndex(3)



    def _on_settings_saved(self, data: dict):

        """Refresh diagnostics-aware pages after settings change."""

        self.status_label.setText("Cài đặt đã lưu")

        if hasattr(self.processing_page, 'refresh_data'):

            self.processing_page.refresh_data()



    def _on_active_video_changed(self, video_id: str):

        """Propagate active video scope to review/export pages."""

        if hasattr(self.review_page, 'set_active_video'):

            self.review_page.set_active_video(video_id)

        if hasattr(self.export_page, 'set_active_video'):

            self.export_page.set_active_video(video_id)

        if hasattr(self.dashboard_page, 'set_active_video'):

            self.dashboard_page.set_active_video(video_id)

        self.status_label.setText(f"Active video set: {video_id[:8]}...")



    def _on_open_review_requested(self, video_id: str):

        self._on_active_video_changed(video_id)

        self.sidebar.set_page(4)

        self.content_stack.setCurrentIndex(4)

        if hasattr(self.review_page, 'refresh_data'):

            self.review_page.refresh_data()



    def _on_open_segment_requested(self, video_id: str, path: str):

        self._on_active_video_changed(video_id)

        self.sidebar.set_page(3)

        self.content_stack.setCurrentIndex(3)

        if hasattr(self.segment_editor_page, 'load_video'):

            title = self.video_manager_page._videos.get(video_id, {}).get('title', '')

            self.segment_editor_page.load_video(video_id, path, 'semi_auto', title)



    def _on_processing_started(self, video_id: str):

        """Switch to processing page when pipeline starts (legacy signal)."""

        self.sidebar.set_page(2)

        self.content_stack.setCurrentIndex(2)

        # If a worker has already been attached through processing_worker_started,

        # do not reset the monitor and disconnect log/progress context.

        if hasattr(self.processing_page, '_worker') and self.processing_page._worker is not None:

            return

        if hasattr(self.processing_page, 'start_monitoring'):

            self.processing_page.start_monitoring(video_id)



    def _on_processing_worker_started(self, worker, source: str):

        """Switch to Processing page and bind worker signals to the monitor."""

        self.sidebar.set_page(2)          # index 2 = Xử Lý

        self.content_stack.setCurrentIndex(2)

        if hasattr(self.processing_page, 'attach_worker'):

            self.processing_page.attach_worker(worker, source)



    def _on_processing_completed(self, video_id: str):

        """Khi pipeline hoàn thành: cập nhật dữ liệu, đặt video active, kết nối nút Xem kết quả."""

        self.status_label.setText(f"✅ Hoàn thành xử lý video {video_id[:8]}...")

        # 1. Cập nhật Quản lý Video
        if hasattr(self.video_manager_page, 'refresh_data'):
            self.video_manager_page.refresh_data()

        # 2. Set active video cho Review + Export
        self._on_active_video_changed(video_id)

        # 3. Kết nối nút "Xem kết quả" → chuyển thẳng sang Kiểm Duyệt
        btn = getattr(self.processing_page, 'view_results_btn', None)
        if btn is not None:
            # ngắt kết nối cũ trước (nếu có) rồi gắn mới
            try:
                btn.clicked.disconnect()
            except RuntimeError:
                pass
            btn.clicked.connect(self._go_to_review_studio)



    def _go_to_review_studio(self):
        """Chuyển thẳng sang trang Kiểm Duyệt và tải lại clip."""
        self.sidebar.set_page(4)
        self.content_stack.setCurrentIndex(4)
        if hasattr(self.review_page, 'refresh_data'):
            self.review_page.refresh_data()
        self.status_label.setText("📍 Kiểm Duyệt — Clip đã sẵn sàng để kiểm tra")

    def _update_gpu_status(self):

        """Check and update GPU availability in status bar"""

        try:

            import torch

            if torch.cuda.is_available():

                gpu_name = torch.cuda.get_device_name(0)

                # Shorten GPU name

                short_name = gpu_name.split("NVIDIA ")[-1] if "NVIDIA" in gpu_name else gpu_name

                gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)

                self.gpu_label.setText(f"🖥️ GPU: {short_name} ({gpu_mem:.0f}GB)")

            else:

                self.gpu_label.setText("🖥️ GPU: CPU Only")

        except ImportError:

            self.gpu_label.setText("🖥️ GPU: N/A")

        except Exception:

            self.gpu_label.setText("🖥️ GPU: Error")



    def navigate_to(self, page_index: int):

        """Programmatically navigate to a page"""

        self.sidebar.set_page(page_index)

        self.content_stack.setCurrentIndex(page_index)



    def set_status(self, message: str):

        """Update status bar message"""

        self.status_label.setText(message)



    def _setup_updater(self):

        """Initialize auto-updater — check for updates on startup"""

        try:

            from ui.updater import UpdateManager

            self._update_manager = UpdateManager(self)

            # Check for updates silently after 3 seconds

            QTimer.singleShot(3000, lambda: self._update_manager.check_for_updates(silent=True))

        except Exception:

            pass  # Updater is optional — don't crash if it fails



    def check_for_updates_manual(self):

        """Manual update check (from settings menu)"""

        if hasattr(self, '_update_manager'):

            self._update_manager.check_for_updates(silent=False)

    def _load_saved_project(self):
        try:
            import json
            from backend.config import settings
            state_path = settings.DATA_DIR / "project_state.json"
            if state_path.exists():
                state = json.loads(state_path.read_text(encoding="utf-8"))
                video_id = state.get("last_active_video_id")
                if video_id:
                    self._on_active_video_changed(video_id)
        except Exception as exc:
            print(f"Error loading saved project state: {exc}")

