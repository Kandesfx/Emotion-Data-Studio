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
from ui.pages.processing_page import ProcessingPage
from ui.pages.review_page import ReviewPage
from ui.pages.export_page import ExportPage
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
        self.processing_page = ProcessingPage()
        self.review_page = ReviewPage()
        self.export_page = ExportPage()

        # Add pages to stack (order matches sidebar NAV_ITEMS)
        self.content_stack.addWidget(self.dashboard_page)   # Index 0
        self.content_stack.addWidget(self.processing_page)  # Index 1
        self.content_stack.addWidget(self.review_page)      # Index 2
        self.content_stack.addWidget(self.export_page)      # Index 3

        # Start on Dashboard
        self.content_stack.setCurrentIndex(0)

        # --- Connect cross-page signals ---
        self._connect_signals()

    def _setup_statusbar(self):
        """Setup the status bar with system info"""
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)

        # Status message
        self.status_label = QLabel("Ready")
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
        self.db_label = QLabel("DB: SQLite ✓")
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
        page_names = ["Dashboard", "Processing Monitor", "Review Studio", "Export & Sync"]
        if 0 <= index < len(page_names):
            self.status_label.setText(f"📍 {page_names[index]}")

        # Refresh page data when switching to it
        current_page = self.content_stack.currentWidget()
        if hasattr(current_page, 'refresh_data'):
            current_page.refresh_data()

    def _connect_signals(self):
        """Connect cross-page communication signals"""
        # Dashboard → Processing: when user starts processing a video
        if hasattr(self.dashboard_page, 'processing_started'):
            self.dashboard_page.processing_started.connect(
                self._on_processing_started
            )

        # Processing → Review: when processing completes
        if hasattr(self.processing_page, 'processing_completed'):
            self.processing_page.processing_completed.connect(
                self._on_processing_completed
            )

    def _on_processing_started(self, video_id: str):
        """Switch to processing page when pipeline starts"""
        self.sidebar.set_page(1)
        self.content_stack.setCurrentIndex(1)
        if hasattr(self.processing_page, 'start_monitoring'):
            self.processing_page.start_monitoring(video_id)

    def _on_processing_completed(self, video_id: str):
        """Notify user when processing completes"""
        self.status_label.setText(f"✅ Processing completed for video {video_id[:8]}...")

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
