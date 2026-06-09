"""
Emotion Data Studio — Desktop Application Entry Point
=======================================================
Launches the PySide6 native desktop application.
Single process: UI + Backend (no HTTP server needed).

Usage:
    python app.py
"""

import os
import sys

# === Fix Windows console encoding for emoji/Unicode ===
# Must be done BEFORE any other imports or print() calls
os.environ['PYTHONIOENCODING'] = 'utf-8'
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

import logging

# Ensure project root is in Python path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def setup_logging():
    """Configure application logging"""
    if getattr(sys, 'frozen', False):
        log_dir = os.path.join(os.environ.get("LOCALAPPDATA", "."), "EmotionDataStudio", "logs")
    else:
        log_dir = os.path.join(PROJECT_ROOT, "data", "logs")
    os.makedirs(log_dir, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(log_dir, "app.log"), encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ]
    )
    return logging.getLogger("EmotionDataStudio")


def init_database():
    """Initialize SQLite database and create tables"""
    try:
        from backend.database.local_db import init_database as _init_db
        _init_db()
        return True
    except Exception as e:
        logging.getLogger("EmotionDataStudio").error(f"Database init failed: {e}")
        return False


def load_stylesheet(app) -> bool:
    """Load and apply the dark theme QSS stylesheet"""
    qss_path = os.path.join(PROJECT_ROOT, "ui", "styles", "dark_theme.qss")
    if os.path.exists(qss_path):
        with open(qss_path, "r", encoding="utf-8") as f:
            app.setStyleSheet(f.read())
        return True
    else:
        logging.getLogger("EmotionDataStudio").warning(
            f"QSS stylesheet not found: {qss_path}"
        )
        return False


def main():
    """Application entry point"""
    logger = setup_logging()
    logger.info("=" * 60)
    logger.info("Emotion Data Studio — Starting...")
    logger.info("=" * 60)

    # Create single-instance mutex (helps installer detect if app is running)
    if sys.platform == 'win32':
        import ctypes
        try:
            # Set AppUserModelID so Windows taskbar displays the custom logo icon instead of python.exe icon
            myappid = 'bcda.emotiondatastudio.app.v1'
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)
            logger.info("AppUserModelID set successfully for taskbar icon")
        except Exception as e:
            logger.warning(f"Failed to set AppUserModelID: {e}")

        try:
            MUTEX_NAME = "{B8C5D9E2-4F1A-4B7D-9E3C-8F2A1D5B6E7C}"
            kernel32 = ctypes.windll.kernel32
            mutex = kernel32.CreateMutexW(None, False, MUTEX_NAME)
            os._app_mutex = mutex  # Keep reference alive
            logger.info("Single-instance mutex created successfully")
        except Exception as e:
            logger.warning(f"Failed to create single-instance mutex: {e}")

    # ── MUST be set BEFORE QApplication is created ──────────────────────
    from PySide6.QtCore import Qt
    Qt.AA_EnableHighDpiScaling if hasattr(Qt, 'AA_EnableHighDpiScaling') else None

    # Create QApplication
    from PySide6.QtWidgets import QApplication
    from PySide6.QtGui import QFont, QIcon

    app = QApplication(sys.argv)

    # Set application icon globally
    icon_path = os.path.join(PROJECT_ROOT, "ui", "assets", "icon.png")
    if os.path.exists(icon_path):
        app.setWindowIcon(QIcon(icon_path))

    # High DPI — must come AFTER QApplication in Qt 6 but BEFORE any widget
    app.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )

    # Application metadata
    from backend.config import settings
    app.setApplicationName("Emotion Data Studio")
    app.setApplicationVersion(settings.VERSION)
    app.setOrganizationName("BCDA Team")

    # Set default font
    default_font = QFont("Inter", 10)
    default_font.setStyleHint(QFont.StyleHint.SansSerif)
    app.setFont(default_font)

    # Load dark theme stylesheet BEFORE splash so it gets the right styles
    logger.info("Loading dark theme stylesheet...")
    load_stylesheet(app)

    # ── BƯỚC 1: Hiện splash và chờ GIF chạy xong hoàn toàn ───────────────
    from ui.splash_screen import SplashScreen
    splash = SplashScreen()
    splash.show()
    app.processEvents()

    logger.info("Splash screen shown — playing logo GIF...")
    splash.play_and_wait()          # ← BLOCK cho đến khi GIF xong đủ 1 vòng
    logger.info("GIF animation complete — starting loading tasks...")

    # ── BƯỚC 2: Khởi tạo database ─────────────────────────────────────────
    splash.set_status("Đang khởi tạo cơ sở dữ liệu...")
    logger.info("Initializing database...")
    if init_database():
        logger.info("Database initialized successfully")
        splash.set_status("Cơ sở dữ liệu sẵn sàng")
    else:
        logger.warning("Database initialization failed — running without DB")
        splash.set_status("Không thể kết nối DB — tiếp tục...")

    # ── BƯỚC 3: Tạo main window ───────────────────────────────────────────
    splash.set_status("Đang tải giao diện...")
    logger.info("Creating main window...")
    from ui.main_window import MainWindow
    window = MainWindow()

    # ── BƯỚC 4: Hoàn tất — đóng splash, hiện main window ─────────────────
    splash.finish()                 # dừng sweep, hiện "Sẵn sàng!", rồi close
    window.show()

    logger.info("Application ready — entering event loop")
    exit_code = app.exec()
    logger.info(f"Application exiting with code {exit_code}")
    sys.exit(exit_code)



if __name__ == "__main__":
    main()
