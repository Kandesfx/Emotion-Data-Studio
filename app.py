"""
Emotion Data Studio — Desktop Application Entry Point
=======================================================
Launches the PySide6 native desktop application.
Single process: UI + Backend (no HTTP server needed).

Usage:
    python app.py
"""

import sys
import os
import logging

# Ensure project root is in Python path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def setup_logging():
    """Configure application logging"""
    log_dir = os.path.join(PROJECT_ROOT, "logs")
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

    # Initialize database
    logger.info("Initializing database...")
    if init_database():
        logger.info("Database initialized successfully")
    else:
        logger.warning("Database initialization failed — running without DB")

    # Create Qt Application
    from PySide6.QtWidgets import QApplication
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QFont

    app = QApplication(sys.argv)

    # Application metadata
    app.setApplicationName("Emotion Data Studio")
    app.setApplicationVersion("1.0.0")
    app.setOrganizationName("BCDA Team")

    # Set default font
    default_font = QFont("Inter", 10)
    default_font.setStyleHint(QFont.StyleHint.SansSerif)
    app.setFont(default_font)

    # Enable high DPI scaling
    app.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )

    # Load dark theme stylesheet
    logger.info("Loading dark theme stylesheet...")
    load_stylesheet(app)

    # Create and show main window
    logger.info("Creating main window...")
    from ui.main_window import MainWindow
    window = MainWindow()
    window.show()

    logger.info("Application ready — entering event loop")

    # Run application
    exit_code = app.exec()

    logger.info(f"Application exiting with code {exit_code}")
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
