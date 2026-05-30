"""
Emotion Data Studio — Auto-Updater
====================================
Checks for new versions from Cloudflare R2 and downloads updates.
Works with PyInstaller builds — no Electron needed.

Update flow:
  1. App starts → check latest.json on R2
  2. Compare current version vs latest version
  3. If newer → show notification dialog
  4. User confirms → download .exe installer in background
  5. Launch installer → close current app
"""

import os
import sys
import json
import logging
import tempfile
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

from PySide6.QtCore import QThread, Signal, QUrl
from PySide6.QtWidgets import QMessageBox, QProgressDialog
from PySide6.QtNetwork import QNetworkAccessManager, QNetworkRequest, QNetworkReply

logger = logging.getLogger(__name__)


# ============================================================
# Version Info
# ============================================================

CURRENT_VERSION = "1.0.0"

# R2 update URL — change to your actual Cloudflare R2 public URL
UPDATE_BASE_URL = os.getenv(
    "EDS_UPDATE_URL",
    "https://updates.your-domain.com/releases"
)


@dataclass
class UpdateInfo:
    """Information about an available update"""
    version: str
    download_url: str
    release_notes: str = ""
    file_size: int = 0
    sha256: str = ""


# ============================================================
# Version Comparison
# ============================================================

def parse_version(version_str: str) -> tuple:
    """Parse version string to tuple for comparison"""
    try:
        parts = version_str.strip().lstrip("v").split(".")
        return tuple(int(p) for p in parts)
    except (ValueError, AttributeError):
        return (0, 0, 0)


def is_newer(remote: str, local: str) -> bool:
    """Check if remote version is newer than local version"""
    return parse_version(remote) > parse_version(local)


# ============================================================
# Update Checker (QThread)
# ============================================================

class UpdateChecker(QThread):
    """
    Background thread to check for updates.
    Fetches latest.json from R2 and compares versions.
    """

    update_available = Signal(object)     # Emits UpdateInfo
    no_update = Signal()                   # No update needed
    check_failed = Signal(str)             # Error message

    def __init__(self, current_version: str = CURRENT_VERSION):
        super().__init__()
        self.current_version = current_version

    def run(self):
        """Check for updates by fetching latest.json from R2"""
        try:
            import urllib.request
            import ssl

            url = f"{UPDATE_BASE_URL}/latest.json"
            logger.info(f"Checking for updates: {url}")

            # Create SSL context (allow self-signed certs for dev)
            ctx = ssl.create_default_context()

            req = urllib.request.Request(url, headers={
                "User-Agent": f"EmotionDataStudio/{self.current_version}"
            })

            try:
                with urllib.request.urlopen(req, timeout=10, context=ctx) as response:
                    data = json.loads(response.read().decode("utf-8"))
            except (urllib.error.URLError, urllib.error.HTTPError) as e:
                logger.warning(f"Update check failed (network): {e}")
                self.check_failed.emit(f"Network error: {e}")
                return

            # Parse update info
            remote_version = data.get("version", "0.0.0")
            download_url = data.get("download_url", "")
            release_notes = data.get("release_notes", "")
            file_size = data.get("file_size", 0)
            sha256 = data.get("sha256", "")

            if is_newer(remote_version, self.current_version):
                update_info = UpdateInfo(
                    version=remote_version,
                    download_url=download_url,
                    release_notes=release_notes,
                    file_size=file_size,
                    sha256=sha256,
                )
                logger.info(f"Update available: {remote_version}")
                self.update_available.emit(update_info)
            else:
                logger.info(f"No update available (current: {self.current_version}, remote: {remote_version})")
                self.no_update.emit()

        except Exception as e:
            logger.error(f"Update check error: {e}")
            self.check_failed.emit(str(e))


# ============================================================
# Update Downloader (QThread)
# ============================================================

class UpdateDownloader(QThread):
    """Download update file in background with progress"""

    progress_updated = Signal(int, int)    # bytes_downloaded, total_bytes
    download_finished = Signal(str)         # downloaded file path
    download_failed = Signal(str)           # error message

    def __init__(self, update_info: UpdateInfo):
        super().__init__()
        self.update_info = update_info

    def run(self):
        """Download the update installer"""
        try:
            import urllib.request
            import hashlib

            url = self.update_info.download_url
            logger.info(f"Downloading update from: {url}")

            # Download to temp directory
            temp_dir = tempfile.mkdtemp(prefix="eds_update_")
            filename = f"EmotionDataStudio-{self.update_info.version}-setup.exe"
            filepath = os.path.join(temp_dir, filename)

            req = urllib.request.Request(url, headers={
                "User-Agent": f"EmotionDataStudio/{CURRENT_VERSION}"
            })

            with urllib.request.urlopen(req, timeout=300) as response:
                total = int(response.headers.get("Content-Length", 0))
                downloaded = 0
                chunk_size = 65536  # 64KB chunks

                with open(filepath, "wb") as f:
                    while True:
                        chunk = response.read(chunk_size)
                        if not chunk:
                            break
                        f.write(chunk)
                        downloaded += len(chunk)
                        self.progress_updated.emit(downloaded, total)

            # Verify SHA256 if provided
            if self.update_info.sha256:
                sha256 = hashlib.sha256()
                with open(filepath, "rb") as f:
                    for block in iter(lambda: f.read(65536), b""):
                        sha256.update(block)
                if sha256.hexdigest() != self.update_info.sha256:
                    os.remove(filepath)
                    self.download_failed.emit("SHA256 checksum mismatch!")
                    return

            logger.info(f"Update downloaded to: {filepath}")
            self.download_finished.emit(filepath)

        except Exception as e:
            logger.error(f"Download failed: {e}")
            self.download_failed.emit(str(e))


# ============================================================
# Update Manager (High-level API)
# ============================================================

class UpdateManager:
    """
    High-level update manager.
    Usage in MainWindow:
        self.updater = UpdateManager(self)
        self.updater.check_for_updates()
    """

    def __init__(self, parent_window):
        self.parent = parent_window
        self._checker = None
        self._downloader = None

    def check_for_updates(self, silent: bool = True):
        """
        Check for updates.
        If silent=True, only show dialog when update is available.
        If silent=False, always show result (for manual check).
        """
        self._silent = silent
        self._checker = UpdateChecker()
        self._checker.update_available.connect(self._on_update_available)
        if not silent:
            self._checker.no_update.connect(self._on_no_update)
            self._checker.check_failed.connect(self._on_check_failed)
        self._checker.start()

    def _on_update_available(self, update_info: UpdateInfo):
        """Show update dialog"""
        size_mb = update_info.file_size / (1024 * 1024) if update_info.file_size else 0

        msg = QMessageBox(self.parent)
        msg.setIcon(QMessageBox.Icon.Information)
        msg.setWindowTitle("Cập nhật mới")
        msg.setText(f"Phiên bản {update_info.version} đã sẵn sàng!")
        msg.setInformativeText(
            f"Phiên bản hiện tại: {CURRENT_VERSION}\n"
            f"Phiên bản mới: {update_info.version}\n"
            f"Kích thước: {size_mb:.1f} MB\n\n"
            f"{update_info.release_notes}"
        )
        msg.setStandardButtons(
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        msg.setDefaultButton(QMessageBox.StandardButton.Yes)
        msg.button(QMessageBox.StandardButton.Yes).setText("Tải và cập nhật")
        msg.button(QMessageBox.StandardButton.No).setText("Để sau")

        if msg.exec() == QMessageBox.StandardButton.Yes:
            self._download_update(update_info)

    def _on_no_update(self):
        """Show no update message (manual check only)"""
        QMessageBox.information(
            self.parent, "Kiểm tra cập nhật",
            f"Bạn đang sử dụng phiên bản mới nhất ({CURRENT_VERSION})."
        )

    def _on_check_failed(self, error: str):
        """Show error message (manual check only)"""
        QMessageBox.warning(
            self.parent, "Lỗi kiểm tra cập nhật",
            f"Không thể kiểm tra cập nhật:\n{error}"
        )

    def _download_update(self, update_info: UpdateInfo):
        """Start downloading the update"""
        # Show progress dialog
        self._progress = QProgressDialog(
            "Đang tải cập nhật...", "Hủy", 0, 100, self.parent
        )
        self._progress.setWindowTitle("Cập nhật")
        self._progress.setMinimumDuration(0)
        self._progress.show()

        self._downloader = UpdateDownloader(update_info)
        self._downloader.progress_updated.connect(self._on_download_progress)
        self._downloader.download_finished.connect(self._on_download_finished)
        self._downloader.download_failed.connect(self._on_download_failed)
        self._progress.canceled.connect(self._downloader.terminate)
        self._downloader.start()

    def _on_download_progress(self, downloaded: int, total: int):
        """Update progress dialog"""
        if total > 0:
            pct = int(downloaded / total * 100)
            self._progress.setValue(pct)
            mb_done = downloaded / (1024 * 1024)
            mb_total = total / (1024 * 1024)
            self._progress.setLabelText(
                f"Đang tải: {mb_done:.1f} / {mb_total:.1f} MB"
            )

    def _on_download_finished(self, filepath: str):
        """Launch installer and quit app"""
        self._progress.close()

        reply = QMessageBox.question(
            self.parent, "Cập nhật đã tải xong",
            "Khởi động lại ứng dụng để cài đặt cập nhật?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )

        if reply == QMessageBox.StandardButton.Yes:
            # Launch installer
            import subprocess
            subprocess.Popen([filepath, "/SILENT"], shell=True)
            # Quit current app
            from PySide6.QtWidgets import QApplication
            QApplication.instance().quit()

    def _on_download_failed(self, error: str):
        """Show download error"""
        self._progress.close()
        QMessageBox.critical(
            self.parent, "Lỗi tải cập nhật",
            f"Không thể tải cập nhật:\n{error}"
        )
