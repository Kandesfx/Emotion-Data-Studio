"""
Emotion Data Studio — Branded Splash Screen
============================================
- GIF giữ nguyên tỉ lệ gốc (1:1), to, chiếm hầu hết màn hình
- Chữ + loading bar nằm gọn bên dưới GIF
- Progress bar chạy giả lập (sweep animation) trong suốt thời gian GIF phát
- App chỉ mở sau khi GIF hoàn thành đủ 1 vòng
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from PySide6.QtCore import (
    Qt, QTimer, QEventLoop, QPropertyAnimation,
    QEasingCurve, Property, QObject, Signal, QSize
)
from PySide6.QtGui import QMovie, QPainter, QColor, QLinearGradient, QPen, QFont
from PySide6.QtWidgets import (
    QApplication,
    QLabel,
    QProgressBar,
    QVBoxLayout,
    QHBoxLayout,
    QWidget,
    QFrame,
    QSizePolicy,
)


def _asset_path(filename: str) -> str:
    """Resolve asset path — works for both dev and PyInstaller bundle."""
    if getattr(sys, 'frozen', False):
        base = Path(sys._MEIPASS) / "ui" / "assets"
    else:
        base = Path(__file__).parent / "assets"
    return str(base / filename)


# ──────────────────────────────────────────────────────────────────────────────
# Indeterminate (sweep) progress bar
# ──────────────────────────────────────────────────────────────────────────────

class SweepProgressBar(QWidget):
    """
    Thanh progress bar chạy sweep liên tục (không xác định tiến trình).
    Vẽ bằng QPainter để đạt hiệu ứng gradient di chuyển mượt.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(4)
        self._offset = 0.0          # 0.0 → 1.0, vị trí highlight
        self._direction = 1
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(16)       # ~60 fps

    def _tick(self):
        self._offset += 0.012 * self._direction
        if self._offset >= 1.0:
            self._offset = 1.0
            self._direction = -1
        elif self._offset <= 0.0:
            self._offset = 0.0
            self._direction = 1
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()

        # Nền mờ
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(255, 255, 255, 14))
        p.drawRoundedRect(0, 0, w, h, 2, 2)

        # Highlight di chuyển
        center_x = int(self._offset * w)
        grad = QLinearGradient(center_x - 120, 0, center_x + 120, 0)
        grad.setColorAt(0.0,  QColor(108, 92, 231, 0))
        grad.setColorAt(0.35, QColor(108, 92, 231, 180))
        grad.setColorAt(0.5,  QColor(162, 155, 254, 255))
        grad.setColorAt(0.65, QColor(108, 92, 231, 180))
        grad.setColorAt(1.0,  QColor(108, 92, 231, 0))

        p.setBrush(grad)
        p.drawRoundedRect(0, 0, w, h, 2, 2)
        p.end()

    def stop(self):
        self._timer.stop()


# ──────────────────────────────────────────────────────────────────────────────
# Splash screen
# ──────────────────────────────────────────────────────────────────────────────

class SplashScreen(QWidget):
    """
    Splash screen toàn màn hình với GIF logo to + info bar bên dưới.

    Vòng đời:
        splash.show()
        splash.play_and_wait()   # block cho đến khi GIF xong 1 vòng
        # ... loading tasks ...
        splash.finish(window)
    """

    def __init__(self):
        super().__init__()
        self._movie: QMovie | None = None
        self._gif_loop = QEventLoop()
        self._gif_total_frames = 0
        self._setup_window()
        self._setup_ui()

    # ── Window ────────────────────────────────────────────────────────────

    def _setup_window(self):
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint |
            Qt.WindowType.WindowStaysOnTopHint |
            Qt.WindowType.SplashScreen
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)

        # Lấy kích thước màn hình để tính splash size
        screen = QApplication.primaryScreen()
        if screen:
            avail = screen.availableGeometry()
            # Splash kích thước vừa phải (rộng 480px), tự động thu nhỏ nếu màn hình quá bé
            splash_w = min(480, avail.width() - 40)
            bottom_h = 110   # chữ + progress bar + padding
            gif_px   = splash_w
            splash_h = gif_px + bottom_h
            if splash_h > avail.height() - 40:
                splash_h = avail.height() - 40
                gif_px = splash_h - bottom_h
                splash_w = gif_px
        else:
            splash_w = 480
            bottom_h = 110   # chữ + progress bar + padding
            gif_px   = splash_w
            splash_h = gif_px + bottom_h

        self._gif_px = gif_px
        self.setFixedSize(splash_w, splash_h)
        self._center()

    def _center(self):
        screen = QApplication.primaryScreen()
        if screen:
            geo = screen.availableGeometry()
            self.move(
                (geo.width()  - self.width())  // 2,
                (geo.height() - self.height()) // 2,
            )

    # ── UI ────────────────────────────────────────────────────────────────

    def _setup_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # Nền bo góc
        container = QWidget()
        container.setObjectName("splashContainer")
        container.setStyleSheet("""
            QWidget#splashContainer {
                background: qlineargradient(
                    x1:0, y1:0, x2:0.5, y2:1,
                    stop:0   #0d0d18,
                    stop:0.7 #0a0818,
                    stop:1   #06060f
                );
                border-radius: 20px;
                border: 1px solid rgba(108, 92, 231, 0.18);
            }
        """)
        inner = QVBoxLayout(container)
        inner.setContentsMargins(0, 0, 0, 0)
        inner.setSpacing(0)

        # ── GIF — không set scaled size để giữ tỉ lệ gốc ─────────────────
        self.logo_label = QLabel()
        self.logo_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.logo_label.setFixedSize(self._gif_px, self._gif_px)
        self.logo_label.setStyleSheet("background: transparent;")

        gif_path = _asset_path("logo.gif")
        if os.path.exists(gif_path):
            self._movie = QMovie(gif_path)
            # Scale theo chiều dài cạnh ngắn hơn để giữ tỉ lệ 1:1
            self._movie.setScaledSize(QSize(self._gif_px, self._gif_px))
            self.logo_label.setMovie(self._movie)
        else:
            # Fallback text
            self.logo_label.setText("EDS")
            self.logo_label.setStyleSheet("""
                font-size: 120px; font-weight: 800;
                color: #a29bfe; background: transparent;
            """)

        inner.addWidget(self.logo_label)

        # ── Bottom info bar ───────────────────────────────────────────────
        bottom = QWidget()
        bottom.setStyleSheet("background: transparent;")
        bottom_layout = QVBoxLayout(bottom)
        bottom_layout.setContentsMargins(32, 8, 32, 24)
        bottom_layout.setSpacing(8)

        # App title + version
        top_row = QHBoxLayout()
        title_lbl = QLabel("Emotion Data Studio")
        title_lbl.setStyleSheet("""
            font-size: 15px; font-weight: 700;
            color: #d0cee8; letter-spacing: 0.3px;
            background: transparent;
        """)
        top_row.addWidget(title_lbl)
        top_row.addStretch()

        try:
            from backend.config import settings
            ver = settings.VERSION
        except Exception:
            ver = "1.1.0"

        ver_lbl = QLabel(f"v{ver}")
        ver_lbl.setStyleSheet("""
            font-size: 11px; color: #4a4860;
            background: transparent;
        """)
        top_row.addWidget(ver_lbl)
        bottom_layout.addLayout(top_row)

        # Sweep progress bar
        self.progress_bar = SweepProgressBar()
        bottom_layout.addWidget(self.progress_bar)

        # Status label
        self.status_label = QLabel("Đang khởi động...")
        self.status_label.setStyleSheet("""
            font-size: 11px; color: #4a4860;
            background: transparent; letter-spacing: 0.3px;
        """)
        bottom_layout.addWidget(self.status_label)

        inner.addWidget(bottom)
        root.addWidget(container)

    # ── Public API ────────────────────────────────────────────────────────

    def play_and_wait(self):
        """
        Phát GIF và BLOCK event loop cho đến khi GIF chạy xong 1 vòng đầy đủ.
        Progress bar chạy sweep animation trong suốt thời gian chờ.
        """
        if self._movie is None:
            # Không có GIF — chờ 2s
            loop = QEventLoop()
            QTimer.singleShot(2000, loop.quit)
            loop.exec()
            return

        self._gif_total_frames = self._movie.frameCount()
        self._movie.start()

        def _on_frame(frame_num: int):
            # Khi tới frame cuối → quit loop
            if self._gif_total_frames > 0 and frame_num >= self._gif_total_frames - 1:
                if self._gif_loop.isRunning():
                    self._gif_loop.quit()

        self._movie.frameChanged.connect(_on_frame)
        self._movie.finished.connect(self._quit_gif_loop)

        # Timeout tuyệt đối 20s
        QTimer.singleShot(20_000, self._quit_gif_loop)

        self._gif_loop.exec()   # ← BLOCK cho đến khi GIF xong

    def _quit_gif_loop(self):
        if self._gif_loop.isRunning():
            self._gif_loop.quit()

    def set_status(self, text: str):
        """Cập nhật dòng status."""
        self.status_label.setText(text)
        QApplication.processEvents()

    def finish(self):
        """Dừng sweep, hiện 'Sẵn sàng!' rồi đóng splash."""
        self.progress_bar.stop()
        self.status_label.setText("Sẵn sàng!")
        QApplication.processEvents()
        if self._movie and self._movie.state() == QMovie.MovieState.Running:
            self._movie.stop()
        QTimer.singleShot(200, self.close)
