"""
Custom frameless window titlebar.

Provides:
- Drag the titlebar to move the window
- Minimize / Maximize-Restore / Close buttons
- Toggle fullscreen (F11) — keeps window controls visible
- Double-click titlebar to toggle maximize
- App icon + title text on the left
"""

from __future__ import annotations

from PySide6.QtCore import Qt, QPoint, Signal
from PySide6.QtGui import QIcon, QMouseEvent
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QWidget,
)

from ui.styles.theme import Colors

try:
    import qtawesome as qta
    _HAS_QTA = True
except ImportError:
    _HAS_QTA = False


def _icon(name: str, color: str = Colors.TEXT_PRIMARY, size: int = 14) -> QIcon:
    if not _HAS_QTA:
        return QIcon()
    return qta.icon(name, color=color, color_active=Colors.TEXT_PRIMARY,
                    color_disabled=Colors.TEXT_TERTIARY)


class TitleBar(QWidget):
    """Frameless titlebar with drag, min/max/close/fullscreen."""

    minimize_clicked = Signal()
    maximize_clicked = Signal()
    close_clicked = Signal()
    fullscreen_clicked = Signal(bool)  # True = now fullscreen

    HEIGHT = 36

    def __init__(self, title: str = "Emotion Data Studio", parent=None):
        super().__init__(parent)
        self.setObjectName("customTitleBar")
        self.setFixedHeight(self.HEIGHT)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        self._drag_pos: QPoint | None = None
        self._is_fullscreen = False

        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 0, 0, 0)
        layout.setSpacing(0)

        # Left: app icon + title
        if _HAS_QTA:
            icon_lbl = QLabel()
            icon_lbl.setPixmap(_icon("fa5s.heart", Colors.ACCENT_LIGHT, size=14).pixmap(14, 14))
            icon_lbl.setStyleSheet("background: transparent; padding-right: 6px;")
            layout.addWidget(icon_lbl)

        self.title_label = QLabel(title)
        self.title_label.setStyleSheet(
            f"color: {Colors.TEXT_PRIMARY}; font-weight: 600;"
            f" font-size: 13px; background: transparent; padding-left: 4px;"
        )
        layout.addWidget(self.title_label)

        layout.addStretch()

        # Right: window controls
        # Fullscreen toggle
        self.fullscreen_btn = QPushButton()
        self.fullscreen_btn.setObjectName("titleBtnFullscreen")
        self.fullscreen_btn.setFixedSize(46, self.HEIGHT)
        self.fullscreen_btn.setToolTip("Bật/tắt toàn màn hình (F11)")
        self.fullscreen_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.fullscreen_btn.setIcon(_icon("fa5s.expand", Colors.TEXT_SECONDARY, size=12))
        self.fullscreen_btn.clicked.connect(self._on_fullscreen_clicked)
        layout.addWidget(self.fullscreen_btn)

        # Minimize
        self.min_btn = QPushButton()
        self.min_btn.setObjectName("titleBtnMin")
        self.min_btn.setFixedSize(46, self.HEIGHT)
        self.min_btn.setToolTip("Thu nhỏ (Win+Down)")
        self.min_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.min_btn.setIcon(_icon("fa5s.window-minimize", Colors.TEXT_SECONDARY, size=12))
        self.min_btn.clicked.connect(self.minimize_clicked.emit)
        layout.addWidget(self.min_btn)

        # Maximize/Restore
        self.max_btn = QPushButton()
        self.max_btn.setObjectName("titleBtnMax")
        self.max_btn.setFixedSize(46, self.HEIGHT)
        self.max_btn.setToolTip("Phóng to / Khôi phục (Win+Up)")
        self.max_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.max_btn.setIcon(_icon("fa5s.window-maximize", Colors.TEXT_SECONDARY, size=12))
        self.max_btn.clicked.connect(self.maximize_clicked.emit)
        layout.addWidget(self.max_btn)

        # Close
        self.close_btn = QPushButton()
        self.close_btn.setObjectName("titleBtnClose")
        self.close_btn.setFixedSize(46, self.HEIGHT)
        self.close_btn.setToolTip("Đóng (Alt+F4)")
        self.close_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.close_btn.setIcon(_icon("fa5s.times", Colors.TEXT_SECONDARY, size=12))
        self.close_btn.clicked.connect(self.close_clicked.emit)
        layout.addWidget(self.close_btn)

    # ── Mouse drag to move window ────────────────────────────────
    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton:
            window = self.window()
            # Không drag khi đang fullscreen
            if window.isFullScreen():
                self._drag_pos = None
                return
            self._drag_pos = event.globalPosition().toPoint() - window.frameGeometry().topLeft()
            event.accept()
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent):
        if self._drag_pos is not None and event.buttons() & Qt.MouseButton.LeftButton:
            window = self.window()
            window.move(event.globalPosition().toPoint() - self._drag_pos)
            event.accept()
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent):
        self._drag_pos = None
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton and not self.window().isFullScreen():
            self.maximize_clicked.emit()
            event.accept()
        else:
            super().mouseDoubleClickEvent(event)

    # ── State sync from main window ─────────────────────────────
    def set_fullscreen_state(self, is_fullscreen: bool):
        self._is_fullscreen = is_fullscreen
        if _HAS_QTA:
            icon_name = "fa5s.compress" if is_fullscreen else "fa5s.expand"
            tooltip = "Thoát toàn màn hình (F11)" if is_fullscreen else "Bật toàn màn hình (F11)"
        else:
            icon_name = ""
            tooltip = "Toggle fullscreen (F11)"
        if _HAS_QTA:
            self.fullscreen_btn.setIcon(_icon(icon_name, Colors.ACCENT_LIGHT, size=12))
        self.fullscreen_btn.setToolTip(tooltip)

    def set_maximized_state(self, is_maximized: bool):
        if _HAS_QTA:
            icon_name = "fa5s.window-restore" if is_maximized else "fa5s.window-maximize"
            self.max_btn.setIcon(_icon(icon_name, Colors.TEXT_SECONDARY, size=12))
        self.max_btn.setToolTip("Khôi phục" if is_maximized else "Phóng to")

    def _on_fullscreen_clicked(self):
        new_state = not self._is_fullscreen
        self.set_fullscreen_state(new_state)
        self.fullscreen_clicked.emit(new_state)