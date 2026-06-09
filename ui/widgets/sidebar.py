"""
Emotion Data Studio — Sidebar Navigation Widget
================================================
Collapsible sidebar with icon-based navigation, version badge, and
smooth hover/active transitions.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal, QSize
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from ui.styles.theme import Sizes


# Unicode icons that render well in most system fonts
NAV_ITEMS = [
    ("⊙",  "Bảng Điều Khiển", 0),
    ("▶",  "Quản Lý Video",   1),
    ("⚙",  "Xử Lý",           2),
    ("✂",  "Soạn Đoạn",         3),
    ("◉",  "Kiểm Duyệt",      4),
    ("↑",  "Xuất & Đồng Bộ",  5),
    ("≡",  "Cài Đặt",          6),
]


class SidebarButton(QPushButton):
    """Individual navigation button with icon + label."""

    def __init__(self, icon: str, label: str, page_index: int, parent=None):
        super().__init__(parent)
        self.page_index = page_index
        self._icon = icon
        self._label = label
        self._build()

    def _build(self):
        self.setObjectName("sidebarBtn")
        self.setCheckable(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setMinimumHeight(44)
        self.setToolTip(self._label)
        self.setText(f" {self._icon}   {self._label}")
        self.setFlat(True)


class Sidebar(QWidget):
    """Left navigation panel — fixed width with page routing."""

    page_changed = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("sidebar")
        self.setFixedWidth(Sizes.SIDEBAR_WIDTH_EXPANDED)
        self._buttons: list[SidebarButton] = []
        self._current_index = 0
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 0, 10, 12)
        layout.setSpacing(0)

        # ── Logo header ───────────────────────────────────────────────────
        header = QWidget()
        header.setObjectName("sidebarHeader")
        header.setFixedHeight(80)
        header_layout = QVBoxLayout(header)
        header_layout.setContentsMargins(10, 14, 10, 8)
        header_layout.setSpacing(2)

        logo = QLabel("Emotion Data Studio")
        logo.setObjectName("sidebarLogo")
        logo.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        header_layout.addWidget(logo)

        try:
            from backend.config import settings
            ver = settings.VERSION
        except Exception:
            ver = "1.1.0"

        version = QLabel(f"v{ver}")
        version.setObjectName("sidebarVersion")
        version.setAlignment(Qt.AlignmentFlag.AlignLeft)
        header_layout.addWidget(version)

        layout.addWidget(header)

        # ── Divider ───────────────────────────────────────────────────────
        divider = QFrame()
        divider.setFrameShape(QFrame.Shape.HLine)
        divider.setObjectName("sidebarDivider")
        divider.setFixedHeight(1)
        layout.addWidget(divider)
        layout.addSpacing(8)

        # ── Nav buttons ───────────────────────────────────────────────────
        for icon, label, index in NAV_ITEMS:
            btn = SidebarButton(icon, label, index)
            btn.clicked.connect(lambda checked=False, i=index: self.set_page(i, emit=True))
            self._buttons.append(btn)
            layout.addWidget(btn)
            layout.addSpacing(2)

        layout.addStretch()

        # ── Bottom info ───────────────────────────────────────────────────
        bottom_divider = QFrame()
        bottom_divider.setFrameShape(QFrame.Shape.HLine)
        bottom_divider.setObjectName("sidebarDivider")
        bottom_divider.setFixedHeight(1)
        layout.addWidget(bottom_divider)
        layout.addSpacing(10)

        build_label = QLabel("BCDA Team · 2025")
        build_label.setObjectName("sidebarFooter")
        build_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(build_label)

        self.set_page(0, emit=False)

    def set_page(self, index: int, emit: bool = False):
        self._current_index = index
        for btn in self._buttons:
            btn.setChecked(btn.page_index == index)
        if emit:
            self.page_changed.emit(index)

    def current_page(self) -> int:
        return self._current_index
