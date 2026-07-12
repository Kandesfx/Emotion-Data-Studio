"""
Emotion Data Studio — Sidebar Navigation Widget
================================================
Collapsible sidebar with icon-based navigation, version badge, and
smooth hover/active transitions.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal, QSize, QPropertyAnimation, QEasingCurve
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ui.styles.theme import Sizes, Colors, Spacing

try:
    import qtawesome as qta
    _HAS_QTA = True
except ImportError:
    _HAS_QTA = False


# QtAwesome icon mapping — chuyên dụng cho từng trang
NAV_ITEMS = [
    ("fa5s.tachometer-alt",  "Bảng Điều Khiển",  0),
    ("fa5s.film",             "Quản Lý Video",    1),
    ("fa5s.cogs",             "Xử Lý",            2),
    ("fa5s.cut",              "Soạn Đoạn",        3),
    ("fa5s.eye",              "Kiểm Duyệt",       4),
    ("fa5s.cloud-download-alt","Xuất & Đồng Bộ", 5),
    ("fa5s.sliders-h",        "Cài Đặt",          6),
]


def _icon(name: str, color: str = Colors.TEXT_PRIMARY, size: int = 16) -> "QIcon":
    if not _HAS_QTA:
        from PySide6.QtGui import QIcon
        return QIcon()
    return qta.icon(name, color=color, color_active=Colors.ACCENT_LIGHT,
                    color_disabled=Colors.TEXT_TERTIARY)


class SidebarButton(QPushButton):
    """Individual navigation button with icon + label."""

    def __init__(self, icon_name: str, label: str, page_index: int, parent=None):
        super().__init__(parent)
        self.page_index = page_index
        self._icon_name = icon_name
        self._label = label
        self._build()

    def _build(self):
        self.setObjectName("sidebarBtn")
        self.setCheckable(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setMinimumHeight(44)
        self.setToolTip(self._label)
        if _HAS_QTA:
            self.setIcon(_icon(self._icon_name, Colors.TEXT_SECONDARY, 16))
            self.setIconSize(QSize(16, 16))
        self.setText(f"  {self._label}  ")  # padding để icon có chỗ
        self.setFlat(True)


class Sidebar(QWidget):
    """Left navigation panel — fixed width with page routing. Supports collapse."""

    page_changed = Signal(int)

    EXPANDED_WIDTH = 220
    COLLAPSED_WIDTH = 64

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("sidebar")
        self.setFixedWidth(self.EXPANDED_WIDTH)
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

        # Logo row: icon + text (icon shown khi collapsed, text chỉ khi expanded)
        logo_row = QHBoxLayout()
        logo_row.setSpacing(8)
        if _HAS_QTA:
            logo_icon = QLabel()
            logo_icon.setPixmap(_icon("fa5s.heart", Colors.ACCENT_LIGHT, 18).pixmap(18, 18))
            logo_icon.setStyleSheet("background: transparent;")
            self._logo_icon_lbl = logo_icon
            logo_row.addWidget(logo_icon)
        logo = QLabel("Emotion Data Studio")
        logo.setObjectName("sidebarLogo")
        logo.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self._logo_text_lbl = logo
        logo_row.addWidget(logo, stretch=1)
        header_layout.addLayout(logo_row)

        try:
            from backend.config import settings
            ver = settings.VERSION
        except Exception:
            ver = "1.1.0"

        version = QLabel(f"v{ver}")
        version.setObjectName("sidebarVersion")
        version.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self._version_lbl = version
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
        for icon_name, label, index in NAV_ITEMS:
            btn = SidebarButton(icon_name, label, index)
            btn.clicked.connect(lambda checked=False, i=index: self.set_page(i, emit=True))
            self._buttons.append(btn)
            layout.addWidget(btn)
            layout.addSpacing(2)

        layout.addStretch()

        # ── Collapse toggle button ───────────────────────────────────────
        self._collapse_btn = QPushButton()
        self._collapse_btn.setObjectName("sidebarCollapseBtn")
        self._collapse_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._collapse_btn.setToolTip("Thu gọn thanh bên")
        self._collapse_btn.setFixedHeight(36)
        if _HAS_QTA:
            self._collapse_btn.setIcon(_icon("fa5s.chevron-left", Colors.TEXT_SECONDARY, 14))
            self._collapse_btn.setIconSize(QSize(14, 14))
        self._collapse_btn.setText("  Thu gọn")
        self._collapse_btn.clicked.connect(lambda: self.set_collapsed(True))
        layout.addWidget(self._collapse_btn)

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
        self._footer_lbl = build_label
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

    def set_collapsed(self, collapsed: bool):
        """Toggle collapsed mode (responsive)."""
        target = self.COLLAPSED_WIDTH if collapsed else self.EXPANDED_WIDTH
        if self.width() == target:
            return

        # Animate width
        try:
            self._anim = QPropertyAnimation(self, b"minimumWidth")
            self._anim.setDuration(200)
            self._anim.setStartValue(self.width())
            self._anim.setEndValue(target)
            self._anim.setEasingCurve(QEasingCurve.Type.InOutQuad)
            self._anim.start()
        except Exception:
            pass

        self.setFixedWidth(target)
        # Hide text labels
        for widget, visible in [
            (getattr(self, '_logo_text_lbl', None), not collapsed),
            (getattr(self, '_version_lbl', None),    not collapsed),
            (getattr(self, '_footer_lbl', None),     not collapsed),
        ]:
            if widget:
                widget.setVisible(visible)

        # Buttons: chỉ hiện icon khi collapsed, icon+text khi expanded
        for btn in self._buttons:
            if collapsed:
                btn.setText("")
                btn.setToolTip(btn._label)
            else:
                btn.setText(f"  {btn._label}  ")

        if collapsed:
            self._collapse_btn.setText("")
            self._collapse_btn.setToolTip("Mở rộng thanh bên")
            if _HAS_QTA:
                self._collapse_btn.setIcon(_icon("fa5s.chevron-right",
                                                  Colors.TEXT_SECONDARY, 14))
                self._collapse_btn.setIconSize(QSize(14, 14))
            # Đổi click handler
            try:
                self._collapse_btn.clicked.disconnect()
            except Exception:
                pass
            self._collapse_btn.clicked.connect(lambda: self.set_collapsed(False))
        else:
            self._collapse_btn.setText("  Thu gọn")
            self._collapse_btn.setToolTip("Thu gọn thanh bên")
            if _HAS_QTA:
                self._collapse_btn.setIcon(_icon("fa5s.chevron-left",
                                                  Colors.TEXT_SECONDARY, 14))
                self._collapse_btn.setIconSize(QSize(14, 14))
            try:
                self._collapse_btn.clicked.disconnect()
            except Exception:
                pass
            self._collapse_btn.clicked.connect(lambda: self.set_collapsed(True))
