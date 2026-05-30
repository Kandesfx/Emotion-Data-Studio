"""
Emotion Data Studio — Sidebar Navigation Widget
=================================================
Animated sidebar with icon + text navigation buttons.
Supports collapse/expand animation.
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QSizePolicy, QSpacerItem, QFrame
)
from PySide6.QtCore import Qt, Signal, QPropertyAnimation, QEasingCurve, QSize
from PySide6.QtGui import QIcon, QFont

from ui.styles.theme import Colors, Spacing, Sizes, Typography


class SidebarButton(QPushButton):
    """
    Sidebar navigation button with icon + text.
    Supports checked/active state.
    """

    def __init__(self, icon_text: str, label: str, page_index: int, parent=None):
        super().__init__(parent)
        self.page_index = page_index
        self.icon_text = icon_text
        self.label_text = label

        self.setText(f"  {icon_text}   {label}")
        self.setObjectName("sidebarBtn")
        self.setCheckable(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setMinimumHeight(42)
        self.setToolTip(label)


class Sidebar(QWidget):
    """
    Left sidebar navigation panel.
    Emits page_changed signal when user clicks a nav button.
    """

    page_changed = Signal(int)  # Emits page index

    # Navigation items: (icon_emoji, label, page_index)
    NAV_ITEMS = [
        ("📊", "Dashboard", 0),
        ("⚙️", "Processing", 1),
        ("🎭", "Review Studio", 2),
        ("📦", "Export & Sync", 3),
    ]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("sidebar")
        self.setFixedWidth(Sizes.SIDEBAR_WIDTH_EXPANDED)
        self.setMinimumHeight(400)

        self._buttons: list[SidebarButton] = []
        self._current_index = 0

        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 0, 8, 8)
        layout.setSpacing(2)

        # --- Logo section ---
        logo_label = QLabel("🎬 EDS")
        logo_label.setObjectName("sidebarLogo")
        layout.addWidget(logo_label)

        version_label = QLabel("Emotion Data Studio v1.0.0")
        version_label.setObjectName("sidebarVersion")
        layout.addWidget(version_label)

        # --- Separator ---
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet("background-color: rgba(255,255,255,0.06); max-height: 1px;")
        layout.addWidget(sep)
        layout.addSpacing(8)

        # --- Section label ---
        nav_title = QLabel("NAVIGATION")
        nav_title.setObjectName("sidebarTitle")
        layout.addWidget(nav_title)

        # --- Navigation buttons ---
        for icon, label, index in self.NAV_ITEMS:
            btn = SidebarButton(icon, label, index, self)
            btn.clicked.connect(lambda checked, idx=index: self._on_button_clicked(idx))
            self._buttons.append(btn)
            layout.addWidget(btn)

        # --- Spacer ---
        layout.addSpacerItem(QSpacerItem(
            20, 40, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding
        ))

        # --- Bottom section ---
        sep2 = QFrame()
        sep2.setFrameShape(QFrame.Shape.HLine)
        sep2.setStyleSheet("background-color: rgba(255,255,255,0.06); max-height: 1px;")
        layout.addWidget(sep2)
        layout.addSpacing(4)

        # Settings button
        settings_btn = SidebarButton("⚙️", "Settings", -1, self)
        settings_btn.setCheckable(False)
        settings_btn.clicked.connect(self._on_settings_clicked)
        layout.addWidget(settings_btn)

        # Set first button active
        self._set_active(0)

    def _on_button_clicked(self, index: int):
        """Handle navigation button click"""
        if index == self._current_index:
            return
        self._set_active(index)
        self.page_changed.emit(index)

    def _set_active(self, index: int):
        """Set the active button by index"""
        self._current_index = index
        for btn in self._buttons:
            btn.setChecked(btn.page_index == index)

    def _on_settings_clicked(self):
        """Open settings dialog (placeholder)"""
        pass  # Will be connected to SettingsDialog

    def set_page(self, index: int):
        """Programmatically set the active page"""
        self._set_active(index)
