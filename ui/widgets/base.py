"""
Shared UI building blocks for all pages.

Designed to make every page visually consistent (same header, card, button,
status pill, etc.) without duplicating style code in 7 page files.

- PageHeader   — large hero header with icon + title + subtitle + right slot
- Card         — rounded card wrapper (standard / elevated variants)
- SectionTitle — small title for sub-sections inside a card
- StatusPill   — colored pill (success / warning / error / info / muted)
- ActionButton — primary / secondary / danger / ghost with icon
- EmptyState   — friendly empty placeholder with icon + message + CTA

Icons use QtAwesome (Font Awesome 5 solid). Falls back to text-only if
QtAwesome is unavailable.
"""

from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Qt, QSize
from PySide6.QtGui import QIcon, QFont
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from ui.styles.theme import Colors, Spacing, BorderRadius, Typography

try:
    import qtawesome as qta
    _HAS_QTA = True
except ImportError:  # graceful fallback
    _HAS_QTA = False


def _icon(name: str, color: str = Colors.TEXT_PRIMARY, size: int = 16) -> QIcon:
    """Build a QIcon from QtAwesome. Returns empty icon if unavailable."""
    if not _HAS_QTA:
        return QIcon()
    return qta.icon(name, color=color, color_active=color, color_disabled=Colors.TEXT_TERTIARY)


# =====================================================================
#  PAGE HEADER  — hero header at the top of every page
# =====================================================================

class PageHeader(QFrame):
    """
    Consistent page header.

        ┌────────────────────────────────────────────────┐
        │ [icon]  Page Title                              │
        │          Subtitle / description          [right] │
        └────────────────────────────────────────────────┘

    Set `right_widget` (QWidget) to put buttons / badges on the right side.
    """

    def __init__(
        self,
        title: str,
        subtitle: str = "",
        icon: str = "",
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self.setObjectName("pageHeader")
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(Spacing.LG, Spacing.LG, Spacing.LG, Spacing.LG)
        layout.setSpacing(Spacing.MD)

        # Icon block
        if icon:
            icon_lbl = QLabel()
            icon_lbl.setPixmap(_icon(icon, Colors.ACCENT_LIGHT, size=28).pixmap(28, 28))
            icon_lbl.setFixedSize(40, 40)
            icon_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            icon_lbl.setStyleSheet(
                f"background-color: {Colors.ACCENT_SUBTLE};"
                f"border-radius: {BorderRadius.MD}px;"
            )
            layout.addWidget(icon_lbl)

        # Title + subtitle column
        text_box = QVBoxLayout()
        text_box.setSpacing(2)

        title_lbl = QLabel(title)
        title_font = QFont(Typography.FAMILY.split(",")[0].strip().strip('"'), Typography.SIZE_XXL)
        title_font.setWeight(QFont.Weight.Bold)
        title_lbl.setFont(title_font)
        title_lbl.setStyleSheet(f"color: {Colors.TEXT_PRIMARY}; background: transparent;")
        text_box.addWidget(title_lbl)

        if subtitle:
            sub_lbl = QLabel(subtitle)
            sub_lbl.setStyleSheet(f"color: {Colors.TEXT_SECONDARY}; background: transparent;")
            sub_lbl.setWordWrap(True)
            text_box.addWidget(sub_lbl)

        layout.addLayout(text_box, stretch=1)

        # Right slot (filled later via set_right_widget)
        self._right_holder = QHBoxLayout()
        self._right_holder.setSpacing(Spacing.SM)
        self._right_holder.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(self._right_holder)

    def set_right_widget(self, widget: QWidget):
        """Replace the right-side widget (buttons, badges, etc.)."""
        # Clear previous
        while self._right_holder.count():
            item = self._right_holder.takeAt(0)
            w = item.widget()
            if w is not None:
                w.setParent(None)
        self._right_holder.addWidget(widget)


# =====================================================================
#  CARD  — rounded container
# =====================================================================

class Card(QFrame):
    """
    Standard card container. Use objectName="card" or "cardElevated"
    so the global QSS can style it consistently.

    Variants:
      - "default"   (white-on-dark subtle)
      - "elevated"  (slightly brighter, used for primary action areas)
      - "accent"    (purple-tinted, used for AI / premium features)
    """

    def __init__(
        self,
        variant: str = "default",
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        if variant == "elevated":
            self.setObjectName("cardElevated")
        elif variant == "accent":
            self.setObjectName("cardAccent")
        else:
            self.setObjectName("card")
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)

        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(Spacing.LG, Spacing.LG, Spacing.LG, Spacing.LG)
        self._layout.setSpacing(Spacing.MD)

    def addLayout(self, layout) -> None:
        self._layout.addLayout(layout)

    def addWidget(self, widget: QWidget) -> None:
        self._layout.addWidget(widget)

    def addStretch(self) -> None:
        self._layout.addStretch()


# =====================================================================
#  SECTION TITLE  — small uppercase-ish title inside a card
# =====================================================================

class SectionTitle(QLabel):
    def __init__(self, text: str, icon: str = "", parent: Optional[QWidget] = None):
        if icon and _HAS_QTA:
            display = f"  {text}"  # icon is on a separate label so font aligns
            super().__init__(display, parent)
            icon_lbl = QLabel()
            icon_lbl.setPixmap(_icon(icon, Colors.ACCENT_LIGHT, size=14).pixmap(14, 14))
            icon_lbl.setStyleSheet("background: transparent;")
            # Build a horizontal layout in a sibling helper is overkill; embed inline
            self._prefix_icon = icon_lbl
        else:
            super().__init__(text, parent)
        self.setObjectName("sectionTitle")
        self.setStyleSheet(
            f"color: {Colors.TEXT_PRIMARY}; font-weight: {Typography.WEIGHT_SEMIBOLD};"
            f" background: transparent;"
        )


# =====================================================================
#  STATUS PILL  — colored pill (✓ OK, ⚠ Warning, ✗ Error, ⓘ Info, • Muted)
# =====================================================================

class StatusPill(QLabel):
    """Colored pill label with icon + text. Use for status indicators."""

    _MAP = {
        "success":  (Colors.SUCCESS,  Colors.SUCCESS_BG,  "fa5s.check"),
        "warning":  (Colors.WARNING,  Colors.WARNING_BG,  "fa5s.exclamation-triangle"),
        "error":    (Colors.ERROR,    Colors.ERROR_BG,    "fa5s.times-circle"),
        "info":     (Colors.INFO,     Colors.INFO_BG,     "fa5s.info-circle"),
        "muted":    (Colors.TEXT_SECONDARY, "rgba(255,255,255,0.05)", "fa5s.circle"),
        "pending":  (Colors.WARNING,  Colors.WARNING_BG,  "fa5s.clock"),
        "running":  (Colors.INFO,     Colors.INFO_BG,     "fa5s.sync"),
        "done":     (Colors.SUCCESS,  Colors.SUCCESS_BG,  "fa5s.check-circle"),
    }

    def __init__(self, status: str = "muted", text: str = "", parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.set_status(status, text)

    def set_status(self, status: str, text: str = ""):
        fg, bg, icon_name = self._MAP.get(status, self._MAP["muted"])
        display = text or {"success": "OK", "warning": "Cảnh báo",
                           "error": "Lỗi", "info": "Thông tin",
                           "muted": "—", "pending": "Chờ",
                           "running": "Đang chạy", "done": "Hoàn thành"}.get(status, status)
        if _HAS_QTA:
            self.setPixmap(_icon(icon_name, fg, size=12).pixmap(12, 12))
            self.setText(f"  {display}")
        else:
            self.setText(f"• {display}")
        self.setStyleSheet(
            f"color: {fg}; background-color: {bg};"
            f" border: 1px solid {fg}40;"
            f" border-radius: {BorderRadius.ROUND}px;"
            f" padding: 3px 10px; font-weight: {Typography.WEIGHT_MEDIUM};"
            f" font-size: {Typography.SIZE_SM}px;"
        )
        self.setMinimumHeight(22)


# =====================================================================
#  ACTION BUTTON  — primary / secondary / danger / ghost with icon
# =====================================================================

class ActionButton(QPushButton):
    """
    Themed action button. Variants:
      - "primary"   — purple gradient (default)
      - "secondary" — lighter purple outline
      - "danger"    — red, for destructive actions
      - "ghost"     — transparent, for tertiary actions
    """

    def __init__(
        self,
        text: str,
        icon: str = "",
        variant: str = "primary",
        parent: Optional[QWidget] = None,
    ):
        super().__init__(text, parent)
        self._variant = variant
        self._icon_name = icon
        self.setMinimumHeight(36)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setObjectName({
            "primary":   "actionPrimary",
            "secondary": "actionSecondary",
            "danger":    "actionDanger",
            "ghost":     "actionGhost",
        }.get(variant, "actionPrimary"))
        if icon:
            self.setIcon(_icon(icon, Colors.TEXT_PRIMARY, size=14))
            self.setIconSize(QSize(14, 14))

    def setText(self, text: str):  # type: ignore[override]
        # Re-apply icon spacing when text changes
        super().setText(f"  {text}" if self._icon_name else text)


# =====================================================================
#  EMPTY STATE  — friendly placeholder when a list is empty
# =====================================================================

class EmptyState(QWidget):
    """Friendly empty placeholder shown when a table / list has no items."""

    def __init__(
        self,
        icon: str = "fa5s.inbox",
        title: str = "Chưa có dữ liệu",
        message: str = "",
        action_text: str = "",
        action_icon: str = "",
        on_action=None,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(Spacing.XL, Spacing.XL, Spacing.XL, Spacing.XL)
        layout.setSpacing(Spacing.MD)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        if icon and _HAS_QTA:
            icon_lbl = QLabel()
            icon_lbl.setPixmap(_icon(icon, Colors.TEXT_TERTIARY, size=56).pixmap(56, 56))
            icon_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            icon_lbl.setStyleSheet("background: transparent;")
            layout.addWidget(icon_lbl)

        title_lbl = QLabel(title)
        title_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_font = QFont(Typography.FAMILY.split(",")[0].strip().strip('"'), Typography.SIZE_LG)
        title_font.setWeight(QFont.Weight.DemiBold)
        title_lbl.setFont(title_font)
        title_lbl.setStyleSheet(f"color: {Colors.TEXT_PRIMARY}; background: transparent;")
        layout.addWidget(title_lbl)

        if message:
            msg_lbl = QLabel(message)
            msg_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            msg_lbl.setWordWrap(True)
            msg_lbl.setStyleSheet(f"color: {Colors.TEXT_SECONDARY}; background: transparent;")
            layout.addWidget(msg_lbl)

        if action_text:
            btn = ActionButton(action_text, action_icon or "fa5s.plus", variant="primary")
            if on_action:
                btn.clicked.connect(on_action)
            btn.setParent(self)
            layout.addWidget(btn, alignment=Qt.AlignmentFlag.AlignCenter)