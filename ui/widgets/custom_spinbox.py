from PySide6.QtWidgets import QDoubleSpinBox, QSpinBox
from PySide6.QtCore import Qt

class FocusDoubleSpinBox(QDoubleSpinBox):
    """QDoubleSpinBox that only changes value on scroll when it has focus."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    def wheelEvent(self, event):
        if self.hasFocus():
            super().wheelEvent(event)
        else:
            event.ignore()

class FocusSpinBox(QSpinBox):
    """QSpinBox that only changes value on scroll when it has focus."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    def wheelEvent(self, event):
        if self.hasFocus():
            super().wheelEvent(event)
        else:
            event.ignore()
