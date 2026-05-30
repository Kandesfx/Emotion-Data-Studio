"""
Emotion Data Studio — Export & Sync Page
=========================================
Export dataset and manage cloud synchronization:
  - Export format selection (Full / Compact / Labels Only)
  - Filter options (approved only, train/val/test split)
  - Export progress
  - Cloud sync status (Google Cloud)
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFrame, QRadioButton, QCheckBox, QGroupBox, QScrollArea,
    QProgressBar, QComboBox, QFileDialog, QMessageBox,
    QSizePolicy, QButtonGroup
)
from PySide6.QtCore import Qt, Signal, Slot, QTimer

from ui.styles.theme import Colors


class ExportPage(QWidget):
    """Export & Cloud Sync Manager page"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)

        scroll_content = QWidget()
        self.main_layout = QVBoxLayout(scroll_content)
        self.main_layout.setContentsMargins(32, 24, 32, 24)
        self.main_layout.setSpacing(24)

        page_layout = QVBoxLayout(self)
        page_layout.setContentsMargins(0, 0, 0, 0)
        page_layout.addWidget(scroll)
        scroll.setWidget(scroll_content)

        # --- Header ---
        header = QVBoxLayout()
        header.setSpacing(4)

        title = QLabel("📦 Export & Sync Manager")
        title.setObjectName("pageTitle")
        header.addWidget(title)

        subtitle = QLabel("Xuất dataset và đồng bộ lên Google Cloud")
        subtitle.setObjectName("pageSubtitle")
        header.addWidget(subtitle)

        self.main_layout.addLayout(header)

        # --- Export Local Section ---
        self._build_export_section()

        # --- Export Stats ---
        self._build_stats_section()

        # --- Cloud Sync Section ---
        self._build_cloud_section()

        self.main_layout.addStretch()

    def _build_export_section(self):
        """Local export configuration"""
        export_card = QFrame()
        export_card.setObjectName("card")

        card_layout = QVBoxLayout(export_card)
        card_layout.setSpacing(12)

        section_title = QLabel("📁 Export Local")
        section_title.setObjectName("sectionTitle")
        card_layout.addWidget(section_title)

        # Format selection
        format_label = QLabel("Export Format:")
        format_label.setObjectName("statLabel")
        card_layout.addWidget(format_label)

        self.format_group = QButtonGroup(self)

        format_row = QHBoxLayout()
        format_row.setSpacing(16)

        self.radio_full = QRadioButton("Full (video + audio + text + labels)")
        self.format_group.addButton(self.radio_full)
        format_row.addWidget(self.radio_full)

        self.radio_compact = QRadioButton("Compact (frames + MFCC + tokens + labels)")
        self.radio_compact.setChecked(True)
        self.format_group.addButton(self.radio_compact)
        format_row.addWidget(self.radio_compact)

        self.radio_labels = QRadioButton("Labels Only (CSV)")
        self.format_group.addButton(self.radio_labels)
        format_row.addWidget(self.radio_labels)

        card_layout.addLayout(format_row)

        # Filter options
        filter_label = QLabel("Filter Options:")
        filter_label.setObjectName("statLabel")
        card_layout.addWidget(filter_label)

        filter_row = QVBoxLayout()
        filter_row.setSpacing(6)

        self.check_approved_only = QCheckBox("Only approved clips")
        self.check_approved_only.setChecked(True)
        filter_row.addWidget(self.check_approved_only)

        self.check_auto_split = QCheckBox("Auto-split train/val/test (70/15/15)")
        self.check_auto_split.setChecked(True)
        filter_row.addWidget(self.check_auto_split)

        self.check_stratified = QCheckBox("Stratified (balance emotions)")
        self.check_stratified.setChecked(True)
        filter_row.addWidget(self.check_stratified)

        card_layout.addLayout(filter_row)

        # Export progress
        self.export_progress = QProgressBar()
        self.export_progress.setObjectName("progressLarge")
        self.export_progress.setValue(0)
        self.export_progress.setVisible(False)
        card_layout.addWidget(self.export_progress)

        self.export_status_label = QLabel("")
        self.export_status_label.setObjectName("mutedText")
        self.export_status_label.setVisible(False)
        card_layout.addWidget(self.export_status_label)

        # Export button
        btn_row = QHBoxLayout()
        btn_row.addStretch()

        self.export_btn = QPushButton("📁 Export to Local Folder")
        self.export_btn.setObjectName("primaryBtn")
        self.export_btn.setMinimumWidth(200)
        self.export_btn.setMinimumHeight(40)
        self.export_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.export_btn.clicked.connect(self._on_export)
        btn_row.addWidget(self.export_btn)

        card_layout.addLayout(btn_row)

        self.main_layout.addWidget(export_card)

    def _build_stats_section(self):
        """Export statistics"""
        stats_card = QFrame()
        stats_card.setObjectName("card")

        card_layout = QHBoxLayout(stats_card)
        card_layout.setSpacing(24)

        # Ready clips
        ready_col = QVBoxLayout()
        self.ready_count_label = QLabel("0")
        self.ready_count_label.setObjectName("statValue")
        self.ready_count_label.setStyleSheet(f"color: {Colors.SUCCESS};")
        ready_col.addWidget(self.ready_count_label)
        ready_desc = QLabel("Clips Ready")
        ready_desc.setObjectName("statLabel")
        ready_col.addWidget(ready_desc)
        card_layout.addLayout(ready_col)

        # Estimated size
        size_col = QVBoxLayout()
        self.est_size_label = QLabel("0 MB")
        self.est_size_label.setObjectName("statValue")
        size_col.addWidget(self.est_size_label)
        size_desc = QLabel("Est. Size")
        size_desc.setObjectName("statLabel")
        size_col.addWidget(size_desc)
        card_layout.addLayout(size_col)

        # Emotion balance
        balance_col = QVBoxLayout()
        self.balance_label = QLabel("N/A")
        self.balance_label.setObjectName("statValue")
        self.balance_label.setStyleSheet(f"color: {Colors.ACCENT_LIGHT};")
        balance_col.addWidget(self.balance_label)
        balance_desc = QLabel("Emotions")
        balance_desc.setObjectName("statLabel")
        balance_col.addWidget(balance_desc)
        card_layout.addLayout(balance_col)

        card_layout.addStretch()

        self.main_layout.addWidget(stats_card)

    def _build_cloud_section(self):
        """Cloud sync section"""
        cloud_card = QFrame()
        cloud_card.setObjectName("card")

        card_layout = QVBoxLayout(cloud_card)
        card_layout.setSpacing(12)

        section_title = QLabel("☁️ Sync to Cloud")
        section_title.setObjectName("sectionTitle")
        card_layout.addWidget(section_title)

        # Cloud status
        status_row = QHBoxLayout()

        self.cloud_status_label = QLabel("⚠️ Not Connected")
        self.cloud_status_label.setObjectName("warningText")
        status_row.addWidget(self.cloud_status_label)

        status_row.addStretch()

        self.cloud_settings_btn = QPushButton("⚙️ Cloud Settings")
        self.cloud_settings_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        status_row.addWidget(self.cloud_settings_btn)

        card_layout.addLayout(status_row)

        # Sync info
        info_row = QHBoxLayout()
        info_row.setSpacing(24)

        self.last_sync_label = QLabel("Last sync: Never")
        self.last_sync_label.setObjectName("mutedText")
        info_row.addWidget(self.last_sync_label)

        self.pending_changes_label = QLabel("Pending changes: 0")
        self.pending_changes_label.setObjectName("mutedText")
        info_row.addWidget(self.pending_changes_label)

        info_row.addStretch()
        card_layout.addLayout(info_row)

        # Sync options
        self.check_sync_metadata = QCheckBox("Metadata → Cloud SQL PostgreSQL")
        self.check_sync_metadata.setChecked(True)
        card_layout.addWidget(self.check_sync_metadata)

        self.check_sync_files = QCheckBox("Processed files → Cloud Storage")
        self.check_sync_files.setChecked(True)
        card_layout.addWidget(self.check_sync_files)

        self.check_sync_raw = QCheckBox("Raw videos → Cloud Storage (⚠️ large files)")
        card_layout.addWidget(self.check_sync_raw)

        # Sync button
        btn_row = QHBoxLayout()
        btn_row.addStretch()

        self.sync_btn = QPushButton("🔄 Sync Now")
        self.sync_btn.setObjectName("primaryBtn")
        self.sync_btn.setMinimumWidth(150)
        self.sync_btn.setEnabled(False)  # Disabled until cloud connected
        self.sync_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_row.addWidget(self.sync_btn)

        card_layout.addLayout(btn_row)

        self.main_layout.addWidget(cloud_card)

    # ================================================================
    # ACTIONS
    # ================================================================

    @Slot()
    def _on_export(self):
        """Handle export button click"""
        # Choose export directory
        export_dir = QFileDialog.getExistingDirectory(
            self, "Chọn thư mục xuất dataset"
        )
        if not export_dir:
            return

        # Determine format
        if self.radio_full.isChecked():
            export_format = "full"
        elif self.radio_compact.isChecked():
            export_format = "compact"
        else:
            export_format = "labels_only"

        # Start export
        self.export_btn.setEnabled(False)
        self.export_btn.setText("⏳ Đang xuất...")
        self.export_progress.setVisible(True)
        self.export_status_label.setVisible(True)
        self.export_progress.setValue(0)
        self.export_status_label.setText("Preparing export...")

        # Run export in background thread
        from ui.workers.export_worker import ExportWorker
        self._export_worker = ExportWorker(
            export_dir=export_dir,
            export_format=export_format,
            approved_only=self.check_approved_only.isChecked(),
            auto_split=self.check_auto_split.isChecked(),
            stratified=self.check_stratified.isChecked(),
        )
        self._export_worker.progress_updated.connect(self._on_export_progress)
        self._export_worker.export_finished.connect(self._on_export_finished)
        self._export_worker.error_occurred.connect(self._on_export_error)
        self._export_worker.start()

    @Slot(int, str)
    def _on_export_progress(self, pct: int, msg: str):
        self.export_progress.setValue(pct)
        self.export_status_label.setText(msg)

    @Slot(str)
    def _on_export_finished(self, output_path: str):
        self.export_btn.setEnabled(True)
        self.export_btn.setText("📁 Export to Local Folder")
        self.export_progress.setValue(100)
        self.export_status_label.setText(f"✅ Exported to: {output_path}")

        QMessageBox.information(
            self, "Export Complete",
            f"Dataset đã được xuất thành công!\n\nĐường dẫn: {output_path}"
        )

    @Slot(str)
    def _on_export_error(self, error_msg: str):
        self.export_btn.setEnabled(True)
        self.export_btn.setText("📁 Export to Local Folder")
        self.export_progress.setVisible(False)
        self.export_status_label.setText(f"❌ Error: {error_msg}")

        QMessageBox.critical(self, "Export Error", f"Lỗi khi xuất:\n{error_msg}")

    # ================================================================
    # DATA
    # ================================================================

    def refresh_data(self):
        """Refresh export statistics"""
        try:
            from backend.database.local_db import get_session
            from backend.database.models import Clip

            session = get_session()
            try:
                ready = session.query(Clip).filter(
                    Clip.status.in_(['approved', 'auto_approved'])
                ).count()
                self.ready_count_label.setText(f"{ready:,}")

                # Estimate size (rough: ~2MB per clip for compact)
                est_mb = ready * 2
                if est_mb > 1024:
                    self.est_size_label.setText(f"{est_mb/1024:.1f} GB")
                else:
                    self.est_size_label.setText(f"{est_mb} MB")

                # Count distinct emotions
                from sqlalchemy import func
                emotions = session.query(
                    func.count(func.distinct(Clip.ai_emotion))
                ).filter(
                    Clip.status.in_(['approved', 'auto_approved'])
                ).scalar()
                self.balance_label.setText(f"{emotions or 0}")

            finally:
                session.close()
        except Exception:
            pass
