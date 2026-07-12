"""
Emotion Data Studio - Export & Sync Manager

Local dataset export UI with quality gate preview, progress, cancel support,
and post-export actions.
"""

from __future__ import annotations

import os
import subprocess
import sys
from collections import Counter
from pathlib import Path

from PySide6.QtCore import Qt, Slot
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QFrame,
    QRadioButton,
    QCheckBox,
    QScrollArea,
    QProgressBar,
    QFileDialog,
    QMessageBox,
    QButtonGroup,
    QPlainTextEdit,
)

from ui.styles.theme import Colors, Spacing
from ui.widgets import PageHeader, Card, ActionButton, StatusPill

try:
    import qtawesome as qta
    _HAS_QTA = True
except ImportError:
    _HAS_QTA = False

from ui.styles.theme import Colors


class ExportPage(QWidget):
    """Export & Cloud Sync Manager page."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._export_worker = None
        self._last_output_path: str | None = None
        self._active_video_id: str | None = None
        self._quality_stats: dict = {}
        self._setup_ui()
        self.refresh_data()

    def _setup_ui(self):
        page_layout = QVBoxLayout(self)
        page_layout.setContentsMargins(0, 0, 0, 0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        page_layout.addWidget(scroll)

        content = QWidget()
        scroll.setWidget(content)
        self.main_layout = QVBoxLayout(content)
        self.main_layout.setContentsMargins(Spacing.XL, Spacing.LG, Spacing.XL, Spacing.LG)
        self.main_layout.setSpacing(Spacing.LG)

        # Hero header
        header = PageHeader(
            "Xuất & Đồng Bộ",
            "Xuất bộ dữ liệu cảm xúc sạch với kiểm tra chất lượng và metadata có thể tái tạo.",
            icon="fa5s.cloud-download-alt",
        )
        self.export_pill = StatusPill("muted", "Sẵn sàng")
        header.set_right_widget(self.export_pill)
        self.main_layout.addWidget(header)

        self._build_quality_gate_section()
        self._build_export_section()
        self._build_output_section()
        self._build_cloud_section()
        self.main_layout.addStretch()

    def _build_quality_gate_section(self):
        card = Card("elevated")

        row = QHBoxLayout()
        row.setSpacing(Spacing.SM)
        title = QLabel("Kiểm Tra Chất Lượng")
        from PySide6.QtGui import QFont
        title.setStyleSheet(
            f"color: {Colors.TEXT_PRIMARY}; font-weight: {int(QFont.Weight.DemiBold)};"
            f" background: transparent;"
        )
        row.addWidget(title)
        row.addStretch()
        self.refresh_stats_btn = ActionButton("Làm mới thống kê", "fa5s.sync-alt", variant="ghost")
        self.refresh_stats_btn.clicked.connect(self.refresh_data)
        row.addWidget(self.refresh_stats_btn)
        card.addLayout(row)

        stats_row = QHBoxLayout()
        stats_row.setSpacing(Spacing.MD)
        self.total_count_label    = self._stat_box(stats_row, "0", "Tổng clip",    Colors.TEXT_PRIMARY)
        self.ready_count_label    = self._stat_box(stats_row, "0", "Xuất được",    Colors.SUCCESS)
        self.rejected_count_label = self._stat_box(stats_row, "0", "Bị loại",     Colors.ERROR)
        self.balance_label        = self._stat_box(stats_row, "0", "Cảm xúc",     Colors.ACCENT_LIGHT)
        self.est_size_label       = self._stat_box(stats_row, "0 MB", "Kích thước ước tính", Colors.INFO)
        stats_row.addStretch()
        card.addLayout(stats_row)

        self.quality_report = QPlainTextEdit()
        self.quality_report.setObjectName("logViewer")
        self.quality_report.setReadOnly(True)
        self.quality_report.setMaximumHeight(160)
        card.addWidget(self.quality_report)
        self.main_layout.addWidget(card)

    def _stat_box(self, parent_layout, value: str, label: str, color: str) -> QLabel:
        col = QVBoxLayout()
        value_label = QLabel(value)
        value_label.setObjectName("statValue")
        value_label.setStyleSheet(f"color: {color};")
        col.addWidget(value_label)
        desc = QLabel(label)
        desc.setObjectName("statLabel")
        col.addWidget(desc)
        parent_layout.addLayout(col)
        return value_label

    def _build_export_section(self):
        card = Card("default")

        from PySide6.QtGui import QFont
        title = QLabel("Xuất Cục Bộ")
        title.setStyleSheet(
            f"color: {Colors.TEXT_PRIMARY}; font-weight: {int(QFont.Weight.DemiBold)};"
            f" background: transparent;"
        )
        card.addWidget(title)
        format_hint = QLabel(
            "Chọn kiểu xuất dataset. Đầy đủ sẽ sao chép clip + audio + annotation + metadata; "
            "metadata gọn phù hợp kiểm tra nhanh; chỉ nhãn dùng cho phân tích nhẹ."
        )
        format_hint.setStyleSheet(f"color: {Colors.TEXT_SECONDARY}; background: transparent;")
        format_hint.setWordWrap(True)
        card.addWidget(format_hint)

        # 3 radio cards grid (cleaner than horizontal row)
        self.format_group = QButtonGroup(self)
        radio_grid = QHBoxLayout()
        radio_grid.setSpacing(Spacing.MD)

        self.radio_full    = QRadioButton("Đầy đủ")
        self.radio_compact = QRadioButton("Metadata gọn")
        self.radio_labels  = QRadioButton("Chỉ nhãn")
        self.radio_compact.setChecked(True)

        radio_descriptions = {
            self.radio_full:    "clip + audio + annotation + metadata",
            self.radio_compact: "nhãn, split, quality report",
            self.radio_labels:  "labels.csv / labels.jsonl",
        }
        for radio, desc in radio_descriptions.items():
            self.format_group.addButton(radio)
            col = QVBoxLayout()
            col.setSpacing(2)
            col.addWidget(radio)
            hint = QLabel(desc)
            hint.setStyleSheet(f"color: {Colors.TEXT_SECONDARY}; background: transparent; padding-left: 22px; font-size: 11px;")
            hint.setWordWrap(True)
            col.addWidget(hint)
            wrap = QFrame()
            wrap.setObjectName("card")
            wrap.setLayout(col)
            wrap.setStyleSheet("background: rgba(255,255,255,0.02); border-radius: 8px; padding: 8px;")
            radio_grid.addWidget(wrap, stretch=1)
        card.addLayout(radio_grid)

        # Checkbox options
        options = QVBoxLayout()
        options.setSpacing(Spacing.XS)
        self.check_approved_only = QCheckBox("Chỉ clip đã duyệt / tự động duyệt")
        self.check_approved_only.setChecked(True)
        self.check_auto_split    = QCheckBox("Tự động chia train/val/test (70/15/15)")
        self.check_auto_split.setChecked(True)
        self.check_stratified    = QCheckBox("Chia phân tầng theo cảm xúc")
        self.check_stratified.setChecked(True)
        for checkbox in (self.check_approved_only, self.check_auto_split, self.check_stratified):
            checkbox.stateChanged.connect(self.refresh_data)
            options.addWidget(checkbox)
        card.addLayout(options)

        self.export_progress = QProgressBar()
        self.export_progress.setObjectName("progressLarge")
        self.export_progress.setValue(0)
        self.export_progress.setVisible(False)
        card.addWidget(self.export_progress)

        self.export_status_label = QLabel("")
        self.export_status_label.setStyleSheet(f"color: {Colors.TEXT_SECONDARY}; background: transparent;")
        self.export_status_label.setVisible(False)
        card.addWidget(self.export_status_label)

        buttons = QHBoxLayout()
        buttons.setSpacing(Spacing.SM)
        self.export_btn = ActionButton("Xuất ra Thư Mục", "fa5s.folder-open", variant="primary")
        self.export_btn.clicked.connect(self._on_export)
        buttons.addWidget(self.export_btn)
        self.cancel_btn = ActionButton("Hủy Xuất", "fa5s.stop", variant="danger")
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.clicked.connect(self._on_cancel_export)
        buttons.addWidget(self.cancel_btn)
        buttons.addStretch()
        self.mmsa_export_btn = ActionButton("Xuất MMSA (.pkl) cho MulT", "fa5s.database", variant="secondary")
        self.mmsa_export_btn.clicked.connect(self._on_mmsa_export)
        buttons.addWidget(self.mmsa_export_btn)
        card.addLayout(buttons)
        self.main_layout.addWidget(card)

    def _build_output_section(self):
        card = QFrame()
        card.setObjectName("card")
        layout = QVBoxLayout(card)
        layout.setSpacing(10)
        title = QLabel("Lần Xuất Cuối")
        title.setObjectName("sectionTitle")
        layout.addWidget(title)
        self.output_path_label = QLabel("Chưa xuất lần nào")
        self.output_path_label.setObjectName("mutedText")
        self.output_path_label.setWordWrap(True)
        layout.addWidget(self.output_path_label)
        actions = QHBoxLayout()
        self.open_folder_btn = QPushButton("Mở thư mục")
        self.open_folder_btn.setEnabled(False)
        self.open_folder_btn.clicked.connect(self._open_last_output)
        actions.addWidget(self.open_folder_btn)
        self.copy_summary_btn = QPushButton("Sao chép tóm tắt")
        self.copy_summary_btn.setEnabled(False)
        self.copy_summary_btn.clicked.connect(self._copy_summary)
        actions.addWidget(self.copy_summary_btn)
        actions.addStretch()
        layout.addLayout(actions)
        self.main_layout.addWidget(card)

    def _build_cloud_section(self):
        card = QFrame()
        card.setObjectName("card")
        layout = QVBoxLayout(card)
        layout.setSpacing(12)

        title = QLabel("Đồng Bộ Đám Mây")
        title.setObjectName("sectionTitle")
        layout.addWidget(title)

        status_note = QLabel(
            "Đồng bộ clips, audio và metadata lên Google Cloud Storage. "
            "Cấu hình credentials trong trang Cài Đặt."
        )
        status_note.setObjectName("mutedText")
        status_note.setWordWrap(True)
        layout.addWidget(status_note)

        self.cloud_status_label = QLabel("Kiểm tra trạng thái...")
        self.cloud_status_label.setObjectName("mutedText")
        layout.addWidget(self.cloud_status_label)

        buttons = QHBoxLayout()
        self.sync_full_btn = QPushButton("Sync Đầy Đủ")
        self.sync_full_btn.setObjectName("primaryBtn")
        self.sync_full_btn.clicked.connect(self._on_sync_full)
        buttons.addWidget(self.sync_full_btn)
        self.sync_files_btn = QPushButton("Sync Files")
        self.sync_files_btn.clicked.connect(self._on_sync_files)
        buttons.addWidget(self.sync_files_btn)
        self.sync_status_btn = QPushButton("Kiểm Tra")
        self.sync_status_btn.clicked.connect(self._on_check_sync_status)
        buttons.addWidget(self.sync_status_btn)
        buttons.addStretch()
        layout.addLayout(buttons)

        self.sync_progress = QProgressBar()
        self.sync_progress.setVisible(False)
        layout.addWidget(self.sync_progress)

        self.sync_log = QPlainTextEdit()
        self.sync_log.setObjectName("logViewer")
        self.sync_log.setReadOnly(True)
        self.sync_log.setMaximumHeight(120)
        layout.addWidget(self.sync_log)

        self.main_layout.addWidget(card)

    # Actions --------------------------------------------------------

    @Slot()
    def _on_export(self):
        self.refresh_data()
        if self._quality_stats.get("exportable", 0) <= 0:
            QMessageBox.warning(
                self,
                "Không có clip xuất được",
                "Không có clip nào qua kiểm tra chất lượng. Hãy duyệt clip và đảm bảo file tồn tại.",
            )
            return

        export_dir = QFileDialog.getExistingDirectory(self, "Chọn thư mục xuất dataset")
        if not export_dir:
            return

        if self.radio_full.isChecked():
            export_format = "full"
        elif self.radio_labels.isChecked():
            export_format = "labels_only"
        else:
            export_format = "compact"

        from ui.workers.export_worker import ExportWorker

        self._export_worker = ExportWorker(
            export_dir=export_dir,
            export_format=export_format,
            approved_only=self.check_approved_only.isChecked(),
            auto_split=self.check_auto_split.isChecked(),
            stratified=self.check_stratified.isChecked(),
            video_id=self._active_video_id,
        )
        self._export_worker.progress_updated.connect(self._on_export_progress)
        self._export_worker.export_finished.connect(self._on_export_finished)
        self._export_worker.error_occurred.connect(self._on_export_error)

        self.export_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        self.export_progress.setVisible(True)
        self.export_status_label.setVisible(True)
        self.export_progress.setValue(0)
        self.export_status_label.setText("Đang chuẩn bị export...")
        self._export_worker.start()

    @Slot()
    def _on_cancel_export(self):
        if self._export_worker is not None and hasattr(self._export_worker, "cancel"):
            self._export_worker.cancel()
        self.cancel_btn.setEnabled(False)
        self.export_status_label.setText("Đã yêu cầu hủy export...")

    @Slot(int, str)
    def _on_export_progress(self, pct: int, msg: str):
        self.export_progress.setValue(pct)
        self.export_status_label.setText(msg)

    @Slot(str)
    def _on_export_finished(self, output_path: str):
        self._last_output_path = output_path
        self.export_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        self.export_progress.setValue(100)
        self.export_status_label.setText(f"Đã xuất tới: {output_path}")
        self.output_path_label.setText(output_path)
        self.open_folder_btn.setEnabled(True)
        self.copy_summary_btn.setEnabled(True)
        QMessageBox.information(self, "Xuất Thành Công", f"Bộ dữ liệu đã xuất thành công:\n\n{output_path}")

    @Slot(str)
    def _on_export_error(self, error_msg: str):
        self.export_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        self.export_status_label.setVisible(True)
        self.export_status_label.setText(f"Lỗi: {error_msg}")
        QMessageBox.critical(self, "Lỗi Xuất", error_msg)

    @Slot()
    def _on_mmsa_export(self):
        from PySide6.QtWidgets import QFileDialog
        output_path, _ = QFileDialog.getSaveFileName(
            self, "Xuất MMSA .pkl", "emotions_dataset.pkl", "Pickle files (*.pkl)"
        )
        if not output_path:
            return

        self.mmsa_export_btn.setEnabled(False)
        self.export_progress.setVisible(True)
        self.export_status_label.setVisible(True)
        self.export_status_label.setText("Đang export MMSA .pkl cho MulT...")

        try:
            from backend.database.local_db import get_session
            from backend.services.exporters.mmsa_exporter import MMSAExporter
            from backend.config import settings

            session = get_session()
            try:
                feature_dir = str(settings.DATA_DIR / "features")
                exporter = MMSAExporter(session)
                result = exporter.export(
                    output_path=output_path,
                    feature_dir=feature_dir,
                    require_aligned=True,
                )
            finally:
                session.close()

            if "error" in result:
                QMessageBox.warning(self, "Export thất bại", result["error"])
                self.export_status_label.setText(f"Lỗi: {result['error']}")
            else:
                summary = (
                    f"✅ Export thành công!\n\n"
                    f"File: {output_path}\n"
                    f"Tổng clip: {result['total_clips']}\n"
                    f"Train: {result['train_clips']} | Val: {result['valid_clips']} | Test: {result['test_clips']}\n"
                    f"Videos: {result['train_videos'] + result['valid_videos'] + result['test_videos']}"
                )
                self.export_status_label.setText(f"✅ Đã export: {result['total_clips']} clips")
                self._last_output_path = str(Path(output_path).parent)
                self.output_path_label.setText(output_path)
                self.open_folder_btn.setEnabled(True)
                QMessageBox.information(self, "Export MMSA Thành Công", summary)
        except Exception as exc:
            QMessageBox.critical(self, "Lỗi", f"Không export được:\n{exc}")
            self.export_status_label.setText(f"Lỗi: {exc}")
        finally:
            self.mmsa_export_btn.setEnabled(True)
            self.export_progress.setValue(0)
            self.export_progress.setVisible(False)
            self.export_status_label.setVisible(False)

    def _open_last_output(self):
        if not self._last_output_path:
            return
        path = Path(self._last_output_path)
        try:
            if os.name == "nt":
                os.startfile(str(path))  # type: ignore[attr-defined]
            elif os.name == "posix":
                subprocess.Popen(["open" if sys.platform == "darwin" else "xdg-open", str(path)])
        except Exception as exc:
            QMessageBox.warning(self, "Không thể mở thư mục", str(exc))

    def _copy_summary(self):
        from PySide6.QtWidgets import QApplication
        QApplication.clipboard().setText(self._build_summary_text())
        self.output_path_label.setText("Đã sao chép tóm tắt vào clipboard")

    # Cloud sync handlers -----------------------------------------------

    def _run_sync_worker(self, sync_type: str, **kwargs):
        from ui.workers.sync_worker import SyncWorker
        if self._export_worker is not None and self._export_worker.isRunning():
            QMessageBox.warning(self, "Đang bận", "Có tiến trình export/sync đang chạy.")
            return
        self._export_worker = SyncWorker(sync_type=sync_type, **kwargs)
        self._export_worker.log_message.connect(self._on_sync_log)
        self._export_worker.progress_updated.connect(self._on_sync_progress)
        self._export_worker.sync_finished.connect(self._on_sync_finished)
        self._export_worker.error_occurred.connect(self._on_export_error)
        self.sync_full_btn.setEnabled(False)
        self.sync_files_btn.setEnabled(False)
        self.sync_progress.setVisible(True)
        self._export_worker.start()

    @Slot()
    def _on_sync_full(self):
        self._run_sync_worker("full", sync_videos=False)

    @Slot()
    def _on_sync_files(self):
        self._run_sync_worker("files", sync_videos=False)

    @Slot(str)
    def _on_sync_log(self, msg: str):
        self.sync_log.appendPlainText(msg)

    @Slot(str, int, int)
    def _on_sync_progress(self, stage: str, current: int, total: int):
        if total > 0:
            self.sync_progress.setValue(int(current / total * 100))

    @Slot(dict)
    def _on_sync_finished(self, report: dict):
        self.sync_full_btn.setEnabled(True)
        self.sync_files_btn.setEnabled(True)
        self.sync_progress.setVisible(False)
        status = report.get("status", "unknown")
        if status == "error":
            self.cloud_status_label.setText(f"❌ Lỗi: {report.get('error', 'unknown')}")
        else:
            meta = report.get("metadata", {})
            files = report.get("files", {})
            meta_errors = meta.get("errors", [])
            file_errors = files.get("errors", [])
            total_errors = len(meta_errors) + len(file_errors)
            if total_errors > 0:
                self.cloud_status_label.setText(
                    f"⚠️ Hoàn tất với {total_errors} lỗi"
                )
            else:
                self.cloud_status_label.setText(
                    f"✅ Hoàn tất. "
                    f"Up: {meta.get('uploaded_videos', 0)} video, "
                    f"{meta.get('uploaded_clips', 0)} clip | "
                    f"Files: {files.get('uploaded_files', 0)}"
                )
        QMessageBox.information(self, "Sync Hoàn Tất", "Cloud sync đã chạy xong.\n\nXem log để biết chi tiết.")

    @Slot()
    def _on_check_sync_status(self):
        try:
            from backend.cloud.sync_manager import SyncManager
            manager = SyncManager()
            avail = manager.is_available
            last = manager.get_sync_status()
            if not avail:
                self.cloud_status_label.setText(
                    "⚠️ Cloud sync chưa cấu hình. "
                    "Đặt GOOGLE_APPLICATION_CREDENTIALS và GCS_BUCKET_NAME trong Settings."
                )
            else:
                last_time = last.get("last_sync") or "chưa bao giờ"
                last_status = last.get("status", "unknown")
                self.cloud_status_label.setText(
                    f"✅ Cloud sync sẵn sàng. Last sync: {last_time} ({last_status})"
                )
        except Exception as exc:
            self.cloud_status_label.setText(f"❌ Lỗi kiểm tra: {exc}")

    # Data -----------------------------------------------------------

    def set_active_video(self, video_id: str | None):
        self._active_video_id = video_id
        self.refresh_data()

    def refresh_data(self):
        try:
            from backend.database.local_db import get_session
            from backend.database.models import Clip

            session = get_session()
            try:
                query = session.query(Clip)
                if self._active_video_id:
                    query = query.filter(Clip.video_id == self._active_video_id)
                clips = query.all()
            finally:
                session.close()

            stats = self._calculate_quality_stats(clips)
            self._quality_stats = stats
            self.total_count_label.setText(f"{stats['total']:,}")
            self.ready_count_label.setText(f"{stats['exportable']:,}")
            self.rejected_count_label.setText(f"{stats['rejected']:,}")
            self.balance_label.setText(str(stats["emotion_count"]))
            self.est_size_label.setText(self._format_size(stats["estimated_full_bytes"]))
            self.quality_report.setPlainText(self._build_quality_report(stats))
        except Exception as exc:
            if hasattr(self, "quality_report"):
                self.quality_report.setPlainText(f"Không thể tải thống kê export: {exc}")

    def _calculate_quality_stats(self, clips) -> dict:
        reasons = Counter()
        emotions = Counter()
        exportable = 0
        estimated_size = 0
        
        approved_only = self.check_approved_only.isChecked() if hasattr(self, "check_approved_only") else True
        
        for clip in clips:
            clip_reasons = []
            
            # Check if approved if approved_only filter is active
            if approved_only and clip.status not in ("approved", "auto_approved"):
                clip_reasons.append("Chưa duyệt thủ công")
                
            label = clip.user_emotion or clip.ai_emotion
            if not label:
                clip_reasons.append("Thiếu nhãn cảm xúc")
                
            if not clip.clip_path or not Path(clip.clip_path).exists():
                clip_reasons.append("Thiếu file video (clip)")
                
            if not clip.duration or clip.duration <= 0:
                clip_reasons.append("Thời lượng không hợp lệ")
                
            if clip_reasons:
                for r in clip_reasons:
                    reasons[r] += 1
            else:
                exportable += 1
                emotions[label] += 1
                try:
                    estimated_size += Path(clip.clip_path).stat().st_size
                except Exception:
                    estimated_size += 2 * 1024 * 1024
                    
        return {
            "total": len(clips),
            "exportable": exportable,
            "rejected": len(clips) - exportable,
            "reasons": reasons,
            "emotions": emotions,
            "emotion_count": len(emotions),
            "estimated_full_bytes": estimated_size,
        }

    def _build_quality_report(self, stats: dict) -> str:
        lines = [
            f"Tổng số phân đoạn xem xét: {stats['total']}",
            f"Số phân đoạn đủ điều kiện xuất: {stats['exportable']}",
            f"Số phân đoạn bị loại / chưa sẵn sàng: {stats['rejected']}",
            "",
            "Lý do chi tiết:",
        ]
        if stats["reasons"]:
            for reason, count in stats["reasons"].most_common():
                lines.append(f"- {reason}: {count} clip")
        else:
            lines.append("- Không có")
        lines.extend(["", "Cân bằng nhãn cảm xúc xuất được:"])
        if stats["emotions"]:
            for emotion, count in sorted(stats["emotions"].items()):
                lines.append(f"- {emotion}: {count} clip")
        else:
            lines.append("- Không có clip nào có nhãn đủ điều kiện xuất")
        return "\n".join(lines)

    def _build_summary_text(self) -> str:
        return "\n".join([
            "Tóm tắt export Emotion Data Studio",
            f"Output: {self._last_output_path or 'N/A'}",
            self.quality_report.toPlainText(),
        ])

    @staticmethod
    def _format_size(num_bytes: int) -> str:
        if num_bytes >= 1024 ** 3:
            return f"{num_bytes / (1024 ** 3):.1f} GB"
        if num_bytes >= 1024 ** 2:
            return f"{num_bytes / (1024 ** 2):.1f} MB"
        return f"{num_bytes / 1024:.1f} KB"
