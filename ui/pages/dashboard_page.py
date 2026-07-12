"""Emotion Data Studio - Dashboard page."""

from __future__ import annotations

import json
from pathlib import Path

from PySide6.QtCore import Qt, QTimer, Signal, Slot
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ui.styles.theme import Colors, Spacing, Typography
from ui.widgets import PageHeader, Card, StatusPill, ActionButton, EmptyState

try:
    import qtawesome as qta
    _HAS_QTA = True
except ImportError:
    _HAS_QTA = False


class StatsCard(QFrame):
    def __init__(self, label: str, value: str = "0", color: str = Colors.TEXT_PRIMARY, parent=None):
        super().__init__(parent)
        self.setObjectName("statsCard")
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 10)
        self.name_label = QLabel(label)
        self.name_label.setObjectName("statLabel")
        layout.addWidget(self.name_label)
        self.value_label = QLabel(value)
        self.value_label.setObjectName("statValue")
        self.value_label.setStyleSheet(f"color: {color};")
        layout.addWidget(self.value_label)

    def set_value(self, value: str):
        self.value_label.setText(value)


class DashboardPage(QWidget):
    processing_started = Signal(str)
    processing_worker_started = Signal(object, str)
    active_video_changed = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._worker = None
        self._active_video_id: str | None = None
        self._setup_ui()
        
        # Load last active video project state on start
        self._active_video_id = self._load_project_state()
        
        QTimer.singleShot(100, self.refresh_data)

    def _setup_ui(self):
        page_layout = QVBoxLayout(self)
        page_layout.setContentsMargins(0, 0, 0, 0)
        page_layout.setSpacing(0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        page_layout.addWidget(scroll)

        content = QWidget()
        scroll.setWidget(content)
        self.main_layout = QVBoxLayout(content)
        self.main_layout.setContentsMargins(Spacing.XL, Spacing.LG, Spacing.XL, Spacing.LG)
        self.main_layout.setSpacing(Spacing.LG)

        # ── Hero header ──
        header = PageHeader(
            "Bảng Điều Khiển",
            "Tổng quan dữ liệu — nhập video mới và theo dõi các tác vụ xử lý cảm xúc.",
            icon="fa5s.tachometer-alt",
        )
        # Right side: small GPU / status badge
        self.gpu_badge = StatusPill("info", "Sẵn sàng")
        header.set_right_widget(self.gpu_badge)
        self.main_layout.addWidget(header)

        # ── Stats cards ──
        self._build_stats_row()
        # ── Import + AI section ──
        self._build_import_section()
        # ── Video table ──
        self._build_video_table()
        self.main_layout.addStretch()

    def _build_stats_row(self):
        """5 stat cards arranged horizontally; expands with window width."""
        stats = QHBoxLayout()
        stats.setSpacing(Spacing.MD)
        self.stat_total = StatsCard("Tổng Clip")
        self.stat_approved = StatsCard("Đã duyệt", color=Colors.SUCCESS)
        self.stat_pending = StatsCard("Chờ duyệt", color=Colors.WARNING)
        self.stat_rejected = StatsCard("Từ chối", color=Colors.ERROR)
        self.stat_videos = StatsCard("Video", color=Colors.ACCENT_LIGHT)
        for card in [self.stat_total, self.stat_approved, self.stat_pending, self.stat_rejected, self.stat_videos]:
            stats.addWidget(card, stretch=1)
        self.main_layout.addLayout(stats)

    def _build_import_section(self):
        card = Card("elevated")

        title_row = QHBoxLayout()
        title_row.setSpacing(Spacing.SM)
        section = QLabel("Nhập Video")
        section.setObjectName("sectionTitle")
        from PySide6.QtGui import QFont
        section.setStyleSheet(
            f"color: {Colors.TEXT_PRIMARY}; font-weight: {int(QFont.Weight.DemiBold)};"
            f" background: transparent;"
        )
        title_row.addWidget(section)
        title_row.addStretch()
        card.addLayout(title_row)

        url_row = QHBoxLayout()
        url_row.setSpacing(Spacing.SM)
        self.url_input = QLineEdit()
        self.url_input.setPlaceholderText("Dán URL YouTube / TikTok / Facebook, hoặc dùng nút Chọn File bên cạnh")
        self.url_input.setMinimumHeight(40)
        self.url_input.returnPressed.connect(self._on_process_clicked)
        url_row.addWidget(self.url_input, stretch=1)
        self.import_file_btn = ActionButton("Chọn File", "fa5s.folder-open", variant="secondary")
        self.import_file_btn.setToolTip("Mở dialog chọn file video từ máy (.mp4, .mkv, .avi, .webm, .mov)")
        url_row.addWidget(self.import_file_btn)
        self.process_btn = ActionButton("Nhập & Xử lý", "fa5s.play", variant="primary")
        self.process_btn.setObjectName("primaryBtn")
        self.process_btn.setToolTip("Bắt đầu pipeline: tải video về, cắt scene, nhận diện cảm xúc")
        self.import_file_btn.clicked.connect(self._on_import_file)
        self.process_btn.clicked.connect(self._on_process_clicked)
        url_row.addWidget(self.process_btn)
        card.addLayout(url_row)

        name_row = QHBoxLayout()
        name_row.setSpacing(Spacing.SM)
        name_label = QLabel("Tên dự án:")
        name_label.setStyleSheet(f"color: {Colors.TEXT_SECONDARY}; background: transparent;")
        name_label.setFixedWidth(90)
        name_row.addWidget(name_label)
        self.movie_name_input = QLineEdit()
        self.movie_name_input.setPlaceholderText("Mặc định lấy từ URL / tên file (ví dụ: 'Phim ABC - Tập 01')")
        self.movie_name_input.setMinimumHeight(36)
        name_row.addWidget(self.movie_name_input, stretch=1)
        card.addLayout(name_row)

        # ── AI Auto-Cut toggle (Vertex AI) ──
        ai_row = QHBoxLayout()
        ai_row.setSpacing(Spacing.SM)
        self.ai_autocut_chk = QCheckBox("AI Auto-Cut (Vertex AI Gemini)")
        if _HAS_QTA:
            self.ai_autocut_chk.setIcon(qta.icon("fa5s.robot", color=Colors.ACCENT_LIGHT))
        self.ai_autocut_chk.setToolTip(
            "Khi bật, Vertex AI (Gemini) sẽ tự quét video và cắt trực tiếp "
            "ra các đoạn cảm xúc mạnh. Nếu Vertex AI lỗi sẽ tự fallback về "
            "scene-split + smart-segmenter cũ."
        )
        self.ai_autocut_chk.setChecked(False)
        ai_row.addWidget(self.ai_autocut_chk)
        ai_row.addSpacing(Spacing.MD)
        thresh_lbl = QLabel("Intensity ≥")
        thresh_lbl.setStyleSheet(f"color: {Colors.TEXT_SECONDARY}; background: transparent;")
        ai_row.addWidget(thresh_lbl)
        self.ai_threshold_spin = QDoubleSpinBox()
        self.ai_threshold_spin.setRange(0.3, 1.0)
        self.ai_threshold_spin.setSingleStep(0.05)
        self.ai_threshold_spin.setValue(0.55)
        self.ai_threshold_spin.setDecimals(2)
        self.ai_threshold_spin.setFixedWidth(70)
        ai_row.addWidget(self.ai_threshold_spin)
        self.ai_status_lbl = StatusPill("muted", "Vertex AI: đang kiểm tra…")
        ai_row.addWidget(self.ai_status_lbl, stretch=1)
        card.addLayout(ai_row)
        self.main_layout.addWidget(card)
        QTimer.singleShot(800, self._refresh_ai_status)

    def _build_video_table(self):
        card = Card("default")

        header = QHBoxLayout()
        header.setSpacing(Spacing.SM)
        section = QLabel("Video Đã Nhập")
        from PySide6.QtGui import QFont
        section.setStyleSheet(
            f"color: {Colors.TEXT_PRIMARY}; font-weight: {int(QFont.Weight.DemiBold)};"
            f" background: transparent;"
        )
        header.addWidget(section)

        self.active_project_label = QLabel("Chưa chọn dự án active")
        self.active_project_label.setStyleSheet(
            f"color: {Colors.ACCENT_LIGHT}; background: transparent;"
            f" font-weight: {int(QFont.Weight.Medium)}; padding-left: {Spacing.SM}px;"
        )
        header.addWidget(self.active_project_label)

        header.addStretch()
        refresh = ActionButton("Làm mới", "fa5s.sync-alt", variant="ghost")
        refresh.clicked.connect(self.refresh_data)
        header.addWidget(refresh)
        card.addLayout(header)

        self.video_table = QTableWidget()
        self.video_table.setColumnCount(6)
        self.video_table.setHorizontalHeaderLabels(["#", "Tên Video", "Clip", "Đã duyệt", "Chờ duyệt", "Trạng thái"])
        self.video_table.horizontalHeader().setStretchLastSection(True)
        self.video_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.video_table.verticalHeader().setVisible(False)
        self.video_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.video_table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.video_table.setMinimumHeight(280)
        self.video_table.itemSelectionChanged.connect(self._on_table_selection_changed)
        card.addWidget(self.video_table)

        self.main_layout.addWidget(card)

    @Slot()
    def _on_process_clicked(self):
        source = self.url_input.text().strip()
        if not source:
            QMessageBox.warning(self, "Thiếu nguồn", "Vui lòng nhập URL YouTube hoặc chọn file video.")
            return
        movie_name = self.movie_name_input.text().strip() or Path(source).stem or "Unknown"
        # Apply AI Auto-Cut settings before starting pipeline
        self._apply_ai_autocut_settings()
        self.process_btn.setEnabled(False)
        self.process_btn.setText("Đang xử lý...")
        self.url_input.setEnabled(False)
        self._start_pipeline(source, movie_name)

    def _apply_ai_autocut_settings(self) -> None:
        """Apply AI Auto-Cut toggle + threshold to backend settings."""
        try:
            from backend.config import settings
            settings.AI_AUTOCUT_ENABLED = bool(self.ai_autocut_chk.isChecked())
            settings.AI_AUTOCUT_INTENSITY_THRESHOLD = float(self.ai_threshold_spin.value())
            if settings.AI_AUTOCUT_ENABLED:
                self.process_btn.setText("🤖 AI Auto-Cut đang chạy...")
        except Exception as exc:
            print(f"[Dashboard] Khong set duoc AI_AUTOCUT_* settings: {exc}")

    def _refresh_ai_status(self) -> None:
        """Kiem tra Vertex AI config va cap nhat label."""
        # Guard: neu widget da bi xoa (page closed hoac app shutdown) -> thoat ngay.
        try:
            from shiboken6 import isValid  # type: ignore
            widgets = (self.ai_status_lbl, self.ai_autocut_chk)
            for w in widgets:
                if not isValid(w):
                    return
        except Exception:
            pass
        try:
            from backend.services.gemini_auto_labeler import is_vertex_configured
            ok, msg = is_vertex_configured()
            if ok:
                # msg có dạng: "Vertex AI (global) ready, project=..., key=..."
                short = msg.replace("Vertex AI (global) ready", "Sẵn sàng")
                self.ai_status_lbl.set_status("success", short)
                self.ai_autocut_chk.setEnabled(True)
            else:
                self.ai_status_lbl.set_status("error", "Chưa cấu hình Vertex AI")
                self.ai_autocut_chk.setEnabled(False)
                self.ai_autocut_chk.setChecked(False)
        except RuntimeError:
            return
        except Exception as exc:
            try:
                self.ai_status_lbl.set_status("warning", f"Không kiểm tra được: {exc}")
                self.ai_autocut_chk.setEnabled(False)
            except RuntimeError:
                return

    def _start_pipeline(self, source: str, movie_name: str):
        try:
            from ui.workers.pipeline_worker import PipelineWorker
            self._worker = PipelineWorker(source, movie_name)
            self._worker.pipeline_finished.connect(self._on_pipeline_finished)
            self._worker.error_occurred.connect(self._on_pipeline_error)
            self.processing_worker_started.emit(self._worker, source)
            self.processing_started.emit(source)
            self._worker.start()
        except Exception as exc:
            self._on_pipeline_error(str(exc))

    def _on_import_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Chọn file video",
            "",
            "File Video (*.mp4 *.mkv *.avi *.webm *.mov);;Tất cả (*.*)",
        )
        if path:
            self.url_input.setText(path)
            if not self.movie_name_input.text().strip():
                self.movie_name_input.setText(Path(path).stem)

    @Slot(dict)
    def _on_pipeline_finished(self, result: dict):
        self.process_btn.setEnabled(True)
        self.process_btn.setText("▶ Nhập & xử lý")
        self.url_input.setEnabled(True)
        self.refresh_data()

    @Slot(str)
    def _on_pipeline_error(self, error_msg: str):
        self.process_btn.setEnabled(True)
        self.process_btn.setText("▶ Nhập & xử lý")
        self.url_input.setEnabled(True)
        QMessageBox.critical(self, "Lỗi xử lý", f"Quá trình xử lý thất bại:\n{error_msg}")

    def refresh_data(self):
        try:
            from backend.database.local_db import get_session
            from backend.database.models import Clip, Video

            session = get_session()
            try:
                total = session.query(Clip).count()
                approved = session.query(Clip).filter(Clip.status.in_(["approved", "auto_approved"])).count()
                pending = session.query(Clip).filter(Clip.status.in_(["pending", "needs_review", "ai_labeled"])).count()
                rejected = session.query(Clip).filter(Clip.status == "rejected").count()
                videos = session.query(Video).order_by(Video.created_at.desc()).all()

                self.stat_total.set_value(str(total))
                self.stat_approved.set_value(str(approved))
                self.stat_pending.set_value(str(pending))
                self.stat_rejected.set_value(str(rejected))
                self.stat_videos.set_value(str(len(videos)))

                self.video_table.blockSignals(True)
                self.video_table.setRowCount(len(videos))
                for row, video in enumerate(videos):
                    values = [
                        str(row + 1),
                        video.title or video.movie_name or video.id[:8],
                        str(video.total_clips or 0),
                        str(video.approved_clips or 0),
                        str(max(0, (video.total_clips or 0) - (video.approved_clips or 0))),
                        self._translate_status(video.status or "pending"),
                    ]
                    for col, value in enumerate(values):
                        item = QTableWidgetItem(value)
                        item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                        if col == 0:
                            item.setData(Qt.ItemDataRole.UserRole, video.id)
                        self.video_table.setItem(row, col, item)
                self.video_table.blockSignals(False)

                self._update_active_project_label()

                # Highlight active video row
                if self._active_video_id:
                    self.video_table.blockSignals(True)
                    for row in range(self.video_table.rowCount()):
                        item = self.video_table.item(row, 0)
                        if item and item.data(Qt.ItemDataRole.UserRole) == self._active_video_id:
                            self.video_table.selectRow(row)
                            break
                    self.video_table.blockSignals(False)
            finally:
                session.close()
        except Exception:
            pass

    @staticmethod
    def _translate_status(status: str) -> str:
        return {
            "pending": "Chờ xử lý",
            "processing": "Đang xử lý",
            "completed": "Hoàn thành",
            "error": "Lỗi",
            "cancelled": "Đã hủy",
        }.get(status, status)

    def set_active_video(self, video_id: str | None):
        self._active_video_id = video_id
        self._save_project_state(video_id)
        if hasattr(self, "active_project_label"):
            self._update_active_project_label()
        
        # Highlight row
        if hasattr(self, "video_table"):
            self.video_table.blockSignals(True)
            for row in range(self.video_table.rowCount()):
                item = self.video_table.item(row, 0)
                if item and item.data(Qt.ItemDataRole.UserRole) == video_id:
                    self.video_table.selectRow(row)
                    break
            self.video_table.blockSignals(False)

    def _on_table_selection_changed(self):
        selected_ranges = self.video_table.selectedRanges()
        if not selected_ranges:
            return
        row = selected_ranges[0].topRow()
        item = self.video_table.item(row, 0)
        if item:
            video_id = item.data(Qt.ItemDataRole.UserRole)
            if video_id and video_id != self._active_video_id:
                self._active_video_id = video_id
                self._save_project_state(video_id)
                self._update_active_project_label()
                self.active_video_changed.emit(video_id)

    def _save_project_state(self, video_id: str | None):
        if not video_id:
            return
        try:
            from backend.config import settings
            state_path = settings.DATA_DIR / "project_state.json"
            state = {"last_active_video_id": video_id}
            state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
        except Exception:
            pass

    def _load_project_state(self) -> str | None:
        try:
            from backend.config import settings
            state_path = settings.DATA_DIR / "project_state.json"
            if state_path.exists():
                state = json.loads(state_path.read_text(encoding="utf-8"))
                return state.get("last_active_video_id")
        except Exception:
            pass
        return None

    def _update_active_project_label(self):
        if not hasattr(self, "active_project_label"):
            return
        if not self._active_video_id:
            self.active_project_label.setText("Chưa chọn dự án active")
            return
        try:
            from backend.database.local_db import get_session
            from backend.database.models import Video
            session = get_session()
            try:
                video = session.query(Video).filter(Video.id == self._active_video_id).first()
                if video:
                    title = video.title or video.movie_name or video.id[:8]
                    self.active_project_label.setText(f"Dự án active: {title}")
                else:
                    self.active_project_label.setText("Chưa chọn dự án active")
            except Exception:
                self.active_project_label.setText(f"Dự án active: {self._active_video_id[:8]}...")
            finally:
                session.close()
        except Exception:
            self.active_project_label.setText(f"Dự án active: {self._active_video_id[:8]}...")
