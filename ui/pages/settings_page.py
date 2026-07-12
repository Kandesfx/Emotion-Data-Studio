"""
Emotion Data Studio - Settings Page

User-facing configuration for paths, runtime, downloader, segmentation defaults
and preflight diagnostics. Settings are stored in data/user_settings.json and
loaded by backend.config on app startup.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QFrame,
    QLineEdit,
    QFileDialog,
    QComboBox,
    QPlainTextEdit,
    QMessageBox,
    QScrollArea,
)

from ui.widgets.custom_spinbox import FocusDoubleSpinBox
from ui.styles.theme import Colors, Spacing
from ui.widgets import PageHeader, Card, ActionButton, StatusPill

try:
    import qtawesome as qta
    _HAS_QTA = True
except ImportError:
    _HAS_QTA = False


class SettingsPage(QWidget):
    """Application settings and diagnostics page."""

    settings_saved = Signal(dict)
    update_check_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()
        self.load_settings()

    def set_responsive_mode(self, stacked: bool):
        """Stack 2-column grid thành 1-column khi window < 900px."""
        try:
            if hasattr(self, '_basic_grid_wrap_l'):
                if stacked:
                    # Move right column children up to top-level layout
                    # Easiest: rebuild layout. Hide the grid instead.
                    self._basic_grid_wrap_l.setVisible(False)
                    self._basic_grid_wrap_r.setVisible(False)
                    # Show all cards at top level
                    if hasattr(self, '_stacked_cards'):
                        for w in self._stacked_cards:
                            w.setVisible(True)
                else:
                    self._basic_grid_wrap_l.setVisible(True)
                    self._basic_grid_wrap_r.setVisible(True)
                    if hasattr(self, '_stacked_cards'):
                        for w in self._stacked_cards:
                            w.setVisible(False)
        except Exception:
            pass

    def _setup_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        root.addWidget(scroll)

        content = QWidget()
        scroll.setWidget(content)
        layout = QVBoxLayout(content)
        layout.setContentsMargins(Spacing.XL, Spacing.LG, Spacing.XL, Spacing.LG)
        layout.setSpacing(Spacing.LG)

        # Hero header
        header = PageHeader(
            "Cài Đặt",
            "FFmpeg, đường dẫn dữ liệu, runtime AI, Vertex AI và các tùy chọn pipeline.",
            icon="fa5s.sliders-h",
        )
        self.settings_pill = StatusPill("info", "Chưa lưu thay đổi")
        header.set_right_widget(self.settings_pill)
        layout.addWidget(header)

        # AI / Vertex AI section first (highlighted as it's the premium feature)
        layout.addWidget(self._build_vertex_ai_card())

        # Hidden stacked cards (visible khi responsive stacked mode)
        self._stacked_cards = [
            self._build_paths_card(),
            self._build_performance_card(),
            self._build_download_card(),
            self._build_pipeline_card(),
        ]
        for w in self._stacked_cards:
            w.setVisible(False)
            layout.addWidget(w)

        # 2-column grid for basic settings (compact)
        col_l = QVBoxLayout()
        col_l.setSpacing(Spacing.LG)
        col_l.addWidget(self._stacked_cards[0])  # paths
        col_l.addWidget(self._stacked_cards[1])  # performance
        col_l.addStretch()
        col_r = QVBoxLayout()
        col_r.setSpacing(Spacing.LG)
        col_r.addWidget(self._stacked_cards[2])  # download
        col_r.addWidget(self._stacked_cards[3])  # pipeline
        col_r.addStretch()

        self._basic_grid_wrap_l = QWidget()
        self._basic_grid_wrap_l.setLayout(col_l)
        self._basic_grid_wrap_l.setStyleSheet("background: transparent;")
        self._basic_grid_wrap_r = QWidget()
        self._basic_grid_wrap_r.setLayout(col_r)
        self._basic_grid_wrap_r.setStyleSheet("background: transparent;")
        grid_row = QHBoxLayout()
        grid_row.setSpacing(Spacing.LG)
        grid_row.addWidget(self._basic_grid_wrap_l, stretch=1)
        grid_row.addWidget(self._basic_grid_wrap_r, stretch=1)
        layout.addLayout(grid_row)

        layout.addWidget(self._build_smart_segmentation_card())
        layout.addWidget(self._build_cloud_card())
        layout.addWidget(self._build_diagnostics_card())
        layout.addStretch()

    def _build_paths_card(self) -> QFrame:
        card = QFrame()
        card.setObjectName("cardElevated")
        layout = QVBoxLayout(card)
        layout.setSpacing(12)

        title = QLabel("Đường Dẫn")
        title.setObjectName("sectionTitle")
        layout.addWidget(title)

        self.ffmpeg_input = self._path_row(layout, "Đường dẫn FFmpeg", browse_file=True)
        self.data_dir_input = self._path_row(layout, "Thư mục dữ liệu", browse_dir=True)
        self.model_cache_input = self._path_row(layout, "Thư mục cache mô hình", browse_dir=True)
        return card

    def _build_performance_card(self) -> QFrame:
        card = QFrame()
        card.setObjectName("card")
        layout = QVBoxLayout(card)
        layout.setSpacing(10)
        title = QLabel("Hiệu Năng & Tài Nguyên")
        title.setObjectName("sectionTitle")
        layout.addWidget(title)
        hint = QLabel("Để 0 là tự động. Nên để CPU còn dư 1–2 luồng để giao diện không bị đứng khi xử lý video dài.")
        hint.setObjectName("mutedText")
        hint.setWordWrap(True)
        layout.addWidget(hint)
        self.cpu_threads = self._double_row(layout, "Số luồng CPU cho AI/tính toán", 0, 128, 0, 1)
        self.ffmpeg_threads = self._double_row(layout, "Số luồng FFmpeg", 0, 128, 0, 1)
        self.pipeline_workers = self._double_row(layout, "Số worker pipeline song song", 1, 8, 1, 1)
        return card

    def _build_download_card(self) -> QFrame:
        card = QFrame()
        card.setObjectName("cardElevated")
        layout = QVBoxLayout(card)
        layout.setSpacing(12)

        title = QLabel("Tải Video Từ URL")
        title.setObjectName("sectionTitle")
        layout.addWidget(title)
        note = QLabel(
            "Cấu hình profile tải, giới hạn độ phân giải, downloader ngoài và cookies cho YouTube/yt-dlp. "
            "Các thay đổi áp dụng cho lần tải tiếp theo."
        )
        note.setObjectName("mutedText")
        note.setWordWrap(True)
        layout.addWidget(note)

        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel("Chế độ tải"))
        self.download_mode_combo = QComboBox()
        self.download_mode_combo.addItems(["balanced", "safe", "turbo"])
        self.download_mode_combo.setToolTip("balanced: cân bằng; safe: ưu tiên ổn định; turbo: ưu tiên tốc độ")
        mode_row.addWidget(self.download_mode_combo, stretch=1)
        layout.addLayout(mode_row)

        self.download_max_height = self._double_row(layout, "Độ phân giải tối đa", 360, 1080, 720, 120)
        self.download_concurrent_fragments = self._double_row(layout, "Số phân đoạn tải song song", 1, 16, 5, 1)
        self.download_throttled_rate_kbps = self._double_row(layout, "Ngưỡng throttling để nối lại (KB/s)", 0, 10240, 100, 10)

        aria_row = QHBoxLayout()
        aria_row.addWidget(QLabel("Dùng aria2c nếu có"))
        self.download_use_aria2_combo = QComboBox()
        self.download_use_aria2_combo.addItems(["false", "true"])
        aria_row.addWidget(self.download_use_aria2_combo, stretch=1)
        layout.addLayout(aria_row)

        browser_row = QHBoxLayout()
        browser_row.addWidget(QLabel("Cookies từ trình duyệt"))
        self.download_cookies_browser_combo = QComboBox()
        self.download_cookies_browser_combo.addItems(["", "chrome", "edge", "firefox"])
        browser_row.addWidget(self.download_cookies_browser_combo, stretch=1)
        layout.addLayout(browser_row)

        self.download_cookie_file_input = self._path_row(layout, "Đường dẫn cookies.txt", browse_file=True)
        return card

    def _build_pipeline_card(self) -> QFrame:
        card = QFrame()
        card.setObjectName("cardElevated")
        layout = QVBoxLayout(card)
        layout.setSpacing(12)

        title = QLabel("Mặc Định Pipeline")
        title.setObjectName("sectionTitle")
        layout.addWidget(title)

        runtime_row = QHBoxLayout()
        runtime_row.addWidget(QLabel("Chế độ chạy"))
        self.runtime_combo = QComboBox()
        self.runtime_combo.addItems(["auto", "cpu", "cuda"])
        runtime_row.addWidget(self.runtime_combo, stretch=1)
        layout.addLayout(runtime_row)

        self.scene_threshold = self._double_row(layout, "Ngưỡng tách cảnh", 1.0, 100.0, 30.0, 1.0)
        self.min_duration = self._double_row(layout, "Thời lượng clip tối thiểu", 0.5, 60.0, 3.0, 0.5)
        self.max_duration = self._double_row(layout, "Thời lượng clip tối đa", 1.0, 300.0, 15.0, 0.5)

        actions = QHBoxLayout()
        actions.setSpacing(Spacing.SM)
        self.save_btn = ActionButton("Lưu cài đặt", "fa5s.save", variant="primary")
        self.save_btn.clicked.connect(self.save_settings)
        actions.addWidget(self.save_btn)
        self.reload_btn = ActionButton("Tải lại", "fa5s.undo", variant="ghost")
        self.reload_btn.clicked.connect(self.load_settings)
        actions.addWidget(self.reload_btn)
        actions.addStretch()
        layout.addLayout(actions)
        return card

    def _build_smart_segmentation_card(self) -> QFrame:
        card = QFrame()
        card.setObjectName("cardElevated")
        layout = QVBoxLayout(card)
        layout.setSpacing(12)

        title = QLabel("Cắt Video Thông Minh")
        title.setObjectName("sectionTitle")
        layout.addWidget(title)
        note = QLabel("Tinh chỉnh cách hệ thống ưu tiên đoạn có khuôn mặt và hội thoại. Các thay đổi sẽ áp dụng cho lần chạy pipeline tiếp theo.")
        note.setObjectName("mutedText")
        note.setWordWrap(True)
        layout.addWidget(note)

        vad_row = QHBoxLayout()
        vad_row.addWidget(QLabel("Chế độ nhận diện hội thoại"))
        self.smart_vad_mode = QComboBox()
        self.smart_vad_mode.addItems(["auto", "energy", "ffmpeg", "disabled"])
        self.smart_vad_mode.setToolTip("auto: ưu tiên VAD năng lượng rồi fallback FFmpeg; energy: chỉ dùng VAD năng lượng; ffmpeg: dùng silencedetect; disabled: bỏ qua hội thoại")
        vad_row.addWidget(self.smart_vad_mode, stretch=1)
        layout.addLayout(vad_row)

        self.smart_face_scan_fps = self._double_row(layout, "FPS quét khuôn mặt", 0.5, 10.0, 2.0, 0.5)
        self.smart_face_confidence = self._double_row(layout, "Ngưỡng tin cậy khuôn mặt", 0.1, 0.99, 0.55, 0.05)
        self.smart_max_missing_face_gap = self._double_row(layout, "Khoảng mất mặt tối đa để vẫn nối đoạn (giây)", 0.0, 5.0, 1.0, 0.1)
        self.smart_target_clip_duration = self._double_row(layout, "Thời lượng clip mục tiêu (giây)", 2.0, 30.0, 6.0, 0.5)
        self.smart_silence_min_duration = self._double_row(layout, "Khoảng im lặng tối thiểu (giây)", 0.1, 3.0, 0.45, 0.05)
        self.smart_max_dialogue_extension = self._double_row(layout, "Mở rộng theo hội thoại tối đa (giây)", 0.0, 5.0, 1.5, 0.1)

        silence_row = QHBoxLayout()
        silence_row.addWidget(QLabel("Ngưỡng im lặng FFmpeg"))
        self.smart_silence_threshold_db = QLineEdit()
        self.smart_silence_threshold_db.setPlaceholderText("Ví dụ: -35dB")
        silence_row.addWidget(self.smart_silence_threshold_db, stretch=1)
        layout.addLayout(silence_row)
        return card

    def _build_vertex_ai_card(self) -> QFrame:
        card = QFrame()
        card.setObjectName("cardElevated")
        layout = QVBoxLayout(card)
        layout.setSpacing(12)

        title = QLabel("Vertex AI — Gemini Auto-Cut")
        title.setObjectName("sectionTitle")
        layout.addWidget(title)
        note = QLabel(
            "Cấu hình kết nối Vertex AI (Gemini) và bật/tắt chế độ AI tự cắt video. "
            "Service account key phải có quyền 'Agent Platform User' trên project 'aura-social-vn'."
        )
        note.setObjectName("mutedText")
        note.setWordWrap(True)
        layout.addWidget(note)

        self.vertex_location_input = self._path_row(
            layout, "Vertex Location (global)", placeholder="global"
        )
        self.gemini_model_input = self._path_row(
            layout, "Gemini model", placeholder="gemini-2.5-flash"
        )
        self.gemini_api_key_input = self._path_row(
            layout, "Gemini API Key (tùy chọn, để dùng AI Studio thay Vertex)",
            placeholder="AIza..."
        )
        self.gac_path_input = self._path_row(
            layout, "GOOGLE_APPLICATION_CREDENTIALS (Service Account JSON)",
            browse_file=True, placeholder="aura-social-vn-*.json"
        )

        autocut_row = QHBoxLayout()
        autocut_row.addWidget(QLabel("Bật AI Auto-Cut mặc định"))
        self.ai_autocut_combo = QComboBox()
        self.ai_autocut_combo.addItems(["false", "true"])
        self.ai_autocut_combo.setToolTip(
            "Khi bật, Vertex AI sẽ thay thế stage scene_split + smart_segmenter. "
            "Pipeline tự fallback nếu Vertex AI lỗi."
        )
        autocut_row.addWidget(self.ai_autocut_combo, stretch=1)
        layout.addLayout(autocut_row)

        self.ai_autocut_threshold = self._double_row(
            layout, "Intensity threshold (0-1)", 0.3, 1.0, 0.55, 0.05
        )
        self.ai_autocut_min_duration = self._double_row(
            layout, "Thời lượng clip tối thiểu (giây)", 1.0, 30.0, 3.0, 0.5
        )
        self.ai_autocut_max_duration = self._double_row(
            layout, "Thời lượng clip tối đa (giây)", 5.0, 60.0, 15.0, 0.5
        )
        self.ai_autocut_max_segments = self._double_row(
            layout, "Số segment tối đa mỗi video", 5, 80, 40, 1
        )
        # Sprint 2 + 3 — Verify pass settings
        self.ai_autocut_verify_strict_combo = QComboBox()
        self.ai_autocut_verify_strict_combo.addItems(["false", "true"])
        self.ai_autocut_verify_strict_combo.setToolTip(
            "Khi true: clip bị Verify reject (unstable/low_quality/emotion_flip) "
            "sẽ bị loại khỏi output. Khi false: giữ lại với flag."
        )
        strict_row = QHBoxLayout()
        strict_row.addWidget(QLabel("Verify strict mode"))
        strict_row.addWidget(self.ai_autocut_verify_strict_combo, stretch=1)
        layout.addLayout(strict_row)
        self.ai_autocut_min_face_coverage = self._double_row(
            layout, "Min face coverage scene", 0.50, 0.95, 0.70, 0.05
        )

        # Trang thai
        check_row = QHBoxLayout()
        check_row.addWidget(QLabel("Trạng thái"))
        self.vertex_status_lbl = QLabel("Chưa kiểm tra")
        self.vertex_status_lbl.setObjectName("mutedText")
        check_row.addWidget(self.vertex_status_lbl, stretch=1)
        self.vertex_check_btn = QPushButton("Kiểm tra ngay")
        self.vertex_check_btn.clicked.connect(self._check_vertex_status)
        check_row.addWidget(self.vertex_check_btn)
        layout.addLayout(check_row)
        return card

    def _check_vertex_status(self) -> None:
        """Kiem tra ngay Vertex AI config va cap nhat status label."""
        try:
            from backend.services.gemini_auto_labeler import is_vertex_configured
            ok, msg = is_vertex_configured()
            if ok:
                self.vertex_status_lbl.setText(f"✅ {msg}")
                self.vertex_status_lbl.setStyleSheet("color: #00b894;")
            else:
                self.vertex_status_lbl.setText(f"❌ {msg}")
                self.vertex_status_lbl.setStyleSheet("color: #e17055;")
        except Exception as exc:
            self.vertex_status_lbl.setText(f"⚠️ {exc}")
            self.vertex_status_lbl.setStyleSheet("color: #9896a8;")

    def _build_cloud_card(self) -> QFrame:
        card = QFrame()
        card.setObjectName("card")
        layout = QVBoxLayout(card)
        layout.setSpacing(12)

        title = QLabel("Đồng Bộ Cloud")
        title.setObjectName("sectionTitle")
        layout.addWidget(title)

        note = QLabel(
            "Cấu hình đồng bộ dữ liệu với Google Cloud Storage và Cloud SQL. "
            "Các biến môi trường cần được đặt trong file .env."
        )
        note.setObjectName("mutedText")
        note.setWordWrap(True)
        layout.addWidget(note)

        self.gcs_bucket_input = self._path_row(layout, "GCS Bucket Name", placeholder="my-bucket-name")
        self.gcs_creds_input = self._path_row(layout, "Service Account JSON", browse_file=True)
        self.gcp_project_input = self._path_row(layout, "GCP Project ID", placeholder="my-gcp-project")
        self.cloudsql_conn_input = self._path_row(layout, "Cloud SQL Connection Name", placeholder="project:region:instance")
        self.cloudsql_db_input = self._path_row(layout, "Cloud SQL Database Name", placeholder="eds_production")
        self.cloudsql_user_input = self._path_row(layout, "Cloud SQL User", placeholder="eds_user")
        self.cloudsql_pass_input = self._path_row(layout, "Cloud SQL Password", placeholder="••••••••")
        self.cloudsql_pass_input.setEchoMode(QLineEdit.EchoMode.Password)

        sync_row = QHBoxLayout()
        sync_row.addWidget(QLabel("Sync videos (large files):"))
        self.sync_videos_check = QComboBox()
        self.sync_videos_check.addItems(["No", "Yes"])
        sync_row.addWidget(self.sync_videos_check)
        sync_row.addStretch()
        layout.addLayout(sync_row)

        self.save_btn = QPushButton("Save Settings")
        self.save_btn.setObjectName("primaryBtn")
        self.save_btn.clicked.connect(self._save_cloud_settings)
        layout.addWidget(self.save_btn)

        self.cloud_status_label = QLabel("")
        self.cloud_status_label.setObjectName("mutedText")
        layout.addWidget(self.cloud_status_label)

        return card

    def _save_cloud_settings(self):
        """Save cloud settings to user_settings.json."""
        import json as _json
        settings_path = Path("d:/Hai/study/DeepLerning/BCDA/tools/emotion-data-studio/data/user_settings.json")
        try:
            data = _json.loads(settings_path.read_text(encoding="utf-8"))
        except Exception:
            data = {}

        data.setdefault("cloud", {})
        data["cloud"]["gcs_bucket"] = self.gcs_bucket_input.text().strip()
        data["cloud"]["gcs_credentials"] = self.gcs_creds_input.text().strip()
        data["cloud"]["gcp_project"] = self.gcp_project_input.text().strip()
        data["cloud"]["cloudsql_connection"] = self.cloudsql_conn_input.text().strip()
        data["cloud"]["cloudsql_db"] = self.cloudsql_db_input.text().strip()
        data["cloud"]["cloudsql_user"] = self.cloudsql_user_input.text().strip()
        data["cloud"]["cloudsql_password"] = self.cloudsql_pass_input.text().strip()
        data["cloud"]["sync_videos"] = self.sync_videos_check.currentText() == "Yes"

        settings_path.parent.mkdir(parents=True, exist_ok=True)
        settings_path.write_text(_json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        self.cloud_status_label.setText("✅ Cloud settings saved to user_settings.json")
        self.settings_saved.emit(data)

    def _build_diagnostics_card(self) -> QFrame:
        card = QFrame()
        card.setObjectName("card")
        layout = QVBoxLayout(card)
        layout.setSpacing(10)

        header = QHBoxLayout()
        from backend.config import settings
        title = QLabel(f"Chẩn Đoán & Cập Nhật (v{settings.VERSION})")
        title.setObjectName("sectionTitle")
        header.addWidget(title)
        header.addStretch()
        self.update_btn = QPushButton("Kiểm tra cập nhật")
        self.update_btn.clicked.connect(self.update_check_requested.emit)
        header.addWidget(self.update_btn)
        self.run_diag_btn = QPushButton("Chạy chẩn đoán")
        self.run_diag_btn.clicked.connect(self.run_diagnostics)
        header.addWidget(self.run_diag_btn)
        self.test_download_btn = QPushButton("Kiểm tra downloader")
        self.test_download_btn.clicked.connect(self.test_downloader)
        header.addWidget(self.test_download_btn)
        layout.addLayout(header)

        self.diagnostics_output = QPlainTextEdit()
        self.diagnostics_output.setReadOnly(True)
        self.diagnostics_output.setMinimumHeight(220)
        self.diagnostics_output.setObjectName("logViewer")
        layout.addWidget(self.diagnostics_output)
        return card

    def _path_row(self, parent_layout: QVBoxLayout, label: str, browse_file=False, browse_dir=False, placeholder: str = "") -> QLineEdit:
        row = QHBoxLayout()
        row.addWidget(QLabel(label))
        edit = QLineEdit()
        if placeholder:
            edit.setPlaceholderText(placeholder)
        row.addWidget(edit, stretch=1)
        browse = QPushButton("Duyệt")
        if browse_file:
            browse.clicked.connect(lambda: self._browse_file(edit))
        elif browse_dir:
            browse.clicked.connect(lambda: self._browse_dir(edit))
        row.addWidget(browse)
        parent_layout.addLayout(row)
        return edit

    def _double_row(self, parent_layout: QVBoxLayout, label: str, minimum: float, maximum: float, value: float, step: float):
        row = QHBoxLayout()
        row.addWidget(QLabel(label))
        spin = FocusDoubleSpinBox()
        spin.setRange(minimum, maximum)
        spin.setSingleStep(step)
        spin.setValue(value)
        row.addWidget(spin, stretch=1)
        parent_layout.addLayout(row)
        return spin

    def _browse_file(self, edit: QLineEdit):
        path, _ = QFileDialog.getOpenFileName(self, "Chọn file", edit.text() or str(Path.home()))
        if path:
            edit.setText(path)

    def _browse_dir(self, edit: QLineEdit):
        path = QFileDialog.getExistingDirectory(self, "Chọn thư mục", edit.text() or str(Path.home()))
        if path:
            edit.setText(path)

    def _settings_path(self) -> Path:
        from backend.config import settings
        return settings.user_settings_path

    def _collect_settings(self) -> dict:
        return {
            "ffmpeg_path": self.ffmpeg_input.text().strip() or "ffmpeg",
            "data_dir": self.data_dir_input.text().strip(),
            "model_cache_dir": self.model_cache_input.text().strip(),
            "runtime_mode": self.runtime_combo.currentText(),
            "scene_threshold": self.scene_threshold.value(),
            "min_clip_duration": self.min_duration.value(),
            "max_clip_duration": self.max_duration.value(),
            "cpu_threads": int(self.cpu_threads.value()),
            "ffmpeg_threads": int(self.ffmpeg_threads.value()),
            "pipeline_workers": int(self.pipeline_workers.value()),
            "download_mode": self.download_mode_combo.currentText(),
            "download_max_height": int(self.download_max_height.value()),
            "download_concurrent_fragments": int(self.download_concurrent_fragments.value()),
            "download_throttled_rate_kbps": int(self.download_throttled_rate_kbps.value()),
            "download_use_aria2": self.download_use_aria2_combo.currentText() == "true",
            "download_cookies_browser": self.download_cookies_browser_combo.currentText().strip(),
            "download_cookie_file": self.download_cookie_file_input.text().strip(),
            "smart_vad_mode": self.smart_vad_mode.currentText(),
            "smart_face_scan_fps": self.smart_face_scan_fps.value(),
            "smart_face_confidence": self.smart_face_confidence.value(),
            "smart_max_missing_face_gap": self.smart_max_missing_face_gap.value(),
            "smart_target_clip_duration": self.smart_target_clip_duration.value(),
            "smart_silence_threshold_db": self.smart_silence_threshold_db.text().strip() or "-35dB",
            "smart_silence_min_duration": self.smart_silence_min_duration.value(),
            "smart_max_dialogue_extension": self.smart_max_dialogue_extension.value(),
            "vertex_location": self.vertex_location_input.text().strip() or "global",
            "gemini_model": self.gemini_model_input.text().strip() or "gemini-2.5-flash",
            "gemini_api_key": self.gemini_api_key_input.text().strip(),
            "google_application_credentials": self.gac_path_input.text().strip(),
            "ai_autocut_enabled": self.ai_autocut_combo.currentText() == "true",
            "ai_autocut_intensity_threshold": self.ai_autocut_threshold.value(),
            "ai_autocut_min_duration": self.ai_autocut_min_duration.value(),
            "ai_autocut_max_duration": self.ai_autocut_max_duration.value(),
            "ai_autocut_max_segments": int(self.ai_autocut_max_segments.value()),
            "ai_autocut_verify_strict": self.ai_autocut_verify_strict_combo.currentText() == "true",
            "ai_autocut_min_face_coverage_in_scene": self.ai_autocut_min_face_coverage.value(),
        }

    def load_settings(self):
        from backend.config import settings

        data = {
            "ffmpeg_path": settings.FFMPEG_PATH,
            "data_dir": str(settings.DATA_DIR),
            "model_cache_dir": str(settings.MODEL_CACHE_DIR),
            "runtime_mode": settings.RUNTIME_MODE,
            "scene_threshold": settings.SCENE_THRESHOLD,
            "min_clip_duration": settings.MIN_CLIP_DURATION,
            "max_clip_duration": settings.MAX_CLIP_DURATION,
            "cpu_threads": settings.EDS_CPU_THREADS,
            "ffmpeg_threads": settings.EDS_FFMPEG_THREADS,
            "pipeline_workers": settings.EDS_PIPELINE_WORKERS,
            "download_mode": os.getenv("EDS_DOWNLOAD_MODE", os.getenv("DOWNLOAD_MODE", "balanced")),
            "download_max_height": int(os.getenv("EDS_DOWNLOAD_MAX_HEIGHT", os.getenv("DOWNLOAD_MAX_HEIGHT", "720"))),
            "download_concurrent_fragments": int(os.getenv("EDS_DOWNLOAD_CONCURRENT_FRAGMENTS", os.getenv("DOWNLOAD_CONCURRENT_FRAGMENTS", "5"))),
            "download_throttled_rate_kbps": int(os.getenv("EDS_DOWNLOAD_THROTTLED_RATE_KBPS", os.getenv("DOWNLOAD_THROTTLED_RATE_KBPS", "100"))),
            "download_use_aria2": (os.getenv("EDS_DOWNLOAD_USE_ARIA2", os.getenv("DOWNLOAD_USE_ARIA2", "false")).lower() == "true"),
            "download_cookies_browser": os.getenv("EDS_DOWNLOAD_COOKIES_BROWSER", os.getenv("DOWNLOAD_COOKIES_BROWSER", "")),
            "download_cookie_file": os.getenv("EDS_DOWNLOAD_COOKIE_FILE", os.getenv("DOWNLOAD_COOKIE_FILE", "")),
            "smart_vad_mode": settings.SMART_VAD_MODE,
            "smart_face_scan_fps": settings.SMART_FACE_SCAN_FPS,
            "smart_face_confidence": settings.SMART_FACE_CONFIDENCE,
            "smart_max_missing_face_gap": settings.SMART_MAX_MISSING_FACE_GAP,
            "smart_target_clip_duration": settings.SMART_TARGET_CLIP_DURATION,
            "smart_silence_threshold_db": settings.SMART_SILENCE_THRESHOLD_DB,
            "smart_silence_min_duration": settings.SMART_SILENCE_MIN_DURATION,
            "smart_max_dialogue_extension": settings.SMART_MAX_DIALOGUE_EXTENSION,
        }
        path = settings.user_settings_path
        if path.exists():
            try:
                data.update(json.loads(path.read_text(encoding="utf-8")))
            except Exception:
                pass

        self.ffmpeg_input.setText(str(data.get("ffmpeg_path") or "ffmpeg"))
        self.data_dir_input.setText(str(data.get("data_dir") or settings.DATA_DIR))
        self.model_cache_input.setText(str(data.get("model_cache_dir") or settings.MODEL_CACHE_DIR))
        self.runtime_combo.setCurrentText(str(data.get("runtime_mode") or "auto"))
        self.scene_threshold.setValue(float(data.get("scene_threshold") or 30.0))
        self.min_duration.setValue(float(data.get("min_clip_duration") or 3.0))
        self.max_duration.setValue(float(data.get("max_clip_duration") or 15.0))
        self.cpu_threads.setValue(float(data.get("cpu_threads") or 0))
        self.ffmpeg_threads.setValue(float(data.get("ffmpeg_threads") or 0))
        self.pipeline_workers.setValue(float(data.get("pipeline_workers") or 1))

        self.download_mode_combo.setCurrentText(str(data.get("download_mode") or "balanced"))
        self.download_max_height.setValue(float(data.get("download_max_height") or 720))
        self.download_concurrent_fragments.setValue(float(data.get("download_concurrent_fragments") or 5))
        self.download_throttled_rate_kbps.setValue(float(data.get("download_throttled_rate_kbps") or 100))
        self.download_use_aria2_combo.setCurrentText("true" if data.get("download_use_aria2") else "false")
        self.download_cookies_browser_combo.setCurrentText(str(data.get("download_cookies_browser") or ""))
        self.download_cookie_file_input.setText(str(data.get("download_cookie_file") or ""))
        if shutil.which("aria2c") is not None and self.download_use_aria2_combo.currentText() != "true":
            self.download_use_aria2_combo.setToolTip("aria2c đã được phát hiện trên máy. Bạn nên bật tùy chọn này để tăng tốc tải video từ URL.")
        else:
            self.download_use_aria2_combo.setToolTip("")

        self.smart_vad_mode.setCurrentText(str(data.get("smart_vad_mode") or "auto"))
        self.smart_face_scan_fps.setValue(float(data.get("smart_face_scan_fps") or 2.0))
        self.smart_face_confidence.setValue(float(data.get("smart_face_confidence") or 0.55))
        self.smart_max_missing_face_gap.setValue(float(data.get("smart_max_missing_face_gap") or 1.0))
        self.smart_target_clip_duration.setValue(float(data.get("smart_target_clip_duration") or 6.0))
        self.smart_silence_threshold_db.setText(str(data.get("smart_silence_threshold_db") or "-35dB"))
        self.smart_silence_min_duration.setValue(float(data.get("smart_silence_min_duration") or 0.45))
        self.smart_max_dialogue_extension.setValue(float(data.get("smart_max_dialogue_extension") or 1.5))

        self.vertex_location_input.setText(str(data.get("vertex_location") or settings.VERTEX_LOCATION or "global"))
        self.gemini_model_input.setText(str(data.get("gemini_model") or settings.GEMINI_MODEL or "gemini-2.5-flash"))
        self.gemini_api_key_input.setText(str(data.get("gemini_api_key") or ""))
        gac = data.get("google_application_credentials") or os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "")
        self.gac_path_input.setText(str(gac))
        self.ai_autocut_combo.setCurrentText("true" if data.get("ai_autocut_enabled") else "false")
        self.ai_autocut_threshold.setValue(float(data.get("ai_autocut_intensity_threshold") or 0.55))
        self.ai_autocut_min_duration.setValue(float(data.get("ai_autocut_min_duration") or 3.0))
        self.ai_autocut_max_duration.setValue(float(data.get("ai_autocut_max_duration") or 15.0))
        self.ai_autocut_max_segments.setValue(float(data.get("ai_autocut_max_segments") or 40))
        self.ai_autocut_verify_strict_combo.setCurrentText(
            "true" if data.get("ai_autocut_verify_strict", False) else "false")
        self.ai_autocut_min_face_coverage.setValue(
            float(data.get("ai_autocut_min_face_coverage_in_scene") or 0.70))

        self.run_diagnostics()

    def save_settings(self):
        data = self._collect_settings()
        if data["min_clip_duration"] > data["max_clip_duration"]:
            QMessageBox.warning(self, "Cài đặt không hợp lệ", "Thời lượng clip tối thiểu phải ≤ tối đa.")
            return

        path = self._settings_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

        os.environ["FFMPEG_PATH"] = data["ffmpeg_path"]
        if data["data_dir"]:
            os.environ["EDS_DATA_DIR"] = data["data_dir"]
        if data["model_cache_dir"]:
            os.environ["MODEL_CACHE_DIR"] = data["model_cache_dir"]
        os.environ["RUNTIME_MODE"] = data["runtime_mode"]
        os.environ["SCENE_THRESHOLD"] = str(data["scene_threshold"])
        os.environ["MIN_CLIP_DURATION"] = str(data["min_clip_duration"])
        os.environ["MAX_CLIP_DURATION"] = str(data["max_clip_duration"])
        os.environ["EDS_CPU_THREADS"] = str(data["cpu_threads"])
        os.environ["EDS_FFMPEG_THREADS"] = str(data["ffmpeg_threads"])
        os.environ["EDS_PIPELINE_WORKERS"] = str(data["pipeline_workers"])

        os.environ["EDS_DOWNLOAD_MODE"] = data["download_mode"]
        os.environ["EDS_DOWNLOAD_MAX_HEIGHT"] = str(data["download_max_height"])
        os.environ["EDS_DOWNLOAD_CONCURRENT_FRAGMENTS"] = str(data["download_concurrent_fragments"])
        os.environ["EDS_DOWNLOAD_THROTTLED_RATE_KBPS"] = str(data["download_throttled_rate_kbps"])
        os.environ["EDS_DOWNLOAD_USE_ARIA2"] = "true" if data["download_use_aria2"] else "false"
        os.environ["EDS_DOWNLOAD_COOKIES_BROWSER"] = data["download_cookies_browser"]
        os.environ["EDS_DOWNLOAD_COOKIE_FILE"] = data["download_cookie_file"]

        os.environ["SMART_VAD_MODE"] = data["smart_vad_mode"]
        os.environ["SMART_FACE_SCAN_FPS"] = str(data["smart_face_scan_fps"])
        os.environ["SMART_FACE_CONFIDENCE"] = str(data["smart_face_confidence"])
        os.environ["SMART_MAX_MISSING_FACE_GAP"] = str(data["smart_max_missing_face_gap"])
        os.environ["SMART_TARGET_CLIP_DURATION"] = str(data["smart_target_clip_duration"])
        os.environ["SMART_SILENCE_THRESHOLD_DB"] = data["smart_silence_threshold_db"]
        os.environ["SMART_SILENCE_MIN_DURATION"] = str(data["smart_silence_min_duration"])
        os.environ["SMART_MAX_DIALOGUE_EXTENSION"] = str(data["smart_max_dialogue_extension"])

        # Vertex AI / Gemini
        os.environ["VERTEX_LOCATION"] = data["vertex_location"]
        os.environ["GEMINI_MODEL"] = data["gemini_model"]
        if data.get("gemini_api_key"):
            os.environ["GEMINI_API_KEY"] = data["gemini_api_key"]
        if data.get("google_application_credentials"):
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = data["google_application_credentials"]
        os.environ["AI_AUTOCUT_ENABLED"] = "true" if data["ai_autocut_enabled"] else "false"
        os.environ["AI_AUTOCUT_INTENSITY_THRESHOLD"] = str(data["ai_autocut_intensity_threshold"])
        os.environ["AI_AUTOCUT_MIN_DURATION"] = str(data["ai_autocut_min_duration"])
        os.environ["AI_AUTOCUT_MAX_DURATION"] = str(data["ai_autocut_max_duration"])
        os.environ["AI_AUTOCUT_MAX_SEGMENTS"] = str(data["ai_autocut_max_segments"])
        os.environ["AI_AUTOCUT_VERIFY_STRICT"] = str(
            data.get("ai_autocut_verify_strict", False)
        ).lower()
        os.environ["AI_AUTOCUT_MIN_FACE_COVERAGE_IN_SCENE"] = str(
            data.get("ai_autocut_min_face_coverage_in_scene", 0.70)
        )

        # Apply to in-memory settings too (khong can restart)
        try:
            from backend.config import settings as _settings
            _settings.VERTEX_LOCATION = data["vertex_location"]
            _settings.GEMINI_MODEL = data["gemini_model"]
            if data.get("gemini_api_key"):
                _settings.GEMINI_API_KEY = data["gemini_api_key"]
            if data.get("google_application_credentials"):
                _settings.GOOGLE_APPLICATION_CREDENTIALS = data["google_application_credentials"]
            _settings.AI_AUTOCUT_ENABLED = bool(data["ai_autocut_enabled"])
            _settings.AI_AUTOCUT_INTENSITY_THRESHOLD = float(data["ai_autocut_intensity_threshold"])
            _settings.AI_AUTOCUT_MIN_DURATION = float(data["ai_autocut_min_duration"])
            _settings.AI_AUTOCUT_MAX_DURATION = float(data["ai_autocut_max_duration"])
            _settings.AI_AUTOCUT_MAX_SEGMENTS = int(data["ai_autocut_max_segments"])
            _settings.AI_AUTOCUT_VERIFY_STRICT = bool(data.get("ai_autocut_verify_strict", False))
            _settings.AI_AUTOCUT_MIN_FACE_COVERAGE_IN_SCENE = float(
                data.get("ai_autocut_min_face_coverage_in_scene", 0.70)
            )
        except Exception:
            pass

        self.settings_saved.emit(data)
        self.run_diagnostics()
        QMessageBox.information(self, "Lưu thành công", "Cài đặt đã lưu. Một số thay đổi có thể cần khởi động lại ứng dụng.")

    def run_diagnostics(self):
        data = self._collect_settings()
        lines = []

        ffmpeg = data["ffmpeg_path"] or "ffmpeg"
        ffmpeg_ok = shutil.which(ffmpeg) is not None or Path(ffmpeg).exists()
        lines.append(f"FFmpeg: {'OK' if ffmpeg_ok else 'FAIL'} - {ffmpeg}")

        aria2_ok = shutil.which("aria2c") is not None
        lines.append(f"aria2c: {'OK' if aria2_ok else 'FAIL'}")

        deno_ok = shutil.which("deno") is not None
        lines.append(f"Deno: {'OK' if deno_ok else 'FAIL'}")

        try:
            from backend.utils.resource_manager import resource_manager
            plan = resource_manager.plan(force_refresh=True)
            lines.append(f"Tài nguyên: thiết bị={plan.device}, CPU={plan.cpu_threads}/{plan.cpu_count} luồng, FFmpeg={plan.ffmpeg_threads} luồng")
            if plan.gpu_name:
                lines.append(f"GPU: {plan.gpu_name} ({plan.gpu_memory_gb} GB VRAM)")
            if plan.ram_gb:
                lines.append(f"RAM: {plan.ram_gb} GB")
        except Exception as exc:
            lines.append(f"Tài nguyên: không đọc được thông tin - {exc}")

        cookie_file = (data.get("download_cookie_file") or "").strip()
        cookie_browser = (data.get("download_cookies_browser") or "").strip()
        cookie_file_ok = Path(cookie_file).exists() if cookie_file else False
        lines.append(f"Downloader: mode={data['download_mode']}, max_height={int(data['download_max_height'])}p, fragments={int(data['download_concurrent_fragments'])}")
        lines.append(f"Downloader: throttled_rate={int(data['download_throttled_rate_kbps'])} KB/s, aria2c={'bật' if data['download_use_aria2'] else 'tắt'}")
        if aria2_ok and not data['download_use_aria2']:
            lines.append("Gợi ý: aria2c đã được cài nhưng đang tắt. Hãy bật 'Dùng aria2c nếu có' để tăng tốc tải video.")
        if not deno_ok:
            lines.append("Gợi ý: cài Deno để tăng độ ổn định khi YouTube thay đổi cơ chế player/signature.")
        lines.append(f"Cookies: browser={cookie_browser or 'không'}, file={'OK' if cookie_file_ok else ('không' if not cookie_file else 'FAIL')}")
        lines.append(f"Cắt thông minh: VAD={data['smart_vad_mode']}, quét mặt={data['smart_face_scan_fps']} FPS, ngưỡng mặt={data['smart_face_confidence']}")
        lines.append(f"Hội thoại: silence={data['smart_silence_threshold_db']}, im lặng tối thiểu={data['smart_silence_min_duration']}s, mở rộng={data['smart_max_dialogue_extension']}s")
        self.diagnostics_output.setPlainText("\n".join(lines))

    def test_downloader(self):
        test_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        try:
            from backend.services.downloader import VideoDownloader
            downloader = VideoDownloader()
            info = downloader.get_video_info(test_url)
            QMessageBox.information(
                self,
                "Downloader hoạt động",
                f"Đã đọc metadata mẫu thành công.\n\nTiêu đề: {info.get('title')}\nThời lượng: {info.get('duration_sec')}s\nID: {info.get('id')}",
            )
        except Exception as exc:
            QMessageBox.warning(
                self,
                "Downloader gặp lỗi",
                f"Không thể đọc metadata mẫu từ YouTube.\n\n{exc}",
            )

    def refresh_data(self):
        self.load_settings()
