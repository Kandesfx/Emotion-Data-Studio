"""
Emotion Data Studio - Processing Monitor Page

Adds a preflight checklist, real-time pipeline progress, structured logs,
resource indicators, and safe cancel controls.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from urllib.parse import urlparse

from PySide6.QtCore import Qt, Signal, Slot, QTimer
from PySide6.QtGui import QTextCursor
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QFrame,
    QProgressBar,
    QPlainTextEdit,
    QScrollArea,
    QSizePolicy,
)


class CheckRow(QFrame):
    """One preflight check row."""

    def __init__(self, label: str, parent=None):
        super().__init__(parent)
        self.setObjectName("checkRow")
        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 6, 10, 6)
        layout.setSpacing(10)

        self.status_label = QLabel("WAIT")
        self.status_label.setFixedWidth(58)
        self.status_label.setObjectName("mutedText")
        layout.addWidget(self.status_label)

        self.label = QLabel(label)
        layout.addWidget(self.label, stretch=1)

        self.detail = QLabel("")
        self.detail.setObjectName("mutedText")
        self.detail.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        layout.addWidget(self.detail)

    def set_result(self, ok: bool, detail: str = ""):
        self.status_label.setText("OK" if ok else "LỖI")
        self.status_label.setObjectName("successText" if ok else "errorText")
        self.status_label.setStyle(self.status_label.style())
        self.detail.setText(detail)

    def reset(self):
        self.status_label.setText("CHỌI")
        self.status_label.setObjectName("mutedText")
        self.status_label.setStyle(self.status_label.style())
        self.detail.setText("")


class StageProgressWidget(QFrame):
    """Single pipeline stage progress indicator."""

    def __init__(self, stage_name: str, stage_code: str, parent=None):
        super().__init__(parent)
        self.stage_name = stage_name
        self.setObjectName("card")

        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(12)

        self.status_label = QLabel("WAIT")
        self.status_label.setFixedWidth(58)
        self.status_label.setObjectName("mutedText")
        layout.addWidget(self.status_label)

        code = QLabel(stage_code)
        code.setFixedWidth(110)
        code.setObjectName("mutedText")
        layout.addWidget(code)

        name_label = QLabel(stage_name)
        name_label.setMinimumWidth(180)
        name_label.setObjectName("statLabel")
        layout.addWidget(name_label)

        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(100)
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar, stretch=1)

        self.pct_label = QLabel("0%")
        self.pct_label.setFixedWidth(48)
        self.pct_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        layout.addWidget(self.pct_label)

    def set_progress(self, current: int, total: int):
        pct = int(current / total * 100) if total else 0
        pct = max(0, min(100, pct))
        self.progress_bar.setValue(pct)
        self.pct_label.setText(f"{pct}%")
        if pct >= 100:
            self.status_label.setText("XONG")
            self.status_label.setObjectName("successText")
        elif pct > 0:
            self.status_label.setText("ĐANG")
            self.status_label.setObjectName("warningText")
        else:
            self.status_label.setText("CHỌI")
            self.status_label.setObjectName("mutedText")
        self.status_label.setStyle(self.status_label.style())

    def mark_failed(self):
        self.status_label.setText("LỖI")
        self.status_label.setObjectName("errorText")
        self.status_label.setStyle(self.status_label.style())

    def reset(self):
        self.progress_bar.setValue(0)
        self.pct_label.setText("0%")
        self.status_label.setText("CHỌI")
        self.status_label.setObjectName("mutedText")
        self.status_label.setStyle(self.status_label.style())


class ProcessingPage(QWidget):
    """Preflight and processing monitor page."""

    processing_completed = Signal(str)

    STAGES = [
        ("queued",        "HÀNG ĐỢI",  "Xếp hàng"),
        ("download",      "TẢI XUỐNG", "Tải video / Nguồn cục bộ"),
        ("scene_split",   "CẢNH",      "Phân tích cảnh"),
        ("prewarm",       "MÔ HÌNH",    "Khởi động AI"),
        ("face_detect",   "KHUÔN MẶT",  "Phát hiện khuôn mặt"),
        ("audio_extract", "ÂM THANH",   "Trích xuất âm thanh"),
        ("transcribe",    "LỚI NÓI",    "Nhận dạng giọng nói"),
        ("emotion_label", "CẢM XÚC",   "Phân tích cảm xúc AI"),
        ("quality_score", "CHẤT LƯợNG", "Chấm điểm chất lượng"),
    ]

    def __init__(self, parent=None):
        super().__init__(parent)
        self._worker = None
        self._stage_widgets: dict[str, StageProgressWidget] = {}
        self._check_widgets: dict[str, CheckRow] = {}
        self._is_processing = False
        self._elapsed_seconds = 0
        self._current_stage = ""
        self._setup_ui()

    def _setup_ui(self):
        page_layout = QVBoxLayout(self)
        page_layout.setContentsMargins(0, 0, 0, 0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        page_layout.addWidget(scroll)

        scroll_content = QWidget()
        scroll.setWidget(scroll_content)
        self.main_layout = QVBoxLayout(scroll_content)
        self.main_layout.setContentsMargins(32, 24, 32, 24)
        self.main_layout.setSpacing(18)

        self.title_label = QLabel("Giám Sát Xử Lý")
        self.title_label.setObjectName("pageTitle")
        self.main_layout.addWidget(self.title_label)

        self.subtitle_label = QLabel("Chưa có tác vụ xử lý nào. Hãy khởi động từ Bảng Điều Khiển.")
        self.subtitle_label.setObjectName("pageSubtitle")
        self.main_layout.addWidget(self.subtitle_label)

        self._build_preflight_card()
        self._build_worker_status_card()
        self._build_overall_card()
        self._build_stage_card()
        self._build_controls()
        self._build_log_card()
        self.main_layout.addStretch()

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._update_elapsed)

        self._resource_timer = QTimer(self)
        self._resource_timer.timeout.connect(self._update_resources)
        self._resource_timer.start(3000)
        self._update_resources()

    def _build_worker_status_card(self):
        card = QFrame()
        card.setObjectName("cardElevated")
        layout = QVBoxLayout(card)
        layout.setSpacing(10)

        header = QHBoxLayout()
        title = QLabel("GPU Workers (Colab)")
        title.setObjectName("sectionTitle")
        header.addWidget(title)
        self.worker_status_label = QLabel("⏸ Chưa kết nối")
        self.worker_status_label.setObjectName("mutedText")
        header.addWidget(self.worker_status_label)
        refresh_btn = QPushButton("⟳")
        refresh_btn.setFixedSize(28, 28)
        refresh_btn.clicked.connect(self._refresh_worker_status)
        header.addWidget(refresh_btn)
        layout.addLayout(header)

        self.worker_info_label = QLabel("Không có worker nào kết nối")
        self.worker_info_label.setWordWrap(True)
        self.worker_info_label.setObjectName("mutedText")
        layout.addWidget(self.worker_info_label)

        self.main_layout.insertWidget(2, card)
        self._worker_status_card = card

        self._worker_poll_timer = QTimer(self)
        self._worker_poll_timer.timeout.connect(self._refresh_worker_status)
        self._worker_poll_timer.start(15000)

    def _refresh_worker_status(self):
        try:
            import httpx
            resp = httpx.get("http://127.0.0.1:8765/api/worker/status", timeout=5)
            data = resp.json()
            workers = data.get("workers", [])
            queue = data.get("queue", {})

            if workers:
                w = workers[0]
                self.worker_status_label.setText(f"✅ {w.get('gpu_name', 'GPU')}")
                self.worker_info_label.setText(
                    f"Worker: {w.get('worker_id', '?')} | "
                    f"GPU: {w.get('gpu_name', '?')} ({w.get('gpu_memory_gb', 0):.0f}GB) | "
                    f"Queue: {queue.get('queued', 0)} queued, {queue.get('running', 0)} running"
                )
                self._worker_poll_timer.start(15000)
            else:
                self.worker_status_label.setText("⏸ Idle")
                self.worker_info_label.setText(
                    f"Không có worker kết nối | "
                    f"Queue: {queue.get('queued', 0)} queued, {queue.get('running', 0)} running"
                )
        except Exception:
            self.worker_status_label.setText("⚠️ Backend offline")
            self.worker_info_label.setText("Local backend không chạy. Chạy: python backend/main.py")

    def _build_preflight_card(self):
        card = QFrame()
        card.setObjectName("cardElevated")
        layout = QVBoxLayout(card)
        layout.setSpacing(10)

        header = QHBoxLayout()
        title = QLabel("Kiểm Tra Sơ Bộ")
        title.setObjectName("sectionTitle")
        header.addWidget(title)
        header.addStretch()
        self.preflight_summary = QLabel("Chưa có nguồn")
        self.preflight_summary.setObjectName("mutedText")
        header.addWidget(self.preflight_summary)
        layout.addLayout(header)

        checks = [
            ("source",   "Nguồn video / URL"),
            ("ffmpeg",   "FFmpeg khả dụng"),
            ("data_dir", "Thư mục dữ liệu có thể ghi"),
            ("database", "Cơ sở dữ liệu SQLite"),
            ("disk",     "Dung lượng ổ đĩa"),
            ("runtime",  "AI runtime"),
        ]
        for key, label in checks:
            row = CheckRow(label)
            self._check_widgets[key] = row
            layout.addWidget(row)

        self.main_layout.addWidget(card)

    def _build_overall_card(self):
        card = QFrame()
        card.setObjectName("cardElevated")
        layout = QVBoxLayout(card)
        layout.setSpacing(8)

        row = QHBoxLayout()
        title = QLabel("Tiến Độ Tổng Thể")
        title.setObjectName("sectionTitle")
        row.addWidget(title)
        row.addStretch()
        self.elapsed_label = QLabel("00:00")
        self.elapsed_label.setObjectName("mutedText")
        row.addWidget(self.elapsed_label)
        layout.addLayout(row)

        self.overall_progress = QProgressBar()
        self.overall_progress.setObjectName("progressLarge")
        self.overall_progress.setMinimum(0)
        self.overall_progress.setMaximum(100)
        self.overall_progress.setValue(0)
        layout.addWidget(self.overall_progress)

        resources = QHBoxLayout()
        self.current_stage_label = QLabel("Giai đoạn: rảnh")
        self.current_stage_label.setObjectName("mutedText")
        resources.addWidget(self.current_stage_label)
        resources.addStretch()
        self.gpu_usage_label = QLabel("GPU: N/A")
        self.gpu_usage_label.setObjectName("mutedText")
        resources.addWidget(self.gpu_usage_label)
        self.ram_usage_label = QLabel("RAM: N/A")
        self.ram_usage_label.setObjectName("mutedText")
        resources.addWidget(self.ram_usage_label)
        layout.addLayout(resources)

        self.main_layout.addWidget(card)

    def _build_stage_card(self):
        title = QLabel("Các Giai Đoạn Xử Lý")
        title.setObjectName("sectionTitle")
        self.main_layout.addWidget(title)
        for stage_key, stage_code, stage_name in self.STAGES:
            widget = StageProgressWidget(stage_name, stage_code)
            self._stage_widgets[stage_key] = widget
            self.main_layout.addWidget(widget)

    def _build_controls(self):
        row = QHBoxLayout()
        row.setSpacing(12)
        self.cancel_btn = QPushButton("Hủy tác vụ")
        self.cancel_btn.setObjectName("dangerBtn")
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.cancel_btn.clicked.connect(self._on_cancel)
        row.addWidget(self.cancel_btn)

        self.clear_log_btn = QPushButton("Xóa nhật ký")
        self.clear_log_btn.setObjectName("ghostBtn")
        self.clear_log_btn.clicked.connect(lambda: self.log_viewer.clear())
        row.addWidget(self.clear_log_btn)

        row.addStretch()
        self.view_results_btn = QPushButton("Xem kết quả")
        self.view_results_btn.setEnabled(False)
        self.view_results_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        row.addWidget(self.view_results_btn)
        self.main_layout.addLayout(row)

    def _build_log_card(self):
        title = QLabel("Nhật Ký Trực Tiếp")
        title.setObjectName("sectionTitle")
        self.main_layout.addWidget(title)

        self.log_viewer = QPlainTextEdit()
        self.log_viewer.setObjectName("logViewer")
        self.log_viewer.setReadOnly(True)
        self.log_viewer.setMinimumHeight(220)
        self.log_viewer.setMaximumHeight(360)
        self.main_layout.addWidget(self.log_viewer)

    # Public API -----------------------------------------------------

    def attach_worker(self, worker, source: str = ""):
        """Attach a PipelineWorker and start monitoring its signals."""
        self._worker = worker
        worker.progress_updated.connect(self.on_progress_updated)
        worker.log_message.connect(self.on_log_message)
        worker.stage_completed.connect(self.on_stage_completed)
        worker.pipeline_finished.connect(self.on_pipeline_finished)
        worker.error_occurred.connect(self.on_error)
        self.start_monitoring(source or getattr(worker, "video_url", "processing"))

    def start_monitoring(self, source: str):
        self._is_processing = True
        self._elapsed_seconds = 0
        self._current_stage = ""
        self.subtitle_label.setText(f"Đang xử lý: {source}")
        self.overall_progress.setValue(0)
        self.current_stage_label.setText("Giai đoạn: chờ")
        self.view_results_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        self.log_viewer.clear()
        self.append_log(f"[BẮT ĐẦU] Pipeline bắt đầu cho {source}")

        for widget in self._stage_widgets.values():
            widget.reset()

        self.run_preflight(source)
        self._timer.start(1000)

    def run_preflight(self, source: str) -> bool:
        """Run non-destructive preflight checks for UI visibility."""
        for row in self._check_widgets.values():
            row.reset()

        results = []
        source_ok, source_detail = self._check_source(source)
        results.append(source_ok)
        self._check_widgets["source"].set_result(source_ok, source_detail)

        try:
            from backend.config import settings
            ffmpeg_ok = shutil.which(settings.FFMPEG_PATH) is not None
            self._check_widgets["ffmpeg"].set_result(ffmpeg_ok, settings.FFMPEG_PATH)
            results.append(ffmpeg_ok)

            settings.DATA_DIR.mkdir(parents=True, exist_ok=True)
            probe = settings.DATA_DIR / ".write_test"
            probe.write_text("ok", encoding="utf-8")
            probe.unlink(missing_ok=True)
            self._check_widgets["data_dir"].set_result(True, str(settings.DATA_DIR))
            results.append(True)
        except Exception as exc:
            self._check_widgets["data_dir"].set_result(False, str(exc))
            results.append(False)

        try:
            from backend.database.local_db import get_session
            from sqlalchemy import text
            session = get_session()
            session.execute(text("SELECT 1"))
            session.close()
            self._check_widgets["database"].set_result(True, "connection OK")
            results.append(True)
        except Exception as exc:
            self._check_widgets["database"].set_result(False, str(exc))
            results.append(False)

        try:
            data_root = Path(os.environ.get("EDS_DATA_ROOT", "."))
            usage = shutil.disk_usage(data_root)
            free_gb = usage.free / (1024 ** 3)
            disk_ok = free_gb >= 2.0
            self._check_widgets["disk"].set_result(disk_ok, f"{free_gb:.1f} GB free")
            results.append(disk_ok)
        except Exception as exc:
            self._check_widgets["disk"].set_result(False, str(exc))
            results.append(False)

        try:
            import torch
            runtime = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU mode"
            self._check_widgets["runtime"].set_result(True, runtime)
            results.append(True)
        except Exception:
            self._check_widgets["runtime"].set_result(True, "CPU mode / torch optional")
            results.append(True)

        passed = sum(1 for ok in results if ok)
        total = len(results)
        all_ok = all(results)
        self.preflight_summary.setText(f"{passed}/{total} kiểm tra qua")
        self.append_log(f"[KIỂM TRA] {passed}/{total} kiểm tra qua")
        if not all_ok:
            self.append_log("[KIỂM TRA] Một hoặc nhiều kiểm tra thất bại. Worker sẽ dừng nếu bị chặn.")
        return all_ok

    @Slot(str, int, int)
    def on_progress_updated(self, stage_name: str, current: int, total: int):
        self._current_stage = stage_name
        self.current_stage_label.setText(f"Giai đoạn: {stage_name}")
        if stage_name in self._stage_widgets:
            self._stage_widgets[stage_name].set_progress(current, total)
        self._update_overall_progress()

    @Slot(str)
    def on_log_message(self, message: str):
        self.append_log(message)

    @Slot(str)
    def on_stage_completed(self, stage_name: str):
        if stage_name in self._stage_widgets:
            self._stage_widgets[stage_name].set_progress(100, 100)
        self._update_overall_progress()

    @Slot(dict)
    def on_pipeline_finished(self, result: dict):
        self._is_processing = False
        self._timer.stop()
        self.overall_progress.setValue(100)
        video_id = result.get("video_id", "")
        total_clips = result.get("total_clips", 0)
        self.subtitle_label.setText(f"Hoàn thành. Clip: {total_clips}")
        self.append_log("[XONG] Pipeline hoàn thành thành công")
        self.cancel_btn.setEnabled(False)
        self.view_results_btn.setEnabled(True)
        self.processing_completed.emit(video_id)

    @Slot(str)
    def on_error(self, error_msg: str):
        self._is_processing = False
        self._timer.stop()
        if self._current_stage in self._stage_widgets:
            self._stage_widgets[self._current_stage].mark_failed()
        self.subtitle_label.setText("Xử lý thất bại. Xem nhật ký để biết thêm.")
        self.append_log(f"[LỖI] {error_msg}")
        self.cancel_btn.setEnabled(False)

    def append_log(self, message: str):
        self.log_viewer.appendPlainText(message)
        cursor = self.log_viewer.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        self.log_viewer.setTextCursor(cursor)

    def refresh_data(self):
        self._update_resources()

    # Internal -------------------------------------------------------

    def _check_source(self, source: str) -> tuple[bool, str]:
        source = (source or "").strip()
        if not source or source == "processing":
            return False, "missing source"
        parsed = urlparse(source)
        if parsed.scheme in {"http", "https"} and parsed.netloc:
            return True, parsed.netloc
        path = Path(source)
        if path.exists() and path.is_file():
            suffix_ok = path.suffix.lower() in {".mp4", ".mkv", ".avi", ".webm", ".mov"}
            return suffix_ok, path.name if suffix_ok else "unsupported extension"
        return False, "file not found or invalid URL"

    def _update_elapsed(self):
        self._elapsed_seconds += 1
        mins = self._elapsed_seconds // 60
        secs = self._elapsed_seconds % 60
        self.elapsed_label.setText(f"{mins:02d}:{secs:02d}")

    def _update_overall_progress(self):
        if not self._stage_widgets:
            self.overall_progress.setValue(0)
            return
        total_pct = sum(widget.progress_bar.value() for widget in self._stage_widgets.values())
        self.overall_progress.setValue(total_pct // len(self._stage_widgets))

    def _update_resources(self):
        try:
            import psutil
            vm = psutil.virtual_memory()
            used = (vm.total - vm.available) / (1024 ** 3)
            total = vm.total / (1024 ** 3)
            self.ram_usage_label.setText(f"RAM: {used:.1f}/{total:.1f} GB")
        except Exception:
            self.ram_usage_label.setText("RAM: N/A")

        try:
            import torch
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated(0) / (1024 ** 3)
                total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
                self.gpu_usage_label.setText(f"GPU: {allocated:.1f}/{total:.1f} GB")
            else:
                self.gpu_usage_label.setText("GPU: CPU mode")
        except Exception:
            self.gpu_usage_label.setText("GPU: N/A")

    def _on_cancel(self):
        if self._worker is not None and hasattr(self._worker, "cancel"):
            self._worker.cancel()
        self._is_processing = False
        self.cancel_btn.setEnabled(False)
        self.subtitle_label.setText("Đã yêu cầu hủy. Đang chờ dừng an toàn...")
        self.append_log("[HỦY] Người dùng yêu cầu hủy")
