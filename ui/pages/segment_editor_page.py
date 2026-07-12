"""
Emotion Data Studio - Segment Editor Page.

Manual and semi-auto segmentation workspace:
- load a local source video
- scrub preview timeline
- set start/end or type exact times
- split/merge/delete segments
- cut clips with SegmentWorker and optionally run AI
"""

from __future__ import annotations

import os
from pathlib import Path

from PySide6.QtCore import Qt, QUrl, Signal, Slot
from PySide6.QtGui import QColor, QKeySequence, QPainter, QPen, QShortcut
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
from PySide6.QtMultimediaWidgets import QVideoWidget
from PySide6.QtWidgets import (
    QAbstractItemView,
    QFileDialog,
    QFrame,
    QHeaderView,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QComboBox,
    QProgressBar,
    QScrollArea,
    QSizePolicy,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ui.widgets.custom_spinbox import FocusDoubleSpinBox
from ui.styles.theme import Colors, Spacing
from ui.widgets import PageHeader, Card, ActionButton, StatusPill

try:
    import qtawesome as qta
    _HAS_QTA = True
except ImportError:
    _HAS_QTA = False


class TimelineWidget(QWidget):
    position_changed = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(56)
        self.setMaximumHeight(56)
        self._duration_ms = 1
        self._position_ms = 0
        self._segments: list[tuple[int, int]] = []
        self._pending_start_ms: int | None = None
        self._dragging = False

    def set_duration(self, duration_ms: int):
        self._duration_ms = max(1, duration_ms)
        self.update()

    def set_position(self, position_ms: int):
        if not self._dragging:
            self._position_ms = max(0, min(position_ms, self._duration_ms))
            self.update()

    def set_segments(self, segments: list[tuple[int, int]]):
        self._segments = segments
        self.update()

    def set_pending_start(self, start_ms: int | None):
        self._pending_start_ms = start_ms
        self.update()

    def _ms_to_x(self, ms: int) -> float:
        margin = 10
        usable = max(1, self.width() - margin * 2)
        return margin + (ms / self._duration_ms) * usable

    def _x_to_ms(self, x: float) -> int:
        margin = 10
        usable = max(1, self.width() - margin * 2)
        ratio = max(0.0, min(1.0, (x - margin) / usable))
        return int(ratio * self._duration_ms)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()
        painter.fillRect(0, 0, w, h, QColor("#0d0d15"))

        track_y = h // 2 - 4
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor("#1a1a28"))
        painter.drawRoundedRect(10, track_y, w - 20, 8, 4, 4)

        for start_ms, end_ms in self._segments:
            sx = self._ms_to_x(start_ms)
            ex = self._ms_to_x(end_ms)
            painter.setBrush(QColor(108, 92, 231, 95))
            painter.drawRoundedRect(int(sx), track_y - 6, max(2, int(ex - sx)), 20, 4, 4)
            painter.setPen(QPen(QColor("#6c5ce7"), 2))
            painter.drawLine(int(sx), track_y - 8, int(sx), track_y + 16)
            painter.drawLine(int(ex), track_y - 8, int(ex), track_y + 16)
            painter.setPen(Qt.PenStyle.NoPen)

        if self._pending_start_ms is not None:
            px = self._ms_to_x(self._pending_start_ms)
            painter.setPen(QPen(QColor("#00b894"), 2, Qt.PenStyle.DashLine))
            painter.drawLine(int(px), 4, int(px), h - 4)

        px = self._ms_to_x(self._position_ms)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor("#e8e6f0"))
        painter.drawEllipse(int(px) - 6, h // 2 - 6, 12, 12)
        painter.setPen(QColor("#9896a8"))
        painter.drawText(10, h - 6, self._format_ms(0))
        dur = self._format_ms(self._duration_ms)
        painter.drawText(w - 10 - painter.fontMetrics().horizontalAdvance(dur), h - 6, dur)
        painter.end()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._dragging = True
            self._seek(event.position().x())

    def mouseMoveEvent(self, event):
        if self._dragging:
            self._seek(event.position().x())

    def mouseReleaseEvent(self, event):
        self._dragging = False

    def _seek(self, x: float):
        self._position_ms = self._x_to_ms(x)
        self.position_changed.emit(self._position_ms)
        self.update()

    @staticmethod
    def _format_ms(ms: int) -> str:
        total = max(0, int(ms / 1000))
        return f"{total // 60}:{total % 60:02d}"


class SegmentEditorPage(QWidget):
    segment_worker_started = Signal(object, str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._video_id: str | None = None
        self._video_path: str | None = None
        self._video_title = ""
        self._segments: list[dict] = []
        self._pending_start_ms: int | None = None
        self._duration_ms = 0
        self._worker = None
        self._setup_ui()
        self._setup_shortcuts()

    def _setup_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        root.addWidget(scroll)

        content = QWidget()
        scroll.setWidget(content)
        self.content_layout = QVBoxLayout(content)
        self.content_layout.setContentsMargins(Spacing.XL, Spacing.LG, Spacing.XL, Spacing.LG)
        self.content_layout.setSpacing(Spacing.LG)

        # Hero header
        header = PageHeader(
            "Trình Soạn Phân Đoạn",
            "Tải video từ máy, tạo và chỉnh sửa các phân đoạn, sau đó tiến hành xử lý.",
            icon="fa5s.cut",
        )
        self.editor_pill = StatusPill("muted", "Chưa tải video")
        header.set_right_widget(self.editor_pill)
        self.content_layout.addWidget(header)

        self._build_source_card()
        self._build_preview_card()
        self._build_segment_controls()
        self._build_segment_table()
        self._build_action_bar()
        self.content_layout.addStretch()

    def _build_source_card(self):
        card = QFrame()
        card.setObjectName("card")
        layout = QHBoxLayout(card)
        layout.setSpacing(10)
        self.source_label = QLabel("Chưa tải video nguồn")
        self.source_label.setObjectName("mutedText")
        layout.addWidget(self.source_label, stretch=1)
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["semi_auto", "manual"])
        self.mode_combo.setToolTip("semi_auto: chạy AI sau khi cắt; manual: chỉ cắt clip để kiểm duyệt thủ công")
        layout.addWidget(self.mode_combo)
        load_btn = ActionButton("Tải Video Từ Máy", "fa5s.file-video", variant="primary")
        load_btn.clicked.connect(self._choose_video)
        layout.addWidget(load_btn)
        self.content_layout.addWidget(card)

    def _build_preview_card(self):
        card = QFrame()
        card.setObjectName("cardElevated")
        layout = QVBoxLayout(card)
        layout.setSpacing(8)
        self.video_widget = QVideoWidget()
        self.video_widget.setMinimumHeight(360)
        self.video_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.video_widget.setStyleSheet("background-color: #000; border-radius: 8px;")
        layout.addWidget(self.video_widget)

        self.media_player = QMediaPlayer(self)
        self.audio_output = QAudioOutput(self)
        self.media_player.setAudioOutput(self.audio_output)
        self.media_player.setVideoOutput(self.video_widget)
        self.media_player.durationChanged.connect(self._on_duration_changed)
        self.media_player.positionChanged.connect(self._on_position_changed)

        self.timeline = TimelineWidget()
        self.timeline.position_changed.connect(self._on_timeline_seek)
        layout.addWidget(self.timeline)

        controls = QHBoxLayout()
        controls.setSpacing(Spacing.SM)
        self.play_btn = ActionButton("", "fa5s.play", variant="primary")
        self.play_btn.setToolTip("Phát / Dừng (Space)")
        self.play_btn.setFixedWidth(40)
        self.play_btn.clicked.connect(self._toggle_play)
        controls.addWidget(self.play_btn)
        self.time_label = QLabel("00:00 / 00:00")
        self.time_label.setStyleSheet(f"color: {Colors.TEXT_SECONDARY}; background: transparent;")
        controls.addWidget(self.time_label)
        controls.addStretch()
        for label, rate in [("0.5x", 0.5), ("1x", 1.0), ("2x", 2.0)]:
            btn = ActionButton(label, "", variant="ghost")
            btn.clicked.connect(lambda checked=False, r=rate: self.media_player.setPlaybackRate(r))
            controls.addWidget(btn)
        layout.addLayout(controls)
        self.content_layout.addWidget(card)

    def _build_segment_controls(self):
        card = QFrame()
        card.setObjectName("card")
        layout = QVBoxLayout(card)
        title = QLabel("Tạo / Chỉnh Sửa Phân Đoạn")
        title.setObjectName("sectionTitle")
        layout.addWidget(title)

        row = QHBoxLayout()
        row.setSpacing(Spacing.SM)
        self.set_start_btn = ActionButton("Đặt Bắt Đầu (S)", "fa5s.play", variant="secondary")
        self.set_start_btn.clicked.connect(self._set_start)
        row.addWidget(self.set_start_btn)
        self.set_end_btn = ActionButton("Đặt Kết Thúc & Thêm (E)", "fa5s.plus", variant="primary")
        self.set_end_btn.setEnabled(False)
        self.set_end_btn.clicked.connect(self._set_end)
        row.addWidget(self.set_end_btn)
        cancel_btn = ActionButton("Huỷ Đánh Dấu", "fa5s.times", variant="ghost")
        cancel_btn.clicked.connect(self._cancel_pending)
        row.addWidget(cancel_btn)
        self.split_btn = ActionButton("Tách tại điểm phát", "fa5s.cut", variant="secondary")
        self.split_btn.clicked.connect(self._split_selected)
        row.addWidget(self.split_btn)
        self.merge_btn = ActionButton("Gộp đang chọn", "fa5s.compress-arrows-alt", variant="secondary")
        self.merge_btn.clicked.connect(self._merge_selected)
        row.addWidget(self.merge_btn)
        layout.addLayout(row)

        self.pending_label = QLabel("")
        self.pending_label.setObjectName("accentText")
        layout.addWidget(self.pending_label)

        manual = QHBoxLayout()
        manual.addWidget(QLabel("Bắt đầu (s):"))
        self.start_input = FocusDoubleSpinBox()
        self.start_input.setRange(0, 99999)
        self.start_input.setDecimals(2)
        self.start_input.setSingleStep(0.5)
        manual.addWidget(self.start_input)
        manual.addWidget(QLabel("Kết thúc (s):"))
        self.end_input = FocusDoubleSpinBox()
        self.end_input.setRange(0, 99999)
        self.end_input.setDecimals(2)
        self.end_input.setSingleStep(0.5)
        manual.addWidget(self.end_input)
        add_btn = QPushButton("Thêm Đoạn Thủ Công")
        add_btn.clicked.connect(self._add_manual_segment)
        manual.addWidget(add_btn)
        manual.addStretch()
        layout.addLayout(manual)
        self.content_layout.addWidget(card)

    def _build_segment_table(self):
        card = QFrame()
        card.setObjectName("card")
        layout = QVBoxLayout(card)
        header = QHBoxLayout()
        title = QLabel("Danh Sách Phân Đoạn")
        title.setObjectName("sectionTitle")
        header.addWidget(title)
        self.segment_count_label = QLabel("0 phân đoạn")
        self.segment_count_label.setObjectName("mutedText")
        header.addWidget(self.segment_count_label)
        header.addStretch()
        self.clear_all_btn = QPushButton("Xoá Tất Cả")
        self.clear_all_btn.setObjectName("dangerBtn")
        self.clear_all_btn.clicked.connect(self._clear_all_segments)
        header.addWidget(self.clear_all_btn)
        layout.addLayout(header)

        self.segment_table = QTableWidget()
        self.segment_table.setColumnCount(5)
        self.segment_table.setHorizontalHeaderLabels(["#", "Bắt Đầu", "Kết Thúc", "Thời Lượng", "Thao Tác"])
        self.segment_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.segment_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Fixed)
        self.segment_table.setColumnWidth(0, 44)
        self.segment_table.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeMode.Fixed)
        self.segment_table.setColumnWidth(4, 150)
        self.segment_table.verticalHeader().setVisible(False)
        self.segment_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.segment_table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.segment_table.setMinimumHeight(220)
        layout.addWidget(self.segment_table)
        self.content_layout.addWidget(card)

    def _build_action_bar(self):
        card = Card("elevated")
        layout = card._layout
        self.mode_label = QLabel("Chế độ: semi_auto")
        self.mode_label.setStyleSheet(f"color: {Colors.ACCENT_LIGHT}; background: transparent;")
        layout.addWidget(self.mode_label)
        layout.addStretch()
        self.process_progress = QProgressBar()
        self.process_progress.setVisible(False)
        self.process_progress.setFixedWidth(220)
        layout.addWidget(self.process_progress)
        self.confirm_btn = ActionButton("Xác Nhận & Tiến Hành Xử Lý", "fa5s.check-double", variant="primary")
        self.confirm_btn.clicked.connect(self._on_confirm)
        layout.addWidget(self.confirm_btn)
        self.content_layout.addWidget(card)

    def _setup_shortcuts(self):
        QShortcut(QKeySequence(Qt.Key.Key_Space), self).activated.connect(self._toggle_play)
        QShortcut(QKeySequence("S"), self).activated.connect(self._set_start)
        QShortcut(QKeySequence("E"), self).activated.connect(self._set_end)
        QShortcut(QKeySequence("Delete"), self).activated.connect(self._delete_selected)

    # Public ---------------------------------------------------------

    def refresh_data(self):
        self.mode_label.setText(f"Chế độ: {self.mode_combo.currentText()}")

    def load_video(self, video_id: str, video_path: str, processing_mode: str = "semi_auto", video_title: str = ""):
        self._video_id = video_id
        self._video_path = video_path
        self._video_title = video_title or Path(video_path).stem
        self.mode_combo.setCurrentText(processing_mode)
        self._segments = []
        self._pending_start_ms = None
        self.source_label.setText(f"{self._video_title} | {video_path}")
        self.subtitle_label.setText("Tạo các clip chính xác để kiểm duyệt cảm xúc.")
        self.media_player.setSource(QUrl.fromLocalFile(video_path))
        self._refresh_table()

    def _choose_video(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Chọn video để phân đoạn",
            "",
            "Video Files (*.mp4 *.mkv *.avi *.webm *.mov);;All Files (*.*)",
        )
        if not path:
            return
        try:
            from backend.database.local_db import get_session
            from backend.database.models import Video

            session = get_session()
            try:
                title = Path(path).stem
                video = Video(title=title, movie_name=title, file_path=path, status="pending", processing_mode=self.mode_combo.currentText())
                session.add(video)
                session.commit()
                session.refresh(video)
                self.load_video(video.id, path, self.mode_combo.currentText(), title)
            finally:
                session.close()
        except Exception as exc:
            QMessageBox.warning(self, "Tải video thất bại", str(exc))

    # Player ---------------------------------------------------------

    def _toggle_play(self):
        if self.media_player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self.media_player.pause()
            self.play_btn.setText("Phát")
        else:
            self.media_player.play()
            self.play_btn.setText("Tạm dừng")

    @Slot(int)
    def _on_duration_changed(self, duration_ms: int):
        self._duration_ms = duration_ms
        self.timeline.set_duration(duration_ms)
        self.start_input.setMaximum(duration_ms / 1000)
        self.end_input.setMaximum(duration_ms / 1000)

    @Slot(int)
    def _on_position_changed(self, position_ms: int):
        self.timeline.set_position(position_ms)
        self.time_label.setText(f"{self._format_ms(position_ms)} / {self._format_ms(self._duration_ms)}")

    @Slot(int)
    def _on_timeline_seek(self, ms: int):
        self.media_player.setPosition(ms)

    # Segment editing ------------------------------------------------

    def _set_start(self):
        if not self._video_path:
            QMessageBox.information(self, "Không có video", "Vui lòng tải video trước.")
            return
        self._pending_start_ms = self.media_player.position()
        self.timeline.set_pending_start(self._pending_start_ms)
        self.set_end_btn.setEnabled(True)
        self.pending_label.setText(f"Đã đánh dấu điểm bắt đầu tại {self._format_ms(self._pending_start_ms)}")

    def _set_end(self):
        if self._pending_start_ms is None:
            return
        self._add_segment_ms(self._pending_start_ms, self.media_player.position())
        self._cancel_pending()

    def _cancel_pending(self):
        self._pending_start_ms = None
        self.timeline.set_pending_start(None)
        self.set_end_btn.setEnabled(False)
        self.pending_label.setText("")

    def _add_manual_segment(self):
        self._add_segment_ms(int(self.start_input.value() * 1000), int(self.end_input.value() * 1000))

    def _add_segment_ms(self, start_ms: int, end_ms: int):
        if end_ms <= start_ms:
            QMessageBox.warning(self, "Phân đoạn không hợp lệ", "Thời gian kết thúc phải sau thời gian bắt đầu.")
            return
        duration = (end_ms - start_ms) / 1000
        if duration < 1.0:
            QMessageBox.warning(self, "Phân đoạn không hợp lệ", "Phân đoạn phải dài tối thiểu 1 giây.")
            return
        if self._duration_ms and end_ms > self._duration_ms:
            QMessageBox.warning(self, "Phân đoạn không hợp lệ", "Điểm kết thúc phân đoạn vượt quá thời lượng video.")
            return
        self._segments.append({"start_ms": start_ms, "end_ms": end_ms})
        self._merge_overlaps()
        self._refresh_table()

    def _split_selected(self):
        rows = self._selected_rows()
        if len(rows) != 1:
            QMessageBox.information(self, "Tách phân đoạn", "Chọn đúng một phân đoạn để tách.")
            return
        index = rows[0]
        seg = self._segments[index]
        pos = self.media_player.position()
        if not (seg["start_ms"] + 500 < pos < seg["end_ms"] - 500):
            QMessageBox.warning(self, "Tách phân đoạn", "Điểm phát phải nằm trong phân đoạn được chọn và cách hai đầu tối thiểu 0.5s.")
            return
        self._segments[index:index + 1] = [
            {"start_ms": seg["start_ms"], "end_ms": pos},
            {"start_ms": pos, "end_ms": seg["end_ms"]},
        ]
        self._refresh_table()

    def _merge_selected(self):
        rows = self._selected_rows()
        if len(rows) < 2:
            QMessageBox.information(self, "Gộp phân đoạn", "Chọn ít nhất hai phân đoạn để tiến hành gộp.")
            return
        starts = [self._segments[i]["start_ms"] for i in rows]
        ends = [self._segments[i]["end_ms"] for i in rows]
        for i in sorted(rows, reverse=True):
            self._segments.pop(i)
        self._segments.append({"start_ms": min(starts), "end_ms": max(ends)})
        self._refresh_table()

    def _delete_segment(self, index: int):
        if 0 <= index < len(self._segments):
            self._segments.pop(index)
            self._refresh_table()

    def _delete_selected(self):
        rows = self._selected_rows()
        for index in sorted(rows, reverse=True):
            self._delete_segment(index)

    def _seek_to_segment(self, index: int):
        if 0 <= index < len(self._segments):
            self.media_player.setPosition(self._segments[index]["start_ms"])

    def _clear_all_segments(self):
        if not self._segments:
            return
        if QMessageBox.question(self, "Xoá tất cả", f"Xoá toàn bộ {len(self._segments)} phân đoạn?") == QMessageBox.StandardButton.Yes:
            self._segments.clear()
            self._refresh_table()

    def _merge_overlaps(self):
        ordered = sorted(self._segments, key=lambda s: s["start_ms"])
        merged = []
        for seg in ordered:
            if not merged or seg["start_ms"] > merged[-1]["end_ms"]:
                merged.append(dict(seg))
            else:
                merged[-1]["end_ms"] = max(merged[-1]["end_ms"], seg["end_ms"])
        self._segments = merged

    def _refresh_table(self):
        self._segments.sort(key=lambda s: s["start_ms"])
        self.segment_table.setRowCount(len(self._segments))
        self.segment_count_label.setText(f"{len(self._segments)} phân đoạn")
        for row, seg in enumerate(self._segments):
            values = [
                str(row + 1),
                self._format_ms(seg["start_ms"]),
                self._format_ms(seg["end_ms"]),
                f"{(seg['end_ms'] - seg['start_ms']) / 1000:.2f}s",
            ]
            for col, text in enumerate(values):
                item = QTableWidgetItem(text)
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                if col == 0:
                    item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self.segment_table.setItem(row, col, item)
            actions = QWidget()
            action_layout = QHBoxLayout(actions)
            action_layout.setContentsMargins(2, 2, 2, 2)
            seek_btn = QPushButton("Đi tới")
            seek_btn.clicked.connect(lambda checked=False, i=row: self._seek_to_segment(i))
            action_layout.addWidget(seek_btn)
            del_btn = QPushButton("Xoá")
            del_btn.setObjectName("dangerBtn")
            del_btn.clicked.connect(lambda checked=False, i=row: self._delete_segment(i))
            action_layout.addWidget(del_btn)
            self.segment_table.setCellWidget(row, 4, actions)
        self.timeline.set_segments([(s["start_ms"], s["end_ms"]) for s in self._segments])

    def _selected_rows(self) -> list[int]:
        return sorted({idx.row() for idx in self.segment_table.selectedIndexes()})

    def _on_confirm(self):
        if not self._video_id or not self._video_path:
            QMessageBox.warning(self, "Không có video", "Vui lòng tải video trước.")
            return
        if not self._segments:
            QMessageBox.warning(self, "Không có phân đoạn", "Vui lòng tạo ít nhất một phân đoạn trước khi xử lý.")
            return
        mode = self.mode_combo.currentText()
        segments_data = []
        for i, seg in enumerate(self._segments):
            start = seg["start_ms"] / 1000
            end = seg["end_ms"] / 1000
            segments_data.append({"segment_index": i, "start_time": start, "end_time": end, "duration": end - start})
        from ui.workers.segment_worker import SegmentWorker
        self._worker = SegmentWorker(self._video_id, self._video_path, segments_data, mode)
        self.segment_worker_started.emit(self._worker, f"phân đoạn thủ công: {self._video_title}")
        self._worker.start()

    @staticmethod
    def _format_ms(ms: int) -> str:
        total = max(0, int(ms / 1000))
        millis = max(0, int(ms % 1000 / 10))
        return f"{total // 60:02d}:{total % 60:02d}.{millis:02d}"
