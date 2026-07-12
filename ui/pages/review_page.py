"""
Emotion Data Studio - Review Workspace

A professional NLE-style review surface:
- left media bin / clip queue
- center preview monitor + timeline strip
- right inspector for AI prediction, transcript, labels and notes
"""

from __future__ import annotations

import os
import json
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import Qt, QUrl, Slot, Signal, QRectF, QPointF
from PySide6.QtGui import (
    QKeySequence,
    QShortcut,
    QPainter,
    QPen,
    QBrush,
    QFont,
    QColor,
    QPolygonF,
    QMouseEvent,
    QPaintEvent,
)
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
from PySide6.QtMultimediaWidgets import QVideoWidget
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QFrame,
    QSplitter,
    QComboBox,
    QCheckBox,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPlainTextEdit,
    QProgressBar,
    QMessageBox,
    QSlider,
    QSizePolicy,
    QScrollArea,
    QApplication,
)

from ui.styles.theme import EMOTION_MAP, Colors, Spacing
from ui.widgets import PageHeader, Card, ActionButton, StatusPill

try:
    import qtawesome as qta
    _HAS_QTA = True
except ImportError:
    _HAS_QTA = False



class ClipListItem(QListWidgetItem):
    """List item that stores a clip dict."""

    def __init__(self, clip: dict):
        label = clip.get("display_name") or f"Clip {clip.get('clip_index', 0):03d}"
        status = clip.get("status") or "pending"
        emotion = clip.get("user_emotion") or clip.get("ai_emotion") or "unknown"
        super().__init__(f"{label}\n{status} | {emotion} | {clip.get('duration', 0):.1f}s")
        self.clip = clip
        self.setToolTip(clip.get("clip_path") or "")


class ScoreRow(QFrame):
    """Compact score row for inspector."""

    def __init__(self, label: str, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 2, 0, 2)
        layout.setSpacing(8)
        self.label = QLabel(label)
        self.label.setFixedWidth(100)
        self.label.setObjectName("statLabel")
        layout.addWidget(self.label)
        self.progress = QProgressBar()
        self.progress.setMinimum(0)
        self.progress.setMaximum(100)
        layout.addWidget(self.progress, stretch=1)
        self.value = QLabel("0%")
        self.value.setFixedWidth(44)
        self.value.setAlignment(Qt.AlignmentFlag.AlignRight)
        layout.addWidget(self.value)

    def set_score(self, score: float):
        pct = max(0, min(100, int((score or 0) * 100)))
        self.progress.setValue(pct)
        self.value.setText(f"{pct}%")


class ClipTimelineBar(QWidget):
    """Adobe Premiere-style timeline: shows all clips as colored segments,
    current position as playhead, click/drag to select clip + seek."""

    clip_selected = Signal(int)  # emits clip index when user clicks/selects a segment
    seek_requested = Signal(int)  # emits absolute ms position when user drags/clicks

    def __init__(self, parent=None):
        super().__init__(parent)
        self._clips: list[dict] = []
        self._total_duration_ms = 0
        self._current_index = -1
        self._playhead_position_ms = 0
        self._is_scrubbing = False

        self.setFixedHeight(95)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setMouseTracking(True)

    def set_clips(self, clips: list[dict], total_duration_ms: int):
        self._clips = clips
        self._total_duration_ms = total_duration_ms
        self.update()

    def set_current_clip(self, index: int):
        if 0 <= index < len(self._clips):
            self._current_index = index
            self.update()

    def set_playhead_position(self, position_ms: int):
        if not self._is_scrubbing:
            self._playhead_position_ms = max(0, min(self._total_duration_ms, position_ms))
            self.update()

    def is_scrubbing(self) -> bool:
        return self._is_scrubbing

    def clear(self):
        self._clips = []
        self._total_duration_ms = 0
        self._current_index = -1
        self._playhead_position_ms = 0
        self._is_scrubbing = False
        self.update()

    def time_to_x(self, time_ms: int) -> float:
        if self._total_duration_ms <= 0:
            return 0
        return (time_ms / self._total_duration_ms) * self.width()

    def x_to_time(self, x: float) -> int:
        if self.width() <= 0:
            return 0
        pct = max(0.0, min(1.0, x / self.width()))
        return int(pct * self._total_duration_ms)

    def paintEvent(self, event: QPaintEvent):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        w = self.width()
        h = self.height()

        # 1. Background
        painter.fillRect(0, 0, w, h, QBrush(QColor("#0d0d15")))

        # Border
        pen = QPen(QColor("rgba(255, 255, 255, 0.08)"), 1)
        painter.setPen(pen)
        painter.drawRect(0, 0, w - 1, h - 1)

        # Ruler Area Background (top 20px)
        painter.fillRect(0, 0, w, 20, QBrush(QColor("#09090f")))
        painter.setPen(QPen(QColor("rgba(255, 255, 255, 0.06)"), 1))
        painter.drawLine(0, 20, w, 20)

        # Draw Ruler Ticks
        if self._total_duration_ms > 0:
            painter.setPen(QPen(QColor("rgba(255, 255, 255, 0.2)"), 1))
            font = QFont("Inter", 8)
            painter.setFont(font)

            total_sec = self._total_duration_ms / 1000.0
            tick_interval_sec = 5.0
            if total_sec > 120:
                tick_interval_sec = 20.0
            elif total_sec > 60:
                tick_interval_sec = 10.0
            elif total_sec < 15:
                tick_interval_sec = 2.0

            num_ticks = int(total_sec / tick_interval_sec)
            for i in range(num_ticks + 1):
                sec = i * tick_interval_sec
                x = (sec * 1000 / self._total_duration_ms) * w
                painter.drawLine(QPointF(x, 12), QPointF(x, 20))

                m = int(sec // 60)
                s = int(sec % 60)
                time_str = f"{m:02d}:{s:02d}"
                rect = QRectF(x - 20, 0, 40, 12)
                painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, time_str)

        # 2. Clips Track
        if self._total_duration_ms > 0:
            track_y = 23
            track_h = h - track_y - 6

            for idx, clip in enumerate(self._clips):
                start = clip.get("start_time", 0) * 1000
                end = clip.get("end_time", 0) * 1000

                x_start = (start / self._total_duration_ms) * w
                x_end = (end / self._total_duration_ms) * w
                clip_w = max(2.0, x_end - x_start)

                status = clip.get("status", "pending")
                is_selected = (idx == self._current_index)

                # Determine colors based on status
                if status == "approved":
                    base_color = QColor("#00b894")  # Green
                    alpha = 210 if is_selected else 90
                elif status == "rejected":
                    base_color = QColor("#e17055")  # Red/orange
                    alpha = 210 if is_selected else 90
                else:
                    # Pending/Needs Review/AI Labeled -> Grayish/Subtle Purple
                    base_color = QColor("#6c5ce7") if is_selected else QColor(100, 100, 110)
                    alpha = 140 if is_selected else 40

                fill_color = QColor(base_color)
                fill_color.setAlpha(alpha)

                clip_rect = QRectF(x_start, track_y, clip_w, track_h)
                painter.fillRect(clip_rect, QBrush(fill_color))

                # Highlight borders
                if is_selected:
                    border_pen = QPen(QColor("#a29bfe"), 2)
                    painter.setPen(border_pen)
                    painter.drawRect(clip_rect.adjusted(1, 1, -1, -1))
                else:
                    border_pen = QPen(QColor("rgba(0, 0, 0, 0.35)"), 1)
                    painter.setPen(border_pen)
                    painter.drawRect(clip_rect)

                # Text inside segment
                if clip_w > 40:
                    label_font = QFont("Inter", 9, QFont.Weight.Bold if is_selected else QFont.Weight.Normal)
                    painter.setFont(label_font)
                    painter.setPen(QColor("#ffffff" if is_selected else "#b8b6c4"))

                    clip_num = clip.get("clip_index", 0)
                    emo_key = clip.get("user_emotion") or clip.get("ai_emotion") or "unknown"
                    emo_info = EMOTION_MAP.get(emo_key, {})
                    emo_lbl = emo_info.get("emoji", "")

                    label_text = f"{clip_num:03d} {emo_lbl}"

                    metrics = painter.fontMetrics()
                    elided_text = metrics.elidedText(label_text, Qt.TextElideMode.ElideRight, int(clip_w - 6))
                    painter.drawText(clip_rect.adjusted(3, 0, -3, 0), Qt.AlignmentFlag.AlignCenter, elided_text)

        # 3. Playhead
        if self._total_duration_ms > 0:
            playhead_x = (self._playhead_position_ms / self._total_duration_ms) * w

            # Vertical line
            line_pen = QPen(QColor("#6c5ce7"), 1.5)
            painter.setPen(line_pen)
            painter.drawLine(QPointF(playhead_x, 0), QPointF(playhead_x, h))

            # Handle (inverted triangle)
            painter.setBrush(QBrush(QColor("#6c5ce7")))
            painter.setPen(Qt.PenStyle.NoPen)

            triangle = QPolygonF([
                QPointF(playhead_x - 6, 0),
                QPointF(playhead_x + 6, 0),
                QPointF(playhead_x + 6, 12),
                QPointF(playhead_x, 18),
                QPointF(playhead_x - 6, 12)
            ])
            painter.drawPolygon(triangle)

            # White dot
            painter.setBrush(QBrush(QColor("#ffffff")))
            painter.drawEllipse(QPointF(playhead_x, 6), 2.5, 2.5)

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton:
            self._is_scrubbing = True
            target_ms = self.x_to_time(event.position().x())
            self._playhead_position_ms = target_ms
            self.seek_requested.emit(target_ms)
            self.update()

    def mouseMoveEvent(self, event: QMouseEvent):
        if self._is_scrubbing:
            target_ms = self.x_to_time(event.position().x())
            self._playhead_position_ms = target_ms
            self.seek_requested.emit(target_ms)
            self.update()

    def mouseReleaseEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton:
            self._is_scrubbing = False
            target_ms = self.x_to_time(event.position().x())
            self._playhead_position_ms = target_ms
            self.seek_requested.emit(target_ms)
            self.update()


class FaceOverlayVideoWidget(QVideoWidget):
    """Video widget that paints face detection boxes from detections.json."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._detections: list[dict] = []
        self._current_sec = 0.0
        self._show_boxes = True
        self._max_time_delta = 0.65
        self._video_size: tuple[int, int] | None = None

    def set_detections(self, detections: list[dict]):
        self._detections = detections or []
        self._video_size = self._infer_video_size(self._detections)
        self.update()

    def set_current_position_ms(self, position_ms: int):
        self._current_sec = max(0.0, position_ms / 1000.0)
        self.update()

    def set_show_boxes(self, show: bool):
        self._show_boxes = show
        self.update()

    def paintEvent(self, event: QPaintEvent):
        super().paintEvent(event)
        if not self._show_boxes or not self._detections:
            return
        detection = self._nearest_detection(self._current_sec)
        if not detection or not detection.get("faces"):
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        font = QFont("Inter", 9, QFont.Weight.Bold)
        painter.setFont(font)

        for face in detection.get("faces", []):
            bbox = face.get("bbox") or []
            if len(bbox) != 4:
                continue
            rect = self._map_bbox(bbox)
            color = QColor("#00e5ff")
            painter.setPen(QPen(color, 2))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRect(rect)

            track_id = face.get('track_id', face.get('face_id', 0))
            label = f"mặt_{track_id} {float(face.get('confidence', 0)):.2f}"
            label_rect = QRectF(rect.left(), max(0, rect.top() - 22), max(92, rect.width()), 20)
            bg = QColor("#000000")
            bg.setAlpha(180)
            painter.fillRect(label_rect, bg)
            painter.setPen(QPen(color, 1))
            painter.drawText(label_rect.adjusted(4, 0, -4, 0), Qt.AlignmentFlag.AlignVCenter, label)

    def _nearest_detection(self, current_sec: float) -> dict | None:
        best = None
        best_delta = float("inf")
        for item in self._detections:
            delta = abs(float(item.get("timestamp", 0.0)) - current_sec)
            if delta < best_delta:
                best = item
                best_delta = delta
        return best if best_delta <= self._max_time_delta else None

    def _map_bbox(self, bbox: list) -> QRectF:
        src_w, src_h = self._video_size or self._infer_size_from_bbox(bbox)
        widget_w, widget_h = max(1, self.width()), max(1, self.height())
        scale = min(widget_w / max(1, src_w), widget_h / max(1, src_h))
        draw_w = src_w * scale
        draw_h = src_h * scale
        offset_x = (widget_w - draw_w) / 2
        offset_y = (widget_h - draw_h) / 2
        x1, y1, x2, y2 = [float(v) for v in bbox]
        return QRectF(
            offset_x + x1 * scale,
            offset_y + y1 * scale,
            max(1.0, (x2 - x1) * scale),
            max(1.0, (y2 - y1) * scale),
        )

    @staticmethod
    def _infer_video_size(detections: list[dict]) -> tuple[int, int] | None:
        for item in detections:
            size = item.get("frame_size")
            if isinstance(size, list) and len(size) == 2 and size[0] and size[1]:
                return int(size[0]), int(size[1])
        return None

    @staticmethod
    def _infer_size_from_bbox(bbox: list) -> tuple[int, int]:
        x2 = max(float(bbox[2]), 640.0)
        y2 = max(float(bbox[3]), 360.0)
        return int(x2), int(y2)


class ReviewPage(QWidget):
    """Review & Labeling Studio."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._all_clips: list[dict] = []
        self._clips: list[dict] = []
        self._active_video_id: str | None = None
        self._current_index = -1
        self._current_clip: dict | None = None
        self._is_scrubbing = False
        self._setup_ui()
        self._setup_shortcuts()

    def _setup_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        self._build_toolbar(root)

        main_splitter = QSplitter(Qt.Orientation.Horizontal)
        main_splitter.addWidget(self._build_media_bin())
        main_splitter.addWidget(self._build_center_workspace())
        main_splitter.addWidget(self._build_inspector())
        main_splitter.setSizes([230, 580, 300])
        main_splitter.setChildrenCollapsible(False)
        root.addWidget(main_splitter, stretch=1)

    def _build_toolbar(self, parent_layout):
        bar = QFrame()
        bar.setObjectName("card")
        bar_layout = QVBoxLayout(bar)
        bar_layout.setContentsMargins(12, 6, 12, 6)
        bar_layout.setSpacing(4)

        # ── Hàng 1: bộ đếm + tìm kiếm + nút tải lại ──────────────────────
        row1 = QHBoxLayout()
        row1.setSpacing(8)

        self.clip_counter = QLabel("Clip 0 / 0")
        self.clip_counter.setObjectName("accentText")
        self.clip_counter.setFixedWidth(90)
        row1.addWidget(self.clip_counter)

        self.search_box = QLineEdit()
        self.search_box.setPlaceholderText("Tìm kiếm transcript / tên clip…")
        self.search_box.setMinimumHeight(36)
        self.search_box.textChanged.connect(self._apply_filters)
        row1.addWidget(self.search_box, stretch=1)

        self.reload_btn = ActionButton("Tải lại", "fa5s.sync-alt", variant="ghost")
        self.reload_btn.clicked.connect(self.refresh_data)
        row1.addWidget(self.reload_btn)
        bar_layout.addLayout(row1)

        # ── Row 2: filters ──────────────────────────────────────
        row2 = QHBoxLayout()
        row2.setSpacing(Spacing.SM)

        status_lbl = QLabel("Trạng thái:")
        status_lbl.setStyleSheet(f"color: {Colors.TEXT_SECONDARY}; background: transparent;")
        row2.addWidget(status_lbl)
        self.filter_status = QComboBox()
        self.filter_status.setMinimumWidth(110)
        self.filter_status.setMinimumHeight(36)
        self.filter_status.addItems(["Tất cả", "pending", "needs_review", "ai_labeled", "approved", "rejected", "failed"])
        self.filter_status.currentTextChanged.connect(self._apply_filters)
        row2.addWidget(self.filter_status)

        emo_lbl = QLabel("Cảm xúc:")
        emo_lbl.setStyleSheet(f"color: {Colors.TEXT_SECONDARY}; background: transparent;")
        row2.addWidget(emo_lbl)
        self.filter_emotion = QComboBox()
        self.filter_emotion.setMinimumWidth(110)
        self.filter_emotion.setMinimumHeight(36)
        emotion_items = ["Tất cả"] + [info.get("label", k) for k, info in EMOTION_MAP.items()]
        self._emotion_filter_keys = ["Tất cả"] + list(EMOTION_MAP.keys())
        self.filter_emotion.addItems(emotion_items)
        self.filter_emotion.currentIndexChanged.connect(self._apply_filters)
        row2.addWidget(self.filter_emotion)

        self.only_incongruity = QCheckBox("Chỉ clip không đồng nhất")
        self.only_incongruity.stateChanged.connect(self._apply_filters)
        row2.addWidget(self.only_incongruity)

        row2.addStretch()
        bar_layout.addLayout(row2)

        parent_layout.addWidget(bar)

    def _build_media_bin(self) -> QWidget:
        panel = QFrame()
        panel.setObjectName("card")
        panel.setMinimumWidth(180)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        title = QLabel("Danh Sách Clip")
        title.setObjectName("sectionTitle")
        layout.addWidget(title)

        self.video_summary = QLabel("Chưa tải clip")
        self.video_summary.setObjectName("mutedText")
        layout.addWidget(self.video_summary)

        self.clip_list = QListWidget()
        self.clip_list.setObjectName("mediaBinList")
        self.clip_list.currentRowChanged.connect(self._on_clip_row_changed)
        layout.addWidget(self.clip_list, stretch=1)

        hint = QLabel("⏎ Space • A duyệt • R từ chối • 1–7 nhãn • ←/→ chuyển")
        hint.setWordWrap(True)
        hint.setObjectName("mutedText")
        layout.addWidget(hint)
        return panel

    def _build_center_workspace(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)

        monitor_card = QFrame()
        monitor_card.setObjectName("cardElevated")
        monitor_layout = QVBoxLayout(monitor_card)
        monitor_layout.setSpacing(8)

        header = QHBoxLayout()
        self.clip_title = QLabel("Màn Hình Xem Trước")
        self.clip_title.setObjectName("sectionTitle")
        header.addWidget(self.clip_title)
        header.addStretch()
        self.clip_meta = QLabel("Chưa chọn clip")
        self.clip_meta.setObjectName("mutedText")
        header.addWidget(self.clip_meta)
        self.show_face_boxes = QCheckBox("Khung mặt")
        self.show_face_boxes.setChecked(True)
        self.show_face_boxes.setToolTip("Hiển thị khung nhận diện khuôn mặt trong khung xem trước")
        self.show_face_boxes.stateChanged.connect(lambda state: self.video_widget.set_show_boxes(state == Qt.CheckState.Checked.value))
        header.addWidget(self.show_face_boxes)
        monitor_layout.addLayout(header)

        self.video_widget = FaceOverlayVideoWidget()
        self.video_widget.setMinimumHeight(360)
        self.video_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.video_widget.setStyleSheet("background-color: #000; border-radius: 8px;")
        monitor_layout.addWidget(self.video_widget, stretch=1)

        self.media_player = QMediaPlayer(self)
        self.audio_output = QAudioOutput(self)
        self.media_player.setAudioOutput(self.audio_output)
        self.media_player.setVideoOutput(self.video_widget)
        self.media_player.positionChanged.connect(self._on_position_changed)
        self.media_player.durationChanged.connect(self._on_duration_changed)

        controls = QHBoxLayout()
        controls.setSpacing(Spacing.SM)
        self.prev_btn = ActionButton("", "fa5s.step-backward", variant="ghost")
        self.prev_btn.setToolTip("Clip trước (←)")
        self.prev_btn.setFixedWidth(40)
        self.prev_btn.clicked.connect(self._go_prev)
        controls.addWidget(self.prev_btn)
        self.play_btn = ActionButton("", "fa5s.play", variant="primary")
        self.play_btn.setToolTip("Phát / Dừng (Space)")
        self.play_btn.setFixedWidth(40)
        self.play_btn.clicked.connect(self._toggle_play)
        controls.addWidget(self.play_btn)
        self.next_btn = ActionButton("", "fa5s.step-forward", variant="ghost")
        self.next_btn.setToolTip("Clip tiếp (→)")
        self.next_btn.setFixedWidth(40)
        self.next_btn.clicked.connect(self._go_next)
        controls.addWidget(self.next_btn)

        self.position_slider = QSlider(Qt.Orientation.Horizontal)
        self.position_slider.setRange(0, 0)
        self.position_slider.sliderPressed.connect(lambda: setattr(self, "_is_scrubbing", True))
        self.position_slider.sliderReleased.connect(self._seek_to_slider)
        controls.addWidget(self.position_slider, stretch=1)

        self.time_label = QLabel("00:00 / 00:00")
        self.time_label.setObjectName("mutedText")
        controls.addWidget(self.time_label)
        monitor_layout.addLayout(controls)
        layout.addWidget(monitor_card, stretch=3)

        timeline_card = QFrame()
        timeline_card.setObjectName("card")
        timeline_layout = QVBoxLayout(timeline_card)
        timeline_title = QLabel("Dòng Thời Gian")
        timeline_title.setObjectName("sectionTitle")
        timeline_layout.addWidget(timeline_title)
        self.timeline_bar = ClipTimelineBar()
        self.timeline_bar.clip_selected.connect(self._select_clip)
        self.timeline_bar.seek_requested.connect(self._on_timeline_seek)
        timeline_layout.addWidget(self.timeline_bar)
        layout.addWidget(timeline_card)
        return panel

    def _build_inspector(self) -> QWidget:
        # Bao bọc toàn bộ inspector trong QScrollArea
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setMinimumWidth(240)

        panel = QFrame()
        panel.setObjectName("card")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        # ── AI prediction ──────────────────────────────────────────────
        ai_frame = QFrame()
        ai_frame.setObjectName("cardElevated")
        ai_layout = QVBoxLayout(ai_frame)
        ai_layout.setContentsMargins(10, 8, 10, 8)
        ai_layout.setSpacing(4)

        ai_header = QLabel("Thanh Tra")
        ai_header.setObjectName("sectionTitle")
        ai_layout.addWidget(ai_header)

        self.pred_emotion_label = QLabel("AI: -")
        self.pred_emotion_label.setObjectName("statValue")
        ai_layout.addWidget(self.pred_emotion_label)

        stats_grid = QHBoxLayout()
        left_col = QVBoxLayout()
        self.confidence_label = QLabel("Độ tin cậy: -")
        self.confidence_label.setObjectName("statLabel")
        left_col.addWidget(self.confidence_label)
        self.agreement_label = QLabel("Đồng thuận: -")
        self.agreement_label.setObjectName("statLabel")
        left_col.addWidget(self.agreement_label)
        stats_grid.addLayout(left_col)
        self.quality_label = QLabel("Chất lượng: -")
        self.quality_label.setObjectName("statLabel")
        stats_grid.addWidget(self.quality_label)
        ai_layout.addLayout(stats_grid)

        self.warning_label = QLabel("")
        self.warning_label.setObjectName("warningText")
        self.warning_label.setWordWrap(True)
        self.warning_label.setVisible(False)
        ai_layout.addWidget(self.warning_label)
        layout.addWidget(ai_frame)

        # ── Thông tin cắt đoạn thông minh ─────────────────────────────
        segment_frame = QFrame()
        segment_frame.setObjectName("subtleCard")
        segment_layout = QVBoxLayout(segment_frame)
        segment_layout.setContentsMargins(10, 8, 10, 8)
        segment_layout.setSpacing(4)
        segment_title = QLabel("Thông Tin Cắt Đoạn")
        segment_title.setObjectName("sectionTitle")
        segment_layout.addWidget(segment_title)
        self.segment_source_label = QLabel("Nguồn cắt: -")
        self.segment_source_label.setObjectName("statLabel")
        segment_layout.addWidget(self.segment_source_label)
        self.segment_face_label = QLabel("Tỷ lệ có mặt: -")
        self.segment_face_label.setObjectName("statLabel")
        segment_layout.addWidget(self.segment_face_label)
        self.segment_speech_label = QLabel("Tỷ lệ hội thoại: -")
        self.segment_speech_label.setObjectName("statLabel")
        segment_layout.addWidget(self.segment_speech_label)
        self.segment_quality_label = QLabel("Đánh giá đoạn: -")
        self.segment_quality_label.setObjectName("statLabel")
        segment_layout.addWidget(self.segment_quality_label)
        layout.addWidget(segment_frame)

        # ── Điểm cảm xúc ───────────────────────────────────────────────
        score_title = QLabel("Điểm Cảm Xúc")
        score_title.setObjectName("sectionTitle")
        layout.addWidget(score_title)
        self.score_rows: dict[str, ScoreRow] = {}
        for key, info in EMOTION_MAP.items():
            row = ScoreRow(info.get("label", key))
            self.score_rows[key] = row
            layout.addWidget(row)

        # ── Lời thoại ─────────────────────────────────────────────────
        transcript_title = QLabel("Lời Thoại")
        transcript_title.setObjectName("sectionTitle")
        layout.addWidget(transcript_title)
        self.transcript_text = QPlainTextEdit()
        self.transcript_text.setReadOnly(True)
        self.transcript_text.setFixedHeight(80)
        self.transcript_text.setPlaceholderText("Không có lời thoại")
        layout.addWidget(self.transcript_text)

        # ── Nhãn thủ công ────────────────────────────────────────────────
        label_header = QHBoxLayout()
        label_title = QLabel("Nhãn Thủ Công")
        label_title.setObjectName("sectionTitle")
        label_header.addWidget(label_title)
        label_header.addStretch()
        layout.addLayout(label_header)

        self._emotion_buttons: dict[str, QPushButton] = {}
        # 2 cột cho các nút nhãn
        btn_grid = QWidget()
        btn_grid_layout = QHBoxLayout(btn_grid)
        btn_grid_layout.setContentsMargins(0, 0, 0, 0)
        btn_grid_layout.setSpacing(6)
        col_left  = QVBoxLayout()
        col_right = QVBoxLayout()
        col_left.setSpacing(4)
        col_right.setSpacing(4)
        items = list(EMOTION_MAP.items())
        mid = (len(items) + 1) // 2
        for i, (key, info) in enumerate(items):
            btn = QPushButton(f"{info.get('shortcut', '')}. {info.get('label', key)}")
            btn.setObjectName("emotionBtn")
            btn.setCheckable(True)
            btn.setMinimumHeight(30)
            btn.clicked.connect(lambda checked=False, k=key: self._on_emotion_selected(k))
            self._emotion_buttons[key] = btn
            if i < mid:
                col_left.addWidget(btn)
            else:
                col_right.addWidget(btn)
        btn_grid_layout.addLayout(col_left)
        btn_grid_layout.addLayout(col_right)
        layout.addWidget(btn_grid)

        # ── Sentiment Score Slider [–3, +3] ──────────────────────────────
        sentiment_frame = QFrame()
        sentiment_frame.setObjectName("subtleCard")
        sentiment_layout = QVBoxLayout(sentiment_frame)
        sentiment_layout.setContentsMargins(10, 8, 10, 8)
        sentiment_layout.setSpacing(4)

        sentiment_header = QHBoxLayout()
        sentiment_title = QLabel("★ Sentiment Score")
        sentiment_title.setObjectName("sectionTitle")
        sentiment_header.addWidget(sentiment_title)
        sentiment_header.addStretch()
        self.sentiment_value_label = QLabel("0.0")
        self.sentiment_value_label.setObjectName("statValue")
        self.sentiment_value_label.setFixedWidth(50)
        self.sentiment_value_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        sentiment_header.addWidget(self.sentiment_value_label)
        sentiment_layout.addLayout(sentiment_header)

        self.sentiment_slider = QSlider(Qt.Orientation.Horizontal)
        self.sentiment_slider.setMinimum(-30)  # -3.0 × 10
        self.sentiment_slider.setMaximum(30)   # +3.0 × 10
        self.sentiment_slider.setValue(0)
        self.sentiment_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.sentiment_slider.setTickInterval(10)  # marks at -3, -2, -1, 0, 1, 2, 3
        self.sentiment_slider.setSingleStep(1)     # 0.1 precision
        self.sentiment_slider.setPageStep(5)       # 0.5 jumps
        self.sentiment_slider.setFixedHeight(28)
        self.sentiment_slider.valueChanged.connect(self._on_sentiment_slider_changed)
        sentiment_layout.addWidget(self.sentiment_slider)

        # Scale labels row
        scale_row = QHBoxLayout()
        scale_row.setContentsMargins(0, 0, 0, 0)
        neg_label = QLabel("-3 Tiêu cực")
        neg_label.setObjectName("mutedText")
        scale_row.addWidget(neg_label)
        scale_row.addStretch()
        neutral_label = QLabel("0")
        neutral_label.setObjectName("mutedText")
        neutral_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        scale_row.addWidget(neutral_label)
        scale_row.addStretch()
        pos_label = QLabel("Tích cực +3")
        pos_label.setObjectName("mutedText")
        pos_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        scale_row.addWidget(pos_label)
        sentiment_layout.addLayout(scale_row)

        layout.addWidget(sentiment_frame)

        # ── Ghi chú ──────────────────────────────────────────────────
        notes_title = QLabel("Ghi Chú Reviewer")
        notes_title.setObjectName("sectionTitle")
        layout.addWidget(notes_title)
        self.notes_text = QPlainTextEdit()
        self.notes_text.setFixedHeight(70)
        self.notes_text.setPlaceholderText("Ghi chú tuù chọn...")
        layout.addWidget(self.notes_text)

        # ── Nút hành động chính ────────────────────────────────────────
        actions = QHBoxLayout()
        actions.setSpacing(8)
        self.approve_btn = QPushButton("✔  Duyệt (A)")
        self.approve_btn.setObjectName("successBtn")
        self.approve_btn.setMinimumHeight(36)
        self.approve_btn.clicked.connect(self._on_approve)
        actions.addWidget(self.approve_btn)
        self.reject_btn = QPushButton("✖  Từ chối (R)")
        self.reject_btn.setObjectName("dangerBtn")
        self.reject_btn.setMinimumHeight(36)
        self.reject_btn.clicked.connect(self._on_reject)
        actions.addWidget(self.reject_btn)
        layout.addLayout(actions)

        self.save_btn = QPushButton("💾  Lưu Ghi Chú / Nhãn")
        self.save_btn.setMinimumHeight(32)
        self.save_btn.clicked.connect(self._save_current_review)
        layout.addWidget(self.save_btn)

        self.gemini_btn = QPushButton("🤖  Gemini Verify")
        self.gemini_btn.setMinimumHeight(32)
        self.gemini_btn.setToolTip("Gọi Gemini 2.5 Flash để verify/re-score clip hiện tại")
        self.gemini_btn.clicked.connect(self._on_gemini_verify)
        layout.addWidget(self.gemini_btn)

        layout.addStretch()
        scroll.setWidget(panel)
        return scroll

    def _setup_shortcuts(self):
        for key, info in EMOTION_MAP.items():
            QShortcut(QKeySequence(info["shortcut"]), self).activated.connect(
                lambda k=key: self._on_emotion_selected(k)
            )
        QShortcut(QKeySequence(Qt.Key.Key_Left), self).activated.connect(self._go_prev)
        QShortcut(QKeySequence(Qt.Key.Key_Right), self).activated.connect(self._go_next)
        QShortcut(QKeySequence("A"), self).activated.connect(self._on_approve)
        QShortcut(QKeySequence("R"), self).activated.connect(self._on_reject)
        QShortcut(QKeySequence(Qt.Key.Key_Space), self).activated.connect(self._toggle_play)
        QShortcut(QKeySequence("Ctrl+S"), self).activated.connect(self._save_current_review)

    # Data -----------------------------------------------------------

    def refresh_data(self):
        self._load_clips()

    def set_active_video(self, video_id: str | None):
        self._active_video_id = video_id
        self._load_clips()

    def _load_clips(self):
        try:
            from backend.database.local_db import get_session
            from backend.database.models import Clip, Video

            session = get_session()
            try:
                query = session.query(Clip)
                if self._active_video_id:
                    query = query.filter(Clip.video_id == self._active_video_id)
                    clips = query.order_by(Clip.start_time.asc()).all()
                else:
                    clips = query.join(Video).order_by(Video.created_at.desc(), Clip.start_time.asc()).all()
                self._all_clips = [self._clip_to_dict(clip) for clip in clips]
            finally:
                session.close()
        except Exception as exc:
            self._all_clips = []
            QMessageBox.warning(self, "Lỗi tải clip", f"Không thể tải danh sách clip:\n{exc}")
        self._apply_filters()

    def _clip_to_dict(self, clip) -> dict:
        display_name = f"Clip {clip.clip_index:03d}"
        per_model = clip.per_model_scores or {}
        gem_meta = per_model.get("gemini_prefilter", {})
        segment_meta = per_model.get("segment", {})
        return {
            "id": clip.id,
            "video_id": clip.video_id,
            "clip_index": clip.clip_index,
            "display_name": display_name,
            "clip_path": clip.clip_path,
            "start_time": clip.start_time or 0,
            "end_time": clip.end_time or 0,
            "duration": clip.duration or 0,
            "transcript": clip.transcript or "",
            "ai_emotion": clip.ai_emotion,
            "ai_confidence": clip.ai_confidence or 0,
            "ai_agreement": clip.ai_agreement,
            "quality_score": clip.quality_score or 0,
            "has_incongruity": bool(clip.has_incongruity),
            "user_emotion": clip.user_emotion,
            "sentiment_score": clip.sentiment_score,
            "reviewer_notes": clip.reviewer_notes or "",
            "status": clip.status or "pending",
            "all_scores": clip.all_scores or {},
            "per_model_scores": per_model,
            "num_faces": clip.num_faces or 0,
            "face_ratio": clip.face_ratio or gem_meta.get("face_coverage", segment_meta.get("face_coverage", 0)),
            "speech_coverage": gem_meta.get("speech_coverage", segment_meta.get("speech_coverage", 0)),
            "has_speech": clip.has_speech or False,
        }

    def _apply_filters(self):
        text = self.search_box.text().strip().lower() if hasattr(self, "search_box") else ""
        status = self.filter_status.currentText() if hasattr(self, "filter_status") else "Tất cả"

        # filter_emotion dùng index để map ngược sang key gốc
        emotion_key = "Tất cả"
        if hasattr(self, "filter_emotion") and hasattr(self, "_emotion_filter_keys"):
            idx = self.filter_emotion.currentIndex()
            emotion_key = self._emotion_filter_keys[idx] if 0 <= idx < len(self._emotion_filter_keys) else "Tất cả"

        only_incongruity = self.only_incongruity.isChecked() if hasattr(self, "only_incongruity") else False

        filtered = []
        for clip in self._all_clips:
            if status != "Tất cả" and clip.get("status") != status:
                continue
            effective_emotion = clip.get("user_emotion") or clip.get("ai_emotion")
            if emotion_key != "Tất cả" and effective_emotion != emotion_key:
                continue
            if only_incongruity and not clip.get("has_incongruity"):
                continue
            haystack = " ".join([
                clip.get("display_name", ""),
                clip.get("clip_path", "") or "",
                clip.get("transcript", "") or "",
            ]).lower()
            if text and text not in haystack:
                continue
            filtered.append(clip)

        self._clips = filtered
        self._populate_lists()

        scope = "video đang chọn" if self._active_video_id else "tất cả video"
        self.video_summary.setText(f"{len(self._clips)} hiển thị / {len(self._all_clips)} clip ({scope})")  
        if self._clips:
            self._select_clip(0)
        else:
            self._clear_display()

    def _populate_lists(self):
        self.clip_list.blockSignals(True)
        self.clip_list.clear()
        for clip in self._clips:
            self.clip_list.addItem(ClipListItem(clip))
        self.clip_list.blockSignals(False)

        # Update timeline_bar
        total_end = max([c.get("end_time", 0) for c in self._all_clips], default=0)
        total_ms = int(total_end * 1000)
        if total_ms <= 0:
            total_ms = sum(int(c.get("duration", 0) * 1000) for c in self._clips)
        self.timeline_bar.set_clips(self._clips, total_ms)

    # Selection / playback -----------------------------------------

    def _select_clip(self, index: int):
        if not self._clips or index < 0 or index >= len(self._clips):
            return
        self._current_index = index
        self._current_clip = self._clips[index]
        self.clip_list.blockSignals(True)
        self.clip_list.setCurrentRow(index)
        self.clip_list.blockSignals(False)
        self.timeline_bar.set_current_clip(index)
        self._load_current_clip()

    def _on_clip_row_changed(self, row: int):
        self._select_clip(row)


    def _load_current_clip(self):
        clip = self._current_clip
        if not clip:
            self._clear_display()
            return
        self.clip_counter.setText(f"Clip {self._current_index + 1} / {len(self._clips)}")
        self.clip_title.setText(clip.get("display_name", "Clip"))
        speech_indicator = "🔊" if clip.get("has_speech") else "🔇"
        self.clip_meta.setText(
            f"{clip.get('start_time', 0):.2f}s - {clip.get('end_time', 0):.2f}s | "
            f"{clip.get('duration', 0):.1f}s | faces: {clip.get('num_faces', 0)} | "
            f"{speech_indicator} speech"
        )

        path = clip.get("clip_path")
        self.video_widget.set_detections(self._load_face_detections_for_clip(clip))
        if path and os.path.exists(path):
            self.media_player.setSource(QUrl.fromLocalFile(path))
            self._is_preview_mode = True
            self.audio_output.setMuted(True)
            self.media_player.setPlaybackRate(2.5)
            self.media_player.play()
            self.play_btn.setIcon(qta.icon("fa5s.play", color=Colors.TEXT_PRIMARY))
        else:
            self.media_player.setSource(QUrl())
            self.clip_meta.setText(self.clip_meta.text() + " | missing file")
            self._is_preview_mode = False

        ai_emotion = clip.get("ai_emotion") or "unknown"
        user_emotion = clip.get("user_emotion")
        ai_info = EMOTION_MAP.get(ai_emotion, {"label": ai_emotion})
        self.pred_emotion_label.setText(f"AI: {ai_info.get('label', ai_emotion)}")
        self.confidence_label.setText(f"Độ tin cậy: {int((clip.get('ai_confidence') or 0) * 100)}%")
        self.agreement_label.setText(f"Đồng thuận: {clip.get('ai_agreement') or '-'}")
        self.quality_label.setText(f"Chất lượng: {(clip.get('quality_score') or 0):.2f}")
        self.warning_label.setVisible(bool(clip.get("has_incongruity")))
        self.warning_label.setText("Phát hiện không đồng nhất giữa các mô hình. Vui lòng kiểm tra kỹ.")
        self._update_segment_info(clip)
        self.transcript_text.setPlainText(clip.get("transcript") or "")
        self.notes_text.setPlainText(clip.get("reviewer_notes") or "")

        all_scores = clip.get("all_scores") or {}
        for key, row in self.score_rows.items():
            row.set_score(all_scores.get(key, 0))

        for key, btn in self._emotion_buttons.items():
            is_checked = (key == user_emotion)
            btn.setChecked(is_checked)
            if is_checked:
                color = EMOTION_MAP.get(key, {}).get("color", "#6c5ce7")
                btn.setStyleSheet(f"background-color: {color}22; border: 1.5px solid {color}; color: {color}; font-weight: bold;")
        else:
                btn.setStyleSheet("")

        # Sentiment slider
        sentiment = clip.get("sentiment_score")
        if sentiment is None:
            # Auto-map from emotion if no explicit score yet
            from backend.database.models import SENTIMENT_MAPPING
            effective_emotion = user_emotion or ai_emotion or "neutral"
            sentiment = SENTIMENT_MAPPING.get(effective_emotion, 0.0)
        self.sentiment_slider.blockSignals(True)
        self.sentiment_slider.setValue(int(round(sentiment * 10)))
        self.sentiment_slider.blockSignals(False)
        self._update_sentiment_display(sentiment)

    def _update_segment_info(self, clip: dict):
        per_model = clip.get("per_model_scores") or {}
        segment = per_model.get("segment") or {}
        gem_meta = per_model.get("gemini_prefilter") or {}
        source_map = {
            "face_dialogue": "Mặt người + hội thoại",
            "face_only": "Theo mặt người",
            "scene_only": "Theo chuyển cảnh",
        }
        quality_map = {
            "good": "Tốt",
            "review": "Cần xem lại",
            "weak": "Yếu",
        }
        source = source_map.get(segment.get("source"), segment.get("source") or "-")
        # Ưu tiên gemini_prefilter > segment metadata
        face_cov = gem_meta.get("face_coverage") if gem_meta.get("face_coverage") is not None else segment.get("face_coverage")
        speech_cov = gem_meta.get("speech_coverage") if gem_meta.get("speech_coverage") is not None else segment.get("speech_coverage")
        quality = quality_map.get(segment.get("quality_hint"), segment.get("quality_hint") or "-")
        avg_faces = segment.get("num_faces_avg")
        has_speech = clip.get("has_speech", False)
        speech_label = "Có lời thoại" if has_speech else "Không có lời thoại"
        self.segment_source_label.setText(f"Nguồn cắt: {source}")
        self.segment_face_label.setText(
            "Tỷ lệ có mặt: -" if face_cov is None else f"Tỷ lệ có mặt: {float(face_cov) * 100:.0f}% | Số mặt TB: {float(avg_faces or 0):.1f}"
        )
        self.segment_speech_label.setText(
            f"Tỷ lệ hội thoại: {'-' if speech_cov is None else f'{float(speech_cov) * 100:.0f}%'} | {speech_label}"
        )
        self.segment_quality_label.setText(f"Đánh giá đoạn: {quality}")

    def _load_face_detections_for_clip(self, clip: dict) -> list[dict]:
        details = clip.get("per_model_scores") or {}
        face_details = details.get("face_extraction") or {}
        detections_path = face_details.get("detections_path")
        if not detections_path:
            return []
        try:
            path = Path(detections_path)
            if not path.exists():
                return []
            data = json.loads(path.read_text(encoding="utf-8"))
            return data if isinstance(data, list) else []
        except Exception:
            return []

    def _clear_display(self):
        self._current_index = -1
        self._current_clip = None
        self.clip_counter.setText("Clip 0 / 0")
        self.clip_title.setText("Màn Hình Xem Trước")
        self.clip_meta.setText("Chưa chọn clip")
        self.media_player.setSource(QUrl())
        self.video_widget.set_detections([])
        self.pred_emotion_label.setText("AI: -")
        self.confidence_label.setText("Độ tin cậy: -")
        self.agreement_label.setText("Đồng thuận: -")
        self.quality_label.setText("Chất lượng: -")
        self.segment_source_label.setText("Nguồn cắt: -")
        self.segment_face_label.setText("Tỷ lệ có mặt: -")
        self.segment_speech_label.setText("Tỷ lệ hội thoại: -")
        self.segment_quality_label.setText("Đánh giá đoạn: -")
        self.warning_label.setVisible(False)
        self.transcript_text.clear()
        self.notes_text.clear()
        for row in self.score_rows.values():
            row.set_score(0)
        for btn in self._emotion_buttons.values():
            btn.setChecked(False)
            btn.setStyleSheet("")
        self.sentiment_slider.blockSignals(True)
        self.sentiment_slider.setValue(0)
        self.sentiment_slider.blockSignals(False)
        self.sentiment_value_label.setText("0.0")
        self.timeline_bar.clear()

    def _toggle_play(self):
        if not self._current_clip:
            return
        if getattr(self, "_is_preview_mode", False):
            # Switch to normal play
            self._is_preview_mode = False
            self.audio_output.setMuted(False)
            self.media_player.setPlaybackRate(1.0)
            self.media_player.setPosition(0)
            self.media_player.play()
            self.play_btn.setIcon(qta.icon("fa5s.pause", color=Colors.TEXT_PRIMARY))
        else:
            if self.media_player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
                # Go back to preview loop
                self._is_preview_mode = True
                self.audio_output.setMuted(True)
                self.media_player.setPlaybackRate(2.5)
                self.media_player.play()
                self.play_btn.setIcon(qta.icon("fa5s.play", color=Colors.TEXT_PRIMARY))
            else:
                self._is_preview_mode = False
                self.audio_output.setMuted(False)
                self.media_player.setPlaybackRate(1.0)
                self.media_player.play()
                self.play_btn.setIcon(qta.icon("fa5s.pause", color=Colors.TEXT_PRIMARY))

    def _go_prev(self):
        if self._current_index > 0:
            self._select_clip(self._current_index - 1)

    def _go_next(self):
        if self._current_index < len(self._clips) - 1:
            self._select_clip(self._current_index + 1)

    @Slot(int)
    def _on_position_changed(self, position: int):
        if not self._is_scrubbing:
            self.position_slider.setValue(position)
        self._update_time_label(position, self.media_player.duration())
        self.video_widget.set_current_position_ms(position)

        # Update timeline playhead
        if self._current_clip and not self.timeline_bar.is_scrubbing():
            abs_time = int(self._current_clip.get("start_time", 0) * 1000) + position
            self.timeline_bar.set_playhead_position(abs_time)

        # GIF loop preview logic
        if getattr(self, "_is_preview_mode", False):
            dur = self.media_player.duration()
            if dur > 200 and position >= dur - 150:
                self.media_player.setPosition(0)

    def _on_timeline_seek(self, absolute_time_ms: int):
        target_sec = absolute_time_ms / 1000.0

        # Check if the current clip contains it first, to avoid reloading player if we just scrub within the same clip
        if self._current_clip:
            start = self._current_clip.get("start_time", 0)
            end = self._current_clip.get("end_time", 0)
            if start <= target_sec <= end:
                relative_ms = int((target_sec - start) * 1000)
                self.media_player.setPosition(relative_ms)
                # Keep playhead updated
                self.timeline_bar.set_playhead_position(absolute_time_ms)
                return

        # Otherwise find the closest clip
        best_idx = -1
        min_dist = float('inf')
        for idx, clip in enumerate(self._clips):
            start = clip.get("start_time", 0)
            end = clip.get("end_time", 0)
            if start <= target_sec <= end:
                best_idx = idx
                break
            dist = min(abs(target_sec - start), abs(target_sec - end))
            if dist < min_dist:
                min_dist = dist
                best_idx = idx

        if best_idx != -1:
            clip = self._clips[best_idx]
            start = clip.get("start_time", 0)
            end = clip.get("end_time", 0)
            self._select_clip(best_idx)
            # Bound seek position within the clip
            rel_sec = max(start, min(end, target_sec)) - start
            self.media_player.setPosition(int(rel_sec * 1000))
            self.timeline_bar.set_playhead_position(absolute_time_ms)

    @Slot(int)
    def _on_duration_changed(self, duration: int):
        self.position_slider.setRange(0, duration)
        self._update_time_label(self.media_player.position(), duration)

    def _seek_to_slider(self):
        self._is_scrubbing = False
        self.media_player.setPosition(self.position_slider.value())

    def _update_time_label(self, position_ms: int, duration_ms: int):
        self.time_label.setText(f"{self._format_ms(position_ms)} / {self._format_ms(duration_ms)}")

    @staticmethod
    def _format_ms(value: int) -> str:
        seconds = max(0, int(value / 1000))
        return f"{seconds // 60:02d}:{seconds % 60:02d}"

    # Review actions -------------------------------------------------

    def _on_sentiment_slider_changed(self, raw_value: int):
        """Called when user drags the sentiment slider."""
        score = raw_value / 10.0  # Convert back to [-3.0, +3.0]
        self._update_sentiment_display(score)
        if self._current_clip:
            self._current_clip["sentiment_score"] = score

    def _update_sentiment_display(self, score: float):
        """Update the sentiment value label text and color."""
        sign = "+" if score > 0 else ""
        self.sentiment_value_label.setText(f"{sign}{score:.1f}")
        # Color coding: green for positive, red for negative, gray for neutral
        if score > 0.5:
            color = "#00b894"  # green
        elif score < -0.5:
            color = "#e17055"  # red
        else:
            color = "#b8b6c4"  # gray/neutral
        self.sentiment_value_label.setStyleSheet(f"color: {color}; font-weight: bold; font-size: 14px;")

    def _on_emotion_selected(self, emotion_key: str):
        if not self._current_clip:
            return
        self._current_clip["user_emotion"] = emotion_key

        # Update styling of buttons
        for key, btn in self._emotion_buttons.items():
            is_checked = (key == emotion_key)
            btn.setChecked(is_checked)
            if is_checked:
                color = EMOTION_MAP.get(key, {}).get("color", "#6c5ce7")
                btn.setStyleSheet(f"background-color: {color}22; border: 1.5px solid {color}; color: {color}; font-weight: bold;")
            else:
                btn.setStyleSheet("")

        # Auto-map sentiment when emotion changes
        from backend.database.models import SENTIMENT_MAPPING
        auto_score = SENTIMENT_MAPPING.get(emotion_key, 0.0)
        self.sentiment_slider.blockSignals(True)
        self.sentiment_slider.setValue(int(round(auto_score * 10)))
        self.sentiment_slider.blockSignals(False)
        self._update_sentiment_display(auto_score)
        if self._current_clip:
            self._current_clip["sentiment_score"] = auto_score

        # Update timeline bar display
        self.timeline_bar.update()
        self._save_current_review(show_message=False)


    def _on_approve(self):
        if not self._current_clip:
            return
        if not self._current_clip.get("user_emotion"):
            self._current_clip["user_emotion"] = self._current_clip.get("ai_emotion")
        # Auto-populate sentiment_score if user hasn't set it explicitly
        if self._current_clip.get("sentiment_score") is None:
            from backend.database.models import SENTIMENT_MAPPING
            emotion = self._current_clip.get("user_emotion") or self._current_clip.get("ai_emotion") or "neutral"
            self._current_clip["sentiment_score"] = SENTIMENT_MAPPING.get(emotion, 0.0)
        self._update_status("approved")
        self._go_next()

    def _on_reject(self):
        if not self._current_clip:
            return
        self._update_status("rejected")
        self._go_next()

    def _on_gemini_verify(self):
        """Gọi Gemini Verify pass (Sprint 2) de re-score clip hien tai."""
        if not self._current_clip:
            return

        clip_path = self._current_clip.get("clip_path")
        if not clip_path or not os.path.exists(clip_path):
            QMessageBox.warning(
                self, "Không tìm thấy clip",
                f"Không tìm được file clip: {clip_path}"
            )
            return

        self.gemini_btn.setEnabled(False)
        self.gemini_btn.setText("Dang phan tich...")
        QApplication.processEvents()

        try:
            from backend.services.ai_video_segmenter import AIVideoSegmenter
            segmenter = AIVideoSegmenter()

            configured, msg = segmenter.is_configured()
            if not configured:
                QMessageBox.warning(
                    self, "Chua cau hinh Gemini",
                    f"Khong the goi Gemini:\n{msg}\n\n"
                    "Chay: gcloud auth application-default login"
                )
                return

            # Lay emotion/intensity hien tai cua clip
            current_emotion = (
                self._current_clip.get("emotion")
                or self._current_clip.get("ai_emotion")
                or self._current_clip.get("gem_emotion")
                or "neutral"
            )
            current_intensity = (
                self._current_clip.get("intensity")
                or self._current_clip.get("ai_intensity")
                or self._current_clip.get("gem_confidence")
                or 0.5
            )
            clip_id = self._current_clip.get("id", "unknown")

            result = segmenter.verify_clip(
                clip_path=clip_path,
                predicted_emotion=str(current_emotion),
                predicted_intensity=float(current_intensity),
                transcript="",  # Transcriber chua chay o day
                audio_features=None,
                face_stats=None,
            )

            verdict = result.get("verdict", "unknown")
            verified_emotion = result.get("emotion", current_emotion)
            verified_intensity = result.get("intensity", current_intensity)
            reasoning = result.get("reasoning", "")
            status = result.get("status", "")

            # Merge voi Stage 2 (current)
            merged = segmenter.combine_verdicts(
                {"emotion": current_emotion, "intensity": current_intensity},
                result,
            )
            final_emotion = merged.get("emotion", verified_emotion)
            final_intensity = merged.get("intensity", verified_intensity)
            verify_status = merged.get("verify_status", "passed")
            rejected = merged.get("rejected_by_verify", False)

            # Cap nhat UI
            self._current_clip["gem_emotion"] = verified_emotion
            self._current_clip["gem_confidence"] = verified_intensity
            self._current_clip["gem_reasoning"] = reasoning
            self._current_clip["verify_verdict"] = verdict
            self._current_clip["verify_status"] = verify_status
            self._current_clip["verify_reasoning"] = reasoning
            self._current_clip["rejected_by_verify"] = rejected

            # Cap nhat label hien thi tren review page
            self._update_verify_badge(verdict, verify_status)

            # Hien thi ket qua
            verdict_emoji = {
                "confirmed": "✅", "wrong_emotion": "⚠️",
                "unstable": "🔄", "low_quality": "❌",
                "stats_mismatch": "❓",
            }.get(verdict, "❓")
            verdict_text = verdict_emoji + " " + verdict.upper()

            msg_box = QMessageBox(self)
            msg_box.setWindowTitle("Gemini Verify Hoan Tat")
            msg_box.setText(
                f"<b>Verdict:</b> {verdict_text}<br><br>"
                f"<b>Emotion:</b> {final_emotion} "
                f"(Verify: {verified_emotion})<br>"
                f"<b>Intensity:</b> {final_intensity:.2f} "
                f"(Verify: {verified_intensity:.2f})<br>"
                f"<b>Status:</b> {'REJECTED' if rejected else 'PASSED'}<br>"
                f"<b>Reasoning:</b><br>{reasoning[:300]}"
            )
            msg_box.setTextFormat(Qt.RichText)
            msg_box.exec()

            # Neu emotion thay doi → hoi user co muon ap dung khong
            if final_emotion and final_emotion != current_emotion:
                reply = QMessageBox.question(
                    self, "Emotion thay doi",
                    f"Gemini de xuat nhan: **{final_emotion.upper()}** "
                    f"(confidence: {final_intensity:.0%})\n\n"
                    f"Co muon ap dung khong?",
                    QMessageBox.Yes | QMessageBox.No,
                )
                if reply == QMessageBox.Yes:
                    self._set_emotion(final_emotion)
                    self._save_current_review()

        except ImportError as exc:
            QMessageBox.critical(
                self, "Thieu thu vien",
                f"google.genai chua duoc cai hoac Vertex AI credentials chua cau hinh:\n{exc}\n\n"
                "Chay: pip install google-genai\n"
                "Hoac: gcloud auth application-default login"
            )
        except Exception as exc:
            QMessageBox.critical(self, "Loi Gemini", str(exc))
        finally:
            self.gemini_btn.setEnabled(True)
            self.gemini_btn.setText("Gemini Verify")

    def _save_current_review(self, show_message: bool = True):
        if not self._current_clip:
            return
        try:
            from backend.database.local_db import get_session
            from backend.database.models import Clip

            session = get_session()
            try:
                clip = session.query(Clip).filter(Clip.id == self._current_clip["id"]).first()
                if clip:
                    clip.user_emotion = self._current_clip.get("user_emotion")
                    clip.sentiment_score = self._current_clip.get("sentiment_score")
                    clip.reviewer_notes = self.notes_text.toPlainText().strip()
                    clip.reviewed_at = datetime.utcnow()
                    session.commit()
                    self._current_clip["reviewer_notes"] = clip.reviewer_notes
                else:
                    raise RuntimeError("Clip no longer exists in database")
            finally:
                session.close()
            if show_message:
                self.video_summary.setText("Review saved")
        except Exception as exc:
            QMessageBox.warning(self, "Save failed", str(exc))

    def _update_verify_badge(self, verdict: str, verify_status: str) -> None:
        """Cap nhat badge/tren thong tin verify tren review page.

        Cap nhat label hoac status tren UI de hien thi verify verdict.
        Neu co attribute tren Clip model → update luon vao DB.
        """
        try:
            from backend.database.local_db import get_session
            from backend.database.models import Clip
            if not self._current_clip:
                return
            clip_id = self._current_clip.get("id")
            if not clip_id:
                return
            session = get_session()
            try:
                clip = session.query(Clip).filter(Clip.id == clip_id).first()
                if clip:
                    clip.verify_verdict = verdict
                    clip.verify_status = verify_status
                    clip.rejected_by_verify = (verify_status == "rejected")
                    session.commit()
                    self._current_clip["verify_verdict"] = verdict
                    self._current_clip["verify_status"] = verify_status
            finally:
                session.close()
        except Exception:
            pass  # Non-critical: badge update fail khong anh huong workflow

    def _update_status(self, new_status: str):
        try:
            from backend.database.local_db import get_session
            from backend.database.models import Clip

            session = get_session()
            try:
                clip = session.query(Clip).filter(Clip.id == self._current_clip["id"]).first()
                if clip:
                    clip.status = new_status
                    clip.user_emotion = self._current_clip.get("user_emotion")
                    clip.sentiment_score = self._current_clip.get("sentiment_score")
                    clip.reviewer_notes = self.notes_text.toPlainText().strip()
                    clip.reviewed_at = datetime.utcnow()
                    session.commit()
                    self._current_clip["status"] = new_status
                    self._current_clip["reviewer_notes"] = clip.reviewer_notes
                else:
                    raise RuntimeError("Clip no longer exists in database")
            finally:
                session.close()
            self._populate_lists()
            self._select_clip(min(self._current_index, len(self._clips) - 1))
        except Exception as exc:
            QMessageBox.warning(self, "Update status failed", str(exc))
