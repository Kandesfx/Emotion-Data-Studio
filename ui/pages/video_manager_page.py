"""
Emotion Data Studio - Video Manager Page.

Project-oriented manager for source videos and generated clips.
Includes search/filter, active-video persistence, batch actions, context menu,
and an embedded preview for the selected video/clip.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from PySide6.QtCore import QSettings, Qt, QUrl, Signal
from PySide6.QtGui import QAction
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
from PySide6.QtMultimediaWidgets import QVideoWidget
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QFrame,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMenu,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ui.styles.theme import Colors, Spacing
from ui.widgets import PageHeader, Card, ActionButton, StatusPill

try:
    import qtawesome as qta
    _HAS_QTA = True
except ImportError:
    _HAS_QTA = False


class VideoManagerPage(QWidget):
    """Video/clip project manager."""

    active_video_changed = Signal(str)
    open_review_requested = Signal(str)
    open_segment_requested = Signal(str, str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._settings = QSettings("EmotionDataStudio", "EmotionDataStudio")
        self._videos: dict[str, dict] = {}
        self._clips: dict[str, dict] = {}
        self._active_video_id: str | None = self._settings.value("active_video_id", None)
        self._selected_entity: tuple[str, str] | None = None
        self._setup_ui()

    # UI -----------------------------------------------------------------

    def _setup_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        root.addWidget(scroll)

        content = QWidget()
        scroll.setWidget(content)
        self.main_layout = QVBoxLayout(content)
        self.main_layout.setContentsMargins(Spacing.XL, Spacing.LG, Spacing.XL, Spacing.LG)
        self.main_layout.setSpacing(Spacing.LG)

        # Hero header
        header = PageHeader(
            "Quản Lý Video",
            "Quản lý video nguồn, clip con, trạng thái duyệt hàng loạt và dự án đang làm.",
            icon="fa5s.film",
        )
        # right slot: active video pill
        self.active_pill = StatusPill("info", "Chưa chọn dự án")
        header.set_right_widget(self.active_pill)
        self.main_layout.addWidget(header)

        self._build_filters(self.main_layout)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self._build_tree_panel())
        splitter.addWidget(self._build_detail_panel())
        splitter.setSizes([560, 660])
        splitter.setChildrenCollapsible(False)
        self.main_layout.addWidget(splitter, stretch=1)

        self._build_actions(self.main_layout)

    def _build_filters(self, root_layout):
        card = Card("default")

        # Row 1: search
        search_row = QHBoxLayout()
        search_row.setSpacing(Spacing.SM)
        if _HAS_QTA:
            search_icon = QLabel()
            search_icon.setPixmap(qta.icon("fa5s.search", color=Colors.TEXT_SECONDARY).pixmap(14, 14))
            search_icon.setStyleSheet("background: transparent;")
            search_row.addWidget(search_icon)
        self.search_box = QLineEdit()
        self.search_box.setPlaceholderText("Tìm theo tên, đường dẫn, nhãn, transcript, ghi chú…")
        self.search_box.setMinimumHeight(36)
        self.search_box.textChanged.connect(self.refresh_data)
        search_row.addWidget(self.search_box, stretch=1)
        refresh_btn = ActionButton("Làm mới", "fa5s.sync-alt", variant="ghost")
        refresh_btn.clicked.connect(self.refresh_data)
        search_row.addWidget(refresh_btn)
        card.addLayout(search_row)

        # Row 2: filters grid
        filter_row = QHBoxLayout()
        filter_row.setSpacing(Spacing.SM)

        for label, widget_factory in [
            ("Video", lambda: self._make_combo(["All", "pending", "processing", "completed", "error", "cancelled"], "status")),
            ("Clip",  lambda: self._make_combo(["All", "pending", "needs_review", "ai_labeled", "approved", "auto_approved", "rejected", "failed"], "clip")),
            ("Nhãn",  lambda: self._make_combo(["All", "unlabeled", "happy", "sad", "angry", "fear", "surprise", "disgust", "neutral"], "label")),
        ]:
            col = QVBoxLayout()
            col.setSpacing(2)
            lbl = QLabel(label)
            lbl.setStyleSheet(f"color: {Colors.TEXT_SECONDARY}; background: transparent; font-size: 11px;")
            col.addWidget(lbl)
            combo = widget_factory()
            col.addWidget(combo)
            wrap = QWidget()
            wrap.setLayout(col)
            wrap.setStyleSheet("background: transparent;")
            filter_row.addWidget(wrap, stretch=1)

        # Show empty checkbox
        empty_col = QVBoxLayout()
        empty_col.setSpacing(2)
        spacer = QLabel("")
        spacer.setStyleSheet("background: transparent; font-size: 11px;")
        empty_col.addWidget(spacer)
        self.show_empty_check = QCheckBox("Hiển thị video rỗng")
        self.show_empty_check.setChecked(True)
        self.show_empty_check.stateChanged.connect(self.refresh_data)
        self.show_empty_check.setStyleSheet("background: transparent;")
        empty_col.addWidget(self.show_empty_check)
        empty_wrap = QWidget()
        empty_wrap.setLayout(empty_col)
        empty_wrap.setStyleSheet("background: transparent;")
        filter_row.addWidget(empty_wrap)
        card.addLayout(filter_row)

        root_layout.addWidget(card)

    def _make_combo(self, items: list, kind: str) -> QComboBox:
        combo = QComboBox()
        combo.addItems(items)
        combo.setMinimumHeight(36)
        combo.currentTextChanged.connect(self.refresh_data)
        if kind == "status":
            self.status_filter = combo
        elif kind == "clip":
            self.clip_status_filter = combo
        elif kind == "label":
            self.label_filter = combo
        return combo

    def _build_tree_panel(self) -> QWidget:
        panel = QFrame()
        panel.setObjectName("card")
        layout = QVBoxLayout(panel)
        layout.setSpacing(10)
        title = QLabel("Project Tree")
        title.setObjectName("sectionTitle")
        layout.addWidget(title)
        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["Source video / clips", "Status", "Label", "Duration"])
        self.tree.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.tree.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.tree.customContextMenuRequested.connect(self._show_context_menu)
        self.tree.itemSelectionChanged.connect(self._on_tree_selection_changed)
        self.tree.itemChanged.connect(self._on_item_checked)
        layout.addWidget(self.tree, stretch=1)
        return panel

    def _build_detail_panel(self) -> QWidget:
        panel = QFrame()
        panel.setObjectName("card")
        layout = QVBoxLayout(panel)
        layout.setSpacing(10)
        title = QLabel("Details & Preview")
        title.setObjectName("sectionTitle")
        layout.addWidget(title)

        self.preview_widget = QVideoWidget()
        self.preview_widget.setMinimumHeight(190)
        self.preview_widget.setStyleSheet("background-color: #000; border-radius: 8px;")
        layout.addWidget(self.preview_widget)
        self.preview_player = QMediaPlayer(self)
        self.preview_audio = QAudioOutput(self)
        self.preview_player.setAudioOutput(self.preview_audio)
        self.preview_player.setVideoOutput(self.preview_widget)

        preview_controls = QHBoxLayout()
        self.preview_play_btn = QPushButton("Play Preview")
        self.preview_play_btn.clicked.connect(self._toggle_preview)
        preview_controls.addWidget(self.preview_play_btn)
        self.preview_path_label = QLabel("No preview source")
        self.preview_path_label.setObjectName("mutedText")
        preview_controls.addWidget(self.preview_path_label, stretch=1)
        layout.addLayout(preview_controls)

        self.detail_table = QTableWidget()
        self.detail_table.setColumnCount(2)
        self.detail_table.setHorizontalHeaderLabels(["Field", "Value"])
        self.detail_table.horizontalHeader().setStretchLastSection(True)
        self.detail_table.verticalHeader().setVisible(False)
        self.detail_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        layout.addWidget(self.detail_table, stretch=1)

        self.summary_label = QLabel("Select a video or clip to see details.")
        self.summary_label.setObjectName("mutedText")
        self.summary_label.setWordWrap(True)
        layout.addWidget(self.summary_label)
        return panel

    def _build_actions(self, root_layout):
        card = Card("elevated")

        # Primary actions row
        primary_row = QHBoxLayout()
        primary_row.setSpacing(Spacing.SM)
        self.import_urls_btn = ActionButton("Nhập URLs", "fa5s.link", variant="primary")
        self.import_urls_btn.clicked.connect(self._open_import_dialog)
        primary_row.addWidget(self.import_urls_btn)
        self.open_review_btn = ActionButton("Mở dự án active trong Duyệt", "fa5s.eye", variant="secondary")
        self.open_review_btn.clicked.connect(self._open_active_review)
        primary_row.addWidget(self.open_review_btn)
        self.open_segment_btn = ActionButton("Mở dự án active trong Soạn Đoạn", "fa5s.cut", variant="secondary")
        self.open_segment_btn.clicked.connect(self._open_active_segment)
        primary_row.addStretch()
        card.addLayout(primary_row)

        # Secondary / batch actions row
        secondary_row = QHBoxLayout()
        secondary_row.setSpacing(Spacing.SM)
        self.batch_status_btn = ActionButton("Đổi trạng thái hàng loạt", "fa5s.tasks", variant="ghost")
        self.batch_status_btn.clicked.connect(self._batch_change_status)
        secondary_row.addWidget(self.batch_status_btn)
        self.batch_label_btn = ActionButton("Gán nhãn hàng loạt", "fa5s.tags", variant="ghost")
        self.batch_label_btn.clicked.connect(self._batch_set_label)
        secondary_row.addWidget(self.batch_label_btn)
        self.rename_btn = ActionButton("Đổi tên / Sửa", "fa5s.edit", variant="ghost")
        self.rename_btn.clicked.connect(self._edit_selected)
        secondary_row.addWidget(self.rename_btn)
        secondary_row.addStretch()
        self.delete_checked_btn = ActionButton("Xóa mục đã chọn", "fa5s.trash-alt", variant="danger")
        self.delete_checked_btn.clicked.connect(self._delete_checked)
        secondary_row.addWidget(self.delete_checked_btn)
        card.addLayout(secondary_row)

        root_layout.addWidget(card)

    # Data ---------------------------------------------------------------

    def refresh_data(self):
        try:
            from backend.database.local_db import get_session
            from backend.database.models import Clip, Video

            session = get_session()
            try:
                videos = session.query(Video).order_by(Video.updated_at.desc(), Video.created_at.desc()).all()
                clips = session.query(Clip).order_by(Clip.video_id.asc(), Clip.clip_index.asc()).all()
                clip_groups: dict[str, list] = {}
                for clip in clips:
                    clip_groups.setdefault(clip.video_id, []).append(clip)

                self._videos = {}
                self._clips = {}
                for video in videos:
                    group = clip_groups.get(video.id, [])
                    video_dict = self._video_to_dict(video, group)
                    clip_dicts = [self._clip_to_dict(clip) for clip in group]
                    clip_dicts = self._filter_clips(clip_dicts)
                    if not self._video_passes_filters(video_dict, clip_dicts):
                        continue
                    self._videos[video.id] = self._video_to_dict(video, clip_dicts)
                    for clip_dict in clip_dicts:
                        self._clips[clip_dict["id"]] = clip_dict
            finally:
                session.close()
        except Exception as exc:
            QMessageBox.warning(self, "Load failed", str(exc))
            self._videos = {}
            self._clips = {}
        if self._active_video_id and self._active_video_id not in self._videos:
            # Keep persisted ID but do not show it as active if filtered out/deleted.
            if not self._settings.value("active_video_id", None):
                self._active_video_id = None
        self._populate_tree()

    def _video_to_dict(self, video, clips: list) -> dict:
        approved = sum(1 for clip in clips if self._obj_get(clip, "status") in {"approved", "auto_approved"})
        pending = sum(1 for clip in clips if self._obj_get(clip, "status") in {"pending", "needs_review", "ai_labeled"})
        rejected = sum(1 for clip in clips if self._obj_get(clip, "status") == "rejected")
        return {
            "id": video.id,
            "title": video.title or video.movie_name or f"Video {video.id[:8]}",
            "movie_name": video.movie_name or "",
            "source_url": video.source_url or "",
            "file_path": video.file_path or "",
            "duration_sec": video.duration_sec or 0,
            "resolution": video.resolution or "",
            "status": video.status or "pending",
            "processing_mode": video.processing_mode or "auto",
            "total_clips": len(clips),
            "approved_clips": approved,
            "pending_clips": pending,
            "rejected_clips": rejected,
            "created_at": str(video.created_at or ""),
            "updated_at": str(video.updated_at or ""),
        }

    def _clip_to_dict(self, clip) -> dict:
        return {
            "id": clip.id,
            "video_id": clip.video_id,
            "clip_index": clip.clip_index,
            "clip_path": clip.clip_path or "",
            "start_time": clip.start_time,
            "end_time": clip.end_time,
            "duration": clip.duration,
            "status": clip.status or "pending",
            "predicted_emotion": clip.predicted_emotion or "",
            "user_emotion": clip.user_emotion or "",
            "confidence": clip.confidence or 0,
            "quality_score": clip.quality_score or 0,
            "transcript": clip.transcript or "",
            "reviewer_notes": clip.reviewer_notes or "",
            "is_manual_segment": bool(clip.is_manual_segment),
            "created_at": str(clip.created_at or ""),
            "updated_at": str(clip.updated_at or ""),
        }

    @staticmethod
    def _obj_get(obj, key: str):
        if isinstance(obj, dict):
            return obj.get(key)
        return getattr(obj, key, None)

    def _filter_clips(self, clips: list[dict]) -> list[dict]:
        status = self.clip_status_filter.currentText()
        label = self.label_filter.currentText()
        text = self.search_box.text().strip().lower()
        result = []
        for clip in clips:
            if status != "All" and clip["status"] != status:
                continue
            effective_label = clip["user_emotion"] or clip["predicted_emotion"] or "unlabeled"
            if label != "All" and effective_label != label:
                continue
            haystack = " ".join([
                clip.get("clip_path", ""),
                clip.get("transcript", ""),
                clip.get("reviewer_notes", ""),
                effective_label,
                clip.get("status", ""),
            ]).lower()
            if text and text not in haystack:
                continue
            result.append(clip)
        return result

    def _video_passes_filters(self, video: dict, visible_clips: list[dict]) -> bool:
        status = self.status_filter.currentText()
        if status != "All" and video["status"] != status:
            return False
        if not self.show_empty_check.isChecked() and not visible_clips:
            return False
        text = self.search_box.text().strip().lower()
        if text:
            video_haystack = " ".join([
                video.get("title", ""),
                video.get("movie_name", ""),
                video.get("source_url", ""),
                video.get("file_path", ""),
                video.get("status", ""),
            ]).lower()
            if text not in video_haystack and not visible_clips:
                return False
        return True

    def _populate_tree(self):
        self.tree.blockSignals(True)
        self.tree.clear()
        for video_id, video in self._videos.items():
            title = video["title"]
            suffix = "  [ACTIVE]" if video_id == self._active_video_id else ""
            item = QTreeWidgetItem([f"{title}{suffix}", video["status"], "", self._format_duration(video["duration_sec"] * 1000)])
            item.setData(0, Qt.ItemDataRole.UserRole, ("video", video_id))
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(0, Qt.CheckState.Unchecked)
            self.tree.addTopLevelItem(item)
            clips = [c for c in self._clips.values() if c["video_id"] == video_id]
            for clip in sorted(clips, key=lambda c: c["clip_index"]):
                label = clip["user_emotion"] or clip["predicted_emotion"] or "unlabeled"
                child = QTreeWidgetItem([
                    f"Clip {clip['clip_index']:03d}",
                    clip["status"],
                    label,
                    self._format_duration((clip["duration"] or 0) * 1000),
                ])
                child.setData(0, Qt.ItemDataRole.UserRole, ("clip", clip["id"]))
                child.setFlags(child.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                child.setCheckState(0, Qt.CheckState.Unchecked)
                item.addChild(child)
            item.setExpanded(video_id == self._active_video_id or bool(self.search_box.text().strip()))
        self.tree.blockSignals(False)
        active_title = self._videos.get(self._active_video_id, {}).get("title") or "none"
        if hasattr(self, 'active_pill') and self.active_pill is not None:
            try:
                short = (active_title[:40] + "…") if len(str(active_title)) > 40 else active_title
                self.active_pill.set_status("success", f"Active: {short}")
            except RuntimeError:
                pass
        self._update_summary()

    # Details / preview --------------------------------------------------

    def _on_tree_selection_changed(self):
        selected = self.tree.selectedItems()
        if not selected:
            self._selected_entity = None
            self._show_details({})
            self._set_preview(None)
            return
        entity = selected[0].data(0, Qt.ItemDataRole.UserRole)
        self._selected_entity = entity
        kind, entity_id = entity
        data = self._videos.get(entity_id) if kind == "video" else self._clips.get(entity_id)
        self._show_details(data or {})
        path = data.get("file_path") if kind == "video" and data else data.get("clip_path") if data else None
        self._set_preview(path)

    def _show_details(self, data: dict):
        self.detail_table.setRowCount(len(data))
        for row, (key, value) in enumerate(data.items()):
            self.detail_table.setItem(row, 0, QTableWidgetItem(str(key)))
            self.detail_table.setItem(row, 1, QTableWidgetItem(str(value)))

    def _set_preview(self, path: str | None):
        self.preview_player.stop()
        if path and Path(path).exists():
            self.preview_player.setSource(QUrl.fromLocalFile(path))
            self.preview_path_label.setText(Path(path).name)
            self.preview_play_btn.setEnabled(True)
        else:
            self.preview_player.setSource(QUrl())
            self.preview_path_label.setText("No preview source")
            self.preview_play_btn.setEnabled(False)
            self.preview_play_btn.setText("Play Preview")

    def _toggle_preview(self):
        if self.preview_player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self.preview_player.pause()
            self.preview_play_btn.setText("Play Preview")
        else:
            self.preview_player.play()
            self.preview_play_btn.setText("Pause Preview")

    def _update_summary(self):
        total_videos = len(self._videos)
        total_clips = len(self._clips)
        active_clips = len([c for c in self._clips.values() if c["video_id"] == self._active_video_id]) if self._active_video_id else 0
        self.summary_label.setText(f"Videos: {total_videos} | Visible clips: {total_clips} | Active video clips: {active_clips}")

    def _on_item_checked(self, item: QTreeWidgetItem, column: int):
        if item.childCount() > 0:
            state = item.checkState(0)
            self.tree.blockSignals(True)
            for i in range(item.childCount()):
                item.child(i).setCheckState(0, state)
            self.tree.blockSignals(False)

    def _checked_entities(self) -> list[tuple[str, str]]:
        checked: list[tuple[str, str]] = []
        for i in range(self.tree.topLevelItemCount()):
            video_item = self.tree.topLevelItem(i)
            if video_item.checkState(0) == Qt.CheckState.Checked:
                checked.append(video_item.data(0, Qt.ItemDataRole.UserRole))
            for j in range(video_item.childCount()):
                child = video_item.child(j)
                if child.checkState(0) == Qt.CheckState.Checked:
                    checked.append(child.data(0, Qt.ItemDataRole.UserRole))
        checked_videos = {entity_id for kind, entity_id in checked if kind == "video"}
        return [(kind, entity_id) for kind, entity_id in checked if not (kind == "clip" and self._clips.get(entity_id, {}).get("video_id") in checked_videos)]

    # Context menu -------------------------------------------------------

    def _show_context_menu(self, pos):
        item = self.tree.itemAt(pos)
        if item is None:
            return
        self.tree.setCurrentItem(item)
        entity = item.data(0, Qt.ItemDataRole.UserRole)
        if not entity:
            return
        kind, entity_id = entity
        menu = QMenu(self)
        open_location = QAction("Open file location", self)
        open_location.triggered.connect(lambda: self._open_file_location(kind, entity_id))
        menu.addAction(open_location)
        copy_path = QAction("Copy path", self)
        copy_path.triggered.connect(lambda: self._copy_path(kind, entity_id))
        menu.addAction(copy_path)
        menu.addSeparator()
        if kind == "video":
            set_active = QAction("Set active video", self)
            set_active.triggered.connect(self._set_selected_video_active)
            menu.addAction(set_active)
            open_seg = QAction("Open in Segment Editor", self)
            open_seg.triggered.connect(self._open_active_segment)
            menu.addAction(open_seg)
        else:
            approve = QAction("Mark approved", self)
            approve.triggered.connect(lambda: self._update_status("clip", entity_id, "approved"))
            menu.addAction(approve)
            reject = QAction("Mark rejected", self)
            reject.triggered.connect(lambda: self._update_status("clip", entity_id, "rejected"))
            menu.addAction(reject)
        edit = QAction("Rename / Edit", self)
        edit.triggered.connect(self._edit_selected)
        menu.addAction(edit)
        delete = QAction("Delete selected/checked", self)
        delete.triggered.connect(self._delete_checked_or_selected)
        menu.addAction(delete)
        menu.exec(self.tree.viewport().mapToGlobal(pos))

    # Actions ------------------------------------------------------------

    def _selected_video_id(self) -> str | None:
        if not self._selected_entity:
            return self._active_video_id
        kind, entity_id = self._selected_entity
        if kind == "video":
            return entity_id
        return self._clips.get(entity_id, {}).get("video_id")

    def _set_selected_video_active(self):
        video_id = self._selected_video_id()
        if not video_id:
            QMessageBox.information(self, "No video", "Select a source video or one of its clips first.")
            return
        self._active_video_id = video_id
        self._settings.setValue("active_video_id", video_id)
        self.active_video_changed.emit(video_id)
        self._populate_tree()

    def _open_active_review(self):
        if not self._active_video_id:
            self._set_selected_video_active()
        if self._active_video_id:
            self.open_review_requested.emit(self._active_video_id)

    def _open_import_dialog(self):
        dialog = ImportUrlsDialog(self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self.refresh_data()

    def _open_active_segment(self):
        video_id = self._selected_video_id() or self._active_video_id
        if not video_id or video_id not in self._videos:
            QMessageBox.information(self, "No video", "Select a source video first.")
            return
        path = self._videos[video_id].get("file_path")
        if not path or not Path(path).exists():
            QMessageBox.warning(self, "Missing file", "The selected source video file is missing.")
            return
        self._active_video_id = video_id
        self._settings.setValue("active_video_id", video_id)
        self.active_video_changed.emit(video_id)
        self.open_segment_requested.emit(video_id, path)

    def _edit_selected(self):
        if not self._selected_entity:
            QMessageBox.information(self, "No selection", "Select a video or clip first.")
            return
        kind, entity_id = self._selected_entity
        if kind == "video":
            current = self._videos[entity_id]["title"]
            value, ok = QInputDialog.getText(self, "Rename video", "Title:", text=current)
            if ok and value.strip():
                self._update_video_title(entity_id, value.strip())
        else:
            current = self._clips[entity_id].get("user_emotion") or ""
            value, ok = QInputDialog.getText(self, "Edit clip label", "User emotion label:", text=current)
            if ok:
                self._update_clip_label(entity_id, value.strip() or None)

    def _batch_change_status(self):
        entities = self._checked_entities()
        clips = [entity_id for kind, entity_id in entities if kind == "clip"]
        videos = [entity_id for kind, entity_id in entities if kind == "video"]
        if not entities:
            QMessageBox.information(self, "Nothing checked", "Tick video/clip checkboxes first.")
            return
        if clips and videos:
            QMessageBox.warning(self, "Mixed selection", "Batch status supports either clips or videos, not both at once.")
            return
        kind = "clip" if clips else "video"
        options = ["pending", "needs_review", "ai_labeled", "approved", "auto_approved", "rejected", "failed"] if kind == "clip" else ["pending", "processing", "completed", "cancelled", "error"]
        value, ok = QInputDialog.getItem(self, "Batch status", "Status:", options, 0, False)
        if ok:
            for entity_id in clips or videos:
                self._update_status(kind, entity_id, value, refresh=False)
            self.refresh_data()

    def _batch_set_label(self):
        clips = [entity_id for kind, entity_id in self._checked_entities() if kind == "clip"]
        if not clips:
            QMessageBox.information(self, "No clips checked", "Tick clip checkboxes first.")
            return
        options = ["", "happy", "sad", "angry", "fear", "surprise", "disgust", "neutral"]
        value, ok = QInputDialog.getItem(self, "Batch label", "User emotion:", options, 0, True)
        if ok:
            label = value.strip() or None
            for clip_id in clips:
                self._update_clip_label(clip_id, label, refresh=False)
            self.refresh_data()

    def _delete_checked_or_selected(self):
        entities = self._checked_entities()
        if not entities and self._selected_entity:
            entities = [self._selected_entity]
        self._delete_entities_with_prompt(entities)

    def _delete_checked(self):
        self._delete_entities_with_prompt(self._checked_entities())

    def _delete_entities_with_prompt(self, entities: list[tuple[str, str]]):
        if not entities:
            QMessageBox.information(self, "Nothing selected", "Tick checkboxes or select an item to delete.")
            return
        delete_files = QMessageBox.question(
            self,
            "Delete media files too?",
            "Delete database records. Also delete local video/clip files from disk?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No | QMessageBox.StandardButton.Cancel,
        )
        if delete_files == QMessageBox.StandardButton.Cancel:
            return
        if QMessageBox.question(self, "Confirm delete", f"Delete {len(entities)} item(s)? This cannot be undone.") != QMessageBox.StandardButton.Yes:
            return
        self._delete_entities(entities, delete_files == QMessageBox.StandardButton.Yes)

    def _open_file_location(self, kind: str, entity_id: str):
        path = self._entity_path(kind, entity_id)
        if not path or not Path(path).exists():
            QMessageBox.warning(self, "Missing file", "File path does not exist.")
            return
        try:
            target = Path(path)
            if os.name == "nt":
                subprocess.Popen(["explorer", "/select,", str(target)])
            elif sys.platform == "darwin":
                subprocess.Popen(["open", "-R", str(target)])
            else:
                subprocess.Popen(["xdg-open", str(target.parent)])
        except Exception as exc:
            QMessageBox.warning(self, "Open location failed", str(exc))

    def _copy_path(self, kind: str, entity_id: str):
        path = self._entity_path(kind, entity_id) or ""
        QApplication.clipboard().setText(path)
        self.summary_label.setText("Path copied to clipboard")

    def _entity_path(self, kind: str, entity_id: str) -> str | None:
        if kind == "video":
            return self._videos.get(entity_id, {}).get("file_path")
        return self._clips.get(entity_id, {}).get("clip_path")

    # Database mutations -------------------------------------------------

    def _update_video_title(self, video_id: str, title: str, refresh: bool = True):
        from backend.database.local_db import get_session
        from backend.database.models import Video
        session = get_session()
        try:
            video = session.query(Video).filter(Video.id == video_id).first()
            if video:
                video.title = title
                video.movie_name = title
                session.commit()
        finally:
            session.close()
        if refresh:
            self.refresh_data()

    def _update_clip_label(self, clip_id: str, label: str | None, refresh: bool = True):
        from backend.database.local_db import get_session
        from backend.database.models import Clip
        session = get_session()
        try:
            clip = session.query(Clip).filter(Clip.id == clip_id).first()
            if clip:
                clip.user_emotion = label
                session.commit()
        finally:
            session.close()
        if refresh:
            self.refresh_data()

    def _update_status(self, kind: str, entity_id: str, status: str, refresh: bool = True):
        from backend.database.local_db import get_session
        from backend.database.models import Clip, Video
        session = get_session()
        try:
            model = Video if kind == "video" else Clip
            obj = session.query(model).filter(model.id == entity_id).first()
            if obj:
                obj.status = status
                session.commit()
        finally:
            session.close()
        if refresh:
            self.refresh_data()

    def _delete_entities(self, entities: list[tuple[str, str]], delete_files: bool):
        from backend.database.local_db import get_session
        from backend.database.models import Clip, Label, Video
        session = get_session()
        paths: list[str] = []
        checked_videos = {entity_id for kind, entity_id in entities if kind == "video"}
        deduped = [(kind, entity_id) for kind, entity_id in entities if not (kind == "clip" and self._clips.get(entity_id, {}).get("video_id") in checked_videos)]
        try:
            for kind, entity_id in deduped:
                if kind == "clip":
                    clip = session.query(Clip).filter(Clip.id == entity_id).first()
                    if clip:
                        if clip.clip_path:
                            paths.append(clip.clip_path)
                        session.query(Label).filter(Label.clip_id == clip.id).delete()
                        session.delete(clip)
                elif kind == "video":
                    video = session.query(Video).filter(Video.id == entity_id).first()
                    if video:
                        if video.file_path:
                            paths.append(video.file_path)
                        for clip in list(video.clips):
                            if clip.clip_path:
                                paths.append(clip.clip_path)
                            session.query(Label).filter(Label.clip_id == clip.id).delete()
                        session.delete(video)
                        if self._active_video_id == entity_id:
                            self._active_video_id = None
                            self._settings.remove("active_video_id")
            session.commit()
        finally:
            session.close()
        if delete_files:
            for path in paths:
                try:
                    if path and Path(path).exists():
                        os.remove(path)
                except Exception:
                    pass
        if self._active_video_id:
            self.active_video_changed.emit(self._active_video_id)
        self.refresh_data()

    @staticmethod
    def _format_duration(ms: float) -> str:
        total = max(0, int(ms / 1000))
        return f"{total // 60:02d}:{total % 60:02d}"


# ─────────────────────────────────────────────────────────────────────────────
# Batch URL Import Dialog
# ─────────────────────────────────────────────────────────────────────────────

class ImportUrlsDialog(QDialog):
    """Dialog for batch-importing video URLs into the EDS pipeline."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Import URLs — Batch Harvest")
        self.setMinimumSize(640, 480)
        self._setup_ui()
        self._emotions = ["", "happy", "sad", "angry", "fear", "surprise", "disgust", "neutral"]

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        header = QLabel(
            "Paste video/playlist/channel URLs (one per line). "
            "Supported: YouTube video, playlist (/playlist?list=...), channel, "
            "TikTok, Facebook, Drive, and direct video files."
        )
        header.setWordWrap(True)
        header.setObjectName("mutedText")
        layout.addWidget(header)

        layout.addWidget(QLabel("URLs:"))
        self.url_input = QPlainTextEdit()
        self.url_input.setPlaceholderText(
            "https://www.youtube.com/watch?v=...\n"
            "https://www.youtube.com/playlist?list=...\n"
            "https://www.youtube.com/@channel\n"
            "https://www.tiktok.com/@user/video/...\n"
            "https://drive.google.com/file/d/...\n"
        )
        layout.addWidget(self.url_input, stretch=1)

        opts = QHBoxLayout()
        opts.addWidget(QLabel("Target emotion:"))
        self.emotion_combo = QComboBox()
        self.emotion_combo.addItems(self._emotions)
        opts.addWidget(self.emotion_combo)

        opts.addWidget(QLabel("Priority:"))
        self.priority_spin = QComboBox()
        self.priority_spin.addItems(["0 — Normal", "1 — High", "2 — Critical"])
        opts.addWidget(self.priority_spin)

        self.auto_start_check = QCheckBox("Auto-start pipeline after import")
        opts.addWidget(self.auto_start_check)
        opts.addStretch()
        layout.addLayout(opts)

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        self.status_label = QLabel("")
        self.status_label.setObjectName("mutedText")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        btns = QHBoxLayout()
        btns.addStretch()
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self.reject)
        btns.addWidget(self.cancel_btn)
        self.import_btn = QPushButton("Import URLs")
        self.import_btn.setObjectName("primaryBtn")
        self.import_btn.clicked.connect(self._do_import)
        btns.addWidget(self.import_btn)
        layout.addLayout(btns)

    def _detect_source_type(self, url: str) -> str:
        host = urlparse(url).netloc.lower()
        if "youtube" in host or "youtu.be" in host:
            return "youtube"
        if "drive.google" in host:
            return "drive"
        if "tiktok" in host:
            return "tiktok"
        if "facebook" in host or "fb.watch" in host:
            return "facebook"
        if "dailymotion" in host:
            return "dailymotion"
        return "url"

    def _derive_title(self, url: str) -> str:
        from urllib.parse import parse_qs, urlparse
        try:
            parsed = urlparse(url)
            if "youtube" in parsed.netloc.lower():
                qs = parse_qs(parsed.query)
                if "v" in qs:
                    return f"YouTube {qs['v'][0]}"
                if "list" in qs:
                    return f"Playlist {qs['list'][0]}"
            return url.split("/")[-1][:60] or url
        except Exception:
            return url[:60]

    def _do_import(self):
        raw = self.url_input.toPlainText()
        urls = [u.strip() for u in raw.splitlines() if u.strip()]
        if not urls:
            self.status_label.setText("No URLs provided.")
            return

        self.import_btn.setEnabled(False)
        self.cancel_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setMaximum(len(urls))
        self.progress_bar.setValue(0)
        self.status_label.setText("Importing...")

        emotion = self.emotion_combo.currentText() or None
        priority = int(self.priority_spin.currentText().split("—")[0].strip())
        auto_start = self.auto_start_check.isChecked()

        from urllib.parse import urlparse as _urlparse

        added = 0
        skipped = 0
        errors = []

        try:
            from backend.database.local_db import get_session
            from backend.database.models import ProcessQueue, Video

            session = get_session()
            try:
                for i, url in enumerate(urls):
                    try:
                        # Check duplicate by URL
                        existing = session.query(Video).filter(
                            Video.source_url == url
                        ).first()
                        if existing:
                            skipped += 1
                            self.progress_bar.setValue(i + 1)
                            continue

                        video = Video(
                            title=self._derive_title(url),
                            source_url=url,
                            source_type=self._detect_source_type(url),
                            status="queued",
                            target_emotion=emotion,
                        )
                        session.add(video)
                        session.flush()

                        queue_item = ProcessQueue(
                            video_id=video.id,
                            priority=priority,
                            target_emotion=emotion,
                            status="queued",
                        )
                        session.add(queue_item)
                        session.commit()
                        added += 1
                    except Exception as e:
                        session.rollback()
                        errors.append(f"  {url[:60]}: {e}")

                    self.progress_bar.setValue(i + 1)

                # Optionally auto-start first item
                if auto_start and added > 0:
                    from backend.services.pipeline_orchestrator import PipelineOrchestrator
                    queue_item = (
                        session.query(ProcessQueue)
                        .filter(ProcessQueue.status == "queued")
                        .order_by(ProcessQueue.priority.desc(), ProcessQueue.id.asc())
                        .first()
                    )
                    if queue_item:
                        queue_item.status = "running"
                        session.commit()
                        try:
                            PipelineOrchestrator().run_pipeline(queue_item.video_id, session)
                            queue_item.status = "done"
                            session.commit()
                        except Exception as e:
                            queue_item.status = "error"
                            queue_item.error_msg = str(e)[:500]
                            session.commit()
                            errors.append(f"  Pipeline error: {e}")

            finally:
                session.close()

            parts = []
            if added:
                parts.append(f"Added {added} video(s)")
            if skipped:
                parts.append(f"skipped {skipped} duplicate(s)")
            if errors:
                parts.append(f"{len(errors)} error(s)")
            self.status_label.setText("✅ " + ", ".join(parts))
            if errors:
                self.status_label.setToolTip("\n".join(errors[:10]))
            self.progress_bar.setValue(len(urls))
            self.import_btn.setText("Done — Close")
            self.import_btn.setEnabled(True)
        except Exception as exc:
            self.status_label.setText(f"❌ Error: {exc}")
            self.import_btn.setEnabled(True)
            self.cancel_btn.setEnabled(True)

