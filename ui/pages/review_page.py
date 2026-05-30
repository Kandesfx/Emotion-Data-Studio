"""
Emotion Data Studio — Review & Labeling Studio Page
=====================================================
The core review interface with:
  - Video player (QMediaPlayer)
  - AI prediction panel (scores, model breakdown)
  - Transcript display
  - Emotion label buttons (keyboard shortcuts)
  - Filter bar (status, emotion, confidence)
  - Clip navigation (prev/next)
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFrame, QSplitter, QComboBox, QCheckBox, QScrollArea,
    QSizePolicy, QProgressBar, QPlainTextEdit, QMessageBox
)
from PySide6.QtCore import Qt, Signal, Slot, QUrl, QTimer
from PySide6.QtGui import QFont, QKeySequence, QShortcut
from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput
from PySide6.QtMultimediaWidgets import QVideoWidget

from ui.styles.theme import Colors, EMOTION_MAP


class AIScoreBar(QFrame):
    """Single model score display with progress bar"""

    def __init__(self, model_name: str, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 2, 0, 2)
        layout.setSpacing(8)

        self.name_label = QLabel(model_name)
        self.name_label.setObjectName("statLabel")
        self.name_label.setFixedWidth(80)
        layout.addWidget(self.name_label)

        self.emotion_label = QLabel("—")
        self.emotion_label.setFixedWidth(60)
        layout.addWidget(self.emotion_label)

        self.progress = QProgressBar()
        self.progress.setMinimum(0)
        self.progress.setMaximum(100)
        self.progress.setValue(0)
        layout.addWidget(self.progress, stretch=1)

        self.conf_label = QLabel("0%")
        self.conf_label.setFixedWidth(40)
        self.conf_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        layout.addWidget(self.conf_label)

    def set_score(self, emotion: str, confidence: float):
        """Update model score"""
        emoji = EMOTION_MAP.get(emotion, {}).get("emoji", "❓")
        self.emotion_label.setText(f"{emoji} {emotion}")
        pct = int(confidence * 100)
        self.progress.setValue(pct)
        self.conf_label.setText(f"{pct}%")


class EmotionScoreBar(QFrame):
    """Emotion score bar for the distribution chart"""

    def __init__(self, emotion_key: str, parent=None):
        super().__init__(parent)
        info = EMOTION_MAP.get(emotion_key, {})
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 1, 0, 1)
        layout.setSpacing(8)

        label_text = f"{info.get('emoji', '❓')} {info.get('label', emotion_key)}"
        self.name_label = QLabel(label_text)
        self.name_label.setFixedWidth(110)
        self.name_label.setObjectName("statLabel")
        layout.addWidget(self.name_label)

        self.progress = QProgressBar()
        self.progress.setMinimum(0)
        self.progress.setMaximum(100)
        self.progress.setValue(0)
        color = info.get("color", "#6c5ce7")
        self.progress.setStyleSheet(f"""
            QProgressBar::chunk {{
                background: {color};
                border-radius: 4px;
            }}
        """)
        layout.addWidget(self.progress, stretch=1)

        self.pct_label = QLabel("0%")
        self.pct_label.setFixedWidth(40)
        self.pct_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        layout.addWidget(self.pct_label)

    def set_score(self, score: float):
        pct = int(score * 100)
        self.progress.setValue(pct)
        self.pct_label.setText(f"{pct}%")


class ReviewPage(QWidget):
    """Review & Labeling Studio — main review interface"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._clips = []           # List of clip data dicts
        self._current_index = 0    # Current clip index
        self._current_clip = None  # Current clip data

        self._setup_ui()
        self._setup_shortcuts()

    def _setup_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # --- Top: Filter Bar ---
        self._build_filter_bar(main_layout)

        # --- Main content: Splitter ---
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left panel: Video + Transcript + Actions
        left_panel = self._build_left_panel()
        splitter.addWidget(left_panel)

        # Right panel: AI Predictions
        right_panel = self._build_right_panel()
        splitter.addWidget(right_panel)

        # Set splitter proportions (60% left, 40% right)
        splitter.setSizes([650, 450])

        main_layout.addWidget(splitter, stretch=1)

    def _build_filter_bar(self, parent_layout):
        """Filter bar at the top"""
        filter_card = QFrame()
        filter_card.setObjectName("card")
        filter_card.setFixedHeight(56)

        layout = QHBoxLayout(filter_card)
        layout.setContentsMargins(16, 0, 16, 0)
        layout.setSpacing(12)

        # Navigation info
        self.clip_counter = QLabel("Clip 0 / 0")
        self.clip_counter.setObjectName("accentText")
        layout.addWidget(self.clip_counter)

        layout.addSpacing(16)

        # Filters
        layout.addWidget(QLabel("Status:"))
        self.filter_status = QComboBox()
        self.filter_status.addItems(["All", "Needs Review", "Auto Approved", "Approved", "Rejected", "Pending"])
        self.filter_status.currentTextChanged.connect(self._on_filter_changed)
        layout.addWidget(self.filter_status)

        layout.addWidget(QLabel("Emotion:"))
        self.filter_emotion = QComboBox()
        self.filter_emotion.addItems(["All"] + [
            f"{v['emoji']} {v['label']}" for v in EMOTION_MAP.values()
        ])
        self.filter_emotion.currentTextChanged.connect(self._on_filter_changed)
        layout.addWidget(self.filter_emotion)

        layout.addWidget(QLabel("Source:"))
        self.filter_source = QComboBox()
        self.filter_source.addItems(["All"])
        self.filter_source.currentTextChanged.connect(self._on_filter_changed)
        layout.addWidget(self.filter_source)

        self.check_incongruity = QCheckBox("Only Incongruity")
        self.check_incongruity.stateChanged.connect(self._on_filter_changed)
        layout.addWidget(self.check_incongruity)

        layout.addStretch()

        # Navigation buttons
        self.prev_btn = QPushButton("⏮ Prev")
        self.prev_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.prev_btn.clicked.connect(self._go_prev)
        layout.addWidget(self.prev_btn)

        self.next_btn = QPushButton("Next ⏭")
        self.next_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.next_btn.clicked.connect(self._go_next)
        layout.addWidget(self.next_btn)

        parent_layout.addWidget(filter_card)

    def _build_left_panel(self) -> QWidget:
        """Left panel: Video player + Transcript + Action buttons"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(16, 12, 8, 12)
        layout.setSpacing(12)

        # --- Video Player ---
        video_card = QFrame()
        video_card.setObjectName("card")
        video_layout = QVBoxLayout(video_card)
        video_layout.setSpacing(8)

        self.video_widget = QVideoWidget()
        self.video_widget.setMinimumHeight(300)
        self.video_widget.setStyleSheet("background-color: #000000; border-radius: 8px;")
        video_layout.addWidget(self.video_widget)

        # Media player
        self.media_player = QMediaPlayer()
        self.audio_output = QAudioOutput()
        self.media_player.setAudioOutput(self.audio_output)
        self.media_player.setVideoOutput(self.video_widget)

        # Player controls
        controls = QHBoxLayout()
        self.play_btn = QPushButton("▶")
        self.play_btn.setObjectName("iconBtn")
        self.play_btn.setFixedSize(36, 36)
        self.play_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.play_btn.clicked.connect(self._toggle_play)
        controls.addWidget(self.play_btn)

        self.time_label = QLabel("0:00 / 0:00")
        self.time_label.setObjectName("mutedText")
        controls.addWidget(self.time_label)

        controls.addStretch()

        self.clip_info_label = QLabel("No clip loaded")
        self.clip_info_label.setObjectName("mutedText")
        controls.addWidget(self.clip_info_label)

        video_layout.addLayout(controls)

        layout.addWidget(video_card)

        # --- Transcript ---
        transcript_card = QFrame()
        transcript_card.setObjectName("card")
        transcript_layout = QVBoxLayout(transcript_card)
        transcript_layout.setSpacing(8)

        transcript_title = QLabel("📝 Transcript")
        transcript_title.setObjectName("sectionTitle")
        transcript_layout.addWidget(transcript_title)

        self.transcript_text = QPlainTextEdit()
        self.transcript_text.setReadOnly(True)
        self.transcript_text.setMaximumHeight(100)
        self.transcript_text.setPlaceholderText("Không có transcript...")
        transcript_layout.addWidget(self.transcript_text)

        layout.addWidget(transcript_card)

        # --- Action Buttons ---
        action_card = QFrame()
        action_card.setObjectName("cardElevated")
        action_layout = QVBoxLayout(action_card)
        action_layout.setSpacing(8)

        action_title = QLabel("🏷️ Gán nhãn cảm xúc")
        action_title.setObjectName("sectionTitle")
        action_layout.addWidget(action_title)

        # Emotion buttons grid
        emotion_grid = QHBoxLayout()
        emotion_grid.setSpacing(6)
        self._emotion_buttons: dict[str, QPushButton] = {}

        for key, info in EMOTION_MAP.items():
            btn = QPushButton(f"{info['emoji']} {info['label']}")
            btn.setObjectName("emotionBtn")
            btn.setCheckable(True)
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.setToolTip(f"Shortcut: {info['shortcut']}")
            btn.clicked.connect(lambda checked, k=key: self._on_emotion_selected(k))
            self._emotion_buttons[key] = btn
            emotion_grid.addWidget(btn)

        action_layout.addLayout(emotion_grid)

        # Approve / Reject buttons
        action_row = QHBoxLayout()
        action_row.setSpacing(8)

        self.approve_btn = QPushButton("✅ Approve (A)")
        self.approve_btn.setObjectName("successBtn")
        self.approve_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.approve_btn.clicked.connect(self._on_approve)
        action_row.addWidget(self.approve_btn)

        self.reject_btn = QPushButton("❌ Reject (R)")
        self.reject_btn.setObjectName("dangerBtn")
        self.reject_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.reject_btn.clicked.connect(self._on_reject)
        action_row.addWidget(self.reject_btn)

        action_row.addStretch()

        shortcut_hint = QLabel("Keys: 1-7 Emotion | A Approve | R Reject | ← → Navigate")
        shortcut_hint.setObjectName("mutedText")
        action_row.addWidget(shortcut_hint)

        action_layout.addLayout(action_row)

        layout.addWidget(action_card)

        return panel

    def _build_right_panel(self) -> QWidget:
        """Right panel: AI predictions and analysis"""
        panel = QWidget()
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)

        scroll_content = QWidget()
        layout = QVBoxLayout(scroll_content)
        layout.setContentsMargins(8, 12, 16, 12)
        layout.setSpacing(12)

        # --- AI Prediction Summary ---
        pred_card = QFrame()
        pred_card.setObjectName("cardElevated")
        pred_layout = QVBoxLayout(pred_card)
        pred_layout.setSpacing(8)

        pred_title = QLabel("🎭 AI Prediction")
        pred_title.setObjectName("sectionTitle")
        pred_layout.addWidget(pred_title)

        # Main prediction
        self.pred_emotion_label = QLabel("—")
        self.pred_emotion_label.setObjectName("statValue")
        self.pred_emotion_label.setStyleSheet(f"font-size: 22px;")
        pred_layout.addWidget(self.pred_emotion_label)

        # Score / Agreement / Quality
        info_grid = QHBoxLayout()

        self.pred_score_label = QLabel("Score: —")
        self.pred_score_label.setObjectName("statLabel")
        info_grid.addWidget(self.pred_score_label)

        self.pred_agreement_label = QLabel("Agreement: —")
        self.pred_agreement_label.setObjectName("statLabel")
        info_grid.addWidget(self.pred_agreement_label)

        self.pred_quality_label = QLabel("Quality: —")
        self.pred_quality_label.setObjectName("statLabel")
        info_grid.addWidget(self.pred_quality_label)

        pred_layout.addLayout(info_grid)

        # Incongruity warning
        self.incongruity_label = QLabel("")
        self.incongruity_label.setObjectName("warningText")
        self.incongruity_label.setVisible(False)
        pred_layout.addWidget(self.incongruity_label)

        layout.addWidget(pred_card)

        # --- Per-Model Breakdown ---
        model_card = QFrame()
        model_card.setObjectName("card")
        model_layout = QVBoxLayout(model_card)
        model_layout.setSpacing(8)

        model_title = QLabel("🔬 Per-Model Breakdown")
        model_title.setObjectName("sectionTitle")
        model_layout.addWidget(model_title)

        self.model_scores: dict[str, AIScoreBar] = {}
        for model_name in ["HSEmotion", "DeepFace", "PhoBERT", "Wav2Vec2"]:
            score_bar = AIScoreBar(model_name)
            self.model_scores[model_name.lower()] = score_bar
            model_layout.addWidget(score_bar)

        layout.addWidget(model_card)

        # --- Emotion Distribution ---
        dist_card = QFrame()
        dist_card.setObjectName("card")
        dist_layout = QVBoxLayout(dist_card)
        dist_layout.setSpacing(6)

        dist_title = QLabel("📊 Emotion Scores")
        dist_title.setObjectName("sectionTitle")
        dist_layout.addWidget(dist_title)

        self.emotion_scores: dict[str, EmotionScoreBar] = {}
        for emotion_key in EMOTION_MAP.keys():
            score_bar = EmotionScoreBar(emotion_key)
            self.emotion_scores[emotion_key] = score_bar
            dist_layout.addWidget(score_bar)

        layout.addWidget(dist_card)

        layout.addStretch()

        scroll.setWidget(scroll_content)

        outer_layout = QVBoxLayout(panel)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        outer_layout.addWidget(scroll)

        return panel

    def _setup_shortcuts(self):
        """Setup keyboard shortcuts for quick labeling"""
        # Emotion shortcuts (1-7)
        for key, info in EMOTION_MAP.items():
            shortcut = QShortcut(QKeySequence(info["shortcut"]), self)
            shortcut.activated.connect(lambda k=key: self._on_emotion_selected(k))

        # Navigation shortcuts
        QShortcut(QKeySequence(Qt.Key.Key_Left), self).activated.connect(self._go_prev)
        QShortcut(QKeySequence(Qt.Key.Key_Right), self).activated.connect(self._go_next)

        # Approve/Reject shortcuts
        QShortcut(QKeySequence("A"), self).activated.connect(self._on_approve)
        QShortcut(QKeySequence("R"), self).activated.connect(self._on_reject)

        # Space for play/pause
        QShortcut(QKeySequence(Qt.Key.Key_Space), self).activated.connect(self._toggle_play)

    # ================================================================
    # ACTIONS
    # ================================================================

    def _toggle_play(self):
        """Toggle video play/pause"""
        if self.media_player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self.media_player.pause()
            self.play_btn.setText("▶")
        else:
            self.media_player.play()
            self.play_btn.setText("⏸")

    def _go_prev(self):
        """Go to previous clip"""
        if self._current_index > 0:
            self._current_index -= 1
            self._load_current_clip()

    def _go_next(self):
        """Go to next clip"""
        if self._current_index < len(self._clips) - 1:
            self._current_index += 1
            self._load_current_clip()

    def _on_emotion_selected(self, emotion_key: str):
        """Handle emotion button click"""
        # Uncheck all, check selected
        for key, btn in self._emotion_buttons.items():
            btn.setChecked(key == emotion_key)

        # Update in database
        if self._current_clip:
            self._save_label(emotion_key)

    def _on_approve(self):
        """Approve current clip"""
        if self._current_clip:
            self._update_status("approved")
            self._go_next()

    def _on_reject(self):
        """Reject current clip"""
        if self._current_clip:
            self._update_status("rejected")
            self._go_next()

    def _on_filter_changed(self):
        """Reload clips based on filters"""
        self._load_clips()

    # ================================================================
    # DATA
    # ================================================================

    def refresh_data(self):
        """Refresh clip list from database"""
        self._load_clips()
        self._load_video_sources()

    def _load_video_sources(self):
        """Load video names for filter dropdown"""
        try:
            from backend.database.local_db import get_session
            from backend.database.models import Video

            session = get_session()
            try:
                videos = session.query(Video).all()
                self.filter_source.clear()
                self.filter_source.addItem("All")
                for v in videos:
                    name = v.title or v.movie_name or f"Video {v.id[:8]}"
                    self.filter_source.addItem(name)
            finally:
                session.close()
        except Exception:
            pass

    def _load_clips(self):
        """Load clips from database with current filters"""
        try:
            from backend.database.local_db import get_session
            from backend.database.models import Clip

            session = get_session()
            try:
                query = session.query(Clip)

                # Apply filters
                status_filter = self.filter_status.currentText()
                if status_filter != "All":
                    status_map = {
                        "Needs Review": "needs_review",
                        "Auto Approved": "auto_approved",
                        "Approved": "approved",
                        "Rejected": "rejected",
                        "Pending": "pending",
                    }
                    if status_filter in status_map:
                        query = query.filter(Clip.status == status_map[status_filter])

                # Order by quality score (needs review first)
                query = query.order_by(Clip.quality_score.desc())

                clips = query.all()
                self._clips = []
                for clip in clips:
                    self._clips.append({
                        "id": clip.id,
                        "video_id": clip.video_id,
                        "clip_path": clip.clip_path,
                        "start_time": clip.start_time,
                        "end_time": clip.end_time,
                        "duration": clip.duration,
                        "transcript": clip.transcript,
                        "ai_emotion": clip.ai_emotion,
                        "ai_confidence": clip.ai_confidence,
                        "ai_agreement": clip.ai_agreement,
                        "quality_score": clip.quality_score,
                        "has_incongruity": clip.has_incongruity,
                        "human_emotion": clip.human_emotion,
                        "status": clip.status,
                        "all_scores": clip.all_scores,
                        "per_model_scores": clip.per_model_scores,
                    })

                self._current_index = 0
                if self._clips:
                    self._load_current_clip()
                else:
                    self._clear_display()

                self.clip_counter.setText(f"Clip {self._current_index + 1} / {len(self._clips)}")

            finally:
                session.close()
        except Exception as e:
            self._clips = []
            self._clear_display()

    def _load_current_clip(self):
        """Load and display the current clip"""
        if not self._clips or self._current_index >= len(self._clips):
            return

        clip = self._clips[self._current_index]
        self._current_clip = clip

        # Update counter
        self.clip_counter.setText(f"Clip {self._current_index + 1} / {len(self._clips)}")

        # Load video
        if clip.get("clip_path"):
            self.media_player.setSource(QUrl.fromLocalFile(clip["clip_path"]))
            self.clip_info_label.setText(
                f"Duration: {clip.get('duration', 0):.1f}s | "
                f"Faces: {clip.get('num_faces', 'N/A')}"
            )

        # Transcript
        self.transcript_text.setPlainText(clip.get("transcript", "") or "Không có transcript")

        # AI Prediction
        ai_emotion = clip.get("ai_emotion", "—")
        ai_conf = clip.get("ai_confidence", 0) or 0
        emoji = EMOTION_MAP.get(ai_emotion, {}).get("emoji", "❓")
        label = EMOTION_MAP.get(ai_emotion, {}).get("label", ai_emotion)
        self.pred_emotion_label.setText(f"{emoji} {label}")

        self.pred_score_label.setText(f"Score: {int(ai_conf * 100)}%")
        self.pred_agreement_label.setText(f"Agreement: {clip.get('ai_agreement', 'N/A')}")
        quality = clip.get("quality_score", 0) or 0
        stars = "⭐" * int(quality * 5) + "☆" * (5 - int(quality * 5))
        self.pred_quality_label.setText(f"Quality: {stars} {quality:.2f}")

        # Incongruity
        if clip.get("has_incongruity"):
            self.incongruity_label.setText("⚠️ Incongruity detected: models disagree")
            self.incongruity_label.setVisible(True)
        else:
            self.incongruity_label.setVisible(False)

        # Emotion distribution
        all_scores = clip.get("all_scores") or {}
        for key, bar in self.emotion_scores.items():
            bar.set_score(all_scores.get(key, 0))

        # Per-model scores
        per_model = clip.get("per_model_scores") or {}
        model_key_map = {
            "hsemotion": "hsemotion",
            "deepface": "deepface",
            "phobert": "phobert",
            "wav2vec2": "wav2vec2",
        }
        for model_key, bar in self.model_scores.items():
            model_data = per_model.get(model_key, {})
            if model_data:
                top_emotion = max(model_data, key=model_data.get) if model_data else "—"
                top_conf = model_data.get(top_emotion, 0) if model_data else 0
                bar.set_score(top_emotion, top_conf)

        # Set emotion button state (if human label exists)
        human_emotion = clip.get("human_emotion")
        for key, btn in self._emotion_buttons.items():
            btn.setChecked(key == human_emotion)

    def _clear_display(self):
        """Clear all displays when no clips available"""
        self.clip_counter.setText("Clip 0 / 0")
        self.pred_emotion_label.setText("—")
        self.pred_score_label.setText("Score: —")
        self.pred_agreement_label.setText("Agreement: —")
        self.pred_quality_label.setText("Quality: —")
        self.incongruity_label.setVisible(False)
        self.transcript_text.clear()
        self.clip_info_label.setText("No clip loaded")

        for bar in self.emotion_scores.values():
            bar.set_score(0)
        for bar in self.model_scores.values():
            bar.set_score("—", 0)
        for btn in self._emotion_buttons.values():
            btn.setChecked(False)

    def _save_label(self, emotion_key: str):
        """Save human label to database"""
        if not self._current_clip:
            return
        try:
            from backend.database.local_db import get_session
            from backend.database.models import Clip

            session = get_session()
            try:
                clip = session.query(Clip).filter(Clip.id == self._current_clip["id"]).first()
                if clip:
                    clip.human_emotion = emotion_key
                    session.commit()
                    self._current_clip["human_emotion"] = emotion_key
            finally:
                session.close()
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to save label: {e}")

    def _update_status(self, new_status: str):
        """Update clip status in database"""
        if not self._current_clip:
            return
        try:
            from backend.database.local_db import get_session
            from backend.database.models import Clip
            from datetime import datetime

            session = get_session()
            try:
                clip = session.query(Clip).filter(Clip.id == self._current_clip["id"]).first()
                if clip:
                    clip.status = new_status
                    clip.reviewed_at = datetime.utcnow()
                    session.commit()
                    self._current_clip["status"] = new_status
            finally:
                session.close()
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to update status: {e}")
