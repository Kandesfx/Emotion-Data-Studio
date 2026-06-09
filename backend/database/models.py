from sqlalchemy import Column, String, Float, Integer, Boolean, DateTime, ForeignKey, Text, JSON
from sqlalchemy.orm import relationship
from datetime import datetime
import uuid
from backend.database.local_db import Base

def generate_uuid():
    return str(uuid.uuid4())

# Auto-mapping: discrete emotion label → default sentiment score [-3.0, +3.0]
# Used as initial suggestion when reviewer has not manually set a score.
SENTIMENT_MAPPING: dict[str, float] = {
    "happy":    +2.0,
    "surprise": +1.0,
    "neutral":   0.0,
    "sad":      -1.0,
    "fear":     -1.5,
    "angry":    -2.0,
    "disgust":  -2.5,
}

class Video(Base):
    __tablename__ = "videos"

    id = Column(String, primary_key=True, default=generate_uuid)
    title = Column(String, nullable=False)
    movie_name = Column(String, nullable=True)
    source_url = Column(String, nullable=True)
    source_type = Column(String, default="local")                # youtube, drive, upload, inbox, tiktok, ...
    file_path = Column(String, nullable=True)                    # Local file path
    gcs_path = Column(String, nullable=True)                     # GCS cloud path
    duration_sec = Column(Float, nullable=True)
    resolution = Column(String, nullable=True)                   # e.g., "1280x720"
    status = Column(String, default="pending")                   # pending, queued, processing, completed, error
    error_msg = Column(Text, nullable=True)
    processing_mode = Column(String, default="auto")              # auto, semi_auto, manual
    total_clips = Column(Integer, default=0)
    approved_clips = Column(Integer, default=0)
    num_clips_raw = Column(Integer, default=0)
    num_clips_ok = Column(Integer, default=0)
    target_emotion = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    clips = relationship("Clip", back_populates="video", cascade="all, delete-orphan")

    def to_dict(self):
        return {
            "id": self.id,
            "title": self.title,
            "movie_name": self.movie_name,
            "source_url": self.source_url,
            "source_type": self.source_type,
            "file_path": self.file_path,
            "gcs_path": self.gcs_path,
            "duration_sec": self.duration_sec,
            "resolution": self.resolution,
            "status": self.status,
            "error_msg": self.error_msg,
            "processing_mode": self.processing_mode,
            "total_clips": self.total_clips,
            "approved_clips": self.approved_clips,
            "num_clips_raw": self.num_clips_raw,
            "num_clips_ok": self.num_clips_ok,
            "target_emotion": self.target_emotion,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None
        }

class Clip(Base):
    __tablename__ = "clips"

    id = Column(String, primary_key=True, default=generate_uuid)
    video_id = Column(String, ForeignKey("videos.id"), nullable=False)
    clip_index = Column(Integer, nullable=False)                 # Clip index in video sequence (0-based)
    start_time = Column(Float, nullable=False)                   # Start time in seconds
    end_time = Column(Float, nullable=False)                     # End time in seconds
    duration = Column(Float, nullable=False)                     # Duration in seconds
    clip_path = Column(String, nullable=True)                    # Local file path
    gcs_path = Column(String, nullable=True)                     # GCS cloud path
    is_manual_segment = Column(Boolean, default=False)            # True nếu do người dùng cắt thủ công
    
    # Face & Visual Metadata
    num_frames = Column(Integer, default=0)
    num_faces = Column(Integer, default=0)
    face_ratio = Column(Float, nullable=True)
    frontal_ratio = Column(Float, nullable=True)
    avg_yaw = Column(Float, nullable=True)
    avg_face_size = Column(Float, nullable=True)
    face_quality = Column(Float, nullable=True)
    
    # Audio & Text Metadata
    transcript = Column(Text, nullable=True)
    transcript_conf = Column(Float, nullable=True)
    speaker_id = Column(String, nullable=True)
    audio_path = Column(String, nullable=True)
    snr_db = Column(Float, nullable=True)
    num_speakers = Column(Integer, nullable=True)
    has_speech = Column(Boolean, default=False)
    
    # AI Sentiment & Scoring
    quality_score = Column(Float, default=0.0)
    status = Column(String, default="pending")                   # pending, needs_review, approved, rejected, auto_approved
    predicted_emotion = Column(String, nullable=True)            # Ensemble voting winner
    emotion_face = Column(String, nullable=True)
    emotion_face_conf = Column(Float, nullable=True)
    emotion_voice = Column(String, nullable=True)
    emotion_voice_conf = Column(Float, nullable=True)
    emotion_text = Column(String, nullable=True)
    emotion_text_conf = Column(Float, nullable=True)
    confidence = Column(Float, default=0.0)                      # Combined score probability
    agreement = Column(String, nullable=True)                    # e.g., "3/4"
    has_incongruity = Column(Boolean, default=False)             # Mâu thuẫn giữa mặt và lời thoại
    decision_by = Column(String, nullable=True)                  # auto, human
    reject_reason = Column(Text, nullable=True)
    pipeline_stage = Column(String, nullable=True)
    
    # Chi tiết kết quả của các mô hình (Lưu dưới dạng JSON tự động parse/serialize)
    all_scores = Column(JSON, nullable=True)                     # Combined scores e.g., {"happy": 0.8, ...}
    per_model_scores = Column(JSON, nullable=True)               # HSEmotion, DeepFace, PhoBERT, Wav2Vec2 individual outputs
    
    # Sentiment Score (continuous label for MMSA training)
    sentiment_score = Column(Float, nullable=True)               # Continuous sentiment [-3.0, +3.0]
    
    # User Intervention (Duyệt thủ công)
    user_emotion = Column(String, nullable=True)                 # Nhãn do user ghi đè
    reviewer_notes = Column(Text, nullable=True)                 # Note của user
    reviewed_at = Column(DateTime, nullable=True)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    video = relationship("Video", back_populates="clips")

    def to_dict(self):
        return {
            "id": self.id,
            "video_id": self.video_id,
            "clip_index": self.clip_index,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration": self.duration,
            "clip_path": self.clip_path,
            "gcs_path": self.gcs_path,
            "num_frames": self.num_frames,
            "num_faces": self.num_faces,
            "face_ratio": self.face_ratio,
            "frontal_ratio": self.frontal_ratio,
            "avg_yaw": self.avg_yaw,
            "avg_face_size": self.avg_face_size,
            "face_quality": self.face_quality,
            "transcript": self.transcript,
            "transcript_conf": self.transcript_conf,
            "speaker_id": self.speaker_id,
            "audio_path": self.audio_path,
            "snr_db": self.snr_db,
            "num_speakers": self.num_speakers,
            "has_speech": self.has_speech,
            "quality_score": self.quality_score,
            "status": self.status,
            "predicted_emotion": self.predicted_emotion,
            "emotion_face": self.emotion_face,
            "emotion_face_conf": self.emotion_face_conf,
            "emotion_voice": self.emotion_voice,
            "emotion_voice_conf": self.emotion_voice_conf,
            "emotion_text": self.emotion_text,
            "emotion_text_conf": self.emotion_text_conf,
            "confidence": self.confidence,
            "agreement": self.agreement,
            "has_incongruity": self.has_incongruity,
            "decision_by": self.decision_by,
            "reject_reason": self.reject_reason,
            "pipeline_stage": self.pipeline_stage,
            "all_scores": self.all_scores or {},
            "per_model_scores": self.per_model_scores or {},
            "sentiment_score": self.sentiment_score,
            "user_emotion": self.user_emotion,
            "reviewer_notes": self.reviewer_notes,
            "reviewed_at": self.reviewed_at.isoformat() if self.reviewed_at else None,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None
        }

    # === Aliases for UI compatibility ===
    @property
    def ai_emotion(self):
        return self.predicted_emotion

    @ai_emotion.setter
    def ai_emotion(self, value):
        self.predicted_emotion = value

    @property
    def ai_confidence(self):
        return self.confidence

    @ai_confidence.setter
    def ai_confidence(self, value):
        self.confidence = value

    @property
    def ai_agreement(self):
        return self.agreement

    @property
    def human_emotion(self):
        return self.user_emotion

    @human_emotion.setter
    def human_emotion(self, value):
        self.user_emotion = value

    @property
    def review_type(self) -> str | None:
        return self.decision_by

    @review_type.setter
    def review_type(self, value: str):
        self.decision_by = value

    @property
    def review_notes(self) -> str | None:
        return self.reviewer_notes

    @review_notes.setter
    def review_notes(self, value: str):
        self.reviewer_notes = value

    @property
    def user_sentiment(self) -> float | None:
        return self.sentiment_score

    @user_sentiment.setter
    def user_sentiment(self, value: float):
        self.sentiment_score = value



class Feature(Base):
    """MOSEI-compatible feature paths for an approved clip."""
    __tablename__ = "features"

    clip_id = Column(String, ForeignKey("clips.id"), primary_key=True)
    text_path = Column(String, nullable=True)
    audio_path = Column(String, nullable=True)
    vision_path = Column(String, nullable=True)
    text_shape = Column(String, nullable=True)
    audio_shape = Column(String, nullable=True)
    vision_shape = Column(String, nullable=True)
    aligned = Column(Boolean, default=False)
    split = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class ProcessQueue(Base):
    """Processing queue for Colab/headless execution."""
    __tablename__ = "process_queue"

    id = Column(Integer, primary_key=True, autoincrement=True)
    video_id = Column(String, ForeignKey("videos.id"), nullable=False)
    priority = Column(Integer, default=0)
    target_emotion = Column(String, nullable=True)
    status = Column(String, default="queued")
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    error_msg = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class Label(Base):
    """Nhãn cuối cùng cho training — mỗi clip có 1 label"""
    __tablename__ = "labels"

    clip_id = Column(String, ForeignKey("clips.id"), primary_key=True)
    emotion_label = Column(String, nullable=False)          # "happy", "sad", ...
    emotion_index = Column(Integer, nullable=False)         # 0-6
    label_source = Column(String, nullable=True)            # "ai_auto", "human_verified", "human_corrected"
    confidence = Column(Float, nullable=True)
    split = Column(String, nullable=True)                   # "train", "val", "test"
    exported = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)


class SyncLog(Base):
    """Log đồng bộ dữ liệu local ↔ cloud"""
    __tablename__ = "sync_log"

    id = Column(Integer, primary_key=True, autoincrement=True)
    direction = Column(String, nullable=True)               # "upload", "download"
    entity_type = Column(String, nullable=True)             # "video", "clip", "label"
    entity_id = Column(String, nullable=True)
    status = Column(String, nullable=True)                  # "success", "failed"
    error_message = Column(Text, nullable=True)
    synced_at = Column(DateTime, default=datetime.utcnow)
