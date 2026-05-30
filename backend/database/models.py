from sqlalchemy import Column, String, Float, Integer, Boolean, DateTime, ForeignKey, Text, JSON
from sqlalchemy.orm import relationship
from datetime import datetime
import uuid
from backend.database.local_db import Base

def generate_uuid():
    return str(uuid.uuid4())

class Video(Base):
    __tablename__ = "videos"

    id = Column(String, primary_key=True, default=generate_uuid)
    title = Column(String, nullable=False)
    movie_name = Column(String, nullable=True)
    source_url = Column(String, nullable=True)
    file_path = Column(String, nullable=True)                    # Local file path
    gcs_path = Column(String, nullable=True)                     # GCS cloud path
    duration_sec = Column(Float, nullable=True)
    resolution = Column(String, nullable=True)                   # e.g., "1280x720"
    status = Column(String, default="pending")                   # pending, processing, completed, error
    total_clips = Column(Integer, default=0)
    approved_clips = Column(Integer, default=0)
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
            "file_path": self.file_path,
            "gcs_path": self.gcs_path,
            "duration_sec": self.duration_sec,
            "resolution": self.resolution,
            "status": self.status,
            "total_clips": self.total_clips,
            "approved_clips": self.approved_clips,
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
    
    # Face & Visual Metadata
    num_frames = Column(Integer, default=0)
    num_faces = Column(Integer, default=0)
    
    # Audio & Text Metadata
    transcript = Column(Text, nullable=True)
    speaker_id = Column(String, nullable=True)
    
    # AI Sentiment & Scoring
    quality_score = Column(Float, default=0.0)
    status = Column(String, default="pending")                   # pending, approved, rejected
    predicted_emotion = Column(String, nullable=True)            # Ensemble voting winner
    confidence = Column(Float, default=0.0)                      # Combined score probability
    agreement = Column(String, nullable=True)                    # e.g., "3/4"
    has_incongruity = Column(Boolean, default=False)             # Mâu thuẫn giữa mặt và lời thoại
    
    # Chi tiết kết quả của các mô hình (Lưu dưới dạng JSON tự động parse/serialize)
    all_scores = Column(JSON, nullable=True)                     # Combined scores e.g., {"happy": 0.8, ...}
    per_model_scores = Column(JSON, nullable=True)               # HSEmotion, DeepFace, PhoBERT, Wav2Vec2 individual outputs
    
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
            "transcript": self.transcript,
            "speaker_id": self.speaker_id,
            "quality_score": self.quality_score,
            "status": self.status,
            "predicted_emotion": self.predicted_emotion,
            "confidence": self.confidence,
            "agreement": self.agreement,
            "has_incongruity": self.has_incongruity,
            "all_scores": self.all_scores or {},
            "per_model_scores": self.per_model_scores or {},
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
    def face_quality(self):
        return None  # Placeholder


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
