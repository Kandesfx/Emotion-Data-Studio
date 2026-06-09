from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime

# ==========================================
# VIDEO SCHEMAS
# ==========================================
class VideoCreate(BaseModel):
    title: str = Field(..., description="Tiêu đề video hoặc tập phim")
    movie_name: Optional[str] = Field(None, description="Tên bộ phim (ví dụ: Về Nhà Đi Con)")
    source_url: Optional[str] = Field(None, description="URL YouTube/Drive/TikTok/... (nếu có)")
    source_type: str = Field("local", description="Nguồn import: local, youtube, drive, upload, inbox, ...")
    file_path: Optional[str] = Field(None, description="Đường dẫn file local (nếu import local)")
    target_emotion: Optional[str] = Field(None, description="Emotion ưu tiên khi harvest")

class VideoResponse(BaseModel):
    id: str
    title: str
    movie_name: Optional[str] = None
    source_url: Optional[str] = None
    source_type: Optional[str] = None
    file_path: Optional[str] = None
    gcs_path: Optional[str] = None
    duration_sec: Optional[float] = None
    resolution: Optional[str] = None
    status: str
    error_msg: Optional[str] = None
    total_clips: int
    approved_clips: int
    num_clips_raw: int = 0
    num_clips_ok: int = 0
    target_emotion: Optional[str] = None
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True

# ==========================================
# CLIP SCHEMAS
# ==========================================
class ClipResponse(BaseModel):
    id: str
    video_id: str
    clip_index: int
    start_time: float
    end_time: float
    duration: float
    clip_path: Optional[str] = None
    gcs_path: Optional[str] = None
    num_frames: int
    num_faces: int
    face_ratio: Optional[float] = None
    frontal_ratio: Optional[float] = None
    avg_yaw: Optional[float] = None
    avg_face_size: Optional[float] = None
    face_quality: Optional[float] = None
    transcript: Optional[str] = None
    transcript_conf: Optional[float] = None
    speaker_id: Optional[str] = None
    audio_path: Optional[str] = None
    snr_db: Optional[float] = None
    num_speakers: Optional[int] = None
    has_speech: bool = False
    quality_score: float
    status: str
    predicted_emotion: Optional[str] = None
    emotion_face: Optional[str] = None
    emotion_face_conf: Optional[float] = None
    emotion_voice: Optional[str] = None
    emotion_voice_conf: Optional[float] = None
    emotion_text: Optional[str] = None
    emotion_text_conf: Optional[float] = None
    confidence: float
    agreement: Optional[str] = None
    has_incongruity: bool
    decision_by: Optional[str] = None
    reject_reason: Optional[str] = None
    pipeline_stage: Optional[str] = None
    all_scores: Dict[str, float] = {}
    per_model_scores: Dict[str, Any] = {}
    user_emotion: Optional[str] = None
    sentiment_score: Optional[float] = None
    reviewer_notes: Optional[str] = None
    reviewed_at: Optional[datetime] = None
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True

class ClipUpdate(BaseModel):
    status: str = Field(..., description="Trạng thái kiểm duyệt: approved, rejected hoặc needs_review")
    user_emotion: Optional[str] = Field(None, description="Nhãn cảm xúc do user chọn thủ công")
    sentiment_score: Optional[float] = Field(None, description="Điểm sentiment liên tục [-3.0, +3.0] cho MMSA training")
    reviewer_notes: Optional[str] = Field(None, description="Ghi chú thêm từ người kiểm duyệt")
    reject_reason: Optional[str] = Field(None, description="Lý do reject nếu có")


class HarvestRequest(BaseModel):
    urls: List[str] = Field(default_factory=list, description="Danh sách URL cần import, mỗi URL một video/playlist/channel")
    target_emotion: Optional[str] = Field(None, description="Emotion ưu tiên")
    priority: int = Field(0, description="Độ ưu tiên xử lý")
    auto_start: bool = Field(False, description="Tự chạy queue ngay sau khi import")


class DriveHarvestRequest(BaseModel):
    drive_url: str = Field(..., description="Google Drive shared file/folder URL")
    target_emotion: Optional[str] = None
    priority: int = 0


class QueueItemResponse(BaseModel):
    id: int
    video_id: str
    priority: int
    target_emotion: Optional[str] = None
    status: str
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_msg: Optional[str] = None
    created_at: datetime

    class Config:
        from_attributes = True

# ==========================================
# STATS & DASHBOARD SCHEMAS
# ==========================================
class EmotionDistribution(BaseModel):
    emotion: str
    count: int
    percentage: float

class DashboardStats(BaseModel):
    total_clips: int
    approved_clips: int
    pending_clips: int
    rejected_clips: int
    total_videos: int
    emotion_distribution: List[EmotionDistribution]
