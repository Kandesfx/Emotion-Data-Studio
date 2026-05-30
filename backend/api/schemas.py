from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime

# ==========================================
# VIDEO SCHEMAS
# ==========================================
class VideoCreate(BaseModel):
    title: str = Field(..., description="Tiêu đề video hoặc tập phim")
    movie_name: Optional[str] = Field(None, description="Tên bộ phim (ví dụ: Về Nhà Đi Con)")
    source_url: Optional[str] = Field(None, description="URL YouTube (nếu có)")
    file_path: Optional[str] = Field(None, description="Đường dẫn file local (nếu import local)")

class VideoResponse(BaseModel):
    id: str
    title: str
    movie_name: Optional[str] = None
    source_url: Optional[str] = None
    file_path: Optional[str] = None
    gcs_path: Optional[str] = None
    duration_sec: Optional[float] = None
    resolution: Optional[str] = None
    status: str
    total_clips: int
    approved_clips: int
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
    transcript: Optional[str] = None
    speaker_id: Optional[str] = None
    quality_score: float
    status: str
    predicted_emotion: Optional[str] = None
    confidence: float
    agreement: Optional[str] = None
    has_incongruity: bool
    all_scores: Dict[str, float] = {}
    per_model_scores: Dict[str, Any] = {}
    user_emotion: Optional[str] = None
    reviewer_notes: Optional[str] = None
    reviewed_at: Optional[datetime] = None
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True

class ClipUpdate(BaseModel):
    status: str = Field(..., description="Trạng thái kiểm duyệt: approved hoặc rejected")
    user_emotion: Optional[str] = Field(None, description="Nhãn cảm xúc do user chọn thủ công")
    reviewer_notes: Optional[str] = Field(None, description="Ghi chú thêm từ người kiểm duyệt")

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
