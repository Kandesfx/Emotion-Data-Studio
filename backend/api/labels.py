from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlalchemy import func
from typing import List, Dict

from backend.database.local_db import get_db
from backend.database.models import Clip, Video
from backend.api.schemas import DashboardStats, EmotionDistribution

router = APIRouter(prefix="/labels", tags=["Labels & Statistics"])

@router.get("/stats", response_model=DashboardStats)
def get_dashboard_statistics(db: Session = Depends(get_db)):
    """Trả về dữ liệu thống kê tổng quan cho trang Dashboard."""
    # 1. Thống kê số lượng clips theo trạng thái
    total_clips = db.query(Clip).count()
    approved_clips = db.query(Clip).filter(Clip.status == "approved").count()
    pending_clips = db.query(Clip).filter(Clip.status == "pending").count()
    rejected_clips = db.query(Clip).filter(Clip.status == "rejected").count()
    
    # 2. Thống kê số lượng video
    total_videos = db.query(Video).count()
    
    # 3. Thống kê phân bố cảm xúc
    # Sử dụng nhãn do user duyệt (user_emotion) làm ưu tiên, nếu chưa có thì lấy predicted_emotion
    # Chỉ thống kê cho các clips có status là "approved" hoặc "pending" (bỏ qua rejected)
    all_clips = db.query(Clip).filter(Clip.status != "rejected").all()
    
    emotion_counts = {}
    for clip in all_clips:
        emotion = clip.user_emotion if clip.user_emotion else clip.predicted_emotion
        if not emotion:
            continue
        emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
        
    total_emotion_labeled = sum(emotion_counts.values())
    
    distribution = []
    if total_emotion_labeled > 0:
        for emotion, count in emotion_counts.items():
            percentage = round((count / total_emotion_labeled) * 100, 2)
            distribution.append(
                EmotionDistribution(
                    emotion=emotion,
                    count=count,
                    percentage=percentage
                )
            )
            
    # Sắp xếp phân bố theo số lượng giảm dần
    distribution.sort(key=lambda x: x.count, reverse=True)
    
    return DashboardStats(
        total_clips=total_clips,
        approved_clips=approved_clips,
        pending_clips=pending_clips,
        rejected_clips=rejected_clips,
        total_videos=total_videos,
        emotion_distribution=distribution
    )
