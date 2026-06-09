from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
from typing import List, Optional
from datetime import datetime
import os

from backend.database.local_db import get_db
from backend.database.models import Clip, Video
from backend.api.schemas import ClipResponse, ClipUpdate

router = APIRouter(prefix="/clips", tags=["Clips"])

@router.get("/", response_model=List[ClipResponse])
def list_clips(
    video_id: Optional[str] = Query(None, description="Lọc theo ID video gốc"),
    status: Optional[str] = Query(None, description="Lọc theo trạng thái kiểm duyệt (pending, approved, rejected)"),
    emotion: Optional[str] = Query(None, description="Lọc theo cảm xúc dự đoán"),
    has_incongruity: Optional[bool] = Query(None, description="Chỉ hiện clips có sự mâu thuẫn (sarcasm)"),
    skip: int = 0,
    limit: int = 50,
    db: Session = Depends(get_db)
):
    """Lấy danh sách các clips phân cảnh kèm bộ lọc nâng cao."""
    query = db.query(Clip)
    
    if video_id:
        query = query.filter(Clip.video_id == video_id)
    if status:
        query = query.filter(Clip.status == status)
    if emotion:
        query = query.filter(Clip.predicted_emotion == emotion)
    if has_incongruity is not None:
        query = query.filter(Clip.has_incongruity == has_incongruity)
        
    clips = query.order_by(Clip.clip_index.asc()).offset(skip).limit(limit).all()
    return clips

@router.get("/{clip_id}", response_model=ClipResponse)
def get_clip(clip_id: str, db: Session = Depends(get_db)):
    """Lấy thông tin chi tiết một clip."""
    clip = db.query(Clip).filter(Clip.id == clip_id).first()
    if not clip:
        raise HTTPException(status_code=404, detail="Không tìm thấy clip phân cảnh")
    return clip

@router.put("/{clip_id}", response_model=ClipResponse)
def update_clip(
    clip_id: str, 
    clip_in: ClipUpdate, 
    db: Session = Depends(get_db)
):
    """Kiểm duyệt thủ công: Cập nhật nhãn cảm xúc và trạng thái duyệt của clip."""
    clip = db.query(Clip).filter(Clip.id == clip_id).first()
    if not clip:
        raise HTTPException(status_code=404, detail="Không tìm thấy clip phân cảnh")
        
    # Cập nhật thông tin
    clip.status = clip_in.status
    clip.user_emotion = clip_in.user_emotion
    clip.sentiment_score = clip_in.sentiment_score
    clip.reviewer_notes = clip_in.reviewer_notes
    clip.reject_reason = clip_in.reject_reason
    clip.decision_by = "human"
    clip.reviewed_at = datetime.utcnow()
    clip.updated_at = datetime.utcnow()
    
    db.commit()
    db.refresh(clip)
    
    # Cập nhật lại số lượng approved_clips của Video gốc tương ứng
    video = db.query(Video).filter(Video.id == clip.video_id).first()
    if video:
        approved_count = db.query(Clip).filter(
            Clip.video_id == video.id,
            Clip.status == "approved"
        ).count()
        video.approved_clips = approved_count
        db.commit()
        
    return clip

@router.get("/{clip_id}/video")
def stream_clip_video(clip_id: str, db: Session = Depends(get_db)):
    """Stream video clip cho trang review trên dashboard."""
    clip = db.query(Clip).filter(Clip.id == clip_id).first()
    if not clip or not clip.clip_path or not os.path.exists(clip.clip_path):
        raise HTTPException(status_code=404, detail="Không tìm thấy file video clip")
    return FileResponse(clip.clip_path, media_type="video/mp4", filename=os.path.basename(clip.clip_path))


@router.get("/{clip_id}/audio")
def stream_clip_audio(clip_id: str, db: Session = Depends(get_db)):
    """Stream audio clip cho kiểm tra chất lượng âm thanh."""
    clip = db.query(Clip).filter(Clip.id == clip_id).first()
    audio_path = clip.audio_path if clip else None
    if not audio_path and clip and clip.per_model_scores:
        audio_path = (clip.per_model_scores.get("audio_features") or {}).get("audio_path")
    if not audio_path or not os.path.exists(audio_path):
        raise HTTPException(status_code=404, detail="Không tìm thấy file audio clip")
    return FileResponse(audio_path, media_type="audio/wav", filename=os.path.basename(audio_path))


@router.delete("/{clip_id}")
def delete_clip(clip_id: str, db: Session = Depends(get_db)):
    """Xóa một clip khỏi cơ sở dữ liệu và dọn dẹp file mp4 cục bộ."""
    clip = db.query(Clip).filter(Clip.id == clip_id).first()
    if not clip:
        raise HTTPException(status_code=404, detail="Không tìm thấy clip phân cảnh")
        
    # Xóa file cục bộ
    if clip.clip_path and os.path.exists(clip.clip_path):
        try:
            os.remove(clip.clip_path)
        except Exception as e:
            print(f"Lỗi khi xóa file clip cục bộ {clip.clip_path}: {e}")
            
    video_id = clip.video_id
    db.delete(clip)
    db.commit()
    
    # Đồng bộ số lượng approved_clips và total_clips trên Video gốc
    video = db.query(Video).filter(Video.id == video_id).first()
    if video:
        video.total_clips = db.query(Clip).filter(Clip.video_id == video_id).count()
        video.approved_clips = db.query(Clip).filter(
            Clip.video_id == video_id,
            Clip.status == "approved"
        ).count()
        db.commit()
        
    return {"message": "Đã xóa clip thành công"}
