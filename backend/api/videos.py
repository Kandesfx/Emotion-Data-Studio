from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Query
from sqlalchemy.orm import Session
from typing import List, Optional
import os
from pathlib import Path

from backend.database.local_db import get_db
from backend.database.models import Video, Clip
from backend.api.schemas import VideoCreate, VideoResponse

router = APIRouter(prefix="/videos", tags=["Videos"])

@router.post("/", response_model=VideoResponse)
def create_video(video_in: VideoCreate, db: Session = Depends(get_db)):
    """Tạo bản ghi video mới từ file local hoặc URL YouTube."""
    # Kiểm tra nếu import local file nhưng path không tồn tại
    if video_in.file_path and not os.path.exists(video_in.file_path):
        raise HTTPException(
            status_code=400,
            detail=f"Đường dẫn file video cục bộ không tồn tại: {video_in.file_path}"
        )
        
    db_video = Video(
        title=video_in.title,
        movie_name=video_in.movie_name,
        source_url=video_in.source_url,
        source_type=video_in.source_type,
        file_path=video_in.file_path,
        target_emotion=video_in.target_emotion,
        status="pending"
    )
    
    db.add(db_video)
    db.commit()
    db.refresh(db_video)
    return db_video

@router.get("/", response_model=List[VideoResponse])
def list_videos(
    skip: int = 0,
    limit: int = 50,
    movie_name: Optional[str] = Query(None, description="Lọc theo tên phim"),
    status: Optional[str] = Query(None, description="Lọc theo trạng thái"),
    db: Session = Depends(get_db)
):
    """Lấy danh sách các video đã import."""
    query = db.query(Video)
    if movie_name:
        query = query.filter(Video.movie_name.ilike(f"%{movie_name}%"))
    if status:
        query = query.filter(Video.status == status)
        
    videos = query.order_by(Video.created_at.desc()).offset(skip).limit(limit).all()
    return videos

@router.get("/{video_id}", response_model=VideoResponse)
def get_video(video_id: str, db: Session = Depends(get_db)):
    """Lấy thông tin chi tiết một video."""
    video = db.query(Video).filter(Video.id == video_id).first()
    if not video:
        raise HTTPException(status_code=404, detail="Không tìm thấy video")
    return video

@router.delete("/{video_id}")
def delete_video(video_id: str, db: Session = Depends(get_db)):
    """Xóa video cùng tất cả các clips liên quan (đồng thời dọn dẹp các tệp tin cục bộ)."""
    video = db.query(Video).filter(Video.id == video_id).first()
    if not video:
        raise HTTPException(status_code=404, detail="Không tìm thấy video")
        
    # Lấy danh sách clips để dọn dẹp file cục bộ
    clips = db.query(Clip).filter(Clip.video_id == video_id).all()
    for clip in clips:
        if clip.clip_path and os.path.exists(clip.clip_path):
            try:
                os.remove(clip.clip_path)
            except Exception as e:
                print(f"Lỗi khi xóa file clip cục bộ {clip.clip_path}: {e}")
                
    # Xóa file video gốc cục bộ (nếu được tải về tự động bởi yt-dlp)
    # LƯU Ý: Không xóa file nếu là import thủ công từ nguồn khác
    if video.file_path and "data/videos" in video.file_path and os.path.exists(video.file_path):
        try:
            os.remove(video.file_path)
        except Exception as e:
            print(f"Lỗi khi xóa file video cục bộ {video.file_path}: {e}")

    db.delete(video)
    db.commit()
    return {"message": "Đã xóa video và toàn bộ dữ liệu đi kèm thành công"}

# ==========================================
# PIPELINE STARTER ACTUAL (Giai đoạn 2)
# ==========================================
def run_pipeline_task(video_id: str, db_session_maker):
    """Tiến trình chạy ngầm thực thi AI Pipeline cho Giai đoạn 2."""
    from backend.services.pipeline_orchestrator import PipelineOrchestrator
    db = db_session_maker()
    try:
        orchestrator = PipelineOrchestrator()
        orchestrator.run_pipeline(video_id, db)
    except Exception as e:
        print(f"Lỗi chạy background task PipelineOrchestrator: {e}")
        try:
            video = db.query(Video).filter(Video.id == video_id).first()
            if video:
                video.status = "error"
                db.commit()
        except:
            pass
    finally:
        db.close()

@router.post("/{video_id}/process", response_model=VideoResponse)
def process_video(
    video_id: str, 
    background_tasks: BackgroundTasks, 
    db: Session = Depends(get_db)
):
    """Bắt đầu tiến trình phân tích AI Pipeline thực tế cho video."""
    video = db.query(Video).filter(Video.id == video_id).first()
    if not video:
        raise HTTPException(status_code=404, detail="Không tìm thấy video")
        
    if video.status == "processing":
        raise HTTPException(status_code=400, detail="Video đang trong tiến trình xử lý rồi")
        
    # Thay đổi trạng thái ban đầu sang processing
    video.status = "processing"
    db.commit()
    db.refresh(video)
    
    # Khởi chạy background task pipeline thực tế
    from backend.database.local_db import SessionLocal
    background_tasks.add_task(run_pipeline_task, video.id, SessionLocal)
    
    return video
