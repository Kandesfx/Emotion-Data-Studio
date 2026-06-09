from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from backend.config import settings

# SQLite yêu cầu connect_args={"check_same_thread": False} khi chạy đa luồng.
# Cần thiết cho cả FastAPI và PySide6 QThread.
connect_args = {}
if settings.database_url.startswith("sqlite"):
    connect_args = {"check_same_thread": False}

# Khởi tạo Engine và Session
engine = create_engine(
    settings.database_url,
    connect_args=connect_args,
    echo=False  # Đặt thành True nếu cần debug câu lệnh SQL
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


def init_database():
    """
    Tạo tất cả các bảng trong database nếu chưa tồn tại.
    Sau đó chạy các schema migration nếu cần.
    Gọi khi khởi động ứng dụng (cả desktop và server).
    """
    from backend.database.models import Video, Clip, Label, SyncLog, Feature, ProcessQueue  # noqa: F401
    Base.metadata.create_all(bind=engine)
    _run_migrations(engine)


def _run_migrations(engine):
    """
    Hệ thống migration đơn giản cho SQLite.
    Mỗi migration là một hàm nhận connection, chạy ALTER TABLE/etc.
    Chỉ chạy migration chưa apply (theo schema_version).
    """
    from sqlalchemy import text, inspect

    with engine.begin() as conn:
        # Tạo bảng schema_version nếu chưa có
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS schema_version (
                version INTEGER PRIMARY KEY,
                applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                description TEXT
            )
        """))

        # Lấy version cao nhất đã apply
        result = conn.execute(text("SELECT COALESCE(MAX(version), 0) FROM schema_version"))
        current_version = result.scalar()

    # Danh sách migration: (version, description, sql_statements)
    # Thêm migration mới vào cuối danh sách khi schema thay đổi
    migrations = [
        (1, "Add sentiment_score column to clips for MMSA training", [
            "ALTER TABLE clips ADD COLUMN sentiment_score REAL"
        ]),
        (2, "Add Colab EDS video and clip metadata", [
            "ALTER TABLE videos ADD COLUMN source_type TEXT DEFAULT 'local'",
            "ALTER TABLE videos ADD COLUMN error_msg TEXT",
            "ALTER TABLE videos ADD COLUMN num_clips_raw INTEGER DEFAULT 0",
            "ALTER TABLE videos ADD COLUMN num_clips_ok INTEGER DEFAULT 0",
            "ALTER TABLE videos ADD COLUMN target_emotion TEXT",
            "ALTER TABLE clips ADD COLUMN face_ratio REAL",
            "ALTER TABLE clips ADD COLUMN frontal_ratio REAL",
            "ALTER TABLE clips ADD COLUMN avg_yaw REAL",
            "ALTER TABLE clips ADD COLUMN avg_face_size REAL",
            "ALTER TABLE clips ADD COLUMN face_quality REAL",
            "ALTER TABLE clips ADD COLUMN transcript_conf REAL",
            "ALTER TABLE clips ADD COLUMN audio_path TEXT",
            "ALTER TABLE clips ADD COLUMN snr_db REAL",
            "ALTER TABLE clips ADD COLUMN num_speakers INTEGER",
            "ALTER TABLE clips ADD COLUMN has_speech BOOLEAN DEFAULT 0",
            "ALTER TABLE clips ADD COLUMN emotion_face TEXT",
            "ALTER TABLE clips ADD COLUMN emotion_face_conf REAL",
            "ALTER TABLE clips ADD COLUMN emotion_voice TEXT",
            "ALTER TABLE clips ADD COLUMN emotion_voice_conf REAL",
            "ALTER TABLE clips ADD COLUMN emotion_text TEXT",
            "ALTER TABLE clips ADD COLUMN emotion_text_conf REAL",
            "ALTER TABLE clips ADD COLUMN decision_by TEXT",
            "ALTER TABLE clips ADD COLUMN reject_reason TEXT",
            "ALTER TABLE clips ADD COLUMN pipeline_stage TEXT"
        ]),
        (3, "Create Colab EDS queue, features and indexes", [
            "CREATE TABLE IF NOT EXISTS features (clip_id TEXT PRIMARY KEY REFERENCES clips(id), text_path TEXT, audio_path TEXT, vision_path TEXT, text_shape TEXT, audio_shape TEXT, vision_shape TEXT, aligned BOOLEAN DEFAULT 0, split TEXT, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)",
            "CREATE TABLE IF NOT EXISTS process_queue (id INTEGER PRIMARY KEY AUTOINCREMENT, video_id TEXT REFERENCES videos(id), priority INTEGER DEFAULT 0, target_emotion TEXT, status TEXT DEFAULT 'queued', started_at TIMESTAMP, completed_at TIMESTAMP, error_msg TEXT, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)",
            "CREATE INDEX IF NOT EXISTS idx_clips_status ON clips(status)",
            "CREATE INDEX IF NOT EXISTS idx_clips_emotion ON clips(predicted_emotion)",
            "CREATE INDEX IF NOT EXISTS idx_clips_video ON clips(video_id)",
            "CREATE INDEX IF NOT EXISTS idx_queue_status ON process_queue(status, priority DESC)"
        ]),
    ]

    with engine.begin() as conn:
        for version, description, statements in migrations:
            if version > current_version:
                for sql in statements:
                    try:
                        conn.execute(text(sql))
                    except Exception as e:
                        # Bỏ qua nếu cột/bảng đã tồn tại (idempotent)
                        if "duplicate column" not in str(e).lower() and "already exists" not in str(e).lower():
                            raise
                conn.execute(text(
                    "INSERT INTO schema_version (version, description) VALUES (:v, :d)"
                ), {"v": version, "d": description})



def get_session():
    """
    Trả về một database session mới.
    Dùng cho PySide6 UI — gọi trực tiếp, tự quản lý đóng session.

    Usage:
        session = get_session()
        try:
            # ... query ...
        finally:
            session.close()
    """
    return SessionLocal()


def get_db():
    """Dependency cung cấp db session cho FastAPI endpoints và tự động đóng khi hoàn thành."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def cleanup_error_data():
    """
    Dọn dẹp dữ liệu lỗi từ các lần chạy thử trước:
    - Xóa Video records có status = 'error'
    - Xóa Clip records mồ côi (video đã bị xóa)
    - Xóa file clip/audio/frame rác trên đĩa
    Trả về dict thống kê số lượng đã xóa.
    """
    import os
    import shutil
    from backend.database.models import Video, Clip
    from backend.config import settings

    session = SessionLocal()
    stats = {"videos_deleted": 0, "clips_deleted": 0, "files_deleted": 0}

    try:
        # 1. Lấy danh sách video lỗi
        error_videos = session.query(Video).filter(Video.status == "error").all()

        for video in error_videos:
            # Xóa clips liên quan
            clips = session.query(Clip).filter(Clip.video_id == video.id).all()
            for clip in clips:
                # Xóa file clip trên đĩa
                if clip.clip_path and os.path.exists(clip.clip_path):
                    try:
                        os.remove(clip.clip_path)
                        stats["files_deleted"] += 1
                    except OSError:
                        pass
                # Xóa thư mục frames
                frames_dir = settings.DATA_DIR / "frames" / clip.id
                if frames_dir.exists():
                    shutil.rmtree(str(frames_dir), ignore_errors=True)
                    stats["files_deleted"] += 1
                # Xóa audio file
                audio_file = settings.DATA_DIR / "audio" / f"{clip.id}.wav"
                if audio_file.exists():
                    try:
                        os.remove(str(audio_file))
                        stats["files_deleted"] += 1
                    except OSError:
                        pass

                session.delete(clip)
                stats["clips_deleted"] += 1

            # Xóa video file trên đĩa
            if video.file_path and os.path.exists(video.file_path):
                try:
                    os.remove(video.file_path)
                    stats["files_deleted"] += 1
                except OSError:
                    pass

            session.delete(video)
            stats["videos_deleted"] += 1

        session.commit()
        print(f"🧹 Cleanup hoàn tất: {stats}")

    except Exception as e:
        session.rollback()
        print(f"❌ Cleanup lỗi: {e}")
    finally:
        session.close()

    return stats

