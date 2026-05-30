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
    Gọi khi khởi động ứng dụng (cả desktop và server).
    """
    from backend.database.models import Video, Clip, Label, SyncLog  # noqa: F401
    Base.metadata.create_all(bind=engine)


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

