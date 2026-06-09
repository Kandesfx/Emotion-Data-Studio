import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.config import settings
from backend.database.local_db import init_database
from backend.api import videos, clips, labels, harvest, stats, gemini_api, colab_worker

# Tự động tạo cấu trúc bảng SQLite cục bộ khi khởi chạy app
init_database()

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.VERSION,
    description="Backend API cho Phần mềm Khai thác & Quản lý Dataset Cảm xúc Đa phương thức"
)

# Cấu hình CORS để Electron app hoặc trình duyệt gọi an toàn
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Trong môi trường local cho phép tất cả các nguồn gọi
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Đăng ký các API Routers
app.include_router(videos.router, prefix="/api")
app.include_router(clips.router, prefix="/api")
app.include_router(labels.router, prefix="/api")
app.include_router(harvest.router, prefix="/api")
app.include_router(stats.router, prefix="/api")
app.include_router(gemini_api.router)
app.include_router(colab_worker.router)

@app.get("/")
def read_root():
    return {
        "status": "online",
        "app_name": settings.APP_NAME,
        "version": settings.VERSION,
        "environment": settings.ENV,
        "data_directory": str(settings.DATA_DIR)
    }

@app.get("/health")
def health_check():
    """Endpoint kiểm tra sức khoẻ hệ thống (health check)"""
    import sqlite3
    db_status = "healthy"
    try:
        # Kiểm tra kết nối database SQLite cục bộ
        conn = sqlite3.connect(settings.DATA_DIR / settings.DB_NAME)
        conn.close()
    except Exception as e:
        db_status = f"unhealthy: {str(e)}"
        
    return {
        "status": "healthy",
        "database": db_status
    }

if __name__ == "__main__":
    uvicorn.run("main:app", host=settings.HOST, port=settings.PORT, reload=True)
