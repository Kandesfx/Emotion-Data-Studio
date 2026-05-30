import os
from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    # App Information
    APP_NAME: str = "Emotion Data Studio"
    VERSION: str = "1.0.0"
    ENV: str = "development"  # development, production
    
    # Network Config
    HOST: str = "127.0.0.1"
    PORT: int = 8765
    
    # Path Config (Local)
    # Nếu chạy đóng gói Electron, EDS_DATA_DIR sẽ được Electron truyền vào thông qua biến môi trường.
    # Mặc định ở chế độ phát triển sẽ là thư mục 'data' nằm ở root của dự án.
    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    DATA_DIR: Path = Path(os.getenv("EDS_DATA_DIR", BASE_DIR / "data"))
    
    # Database config
    DB_NAME: str = "studio.db"
    
    # GCS & Google Cloud Configuration
    GOOGLE_APPLICATION_CREDENTIALS: Optional[str] = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", None)
    GCP_PROJECT_ID: Optional[str] = os.getenv("GCP_PROJECT_ID", None)
    GCS_BUCKET_NAME: Optional[str] = os.getenv("GCS_BUCKET_NAME", None)
    CLOUD_SQL_CONNECTION_NAME: Optional[str] = os.getenv("CLOUD_SQL_CONNECTION_NAME", None)
    CLOUD_SQL_USER: Optional[str] = os.getenv("CLOUD_SQL_USER", None)
    CLOUD_SQL_PASSWORD: Optional[str] = os.getenv("CLOUD_SQL_PASSWORD", None)
    CLOUD_SQL_DB: Optional[str] = os.getenv("CLOUD_SQL_DB", None)
    
    # R2 / S3 Configuration for Releases
    R2_ENDPOINT: Optional[str] = os.getenv("R2_ENDPOINT", None)
    AWS_ACCESS_KEY_ID: Optional[str] = os.getenv("AWS_ACCESS_KEY_ID", None)
    AWS_SECRET_ACCESS_KEY: Optional[str] = os.getenv("AWS_SECRET_ACCESS_KEY", None)

    # External Binaries Path
    # Nếu chạy đóng gói, EDS_FFMPEG_PATH sẽ được Electron truyền vào.
    # Mặc định sẽ tìm kiếm trong system PATH hoặc trong thư mục bin/
    FFMPEG_PATH: str = os.getenv("EDS_FFMPEG_PATH", "ffmpeg")
    FFPROBE_PATH: str = os.getenv("EDS_FFPROBE_PATH", "ffprobe")
    
    # AI Engine Lazy loading & hardware configuration
    USE_GPU: bool = True
    MODEL_CACHE_DIR: Path = DATA_DIR / "models_cache"

    # Pydantic Settings Config
    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    @property
    def database_url(self) -> str:
        """Đường dẫn kết nối SQLite cục bộ."""
        db_path = self.DATA_DIR / self.DB_NAME
        return f"sqlite:///{db_path.as_posix()}"

    def initialize_directories(self):
        """Khởi tạo tất cả các thư mục lưu trữ dữ liệu cục bộ nếu chưa có."""
        subdirs = [
            "videos",
            "clips",
            "frames",
            "audio",
            "transcripts",
            "exports",
            "models_cache",
            "logs"
        ]
        
        # Đảm bảo DATA_DIR tồn tại
        self.DATA_DIR.mkdir(parents=True, exist_ok=True)
        
        # Khởi tạo các thư mục con
        for folder in subdirs:
            folder_path = self.DATA_DIR / folder
            folder_path.mkdir(parents=True, exist_ok=True)

# Khởi tạo biến settings toàn cục
settings = Settings()
# Tự động tạo thư mục khi load settings
settings.initialize_directories()
