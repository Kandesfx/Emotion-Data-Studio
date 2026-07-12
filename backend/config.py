import json
import os
import sys
from pathlib import Path
from typing import Optional
try:
    from pydantic_settings import BaseSettings, SettingsConfigDict
except Exception:  # pragma: no cover - fallback for environments without pydantic-settings
    from pydantic import BaseSettings  # type: ignore

    def SettingsConfigDict(**kwargs):  # type: ignore
        return kwargs

# Tự động đồng bộ PATH trên Windows
if os.name == 'nt':
    try:
        import winreg
        merged_path = os.environ.get("PATH", "")
        
        # Đọc User PATH từ Registry
        try:
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as key:
                user_path, _ = winreg.QueryValueEx(key, "Path")
                merged_path += ";" + user_path
        except Exception:
            pass
            
        # Đọc System PATH từ Registry
        try:
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"System\CurrentControlSet\Control\Session Manager\Environment") as key:
                system_path, _ = winreg.QueryValueEx(key, "Path")
                merged_path += ";" + system_path
        except Exception:
            pass
            
        # Loại bỏ đường dẫn trùng lặp
        resolved_paths = []
        for p in merged_path.split(";"):
            p_expanded = os.path.expandvars(p).strip()
            if p_expanded and p_expanded not in resolved_paths:
                resolved_paths.append(p_expanded)
                
        os.environ["PATH"] = ";".join(resolved_paths)
    except Exception as e:
        print(f"Lỗi đồng bộ registry PATH: {e}")

BASE_DIR = Path(__file__).resolve().parent.parent

# Ưu tiên thư mục bin của ứng dụng để tìm ffmpeg/aria2c/deno trong bản build.
try:
    app_bin_dir = BASE_DIR / "bin"
    if app_bin_dir.exists():
        current_path = os.environ.get("PATH", "")
        bin_str = str(app_bin_dir)
        path_parts = [p for p in current_path.split(os.pathsep) if p]
        if bin_str not in path_parts:
            os.environ["PATH"] = bin_str + os.pathsep + current_path if current_path else bin_str
except Exception as e:
    print(f"Không thể ưu tiên thư mục bin của ứng dụng: {e}")

def _default_user_settings_path() -> Path:
    if getattr(sys, 'frozen', False):
        return Path(os.environ.get("LOCALAPPDATA", ".")) / "EmotionDataStudio" / "user_settings.json"
    return BASE_DIR / "data" / "user_settings.json"

USER_SETTINGS_PATH = Path(os.getenv("EDS_USER_SETTINGS", _default_user_settings_path()))

def _load_user_settings_into_env() -> None:
    """Load user-editable desktop settings before BaseSettings is created."""
    if not USER_SETTINGS_PATH.exists():
        return
    try:
        data = json.loads(USER_SETTINGS_PATH.read_text(encoding="utf-8"))
    except Exception:
        return
    mapping = {
        "data_dir": "EDS_DATA_DIR",
        "ffmpeg_path": "FFMPEG_PATH",
        "model_cache_dir": "MODEL_CACHE_DIR",
        "runtime_mode": "RUNTIME_MODE",
        "scene_threshold": "SCENE_THRESHOLD",
        "min_clip_duration": "MIN_CLIP_DURATION",
        "max_clip_duration": "MAX_CLIP_DURATION",
        "smart_face_scan_fps": "SMART_FACE_SCAN_FPS",
        "smart_face_confidence": "SMART_FACE_CONFIDENCE",
        "smart_max_missing_face_gap": "SMART_MAX_MISSING_FACE_GAP",
        "smart_target_clip_duration": "SMART_TARGET_CLIP_DURATION",
        "smart_silence_threshold_db": "SMART_SILENCE_THRESHOLD_DB",
        "smart_silence_min_duration": "SMART_SILENCE_MIN_DURATION",
        "smart_max_dialogue_extension": "SMART_MAX_DIALOGUE_EXTENSION",
        "smart_vad_mode": "SMART_VAD_MODE",
        "cpu_threads": "EDS_CPU_THREADS",
        "ffmpeg_threads": "EDS_FFMPEG_THREADS",
        "pipeline_workers": "EDS_PIPELINE_WORKERS",
        "download_mode": "EDS_DOWNLOAD_MODE",
        "download_max_height": "EDS_DOWNLOAD_MAX_HEIGHT",
        "download_concurrent_fragments": "EDS_DOWNLOAD_CONCURRENT_FRAGMENTS",
        "download_throttled_rate_kbps": "EDS_DOWNLOAD_THROTTLED_RATE_KBPS",
        "download_use_aria2": "EDS_DOWNLOAD_USE_ARIA2",
        "download_cookies_browser": "EDS_DOWNLOAD_COOKIES_BROWSER",
        "download_cookie_file": "EDS_DOWNLOAD_COOKIE_FILE",
        "gemini_api_key": "GEMINI_API_KEY",
        "vertex_location": "VERTEX_LOCATION",
        "gemini_model": "GEMINI_MODEL",
        "ai_autocut_enabled": "AI_AUTOCUT_ENABLED",
        "ai_autocut_intensity_threshold": "AI_AUTOCUT_INTENSITY_THRESHOLD",
        "ai_autocut_min_duration": "AI_AUTOCUT_MIN_DURATION",
        "ai_autocut_max_duration": "AI_AUTOCUT_MAX_DURATION",
        "ai_autocut_padding_before": "AI_AUTOCUT_PADDING_BEFORE",
        "ai_autocut_padding_after": "AI_AUTOCUT_PADDING_AFTER",
        "ai_autocut_max_segments": "AI_AUTOCUT_MAX_SEGMENTS",
        "ai_autocut_min_face_coverage_in_scene": "AI_AUTOCUT_MIN_FACE_COVERAGE_IN_SCENE",
        "ai_autocut_verify_strict": "AI_AUTOCUT_VERIFY_STRICT",
        # Sprint 1 — hard filter & emotion lock (xem docs/03_ai_autocut_optimization.md)
        "hard_filter_min_face_coverage": "HARD_FILTER_MIN_FACE_COVERAGE",
        "hard_filter_min_intensity": "HARD_FILTER_MIN_INTENSITY",
        "hard_filter_min_frontal_ratio": "HARD_FILTER_MIN_FRONTAL_RATIO",
        "hard_filter_max_yaw_deg": "HARD_FILTER_MAX_YAW_DEG",
        "hard_filter_min_face_size_ratio": "HARD_FILTER_MIN_FACE_SIZE_RATIO",
        "hard_filter_min_snr_db": "HARD_FILTER_MIN_SNR_DB",
        "hard_filter_min_speech_segments": "HARD_FILTER_MIN_SPEECH_SEGMENTS",
        "hard_filter_min_duration_sec": "HARD_FILTER_MIN_DURATION_SEC",
        "hard_filter_max_duration_sec": "HARD_FILTER_MAX_DURATION_SEC",
        "hard_filter_max_people_in_clip": "HARD_FILTER_MAX_PEOPLE_IN_CLIP",
        "hard_filter_min_word_count": "HARD_FILTER_MIN_WORD_COUNT",
        "hard_filter_strict_mode": "HARD_FILTER_STRICT_MODE",
        "emotion_lock_enabled": "EMOTION_LOCK_ENABLED",
        "emotion_lock_max_flip_score": "EMOTION_LOCK_MAX_FLIP_SCORE",
    }
    for key, env_key in mapping.items():
        value = data.get(key)
        if value not in (None, ""):
            os.environ.setdefault(env_key, str(value))

_load_user_settings_into_env()

class Settings(BaseSettings):
    """Application configuration shared by desktop UI and backend services."""

    APP_NAME: str = "Emotion Data Studio"
    VERSION: str = "1.0.0"
    ENV: str = "development"

    HOST: str = "127.0.0.1"
    PORT: int = 8765

    BASE_DIR: Path = BASE_DIR
    
    @staticmethod
    def _default_data_dir() -> Path:
        if getattr(sys, 'frozen', False):  # PyInstaller build
            return Path(os.environ.get("LOCALAPPDATA", ".")) / "EmotionDataStudio"
        return Path(__file__).resolve().parent.parent / "data"

    DATA_DIR: Path = Path(os.getenv("EDS_DATA_DIR", _default_data_dir()))
    DB_NAME: str = "studio.db"

    FFMPEG_PATH: str = os.getenv("EDS_FFMPEG_PATH", os.getenv("FFMPEG_PATH", "ffmpeg"))
    FFPROBE_PATH: str = os.getenv("EDS_FFPROBE_PATH", os.getenv("FFPROBE_PATH", "ffprobe"))
    MODEL_CACHE_DIR: Path = Path(os.getenv("MODEL_CACHE_DIR", str(DATA_DIR / "models_cache")))
    RUNTIME_MODE: str = "auto"  # auto, cpu, cuda

    SCENE_THRESHOLD: float = float(os.getenv("SCENE_THRESHOLD", "30.0"))
    MIN_CLIP_DURATION: float = float(os.getenv("MIN_CLIP_DURATION", "3.0"))
    MAX_CLIP_DURATION: float = float(os.getenv("MAX_CLIP_DURATION", "15.0"))

    # Smart segmentation / dataset mining
    SMART_FACE_SCAN_FPS: float = float(os.getenv("SMART_FACE_SCAN_FPS", "2.0"))
    SMART_FACE_CONFIDENCE: float = float(os.getenv("SMART_FACE_CONFIDENCE", "0.55"))
    SMART_MAX_MISSING_FACE_GAP: float = float(os.getenv("SMART_MAX_MISSING_FACE_GAP", "1.0"))
    SMART_TARGET_CLIP_DURATION: float = float(os.getenv("SMART_TARGET_CLIP_DURATION", "6.0"))
    SMART_SILENCE_THRESHOLD_DB: str = os.getenv("SMART_SILENCE_THRESHOLD_DB", "-35dB")
    SMART_SILENCE_MIN_DURATION: float = float(os.getenv("SMART_SILENCE_MIN_DURATION", "0.45"))
    SMART_MAX_DIALOGUE_EXTENSION: float = float(os.getenv("SMART_MAX_DIALOGUE_EXTENSION", "1.5"))
    SMART_VAD_MODE: str = os.getenv("SMART_VAD_MODE", "auto")  # auto, energy, ffmpeg, disabled

    # Resource utilization
    EDS_CPU_THREADS: int = int(os.getenv("EDS_CPU_THREADS", "0") or "0")
    EDS_FFMPEG_THREADS: int = int(os.getenv("EDS_FFMPEG_THREADS", "0") or "0")
    EDS_PIPELINE_WORKERS: int = int(os.getenv("EDS_PIPELINE_WORKERS", "1") or "1")

    # URL download tuning
    EDS_DOWNLOAD_MODE: str = os.getenv("EDS_DOWNLOAD_MODE", os.getenv("DOWNLOAD_MODE", "balanced"))
    EDS_DOWNLOAD_MAX_HEIGHT: int = int(os.getenv("EDS_DOWNLOAD_MAX_HEIGHT", os.getenv("DOWNLOAD_MAX_HEIGHT", "720")) or "720")
    EDS_DOWNLOAD_CONCURRENT_FRAGMENTS: int = int(os.getenv("EDS_DOWNLOAD_CONCURRENT_FRAGMENTS", os.getenv("DOWNLOAD_CONCURRENT_FRAGMENTS", "5")) or "5")
    EDS_DOWNLOAD_THROTTLED_RATE_KBPS: int = int(os.getenv("EDS_DOWNLOAD_THROTTLED_RATE_KBPS", os.getenv("DOWNLOAD_THROTTLED_RATE_KBPS", "100")) or "100")
    EDS_DOWNLOAD_USE_ARIA2: bool = (os.getenv("EDS_DOWNLOAD_USE_ARIA2", os.getenv("DOWNLOAD_USE_ARIA2", "false")).strip().lower() == "true")
    EDS_DOWNLOAD_COOKIES_BROWSER: str = os.getenv("EDS_DOWNLOAD_COOKIES_BROWSER", os.getenv("DOWNLOAD_COOKIES_BROWSER", ""))
    EDS_DOWNLOAD_COOKIE_FILE: str = os.getenv("EDS_DOWNLOAD_COOKIE_FILE", os.getenv("DOWNLOAD_COOKIE_FILE", ""))

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
    EDS_UPDATE_URL: Optional[str] = os.getenv("EDS_UPDATE_URL", "https://pub-74b3008a5f904815b3951f8d440264cc.r2.dev")

    # Gemini Auto-Labeler
    GEMINI_API_KEY: Optional[str] = os.getenv("GEMINI_API_KEY", None)
    # Per docs/09_vertex_ai_integration.md: Gemini Enterprise Agent Platform
    # yeu cau location="global". User co the override qua env neu su dung AI Studio.
    VERTEX_LOCATION: str = os.getenv("VERTEX_LOCATION", "global")
    GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    GEMINI_TEMPERATURE: float = float(os.getenv("GEMINI_TEMPERATURE", "0.2"))
    GEMINI_MAX_TOKENS: int = int(os.getenv("GEMINI_MAX_TOKENS", "8192"))
    GEMINI_INTENSITY_THRESHOLD: float = float(os.getenv("GEMINI_INTENSITY_THRESHOLD", "0.6"))
    GEMINI_MONTHLY_BUDGET_USD: float = float(os.getenv("GEMINI_MONTHLY_BUDGET_USD", "500.0"))
    GEMINI_COST_TRACKING_ENABLED: bool = (
        os.getenv("GEMINI_COST_TRACKING_ENABLED", "true").strip().lower() == "true"
    )

    # Vertex AI Agent Studio — Cloud Run endpoint
    # Deployed agent URL (e.g. https://genai-app-eds-xxx.us-central1.run.app)
    AGENT_RUNTIME_URL: Optional[str] = os.getenv("AGENT_RUNTIME_URL", None)
    # API key / secret for the deployed agent (aiplatform.googleapis.com/app-secret-key)
    AGENT_API_KEY: Optional[str] = os.getenv("AGENT_API_KEY", None)

    # AI Auto-Cut mode (Vertex AI-driven segmentation)
    # Khi bat, Stage 2 (scene_split + smart_segmenter) se duoc thay the bang
    # Gemini scan + FFmpeg cut truc tiep. Neu Vertex AI loi -> fallback
    # ve SceneSplitter + SmartSegmenter cu.
    AI_AUTOCUT_ENABLED: bool = (
        os.getenv("AI_AUTOCUT_ENABLED", "false").strip().lower() == "true"
    )
    AI_AUTOCUT_INTENSITY_THRESHOLD: float = float(
        os.getenv("AI_AUTOCUT_INTENSITY_THRESHOLD", "0.55")
    )
    AI_AUTOCUT_MIN_DURATION: float = float(os.getenv("AI_AUTOCUT_MIN_DURATION", "3.0"))
    AI_AUTOCUT_MAX_DURATION: float = float(os.getenv("AI_AUTOCUT_MAX_DURATION", "15.0"))
    AI_AUTOCUT_PADDING_BEFORE: float = float(os.getenv("AI_AUTOCUT_PADDING_BEFORE", "0.5"))
    AI_AUTOCUT_PADDING_AFTER: float = float(os.getenv("AI_AUTOCUT_PADDING_AFTER", "0.5"))
    AI_AUTOCUT_MAX_SEGMENTS: int = int(os.getenv("AI_AUTOCUT_MAX_SEGMENTS", "40"))
    AI_AUTOCUT_MIN_FACE_COVERAGE_IN_SCENE: float = float(
        os.getenv("AI_AUTOCUT_MIN_FACE_COVERAGE_IN_SCENE", "0.70")
    )
    AI_AUTOCUT_VERIFY_STRICT: bool = (
        os.getenv("AI_AUTOCUT_VERIFY_STRICT", "false").lower() in {"1", "true", "yes"}
    )

    # ── Sprint 1 — Hard Filter & Emotion Lock ────────────────────────────────
    # Sau Vertex AI scan/verify, segments phai dat nguong chat luong de duoc
    # dua vao training set. Xem chi tiet: docs/03_ai_autocut_optimization.md §5.2
    #
    # HARD_FILTER_MIN_FACE_COVERAGE = 0.70 theo yeu cau cua user:
    # chi nhan clip co >= 70% frames co mat chinh dien.
    HARD_FILTER_MIN_FACE_COVERAGE: float = float(
        os.getenv("HARD_FILTER_MIN_FACE_COVERAGE", "0.70")
    )
    HARD_FILTER_MIN_INTENSITY: float = float(
        os.getenv("HARD_FILTER_MIN_INTENSITY", "0.80")
    )
    HARD_FILTER_MIN_FRONTAL_RATIO: float = float(
        os.getenv("HARD_FILTER_MIN_FRONTAL_RATIO", "0.75")
    )
    HARD_FILTER_MAX_YAW_DEG: float = float(
        os.getenv("HARD_FILTER_MAX_YAW_DEG", "30.0")
    )
    HARD_FILTER_MIN_FACE_SIZE_RATIO: float = float(
        os.getenv("HARD_FILTER_MIN_FACE_SIZE_RATIO", "0.15")
    )
    HARD_FILTER_MIN_SNR_DB: float = float(
        os.getenv("HARD_FILTER_MIN_SNR_DB", "12.0")
    )
    HARD_FILTER_MIN_SPEECH_SEGMENTS: int = int(
        os.getenv("HARD_FILTER_MIN_SPEECH_SEGMENTS", "1")
    )
    HARD_FILTER_MIN_DURATION_SEC: float = float(
        os.getenv("HARD_FILTER_MIN_DURATION_SEC", "3.0")
    )
    HARD_FILTER_MAX_DURATION_SEC: float = float(
        os.getenv("HARD_FILTER_MAX_DURATION_SEC", "12.0")
    )
    HARD_FILTER_MAX_PEOPLE_IN_CLIP: int = int(
        os.getenv("HARD_FILTER_MAX_PEOPLE_IN_CLIP", "1")
    )
    HARD_FILTER_MIN_WORD_COUNT: int = int(
        os.getenv("HARD_FILTER_MIN_WORD_COUNT", "3")
    )
    # strict_mode=False: chi log warning khi segment bi loai, van tra ve
    # danh sach segments (gom reject_reason). True: raise error neu 0 segment pass.
    HARD_FILTER_STRICT_MODE: bool = (
        os.getenv("HARD_FILTER_STRICT_MODE", "false").strip().lower() == "true"
    )

    # Emotion lock — Verify pass se so sanh emotion Scan vs Verify.
    # Neu FLIP_SCORE > max_flip_score -> segment bi danh dau "unstable".
    EMOTION_LOCK_ENABLED: bool = (
        os.getenv("EMOTION_LOCK_ENABLED", "true").strip().lower() == "true"
    )
    EMOTION_LOCK_MAX_FLIP_SCORE: float = float(
        os.getenv("EMOTION_LOCK_MAX_FLIP_SCORE", "0.3")
    )

    # ── Sprint 4 — Self-tuning Review Queue Agent ─────────────────────────────
    # Routing thresholds for the read-only agent that triages clips into
    # auto-approve / auto-reject / needs-human-review buckets.
    # Docs: docs/04_review_queue_agent_spec.md
    REVIEW_QUEUE_AUTO_APPROVE_CONFIDENCE: float = float(
        os.getenv("REVIEW_QUEUE_AUTO_APPROVE_CONFIDENCE", "0.85")
    )
    REVIEW_QUEUE_AUTO_APPROVE_QUALITY: float = float(
        os.getenv("REVIEW_QUEUE_AUTO_APPROVE_QUALITY", "0.85")
    )
    REVIEW_QUEUE_AUTO_REJECT_CONFIDENCE: float = float(
        os.getenv("REVIEW_QUEUE_AUTO_REJECT_CONFIDENCE", "0.40")
    )
    # Soft targets logged for drift detection — not enforced.
    REVIEW_QUEUE_BUCKET_AUTO_APPROVE_RATIO: float = float(
        os.getenv("REVIEW_QUEUE_BUCKET_AUTO_APPROVE_RATIO", "0.30")
    )
    REVIEW_QUEUE_BUCKET_AUTO_REJECT_RATIO: float = float(
        os.getenv("REVIEW_QUEUE_BUCKET_AUTO_REJECT_RATIO", "0.20")
    )
    # Confidence-score weights for auto_approve candidates.
    REVIEW_QUEUE_W_CONFIDENCE: float = float(os.getenv("REVIEW_QUEUE_W_CONFIDENCE", "0.30"))
    REVIEW_QUEUE_W_QUALITY: float = float(os.getenv("REVIEW_QUEUE_W_QUALITY", "0.20"))
    REVIEW_QUEUE_W_VERIFY: float = float(os.getenv("REVIEW_QUEUE_W_VERIFY", "0.40"))
    REVIEW_QUEUE_W_NO_INCONGRUITY: float = float(os.getenv("REVIEW_QUEUE_W_NO_INCONGRUITY", "0.10"))

    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @property
    def database_url(self) -> str:
        self.DATA_DIR.mkdir(parents=True, exist_ok=True)
        return f"sqlite:///{self.DATA_DIR / self.DB_NAME}"

    @property
    def user_settings_path(self) -> Path:
        return USER_SETTINGS_PATH

    def ensure_directories(self) -> None:
        self.DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        for name in ("videos", "clips", "frames", "audio", "exports", "logs", "inbox", "features", "faces", "config", "cache"):
            (self.DATA_DIR / name).mkdir(parents=True, exist_ok=True)

    @property
    def inbox_dir(self) -> Path:
        return self.DATA_DIR / "inbox"

    @property
    def colab_config_path(self) -> Path:
        return self.DATA_DIR / "config" / "pipeline_config.json"

settings = Settings()
settings.ensure_directories()
