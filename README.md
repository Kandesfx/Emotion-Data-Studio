# 🎬 Emotion Data Studio (EDS)

> Tool desktop hỗ trợ chuẩn bị dữ liệu cho mô hình nhận diện cảm xúc đa phương thức.

## ✨ Tính năng

- **📥 Import Video** — Tải video từ YouTube hoặc file local
- **✂️ Auto Scene Split** — Tự động cắt cảnh bằng PySceneDetect
- **✂️ Segment Editor** — Cắt phân đoạn thủ công với timeline, playback controls, keyboard shortcuts
- **👤 Face Detection** — Phát hiện và tracking khuôn mặt (SCRFD + ByteTrack)
- **🔊 Audio Analysis** — Trích xuất audio, MFCC, chuyển giọng nói thành text (Whisper)
- **🎭 AI Emotion Labeling** — Ensemble voting từ 4 model (HSEmotion, DeepFace, PhoBERT, Wav2Vec2)
- **⭐ Quality Scoring** — Chấm điểm chất lượng clip tự động
- **📊 Review Studio** — Giao diện duyệt và gán nhãn thủ công (keyboard shortcuts, reviewer notes)
- **📦 Export** — Xuất dataset với train/val/test split
- **☁️ Cloud Sync** — Đồng bộ Google Cloud (GCS + Cloud SQL)
- **🔄 Auto-Updater** — Tự kiểm tra và cập nhật phiên bản mới từ Cloudflare R2

## ⚙️ 3 Chế Độ Xử Lý

| Chế độ | Mô tả |
|---|---|
| 🤖 **Full Auto** | Tải video → Scene Detect → AI gán nhãn → Review |
| 🔀 **Semi-Auto** | Tải video → Người dùng cắt thủ công → AI gán nhãn → Review |
| ✋ **Full Manual** | Tải video → Người dùng cắt + gán nhãn → Trích xuất audio/text |

## 🚀 Quick Start

```bash
# Cài đặt dependencies
pip install -r requirements.txt

# Chạy ứng dụng desktop
python app.py
```

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| GUI | PySide6 (Qt6) — Native desktop |
| Database | SQLite (local) + PostgreSQL (cloud) |
| AI Pipeline | PyTorch, Whisper, DeepFace, PhoBERT |
| Video | FFmpeg, PySceneDetect, yt-dlp |
| Cloud | Google Cloud Storage, Cloud SQL, Cloud Run |
| Updates | Cloudflare R2 + Auto-Updater |
| Packaging | PyInstaller + Inno Setup |

## 📁 Cấu trúc

```
emotion-data-studio/
├── app.py                 # Desktop entry point
├── app_cloud.py           # Cloud Run entry point
├── .env.example           # Environment config template
├── backend/               # Backend services
│   ├── config.py
│   ├── database/          # SQLAlchemy models + SQLite
│   ├── services/          # AI pipeline (7 services)
│   ├── ai_models/         # Model manager (singleton, pre-warming)
│   ├── cloud/             # GCS, Cloud SQL, Sync Manager
│   └── api/               # FastAPI REST endpoints
├── ui/                    # PySide6 Desktop UI
│   ├── main_window.py
│   ├── updater.py         # Auto-updater (R2 check + download)
│   ├── pages/             # 5 pages
│   │   ├── dashboard_page.py      # Import + Stats + 3 mode selection
│   │   ├── processing_page.py     # Real-time pipeline monitor
│   │   ├── segment_editor_page.py # Manual clip cutting
│   │   ├── review_page.py         # AI review + labeling
│   │   └── export_page.py         # Dataset export
│   ├── widgets/           # Sidebar, custom widgets
│   ├── workers/           # QThread workers
│   │   ├── pipeline_worker.py     # Full/Download-only pipeline
│   │   ├── segment_worker.py      # Manual segment processing
│   │   └── export_worker.py       # Export worker
│   └── styles/            # Dark theme QSS
├── build/                 # Build pipeline
│   ├── build.ps1              # PyInstaller + Inno Setup
│   ├── publish_release.ps1    # Upload release to R2
│   └── emotion_studio.spec   # PyInstaller spec
├── deploy/                # Cloud deployment
│   ├── Dockerfile             # Cloud Run container
│   └── cloudbuild.yaml        # CI/CD config
└── installer/             # Windows installer
    └── emotion_data_studio.iss  # Inno Setup script
```

## 📦 Build & Release

```powershell
# Build app + installer
.\build\build.ps1 -Version "1.0.0"

# Publish release to Cloudflare R2
.\build\publish_release.ps1 -Version "1.0.0" -ReleaseNotes "Bug fixes"

# Dry run (test without uploading)
.\build\publish_release.ps1 -Version "1.0.0" -DryRun
```

## ☁️ Cloud Deploy

```bash
# Deploy Cloud Run API
gcloud builds submit --config deploy/cloudbuild.yaml

# Health check
curl https://your-url.run.app/health
```

## 📋 License

Internal tool — BCDA Team
