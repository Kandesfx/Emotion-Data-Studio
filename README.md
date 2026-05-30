# 🎬 Emotion Data Studio (EDS)

> Tool desktop hỗ trợ chuẩn bị dữ liệu cho mô hình nhận diện cảm xúc đa phương thức.

## ✨ Tính năng

- **📥 Import Video** — Tải video từ YouTube hoặc file local
- **✂️ Auto Scene Split** — Tự động cắt cảnh bằng PySceneDetect
- **👤 Face Detection** — Phát hiện và tracking khuôn mặt (SCRFD + ByteTrack)
- **🔊 Audio Analysis** — Trích xuất audio, MFCC, chuyển giọng nói thành text (Whisper)
- **🎭 AI Emotion Labeling** — Ensemble voting từ 4 model (HSEmotion, DeepFace, PhoBERT, Wav2Vec2)
- **⭐ Quality Scoring** — Chấm điểm chất lượng clip tự động
- **📊 Review Studio** — Giao diện duyệt và gán nhãn thủ công (keyboard shortcuts)
- **📦 Export** — Xuất dataset với train/val/test split
- **☁️ Cloud Sync** — Đồng bộ Google Cloud (GCS + Cloud SQL)

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
| Packaging | PyInstaller + Inno Setup |

## 📁 Cấu trúc

```
emotion-data-studio/
├── app.py                 # Desktop entry point
├── app_cloud.py           # Cloud Run entry point
├── backend/               # Backend services
│   ├── config.py
│   ├── database/          # SQLAlchemy models + SQLite
│   ├── services/          # AI pipeline (7 services)
│   ├── ai_models/         # Model manager
│   └── cloud/             # GCS, Cloud SQL, Sync
├── ui/                    # PySide6 Desktop UI
│   ├── main_window.py
│   ├── pages/             # 4 pages (Dashboard, Processing, Review, Export)
│   ├── widgets/           # Sidebar, custom widgets
│   ├── workers/           # QThread workers
│   └── styles/            # Dark theme QSS
├── build/                 # PyInstaller spec + build script
├── deploy/                # Dockerfile + Cloud Build
└── installer/             # Inno Setup script
```

## 📋 License

Internal tool — BCDA Team
