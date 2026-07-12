# 🎬 Emotion Data Studio (EDS)

> Tool desktop + REST API hỗ trợ chuẩn bị dữ liệu huấn luyện cho mô hình nhận diện cảm xúc đa phương thức (MMSA / MulT).
>
> **Phiên bản hiện tại: 1.3.0** — đã tích hợp **Vertex AI (Gemini)** với **2-pass Verify** cho chế độ AI Auto-Cut.

## ✨ Tính năng

- **📥 Import Video** — Tải video từ YouTube (yt-dlp với 3 profile + cookies) hoặc file local
- **✂️ Auto Scene Split** — PySceneDetect + SmartSegmenter (face/dialogue-aware) — classic mode
- **🤖 AI Auto-Cut (Vertex AI)** — Gemini 2.5 Flash tự quét video và cắt trực tiếp ra clip cảm xúc mạnh — **mode mới**
- **🔍 Verify Pass (Sprint 2)** — Gemini verify lại emotion/intensity + quality cho từng clip với 5 verdict (`confirmed` / `wrong_emotion` / `unstable` / `low_quality` / `stats_mismatch`)
- **🛡️ Hard Filter (Sprint 1)** — Pre-filter Gemini raw output với 5 tiêu chí (face coverage, frontal, motion, speech, people) → chỉ giữ clip chất lượng train
- **⚡ Cache Layer (Sprint 3.2)** — Disk-based JSON cache 7 ngày TTL, tiết kiệm cost khi re-run
- **💰 Cost Breakdown** — Track riêng scan cost + verify cost + grand_total cho mỗi video
- **🛡️ Strict Mode** — Bật → tự loại clip bị Verify reject khỏi output
- **✂️ Manual Segment Editor** — Timeline + playback + keyboard shortcuts
- **👤 Face Detection** — SCRFD / MTCNN + ByteTrack, lưu detections.json
- **🔊 Audio Analysis** — FFmpeg → WAV 16kHz → MFCC 74-dim (COVAREP-compatible)
- **📝 Speech-to-Text** — Whisper medium + faster-whisper (Vietnamese prompt)
- **🎭 AI Ensemble Voting** — DeepFace (visual) + PhoBERT-ViLexicon (text) + Wav2Vec2 (audio)
- **⭐ Quality Scoring + Auto-Decision** — 9-criteria gate → auto_approved / needs_review / rejected
- **📊 Review Studio (NLE-style)** — Media bin + preview + timeline + inspector, keyboard shortcuts, **Verify verdict badge**
- **🎯 MMSA Export** — Xuất `.pkl` đúng format MulT: `(50, 768)` text / `(50, 74)` audio / `(50, 35)` vision
- **📦 Dataset Export** — Full / Compact / Labels-only, stratified split 70/15/15
- **☁️ Cloud Sync** — GCS + Cloud SQL bidirectional
- **🔄 Auto-Updater** — Cloudflare R2 endpoint
- **🐳 Cloud Run + Colab Worker** — GPU job queue từ xa

## ⚙️ 4 Chế Độ Xử Lý (sau khi tích hợp Vertex AI)

| Chế độ | Mô tả | Cắt bằng | Lọc bằng |
|---|---|---|---|
| 🤖 **Full Auto** (Vertex AI) | Gemini tự quét + cắt → ensemble voting → review | Vertex AI scan + FFmpeg | Ensemble + 9-criteria gate |
| 🤖 **Full Auto** (classic) | Scene-detect + AI gán nhãn → review | SceneSplitter + SmartSegmenter | Ensemble + 9-criteria gate |
| 🔀 **Semi-Auto** | User cắt thủ công → AI gán nhãn → review | UI Segment Editor | Ensemble + 9-criteria gate |
| ✋ **Full Manual** | User cắt + gán nhãn, AI trích xuất feature | UI Segment Editor | User review |

**Fallback tự động**: nếu bật AI Auto-Cut nhưng Vertex AI lỗi → tự rơi về classic Full Auto (PySceneDetect).

## 🆕 Sprint 3.9 — Verify pass qua UI/API/DB (10/07/2026)

Sprint 1–3 đã hoàn tất backend pipeline (hard filter + verify pass + cache + cost breakdown), nhưng chưa wiring vào UI/API/DB. Sprint 3.9 đóng các gap cuối để user có thể sử dụng Verify pass qua UI mà không cần gọi code trực tiếp.

| Thành phần | Thay đổi chính |
|---|---|
| **API `POST /api/gemini/cut-and-create`** | Tích hợp Verify pass sau `cut_video()`. Response mới có `verify_summary`, `stage_counters`, `cost_breakdown`, per-clip `verify_verdict`. |
| **Settings page** | Thêm 2 control: `Verify strict mode` (combo) + `Min face coverage scene` (spinbox 0.50–0.95). |
| **Review page** | `_on_gemini_verify()` rewrite: dùng Sprint 2 `verify_clip()` + `combine_verdicts()`. Dialog hiển thị verdict emoji + reasoning + status. |
| **DB `Clip` model** | +4 columns: `verify_verdict`, `verify_status`, `verify_reasoning`, `rejected_by_verify`. |
| **Migration script** | `scripts/migrate_add_verify_columns.py` — idempotent ALTER TABLE. |
| **Test scripts** | `data/scripts/test_e2e_bongma.py` (E2E trên video BÓNG MA), `data/scripts/test_multi_emotion.py` (multi-label). |

**Test**: 106/106 unit tests pass + E2E trên BÓNG MA HẠNH PHÚC Tập 1 (4 clips, $1.23, all `confirmed`).

Chi tiết đầy đủ: [`docs/03_ai_autocut_optimization.md`](docs/03_ai_autocut_optimization.md) §7.

## 🚀 Quick Start

### Yêu cầu

- Python 3.12+
- FFmpeg trong PATH (hoặc set `FFMPEG_PATH`)
- Service Account GCP key (cho Vertex AI — tùy chọn)

### Cài đặt

```bash
cd tools/emotion-data-studio
pip install -r requirements.txt
```

### Chạy Desktop App

```bash
python app.py
```

### Chạy REST API server (cho Electron/web client)

```bash
uvicorn backend.main:app --reload --port 8765
```

### Kiểm tra tích hợp Vertex AI

```powershell
$env:GOOGLE_APPLICATION_CREDENTIALS = "d:\...\aura-social-vn-e7a147284c33.json"
python scripts/test_ai_autocut_integration.py
```

### Bật AI Auto-Cut

**Cách 1 — Settings UI**: Mở app → tab **Cài Đặt** → section "Vertex AI — Gemini Auto-Cut":
- Tick `Bật AI Auto-Cut mặc định`
- Tick `Verify strict mode` *(tùy chọn — bật sẽ tự loại clip bị Verify reject)*
- Chỉnh `Intensity threshold` (mặc định 0.55)
- Chỉnh `Min face coverage scene` *(mặc định 0.70)*
- Bấm `Lưu cài đặt` → status label sẽ hiển thị `✅ Vertex AI (global) ready`

**Cách 2 — Dashboard toggle**: Tab "Bảng Điều Khiển" → tick checkbox "🤖 AI Auto-Cut" → bấm "Xử lý"

**Cách 3 — REST API**:
```bash
curl -X POST http://127.0.0.1:8765/api/gemini/cut-and-create \
     -H "Content-Type: application/json" \
     -d '{"video_id": "<uuid>", "intensity_threshold": 0.55}'
```

Response (Sprint 3.9):
```json
{
  "status": "ok",
  "total_segments": 4,
  "clips_inserted": 4,
  "cost_breakdown": {
    "scan": {"total_usd": 1.19, "input_tokens": 25000, "output_tokens": 800},
    "verify": {"total_usd": 0.04, "n_clips": 4},
    "grand_total_usd": 1.23
  },
  "stage1_candidates": 16,
  "stage2_passed": 4,
  "stage3_cut": 4,
  "stage4_verified": 4,
  "verify_summary": {
    "total": 4,
    "passed": 4,
    "rejected": 0,
    "by_verdict": {"confirmed": 4}
  },
  "clips": [
    {
      "clip_id": "...",
      "verify_verdict": "confirmed",
      "verify_status": "passed",
      "rejected_by_verify": false
    }
  ]
}
```

**Cách 4 — `.env`** (cho Docker / Colab / startup):
```env
AI_AUTOCUT_ENABLED=true
AI_AUTOCUT_VERIFY_STRICT=false
AI_AUTOCUT_MIN_FACE_COVERAGE_IN_SCENE=0.70
VERTEX_LOCATION=global                      # BAT BUOC
GOOGLE_APPLICATION_CREDENTIALS=.../aura-social-vn-e7a147284c33.json
```

### Migrate DB (một lần khi upgrade từ Sprint 1)

```bash
python scripts/migrate_add_verify_columns.py
```

Output mong đợi:
```
+ Added column: verify_verdict (TEXT)
+ Added column: verify_status (TEXT DEFAULT 'not_run')
+ Added column: verify_reasoning (TEXT)
+ Added column: rejected_by_verify (INTEGER DEFAULT 0)
Migration completed successfully.
```

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| GUI | PySide6 (Qt6) — Native desktop |
| Database | SQLite (local) + PostgreSQL (cloud) |
| AI Video | **Vertex AI Gemini 2.5 Flash** (global, google-genai SDK) — **2-pass scan + verify** |
| Cache | Disk-based JSON cache (7 ngày TTL) |
| Local AI | PyTorch, Whisper, DeepFace, PhoBERT, MTCNN, ByteTrack |
| Video | FFmpeg, PySceneDetect, yt-dlp |
| Cloud | Google Cloud Storage, Cloud SQL, Cloud Run |
| Updates | Cloudflare R2 + Auto-Updater |
| Packaging | PyInstaller + Inno Setup |
| CI | GitHub Actions — `requirements-ci.txt` + 3 sprint test suites |

## 📐 Engineering Practices

| Doc | Purpose |
|---|---|
| [docs/AI_CODING_WORKFLOW.md](docs/AI_CODING_WORKFLOW.md) | Spec-first workflow used with AI coding agents; review checklist; real examples from Sprint 1–3.9 |
| [docs/SECURITY_REVIEW.md](docs/SECURITY_REVIEW.md) | Threat model + mitigations with file:line references; residual risks; follow-ups |
| [docs/04_review_queue_agent_spec.md](docs/04_review_queue_agent_spec.md) | Sprint 4 — Self-tuning Review Queue Agent: routing rules, confidence scoring, anti-goals |
| [`.env.example`](.env.example) | Template for environment variables — copy to `.env` and fill in. Never commit real credentials |
| [docs/03_ai_autocut_optimization.md](docs/03_ai_autocut_optimization.md) | Sprint log: every decision, deviation, bug, and lesson learned from Sprint 1 → 3.9 |
| [.github/workflows/test.yml](.github/workflows/test.yml) | CI: import checks + 181 unit tests (Sprint 1–4), no torch/whisper needed |

## 📁 Cấu trúc (cập nhật sau Vertex AI integration)

```
emotion-data-studio/
├── app.py                          # Desktop entry point (PySide6)
├── app_cloud.py                    # Cloud Run entry (FastAPI)
├── web/main.py                     # Colab Web dashboard (FastAPI)
├── .env / .env.example             # Environment config (cần VERTEX_LOCATION=global)
│
├── backend/
│   ├── config.py                   # ★ AI_AUTOCUT_* settings + VERIFY_STRICT + MIN_FACE_COVERAGE
│   ├── main.py                     # FastAPI app
│   ├── database/                   # SQLAlchemy + SQLite
│   │   └── models.py               # ★ Clip + verify_verdict/status/reasoning/rejected_by_verify
│   ├── services/
│   │   ├── pipeline_orchestrator.py # ★ Stage 2 dispatcher: AI vs classic
│   │   ├── gemini_auto_labeler.py   # ★ Refactor: get_genai_client + is_vertex_configured
│   │   ├── ai_video_segmenter.py    # ★ Vertex AI scan + verify_clip + combine_verdicts + persist_clips
│   │   ├── gemini_cache.py          # ★ Sprint 3.2 — Disk-based cache (7-day TTL)
│   │   ├── scene_splitter.py
│   │   ├── smart_segmenter.py
│   │   ├── face_extractor.py
│   │   ├── audio_extractor.py
│   │   ├── transcriber.py
│   │   ├── emotion_analyzer.py
│   │   ├── quality_scorer.py
│   │   ├── auto_decision.py
│   │   └── feature_extractors/      # MMSA-compatible
│   ├── cloud/                       # GCS + Cloud SQL
│   └── api/                         # FastAPI routers (8)
│       └── gemini_api.py            # ★ /cut-and-create tích hợp Verify pass (Sprint 3.9)
│
├── ui/                              # PySide6 Desktop UI (7 pages)
│   ├── main_window.py
│   ├── pages/
│   │   ├── dashboard_page.py        # ★ CheckBox AI AutoCut + intensity
│   │   ├── processing_page.py       # ★ Stage "AI CUT"
│   │   ├── segment_editor_page.py
│   │   ├── review_page.py           # ★ _on_gemini_verify rewrite + Verify verdict badge
│   │   └── settings_page.py         # ★ Card Vertex AI + Verify strict + Min face coverage
│   ├── widgets/
│   ├── workers/                     # QThread (pipeline/segment/export)
│   └── styles/
│
├── scripts/                          # ★ Test scripts + migration
│   ├── test_ai_autocut_integration.py  # ★ 4 + 1 integration tests
│   ├── verify_integration_imports.py   # ★ 10 import checks
│   └── migrate_add_verify_columns.py   # ★ Sprint 3.9 — idempotent DB migration
│
├── data/
│   ├── scripts/
│   │   ├── test_e2e_bongma.py       # ★ Sprint 3.9 — E2E test trên video BÓNG MA
│   │   └── test_multi_emotion.py    # ★ Sprint 3.9 — Test multi-label emotion
│   ├── test_outputs/                # Raw Gemini responses, results JSON
│   ├── clips/                       # Generated clip files
│   └── cache/gemini/                # Sprint 3.2 cache (7-day TTL)
│
├── docs/                             # ★ Tài liệu kỹ thuật
│   ├── 01_vertex_ai_integration.md   # Hướng dẫn tích hợp Vertex AI
│   ├── 02_ai_autocut_design.md       # Thiết kế AI Auto-Cut
│   └── 03_ai_autocut_optimization.md # ★ Sprint log: Sprint 1 → 3.9 (live doc)
│
├── build/                          # PyInstaller + R2 publish
├── deploy/                         # Dockerfile + cloudbuild
└── installer/                      # Inno Setup script
```

## 📚 Tài liệu

| Doc | Mô tả |
|---|---|
| [docs/01_vertex_ai_integration.md](docs/01_vertex_ai_integration.md) | Hướng dẫn kết nối Vertex AI từ A→Z: pattern `global`, SDK `google-genai`, troubleshooting |
| [docs/02_ai_autocut_design.md](docs/02_ai_autocut_design.md) | Thiết kế AI Auto-Cut: kiến trúc, contracts, fallback, UI/UX, edge cases |
| [docs/03_ai_autocut_optimization.md](docs/03_ai_autocut_optimization.md) | **Sprint log**: tất cả thay đổi, bug fixes, lessons learned từ Sprint 1 → 3.9 |

## 📦 Build & Release

```powershell
# Build app + installer
.\build\build.ps1 -Version "1.3.0"

# Publish release lên Cloudflare R2
.\build\publish_release.ps1 -Version "1.3.0" -ReleaseNotes "Verify pass + UI integration (Sprint 3.9)"

# Dry run
.\build\publish_release.ps1 -Version "1.3.0" -DryRun
```

## ☁️ Cloud Deploy

```bash
gcloud builds submit --config deploy/cloudbuild.yaml
curl https://your-url.run.app/health
```

## 🧪 Test

```bash
# Test Vertex AI tích hợp (cần GOOGLE_APPLICATION_CREDENTIALS)
$env:GOOGLE_APPLICATION_CREDENTIALS = "..../aura-social-vn-e7a147284c33.json"
python scripts/test_ai_autocut_integration.py

# Full AutoCut với 1 video mẫu
python scripts/test_ai_autocut_integration.py --with-video data/test.mp4

# Quick verify imports (không cần network)
python scripts/verify_integration_imports.py

# Sprint 1-3 unit tests (106 tests total)
python ../tests/test_sprint1_autocut.py
python ../tests/test_sprint2_autocut.py
python ../tests/test_sprint3_autocut.py

# End-to-end test trên video BÓNG MA HẠNH PHÚC (Sprint 3.9)
python data/scripts/test_e2e_bongma.py

# Multi-emotion test — test nhiều nhãn emotion/sentiment (Sprint 3.9)
python data/scripts/test_multi_emotion.py --all --max-size 5
```

Output mong đợi:

```
✅ PASS  vertex_config     # is_vertex_configured() -> True, project=aura-social-vn, location=global
✅ PASS  text_gen          # Gemini text "Xin chào!"
✅ PASS  multimodal        # Gemini vision/image
✅ PASS  ai_segmenter      # AIVideoSegmenter ready
✅ PASS  full_autocut      # Full end-to-end (optional)
```

E2E test trên BÓNG MA (Sprint 3.9):
```
Stage 2 (scan):        ✅ PASS
Stage 4 (verify):      ✅ PASS
Cost breakdown:        ✅ PRESENT
Cache layer:           ✅ 3 entries
Persist:               ✅ 2 clips
```

## 📋 License

Internal tool — BCDA Team.

---

### ℹ️ Lịch sử thay đổi gần đây

**1.3.0** *(10/07/2026 — Sprint 3.9)*
- ✨ **UI/API/DB integration** — Verify pass chạy qua UI, API, DB
- ✨ `POST /api/gemini/cut-and-create` tích hợp Verify pass cho từng clip
- ✨ Review page `_on_gemini_verify()` rewrite dùng Sprint 2 pipeline
- ✨ Settings page thêm 2 control: `Verify strict mode` + `Min face coverage scene`
- ✨ `Clip` model +4 columns: `verify_verdict`, `verify_status`, `verify_reasoning`, `rejected_by_verify`
- ✨ Migration script `scripts/migrate_add_verify_columns.py` (idempotent)
- 🧪 Test scripts mới: `data/scripts/test_e2e_bongma.py`, `data/scripts/test_multi_emotion.py`
- 📚 Docs: `docs/03_ai_autocut_optimization.md` §7 (Sprint 3.9 log)

**1.2.0** *(Jul 2026 — Sprint 1-3)*
- ✨ **Gemini Verify pass (Sprint 2)** — verify lại emotion/intensity + quality cho từng clip
- ✨ **Hard filter (Sprint 1)** — 5 tiêu chí chất lượng pre-filter raw Gemini output
- ✨ **Cache layer (Sprint 3.2)** — Disk-based JSON cache 7 ngày TTL
- ✨ **Cost breakdown** — Track scan + verify riêng, per-video grand total
- 🛡️ **Strict mode** — Tùy chọn tự loại clip bị Verify reject
- 🧪 106/106 unit tests pass (Sprint 2: 59, Sprint 3: 47)

**1.1.0** *(Jul 2026 — Vertex AI integration)*
- ✨ Tích hợp **Vertex AI (Gemini 2.5 Flash global)** cho AI Auto-Cut
- ✨ Service mới `backend/services/ai_video_segmenter.py`
- ✨ Endpoint REST `POST /api/gemini/cut-and-create`
- ✨ UI: Dashboard toggle + Settings card + Processing stage
- 🔧 Refactor `gemini_auto_labeler.py`: tách helper `get_genai_client()`, `is_vertex_configured()`
- 🔧 `.env`: `VERTEX_LOCATION` mặc định = `global` (không `us-central1`)
- 📚 Docs mới: `docs/01_vertex_ai_integration.md`, `docs/02_ai_autocut_design.md`
- 🧪 Test scripts: `test_ai_autocut_integration.py`, `verify_integration_imports.py`

**1.0.0** *(Jul 2026)*
- ✨ Tích hợp **Vertex AI (Gemini 2.5 Flash global)** cho AI Auto-Cut
- ✨ Service mới `backend/services/ai_video_segmenter.py`
- ✨ Endpoint REST `POST /api/gemini/cut-and-create`
- ✨ UI: Dashboard toggle + Settings card + Processing stage
- 🔧 Refactor `gemini_auto_labeler.py`: tách helper `get_genai_client()`, `is_vertex_configured()` theo pattern `docs/09_vertex_ai_integration.md`
- 🔧 `.env`: `VERTEX_LOCATION` mặc định = `global` (không `us-central1`)
- 📚 Docs mới: `docs/01_vertex_ai_integration.md`, `docs/02_ai_autocut_design.md`
- 🧪 Test scripts: `test_ai_autocut_integration.py`, `verify_integration_imports.py`
