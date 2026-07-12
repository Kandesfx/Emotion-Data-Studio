# Vertex AI Integration — Emotion Data Studio

Tích hợp Google Vertex AI (Gemini Enterprise Agent Platform) để:
1. **Phân tích cảm xúc đa phương thức** (text / ảnh / video native input).
2. **AI Auto-Cut**: AI tự quét video và cắt trực tiếp ra các đoạn cảm xúc mạnh — thay thế cho stage scene-split + smart-segmenter khi bật.

---

## 1. Vấn đề ban đầu

Lỗi `404 NOT_FOUND` khi gọi model Generative trên Vertex AI:

> `Publisher Model projects/aura-social-vn/locations/us-central1/publishers/google/models/gemini-2.0-flash was not found or your project does not have access to it.`

Dù `aiplatform.googleapis.com` đã enable, `roles/aiplatform.user` đã gán, `gcloud auth` đã xác thực.

## 2. Giải pháp (theo `docs/09_vertex_ai_integration.md`)

### A. Location phải là `global`

Gemini Enterprise Agent Platform chỉ phục vụ model qua `locations/global`, **không** qua region địa lý (`us-central1`, `asia-southeast1`, …).

```
https://aiplatform.googleapis.com/v1/projects/{PROJECT}/locations/global/publishers/google/models/{MODEL}:generateContent
```

### B. SDK chuẩn — `google-genai`

Thay vì gọi REST thủ công hoặc qua `subprocess gcloud`, dùng SDK chính thức:

```python
import google.genai as genai

client = genai.Client(
    vertexai=True,
    project="aura-social-vn",
    location="global",
    credentials=service_account_credentials,
    http_options={"api_version": "v1"},
)
```

### C. Service Account JSON

Project `aura-social-vn` sử dụng Service Account key tại:

```
tools/emotion-data-studio/aura-social-vn-e7a147284c33.json
```

Key phải có quyền tối thiểu `Agent Platform User` (`roles/aiplatform.user`).

---

## 3. Cấu hình

### `.env`

```env
# Per docs/09_vertex_ai_integration.md
GOOGLE_APPLICATION_CREDENTIALS=d:/.../aura-social-vn-e7a147284c33.json
GCP_PROJECT_ID=aura-social-vn
GCS_BUCKET_NAME=eds-data-bucket-aura

# BAT BUOC: global, KHONG dung us-central1
VERTEX_LOCATION=global
GEMINI_MODEL=gemini-2.5-flash

# AI Auto-Cut (optional)
AI_AUTOCUT_ENABLED=false           # bat/tat o Settings UI hoac Dashboard toggle
AI_AUTOCUT_INTENSITY_THRESHOLD=0.55 # chi giu segment co intensity >= 0.55
AI_AUTOCUT_MIN_DURATION=3.0        # giay
AI_AUTOCUT_MAX_DURATION=15.0       # giay
AI_AUTOCUT_PADDING_BEFORE=0.5      # them buffer truoc
AI_AUTOCUT_PADDING_AFTER=0.5       # them buffer sau
AI_AUTOCUT_MAX_SEGMENTS=40         # toi da moi video
```

### `backend/config.py`

| Setting | Default | Mô tả |
|---|---|---|
| `VERTEX_LOCATION` | `global` | Bắt buộc. Override qua env nếu dùng AI Studio. |
| `GEMINI_MODEL` | `gemini-2.5-flash` | Model cho cả pre-filter + AutoCut + verify. |
| `GEMINI_TEMPERATURE` | `0.2` | Có thể chỉnh trong code khi tạo `GeminiAutoLabeler()`. |
| `GEMINI_MAX_TOKENS` | `8192` | Token tối đa cho mỗi response. |
| `AI_AUTOCUT_*` | xem trên | Pipeline orchestrator đọc ở runtime để dispatch sang AI Auto-Cut. |

### `user_settings.json` (do Settings UI quản lý)

Settings UI ghi đè vào cả `os.environ` lẫn in-memory `settings`, nên thay đổi có hiệu lực ngay trong session — không cần restart app.

---

## 4. Kiến trúc

### A. Helper — `backend/services/gemini_auto_labeler.py`

Tách riêng 3 hàm helper ở đầu file, dùng chung cho cả `GeminiAutoLabeler` và `AIVideoSegmenter`:

```python
VERTEX_GLOBAL_LOCATION = "global"

def _resolve_credentials() -> tuple[Any, str]:
    """Service account -> credentials, project_id"""

def get_genai_client(location="global", force_vertex=True):
    """Tra ve (genai.Client, project_id, location)"""

def is_vertex_configured() -> tuple[bool, str]:
    """Kiem tra san sang, khong can round-trip network"""
```

### B. Service chính — `backend/services/ai_video_segmenter.py` (~370 dòng)

Service mới, hoàn toàn độc lập với SceneSplitter / SmartSegmenter cũ.

```text
INPUT: video_path, video_id
   │
   ├─ 1. ffprobe -> duration
   ├─ 2. _call_gemini_segments()
   │     ├─ uu tien GCS native input (neu co GCS_BUCKET_NAME)
   │     ├─ fallback: ffmpeg 1fps -> base64 frames (batch 24)
   │     └─ goi Gemini -> JSON list segments
   ├─ 3. _parse_segments() → validate + dedup overlap + clamp duration + padding
   ├─ 4. _cut_with_ffmpeg() cho tung segment → clips/*.mp4
   └─ 5. persist_clips() → bulk insert Clip records

OUTPUT: AutoCutResult {
  clips: [AutoCutSegment],
  total_segments, total_cost_usd,
  video_duration, source="ai_autocut"
}
```

**Clip record schema:**

| Field | Value |
|---|---|
| `id` | `{video_id}_ai_{idx}_{uuid8}` |
| `clip_index` | 0..N |
| `status` | `needs_review` |
| `decision_by` | `gemini_autocut` |
| `pipeline_stage` | `ai_autocut_done` |
| `predicted_emotion` | emotion của AI |
| `confidence` | intensity (0-1) |
| `face_ratio` | face_coverage |
| `reviewer_notes` | `[Gemini AutoCut] subject=... \| reasoning=...` |
| `per_model_scores.gemini_autocut` | emotion / intensity / subject / reasoning / has_transcript |
| `per_model_scores.autocut_meta` | source / cost_usd / model |

### C. Tích hợp pipeline — `backend/services/pipeline_orchestrator.py`

Stage 2 được dispatch thành 2 helper:

```python
# Trong run_pipeline(), sau stage download:
if settings.AI_AUTOCUT_ENABLED and self._vertex_ai_ready():
    clips_metadata = self._ai_autocut_stage(...)        # Vertex AI scan + FFmpeg cut
    if total_clips == 0:
        clips_metadata = self._classic_cut_stage(...)  # fallback scene-split
else:
    clips_metadata = self._classic_cut_stage(...)
```

**Fallback chain:**

```
AI_AUTOCUT_ENABLED = True + Vertex AI ready  → AI AutoCut (stage 2)
AI_AUTOCUT_ENABLED = True + Vertex AI lỗi     → classic cut
AI_AUTOCUT_ENABLED = False                    → classic cut
```

Stage 3-6 (face detect, audio extract, transcribe, ensemble, feature extraction) **chạy bình thường** cho cả 2 mode — không phải AI AutoCut thì bỏ qua stage nào.

### D. REST API — `backend/api/gemini_api.py`

```http
POST /api/gemini/cut-and-create
Content-Type: application/json
{
  "video_id": "<uuid>",
  "intensity_threshold": 0.55,  // optional override
  "max_segments": 30             // optional override
}
```

Response:

```json
{
  "status": "ok",
  "video_id": "...",
  "video_duration_sec": 120.5,
  "total_segments": 8,
  "clips_inserted": 8,
  "estimated_cost_usd": 0.0278,
  "clips": [
    {"clip_id": "...", "start_time": 12.5, "end_time": 24.0,
     "emotion": "angry", "intensity": 0.87, "subject": "người đàn ông trung niên..."},
    ...
  ]
}
```

### E. UI Desktop — 3 nơi tích hợp

**`ui/pages/dashboard_page.py`**
- Checkbox "🤖 AI Auto-Cut (Vertex AI)" + QDoubleSpinBox intensity
- QTimer.singleShot → `_refresh_ai_status()` check credentials ngay khi mở trang
- `_apply_ai_autocut_settings()` ghi vào `settings` runtime trước khi `_start_pipeline()`

**`ui/pages/settings_page.py`** — Card "Vertex AI — Gemini Auto-Cut"
- `vertex_location_input` (default `global`)
- `gemini_model_input` (default `gemini-2.5-flash`)
- `gemini_api_key_input` (optional, để dùng AI Studio)
- `gac_path_input` (Service Account JSON, browse file)
- `ai_autocut_combo` (true/false default)
- `ai_autocut_threshold`, `min_duration`, `max_duration`, `max_segments`
- Button "Kiểm tra ngay" → `is_vertex_configured()` hiển thị status

**`ui/pages/processing_page.py`** — STAGES list thêm entry:
```python
("ai_autocut", "AI CUT", "Vertex AI quét + cắt tự động"),
```

---

## 5. Kiểm tra & Debug

### Test integration (4 + 1 tests):

```bash
cd tools/emotion-data-studio
$env:GOOGLE_APPLICATION_CREDENTIALS = ".../aura-social-vn-e7a147284c33.json"
python scripts/test_ai_autocut_integration.py
python scripts/test_ai_autocut_integration.py --with-video path/to/short.mp4
```

Output mong đợi:

```
✅ PASS  vertex_config      # is_vertex_configured() -> True, project=aura-social-vn, location=global
✅ PASS  text_gen           # Gemini text reply "Xin chào!"
✅ PASS  multimodal         # Gemini mo ta image (PNG test)
✅ PASS  ai_segmenter       # AIVideoSegmenter.is_configured() -> True
✅ PASS  full_autocut       # (with --with-video) Gemini scan + FFmpeg cut thanh cong
```

### Verify imports (10 module):

```bash
python scripts/verify_integration_imports.py
```

### Debug checklist

Khi gặp lỗi, kiểm tra theo thứ tự:

1. **Credentials**: Service Account key còn hạn không? Có quyền `Agent Platform User` chưa?
   ```bash
   python -c "from backend.services.gemini_auto_labeler import is_vertex_configured; print(is_vertex_configured())"
   ```
2. **Location**: Phải là `global` (KHÔNG `us-central1`):
   ```bash
   python -c "from backend.config import settings; print(settings.VERTEX_LOCATION)"
   ```
3. **SDK**: `google-genai` đã cài chưa:
   ```bash
   pip show google-genai
   ```
4. **Quota**: Project `aura-social-vn` còn quota Vertex AI không? (Google Cloud Console → IAM & Admin → Quotas)
5. **Network**: Trong Colab / Docker cần outbound HTTPS tới `aiplatform.googleapis.com` (port 443).

---

## 6. Chi phí ước tính

Gemini 2.5 Flash pricing (Vertex AI):

| Input | Output |
|---|---|
| $0.30 / 1M tokens | $2.50 / 1M tokens |

Ước tính (theo code `_estimate_cost`):

```
input_tokens  = duration_sec * 7000
output_tokens = 1024
total_usd     = input * 0.30e-6 + output * 2.50e-6
```

| Video | Chi phí ước tính |
|---|---|
| 1 phút | ~$0.025 |
| 10 phút | ~$0.22 |
| 30 phút | ~$0.65 |
| 1 giờ | ~$1.30 |

Có thể tối ưu bằng **GCS native video input** (giảm ~40% chi phí so với frame extraction) — chỉ cần `GCS_BUCKET_NAME` hợp lệ.

---

## 7. Files đã thay đổi

| File | Loại | Mô tả |
|---|---|---|
| `backend/services/gemini_auto_labeler.py` | Refactor | Tách `get_genai_client`, `is_vertex_configured`, `_resolve_credentials`; default `location="global"` |
| `backend/services/ai_video_segmenter.py` | **Mới** | Service AutoCut (~370 dòng) |
| `backend/services/pipeline_orchestrator.py` | Tích hợp | Stage 2 dispatcher: `_ai_autocut_stage` ↔ `_classic_cut_stage` với fallback |
| `backend/api/gemini_api.py` | Tích hợp | Thêm router `POST /api/gemini/cut-and-create` |
| `backend/config.py` | Tích hợp | Thêm settings `AI_AUTOCUT_*`, mapping `user_settings.json` |
| `ui/pages/dashboard_page.py` | Tích hợp | Checkbox + spinbox + real-time status |
| `ui/pages/settings_page.py` | Tích hợp | Card "Vertex AI — Gemini Auto-Cut" đầy đủ |
| `ui/pages/processing_page.py` | Tích hợp | Stage "AI CUT" trong progress |
| `.env` | Cập nhật | `VERTEX_LOCATION=global`, `GOOGLE_APPLICATION_CREDENTIALS` trỏ đúng file key |
| `scripts/test_ai_autocut_integration.py` | **Mới** | 4 + 1 test tích hợp |
| `scripts/verify_integration_imports.py` | **Mới** | 10 import checks |
