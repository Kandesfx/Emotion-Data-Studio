# AI Auto-Cut — Design Document

Thiết kế và quy trình kỹ thuật cho chế độ **AI Auto-Cut**: Vertex AI (Gemini) tự động phát hiện đoạn cảm xúc mạnh trong video và cắt trực tiếp ra file `.mp4` — không qua PySceneDetect / SmartSegmenter cũ.

---

## 1. Mục tiêu

| # | Mục tiêu |
|---|---|
| M1 | Tận dụng khả năng hiểu ngữ cảnh đa phương thức của Gemini 2.5 Flash (visual + audio + speech cues) |
| M2 | Thay thế pipeline scene-detect + smart-segmenter cũ cho dataset harvest tự động |
| M3 | Giữ fallback an toàn — nếu Vertex AI lỗi phải KHÔNG làm hỏng job đang chạy |
| M4 | Reviewer vẫn có thể duyệt / sửa nhãn (stage Review không đổi) |
| M5 | Cost estimate rõ ràng + UI toggle 1-click |
| M6 | Tương thích schema DB hiện tại (`Clip` table, `per_model_scores.*`, `decision_by` enum) |

## 2. Không nằm trong scope

- Không thay đổi Stage 3 (face detect, audio extract, transcribe, ensemble).
- Không thay đổi Stage 6 (MMSA feature extraction).
- Không train model mới.
- Không thay đổi format MMSA `.pkl` exporter.

---

## 3. Kiến trúc tổng thể

```
┌────────────────────────────────────────────────────────────────────┐
│ USER (Dashboard / REST API)                                        │
│   - tick "AI Auto-Cut"   hoặc   POST /api/gemini/cut-and-create    │
└────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌────────────────────────────────────────────────────────────────────┐
│ PipelineOrchestrator.run_pipeline(video_id, db, progress_callback) │
│                                                                    │
│   Stage 1: Download (yt-dlp)                                       │
│   Stage 2: ┌──────────────────────────────────────────┐            │
│            │ if AI_AUTOCUT_ENABLED && vertex_ai_ready:│            │
│            │   AIVideoSegmenter.cut_video()           │            │
│            │ else:                                    │            │
│            │   SceneSplitter + SmartSegmenter         │            │
│            └──────────────────────────────────────────┘            │
│   Stage 3: Face + Audio + Whisper + Ensemble (giữ nguyên)          │
│   Stage 4: Quality Scorer + Auto-Decision (giữ nguyên)             │
│   Stage 5: Insert Clip records, status routing                     │
│   Stage 6: MMSA Feature extraction (chỉ approved clips)            │
└────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌────────────────────────────────────────────────────────────────────┐
│ AIVideoSegmenter.cut_video(video_path, video_id)                   │
│                                                                    │
│   1. ffprobe → duration                                            │
│   2. resolve gemini client (Vertex AI global)                      │
│   3. GCS upload? ─yes─► native video input                         │
│                │no                                                 │
│                ▼                                                   │
│   4. ffmpeg 1fps → base64 jpeg frames (batch 24)                  │
│   5. Gemini call → JSON list segments                              │
│   6. _parse_segments(): validate + clamp + padding                 │
│   7. _dedup_overlaps() (giữ intensity cao nhất)                   │
│   8. Với mỗi segment: FFmpeg stream-copy → clips/{uuid}.mp4       │
│   9. persist_clips() → bulk insert với status='needs_review'       │
└────────────────────────────────────────────────────────────────────┘
```

---

## 4. Hợp đồng (Contracts)

### 4.1. System prompt (`AUTOCUT_SYSTEM_PROMPT`)

Có cấu trúc rõ ràng, dùng `{intensity_threshold}` / `{min_duration}` / `{max_duration}` từ settings.

Đặc điểm:
- Bắt buộc JSON array, không kèm text khác.
- Mỗi segment liệt kê: `start_time`, `end_time`, `emotion`, `intensity`, `face_coverage`, `speaker_visible`, `has_transcript`, `subject`, `reasoning`.
- Ràng buộc cứng: intensity ≥ threshold, duration ∈ [min, max], không overlap, có mặt chính diện.

### 4.2. Validation (`_parse_segments`)

```
if duration < MIN   → drop
if duration > MAX   → clamp end_time
if end_time > video_duration → clamp về duration
if emotion not in 7 labels   → drop
if intensity < threshold     → drop
```

### 4.3. Dedup overlap (`_dedup_overlaps`)

Sắp xếp theo **intensity giảm dần**, duyệt từng segment, loại bỏ nếu overlap > 30% với segment đã chọn. Kết quả: giữ segment có emotion rõ nhất trên mỗi khoảng thời gian.

### 4.4. FFmpeg cutting (`_cut_with_ffmpeg`)

```
1. Thử stream copy (-c:v copy -c:a copy)  ← nhanh, giữ nguyên quality
2. Nếu fail (codec không tương thích) → re-encode H.264 + AAC
```

Padding: `start_time -= AI_AUTOCUT_PADDING_BEFORE` (mặc định 0.5s), `end_time += AI_AUTOCUT_PADDING_AFTER`. Clamp về [0, duration].

### 4.5. Persist (`persist_clips`)

Mỗi AutoCutSegment → 1 row `Clip`:

```python
Clip(
  id=f"{video_id}_ai_{idx}_{uuid8}",
  video_id, clip_index=idx,
  start_time, end_time, duration,
  clip_path=path,
  is_manual_segment=False,
  predicted_emotion, confidence=intensity,
  face_ratio=face_coverage,
  has_speech=has_transcript,
  status="needs_review",
  decision_by="gemini_autocut",
  pipeline_stage="ai_autocut_done",
  reviewer_notes="[Gemini AutoCut] subject=.. | reasoning=..",
  per_model_scores={
    "gemini_autocut": {emotion, intensity, face_coverage, subject, reasoning, has_transcript},
    "autocut_meta":   {source="vertex_ai_global", cost_usd, model},
  },
  all_scores={emotion: intensity},
)
```

**Idempotent**: nếu `clip_id` đã tồn tại → update emotion/confidence, không tạo row mới.

---

## 5. Fallback strategy

| Điều kiện | Stage 2 chạy | Ghi chú |
|---|---|---|
| `AI_AUTOCUT_ENABLED=true` + Vertex AI OK | `_ai_autocut_stage()` | AutoCut |
| `AI_AUTOCUT_ENABLED=true` + Vertex AI fail to get list | `_ai_autocut_stage()` | `try/except` rơi về `_classic_cut_stage()` |
| `AI_AUTOCUT_ENABLED=true` + Vertex AI OK nhưng 0 segments | `_ai_autocut_stage()` | 0 clip → `_classic_cut_stage()` |
| `AI_AUTOCUT_ENABLED=false` | `_classic_cut_stage()` | SceneSplitter + SmartSegmenter |
| Không có FFmpeg | Stage 2 fail | raise → toàn pipeline fail + UI error |

Lý do fallback 0 segments: tránh trường hợp AI không tìm được gì nhưng user vẫn cần dataset từ video đó (dùng clip thường).

---

## 6. UI/UX

### 6.1. Dashboard toggle

```
┌─────────────────────────────────────────────────────────────────┐
│ URL: [https://youtu.be/...]    [Xử lý]                          │
│ Tên phim: [...]                              [Chọn File]        │
│ ☑ 🤖 AI Auto-Cut (Vertex AI)  Intensity ≥ [0.55]  ✅ ready, ...  │
└─────────────────────────────────────────────────────────────────┘
```

- Checkbox **mặc định tắt** (zero-config khi Vertex AI chưa sẵn sàng).
- `Intensity` spinbox [0.30 — 1.00], step 0.05.
- Status label real-time (`refresh_ai_status()` sau 800ms).
- Nếu Vertex AI chưa sẵn sàng → checkbox bị disable.

### 6.2. Settings card

Đặt **giữa** Smart Segmentation card và Cloud card. Cấu hình chi tiết + button "Kiểm tra ngay".

### 6.3. Processing monitor

Stage widget mới:

```
[ai_autocut]  AI CUT   [░░░░░░░░░░] 45%   "Đã cắt 12/27: angry (0.87)"
```

---

## 7. Chi phí & quota

**Công thức ước tính** (constant-time, không cần gọi Gemini):

```python
input_tokens  = duration_sec * 7000   # ~7K tokens/frame @ 1fps
output_tokens = 1024
total_usd     = input_tokens * 0.3e-6 + output_tokens * 2.5e-6
```

| Duration | Cost estimate |
|---|---|
| 1 min | ~$0.025 |
| 10 min | ~$0.22 |
| 30 min | ~$0.65 |
| 1 hour | ~$1.30 |

**Tối ưu**: bật `GCS_BUCKET_NAME` → Gemini dùng native video input, giảm ~40% input tokens.

---

## 8. Testing

### 8.1. Test tích hợp (`scripts/test_ai_autocut_integration.py`)

| Test | Mục đích |
|---|---|
| 1. `vertex_config` | Credentials + project + location OK |
| 2. `text_gen` | Gemini trả lời text thường |
| 3. `multimodal` | Gemini mô tả ảnh (PNG → text) |
| 4. `ai_segmenter` | `AIVideoSegmenter.is_configured()` true |
| 5. `full_autocut` *(optional)* | End-to-end với video thật |

### 8.2. Verify imports (`scripts/verify_integration_imports.py`)

10 module-level assertions (không cần network):

```
✓ gemini_auto_labeler
✓ ai_video_segmenter
✓ pipeline_orchestrator (có _vertex_ai_ready, _ai_autocut_stage, _classic_cut_stage)
✓ gemini_api (7 routes, có /cut-and-create)
✓ config (AI_AUTOCUT, VERTEX_LOCATION)
✓ ui/pages/dashboard_page (có ai_autocut_chk, _refresh_ai_status)
✓ ui/pages/settings_page (có _build_vertex_ai_card)
✓ ui/pages/processing_page (có ai_autocut trong STAGES)
✓ ui/main_window
```

### 8.3. CI / Build-time

Nên tích hợp `verify_integration_imports.py` vào GitHub Actions pre-commit (chưa có workflow — cần tạo `tools/emotion-data-studio/.github/workflows/test.yml` nếu muốn).

---

## 9. Edge cases & lưu ý

| Case | Hành vi |
|---|---|
| Video 0s / ffprobe fail | raise `RuntimeError("Cannot determine duration")` |
| Gemini trả text không phải JSON | `_parse_segments()` fallback regex hoặc `[]` |
| Frame extraction fail (codec lạ) | raise, không retry — user cần re-encode video trước |
| FFmpeg cut fail trên 1 segment | log warning, skip segment đó, tiếp tục các segment khác |
| Gemini response bị truncate do MAX_TOKENS | parse partial JSON nếu được, không raise |
| `GEMINI_MONTHLY_BUDGET_USD` vượt | (TODO) thêm budget guard trước khi gọi |
| Concurrent jobs chạy song song | PipelineOrchestrator là **singleton** — chỉ 1 job tại 1 thời điểm (đã đảm bảo từ code cũ) |
| Service Account key hết hạn | `is_vertex_configured()` trả False → UI disable + fallback classic |
| User thay đổi settings giữa job | Settings override áp dụng ngay cho job tiếp theo (in-memory) |

---

## 10. Roadmap (chưa làm, ngoài scope hiện tại)

| Item | Priority | Mô tả |
|---|---|---|
| Budget guard | Medium | Refuse job nếu `cost_estimate > monthly_budget` |
| Re-verify stage | Medium | Sau khi cut, gọi Gemini verify lại từng clip (depth=2) |
| Native multimodal video | High | Hỗ trợ upload MP4 trực tiếp qua `file_data` thay vì frame base64 |
| Async/queue | Low | Cho batch nhiều video, dùng `asyncio.gather` |
| Multi-model ensemble | Low | Kết hợp Gemini + Claude / GPT-4V voting |
| Persistent cache | Low | Cache Gemini response theo video hash → không gọi lại cùng video |
