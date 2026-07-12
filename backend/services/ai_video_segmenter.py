"""
Emotion Data Studio — AI Video Segmenter
========================================
Orchestrator giup AI (Vertex AI / Gemini) tu dong quet video va cat truc tiep
ra cac doan co cam xuc manh. Thay the cho stage 2 (scene_split +
smart_segmenter) khi AI_AUTOCUT_ENABLED = true.

Quy trinh:
  1. Lay video duration (ffprobe).
  2. Estimate cost (token / USD).
  3. Goi Gemini (Vertex AI global) phan tich -> list segments
     (start_time, end_time, emotion, intensity, reasoning, ...).
  4. Validate + dedup + clamp theo min/max duration + padding.
  5. FFmpeg stream-copy tung segment -> file .mp4 trong DATA_DIR/clips.
  6. Bulk insert Clip records (status="needs_review", decision_by="gemini").
  7. Tra ve danh sach Clip metadata cho pipeline_orchestrator.

Fallback:
  - Neu Vertex AI chua cau hinh hoac loi -> raise RuntimeError
    de pipeline_orchestrator tu fallback ve scene_split + smart_segmenter.
"""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from sqlalchemy.orm import Session

logger = logging.getLogger("EDS-AISegmenter")


# ── Constants ─────────────────────────────────────────────

EMOTION_LABELS = ["happy", "sad", "angry", "fear", "surprise", "disgust", "neutral"]


# ── Sprint 1 — response_schema cho Gemini ────────────────────────────────
# Gemini 2.5 Flash support response_schema + response_mime_type="application/json".
# Giup Gemini tra JSON dung format 100%, khong can regex fallback.
# Xem chi tiet: docs/03_ai_autocut_optimization.md §2.5.
def _build_response_schema() -> dict[str, Any] | None:
    """Build JSON schema cho Gemini scan output.

    Tra ve None neu google-genai SDK chua import duoc (de caller fallback
    prompt-only)."""
    try:
        from google.genai import types  # type: ignore
    except Exception:
        return None

    return {
        "type": "ARRAY",
        "items": {
            "type": "OBJECT",
            "properties": {
                "start_time":      {"type": "NUMBER"},
                "end_time":        {"type": "NUMBER"},
                "emotion":         {"type": "STRING", "enum": EMOTION_LABELS},
                "intensity":       {"type": "NUMBER", "minimum": 0.0, "maximum": 1.0},
                "face_coverage":   {"type": "NUMBER", "minimum": 0.0, "maximum": 1.0},
                "frontal_ratio":   {"type": "NUMBER", "minimum": 0.0, "maximum": 1.0},
                "speaker_visible": {"type": "BOOLEAN"},
                "has_transcript":  {"type": "BOOLEAN"},
                "speaker_id":      {"type": "STRING"},
                "subject":         {"type": "STRING"},
                "reasoning":       {"type": "STRING"},
                "speech_quality":  {"type": "STRING", "enum": ["good", "fair", "poor", "none"]},
                "people_count":    {"type": "INTEGER", "minimum": 1, "maximum": 10},
            },
            "required": ["start_time", "end_time", "emotion", "intensity"],
        },
    }


def _call_gemini_with_json_enforced(
    client: Any,
    model: str,
    contents: list,
    *,
    system_instruction: str,
    temperature: float,
    max_output_tokens: int,
) -> Any:
    """Goi Gemini ep output JSON qua response_schema.

    Neu google-genai SDK khong support schema, fallback prompt-only.
    """
    schema = _build_response_schema()
    config: dict[str, Any] = {
        "temperature": temperature,
        "max_output_tokens": max_output_tokens,
        "system_instruction": system_instruction,
    }
    if schema is not None:
        config["response_mime_type"] = "application/json"
        config["response_schema"] = schema
    return client.models.generate_content(
        model=model,
        contents=contents,
        config=config,
    )


# Sprint 1 — chain-of-thought + 13 tieu chat chat luong + 3 vi du.
# Khong gò bó Gemini phai tra segment: neu khong dat chuan → [].
# Xem chi tiet: docs/03_ai_autocut_optimization.md §2.4 va §5.2.
AUTOCUT_SYSTEM_PROMPT = """Bạn là chuyên gia phân tích cảm xúc cho dataset MMSA/MulT tiếng Việt.

NHIỆM VỤ: Xem video và xác định các clip có thể dùng để HUẤN LUYỆN mô hình đa phương thức.
Mỗi clip phải đạt TIÊU CHUẨN NGHIÊM NGẶT — nếu không đạt thì KHÔNG trả về
(tốt hơn bỏ sót còn hơn trả clip xấu).

═══════════════════════════════════════════════════
TIÊU CHUẨN BẮT BUỘC (mỗi segment phải thoả mãn TẤT CẢ)
═══════════════════════════════════════════════════

[CHẤT LƯỢNG HÌNH ẢNH]
1. Mặt người CHÍNH DIỆN (yaw < {yaw_deg}°) xuất hiện ≥ {face_coverage:.0%} số frame trong clip.
2. Khuôn mặt có kích thước ≥ {face_size_ratio:.0%} diện tích frame (đủ để facial AU rõ).
3. KHÔNG che mặt bởi tay/vật, KHÔNG blur nặng, KHÔNG ánh sáng ngược silhouette.
4. CHỈ 1 NGƯỜI nói/biểu lộ trong clip (clip nhiều người → bỏ).

[CHẤT LƯỢNG ÂM THANH]
5. Có lời nói rõ ràng ≥ 1 đoạn speech (RMS đủ, không bị nhiễu nền).
6. Nếu nhạc nền to hơn lời thoại → giảm intensity hoặc bỏ.
7. SNR cảm nhận tốt (nếu nghe ù/hiss rõ → bỏ).

[CẢM XÚC]
8. Cường độ cảm xúc ≥ {intensity_threshold:.2f} (thang 0.0-1.0).
9. Emotion ỔN ĐỊNH trong suốt clip — KHÔNG chuyển sang emotion khác.
10. Biểu cảm PHẢI PHÙ HỢP với giọng nói và nội dung lời thoại.
    Nếu cười nhưng giọng buồn → không chọn, hoặc intensity < 0.6.

[NGÔN NGỮ & NỘI DUNG]
11. Lời thoại phải là tiếng Việt rõ ràng ≥ {min_words} từ (nếu có speech).
12. Tránh clip có quảng cáo / intro / outro / credit / nhạc chuyển cảnh.
13. Ưu tiên clip có SỰ KIỆN cảm xúc rõ: cãi vã, khóc, bất ngờ, hùng biện, xúc động.

═══════════════════════════════════════════════════
QUY TRÌNH PHÂN TÍCH (CHAIN-OF-THOUGHT — BẮT BUỘC)
═══════════════════════════════════════════════════

Trước khi đưa JSON cuối, hãy suy luận theo từng bước:

Bước 1: LIỆT KÊ 3-5 scene "ứng viên" (mỗi scene 1 dòng: time-range, ai, nói gì).
Bước 2: VỚI MỖI ỨNG VIÊN, kiểm tra 13 tiêu chí ở trên (✓/✗ cho từng tiêu chí).
Bước 3: CHỈ GIỮ ứng viên thoả mãn TẤT CẢ tiêu chí bắt buộc.
Bước 4: Xác định start/end time chính xác (±0.5s).
Bước 5: Output JSON array cuối (KHÔNG kèm text khác ngoài JSON).

═══════════════════════════════════════════════════
NHÃN CẢM XÚC (chỉ dùng đúng các nhãn này, tiếng Anh)
═══════════════════════════════════════════════════

- happy: vui, cười, phấn khởi, hân hoan, hài lòng, yêu thương
- sad: buồn, khóc, tuyệt vọng, thất vọng, cô đơn, tủi thân
- angry: giận, tức, bực, gào, cãi, phẫn nộ
- fear: sợ, hoảng, lo lắng, bất an, run rẩy
- surprise: bất ngờ, ngạc nhiên, sốc, choáng
- disgust: ghê tởm, chán ghét, khinh miệt
- neutral: KHÔNG dùng trừ khi thật sự cần thiết (background scene, không có emotion rõ)

═══════════════════════════════════════════════════
VÍ DỤ MẪU (tham khảo format và tiêu chí)
═══════════════════════════════════════════════════

Ví dụ 1 (clip 8s — ĐẠT):
Input: 1 người phụ nữ, mắt đỏ, tay run, nói giọng nghẹn.
Output:
[
  {{
    "start_time": 45.20, "end_time": 53.00, "emotion": "sad",
    "intensity": 0.92, "face_coverage": 0.88, "frontal_ratio": 0.85,
    "speaker_visible": true, "has_transcript": true,
    "subject": "người phụ nữ trung niên, tóc dài, áo trắng",
    "reasoning": "Mắt đỏ hoe, nước mắt chảy, giọng nghẹn, nói 'Con ơi...' — buồn rõ ràng xuyên suốt clip, không flip.",
    "speech_quality": "good", "people_count": 1
  }}
]

Ví dụ 2 (clip 5s — BỎ vì 2 người):
Input: 2 người đang cãi nhau, cả 2 đều biểu cảm angry.
Output: []   (KHÔNG trả clip này vì vi phạm tiêu chí 4 — nhiều người)

Ví dụ 3 (clip 6s — BỎ vì mặt nghiêng):
Input: 1 người nhưng mặt xoay 60°, chỉ thấy gáy.
Output: []   (frontal_ratio quá thấp, không đạt 70% mặt chính diện)

═══════════════════════════════════════════════════
OUTPUT FORMAT (BẮT BUỘC — chỉ trả JSON array, KHÔNG kèm text khác)
═══════════════════════════════════════════════════

[
  {{
    "start_time": 12.50,
    "end_time": 24.00,
    "emotion": "angry",
    "intensity": 0.87,
    "face_coverage": 0.85,
    "frontal_ratio": 0.80,
    "speaker_visible": true,
    "has_transcript": true,
    "speaker_id": "speaker_0",
    "subject": "người đàn ông trung niên, áo sơ mi đen, tóc ngắn",
    "reasoning": "Giọng nói cao và run, mặt căng, lông mày nhíu, cử chỉ tay đập bàn. Lời thoại: 'Tôi không chịu được nữa!'. Biểu cảm angry rõ ràng xuyên suốt clip.",
    "speech_quality": "good",
    "people_count": 1
  }}
]

LƯU Ý CUỐI:
- Nếu KHÔNG tìm được segment nào đạt chuẩn → trả về [] (không ép buộc).
- start/end time làm tròn 2 chữ số thập phân.
- Mỗi video 30 phút thường cho 3-8 segments đạt chuẩn — đừng cố trả nhiều hơn.
- Reasoning phải MÔ TẢ CỤ THỂ: biểu cảm gì, giọng ra sao, lời nói gì.
"""


@dataclass
class AutoCutSegment:
    """Mot doan AI cat ra. Se duoc cat thanh file rieng."""
    start_time: float
    end_time: float
    emotion: str
    intensity: float
    face_coverage: float = 0.0
    speaker_visible: bool = True
    has_transcript: bool = False
    subject: str = ""
    reasoning: str = ""
    clip_path: str = ""
    clip_id: str = ""
    # Sprint 2 — Verify pass metadata (them sau Stage 4)
    verify_verdict: str = ""           # confirmed | wrong_emotion | unstable | low_quality | stats_mismatch | ""
    verify_status: str = "not_run"     # not_run | passed | rejected
    verify_reasoning: str = ""
    rejected_by_verify: bool = False
    reject_reason: str = ""


@dataclass
class AutoCutResult:
    """Ket qua cua toan bo qua trinh AI auto-cut."""
    video_id: str
    video_path: str
    video_duration: float
    total_segments: int
    total_cost_usd: float
    clips: list[AutoCutSegment] = field(default_factory=list)
    raw_gemini_response: dict[str, Any] = field(default_factory=dict)
    source: str = "ai_autocut"
    error: str | None = None
    # Sprint 2 — toan bo run state de UI/debug nhin thay
    last_rejected: list[dict[str, Any]] = field(default_factory=list)
    verify_summary: dict[str, Any] = field(default_factory=dict)
    stage1_candidates: int = 0      # so segments Gemini de xuat (truoc hard filter)
    stage2_passed: int = 0          # segments vuot hard filter
    stage3_cut: int = 0             # segments da cat bang FFmpeg
    stage4_verified: int = 0        # segments qua verify (passed + rejected)
    cost_breakdown: dict[str, Any] = field(default_factory=dict)  # Sprint 3

    def to_dict(self) -> dict[str, Any]:
        return {
            "video_id": self.video_id,
            "video_path": self.video_path,
            "video_duration": self.video_duration,
            "total_segments": self.total_segments,
            "total_cost_usd": self.total_cost_usd,
            "clips": [c.__dict__ for c in self.clips],
            "source": self.source,
            "error": self.error,
            "last_rejected": list(self.last_rejected),
            "verify_summary": dict(self.verify_summary),
            "stage1_candidates": self.stage1_candidates,
            "stage2_passed": self.stage2_passed,
            "stage3_cut": self.stage3_cut,
            "stage4_verified": self.stage4_verified,
            "cost_breakdown": dict(self.cost_breakdown),
        }


# ── Sprint 2 — Verify pass (Stage 4) ─────────────────────────────────────
# Sau khi Stage 3 FFmpeg cat xong, tung clip duoc gui lai cho Gemini de
# xac minh emotion + face/audio quality. Neu 2 pass khop → intensity boost,
# neu khong khop → emotion_lock unlocked, danh "needs_review".
# Xem chi tiet: docs/03_ai_autocut_optimization.md §2.2.

VERIFY_SYSTEM_PROMPT = """Bạn đang verify 1 clip đã cắt ra từ video gốc. KHÔNG đoán emotion mới,
mà XÁC MINH emotion mà Stage 2 (AI Auto-Cut) đã gán.

═══════════════════════════════════════════════════
NHIỆM VỤ VERIFY (4 bước — bắt buộc)
═══════════════════════════════════════════════════

Bước 1: Xem clip — emotion Stage 2 đã gán có khớp thực tế không?
   - KHỚP & intensity đúng        → confidence cao (0.85-1.0)
   - KHÔNG KHỚP                  → đề xuất emotion mới + intensity < 0.7
   - KHÔNG RÕ RÀNG                → intensity < 0.6, đánh dấu "low_quality"

Bước 2: Kiểm tra EMOTION ỔN ĐỊNH trong clip
   - Có chuyển emotion giữa chừng không? (vd: buồn → vui)
   - Nếu CÓ → flag "unstable", intensity < 0.5

Bước 3: Kiểm tra FACE và AUDIO chất lượng THỰC TẾ
   - Face có rõ, frontal, đủ size không? (so với stats đầu vào)
   - Audio có rõ, khớp emotion không?
   - Nếu stats nói rõ mà thực tế không → flag "stats_mismatch"

Bước 4: ĐƯA RA VERDICT
   - "confirmed"       : emotion + intensity đúng, dùng làm ground truth
   - "wrong_emotion"   : emotion khác verdict Stage 2 (low confidence)
   - "unstable"        : emotion flip trong clip
   - "low_quality"     : clip xấu, không nên dùng training
   - "stats_mismatch"  : stats Stage 2 không khớp thực tế

═══════════════════════════════════════════════════
RÀNG BUỘC KHI ĐÁNH GIÁ
═══════════════════════════════════════════════════

- Mặt người CHÍNH DIỆN (yaw < 30°) ≥ 70% số frame? Nếu KHÔNG → intensity giảm 30%.
- Có ≥ 1 đoạn speech rõ ràng? Nếu KHÔNG → giảm intensity hoặc flag "no_speech".
- Nếu lời nói và biểu cảm mâu thuẫn (cười nhưng giọng buồn):
  → chọn emotion mạnh hơn, ghi reasoning giải thích.

═══════════════════════════════════════════════════
OUTPUT (chỉ JSON object, KHÔNG kèm text khác)
═══════════════════════════════════════════════════

{
  "verdict": "confirmed" | "wrong_emotion" | "unstable" | "low_quality" | "stats_mismatch",
  "emotion": "angry",
  "intensity": 0.85,
  "face_coverage": 0.82,
  "frontal_ratio": 0.78,
  "speech_quality": "good",
  "stable": true,
  "speech_emotion_match": true,
  "reasoning": "Xác nhận angry rõ, giọng run + mặt căng + lông mày nhíu. Ổn định xuyên suốt clip, không flip."
}
"""


def _build_verify_response_schema() -> dict[str, Any] | None:
    """Schema riêng cho Verify pass (single object, không phải array)."""
    try:
        import google.genai as genai  # type: ignore  # noqa: F401
    except Exception:
        return None
    return {
        "type": "OBJECT",
        "properties": {
            "verdict": {
                "type": "STRING",
                "enum": [
                    "confirmed", "wrong_emotion", "unstable",
                    "low_quality", "stats_mismatch",
                ],
            },
            "emotion":           {"type": "STRING", "enum": EMOTION_LABELS},
            "intensity":         {"type": "NUMBER", "minimum": 0.0, "maximum": 1.0},
            "face_coverage":     {"type": "NUMBER", "minimum": 0.0, "maximum": 1.0},
            "frontal_ratio":     {"type": "NUMBER", "minimum": 0.0, "maximum": 1.0},
            "speech_quality":    {"type": "STRING", "enum": ["good", "fair", "poor", "none"]},
            "stable":            {"type": "BOOLEAN"},
            "speech_emotion_match": {"type": "BOOLEAN"},
            "reasoning":         {"type": "STRING"},
        },
        "required": ["verdict", "emotion", "intensity"],
    }


# ── Service ───────────────────────────────────────────────

class AIVideoSegmenter:
    """
    Su dung Gemini (Vertex AI global) de quet video va cat truc tiep.
    """

    def __init__(self, labeler: Any | None = None):
        self.labeler = labeler

    # ── Helpers ─────────────────────────────────────────

    @staticmethod
    def _get_ffmpeg_path() -> str:
        from backend.config import settings
        return settings.FFMPEG_PATH

    def _get_duration(self, video_path: Path) -> float:
        try:
            result = subprocess.run(
                [
                    "ffprobe", "-v", "error", "-show_entries", "format=duration",
                    "-of", "default=noprint_wrappers=1:nokey=1", str(video_path),
                ],
                capture_output=True, text=True, timeout=30,
            )
            return float(result.stdout.strip())
        except Exception:
            return 0.0

    def _ensure_labeler(self):
        if self.labeler is None:
            from backend.services.gemini_auto_labeler import GeminiAutoLabeler
            self.labeler = GeminiAutoLabeler()
        return self.labeler

    # ── Vertex AI call ──────────────────────────────────

    def _build_prompt(self, video_duration: float) -> tuple[str, str]:
        """Build system + user prompt.

        Cac tham so duoc fill tu settings de prompt luon phan anh config hien tai.
        """
        from backend.config import settings
        sys_prompt = AUTOCUT_SYSTEM_PROMPT.format(
            intensity_threshold=settings.AI_AUTOCUT_INTENSITY_THRESHOLD,
            min_duration=settings.AI_AUTOCUT_MIN_DURATION,
            max_duration=settings.AI_AUTOCUT_MAX_DURATION,
            # Sprint 1 — them cac nguong hard filter vao prompt
            face_coverage=settings.HARD_FILTER_MIN_FACE_COVERAGE,
            face_size_ratio=settings.HARD_FILTER_MIN_FACE_SIZE_RATIO,
            yaw_deg=settings.HARD_FILTER_MAX_YAW_DEG,
            min_words=settings.HARD_FILTER_MIN_WORD_COUNT,
        )
        user_prompt = (
            f"Video dai {video_duration:.1f} giay. "
            "Hay phan tich va tra ve JSON array cac segment theo dung format quy dinh. "
            "Neu khong co doan nao dat yeu cau, tra ve []."
        )
        return sys_prompt, user_prompt

    def _call_gemini_segments(
        self,
        video_path: Path,
        video_duration: float,
        progress_callback: Optional[Callable[[str, int, int, str], None]] = None,
    ) -> list[dict[str, Any]]:
        """
        Goi Gemini de lay segments. Dung native video input neu co GCS,
        nguoc lai fallback frame extraction.

        Sprint 3 — Cache layer:
        Cache JSON response theo key (video_path, stage="scan", prompt_hash).
        Neu cache hit → khong goi Gemini, tra lai luon.
        """
        from backend.config import settings
        from backend.services import gemini_cache

        # ── Cache check ─────────────────────────────────
        labeler = self._ensure_labeler()
        configured, msg = labeler.is_configured()
        if not configured:
            raise RuntimeError(f"Gemini chua cau hinh: {msg}")

        # Build prompt truoc de tinh hash
        sys_prompt, user_prompt = self._build_prompt(video_duration)
        cache_key = gemini_cache.make_key(
            str(video_path),
            stage="scan",
            prompt=sys_prompt + user_prompt,
            params={"duration": video_duration, "model": labeler.model},
        )
        cached = gemini_cache.get(cache_key)
        if cached is not None:
            if progress_callback:
                progress_callback("ai_autocut", 50, 100,
                                  f"⚡ Cache hit: {len(cached)} segments")
            return list(cached)

        # Neu co AGENT_RUNTIME_URL -> uu tien
        if getattr(labeler, "agent_url", None) and getattr(labeler, "agent_api_key", None):
            if progress_callback:
                progress_callback("ai_autocut", 10, 100, "Calling Agent Runtime (Cloud Run)...")
            result = labeler._call_agent_runtime(user_prompt, None)
            text = result.get("text", "") if isinstance(result, dict) else str(result)
            segments = self._parse_segments(text, video_duration)
            gemini_cache.put(cache_key, segments)
            return segments

        # Thu native GCS input truoc
        gcs_uri = self._try_upload_to_gcs(video_path)
        if gcs_uri:
            if progress_callback:
                progress_callback(
                    "ai_autocut", 15, 100,
                    f"Uploaded to GCS, asking Gemini via native video..."
                )
            segments = self._call_gemini_native_video(
                labeler, gcs_uri, video_duration, sys_prompt, user_prompt
            )
            gemini_cache.put(cache_key, segments)
            return segments

        # Fallback: frame extraction
        if progress_callback:
            progress_callback("ai_autocut", 15, 100, "Extracting frames (1 fps)...")
        frames = self._extract_frames_b64(video_path, max_fps=1.0, max_frames=400)
        if not frames:
            raise RuntimeError("Khong the trich xuat frames tu video.")
        if progress_callback:
            progress_callback("ai_autocut", 30, 100, f"Sending {len(frames)} frames to Gemini...")

        segments = self._call_gemini_with_frames(
            labeler, frames, video_duration, sys_prompt, user_prompt
        )
        gemini_cache.put(cache_key, segments)
        return segments

    def _call_gemini_native_video(self, labeler, gcs_uri, duration, sys_prompt, user_prompt):
        try:
            client = labeler._resolve_client()
            response = _call_gemini_with_json_enforced(
                client,
                labeler.model,
                contents=[{
                    "role": "user",
                    "parts": [
                        {"file_data": {"mime_type": "video/mp4", "file_uri": gcs_uri}},
                        {"text": user_prompt},
                    ],
                }],
                system_instruction=sys_prompt,
                temperature=labeler.temperature,
                max_output_tokens=labeler.max_output_tokens,
            )
            text = labeler._get_text(response)
            return self._parse_segments(text, duration)
        except Exception as exc:
            logger.warning(f"Native GCS call failed, fallback frames: {exc}")
            raise

    def _call_gemini_with_frames(self, labeler, frames_b64, duration, sys_prompt, user_prompt):
        client = labeler._resolve_client()
        batch_size = 24
        all_segments: list[dict] = []
        total_batches = (len(frames_b64) + batch_size - 1) // batch_size
        for i in range(0, len(frames_b64), batch_size):
            batch = frames_b64[i:i + batch_size]
            batch_num = i // batch_size + 1
            time_offset = i
            try:
                image_parts = [
                    {"inline_data": {"mime_type": "image/jpeg", "data": f}} for f in batch
                ]
                response = _call_gemini_with_json_enforced(
                    labeler._resolve_client(),
                    labeler.model,
                    contents=[{"role": "user", "parts": image_parts + [{"text": user_prompt}]}],
                    system_instruction=sys_prompt,
                    temperature=labeler.temperature,
                    max_output_tokens=labeler.max_output_tokens,
                )
                text = labeler._get_text(response)
                if not text:
                    continue
                batch_segs = self._parse_segments(text, duration)
                for seg in batch_segs:
                    seg["start_time"] = float(seg.get("start_time", 0)) + time_offset
                    seg["end_time"] = float(seg.get("end_time", 0)) + time_offset
                all_segments.extend(batch_segs)
                logger.info(f"Autocut batch {batch_num}/{total_batches}: {len(batch_segs)} segs")
            except Exception as exc:
                logger.warning(f"Autocut batch {batch_num} failed: {exc}")
                continue
        return all_segments

    # ── Frame extraction + GCS upload ───────────────────

    def _extract_frames_b64(self, video_path: Path, max_fps: float = 1.0, max_frames: int = 400) -> list[str]:
        import base64
        frames: list[str] = []
        tmp_dir = video_path.parent / f".tmp_autocut_{os.getpid()}"
        tmp_dir.mkdir(exist_ok=True)
        try:
            result = subprocess.run(
                [
                    self._get_ffmpeg_path(), "-y", "-i", str(video_path),
                    "-vf", f"fps={max_fps}",
                    "-q:v", "3",
                    "-frames:v", str(max_frames),
                    str(tmp_dir / "frame_%04d.jpg"),
                ],
                capture_output=True, text=True, timeout=600,
            )
            if result.returncode != 0:
                logger.warning(f"ffmpeg frame extract failed: {result.stderr[:200]}")
                return []
            for f in sorted(tmp_dir.iterdir()):
                if f.suffix.lower() in (".jpg", ".jpeg", ".png"):
                    frames.append(base64.b64encode(f.read_bytes()).decode("utf-8"))
        finally:
            try:
                for f in tmp_dir.iterdir():
                    f.unlink()
                tmp_dir.rmdir()
            except Exception:
                pass
        return frames

    def _try_upload_to_gcs(self, video_path: Path) -> str | None:
        """Upload len GCS neu co gsutil + bucket name."""
        bucket = os.getenv("GCS_BUCKET_NAME")
        if not bucket:
            return None
        if not video_path.exists():
            return None
        filename = f"autocut/{video_path.stem}_{os.getpid()}{video_path.suffix}"
        gcs_uri = f"gs://{bucket}/{filename}"
        try:
            result = subprocess.run(
                ["gsutil", "cp", str(video_path), gcs_uri],
                capture_output=True, text=True, timeout=600,
            )
            if result.returncode == 0:
                logger.info(f"Autocut: uploaded to {gcs_uri}")
                return gcs_uri
            logger.warning(f"gsutil upload failed: {result.stderr[:200]}")
        except FileNotFoundError:
            logger.warning("gsutil not found, fallback to frame extraction")
        except Exception as exc:
            logger.warning(f"GCS upload error: {exc}")
        return None

    # ── Response parsing ────────────────────────────────

    def _parse_segments(self, text: str, video_duration: float) -> list[dict[str, Any]]:
        from backend.config import settings
        text = (text or "").strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text)
        data: Any = None
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            # Sprint 1: response_schema da ep Gemini tra JSON hop le.
            # Neu van khong parse duoc → log warning, return [].
            # KHONG fallback regex nua (de xoa `_extract_fallback`).
            logger.warning(
                f"[AISegmenter] Gemini tra JSON khong hop le, bo qua. "
                f"Text dau vao (first 200 chars): {text[:200]!r}. "
                f"ParseError: {exc}"
            )
            return []

        if isinstance(data, dict):
            for key in ("segments", "results", "clips", "highlights"):
                if key in data and isinstance(data[key], list):
                    data = data[key]
                    break
            else:
                data = [data]
        if not isinstance(data, list):
            return []

        valid: list[dict[str, Any]] = []
        for seg in data:
            if not isinstance(seg, dict):
                continue
            try:
                start = float(seg.get("start_time", 0))
                end = float(seg.get("end_time", 0))
                emotion = str(seg.get("emotion", "")).lower().strip()
                intensity = float(seg.get("intensity", 0))
            except (TypeError, ValueError):
                continue
            if start < 0 or end <= start:
                continue
            if emotion not in EMOTION_LABELS:
                continue
            if intensity < settings.AI_AUTOCUT_INTENSITY_THRESHOLD:
                continue
            dur = end - start
            if dur < settings.AI_AUTOCUT_MIN_DURATION:
                continue
            if dur > settings.AI_AUTOCUT_MAX_DURATION:
                end = start + settings.AI_AUTOCUT_MAX_DURATION
            if end > video_duration:
                end = video_duration
            if (end - start) < settings.AI_AUTOCUT_MIN_DURATION:
                continue
            valid.append({
                "start_time": round(start, 2),
                "end_time": round(end, 2),
                "emotion": emotion,
                "intensity": round(min(1.0, intensity), 3),
                "face_coverage": round(float(seg.get("face_coverage", 0)), 3),
                "speaker_visible": bool(seg.get("speaker_visible", True)),
                "has_transcript": bool(seg.get("has_transcript", False)),
                "subject": str(seg.get("subject", ""))[:200],
                "reasoning": str(seg.get("reasoning", ""))[:500],
            })
        valid.sort(key=lambda s: s["start_time"])
        return self._dedup_overlaps(valid)

    @staticmethod
    def _dedup_overlaps(segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Giu doan co intensity cao hon khi overlap, loai bo doan yeu hon."""
        if not segments:
            return []
        sorted_segs = sorted(segments, key=lambda s: (-s["intensity"], s["start_time"]))
        used: list[dict[str, Any]] = []
        for seg in sorted_segs:
            s, e = seg["start_time"], seg["end_time"]
            conflict = False
            for u in used:
                us, ue = u["start_time"], u["end_time"]
                overlap = max(0, min(e, ue) - max(s, us))
                if overlap > 0 and (overlap / max(1e-6, min(e - s, ue - us))) > 0.3:
                    conflict = True
                    break
            if not conflict:
                used.append(seg)
        return sorted(used, key=lambda s: s["start_time"])

    # ── Sprint 1 — Hard filter (loại segment không đạt chuẩn training) ─────

    # Schema chuẩn hoá segment sau khi parse từ Gemini.
    # Field `reject_reason` được thêm nếu segment bị hard filter loại.
    # Field `quality_score` 0-1 tổng hợp các tiêu chí đạt được.
    def _hard_filter_clip_quality(
        self,
        segments: list[dict[str, Any]],
        *,
        audio_snr_db: float | None = None,
        transcript_word_count: int | None = None,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Lọc segments theo 11 tiêu chí trong HARD_FILTER_CONFIG.

        Args:
            segments: list segments thô từ Gemini (đã parse).
            audio_snr_db: SNR trung bình toàn video (optional, dùng cho filter).
            transcript_word_count: số từ trong transcript toàn video (optional).

        Returns:
            (passed, rejected) — passed là segments đạt chuẩn, rejected kèm lý do.
        """
        from backend.config import settings

        cfg = {
            "min_face_coverage":      settings.HARD_FILTER_MIN_FACE_COVERAGE,
            "min_intensity":          settings.HARD_FILTER_MIN_INTENSITY,
            "min_frontal_ratio":      settings.HARD_FILTER_MIN_FRONTAL_RATIO,
            "max_yaw_deg":            settings.HARD_FILTER_MAX_YAW_DEG,
            "min_face_size_ratio":    settings.HARD_FILTER_MIN_FACE_SIZE_RATIO,
            "min_snr_db":             settings.HARD_FILTER_MIN_SNR_DB,
            "min_speech_segments":    settings.HARD_FILTER_MIN_SPEECH_SEGMENTS,
            "min_duration_sec":       settings.HARD_FILTER_MIN_DURATION_SEC,
            "max_duration_sec":       settings.HARD_FILTER_MAX_DURATION_SEC,
            "max_people_in_clip":     settings.HARD_FILTER_MAX_PEOPLE_IN_CLIP,
            "min_word_count":         settings.HARD_FILTER_MIN_WORD_COUNT,
        }

        passed: list[dict[str, Any]] = []
        rejected: list[dict[str, Any]] = []

        for seg in segments:
            reasons: list[str] = []
            dur = float(seg.get("end_time", 0)) - float(seg.get("start_time", 0))

            # 1. duration bounds
            if dur < cfg["min_duration_sec"]:
                reasons.append(f"duration_too_short({dur:.1f}s<{cfg['min_duration_sec']}s)")
            if dur > cfg["max_duration_sec"]:
                reasons.append(f"duration_too_long({dur:.1f}s>{cfg['max_duration_sec']}s)")

            # 2. intensity
            intensity = float(seg.get("intensity", 0))
            if intensity < cfg["min_intensity"]:
                reasons.append(f"intensity_low({intensity:.2f}<{cfg['min_intensity']:.2f})")

            # 3. face coverage (Gemini tu danh gia)
            face_cov = float(seg.get("face_coverage", 0))
            if face_cov < cfg["min_face_coverage"]:
                reasons.append(
                    f"face_coverage_low({face_cov:.2f}<{cfg['min_face_coverage']:.2f})"
                )

            # 4. frontal ratio (chi check neu Gemini co tra ve field)
            # Neu khong co (default 0.0) → khong check, vi nhieu video Gemini
            # khong estimate duoc frontal_ratio tu frames → tranh loai nham.
            frontal = float(seg.get("frontal_ratio", 0.0))
            has_frontal_field = "frontal_ratio" in seg and frontal > 0
            if has_frontal_field and frontal < cfg["min_frontal_ratio"]:
                reasons.append(
                    f"frontal_ratio_low({frontal:.2f}<{cfg['min_frontal_ratio']:.2f})"
                )

            # 5. people count (chi check neu Gemini co tra ve field)
            # Neu khong co → default 1, giu nguyen.
            has_people_field = "people_count" in seg
            people = int(seg.get("people_count", 1)) if has_people_field else 1
            if has_people_field and people > cfg["max_people_in_clip"]:
                reasons.append(f"too_many_people({people}>{cfg['max_people_in_clip']})")

            # 6. speech_quality (chi check khi Gemini tu danh gia = "none")
            # Neu khong co field → khong check (co the Gemini khong danh gia duoc).
            speech_q = str(seg.get("speech_quality", "")).lower()
            if "speech_quality" in seg and cfg["min_speech_segments"] >= 1 and speech_q == "none":
                reasons.append("no_speech")

            # 7. transcript word count (global, optional)
            if (
                transcript_word_count is not None
                and cfg["min_word_count"] >= 1
                and transcript_word_count < cfg["min_word_count"]
                and seg.get("has_transcript", False)
            ):
                reasons.append(
                    f"transcript_too_short({transcript_word_count}<{cfg['min_word_count']})"
                )

            if reasons:
                rejected.append({**seg, "reject_reason": ",".join(reasons)})
                logger.info(
                    f"[HardFilter] Bỏ segment [{seg.get('start_time'):.1f}-"
                    f"{seg.get('end_time'):.1f}] emotion={seg.get('emotion')} "
                    f"intensity={intensity:.2f}: {','.join(reasons)}"
                )
            else:
                # Tinh quality_score tong hop (0-1)
                # Trong luong: face_coverage 0.4, intensity 0.3, frontal 0.3
                quality = (
                    0.4 * min(1.0, face_cov / cfg["min_face_coverage"])
                    + 0.3 * min(1.0, intensity / cfg["min_intensity"])
                    + 0.3 * min(1.0, frontal / cfg["min_frontal_ratio"])
                )
                passed.append({**seg, "quality_score": round(min(1.0, quality), 3)})

        logger.info(
            f"[HardFilter] {len(passed)}/{len(segments)} segments đạt chuẩn, "
            f"{len(rejected)} bị loại."
        )
        return passed, rejected

    # ── FFmpeg cutting ──────────────────────────────────

    def _cut_with_ffmpeg(
        self,
        ffmpeg_path: str,
        video_path: Path,
        start: float,
        duration: float,
        output: Path,
    ) -> None:
        startupinfo = None
        if os.name == "nt":
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW

        copy_cmd = [
            ffmpeg_path, "-y", "-ss", str(start), "-t", str(duration),
            "-i", str(video_path),
            "-c:v", "copy", "-c:a", "copy",
            "-avoid_negative_ts", "1", str(output.resolve()),
        ]
        try:
            subprocess.run(
                copy_cmd,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                check=True, startupinfo=startupinfo, timeout=120,
            )
        except subprocess.CalledProcessError:
            reencode_cmd = [
                ffmpeg_path, "-y", "-ss", str(start), "-t", str(duration),
                "-i", str(video_path),
                "-c:v", "libx264", "-preset", "veryfast", "-c:a", "aac",
                str(output.resolve()),
            ]
            try:
                subprocess.run(
                    reencode_cmd,
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                    check=True, startupinfo=startupinfo, timeout=300,
                )
            except subprocess.CalledProcessError as exc:
                stderr = exc.stderr.decode("utf-8", errors="replace") if exc.stderr else ""
                raise RuntimeError(
                    f"FFmpeg khong the cat doan [{start:.1f}-{start + duration:.1f}]: "
                    f"{stderr[-500:]}"
                ) from exc

    # ── Public API ──────────────────────────────────────

    def is_configured(self) -> tuple[bool, str]:
        from backend.services.gemini_auto_labeler import is_vertex_configured
        return is_vertex_configured()

    def cut_video(
        self,
        video_path: str | Path,
        video_id: str,
        progress_callback: Optional[Callable[[str, int, int, str], None]] = None,
        cancel_check: Optional[Callable[[], bool]] = None,
    ) -> AutoCutResult:
        """
        Quet video bang Gemini va cat truc tiep ra cac clip.

        Args:
            video_path: duong dan file video
            video_id: id cua Video record (de dat ten clip)
            progress_callback: callback(stage, current, total, message)
            cancel_check: callable() -> True neu user huy

        Returns:
            AutoCutResult chua danh sach clip da cat + metadata.

        Raises:
            RuntimeError: neu Vertex AI chua cau hinh hoac loi.
        """
        from backend.config import settings

        path = Path(video_path)
        if not path.exists():
            raise FileNotFoundError(f"Video khong ton tai: {path}")

        if progress_callback:
            progress_callback("ai_autocut", 0, 100, "Đang chuẩn bị...")

        duration = self._get_duration(path)
        if duration <= 0:
            raise RuntimeError("Khong xac dinh duoc duration video (ffprobe).")

        # 1. Call Gemini
        if progress_callback:
            progress_callback("ai_autocut", 5, 100, f"Video {duration:.0f}s — gọi Vertex AI...")
        try:
            raw_segments = self._call_gemini_segments(path, duration, progress_callback)
        except Exception as exc:
            logger.error(f"AI autocut Vertex AI call failed: {exc}")
            raise RuntimeError(f"Vertex AI failed: {exc}") from exc

        if progress_callback:
            progress_callback(
                "ai_autocut", 60, 100,
                f"Gemini trả về {len(raw_segments)} đoạn hợp lệ."
            )

        # 2. Apply padding
        for seg in raw_segments:
            seg["start_time"] = max(0.0, seg["start_time"] - settings.AI_AUTOCUT_PADDING_BEFORE)
            seg["end_time"] = min(duration, seg["end_time"] + settings.AI_AUTOCUT_PADDING_AFTER)

        # 3. Cap so luong
        raw_segments = raw_segments[: settings.AI_AUTOCUT_MAX_SEGMENTS]

        # 3.5. Sprint 1 — Hard filter theo 11 tiêu chí (xem _hard_filter_clip_quality)
        # Loc ngay sau Gemini, truoc FFmpeg cut → tiet kiem disk + thoi gian cat.
        passed_segs, rejected_segs = self._hard_filter_clip_quality(raw_segments)
        if settings.HARD_FILTER_STRICT_MODE and not passed_segs:
            raise RuntimeError(
                f"Hard filter loai het {len(rejected_segs)} segments (strict_mode=True)."
            )
        raw_segments = passed_segs
        # Sprint 2 — Track stage counters cho verify_summary
        stage1_candidates = len(raw_segments) + len(rejected_segs)
        stage2_passed = len(raw_segments)
        self._last_rejected = rejected_segs  # type: ignore[attr-defined]
        self._last_stage1 = stage1_candidates  # type: ignore[attr-defined]
        self._last_stage2 = stage2_passed  # type: ignore[attr-defined]

        if not raw_segments:
            if progress_callback:
                progress_callback(
                    "ai_autocut", 100, 100,
                    f"Gemini trả về 0 segments đạt chuẩn (đã lọc {len(rejected_segs)} segments yếu)."
                )
            return AutoCutResult(
                video_id=video_id,
                video_path=str(path),
                video_duration=duration,
                total_segments=0,
                total_cost_usd=self._estimate_cost(duration),
                clips=[],
                source="ai_autocut",
                raw_gemini_response={"rejected_after_filter": rejected_segs},
                last_rejected=getattr(self, "_last_rejected", []),
                verify_summary={},
                stage1_candidates=stage1_candidates,
                stage2_passed=len(raw_segments),
                stage3_cut=0,
                stage4_verified=0,
                cost_breakdown={
                    "scan": self._estimate_cost_scan(duration),
                    "verify": {"stage": "verify_total", "total_usd": 0.0,
                               "input_tokens": 0, "output_tokens": 0,
                               "input_cost_usd": 0.0, "output_cost_usd": 0.0},
                    "grand_total_usd": self._estimate_cost_scan(duration)["total_usd"],
                    "video_duration_sec": duration,
                    "n_clips": 0,
                },
            )

        # 4. Cut tung segment
        clips_dir = settings.DATA_DIR / "clips"
        clips_dir.mkdir(parents=True, exist_ok=True)
        ffmpeg_path = self._get_ffmpeg_path()
        cut_clips: list[AutoCutSegment] = []
        total = len(raw_segments)

        for idx, seg in enumerate(raw_segments):
            if cancel_check and cancel_check():
                logger.info("AI autocut cancelled by user")
                break
            clip_id = f"{video_id}_ai_{idx}_{uuid.uuid4().hex[:8]}"
            clip_path = clips_dir / f"{clip_id}.mp4"
            dur = seg["end_time"] - seg["start_time"]
            try:
                self._cut_with_ffmpeg(
                    ffmpeg_path, path,
                    seg["start_time"], dur, clip_path,
                )
            except Exception as exc:
                logger.warning(f"Skip segment {idx} (cut failed): {exc}")
                continue
            cut_clips.append(AutoCutSegment(
                start_time=seg["start_time"],
                end_time=seg["end_time"],
                emotion=seg["emotion"],
                intensity=seg["intensity"],
                face_coverage=seg["face_coverage"],
                speaker_visible=seg["speaker_visible"],
                has_transcript=seg["has_transcript"],
                subject=seg.get("subject", ""),
                reasoning=seg.get("reasoning", ""),
                clip_path=str(clip_path.resolve()),
                clip_id=clip_id,
            ))
            if progress_callback:
                pct = 60 + int(35 * (idx + 1) / total)
                progress_callback(
                    "ai_autocut", pct, 100,
                    f"Đã cắt {idx + 1}/{total}: {seg['emotion']} ({seg['intensity']:.0%})",
                )

        if progress_callback:
            progress_callback(
                "ai_autocut", 100, 100,
                f"Hoàn tất AI Auto-Cut: {len(cut_clips)} clip.",
            )

        return AutoCutResult(
            video_id=video_id,
            video_path=str(path),
            video_duration=duration,
            total_segments=len(cut_clips),
            total_cost_usd=self._estimate_cost(duration),
            clips=cut_clips,
            source="ai_autocut",
            last_rejected=getattr(self, "_last_rejected", []),
            verify_summary={},
            stage1_candidates=getattr(self, "_last_stage1", len(cut_clips)),
            stage2_passed=getattr(self, "_last_stage2", len(cut_clips)),
            stage3_cut=len(cut_clips),
            stage4_verified=0,  # Verify pass chua chay (se duoc set khi goi verify_clip)
            cost_breakdown=self.estimate_total_cost(
                duration,
                n_clips=len(cut_clips),
                avg_clip_duration=sum(c.end_time - c.start_time for c in cut_clips) / max(1, len(cut_clips)),
            ),
        )

    # ── DB persistence ──────────────────────────────────

    def persist_clips(
        self,
        result: AutoCutResult,
        db: Session,
    ) -> int:
        """
        Insert/update Clip records tu AutoCutResult.
        Tra ve so clip da insert moi (khong tinh updated).
        """
        from backend.database.models import Clip
        from datetime import datetime

        inserted = 0
        for idx, seg in enumerate(result.clips):
            existing = db.query(Clip).filter(Clip.id == seg.clip_id).first()
            if existing:
                existing.predicted_emotion = seg.emotion
                existing.confidence = seg.intensity
                existing.face_ratio = seg.face_coverage
                # Sprint 2 — Verify pass fields
                existing.verify_verdict = seg.verify_verdict or None
                existing.verify_status = seg.verify_status or "not_run"
                existing.verify_reasoning = seg.verify_reasoning or None
                existing.rejected_by_verify = seg.rejected_by_verify
                if seg.reject_reason:
                    existing.reject_reason = seg.reject_reason
                existing.updated_at = datetime.utcnow()
                continue
            clip = Clip(
                id=seg.clip_id,
                video_id=result.video_id,
                clip_index=idx,
                start_time=seg.start_time,
                end_time=seg.end_time,
                duration=seg.end_time - seg.start_time,
                clip_path=seg.clip_path,
                is_manual_segment=False,
                predicted_emotion=seg.emotion,
                confidence=seg.intensity,
                face_ratio=seg.face_coverage,
                has_speech=seg.has_transcript,
                status="needs_review",
                decision_by="gemini_autocut",
                pipeline_stage="ai_autocut_done",
                # Sprint 2 — Verify pass fields
                verify_verdict=seg.verify_verdict or None,
                verify_status=seg.verify_status or "not_run",
                verify_reasoning=seg.verify_reasoning or None,
                rejected_by_verify=seg.rejected_by_verify,
                reject_reason=seg.reject_reason or None,
                reviewer_notes=f"[Gemini AutoCut] subject={seg.subject} | {seg.reasoning}"[:1000],
                per_model_scores={
                    "gemini_autocut": {
                        "emotion": seg.emotion,
                        "intensity": seg.intensity,
                        "face_coverage": seg.face_coverage,
                        "subject": seg.subject,
                        "reasoning": seg.reasoning,
                        "has_transcript": seg.has_transcript,
                    },
                    "autocut_meta": {
                        "source": "vertex_ai_global",
                        "cost_usd": result.total_cost_usd,
                        "model": getattr(self._ensure_labeler(), "model", "gemini-2.5-flash"),
                    },
                },
                all_scores={seg.emotion: seg.intensity},
            )
            db.add(clip)
            inserted += 1
        if inserted:
            db.commit()
        return inserted

    # ── Sprint 2 — Verify pass (Stage 4) ───────────────────────────────────

    # Cac verdict chuan hoa. Verify schema cung dung set nay (xem _build_verify_response_schema).
    VERIFY_VERDICTS = {
        "confirmed", "wrong_emotion", "unstable", "low_quality", "stats_mismatch",
    }

    def verify_clip(
        self,
        clip_path: str | Path,
        *,
        predicted_emotion: str,
        predicted_intensity: float,
        transcript: str = "",
        audio_features: dict[str, Any] | None = None,
        face_stats: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Verify 1 clip da cat (Stage 4).

        Args:
            clip_path: duong dan file clip (.mp4).
            predicted_emotion: emotion Stage 2 da gan.
            predicted_intensity: intensity Stage 2 (0-1).
            transcript: transcript tu Whisper (optional).
            audio_features: dict tu audio_extractor (snr_db, has_speech, rms, ...).
            face_stats: dict tu face_extractor (num_faces, frontal_ratio, avg_yaw, ...).

        Returns:
            dict gom:
              - verdict: confirmed | wrong_emotion | unstable | low_quality | stats_mismatch
              - emotion, intensity (sau verify)
              - face_coverage, frontal_ratio, speech_quality, stable
              - reasoning (tieng Viet)
              - error: str neu khong goi duoc Gemini
        """
        path = Path(clip_path)
        if not path.exists():
            return self._empty_verify_result(path, error="clip_not_found")

        # Build user prompt voi day du context
        duration = self._get_duration(path)
        user_prompt = self._build_verify_user_prompt(
            duration=duration,
            predicted_emotion=predicted_emotion,
            predicted_intensity=predicted_intensity,
            transcript=transcript,
            audio_features=audio_features or {},
            face_stats=face_stats or {},
        )

        # Sprint 3 — Cache layer
        from backend.services import gemini_cache
        cache_key = gemini_cache.make_key(
            str(path),
            stage="verify",
            prompt=VERIFY_SYSTEM_PROMPT + user_prompt,
            params={
                "predicted_emotion": predicted_emotion,
                "predicted_intensity": predicted_intensity,
            },
        )
        cached = gemini_cache.get(cache_key)
        if cached is not None:
            cached["from_cache"] = True
            return cached

        try:
            client = self._ensure_labeler()._resolve_client()
            schema = _build_verify_response_schema()
            config: dict[str, Any] = {
                "temperature": 0.1,             # rat thap de khop Stage 2
                # 4096 de du cho JSON schema 9 fields + reasoning dai (co the 500+ chars).
                "max_output_tokens": 4096,
                "system_instruction": VERIFY_SYSTEM_PROMPT,
            }
            if schema is not None:
                config["response_mime_type"] = "application/json"
                config["response_schema"] = schema

            response = client.models.generate_content(
                model=self._ensure_labeler().model,
                contents=[{"role": "user", "parts": [{"text": user_prompt}]}],
                config=config,
            )
            text = self._ensure_labeler()._get_text(response)
            result = self._parse_verify_response(text, predicted_emotion, predicted_intensity)
            # Sprint 3 — luu cache (tru truong hop parse_fail)
            if result.get("verdict") != "stats_mismatch" or "parse_fail" not in result.get("error", ""):
                gemini_cache.put(cache_key, result)
            return result
        except Exception as exc:
            logger.warning(f"[Verify] Gemini call failed: {exc}")
            return self._empty_verify_result(path, error=str(exc))

    def _build_verify_user_prompt(
        self,
        *,
        duration: float,
        predicted_emotion: str,
        predicted_intensity: float,
        transcript: str,
        audio_features: dict,
        face_stats: dict,
    ) -> str:
        """Build user prompt voi context day du cho Verify pass."""
        clean_transcript = (transcript or "").strip().replace("\n", " ")[:300]
        prompt_parts = [
            f"Clip {duration:.1f}s. Stage 2 verdict: emotion={predicted_emotion}, "
            f"intensity={predicted_intensity:.2f}.",
        ]
        if clean_transcript:
            prompt_parts.append(f'Lời thoại: "{clean_transcript}"')
        if audio_features:
            snr = audio_features.get("snr_db")
            has_speech = audio_features.get("has_speech_energy")
            prompt_parts.append(
                f"Audio: SNR={snr:.1f}dB" if snr is not None else "Audio: SNR=?"
            )
            if has_speech is not None:
                prompt_parts.append(f"has_speech={has_speech}")
        if face_stats:
            num_faces = face_stats.get("num_faces", 1)
            frontal = face_stats.get("frontal_ratio", 0.0)
            yaw = face_stats.get("avg_yaw", 0.0)
            prompt_parts.append(
                f"Face stats: {num_faces} face(s), frontal_ratio={frontal:.2f}, avg_yaw={yaw:.1f}°"
            )
        prompt_parts.append(
            "Verify theo 4 bước trong system prompt. Trả JSON (chỉ object, không text khác)."
        )
        return " ".join(prompt_parts)

    def _parse_verify_response(
        self,
        text: str,
        predicted_emotion: str,
        predicted_intensity: float,
    ) -> dict[str, Any]:
        """Parse JSON tu Verify pass, validate schema, fallback neu parse fail."""
        text = (text or "").strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text)
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            logger.warning(f"[Verify] JSON parse fail: {exc}; text={text[:200]!r}")
            # Fallback: giu nguyen Stage 2 verdict, danh "stats_mismatch" de review thu cong
            return {
                "verdict": "stats_mismatch",
                "emotion": predicted_emotion,
                "intensity": predicted_intensity,
                "face_coverage": 0.0,
                "frontal_ratio": 0.0,
                "speech_quality": "unknown",
                "stable": True,
                "speech_emotion_match": True,
                "reasoning": "Verify parse fail, fallback Stage 2 verdict.",
                "error": f"parse_fail: {exc}",
            }
        if not isinstance(data, dict):
            data = {}
        verdict = str(data.get("verdict", "stats_mismatch"))
        if verdict not in self.VERIFY_VERDICTS:
            verdict = "stats_mismatch"
        return {
            "verdict": verdict,
            "emotion": str(data.get("emotion", predicted_emotion)),
            "intensity": float(data.get("intensity", predicted_intensity)),
            "face_coverage": float(data.get("face_coverage", 0.0)),
            "frontal_ratio": float(data.get("frontal_ratio", 0.0)),
            "speech_quality": str(data.get("speech_quality", "unknown")),
            "stable": bool(data.get("stable", True)),
            "speech_emotion_match": bool(data.get("speech_emotion_match", True)),
            "reasoning": str(data.get("reasoning", ""))[:500],
        }

    @staticmethod
    def _empty_verify_result(clip_path: Path, error: str = "") -> dict[str, Any]:
        """Fallback khi Verify call fail (clip khong ton tai hoac Gemini loi)."""
        return {
            "verdict": "stats_mismatch",
            "emotion": "",
            "intensity": 0.0,
            "face_coverage": 0.0,
            "frontal_ratio": 0.0,
            "speech_quality": "unknown",
            "stable": True,
            "speech_emotion_match": True,
            "reasoning": "",
            "error": error,
        }

    def combine_verdicts(
        self,
        stage2_seg: dict[str, Any],
        verify_result: dict[str, Any],
    ) -> dict[str, Any]:
        """Merge Stage 2 verdict voi Verify pass → emotion/intensity cuoi cung.

        Decision logic (xem docs/03 §2.2):
        - confirmed          → emotion = verify (hoac stage2 neu thieu), intensity *= 1.05
        - wrong_emotion      → emotion = verify, intensity *= 0.7
        - unstable           → rejected (emotion flip trong clip)
        - low_quality        → rejected (clip xau)
        - stats_mismatch     → emotion = stage2, intensity = (s2 + v)/2

        Returns:
            dict seg da merge, co them key `verify_status`, `verify_verdict`,
            `rejected_by_verify` neu bi loai.
        """
        from backend.config import settings

        s2_emo = stage2_seg.get("emotion", "")
        s2_int = float(stage2_seg.get("intensity", 0))
        verdict = verify_result.get("verdict", "stats_mismatch")
        v_emo = verify_result.get("emotion") or s2_emo
        v_int = float(verify_result.get("intensity", s2_int))

        merged = dict(stage2_seg)
        merged["verify_verdict"] = verdict
        merged["verify_status"] = "rejected" if verdict in {"unstable", "low_quality"} else "passed"
        merged["verify_reasoning"] = verify_result.get("reasoning", "")

        if verdict == "confirmed":
            merged["emotion"] = v_emo
            merged["intensity"] = round(min(1.0, max(s2_int, v_int) * 1.05), 3)
        elif verdict == "wrong_emotion":
            merged["emotion"] = v_emo
            merged["intensity"] = round(min(1.0, v_int * 0.7), 3)
        elif verdict == "unstable":
            merged["rejected_by_verify"] = True
            merged["reject_reason"] = "emotion_unstable"
        elif verdict == "low_quality":
            merged["rejected_by_verify"] = True
            merged["reject_reason"] = "low_quality_detected"
        elif verdict == "stats_mismatch":
            merged["emotion"] = s2_emo
            merged["intensity"] = round((s2_int + v_int) / 2, 3)
        else:
            # Fallback khong xac dinh
            merged["emotion"] = s2_emo
            merged["intensity"] = s2_int

        # Emotion lock: neu flip_score > max_flip_score → unstable
        if settings.EMOTION_LOCK_ENABLED and s2_emo and v_emo:
            if s2_emo != v_emo:
                # Tinh flip_score heuristic: 1.0 neu khac emotion, giam theo intensity
                flip = 1.0 - abs(s2_int - v_int)
                if flip > settings.EMOTION_LOCK_MAX_FLIP_SCORE:
                    merged["rejected_by_verify"] = True
                    merged["reject_reason"] = "emotion_flip_detected"
                    merged["verify_status"] = "rejected"
                    logger.info(
                        f"[EmotionLock] flip_score={flip:.2f} > "
                        f"{settings.EMOTION_LOCK_MAX_FLIP_SCORE:.2f} → reject"
                    )
        return merged

    # ── Cost estimation ─────────────────────────────────

    @staticmethod
    def _estimate_cost(video_duration: float) -> float:
        """Uoc tinh USD cho Stage 2 (scan) voi 1 fps frame sampling.

        Luu y: Verify pass (Stage 4) tinh rieng. Tong cost =
        _estimate_cost_scan + _estimate_cost_verify * so_clip.
        """
        # gemini-2.5-flash gia ui (cap nhat T7/2026):
        # input ~ 7K tokens/frame, output ~ 200 tokens/segment
        input_tokens = int(video_duration) * 7000
        output_tokens = 1024
        input_cost = input_tokens * 0.30 / 1_000_000
        output_cost = output_tokens * 2.50 / 1_000_000
        return round(input_cost + output_cost, 6)

    @staticmethod
    def _estimate_cost_scan(video_duration: float) -> dict[str, Any]:
        """Sprint 3 — Estimate chi tiet cho Stage 2 (scan).

        Returns:
            dict voi input_tokens, output_tokens, input_cost_usd,
            output_cost_usd, total_usd.
        """
        input_tokens = int(video_duration) * 7000
        output_tokens = 1024
        return {
            "stage": "scan",
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "input_cost_usd": round(input_tokens * 0.30 / 1_000_000, 6),
            "output_cost_usd": round(output_tokens * 2.50 / 1_000_000, 6),
            "total_usd": round(
                (input_tokens * 0.30 + output_tokens * 2.50) / 1_000_000, 6
            ),
        }

    @staticmethod
    def _estimate_cost_verify(clip_duration: float) -> dict[str, Any]:
        """Sprint 3 — Estimate chi tiet cho Stage 4 (Verify pass).

        Verify pass doc 1 clip (khong phai full video) → input tokens
        tinh theo duration cua clip.
        """
        # Clip duration thuong 3-15s, frame extraction 1fps
        # Input tokens: duration * 7K/frame + ~500 cho system+user prompt
        input_tokens = int(clip_duration) * 7000 + 500
        output_tokens = 256
        return {
            "stage": "verify",
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "input_cost_usd": round(input_tokens * 0.30 / 1_000_000, 6),
            "output_cost_usd": round(output_tokens * 2.50 / 1_000_000, 6),
            "total_usd": round(
                (input_tokens * 0.30 + output_tokens * 2.50) / 1_000_000, 6
            ),
        }

    @staticmethod
    def estimate_total_cost(
        video_duration: float,
        n_clips: int,
        avg_clip_duration: float = 7.0,
    ) -> dict[str, Any]:
        """Tong hop cost scan + verify cho 1 video."""
        scan = AIVideoSegmenter._estimate_cost_scan(video_duration)
        verify_total = {
            "stage": "verify_total",
            "input_tokens": 0,
            "output_tokens": 0,
            "input_cost_usd": 0.0,
            "output_cost_usd": 0.0,
            "total_usd": 0.0,
        }
        for _ in range(n_clips):
            v = AIVideoSegmenter._estimate_cost_verify(avg_clip_duration)
            for k in ("input_tokens", "output_tokens", "input_cost_usd",
                      "output_cost_usd", "total_usd"):
                verify_total[k] += v[k]
        for k in ("input_cost_usd", "output_cost_usd", "total_usd"):
            verify_total[k] = round(verify_total[k], 6)
        return {
            "scan": scan,
            "verify": verify_total,
            "grand_total_usd": round(scan["total_usd"] + verify_total["total_usd"], 6),
            "video_duration_sec": video_duration,
            "n_clips": n_clips,
        }
