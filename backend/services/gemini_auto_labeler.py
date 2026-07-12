"""
Emotion Data Studio — Vertex AI Client
=====================================
Helper chuẩn hoá việc khởi tạo Google GenAI client cho Vertex AI.

Theo docs/09_vertex_ai_integration.md:
  - Gemini Enterprise Agent Platform bắt buộc dùng location="global".
  - Ưu tiên `google-genai` SDK, fallback thủ công qua REST.
  - Project ID đọc từ GOOGLE_APPLICATION_CREDENTIALS (file JSON key).
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger("EDS-VertexAI")

VERTEX_GLOBAL_LOCATION = "global"


def _load_service_account_info(path: str | Path) -> dict[str, Any]:
    """Đọc file key JSON của service account, trả về dict thông tin."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Service account key not found: {p}")
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(f"Khong doc duoc service account key {p}: {exc}") from exc


def _resolve_credentials() -> tuple[Any, str]:
    """
    Tra ve (credentials, project_id) tu service account JSON.

    Uu tien:
      1. GOOGLE_APPLICATION_CREDENTIALS env
      2. backend.config.settings.GOOGLE_APPLICATION_CREDENTIALS
      3. application-default credentials (gcloud auth login)
    """
    creds_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if not creds_path:
        try:
            from backend.config import settings
            creds_path = getattr(settings, "GOOGLE_APPLICATION_CREDENTIALS", None)
        except Exception:
            creds_path = None

    if creds_path and Path(creds_path).exists():
        info = _load_service_account_info(creds_path)
        project_id = (
            os.getenv("GCP_PROJECT_ID")
            or info.get("project_id")
            or "aura-social-vn"
        )
        try:
            from google.oauth2 import service_account as sa  # type: ignore
            scopes = ["https://www.googleapis.com/auth/cloud-platform"]
            creds = sa.Credentials.from_service_account_file(creds_path, scopes=scopes)
            return creds, project_id
        except Exception as exc:
            logger.warning(f"Khong tao duoc service_account.Credentials: {exc}")
            # Fallback: tra info de caller tu xu ly
            return {"_raw_info": info, "_path": creds_path}, project_id

    # Fallback: application-default credentials
    import google.auth as gauth  # type: ignore
    creds, project = gauth.default()
    if not creds.token or not creds.valid:
        try:
            from google.auth.transport import requests as grequests  # type: ignore
            creds.refresh(grequests.Request())
        except Exception as exc:
            raise RuntimeError(
                "Khong xac thuc duoc Google credentials. "
                "Hay dat GOOGLE_APPLICATION_CREDENTIALS hoac chay 'gcloud auth application-default login'. "
                f"Chi tiet: {exc}"
            ) from exc
    return creds, (os.getenv("GCP_PROJECT_ID") or project or "aura-social-vn")


def get_genai_client(
    location: str = VERTEX_GLOBAL_LOCATION,
    api_version: str = "v1",
    force_vertex: bool = True,
) -> tuple[Any, str, str]:
    """
    Tra ve (client, project_id, location) cho google-genai SDK.

    - location="global" mac dinh (theo docs/09).
    - neu force_vertex=False va co GEMINI_API_KEY thi tra ve AI Studio client.
    - Neu SDK chua duoc cai, raise RuntimeError voi huong dan.
    """
    try:
        import google.genai as genai  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "Chua cai google-genai SDK. Chay: pip install google-genai"
        ) from exc

    if not force_vertex:
        api_key = os.getenv("GEMINI_API_KEY")
        if api_key:
            return genai.Client(api_key=api_key), "ai-studio", "global"

    creds, project_id = _resolve_credentials()
    return (
        genai.Client(
            vertexai=True,
            credentials=creds,
            project=project_id,
            location=location,
            http_options={"api_version": api_version},
        ),
        project_id,
        location,
    )


def is_vertex_configured() -> tuple[bool, str]:
    """Kiem tra Vertex AI da san sang chua (credentials + project)."""
    try:
        creds, project = _resolve_credentials()
        creds_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
        if not creds_path:
            try:
                from backend.config import settings
                creds_path = getattr(settings, "GOOGLE_APPLICATION_CREDENTIALS", None)
            except Exception:
                pass
        if creds_path:
            return True, f"Vertex AI (global) ready, project={project}, key={Path(creds_path).name}"
        return True, f"Vertex AI (global) ready, project={project}, ADC"
    except Exception as exc:
        return False, f"Vertex AI chua cau hinh: {exc}"


EMOTION_LABELS = ["happy", "sad", "angry", "fear", "surprise", "disgust", "neutral"]
EMOTION_DESCRIPTIONS = {
    "happy": "vui vẻ, hạnh phúc, cười đùa, phấn khích",
    "sad": "buồn bã, thất vọng, khóc, chán nản, tuyệt vọng",
    "angry": "tức giận, bực bội, khó chịu, bùng nổ cảm xúc",
    "fear": "sợ hãi, lo lắng, bất an, hoảng sợ, căng thẳng",
    "surprise": "bất ngờ, ngạc nhiên, choáng ngợp, shock",
    "disgust": "ghê tởm, chán ghét, khinh miệt, ác cảm",
    "neutral": "bình thường, không có cảm xúc mạnh",
}

SYSTEM_PROMPT = f"""Bạn là chuyên gia phân tích cảm xúc trên video cho bộ dữ liệu huấn luyện.

NHIỆM VỤ: Xem video, xác định TẤT CẢ các khoảnh khắc có cảm xúc mạnh, và gán nhãn cảm xúc chính xác.

Các nhãn cảm xúc và mô tả:
{chr(10).join(f"- {k}: {v}" for k, v in EMOTION_DESCRIPTIONS.items())}

DANH SÁCH RÀNG BUỘC BẮT BUỘC (không được bỏ qua bất kỳ điều nào):

[RÀNG BUỘC CHẤT LƯỢNG KHUNG HÌNH]
1. Chỉ đánh dấu các đoạn có CƯỜNG ĐỘ cảm xúc >= 0.6 (thang 0.0 - 1.0). Đoạn yếu hơn → bỏ qua.
2. Đoạn phải dài tối thiểu 3 giây, tối đa 30 giây.
3. Trong suốt đoạn phải có ÍT NHẤT 60% số frame có KHUÔN MẶT CHÍNH DIỆN rõ ràng của chủ thể đang biểu lộ cảm xúc.
4. Khuôn mặt không được bị che khuất, quay nghiêng quá 45 độ, hoặc bị blur nặng.
5. Nếu khuôn mặt chủ thể bị thay đổi liên tục (chuyển cảnh, nhiều người), CHỈ đánh dấu khoảnh khắc của CHỦ THỂ CHÍNH đang nói/biểu lộ cảm xúc.

[RÀNG BUỘC VỀ NỘI DUNG VÀ NGỮ CẢNH]
6. Ưu tiên các đoạn có LỜI NÓI / TRANSCRIPT gắn liền với cảm xúc (giọng nói + nội dung).
7. Nếu đoạn có người đang NÓI, cảm xúc phải phù hợp với NỘI DUNG LỜI NÓI và GIỌNG NÓI. Nếu lời nói và biểu cảm mâu thuẫn, ghi chú trong reasoning.
8. Nếu không có lời nói, cảm xúc phải thể hiện rõ qua BIỂU CẢM MẶT VÀ CỬ CHỈ.

[RÀNG BUỘC VỀ CƯỜNG ĐỘ CẢM XÚC]
9. Cường độ >= 0.9: biểu cảm mặt rõ ràng + giọng nói/hành động phù hợp. Cảm xúc không thể nhầm lẫn.
10. Cường độ 0.7-0.9: có biểu cảm rõ nhưng có thể ngắn hoặc có yếu tố khác xen vào.
11. Cường độ 0.6-0.7: có cảm xúc nhưng mờ nhạt hoặc chỉ một phần đoạn có biểu hiện.

[RÀNG BUỘC VỀ OUTPUT]
12. Trả lời BẰNG TIẾNG VIỆT cho reasoning.
13. LUÔN trả về JSON hợp lệ, không thêm text khác ngoài JSON.
14. Nếu không có đoạn nào đủ mạnh, trả về mảng rỗng: [].
15. reasoning phải MÔ TẢ CỤ THỂ: biểu cảm gì (mắt, miệng, lông mày), giọng nói ra sao, lời nói nói gì.

Định dạng output:
```json
[
  {{
    "start_time": 12.5,
    "end_time": 28.3,
    "emotion": "angry",
    "intensity": 0.87,
    "face_coverage": 0.82,
    "speaker_visible": true,
    "has_transcript": true,
    "reasoning": "Giọng nói cao và run, khuôn mặt căng thẳng, lông mày nhíu chặt, cử chỉ tay mạnh mẽ. Người nói đang thể hiện sự tức giận rõ ràng."
  }}
]
```

TRƯỜNG HỢP ĐẶC BIỆT CẦN XỬ LÝ:
- Đoạn hành động (không có lời nói, không rõ cảm xúc): dùng neutral + ghi chú reason
- Đoạn có nhiều người: xác định CHỦ THỂ CHÍNH, bỏ qua фон/background
- Đoạn che mặt (tay, vật thể): không đánh dấu
- Đoạn quay nghiêng/xoáy camera: giảm intensity hoặc bỏ qua
- Đoạn có cười nhưng giọng buồn: ghi nhận incongruity, chọn emotion mạnh hơn
"""


class GeminiAutoLabeler:
    """
    Phân tích cảm xúc video bằng Gemini 2.5 Flash trên Vertex AI.
    Hỗ trợ video local (qua frame extraction) và video trên GCS.
    """

    def __init__(
        self,
        model: str | None = None,
        temperature: float | None = None,
        max_output_tokens: int | None = None,
        location: str | None = None,
    ):
        from backend.config import settings
        self.model = model or getattr(settings, "GEMINI_MODEL", None) or DEFAULT_MODEL
        self.temperature = temperature or getattr(settings, "GEMINI_TEMPERATURE", None) or 0.15
        self.max_output_tokens = max_output_tokens or getattr(settings, "GEMINI_MAX_TOKENS", None) or 8192
        # Per docs/09_vertex_ai_integration.md: Gemini Enterprise phai dung "global"
        # cho Agent Studio; default neu user khong override.
        self.location = (
            location
            or getattr(settings, "VERTEX_LOCATION", None)
            or VERTEX_GLOBAL_LOCATION
        )

        self.agent_url: str | None = getattr(settings, "AGENT_RUNTIME_URL", None)
        self.agent_api_key: str | None = getattr(settings, "AGENT_API_KEY", None)

        self._client: Optional[Any] = None

    # ── Agent Runtime (Cloud Run via Vertex AI Agent Studio) ──

    def _call_agent_runtime(self, prompt: str, frames_b64: list[str] | None = None) -> dict[str, Any]:
        """Goi deployed agent qua Cloud Run endpoint (Vertex AI Agent Studio)."""
        import urllib.request
        import urllib.error

        if not self.agent_url or not self.agent_api_key:
            raise RuntimeError(
                "Agent runtime chua duoc cau hinh. "
                "Dat AGENT_RUNTIME_URL va AGENT_API_KEY trong config."
            )

        url = self.agent_url.rstrip("/")

        image_parts = []
        if frames_b64:
            for i, frame in enumerate(frames_b64):
                image_parts.append(
                    {
                        "inline_data": {
                            "mime_type": "image/jpeg",
                            "data": frame,
                        }
                    }
                )

        contents = [
            {
                "role": "user",
                "parts": image_parts + [{"text": prompt}]
                if image_parts
                else [{"text": prompt}],
            }
        ]

        payload = json.dumps({"contents": contents}).encode("utf-8")
        headers = {
            "x-api-key": self.agent_api_key,
            "Content-Type": "application/json",
        }

        try:
            req = urllib.request.Request(url, data=payload, headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=300) as resp:
                result = json.loads(resp.read().decode("utf-8"))
                return self._parse_agent_response(result)
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")[:500]
            logger.error(f"Agent runtime HTTP {e.code}: {body}")
            raise RuntimeError(f"Agent runtime loi HTTP {e.code}: {body}") from e
        except Exception as exc:
            logger.error(f"Agent runtime failed: {exc}")
            raise RuntimeError(f"Agent runtime failed: {exc}") from exc

    def _parse_agent_response(self, result: dict[str, Any]) -> dict[str, Any]:
        """Parse response tu Vertex AI Agent Studio runtime."""
        try:
            candidates = result.get("candidates", [])
            if candidates:
                content = candidates[0].get("content", {})
                parts = content.get("parts", [])
                for part in parts:
                    if "text" in part:
                        text = part["text"].strip()
                        if text.startswith("```"):
                            text = re.sub(r"^```(?:json)?\s*", "", text)
                            text = re.sub(r"\s*```$", "", text)
                        return {"text": text}
            fallback = result.get("text") or result.get("raw", "")
            return {"text": fallback}
        except Exception as exc:
            logger.warning(f"Could not parse agent response: {exc}")
            return {"text": str(result)}

    # ── Client (Vertex AI only) ────────────────────────────

    def _resolve_client(self) -> Any:
        """Tạo google.genai.Client — ưu tiên API Key, fallback Vertex AI global."""
        if self._client is not None:
            return self._client

        from backend.services.gemini_auto_labeler import get_genai_client

        api_key = os.getenv("GEMINI_API_KEY") or getattr(
            __import__("backend.config", fromlist=["settings"]).settings,
            "GEMINI_API_KEY", None
        )
        try:
            if api_key:
                self._client, project_id, location = get_genai_client(
                    location="global", force_vertex=False
                )
                logger.info(f"GeminiAutoLabeler: Gemini API Key, model={self.model}")
                return self._client
        except Exception as exc:
            logger.warning(f"API key client init failed, fallback Vertex AI: {exc}")

        self._client, project_id, location = get_genai_client(
            location=self.location, force_vertex=True
        )
        logger.info(
            f"GeminiAutoLabeler: Vertex AI, project={project_id}, "
            f"location={location}, model={self.model}"
        )
        return self._client

    # ── Video helpers ─────────────────────────────────────

    def _get_video_duration(self, video_path: Path) -> float:
        """Lấy duration video bằng ffprobe."""
        try:
            result = subprocess.run(
                ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                 "-of", "default=noprint_wrappers=1:nokey=1", str(video_path)],
                capture_output=True, text=True, timeout=30,
            )
            return float(result.stdout.strip())
        except Exception:
            return 60.0

    def _video_to_frames_base64(self, video_path: Path, max_fps: float = 1.0, max_frames: int = 300) -> list[str]:
        """Extract frames at given FPS, return list of base64-encoded JPEG."""
        frames: list[str] = []
        tmp_dir = video_path.parent / f".tmp_frames_{os.getpid()}"
        tmp_dir.mkdir(exist_ok=True)
        try:
            result = subprocess.run(
                [
                    "ffmpeg", "-y", "-i", str(video_path),
                    "-vf", f"fps={max_fps}",
                    "-q:v", "3",
                    "-frames:v", str(max_frames),
                    str(tmp_dir / "frame_%04d.jpg"),
                ],
                capture_output=True, text=True, timeout=300,
            )
            if result.returncode != 0:
                logger.warning(f"ffmpeg failed: {result.stderr[:200]}")
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

    def _upload_video_to_gcs(self, video_path: Path) -> str | None:
        """Upload video lên GCS, trả về GCS URI. Cần gsutil."""
        bucket = os.getenv("GCS_BUCKET_NAME")
        if not bucket:
            return None
        filename = f"temp/{video_path.stem}_{os.getpid()}{video_path.suffix}"
        gcs_uri = f"gs://{bucket}/{filename}"
        try:
            result = subprocess.run(
                ["gsutil", "cp", str(video_path), gcs_uri],
                capture_output=True, text=True, timeout=300,
            )
            if result.returncode == 0:
                logger.info(f"Uploaded to GCS: {gcs_uri}")
                return gcs_uri
            else:
                logger.warning(f"gsutil upload failed: {result.stderr[:200]}")
                return None
        except FileNotFoundError:
            logger.warning("gsutil not found, skipping GCS upload")
            return None

    # ── Response parsing ───────────────────────────────────

    def _get_text(self, response: Any) -> str:
        """Lấy text từ Gemini response, xử lý trường hợp None."""
        text = getattr(response, "text", None)
        if text:
            return text.strip()
        if response.candidates:
            candidate = response.candidates[0]
            if hasattr(candidate, "content") and candidate.content:
                parts = list(candidate.content.parts)
                for p in parts:
                    if hasattr(p, "text") and p.text:
                        return p.text.strip()
        return ""

    def _parse_response(self, text: str) -> list[dict[str, Any]]:
        text = text.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text)
        try:
            data = json.loads(text)
            if isinstance(data, list):
                return self._validate_segments(data)
            for key in ("segments", "results", "emotional_segments", "emotions", "highlights"):
                if key in data and isinstance(data[key], list):
                    return self._validate_segments(data[key])
            return self._validate_segments([data])
        except json.JSONDecodeError:
            pass
        return self._extract_fallback(text)

    def _validate_segments(self, segments: list) -> list[dict[str, Any]]:
        valid = []
        for seg in segments:
            if not isinstance(seg, dict):
                continue
            start = float(seg.get("start_time", 0))
            end = float(seg.get("end_time", 0))
            emotion = str(seg.get("emotion", "")).lower().strip()
            intensity = float(seg.get("intensity", 0))
            duration = end - start
            if not (3.0 <= duration <= 30.0):
                continue
            if start < 0 or end <= start:
                continue
            if emotion not in EMOTION_LABELS:
                continue
            if intensity < 0.6:
                continue
            face_cov = float(seg.get("face_coverage", 0))
            # Proxy: if no face visible, speaker is not visible
            speaker_visible = bool(seg.get("speaker_visible", face_cov >= 0.3))
            has_transcript = bool(seg.get("has_transcript", False))
            valid.append({
                "start_time": round(start, 2),
                "end_time": round(end, 2),
                "emotion": emotion,
                "intensity": round(min(1.0, intensity), 3),
                "face_coverage": round(face_cov, 3),
                "speaker_visible": speaker_visible,
                "has_transcript": has_transcript,
                "reasoning": str(seg.get("reasoning", ""))[:500],
            })
        valid.sort(key=lambda s: s["start_time"])
        return valid

    def _extract_fallback(self, text: str) -> list[dict[str, Any]]:
        segments = []
        # Pattern: 12.5 - 28.3: angry (0.87)
        pattern = r"(\d+\.?\d*)\s*[-–to]+\s*(\d+\.?\d*)\s*[:\-\|]\s*(\w+)\s*\(?(\d+\.?\d*)\)?"
        for m in re.finditer(pattern, text):
            start, end, emotion, intensity = m.groups()
            emotion = emotion.lower()
            if emotion in EMOTION_LABELS and float(intensity) >= 0.6:
                segments.append({
                    "start_time": float(start), "end_time": float(end),
                    "emotion": emotion, "intensity": float(intensity),
                    "reasoning": "Extracted from text",
                })
        return self._validate_segments(segments)

    # ── Cost estimation ────────────────────────────────────

    def _estimate_cost(self, video_duration_sec: float, num_frames: int) -> dict[str, float]:
        """Ước tính chi phí dựa trên số frames."""
        # gemini-2.5-flash pricing on Vertex AI
        # 1 frame 720p ≈ ~7K tokens; 1 min video @ 1fps ≈ 420K tokens input
        input_tokens = num_frames * 7000
        output_tokens = min(self.max_output_tokens, num_frames * 200)
        input_cost = input_tokens * 0.30 / 1_000_000
        output_cost = output_tokens * 2.50 / 1_000_000
        return {
            "input_tokens_estimate": input_tokens,
            "output_tokens_estimate": output_tokens,
            "estimated_input_cost_usd": round(input_cost, 6),
            "estimated_output_cost_usd": round(output_cost, 6),
            "estimated_total_usd": round(input_cost + output_cost, 6),
        }

    # ── Core analysis methods ──────────────────────────────

    def _analyze_frames(
        self,
        frames_b64: list[str],
        video_duration: float,
        batch_start_offset: float = 0.0,
    ) -> list[dict[str, Any]]:
        """Gui frames len Gemini (Vertex AI hoac Cloud Run), nhan segments."""
        batch_size = 30
        user_prompt = (
            f"Video dai {video_duration:.1f} giay ({len(frames_b64)} frames @ 1 FPS). "
            "Phan tich tat ca cac khoanh khac co cam xuc manh (intensity >= 0.6). "
            "Tra ve JSON array. Neu khong co doan nao, tra ve []."
        )

        all_segments: list[dict] = []
        for i in range(0, len(frames_b64), batch_size):
            batch = frames_b64[i:i + batch_size]
            batch_num = i // batch_size + 1
            total_batches = (len(frames_b64) + batch_size - 1) // batch_size
            time_offset = batch_start_offset + i

            try:
                if self.agent_url and self.agent_api_key:
                    prompt = f"Batch {batch_num}/{total_batches}: frames {i+1}-{i+len(batch)} | {user_prompt}"
                    result = self._call_agent_runtime(prompt, batch)
                    text = result.get("text", "") if isinstance(result, dict) else str(result)
                else:
                    client = self._resolve_client()
                    image_parts = [
                        {"inline_data": {"mime_type": "image/jpeg", "data": frame}}
                        for frame in batch
                    ]
                    prompt_part = [{"text": f"Batch {batch_num}/{total_batches}: frames {i+1}-{i+len(batch)} | {user_prompt}"}]
                    response = client.models.generate_content(
                        model=self.model,
                        contents=[{"role": "user", "parts": image_parts + prompt_part}],
                        config={
                            "temperature": self.temperature,
                            "max_output_tokens": self.max_output_tokens,
                            "system_instruction": SYSTEM_PROMPT,
                        },
                    )
                    text = self._get_text(response)
                    if not text:
                        logger.warning(f"Batch {batch_num} returned empty response")
                        continue

                batch_segs = self._parse_response(text)
                for seg in batch_segs:
                    seg["start_time"] += time_offset
                    seg["end_time"] += time_offset
                all_segments.extend(batch_segs)
                logger.info(f"Batch {batch_num}/{total_batches}: {len(batch_segs)} segments")
            except Exception as exc:
                logger.warning(f"Batch {batch_num} failed: {exc}")
                continue

        return all_segments

    def _analyze_video_gcs(self, gcs_uri: str, video_duration: float) -> list[dict[str, Any]]:
        """Phan tich video truc tiep tu GCS (khong can frame extraction)."""
        if self.agent_url and self.agent_api_key:
            prompt = (
                f"Video dai {video_duration:.1f} giay. "
                "Phan tich tat ca cac khoanh khac co cam xuc manh (intensity >= 0.6). "
                "Tra ve JSON array. Neu khong co doan nao, tra ve []."
            )
            result = self._call_agent_runtime(prompt, None)
            text = result.get("text", "") if isinstance(result, dict) else str(result)
            return self._parse_response(text)

        client = self._resolve_client()
        try:
            response = client.models.generate_content(
                model=self.model,
                contents=[{
                    "role": "user",
                    "parts": [
                        {"file_data": {"mime_type": "video/mp4", "file_uri": gcs_uri}},
                        {"text": (
                            f"Video dai {video_duration:.1f} giay. "
                            "Phan tich tat ca cac khoanh khac co cam xuc manh (intensity >= 0.6). "
                            "Tra ve JSON array. Neu khong co doan nao, tra ve []."
                        )},
                    ]
                }],
                config={
                    "temperature": self.temperature,
                    "max_output_tokens": self.max_output_tokens,
                    "system_instruction": SYSTEM_PROMPT,
                },
            )
            text = self._get_text(response)
            if not text:
                logger.warning(f"GCS analysis empty: {response.candidates}")
                return []
            return self._parse_response(text)
        except Exception as exc:
            logger.error(f"GCS video analysis failed: {exc}")
            return []

    # ── Public API ─────────────────────────────────────────

    def analyze_video(
        self,
        video_path: str | Path | None = None,
        gcs_uri: str | None = None,
        intensity_threshold: float = 0.6,
        max_segments: int = 20,
        use_gcs_native: bool = True,
    ) -> dict[str, Any]:
        """
        Phân tích video, trả về các đoạn có cảm xúc mạnh.

        Args:
            video_path: Đường dẫn local đến video file
            gcs_uri: GCS URI của video (ưu tiên hơn video_path nếu có)
            intensity_threshold: Chỉ giữ segments có intensity >= ngưỡng
            max_segments: Giới hạn số segments trả về
            use_gcs_native: Dùng native video input của Gemini thay vì frame extraction
        """
        if not video_path and not gcs_uri:
            raise ValueError("video_path hoac gcs_uri is required")

        duration: float = 60.0
        all_segments: list[dict] = []

        if gcs_uri or use_gcs_native and video_path:
            # Ưu tiên 1: Dùng native video input của Gemini
            if gcs_uri:
                duration = self._get_video_duration(Path(video_path)) if video_path else 180.0
                all_segments = self._analyze_video_gcs(gcs_uri, duration)
            elif video_path:
                path = Path(video_path)
                if path.exists():
                    # Upload lên GCS trước
                    gcs = self._upload_video_to_gcs(path)
                    if gcs:
                        duration = self._get_video_duration(path)
                        all_segments = self._analyze_video_gcs(gcs, duration)
                    else:
                        # Fallback: dùng frames
                        all_segments = self._analyze_with_frames(path)
            if all_segments:
                logger.info(f"Native GCS analysis: {len(all_segments)} segments")
                return self._build_result(all_segments, duration, use_gcs_native, gcs_uri or str(video_path))

        # Fallback 2: Frame extraction
        if video_path:
            all_segments = self._analyze_with_frames(Path(video_path))
            duration = self._get_video_duration(Path(video_path))
            return self._build_result(all_segments, duration, False, str(video_path))

        raise RuntimeError("Khong the phan tich video")

    def _analyze_with_frames(self, video_path: Path) -> list[dict[str, Any]]:
        frames_b64 = self._video_to_frames_base64(video_path)
        if not frames_b64:
            raise RuntimeError("Khong the trich xuat frames tu video. Kiem tra ffmpeg.")
        return self._analyze_frames(frames_b64, self._get_video_duration(video_path))

    def _build_result(
        self,
        segments: list[dict],
        duration: float,
        used_gcs: bool,
        source: str,
    ) -> dict[str, Any]:
        num_frames = int(duration)
        # Deduplicate overlapping segments (keep higher intensity)
        seen: dict[str, dict] = {}
        for seg in segments:
            key = f"{seg['start_time']:.1f}-{seg['end_time']:.1f}-{seg['emotion']}"
            if key not in seen or seg["intensity"] > seen[key]["intensity"]:
                seen[key] = seg

        unique = sorted(seen.values(), key=lambda s: s["start_time"])
        cost = self._estimate_cost(duration, num_frames)

        return {
            "segments": unique,
            "cost_estimate": cost,
            "video_duration": duration,
            "model_used": self.model,
            "total_cost_usd": cost["estimated_total_usd"],
            "used_gcs_native": used_gcs,
            "source": source,
        }

    def analyze_clip(
        self,
        clip_path: str | Path,
        intensity_threshold: float = 0.5,
        transcript: str = "",
    ) -> dict[str, Any]:
        """Phan tich mot clip da cat — tra ve emotion cua toan bo clip."""
        path = Path(clip_path)
        if not path.exists():
            raise FileNotFoundError(f"Clip not found: {clip_path}")

        duration = self._get_video_duration(path)
        frames_b64 = self._video_to_frames_base64(path)
        if not frames_b64:
            raise RuntimeError("Khong the trich xuat frames tu clip")

        cost = self._estimate_cost(duration, len(frames_b64))
        has_transcript = bool(transcript and transcript.strip())
        prompt_parts = [
            f"Clip dai {duration:.1f} giay.",
            "Phan tich cam xuc CHINH cua toan bo clip.",
        ]
        if has_transcript:
            clean = transcript.strip().replace("\n", " ")[:300]
            prompt_parts.append(f"Loi thoai: \"{clean}...\"")
        prompt_parts.append(
            "Tra ve JSON voi emotion, intensity (0-1), face_coverage, speaker_visible, "
            "has_transcript, reasoning tieng Viet. "
            "Chi tra ve JSON, khong them text khac."
        )
        prompt = " ".join(prompt_parts)

        try:
            if self.agent_url and self.agent_api_key:
                result = self._call_agent_runtime(prompt, frames_b64)
                text = result.get("text", "") if isinstance(result, dict) else str(result)
            else:
                client = self._resolve_client()
                image_parts = [
                    {"inline_data": {"mime_type": "image/jpeg", "data": frame}}
                    for frame in frames_b64
                ]
                response = client.models.generate_content(
                    model=self.model,
                    contents=[{"role": "user", "parts": image_parts + [{"text": prompt}]}],
                    config={
                        "temperature": 0.15,
                        "max_output_tokens": 2048,
                        "system_instruction": SYSTEM_PROMPT,
                    },
                )
                text = self._get_text(response)

            if text.startswith("```"):
                text = re.sub(r"^```(?:json)?\s*", "", text)
                text = re.sub(r"\s*```$", "", text)
            try:
                data = json.loads(text)
                if isinstance(data, list) and data:
                    data = data[0]
            except json.JSONDecodeError:
                data = {"raw": text[:500]}

            return {
                "clip_path": str(path),
                "duration": duration,
                "emotion": data.get("emotion", data.get("predicted_emotion")),
                "intensity": data.get("intensity", data.get("confidence", 0)),
                "face_coverage": data.get("face_coverage", 0),
                "speaker_visible": data.get("speaker_visible", True),
                "has_transcript": has_transcript,
                "transcript": transcript if has_transcript else "",
                "reasoning": data.get("reasoning", ""),
                "analysis": data,
                "cost_estimate": cost,
                "total_cost_usd": cost["estimated_total_usd"],
            }
        except Exception as exc:
            logger.error(f"Clip analysis failed: {exc}")
            raise RuntimeError(f"Clip analysis failed: {exc}") from exc

    def batch_analyze(
        self,
        video_paths: list[str | Path],
        intensity_threshold: float = 0.6,
        max_segments_per_video: int = 20,
    ) -> list[dict[str, Any]]:
        """Phân tích nhiều video liên tiếp."""
        results = []
        for path in video_paths:
            try:
                result = self.analyze_video(
                    video_path=path,
                    intensity_threshold=intensity_threshold,
                    max_segments=max_segments_per_video,
                )
                results.append(result)
            except Exception as exc:
                results.append({"video_path": str(path), "error": str(exc), "segments": []})
        return results

    # ── Config / Status ───────────────────────────────────

    def is_configured(self) -> tuple[bool, str]:
        """Kiem tra Gemini credentials (Vertex AI global hoac API key)."""
        api_key = os.getenv("GEMINI_API_KEY") or getattr(
            __import__("backend.config", fromlist=["settings"]).settings,
            "GEMINI_API_KEY", None
        )
        if api_key:
            return True, f"San sang (Gemini API Key, model={self.model})"

        try:
            from backend.services.gemini_auto_labeler import is_vertex_configured
            return is_vertex_configured()
        except Exception as exc:
            return False, f"Vertex AI chua cau hinh: {exc}"

    def status(self) -> dict[str, Any]:
        """Tra ve trang thai cau hinh."""
        configured, message = self.is_configured()
        return {
            "configured": configured,
            "message": message,
            "model": self.model,
            "location": self.location,
            "temperature": self.temperature,
            "max_output_tokens": self.max_output_tokens,
            "agent_runtime": {
                "enabled": bool(self.agent_url and self.agent_api_key),
                "url": self.agent_url,
                "api_key_set": bool(self.agent_api_key),
            },
        }
