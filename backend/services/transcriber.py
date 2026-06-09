"""Vietnamese speech transcription service."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict


class SpeechTranscriber:
    def __init__(self, language: str = "vi"):
        self.language = language

    def transcribe_audio_clip(self, audio_path: str, clip_id: str) -> Dict[str, Any]:
        if not Path(audio_path).exists():
            return self._empty("audio_missing")
        try:
            from backend.ai_models.model_manager import model_manager
            model = model_manager.load_model("whisper")
            result = model.transcribe(
                audio_path,
                language=self.language,
                task="transcribe",
                fp16=self._use_fp16(),
                verbose=False,
                condition_on_previous_text=False,
                temperature=0,
                no_speech_threshold=0.35,
                logprob_threshold=-1.0,
                compression_ratio_threshold=2.8,
                initial_prompt="Đây là lời thoại tiếng Việt trong phim hoặc video hội thoại.",
            )
            transcript = self._normalize_text(result.get("text") or "")
            segments = result.get("segments") or []
            return {
                "transcript": transcript,
                "segments": segments,
                "language": result.get("language", self.language) or self.language,
                "main_speaker": "speaker_0" if transcript else None,
                "word_count": len(transcript.split()),
            }
        except Exception as exc:
            print(f"⚠️ [Transcriber] Clip {clip_id} lỗi nhận diện lời thoại: {exc}")
            data = self._empty(f"transcription_failed: {exc}")
            return data

    @staticmethod
    def _use_fp16() -> bool:
        try:
            import torch
            return torch.cuda.is_available()
        except Exception:
            return False

    def _empty(self, warning: str) -> Dict[str, Any]:
        return {
            "transcript": "",
            "segments": [],
            "language": self.language,
            "main_speaker": None,
            "word_count": 0,
            "warning": warning,
        }

    @staticmethod
    def _normalize_text(text: str) -> str:
        return " ".join(text.replace("\n", " ").strip().split())
