"""Central lazy model registry for Emotion Data Studio.

The desktop pipeline must be able to run on machines with partial model
availability. This manager therefore:
- lazy-loads heavy models once
- reports unavailable models without crashing the whole app
- exposes a stable set of model keys used by services
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class ModelStatus:
    key: str
    loaded: bool
    error: str | None = None


class FasterWhisperAdapter:
    """Adapter để faster-whisper có interface gần giống openai-whisper."""

    def __init__(self, model: Any):
        self.model = model

    def transcribe(self, audio_path: str, **kwargs) -> dict:
        segments, info = self.model.transcribe(
            audio_path,
            language=kwargs.get("language"),
            task=kwargs.get("task", "transcribe"),
            beam_size=5,
            vad_filter=True,
            initial_prompt=kwargs.get("initial_prompt"),
        )
        rows = []
        texts = []
        for seg in segments:
            text = (seg.text or "").strip()
            if text:
                texts.append(text)
            rows.append({
                "start": float(seg.start),
                "end": float(seg.end),
                "text": text,
            })
        return {
            "text": " ".join(texts),
            "segments": rows,
            "language": getattr(info, "language", kwargs.get("language", "vi")),
        }


class ModelManager:
    def __init__(self):
        self._models: dict[str, Any] = {}
        self._errors: dict[str, str] = {}
        self._loaders: dict[str, Callable[[], Any]] = {
            "whisper": self._load_whisper,
            "deepface": self._load_deepface,
            "mtcnn": self._load_mtcnn,
            "text_emotion": self._load_text_emotion,
            "audio_emotion": self._load_audio_emotion,
        }

    def load_model(self, key: str) -> Any:
        if key in self._models:
            return self._models[key]
        if key not in self._loaders:
            raise KeyError(f"Unknown model key: {key}")
        try:
            model = self._loaders[key]()
            self._models[key] = model
            self._errors.pop(key, None)
            return model
        except Exception as exc:
            self._errors[key] = str(exc)
            raise

    def get_model(self, key: str) -> Any | None:
        try:
            return self.load_model(key)
        except Exception:
            return None

    def prewarm_models(self, keys: list[str] | None = None) -> tuple[int, int]:
        # Prewarm only the core models needed for the current data-mining workflow.
        # Text/audio emotion models are optional and loaded lazily to avoid slowing
        # down Vietnamese video processing when visual labeling is the priority.
        keys = keys or ["whisper", "deepface", "mtcnn"]
        loaded = 0
        failed = 0
        for key in keys:
            try:
                self.load_model(key)
                loaded += 1
            except Exception:
                failed += 1
        return loaded, failed

    def status(self) -> list[ModelStatus]:
        keys = sorted(self._loaders)
        return [ModelStatus(key, key in self._models, self._errors.get(key)) for key in keys]

    def _device(self) -> str:
        try:
            from backend.utils.resource_manager import resource_manager
            return resource_manager.apply().device
        except Exception:
            try:
                import torch
                return "cuda" if torch.cuda.is_available() else "cpu"
            except Exception:
                return "cpu"

    def _load_whisper(self):
        model_name = os.getenv("EDS_WHISPER_MODEL", "medium")
        try:
            import whisper
            return whisper.load_model(model_name, device=self._device())
        except Exception as openai_exc:
            try:
                from faster_whisper import WhisperModel
                device = self._device()
                compute_type = "float16" if device == "cuda" else "int8"
                model = WhisperModel(model_name, device=device, compute_type=compute_type)
                return FasterWhisperAdapter(model)
            except Exception as faster_exc:
                raise RuntimeError(
                    f"Không load được Whisper. openai-whisper: {openai_exc}; faster-whisper: {faster_exc}"
                ) from faster_exc

    def _load_deepface(self):
        from deepface import DeepFace
        return DeepFace

    def _load_mtcnn(self):
        from facenet_pytorch import MTCNN
        return MTCNN(keep_all=True, device=self._device())

    def _load_audio_emotion(self):
        from backend.ai_models.audio_emotion_model import audio_emotion_classifier
        return audio_emotion_classifier

    def _load_text_emotion(self):
        from backend.ai_models.text_emotion_model import text_emotion_classifier
        return text_emotion_classifier


model_manager = ModelManager()
