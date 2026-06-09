"""Multimodal emotion analysis for Vietnamese video clips.

The service intentionally treats text as Vietnamese-first and visual/audio as
secondary evidence. Final labels are conservative: uncertain or conflicting
clips go to human review instead of being over-confidently auto-approved.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable


EMOTIONS = ["happy", "sad", "angry", "fear", "surprise", "disgust", "neutral"]

VI_LEXICON = {
    "happy": ["vui", "hanh phuc", "cuoi", "thich", "yeu", "tuyet", "may qua", "mung", "suong", "vui ve", "hanh phuc", "hoan hoan", "ron rang"],
    "sad": ["buon", "khoc", "dau long", "co don", "mat", "nho", "tui", "that vong", "tui nhu", "nuoc mat", "bi luy", "sau", "thuong"],
    "angry": ["gian", "tuc", "buc", "do khon", "im di", "cam", "ghet", "dien", "khong tha", "cam han", "tuc gian", "buc boi"],
    "fear": ["so", "lo", "hoang", "cuu", "nguy hiem", "chay di", "dung", "hai", "run", "rung minh", "lo lang", "bat an", "hoang loan"],
    "surprise": ["sao", "gi co", "that an", "khong the", "bat ngo", "troi oi", "ua", "chu sao", "lam sao", "that khong", "khong tin noi"],
    "disgust": ["ghe", "kinh", "tom", "ban", "khinh", "dang ghet", "buon non", "ghe tom", "kinh tom", "buc", "ngay", "ac"],
}


class EmotionAnalyzer:
    def analyze_clip(self, face_images: Iterable[str] | None = None, transcript: str = "", audio_path: str | None = None) -> Dict[str, Any]:
        per_model: dict[str, dict[str, float]] = {}
        diagnostics: dict[str, Any] = {}

        visual = self._visual_emotion(face_images or [])
        if visual:
            per_model["visual_deepface"] = visual

        text = self._vietnamese_text_emotion(transcript or "")
        if text:
            per_model["text_vi_lexicon"] = text

        audio = self._audio_emotion(audio_path)
        if audio:
            per_model["audio_wav2vec"] = audio

        weights = {
            "visual_deepface": 0.45,
            "text_vi_lexicon": 0.35,
            "audio_wav2vec": 0.20,
        }
        combined = self._weighted_combine(per_model, weights)
        predicted = max(combined, key=combined.get) if combined else "unknown"
        confidence = float(combined.get(predicted, 0.0)) if combined else 0.0
        winners = {name: max(scores, key=scores.get) for name, scores in per_model.items() if scores}
        agreement_count = sum(1 for value in winners.values() if value == predicted)
        agreement = f"{agreement_count}/{len(winners)}" if winners else "0/0"
        diagnostics["model_winners"] = winners
        diagnostics["text_length"] = len((transcript or "").split())

        return {
            "predicted_emotion": predicted,
            "confidence": confidence,
            "agreement": agreement,
            "has_incongruity": len(set(winners.values())) > 1 if winners else False,
            "all_scores": combined or {emotion: 0.0 for emotion in EMOTIONS},
            "per_model_scores": per_model,
            "diagnostics": diagnostics,
        }

    def _visual_emotion(self, face_images: Iterable[str]) -> dict[str, float]:
        paths = [p for p in face_images if p and Path(p).exists()]
        if not paths:
            return {}
        aggregate = defaultdict(float)
        used = 0
        try:
            from backend.ai_models.model_manager import model_manager
            DeepFace = model_manager.load_model("deepface")
            for path in paths[:5]:
                result = DeepFace.analyze(path, actions=["emotion"], enforce_detection=False, silent=True)
                item = result[0] if isinstance(result, list) else result
                raw = item.get("emotion", {}) or {}
                normalized = self._normalize({self._map_emotion(k): float(v) for k, v in raw.items()})
                for emotion, score in normalized.items():
                    aggregate[emotion] += score
                used += 1
        except Exception as exc:
            print(f"⚠️ [EmotionAnalyzer] DeepFace visual emotion failed: {exc}")
            return {}
        if not used:
            return {}
        return self._normalize({emotion: aggregate[emotion] / used for emotion in EMOTIONS})

    def _vietnamese_text_emotion(self, transcript: str) -> dict[str, float]:
        if not transcript or not transcript.strip():
            return {}
        try:
            from backend.ai_models.text_emotion_model import text_emotion_classifier
            result = text_emotion_classifier.predict(transcript)
            return self._normalize({row["label"]: float(row["score"]) for row in result})
        except Exception:
            pass
        # Fallback: simple lexicon matching
        return self._lexicon_fallback(transcript)

    def _audio_emotion(self, audio_path: str | None) -> dict[str, float]:
        if not audio_path or not Path(audio_path).exists():
            return {}
        try:
            from backend.ai_models.audio_emotion_model import audio_emotion_classifier
            result = audio_emotion_classifier.predict(audio_path)
            return self._normalize({row["label"]: float(row["score"]) for row in result})
        except Exception:
            return {}

    def _weighted_combine(self, per_model: dict[str, dict[str, float]], weights: dict[str, float]) -> dict[str, float]:
        if not per_model:
            return {}
        acc = defaultdict(float)
        total_weight = 0.0
        for name, scores in per_model.items():
            weight = weights.get(name, 0.1)
            normalized = self._normalize(scores)
            for emotion, score in normalized.items():
                acc[emotion] += score * weight
            total_weight += weight
        if total_weight <= 0:
            return {"neutral": 1.0}
        return self._normalize({emotion: acc[emotion] / total_weight for emotion in EMOTIONS})

    @staticmethod
    def _normalize(scores: dict[str, float]) -> dict[str, float]:
        mapped = {emotion: max(0.0, float(scores.get(emotion, 0.0))) for emotion in EMOTIONS}
        total = sum(mapped.values())
        if total <= 0:
            return {"neutral": 1.0}
        return {emotion: value / total for emotion, value in mapped.items()}

    @staticmethod
    def _lexicon_fallback(text: str) -> dict[str, float]:
        """Simple lexicon fallback when PhoBERT encoding is unavailable."""
        import unicodedata
        normalized = "".join(
            c for c in unicodedata.normalize("NFD", text.lower().strip())
            if unicodedata.category(c) != "Mn"
        )
        scores = {emotion: 0.0 for emotion in EMOTIONS}
        for emotion, phrases in VI_LEXICON.items():
            for phrase in phrases:
                norm_phrase = "".join(
                    c for c in unicodedata.normalize("NFD", phrase)
                    if unicodedata.category(c) != "Mn"
                )
                # Require at least 3 chars to avoid single-character false matches
                if len(norm_phrase) >= 3 and norm_phrase in normalized:
                    scores[emotion] += 1.0 + min(1.0, len(phrase.split()) * 0.15)
        if sum(scores.values()) == 0:
            return {}
        scores["neutral"] += 0.25
        return EmotionAnalyzer._normalize(scores)

    @staticmethod
    def _map_emotion(label: str) -> str:
        label = (label or "").lower()
        if "happy" in label or "joy" in label or "happiness" in label:
            return "happy"
        if "sad" in label or "sadness" in label:
            return "sad"
        if "ang" in label:
            return "angry"
        if "fear" in label:
            return "fear"
        if "sur" in label:
            return "surprise"
        if "dis" in label:
            return "disgust"
        return "neutral"
