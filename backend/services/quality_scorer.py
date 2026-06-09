"""Quality scoring and routing for generated emotion labels."""

from __future__ import annotations


class QualityScorer:
    def calculate_score(
        self,
        confidence: float = 0.0,
        agreement_str: str | None = None,
        sampled_frames_count: int = 0,
        cropped_faces_count: int = 0,
        audio_clarity: float = 0.0,
    ) -> dict:
        confidence = self._clamp(confidence)
        agreement = self._parse_agreement(agreement_str)
        face_coverage = self._clamp((cropped_faces_count or 0) / max(1, sampled_frames_count or 1))
        audio_signal = self._clamp(float(audio_clarity or 0.0) * 18.0)

        quality = (
            0.42 * confidence
            + 0.28 * agreement
            + 0.20 * face_coverage
            + 0.10 * audio_signal
        )
        quality = self._clamp(quality)

        if quality >= 0.80 and confidence >= 0.72 and agreement >= 0.67 and face_coverage >= 0.20:
            status = "auto_approved"
        elif quality >= 0.42 and confidence >= 0.38:
            status = "needs_review"
        else:
            status = "failed"

        return {
            "quality_score": quality,
            "status": status,
            "confidence_score": confidence,
            "agreement_score": agreement,
            "face_score": face_coverage,
            "audio_score": audio_signal,
        }

    @staticmethod
    def _parse_agreement(value: str | None) -> float:
        if not value or "/" not in value:
            return 0.0
        try:
            left, right = value.split("/", 1)
            right_i = int(right)
            return int(left) / right_i if right_i else 0.0
        except Exception:
            return 0.0

    @staticmethod
    def _clamp(value: float) -> float:
        return max(0.0, min(1.0, float(value or 0.0)))
