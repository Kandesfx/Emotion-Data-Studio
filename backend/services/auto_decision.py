from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Decision:
    status: str
    decision_by: str
    reject_reason: str | None = None


class AutoDecisionEngine:
    """Nine-criteria auto decision gate from COLAB_EDS_SYSTEM_PLAN.md."""

    APPROVE_AGREEMENTS = {"3/3", "2/3", "3/4", "4/4"}

    def decide(self, metrics: dict[str, Any]) -> Decision:
        transcript = (metrics.get("transcript") or "").strip()
        word_count = len(transcript.split())
        duration = float(metrics.get("duration") or 0.0)
        snr_db = self._float(metrics.get("snr_db"), default=20.0)
        frontal_ratio = self._float(metrics.get("frontal_ratio"), default=1.0)
        face_quality = self._float(metrics.get("face_quality"), default=metrics.get("quality_score") or 0.0)
        quality_score = self._float(metrics.get("quality_score"), default=0.0)
        confidence = self._float(metrics.get("confidence"), default=0.0)
        agreement = metrics.get("agreement") or ""
        predicted = metrics.get("predicted_emotion") or metrics.get("emotion_final")
        num_faces = int(metrics.get("num_faces") or 0)
        has_incongruity = bool(metrics.get("has_incongruity"))

        reject_reasons = []
        if num_faces == 0:
            reject_reasons.append("no_face")
        if word_count < 2:
            reject_reasons.append("transcript_too_short")
        if duration < 2.0:
            reject_reasons.append("duration_too_short")
        if snr_db < 5.0:
            reject_reasons.append("snr_too_low")
        if frontal_ratio < 0.30:
            reject_reasons.append("head_pose_bad")
        if face_quality < 0.20:
            reject_reasons.append("face_quality_too_low")
        if not predicted or predicted == "unknown":
            reject_reasons.append("unknown_emotion")

        if reject_reasons:
            return Decision(status="rejected", decision_by="auto", reject_reason=", ".join(reject_reasons))

        approve = all(
            [
                quality_score >= 0.80,
                confidence >= 0.70,
                agreement in self.APPROVE_AGREEMENTS,
                not has_incongruity,
                word_count >= 3,
                3.0 <= duration <= 15.0,
                snr_db >= 15.0,
                frontal_ratio >= 0.80,
                face_quality >= 0.60,
            ]
        )
        if approve:
            return Decision(status="approved", decision_by="auto")

        return Decision(status="needs_review", decision_by="auto")

    @staticmethod
    def _float(value: Any, default: float) -> float:
        try:
            if value is None:
                return default
            return float(value)
        except (TypeError, ValueError):
            return default
