"""Self-tuning Review Queue Agent — Sprint 4.

Read-only deterministic agent that triages clips into three buckets:
  - auto_approve_candidates: clips a human reviewer will almost certainly approve
  - auto_reject_candidates: clips a human reviewer will almost certainly reject
  - needs_human_review: clips that need human attention

The agent never modifies the database, never calls Gemini, never auto-writes
a status. It is purely a routing hint consumed by the UI or by humans who
want to skip the obvious 60% and focus on the 40% that matter.

See: docs/04_review_queue_agent_spec.md (source of truth for behavior)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sqlalchemy.orm import Session

from backend.config import settings


# ── Bucket labels (string constants for routing) ──────────────────────────

BUCKET_AUTO_APPROVE = "auto_approve_candidates"
BUCKET_AUTO_REJECT = "auto_reject_candidates"
BUCKET_NEEDS_REVIEW = "needs_human_review"


@dataclass
class ClipBucket:
    """One clip's routing decision."""

    clip_id: str
    bucket: str
    confidence: float
    reasons: list[str] = field(default_factory=list)
    clip_index: int | None = None
    duration: float | None = None
    ai_emotion: str | None = None
    quality_score: float | None = None


@dataclass
class AgentResult:
    """Output of one agent.run() call."""

    auto_approve_candidates: list[ClipBucket]
    auto_reject_candidates: list[ClipBucket]
    needs_human_review: list[ClipBucket]
    summary: dict

    def to_log_dict(self) -> dict:
        """Compact dict for JSONL log line."""
        return {
            "ts": datetime.utcnow().isoformat(),
            "n_clips": self.summary["n_clips"],
            "n_auto_approve": self.summary["n_auto_approve"],
            "n_auto_reject": self.summary["n_auto_reject"],
            "n_needs_review": self.summary["n_needs_review"],
            "auto_approve_ratio": self.summary["auto_approve_ratio"],
            "auto_reject_ratio": self.summary["auto_reject_ratio"],
        }


class ReviewQueueAgent:
    """Routes clips into auto-approve / auto-reject / needs-human-review.

    Constructor takes a SQLAlchemy Session. Reads Clip rows. Never writes.

    Usage:
        agent = ReviewQueueAgent(session)
        result = agent.run()
        for entry in result.auto_approve_candidates:
            print(entry.clip_id, entry.confidence, entry.reasons)
    """

    def __init__(self, session: "Session", video_id: str | None = None):
        self.session = session
        self.video_id = video_id
        self.log_path = self._default_log_path()

    # ── Public API ──────────────────────────────────────────────────────

    def run(self, log_to_file: bool = True) -> AgentResult:
        """Route all clips (or video_id-scoped subset) into buckets."""
        clips = self._fetch_clips()
        auto_approve, auto_reject, needs_review = [], [], []
        for clip in clips:
            entry = self._classify(clip)
            if entry.bucket == BUCKET_AUTO_APPROVE:
                auto_approve.append(entry)
            elif entry.bucket == BUCKET_AUTO_REJECT:
                auto_reject.append(entry)
            else:
                needs_review.append(entry)

        # Order each bucket by confidence DESC (most confident first)
        auto_approve.sort(key=lambda e: e.confidence, reverse=True)
        auto_reject.sort(key=lambda e: e.confidence, reverse=True)
        needs_review.sort(key=lambda e: e.confidence, reverse=True)

        n = len(clips)
        summary = {
            "n_clips": n,
            "n_auto_approve": len(auto_approve),
            "n_auto_reject": len(auto_reject),
            "n_needs_review": len(needs_review),
            "auto_approve_ratio": (len(auto_approve) / n) if n else 0.0,
            "auto_reject_ratio": (len(auto_reject) / n) if n else 0.0,
            "target_auto_approve_ratio": settings.REVIEW_QUEUE_BUCKET_AUTO_APPROVE_RATIO,
            "target_auto_reject_ratio": settings.REVIEW_QUEUE_BUCKET_AUTO_REJECT_RATIO,
        }
        result = AgentResult(
            auto_approve_candidates=auto_approve,
            auto_reject_candidates=auto_reject,
            needs_human_review=needs_review,
            summary=summary,
        )
        if log_to_file:
            self._write_log(result)
        return result

    # ── Classification logic ────────────────────────────────────────────

    def _classify(self, clip) -> ClipBucket:
        """Apply rules from spec §"Routing rules" to one clip."""
        ai_conf = float(clip.confidence or 0.0)
        quality = float(clip.quality_score or 0.0)
        verify_verdict = (clip.verify_verdict or "").strip()
        verify_status = (clip.verify_status or "").strip()
        rejected_by_verify = bool(clip.rejected_by_verify)
        has_incongruity = bool(clip.has_incongruity)

        reasons: list[str] = []

        # ── AUTO-REJECT rules (applied first so they win ties) ──────────
        if rejected_by_verify or verify_verdict == "wrong_emotion":
            reasons.append(
                f"verify_verdict={verify_verdict or 'rejected_by_verify=True'}"
            )
            return self._make_entry(
                clip,
                bucket=BUCKET_AUTO_REJECT,
                confidence=0.95,
                reasons=reasons,
            )

        if ai_conf < settings.REVIEW_QUEUE_AUTO_REJECT_CONFIDENCE:
            reasons.append(
                f"ai_confidence={ai_conf:.2f} < {settings.REVIEW_QUEUE_AUTO_REJECT_CONFIDENCE:.2f}"
            )
            return self._make_entry(
                clip,
                bucket=BUCKET_AUTO_REJECT,
                confidence=0.80,
                reasons=reasons,
            )

        # ── AUTO-APPROVE rules ──────────────────────────────────────────
        if (
            verify_status == "passed"
            and verify_verdict == "confirmed"
            and ai_conf >= settings.REVIEW_QUEUE_AUTO_APPROVE_CONFIDENCE
            and quality >= settings.REVIEW_QUEUE_AUTO_APPROVE_QUALITY
            and not has_incongruity
        ):
            confidence = self._compute_auto_approve_confidence(
                ai_conf, quality, verify_verdict, has_incongruity
            )
            reasons.extend([
                f"verify={verify_verdict}",
                f"ai_confidence={ai_conf:.2f}",
                f"quality_score={quality:.2f}",
                "no_incongruity",
            ])
            return self._make_entry(
                clip,
                bucket=BUCKET_AUTO_APPROVE,
                confidence=confidence,
                reasons=reasons,
            )

        # ── NEEDS-HUMAN-REVIEW (default) ────────────────────────────────
        if has_incongruity and verify_status != "rejected":
            reasons.append("has_incongruity=True")
        if verify_status == "not_run":
            reasons.append("verify_not_yet_run")
        if not reasons:
            reasons.append("default_to_human_review")
        return self._make_entry(
            clip,
            bucket=BUCKET_NEEDS_REVIEW,
            confidence=0.5,
            reasons=reasons,
        )

    def _compute_auto_approve_confidence(
        self,
        ai_conf: float,
        quality: float,
        verify_verdict: str,
        has_incongruity: bool,
    ) -> float:
        """Weighted score in [0.0, 1.0]. See spec §"Confidence scoring"."""
        high_conf = settings.REVIEW_QUEUE_AUTO_APPROVE_CONFIDENCE or 0.85
        high_quality = settings.REVIEW_QUEUE_AUTO_APPROVE_QUALITY or 0.85

        conf_term = max(0.0, min(1.0, 1.0 - (high_conf - ai_conf) / high_conf))
        quality_term = max(0.0, min(1.0, 1.0 - (high_quality - quality) / high_quality))
        verify_term = 1.0 if verify_verdict == "confirmed" else 0.0
        incongruity_term = 0.0 if has_incongruity else 1.0

        total = (
            settings.REVIEW_QUEUE_W_CONFIDENCE * conf_term
            + settings.REVIEW_QUEUE_W_QUALITY * quality_term
            + settings.REVIEW_QUEUE_W_VERIFY * verify_term
            + settings.REVIEW_QUEUE_W_NO_INCONGRUITY * incongruity_term
        )
        return round(max(0.0, min(1.0, total)), 4)

    # ── Helpers ─────────────────────────────────────────────────────────

    def _make_entry(
        self, clip, bucket: str, confidence: float, reasons: list[str]
    ) -> ClipBucket:
        return ClipBucket(
            clip_id=clip.id,
            bucket=bucket,
            confidence=confidence,
            reasons=reasons,
            clip_index=clip.clip_index,
            duration=clip.duration,
            ai_emotion=clip.predicted_emotion,
            quality_score=clip.quality_score,
        )

    def _fetch_clips(self) -> list:
        """Lazy import to avoid circular import at module load time."""
        from backend.database.models import Clip, Video

        query = self.session.query(Clip)
        if self.video_id:
            query = query.filter(Clip.video_id == self.video_id)
        else:
            query = query.join(Video).order_by(Video.created_at.desc(), Clip.start_time.asc())
        return query.order_by(Clip.start_time.asc()).all()

    def _default_log_path(self) -> Path:
        from backend.config import BASE_DIR
        log_dir = BASE_DIR / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        return log_dir / "review_queue_agent.jsonl"

    def _write_log(self, result: AgentResult) -> None:
        try:
            with self.log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(result.to_log_dict(), ensure_ascii=False) + "\n")
        except OSError:
            # Logging is best-effort; never fail the agent run on I/O error.
            pass


# ── Convenience function for ad-hoc use ──────────────────────────────────

def run_agent(session, video_id: str | None = None) -> AgentResult:
    """One-liner wrapper. Prefer using ReviewQueueAgent directly in tests."""
    return ReviewQueueAgent(session, video_id=video_id).run()