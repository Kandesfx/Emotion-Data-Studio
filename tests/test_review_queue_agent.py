"""Test Sprint 4 — Self-tuning Review Queue Agent.

Chạy:  python tests/test_review_queue_agent.py

Test coverage:
- BUCKET_AUTO_REJECT: rejected_by_verify=True
- BUCKET_AUTO_REJECT: verify_verdict=wrong_emotion
- BUCKET_AUTO_REJECT: ai_confidence too low
- BUCKET_AUTO_APPROVE: high quality + verified + no incongruity
- BUCKET_AUTO_APPROVE confidence is in [0.0, 1.0]
- BUCKET_AUTO_APPROVE rejected when any one of 4 conditions fails
- BUCKET_NEEDS_REVIEW: default; verify_not_yet_run; incongruity only
- Buckets are sorted by confidence DESC within bucket
- Reasons list is non-empty for every clip
- AgentResult.to_log_dict() shape
- Integration test on 30 synthetic clips: bucket distribution within ±20%

Spec: docs/04_review_queue_agent_spec.md
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# Force UTF-8 cho Windows console
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

# Make repo importable when chạy script trực tiếp
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from types import SimpleNamespace

from backend.services.review_queue_agent import (  # noqa: E402
    BUCKET_AUTO_APPROVE,
    BUCKET_AUTO_REJECT,
    BUCKET_NEEDS_REVIEW,
    AgentResult,
    ClipBucket,
    ReviewQueueAgent,
)


PASSED = 0
FAILED = 0


def check(label: str, condition: bool, detail: str = "") -> None:
    global PASSED, FAILED
    if condition:
        print(f"  PASS  {label}")
        PASSED += 1
    else:
        print(f"  FAIL  {label}  {detail}")
        FAILED += 1


# ── Test fixtures ────────────────────────────────────────────────────────


def make_clip(
    clip_id: str,
    *,
    confidence: float = 0.6,
    quality_score: float = 0.7,
    predicted_emotion: str = "happy",
    verify_verdict: str | None = None,
    verify_status: str = "not_run",
    rejected_by_verify: bool = False,
    has_incongruity: bool = False,
    clip_index: int = 0,
    duration: float = 5.0,
) -> SimpleNamespace:
    """Build a fake Clip row (SimpleNamespace avoids SQLAlchemy session setup)."""
    return SimpleNamespace(
        id=clip_id,
        clip_index=clip_index,
        duration=duration,
        predicted_emotion=predicted_emotion,
        confidence=confidence,
        quality_score=quality_score,
        verify_verdict=verify_verdict,
        verify_status=verify_status,
        rejected_by_verify=rejected_by_verify,
        has_incongruity=has_incongruity,
    )


class FakeSession:
    """Minimal Session stub that returns a fixed list of clips."""

    def __init__(self, clips: list):
        self._clips = clips
        self._joined_video = False

    def query(self, model_cls):
        return _FakeQuery(self._clips)


class _FakeQuery:
    def __init__(self, clips: list):
        self._clips = clips
        self._video_id = None

    def filter(self, *args, **kwargs):
        return self

    def join(self, *args, **kwargs):
        return self

    def order_by(self, *args, **kwargs):
        return self

    def all(self) -> list:
        return list(self._clips)


# ── Unit tests ───────────────────────────────────────────────────────────


def test_auto_reject_by_verify_flag() -> None:
    """rejected_by_verify=True → auto-reject, regardless of other signals."""
    print("\n[Test 1] rejected_by_verify=True → BUCKET_AUTO_REJECT")
    clip = make_clip("c1", rejected_by_verify=True, confidence=0.99)
    agent = ReviewQueueAgent(FakeSession([clip]))
    entry = agent._classify(clip)
    check("bucket = auto_reject", entry.bucket == BUCKET_AUTO_REJECT)
    check("confidence >= 0.9", entry.confidence >= 0.9)
    check("reasons non-empty", len(entry.reasons) > 0)
    check("rejected_by_verify in reasons", any("verify_verdict" in r or "rejected_by_verify" in r for r in entry.reasons))


def test_auto_reject_by_wrong_emotion() -> None:
    """verify_verdict='wrong_emotion' → auto-reject."""
    print("\n[Test 2] verify_verdict=wrong_emotion → BUCKET_AUTO_REJECT")
    clip = make_clip("c2", verify_verdict="wrong_emotion", verify_status="rejected", confidence=0.9)
    agent = ReviewQueueAgent(FakeSession([clip]))
    entry = agent._classify(clip)
    check("bucket = auto_reject", entry.bucket == BUCKET_AUTO_REJECT)


def test_auto_reject_low_confidence() -> None:
    """ai_confidence below threshold → auto-reject."""
    print("\n[Test 3] low ai_confidence → BUCKET_AUTO_REJECT")
    clip = make_clip("c3", confidence=0.30, verify_status="passed", verify_verdict="confirmed")
    agent = ReviewQueueAgent(FakeSession([clip]))
    entry = agent._classify(clip)
    check("bucket = auto_reject", entry.bucket == BUCKET_AUTO_REJECT)
    check("reason mentions confidence", any("ai_confidence" in r for r in entry.reasons))


def test_auto_approve_happy_path() -> None:
    """All 4 conditions met → auto-approve with reasonable confidence."""
    print("\n[Test 4] all 4 conditions met → BUCKET_AUTO_APPROVE")
    clip = make_clip(
        "c4",
        confidence=0.95,
        quality_score=0.95,
        verify_status="passed",
        verify_verdict="confirmed",
        has_incongruity=False,
    )
    agent = ReviewQueueAgent(FakeSession([clip]))
    entry = agent._classify(clip)
    check("bucket = auto_approve", entry.bucket == BUCKET_AUTO_APPROVE)
    check("confidence in [0.0, 1.0]", 0.0 <= entry.confidence <= 1.0)
    check("confidence >= 0.5 (high signal)", entry.confidence >= 0.5)
    check("has 4 reasons (one per condition)", len(entry.reasons) >= 4)


def test_auto_approve_rejected_when_quality_too_low() -> None:
    """Quality below threshold breaks auto-approve even if everything else OK."""
    print("\n[Test 5] low quality_score breaks auto-approve")
    clip = make_clip(
        "c5",
        confidence=0.95,
        quality_score=0.50,
        verify_status="passed",
        verify_verdict="confirmed",
    )
    agent = ReviewQueueAgent(FakeSession([clip]))
    entry = agent._classify(clip)
    check("bucket = needs_review (NOT auto_approve)", entry.bucket == BUCKET_NEEDS_REVIEW)


def test_auto_approve_rejected_when_incongruity() -> None:
    """has_incongruity=True breaks auto-approve."""
    print("\n[Test 6] has_incongruity=True breaks auto-approve")
    clip = make_clip(
        "c6",
        confidence=0.95,
        quality_score=0.95,
        verify_status="passed",
        verify_verdict="confirmed",
        has_incongruity=True,
    )
    agent = ReviewQueueAgent(FakeSession([clip]))
    entry = agent._classify(clip)
    check("bucket = needs_review", entry.bucket == BUCKET_NEEDS_REVIEW)
    check("reason mentions incongruity", any("incongruity" in r for r in entry.reasons))


def test_auto_approve_rejected_when_verify_not_run() -> None:
    """verify_status='not_run' → cannot auto-approve."""
    print("\n[Test 7] verify_status=not_run → cannot auto-approve")
    clip = make_clip(
        "c7",
        confidence=0.95,
        quality_score=0.95,
        verify_status="not_run",
    )
    agent = ReviewQueueAgent(FakeSession([clip]))
    entry = agent._classify(clip)
    check("bucket = needs_review", entry.bucket == BUCKET_NEEDS_REVIEW)


def test_needs_review_default() -> None:
    """Clip with no special signals → needs_review with default reason."""
    print("\n[Test 8] default → needs_review")
    clip = make_clip("c8", confidence=0.6, quality_score=0.6, verify_status="not_run")
    agent = ReviewQueueAgent(FakeSession([clip]))
    entry = agent._classify(clip)
    check("bucket = needs_review", entry.bucket == BUCKET_NEEDS_REVIEW)
    check("reasons non-empty", len(entry.reasons) > 0)


def test_needs_review_incongruity() -> None:
    """has_incongruity + verify passed → still needs_review (incongruity reason)."""
    print("\n[Test 9] incongruity without reject → needs_review")
    clip = make_clip(
        "c9",
        confidence=0.95,
        quality_score=0.95,
        verify_status="passed",
        verify_verdict="confirmed",
        has_incongruity=True,
    )
    agent = ReviewQueueAgent(FakeSession([clip]))
    entry = agent._classify(clip)
    check("bucket = needs_review", entry.bucket == BUCKET_NEEDS_REVIEW)
    check("reason mentions incongruity", any("incongruity" in r for r in entry.reasons))


# ── Bucketing + sorting tests ───────────────────────────────────────────


def test_run_buckets_and_sorting() -> None:
    """run() returns three buckets, each sorted by confidence DESC."""
    print("\n[Test 10] run() buckets + sorting")
    clips = [
        # Auto-approve candidates with varying confidence
        make_clip("aa_high", confidence=0.99, quality_score=0.99, verify_verdict="confirmed", verify_status="passed"),
        make_clip("aa_mid", confidence=0.92, quality_score=0.90, verify_verdict="confirmed", verify_status="passed"),
        make_clip("aa_low", confidence=0.87, quality_score=0.87, verify_verdict="confirmed", verify_status="passed"),
        # Auto-reject candidates
        make_clip("ar_low_conf", confidence=0.20, verify_verdict="confirmed", verify_status="passed"),
        make_clip("ar_wrong", confidence=0.85, verify_verdict="wrong_emotion", verify_status="rejected"),
        # Needs-review
        make_clip("nr_default", confidence=0.5, verify_status="not_run"),
    ]
    agent = ReviewQueueAgent(FakeSession(clips))
    result = agent.run(log_to_file=False)

    check("3 auto_approve", len(result.auto_approve_candidates) == 3)
    check("2 auto_reject", len(result.auto_reject_candidates) == 2)
    check("1 needs_review", len(result.needs_human_review) == 1)
    check("n_clips = 6", result.summary["n_clips"] == 6)

    # Auto-approve sorted DESC by confidence
    aa_confs = [e.confidence for e in result.auto_approve_candidates]
    check("auto_approve sorted DESC", aa_confs == sorted(aa_confs, reverse=True))
    check("aa_high first", result.auto_approve_candidates[0].clip_id == "aa_high")


def test_empty_input() -> None:
    """Empty clip list returns empty buckets and zero ratios."""
    print("\n[Test 11] empty input")
    agent = ReviewQueueAgent(FakeSession([]))
    result = agent.run(log_to_file=False)
    check("no auto_approve", len(result.auto_approve_candidates) == 0)
    check("no auto_reject", len(result.auto_reject_candidates) == 0)
    check("no needs_review", len(result.needs_human_review) == 0)
    check("n_clips = 0", result.summary["n_clips"] == 0)
    check("auto_approve_ratio = 0", result.summary["auto_approve_ratio"] == 0.0)
    check("auto_reject_ratio = 0", result.summary["auto_reject_ratio"] == 0.0)


def test_reasons_nonempty_for_every_clip() -> None:
    """Spec §Done criteria: reasons list is non-empty for every clip."""
    print("\n[Test 12] every clip has ≥1 reason")
    clips = [
        make_clip("c_a", confidence=0.99, quality_score=0.99, verify_verdict="confirmed", verify_status="passed"),
        make_clip("c_b", confidence=0.20),
        make_clip("c_c", confidence=0.5),
        make_clip("c_d", verify_verdict="wrong_emotion", verify_status="rejected", confidence=0.9),
    ]
    agent = ReviewQueueAgent(FakeSession(clips))
    result = agent.run(log_to_file=False)
    all_entries = (
        result.auto_approve_candidates
        + result.auto_reject_candidates
        + result.needs_human_review
    )
    check("all 4 clips returned", len(all_entries) == 4)
    for entry in all_entries:
        check(
            f"  {entry.clip_id} has ≥1 reason",
            len(entry.reasons) > 0,
            detail=f"got {entry.reasons}",
        )


def test_confidence_in_unit_interval() -> None:
    """Confidence is always in [0.0, 1.0]."""
    print("\n[Test 13] all confidences in [0.0, 1.0]")
    clips = [
        make_clip(f"c{i}", confidence=conf, quality_score=conf, verify_verdict=vv, verify_status=vs)
        for i, (conf, vv, vs) in enumerate([
            (1.5, "confirmed", "passed"),
            (-0.1, "confirmed", "passed"),
            (0.5, "wrong_emotion", "rejected"),
            (0.85, "confirmed", "passed"),
        ])
    ]
    agent = ReviewQueueAgent(FakeSession(clips))
    result = agent.run(log_to_file=False)
    all_entries = (
        result.auto_approve_candidates
        + result.auto_reject_candidates
        + result.needs_human_review
    )
    for entry in all_entries:
        check(
            f"  {entry.clip_id} confidence in [0,1]",
            0.0 <= entry.confidence <= 1.0,
            detail=f"got {entry.confidence}",
        )


def test_agent_result_log_dict_shape() -> None:
    """AgentResult.to_log_dict() returns the expected keys."""
    print("\n[Test 14] AgentResult.to_log_dict shape")
    result = AgentResult(
        auto_approve_candidates=[],
        auto_reject_candidates=[],
        needs_human_review=[],
        summary={
            "n_clips": 0,
            "n_auto_approve": 0,
            "n_auto_reject": 0,
            "n_needs_review": 0,
            "auto_approve_ratio": 0.0,
            "auto_reject_ratio": 0.0,
        },
    )
    log = result.to_log_dict()
    expected_keys = {
        "ts", "n_clips", "n_auto_approve", "n_auto_reject", "n_needs_review",
        "auto_approve_ratio", "auto_reject_ratio",
    }
    check("log_dict has expected keys", set(log.keys()) == expected_keys)
    check("ts is ISO string", isinstance(log["ts"], str) and "T" in log["ts"])


# ── Integration test ────────────────────────────────────────────────────


def test_integration_30_clips_distribution() -> None:
    """30 synthetic clips → bucket distribution is non-zero and plausible."""
    print("\n[Test 15] integration: 30 clips")
    clips = []
    # 8 high-quality auto-approve candidates
    for i in range(8):
        clips.append(make_clip(
            f"aa_{i}", clip_index=i,
            confidence=0.95, quality_score=0.92,
            verify_verdict="confirmed", verify_status="passed",
        ))
    # 5 auto-reject by verify
    for i in range(5):
        clips.append(make_clip(
            f"ar1_{i}", clip_index=10 + i,
            confidence=0.7, verify_verdict="wrong_emotion", verify_status="rejected",
        ))
    # 4 auto-reject by low confidence
    for i in range(4):
        clips.append(make_clip(
            f"ar2_{i}", clip_index=20 + i,
            confidence=0.20, verify_status="not_run",
        ))
    # 13 needs-review (default)
    for i in range(13):
        clips.append(make_clip(
            f"nr_{i}", clip_index=30 + i,
            confidence=0.6, quality_score=0.6, verify_status="not_run",
        ))

    agent = ReviewQueueAgent(FakeSession(clips))
    result = agent.run(log_to_file=False)

    check("30 clips total", result.summary["n_clips"] == 30)
    check("8 auto_approve", result.summary["n_auto_approve"] == 8)
    check("9 auto_reject (5 verify + 4 low conf)", result.summary["n_auto_reject"] == 9)
    check("13 needs_review", result.summary["n_needs_review"] == 13)

    # Distribution check: within ±20% of predicted
    aa_ratio = result.summary["auto_approve_ratio"]
    ar_ratio = result.summary["auto_reject_ratio"]
    check("auto_approve ratio in [0.1, 0.5]", 0.1 <= aa_ratio <= 0.5, detail=f"got {aa_ratio:.2f}")
    check("auto_reject ratio in [0.1, 0.5]", 0.1 <= ar_ratio <= 0.5, detail=f"got {ar_ratio:.2f}")


# ── Test runner ──────────────────────────────────────────────────────────


def main() -> int:
    print("=" * 60)
    print("  Review Queue Agent — Sprint 4 tests")
    print("  Spec: docs/04_review_queue_agent_spec.md")
    print("=" * 60)

    test_auto_reject_by_verify_flag()
    test_auto_reject_by_wrong_emotion()
    test_auto_reject_low_confidence()
    test_auto_approve_happy_path()
    test_auto_approve_rejected_when_quality_too_low()
    test_auto_approve_rejected_when_incongruity()
    test_auto_approve_rejected_when_verify_not_run()
    test_needs_review_default()
    test_needs_review_incongruity()
    test_run_buckets_and_sorting()
    test_empty_input()
    test_reasons_nonempty_for_every_clip()
    test_confidence_in_unit_interval()
    test_agent_result_log_dict_shape()
    test_integration_30_clips_distribution()

    print()
    print("=" * 60)
    print(f"  RESULT:  {PASSED} passed,  {FAILED} failed")
    print("=" * 60)
    return 0 if FAILED == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
