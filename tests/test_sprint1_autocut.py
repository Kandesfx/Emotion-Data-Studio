"""Smoke test Sprint 1 — hard filter + schema."""
import sys
from pathlib import Path

# Cho phep `python tests/sprint1_hard_filter.py` chay duoc tu root project.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from backend.services.ai_video_segmenter import (
    AIVideoSegmenter,
    _build_response_schema,
)


def main():
    seg = AIVideoSegmenter()
    test_segments = [
        # 1) ĐẠT: đủ mặt, đủ intensity, 1 người
        {
            "start_time": 10.0, "end_time": 17.0, "emotion": "angry",
            "intensity": 0.85, "face_coverage": 0.80, "frontal_ratio": 0.85,
            "people_count": 1, "speech_quality": "good", "has_transcript": True,
        },
        # 2) LOẠI: face_coverage 0.50 < 0.70
        {
            "start_time": 30.0, "end_time": 33.0, "emotion": "happy",
            "intensity": 0.75, "face_coverage": 0.50, "frontal_ratio": 0.80,
            "people_count": 1, "speech_quality": "good", "has_transcript": True,
        },
        # 3) LOẠI: 2 người
        {
            "start_time": 50.0, "end_time": 55.0, "emotion": "sad",
            "intensity": 0.92, "face_coverage": 0.75, "frontal_ratio": 0.78,
            "people_count": 2, "speech_quality": "good", "has_transcript": True,
        },
        # 4) LOẠI: intensity 0.65 < 0.80
        {
            "start_time": 70.0, "end_time": 73.0, "emotion": "angry",
            "intensity": 0.65, "face_coverage": 0.80, "frontal_ratio": 0.80,
            "people_count": 1, "speech_quality": "good", "has_transcript": True,
        },
        # 5) LOẠI: duration < 3s
        {
            "start_time": 90.0, "end_time": 91.5, "emotion": "fear",
            "intensity": 0.90, "face_coverage": 0.85, "frontal_ratio": 0.90,
            "people_count": 1, "speech_quality": "good", "has_transcript": True,
        },
    ]
    passed, rejected = seg._hard_filter_clip_quality(test_segments)
    print(f"Passed ({len(passed)}):")
    for s in passed:
        print(f"  [{s['start_time']:.1f}-{s['end_time']:.1f}] "
              f"emotion={s['emotion']} intensity={s['intensity']} "
              f"quality={s.get('quality_score')}")
    print(f"\nRejected ({len(rejected)}):")
    for s in rejected:
        print(f"  [{s['start_time']:.1f}-{s['end_time']:.1f}] "
              f"emotion={s['emotion']} -> {s['reject_reason']}")
    print(f"\nSchema available: {_build_response_schema() is not None}")

    # Sanity: assert exactly 1 passed
    assert len(passed) == 1, f"Expected 1 passed, got {len(passed)}"
    assert passed[0]["start_time"] == 10.0
    assert len(rejected) == 4
    reasons_concat = " ".join(s["reject_reason"] for s in rejected)
    for needed in ["face_coverage_low", "too_many_people", "intensity_low", "duration_too_short"]:
        assert needed in reasons_concat, f"Missing reject reason: {needed}"
    print("\nOK — all assertions passed.")


if __name__ == "__main__":
    main()
