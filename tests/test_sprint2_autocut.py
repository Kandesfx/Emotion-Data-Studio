"""Test Sprint 2 — Verify pass + Smart pre-cut + AutoCutResult serialize.

Chạy:  python tests/test_sprint2_autocut.py

Test coverage:
- _parse_verify_response: JSON parse OK, schema invalid, fallback
- combine_verdicts: 5 verdict branches + emotion lock
- AutoCutSegment: Sprint 2 fields default values
- AutoCutResult: serialize round-trip, stage counters
- _build_verify_response_schema: structure OK
- SmartSegmenter: min_face_coverage_in_scene filter
"""

from __future__ import annotations

import json
import sys
import os
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

from backend.config import settings  # noqa: E402
from backend.services.ai_video_segmenter import (  # noqa: E402
    AutoCutResult,
    AutoCutSegment,
    AIVideoSegmenter,
    _build_verify_response_schema,
    VERIFY_SYSTEM_PROMPT,
)
from backend.services.smart_segmenter import (  # noqa: E402
    FaceSample,
    SmartSegmenter,
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


def test_parse_verify_response() -> None:
    """Parse JSON response tu Verify pass."""
    print("\n[Test 1] _parse_verify_response")
    seg = AIVideoSegmenter()
    # Case 1: JSON hợp lệ, full schema
    r = seg._parse_verify_response(
        json.dumps({
            "verdict": "confirmed",
            "emotion": "happy",
            "intensity": 0.85,
            "face_coverage": 0.82,
            "frontal_ratio": 0.78,
            "speech_quality": "good",
            "stable": True,
            "speech_emotion_match": True,
            "reasoning": "Xác nhận rõ ràng",
        }),
        predicted_emotion="happy",
        predicted_intensity=0.7,
    )
    check("verdict = confirmed", r["verdict"] == "confirmed")
    check("intensity = 0.85", abs(r["intensity"] - 0.85) < 1e-6)
    check("stable = True", r["stable"] is True)

    # Case 2: code block markdown ```json ... ```
    r2 = seg._parse_verify_response(
        '```json\n{"verdict": "wrong_emotion", "emotion": "sad", "intensity": 0.5}\n```',
        predicted_emotion="happy",
        predicted_intensity=0.7,
    )
    check("code-block parse → wrong_emotion", r2["verdict"] == "wrong_emotion")
    check("code-block emotion = sad", r2["emotion"] == "sad")

    # Case 3: JSON invalid → fallback
    r3 = seg._parse_verify_response(
        "not json at all",
        predicted_emotion="angry",
        predicted_intensity=0.9,
    )
    check("invalid JSON → fallback stats_mismatch", r3["verdict"] == "stats_mismatch")
    check("fallback emotion = predicted", r3["emotion"] == "angry")
    check("fallback error key present", "error" in r3)

    # Case 4: verdict invalid enum → fallback
    r4 = seg._parse_verify_response(
        json.dumps({"verdict": "bogus", "emotion": "x", "intensity": 0.5}),
        predicted_emotion="neutral",
        predicted_intensity=0.5,
    )
    check("invalid verdict enum → stats_mismatch", r4["verdict"] == "stats_mismatch")


def test_combine_verdicts() -> None:
    """Combine Stage 2 + Verify verdicts theo 5 branches."""
    print("\n[Test 2] combine_verdicts")
    seg = AIVideoSegmenter()
    stage2 = {
        "emotion": "happy",
        "intensity": 0.80,
        "start_time": 0.0,
        "end_time": 5.0,
    }

    # Case A: confirmed → intensity boost 5%
    r = seg.combine_verdicts(stage2, {
        "verdict": "confirmed",
        "emotion": "happy",
        "intensity": 0.85,
        "stable": True,
        "speech_emotion_match": True,
        "reasoning": "ok",
    })
    check("confirmed → emotion kept", r["emotion"] == "happy")
    check("confirmed → intensity boosted (≥0.84)", r["intensity"] >= 0.84)
    check("confirmed → verify_status passed", r["verify_status"] == "passed")
    check("confirmed → no reject flag", not r.get("rejected_by_verify", False))

    # Case B: wrong_emotion → intensity *= 0.7
    r = seg.combine_verdicts(stage2, {
        "verdict": "wrong_emotion",
        "emotion": "sad",
        "intensity": 0.90,
    })
    check("wrong_emotion → emotion override", r["emotion"] == "sad")
    check("wrong_emotion → intensity dampened (≤0.63)", r["intensity"] <= 0.63)

    # Case C: unstable → rejected
    r = seg.combine_verdicts(stage2, {
        "verdict": "unstable",
        "emotion": "happy",
        "intensity": 0.5,
    })
    check("unstable → rejected_by_verify", r.get("rejected_by_verify") is True)
    check("unstable → reject_reason = emotion_unstable",
          r.get("reject_reason") == "emotion_unstable")
    check("unstable → verify_status rejected", r["verify_status"] == "rejected")

    # Case D: low_quality → rejected
    r = seg.combine_verdicts(stage2, {
        "verdict": "low_quality",
        "emotion": "happy",
        "intensity": 0.5,
    })
    check("low_quality → rejected", r.get("rejected_by_verify") is True)
    check("low_quality → reject_reason", r.get("reject_reason") == "low_quality_detected")

    # Case E: stats_mismatch → emotion = stage2, intensity avg
    r = seg.combine_verdicts(stage2, {
        "verdict": "stats_mismatch",
        "emotion": "happy",
        "intensity": 0.6,
    })
    check("stats_mismatch → emotion = stage2", r["emotion"] == "happy")
    check("stats_mismatch → intensity = avg",
          abs(r["intensity"] - 0.70) < 1e-6)


def test_emotion_lock() -> None:
    """Emotion flip lớn → reject ngay cả khi verdict = confirmed."""
    print("\n[Test 3] Emotion lock")
    seg = AIVideoSegmenter()
    # Force emotion lock on
    settings.EMOTION_LOCK_ENABLED = True
    settings.EMOTION_LOCK_MAX_FLIP_SCORE = 0.5

    stage2 = {"emotion": "happy", "intensity": 0.95}
    # Flip sang sad → flip_score = 1 - |0.95 - 0.20| = 0.25 < 0.5 → không reject
    r = seg.combine_verdicts(stage2, {
        "verdict": "confirmed",
        "emotion": "sad",
        "intensity": 0.20,
    })
    check("flip_score < threshold → keep", not r.get("rejected_by_verify", False))

    # Flip với intensity gần giống → flip_score cao → reject
    r = seg.combine_verdicts(stage2, {
        "verdict": "confirmed",
        "emotion": "angry",
        "intensity": 0.93,
    })
    check("flip_score > threshold → rejected", r.get("rejected_by_verify") is True)
    check("emotion lock → reject_reason = emotion_flip_detected",
          r.get("reject_reason") == "emotion_flip_detected")

    # Reset (settings la pydantic BaseSettings, khong can reset vi da set flag)


def test_verify_schema_structure() -> None:
    """Schema JSON cho Verify pass phải đúng shape."""
    print("\n[Test 4] _build_verify_response_schema")
    schema = _build_verify_response_schema()
    if schema is None:
        print("  SKIP  google-genai chưa import được")
        return
    check("schema type = OBJECT", schema["type"] == "OBJECT")
    check("schema có verdict field", "verdict" in schema["properties"])
    check("verdict enum có 5 giá trị",
          len(schema["properties"]["verdict"]["enum"]) == 5)
    check("verdict required", "verdict" in schema["required"])
    check("intensity required", "intensity" in schema["required"])
    check("emotion required", "emotion" in schema["required"])


def test_verify_prompt_quality() -> None:
    """Verify prompt phải có đủ 4 bước + ràng buộc quan trọng."""
    print("\n[Test 5] VERIFY_SYSTEM_PROMPT")
    check("có Bước 1", "Bước 1" in VERIFY_SYSTEM_PROMPT)
    check("có Bước 2", "Bước 2" in VERIFY_SYSTEM_PROMPT)
    check("có Bước 3", "Bước 3" in VERIFY_SYSTEM_PROMPT)
    check("có Bước 4", "Bước 4" in VERIFY_SYSTEM_PROMPT)
    check("có 70% rule", "70%" in VERIFY_SYSTEM_PROMPT)
    check("có 5 verdict", "confirmed" in VERIFY_SYSTEM_PROMPT
          and "wrong_emotion" in VERIFY_SYSTEM_PROMPT
          and "unstable" in VERIFY_SYSTEM_PROMPT
          and "low_quality" in VERIFY_SYSTEM_PROMPT
          and "stats_mismatch" in VERIFY_SYSTEM_PROMPT)


def test_autocut_segment_default() -> None:
    """AutoCutSegment phải có Sprint 2 fields với default đúng."""
    print("\n[Test 6] AutoCutSegment defaults")
    seg = AutoCutSegment(start_time=0, end_time=5, emotion="happy", intensity=0.8)
    check("verify_verdict default = ''", seg.verify_verdict == "")
    check("verify_status default = 'not_run'", seg.verify_status == "not_run")
    check("verify_reasoning default = ''", seg.verify_reasoning == "")
    check("rejected_by_verify default = False", seg.rejected_by_verify is False)
    check("reject_reason default = ''", seg.reject_reason == "")


def test_autocut_result_serialize() -> None:
    """AutoCutResult.to_dict phải round-trip đầy đủ Sprint 2 fields."""
    print("\n[Test 7] AutoCutResult to_dict")
    clips = [
        AutoCutSegment(start_time=0, end_time=5, emotion="happy", intensity=0.8),
        AutoCutSegment(
            start_time=6, end_time=12, emotion="sad", intensity=0.7,
            verify_verdict="confirmed", verify_status="passed",
            verify_reasoning="ok", clip_id="clip_001",
        ),
    ]
    rejected = [
        {"start_time": 13, "end_time": 18, "reason": "low_face_coverage"},
    ]
    verify_summary = {"total_verified": 2, "passed": 1, "rejected": 0, "errors": 1}
    result = AutoCutResult(
        video_id="vid_42",
        video_path="/tmp/v.mp4",
        video_duration=60.0,
        total_segments=2,
        total_cost_usd=0.12,
        clips=clips,
        source="ai_autocut",
        last_rejected=rejected,
        verify_summary=verify_summary,
        stage1_candidates=5,
        stage2_passed=2,
        stage3_cut=2,
        stage4_verified=2,
    )
    d = result.to_dict()
    check("serialized clips count = 2", len(d["clips"]) == 2)
    check("serialized last_rejected[0].reason = low_face_coverage",
          d["last_rejected"][0]["reason"] == "low_face_coverage")
    check("serialized verify_summary.passed = 1",
          d["verify_summary"]["passed"] == 1)
    check("serialized stage1_candidates = 5", d["stage1_candidates"] == 5)
    check("serialized stage2_passed = 2", d["stage2_passed"] == 2)
    check("serialized stage3_cut = 2", d["stage3_cut"] == 2)
    check("serialized stage4_verified = 2", d["stage4_verified"] == 2)
    check("clip[1] verify_verdict round-trip",
          d["clips"][1]["verify_verdict"] == "confirmed")

    # Round-trip JSON
    blob = json.dumps(d, ensure_ascii=False)
    d2 = json.loads(blob)
    check("JSON round-trip stage counters",
          d2["stage1_candidates"] == 5 and d2["stage4_verified"] == 2)


def test_smart_precut() -> None:
    """SmartSegmenter phải drop candidate có face_coverage < 0.70."""
    print("\n[Test 8] SmartSegmenter min_face_coverage_in_scene")
    seg = SmartSegmenter(min_face_coverage_in_scene=0.70, metadata_dir=Path("/tmp/_test_seg"))
    check("threshold mặc định = 0.70",
          abs(seg.min_face_coverage_in_scene - 0.70) < 1e-6)

    # Test helper: tạo FaceSample fixtures
    # 10 samples @ 2fps, đoạn 0-5s
    # 8/10 có face → coverage = 0.8 (giữ)
    samples_good = [
        FaceSample(timestamp=i * 0.5, has_face=(i < 8))
        for i in range(10)
    ]
    cov_good = seg._coverage_from_samples((0.0, 5.0), samples_good)
    check("_coverage_from_samples = 0.8 với 8/10 face", abs(cov_good - 0.8) < 1e-6)

    # 4/10 có face → coverage = 0.4 (bỏ)
    samples_bad = [
        FaceSample(timestamp=i * 0.5, has_face=(i < 4))
        for i in range(10)
    ]
    cov_bad = seg._coverage_from_samples((0.0, 5.0), samples_bad)
    check("_coverage_from_samples = 0.4 với 4/10 face", abs(cov_bad - 0.4) < 1e-6)


def test_settings_present() -> None:
    """Settings mới phải có trong config."""
    print("\n[Test 9] settings")
    check("AI_AUTOCUT_MIN_FACE_COVERAGE_IN_SCENE có giá trị",
          hasattr(settings, "AI_AUTOCUT_MIN_FACE_COVERAGE_IN_SCENE"))
    check("default value = 0.70",
          abs(settings.AI_AUTOCUT_MIN_FACE_COVERAGE_IN_SCENE - 0.70) < 1e-6)


def test_empty_verify_result() -> None:
    """empty verify khi clip không tồn tại."""
    print("\n[Test 10] _empty_verify_result")
    seg = AIVideoSegmenter()
    from pathlib import Path
    r = seg._empty_verify_result(Path("/nope/does/not/exist.mp4"), error="clip_not_found")
    check("error = clip_not_found", r["error"] == "clip_not_found")
    check("verdict fallback = stats_mismatch", r["verdict"] == "stats_mismatch")
    check("intensity = 0.0", r["intensity"] == 0.0)


def main() -> int:
    print("=" * 60)
    print("Sprint 2 — Unit Tests")
    print("=" * 60)
    test_parse_verify_response()
    test_combine_verdicts()
    test_emotion_lock()
    test_verify_schema_structure()
    test_verify_prompt_quality()
    test_autocut_segment_default()
    test_autocut_result_serialize()
    test_smart_precut()
    test_settings_present()
    test_empty_verify_result()

    print("\n" + "=" * 60)
    print(f"PASSED: {PASSED}")
    print(f"FAILED: {FAILED}")
    print("=" * 60)
    return 0 if FAILED == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
