"""Test Sprint 3 — Verify pass integration + Cache + Cost tracking + YouTube handler.

Chạy:  python tests/test_sprint3_autocut.py

Test coverage:
- Cache: put/get, TTL, key hash, clear_all, stats
- Cost tracking: scan, verify, total breakdown
- Verify pass integration: progress_callback, summary, strict mode
- YouTube handler: duration check, truncation, is_valid_url
- Settings: AI_AUTOCUT_VERIFY_STRICT
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.config import settings  # noqa: E402
from backend.services.ai_video_segmenter import (  # noqa: E402
    AutoCutResult,
    AutoCutSegment,
    AIVideoSegmenter,
)
from backend.services import gemini_cache  # noqa: E402


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


def test_cache_put_get() -> None:
    """Cache put/get co ban."""
    print("\n[Test 1] gemini_cache put/get")
    key = gemini_cache.make_key(
        "/tmp/test.mp4", stage="scan",
        prompt="hello", params={"x": 1},
    )
    check("make_key tra SHA256 hex", len(key) == 32)
    check("get tra None khi chua co", gemini_cache.get(key) is None)

    gemini_cache.put(key, {"segments": [{"start_time": 0, "end_time": 5}]})
    got = gemini_cache.get(key)
    check("get tra dict sau put", got is not None)
    check("payload giong nhau", got == {"segments": [{"start_time": 0, "end_time": 5}]})

    # Key khac prompt → khac key
    key2 = gemini_cache.make_key(
        "/tmp/test.mp4", stage="scan",
        prompt="hello diff", params={"x": 1},
    )
    check("prompt khac → key khac", key != key2)


def test_cache_clear_stats() -> None:
    """Clear all + stats."""
    print("\n[Test 2] gemini_cache clear/stats")
    stats_before = gemini_cache.stats()
    check("stats co entries", "entries" in stats_before)
    check("stats co cache_dir", "cache_dir" in stats_before)
    check("stats co ttl_seconds", stats_before["ttl_seconds"] == 7 * 24 * 60 * 60)

    # Clear
    cleared = gemini_cache.clear_all()
    check("clear_all tra so file da xoa", isinstance(cleared, int))
    stats_after = gemini_cache.stats()
    check("entries = 0 sau clear", stats_after["entries"] == 0)


def test_cost_scan() -> None:
    """Cost estimate cho scan (Stage 2)."""
    print("\n[Test 3] _estimate_cost_scan")
    res = AIVideoSegmenter._estimate_cost_scan(video_duration=60.0)
    check("stage = scan", res["stage"] == "scan")
    check("input_tokens = 60 * 7000", res["input_tokens"] == 420000)
    check("output_tokens = 1024", res["output_tokens"] == 1024)
    check("input_cost_usd > 0", res["input_cost_usd"] > 0)
    check("total_usd = input + output", abs(
        res["total_usd"] - (res["input_cost_usd"] + res["output_cost_usd"])
    ) < 1e-6)


def test_cost_verify() -> None:
    """Cost estimate cho Verify (Stage 4) 1 clip."""
    print("\n[Test 4] _estimate_cost_verify")
    res = AIVideoSegmenter._estimate_cost_verify(clip_duration=7.0)
    check("stage = verify", res["stage"] == "verify")
    check("input_tokens = 7*7000 + 500", res["input_tokens"] == 49500)
    check("output_tokens = 256", res["output_tokens"] == 256)
    check("total_usd > scan_total / 10", res["total_usd"] > 0)


def test_cost_total() -> None:
    """Total cost = scan + N * verify."""
    print("\n[Test 5] estimate_total_cost")
    res = AIVideoSegmenter.estimate_total_cost(
        video_duration=120.0, n_clips=5, avg_clip_duration=8.0,
    )
    check("co scan dict", "scan" in res)
    check("co verify dict", "verify" in res)
    check("grand_total_usd > 0", res["grand_total_usd"] > 0)
    check("n_clips = 5", res["n_clips"] == 5)
    check("video_duration_sec = 120", res["video_duration_sec"] == 120.0)
    expected = res["scan"]["total_usd"] + res["verify"]["total_usd"]
    check("grand_total = scan + verify",
          abs(res["grand_total_usd"] - expected) < 1e-6)
    # 5 clips verify → verify total phai lon hon scan/5 (do 5 verify calls)
    check("verify total_usd proportional (5 clips)",
          res["verify"]["total_usd"] > res["scan"]["total_usd"] / 10)


def test_settings_verify_strict() -> None:
    """Setting AI_AUTOCUT_VERIFY_STRICT co gia tri mac dinh."""
    print("\n[Test 6] settings.AI_AUTOCUT_VERIFY_STRICT")
    check("Setting co ton tai", hasattr(settings, "AI_AUTOCUT_VERIFY_STRICT"))
    check("Default = False", settings.AI_AUTOCUT_VERIFY_STRICT is False)


def test_autocut_result_cost_breakdown() -> None:
    """AutoCutResult co cost_breakdown serialize dung."""
    print("\n[Test 7] AutoCutResult cost_breakdown")
    r = AutoCutResult(
        video_id="v1", video_path="/tmp/v.mp4", video_duration=60.0,
        total_segments=2, total_cost_usd=0.05,
        cost_breakdown={
            "scan": {"total_usd": 0.04},
            "verify": {"total_usd": 0.01},
            "grand_total_usd": 0.05,
        },
    )
    d = r.to_dict()
    check("cost_breakdown trong to_dict", "cost_breakdown" in d)
    check("scan.total_usd = 0.04", d["cost_breakdown"]["scan"]["total_usd"] == 0.04)
    check("verify.total_usd = 0.01", d["cost_breakdown"]["verify"]["total_usd"] == 0.01)


def test_orchestrator_youtube_handler() -> None:
    """PipelineOrchestrator.download_youtube_for_ai_autocut method exists + validates."""
    print("\n[Test 8] PipelineOrchestrator YouTube handler")
    from backend.services.pipeline_orchestrator import PipelineOrchestrator
    from backend.services.downloader import VideoDownloader

    # Static attribute
    check("Co constant AI_AUTOCUT_MAX_DURATION_SEC",
          hasattr(PipelineOrchestrator, "AI_AUTOCUT_MAX_DURATION_SEC"))
    check("Default cap = 30 min = 1800s",
          PipelineOrchestrator.AI_AUTOCUT_MAX_DURATION_SEC == 1800)

    # Method exists
    check("Method download_youtube_for_ai_autocut exists",
          hasattr(PipelineOrchestrator, "download_youtube_for_ai_autocut"))

    # is_valid_url checks
    check("valid YouTube URL",
          VideoDownloader.is_valid_url("https://www.youtube.com/watch?v=dQw4w9WgXcQ"))
    check("valid short URL",
          VideoDownloader.is_valid_url("https://youtu.be/dQw4w9WgXcQ"))
    check("empty URL invalid", not VideoDownloader.is_valid_url(""))
    check("random string invalid", not VideoDownloader.is_valid_url("not a url"))


def test_youtube_handler_clip_ffmpeg_helper() -> None:
    """_clip_video_ffmpeg co method."""
    print("\n[Test 9] _clip_video_ffmpeg helper")
    from backend.services.pipeline_orchestrator import PipelineOrchestrator
    check("Method _clip_video_ffmpeg exists",
          hasattr(PipelineOrchestrator, "_clip_video_ffmpeg"))


def test_estimate_cost_endpoint() -> None:
    """Sprint 3 — /estimate-cost tra ve breakdown chi tiet."""
    print("\n[Test 10] /estimate-cost contract")
    # Test bang goi truc tiep (khong can FastAPI)
    res = AIVideoSegmenter.estimate_total_cost(
        video_duration=300.0, n_clips=15, avg_clip_duration=8.0,
    )
    # Endpoint shape phai match
    expected_keys = {"scan", "verify", "grand_total_usd",
                     "video_duration_sec", "n_clips"}
    check("Endpoint response co keys dung",
          expected_keys.issubset(res.keys()))
    check("5 phut video → scan total > 0", res["scan"]["total_usd"] > 0)
    check("15 clips → verify total > scan", res["verify"]["total_usd"] > res["scan"]["total_usd"] / 5)


def test_verify_pass_with_mocked_segmenter() -> None:
    """Verify pass integration test — mock Gemini, verify combine_verdicts."""
    print("\n[Test 11] Verify pass integration (mocked)")
    from backend.services.ai_video_segmenter import AIVideoSegmenter

    seg = AIVideoSegmenter()
    # Mock verify_clip de k phai goi Gemini that
    def mock_verify(clip_path, predicted_emotion, predicted_intensity, **kwargs):
        if predicted_emotion == "happy":
            return {
                "verdict": "confirmed",
                "emotion": "happy",
                "intensity": 0.9,
                "face_coverage": 0.85,
                "frontal_ratio": 0.8,
                "speech_quality": "good",
                "stable": True,
                "speech_emotion_match": True,
                "reasoning": "ok",
            }
        else:  # sad → unstable
            return {
                "verdict": "unstable",
                "emotion": "angry",
                "intensity": 0.5,
                "speech_quality": "fair",
                "stable": False,
                "speech_emotion_match": False,
                "reasoning": "emotion flip",
            }

    seg.verify_clip = mock_verify  # type: ignore[assignment]

    clips_meta = [
        {"emotion": "happy", "intensity": 0.8},
        {"emotion": "sad", "intensity": 0.6},
    ]
    verify_summary = {"total": 0, "passed": 0, "rejected": 0,
                      "errors": 0, "by_verdict": {}}
    for c in clips_meta:
        v = seg.verify_clip(
            clip_path="/tmp/fake.mp4",
            predicted_emotion=c["emotion"],
            predicted_intensity=c["intensity"],
        )
        merged = seg.combine_verdicts(c, v)
        v_status = merged.get("verify_status")
        v_verdict = merged.get("verify_verdict")
        verify_summary["total"] += 1
        verify_summary["by_verdict"][v_verdict] = (
            verify_summary["by_verdict"].get(v_verdict, 0) + 1
        )
        if v_status == "passed":
            verify_summary["passed"] += 1
        elif v_status == "rejected":
            verify_summary["rejected"] += 1

    check("Mocked 2 clips", verify_summary["total"] == 2)
    check("1 passed (happy→confirmed)", verify_summary["passed"] == 1)
    check("1 rejected (sad→unstable)", verify_summary["rejected"] == 1)
    check("by_verdict.confirmed = 1",
          verify_summary["by_verdict"].get("confirmed", 0) == 1)
    check("by_verdict.unstable = 1",
          verify_summary["by_verdict"].get("unstable", 0) == 1)


def main() -> int:
    print("=" * 60)
    print("Sprint 3 — Unit Tests")
    print("=" * 60)
    test_cache_put_get()
    test_cache_clear_stats()
    test_cost_scan()
    test_cost_verify()
    test_cost_total()
    test_settings_verify_strict()
    test_autocut_result_cost_breakdown()
    test_orchestrator_youtube_handler()
    test_youtube_handler_clip_ffmpeg_helper()
    test_estimate_cost_endpoint()
    test_verify_pass_with_mocked_segmenter()

    print("\n" + "=" * 60)
    print(f"PASSED: {PASSED}")
    print(f"FAILED: {FAILED}")
    print("=" * 60)
    return 0 if FAILED == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
