"""
Emotion Data Studio — AI Auto-Cut integration test
==================================================
Test doc lap theo pattern docs/09_vertex_ai_integration.md:
  1. Vertex AI config (credentials, project, location)
  2. Vertex AI text gen (smoke test)
  3. Vertex AI multimodal gen (image -> text)
  4. AIVideoSegmenter.is_configured()

Chay:
  python scripts/test_ai_autocut_integration.py
  python scripts/test_ai_autocut_integration.py --with-video path/to/short.mp4
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

# Cho phép chạy từ thư mục gốc repo
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def banner(text: str) -> None:
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70)


def section(text: str) -> None:
    print(f"\n--- {text} ---")


def _set_utf8_stdout() -> None:
    """Fix Unicode print loi tren Windows cp1252."""
    try:
        import sys
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def test_vertex_config() -> dict | None:
    banner("TEST 1/4: Vertex AI config")
    try:
        from backend.services.gemini_auto_labeler import is_vertex_configured
        ok, msg = is_vertex_configured()
        print(f"Configured: {ok}")
        print(f"Message: {msg}")
        if not ok:
            print("\n[HUONG DAN]")
            print("  1. Dat GOOGLE_APPLICATION_CREDENTIALS trong .env hoac env:")
            print("     GOOGLE_APPLICATION_CREDENTIALS=d:/Hai/study/DeepLerning/BCDA/"
                  "tools/emotion-data-studio/aura-social-vn-e7a147284c33.json")
            print("  2. Hoac set trong Settings UI (Path tab).")
            return None
        return {"ok": True, "msg": msg}
    except Exception as exc:
        print(f"FAIL: {exc}")
        traceback.print_exc()
        return None


def test_text_gen() -> bool:
    banner("TEST 2/4: Vertex AI text generation (smoke)")
    try:
        from backend.services.gemini_auto_labeler import get_genai_client
        client, project, location = get_genai_client(force_vertex=True)
        print(f"project={project}, location={location}")
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[{"role": "user", "parts": [{"text": "Tra loi 1 cau tieng Viet: Xin chao"}]}],
            config={"temperature": 0.2, "max_output_tokens": 64},
        )
        text = getattr(response, "text", "") or ""
        if not text:
            # Fallback: dig into candidates
            if response.candidates:
                content = response.candidates[0].content
                for p in content.parts:
                    if getattr(p, "text", ""):
                        text = p.text
                        break
        print(f"Response: {text!r}")
        return bool(text.strip())
    except Exception as exc:
        print(f"FAIL: {exc}")
        traceback.print_exc()
        return False


def test_multimodal_gen() -> bool:
    banner("TEST 3/4: Vertex AI multimodal (image -> text)")
    try:
        from backend.services.gemini_auto_labeler import get_genai_client
        import base64
        import io

        # Tao PNG 4x4 mau do bang PIL/Pillow neu co, fallback bytes
        try:
            from PIL import Image  # type: ignore
            buf = io.BytesIO()
            Image.new("RGB", (4, 4), color=(220, 20, 20)).save(buf, format="PNG")
            png_bytes = buf.getvalue()
        except Exception:
            # PNG 4x4 mau do toi gian (CHUAN PNG)
            png_bytes = (
                b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x04"
                b"\x00\x00\x00\x04\x08\x02\x00\x00\x00\x2e\x9c\x6d\xf3"
                b"\x00\x00\x00\x16IDATx\x9cc\xfc\xcf\xc0\xf0\x9f\x81"
                b"\x81\x09\x80\x18\x80\x18\x80\x18\x80\x18\x80\x18\x80"
                b"\x00\x00\x00\xff\xff\x03\x00\x06\x05\x02\xfd\xa3\x9b"
                b"\x60\x00\x00\x00\x00IEND\xaeB`\x82"
            )
        img_b64 = base64.b64encode(png_bytes).decode("utf-8")

        client, _, _ = get_genai_client(force_vertex=True)
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[{
                "role": "user",
                "parts": [
                    {"inline_data": {"mime_type": "image/png", "data": img_b64}},
                    {"text": "Mo ta hinh nay bang 1 cau tieng Viet."},
                ],
            }],
            config={"temperature": 0.2, "max_output_tokens": 128},
        )
        text = getattr(response, "text", "") or ""
        if not text and response.candidates:
            for p in response.candidates[0].content.parts:
                if getattr(p, "text", ""):
                    text = p.text
                    break
        print(f"Response: {text!r}")
        return bool(text.strip())
    except Exception as exc:
        print(f"FAIL: {exc}")
        traceback.print_exc()
        return False


def test_ai_segmenter_config() -> bool:
    banner("TEST 4/4: AIVideoSegmenter.is_configured()")
    try:
        from backend.services.ai_video_segmenter import AIVideoSegmenter
        segmenter = AIVideoSegmenter()
        ok, msg = segmenter.is_configured()
        print(f"AI AutoCut ready: {ok}")
        print(f"Message: {msg}")
        return ok
    except Exception as exc:
        print(f"FAIL: {exc}")
        traceback.print_exc()
        return False


def test_full_autocut(video_path: str) -> bool:
    banner("BONUS: Full AI Auto-Cut (Gemini scan + FFmpeg cut)")
    try:
        from backend.services.ai_video_segmenter import AIVideoSegmenter
        segmenter = AIVideoSegmenter()
        result = segmenter.cut_video(video_path=video_path, video_id="test_integration")
        print(f"Duration: {result.video_duration:.1f}s")
        print(f"Segments: {result.total_segments}")
        print(f"Cost: ${result.total_cost_usd:.4f}")
        for idx, c in enumerate(result.clips[:3]):
            print(f"  [{idx}] {c.start_time:.1f}s-{c.end_time:.1f}s "
                  f"emotion={c.emotion} intensity={c.intensity:.2f} "
                  f"-> {Path(c.clip_path).name}")
        return True
    except Exception as exc:
        print(f"FAIL: {exc}")
        traceback.print_exc()
        return False


def main() -> int:
    _set_utf8_stdout()
    parser = argparse.ArgumentParser()
    parser.add_argument("--with-video", help="Test full AI Auto-Cut với 1 file video")
    args = parser.parse_args()

    print("Emotion Data Studio — AI AutoCut Integration Test")
    print(f"Project root: {ROOT}")

    cfg = test_vertex_config()
    if not cfg:
        return 1

    results = {
        "vertex_config": True,
        "text_gen": test_text_gen(),
        "multimodal": test_multimodal_gen(),
        "ai_segmenter": test_ai_segmenter_config(),
    }
    if args.with_video:
        results["full_autocut"] = test_full_autocut(args.with_video)

    banner("SUMMARY")
    for name, ok in results.items():
        status = "✅ PASS" if ok else "❌ FAIL"
        print(f"  {status}  {name}")

    return 0 if all(results.values()) else 2


if __name__ == "__main__":
    sys.exit(main())
