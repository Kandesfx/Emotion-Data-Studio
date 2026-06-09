"""Quick dependency check script for Emotion Data Studio."""
import sys
import shutil

sys.stdout.reconfigure(encoding='utf-8')

print(f"Python: {sys.executable}")
print(f"Python version: {sys.version}")
print()

checks = {
    "openai-whisper": "whisper",
    "faster-whisper": "faster_whisper",
    "deepface": "deepface",
    "facenet-pytorch": "facenet_pytorch",
    "torch": "torch",
    "librosa": "librosa",
    "transformers": "transformers",
    "opencv": "cv2",
    "Pillow": "PIL",
    "numpy": "numpy",
    "soundfile": "soundfile",
    "scenedetect": "scenedetect",
}

for name, module in checks.items():
    try:
        m = __import__(module)
        ver = getattr(m, "__version__", "?")
        print(f"  OK: {name:25s} (version: {ver})")
    except Exception as e:
        print(f"  MISSING: {name:25s} ({e})")

# Torch CUDA check
try:
    import torch
    print(f"\n  Torch CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  CUDA device: {torch.cuda.get_device_name(0)}")
except Exception:
    pass

# FFmpeg
ffmpeg = shutil.which("ffmpeg")
print(f"\n  FFmpeg: {ffmpeg or 'NOT FOUND IN PATH'}")
ffprobe = shutil.which("ffprobe")
print(f"  FFprobe: {ffprobe or 'NOT FOUND IN PATH'}")

# Try loading model_manager and prewarming
print("\n--- Model Manager Test ---")
try:
    sys.path.insert(0, ".")
    from backend.ai_models.model_manager import model_manager
    for key in ["whisper", "deepface", "mtcnn"]:
        try:
            model_manager.load_model(key)
            print(f"  Model '{key}': LOADED OK")
        except Exception as e:
            print(f"  Model '{key}': FAILED - {e}")
except Exception as e:
    print(f"  Model Manager import failed: {e}")

# Test transcriber with a dummy call
print("\n--- Transcriber Test ---")
try:
    from backend.services.transcriber import SpeechTranscriber
    t = SpeechTranscriber()
    result = t.transcribe_audio_clip("nonexistent.wav", "test_clip")
    print(f"  Transcriber initialized OK (empty test: warning={result.get('warning', 'none')})")
except Exception as e:
    print(f"  Transcriber failed: {e}")

print("\nDone.")
