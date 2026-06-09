"""Audio extraction and acoustic feature service."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict

import numpy as np

from backend.config import settings


class AudioExtractor:
    def __init__(self, output_dir: Path | None = None):
        self.output_dir = output_dir or (settings.DATA_DIR / "audio")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.target_sr = 16000

    def extract_audio_from_clip(self, clip_path: str, clip_id: str) -> Dict[str, Any]:
        clip = Path(clip_path)
        if not clip.exists():
            raise FileNotFoundError(f"Clip not found: {clip_path}")

        ffmpeg_path = settings.FFMPEG_PATH or "ffmpeg"
        if shutil.which(ffmpeg_path) is None and not Path(ffmpeg_path).exists():
            raise FileNotFoundError(f"FFmpeg not found: {ffmpeg_path}")

        audio_path = self.output_dir / f"{clip_id}.wav"
        cmd = [
            ffmpeg_path,
            "-y",
            "-i", str(clip),
            "-vn",
            "-acodec", "pcm_s16le",
            "-ar", str(self.target_sr),
            "-ac", "1",
            str(audio_path),
        ]
        startupinfo = None
        if os.name == "nt":
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, startupinfo=startupinfo)
        if result.returncode != 0:
            raise RuntimeError(result.stderr.decode("utf-8", errors="ignore")[-1200:])

        features = self._extract_features(audio_path)
        features["audio_path"] = str(audio_path.resolve())
        return features

    def _extract_features(self, audio_path: Path) -> Dict[str, Any]:
        try:
            import librosa
            y, sr = librosa.load(str(audio_path), sr=self.target_sr, mono=True)
            if len(y) == 0:
                return self._empty_features(str(audio_path))
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
            rms = librosa.feature.rms(y=y)
            zcr = librosa.feature.zero_crossing_rate(y=y)
            tempo = librosa.feature.tempo(y=y, sr=sr)
            return {
                "sample_rate": sr,
                "duration_sec": float(len(y) / sr),
                "mfccs_mean": np.mean(mfcc.T, axis=0).astype(float).tolist(),
                "audio_clarity": float(np.mean(rms)),
                "zero_crossing_rate": float(np.mean(zcr)),
                "tempo": float(tempo[0]) if len(tempo) else 0.0,
                "has_speech_energy": bool(float(np.mean(rms)) > 0.003),
            }
        except Exception as exc:
            data = self._empty_features(str(audio_path))
            data["warning"] = f"audio_features_failed: {exc}"
            return data

    def _empty_features(self, audio_path: str) -> Dict[str, Any]:
        return {
            "audio_path": audio_path,
            "sample_rate": self.target_sr,
            "duration_sec": 0.0,
            "mfccs_mean": [],
            "audio_clarity": 0.0,
            "zero_crossing_rate": 0.0,
            "tempo": 0.0,
            "has_speech_energy": False,
        }
