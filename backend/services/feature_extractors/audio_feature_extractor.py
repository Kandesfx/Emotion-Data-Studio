"""Frame-level audio feature extractor — COVAREP-like 74-dimensional features.

Replicates the 74-dimensional acoustic feature vector used in CMU-MOSEI (COVAREP).
Since COVAREP binary is not available on Windows, we reconstruct equivalent features
using Librosa.

Feature groups (targeting 74 dims):
  - MFCC static + delta + delta-delta: 13 × 3 = 39
  - Chroma STFT: 12
  - Spectral features: 6 (ZCR, RMS, centroid, bandwidth, rolloff, flatness)
  - F0 + voiced flag: 2
  - HNR: 1
  - Tonnetz: 6
  - Spectral contrast: 7
  Total: 39 + 12 + 6 + 2 + 1 + 6 + 7 = 73 → pad to 74
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

TARGET_SR = 16000        # 16 kHz, matches COVAREP
HOP_LENGTH = 160          # 10 ms hop → 100 fps (matches COVAREP frame rate)
N_FFT = 512              # 32 ms window
MAX_SEQ_LEN = 50         # matches MMSA DataLoader

AUDIO_DIM = 74


class AudioFeatureExtractor:
    def __init__(self, output_dir: Path | None = None):
        self.output_dir = output_dir
        self.target_sr = TARGET_SR
        self.hop_length = HOP_LENGTH
        self.n_fft = N_FFT

    def extract_features(
        self,
        audio_path: str,
        clip_id: str,
        word_timestamps: list[dict] | None = None,
    ) -> dict[str, Any]:
        """Extract frame-level audio features aligned to word timestamps.

        Args:
            audio_path: Path to 16kHz mono .wav file.
            clip_id: Clip identifier, used for output filename.
            word_timestamps: List of {"word": str, "start": float, "end": float}
                             from Whisper transcription. If None, uniform sampling.

        Returns:
            {
                "features": np.ndarray (T, 74), float64,
                "feature_path": str,
                "shape": str,
                "num_frames": int,
                "duration_sec": float,
                "aligned": bool,
            }
        """
        import librosa

        audio_path = Path(audio_path)
        if not audio_path.exists():
            return self._empty_result(str(audio_path))

        try:
            y, sr = librosa.load(str(audio_path), sr=self.target_sr, mono=True)
        except Exception as exc:
            return self._empty_result(str(audio_path), warning=f"load_failed: {exc}")

        if len(y) == 0:
            return self._empty_result(str(audio_path), warning="empty_audio")

        # ── Extract all feature groups ─────────────────────────────────────────
        features_77 = self._extract_all_features(y, sr)

        # ── Resample to target length (MAX_SEQ_LEN = 50) ──────────────────────
        # librosa already returns frames at 100fps from hop_length=160 @ 16kHz
        # If word_timestamps provided: align per-word; else: uniform resample
        if word_timestamps:
            features_aligned = self._align_to_words(
                features_77, y, sr, word_timestamps
            )
            aligned = True
        else:
            features_aligned = self._uniform_resample(features_77, MAX_SEQ_LEN)
            aligned = False

        # Ensure exact (MAX_SEQ_LEN, 74) shape
        features_aligned = self._pad_or_truncate(features_aligned, MAX_SEQ_LEN)

        # Replace any -inf / nan with 0 (same as MMSA DataLoader convention)
        features_aligned = np.where(np.isfinite(features_aligned), features_aligned, 0.0)

        # Save to disk
        output_dir = self.output_dir or (Path(audio_path).parent / "features")
        output_dir.mkdir(parents=True, exist_ok=True)
        feature_path = output_dir / f"{clip_id}_audio_features.npy"
        np.save(str(feature_path), features_aligned.astype(np.float64))

        return {
            "features": features_aligned,
            "feature_path": str(feature_path.resolve()),
            "shape": f"({features_aligned.shape[0]}, {features_aligned.shape[1]})",
            "num_frames": features_aligned.shape[0],
            "duration_sec": float(len(y) / sr),
            "aligned": aligned,
        }

    # ── Feature extraction helpers ─────────────────────────────────────────────

    def _extract_all_features(self, y: np.ndarray, sr: int) -> np.ndarray:
        """Build 77-dim feature matrix from audio waveform."""
        import librosa

        parts: list[np.ndarray] = []

        # 1. MFCC (13) + delta (13) + delta-delta (13) = 39
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, n_fft=self.n_fft, hop_length=self.hop_length)
        mfcc_delta = librosa.feature.delta(mfcc, mode="interp")
        mfcc_delta2 = librosa.feature.delta(mfcc, order=2, mode="interp")
        parts.extend([mfcc, mfcc_delta, mfcc_delta2])   # (39, T)

        # 2. Chroma STFT (12)
        chroma = librosa.feature.chroma_stft(
            y=y, sr=sr, n_fft=self.n_fft, hop_length=self.hop_length
        )
        parts.append(chroma)   # (12, T)

        # 3. Spectral features (6)
        zcr = librosa.feature.zero_crossing_rate(y=y, hop_length=self.hop_length)
        rms = librosa.feature.rms(y=y, hop_length=self.hop_length)
        cent = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=self.n_fft, hop_length=self.hop_length)
        bw = librosa.feature.spectral_bandwidth(y=y, sr=sr, n_fft=self.n_fft, hop_length=self.hop_length)
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, n_fft=self.n_fft, hop_length=self.hop_length)
        flatness = librosa.feature.spectral_flatness(y=y, n_fft=self.n_fft, hop_length=self.hop_length)
        parts.extend([zcr, rms, cent, bw, rolloff, flatness])   # (6, T)

        # 4. F0 (fundamental frequency) + voiced flag (2)
        f0, voiced_flag, voiced_probs = librosa.pyin(
            y, fmin=librosa.note_to_hz("C1"), fmax=librosa.note_to_hz("C8"),
            sr=sr, hop_length=self.hop_length
        )
        f0 = np.nan_to_num(f0, nan=0.0)
        voiced_flag = voiced_flag.astype(np.float64)
        parts.extend([f0[np.newaxis, :], voiced_flag[np.newaxis, :]])   # (2, T)

        # 5. Harmonic-to-Noise Ratio (1)
        try:
            harmonic, _ = librosa.effects.hpss(y, hop_length=self.hop_length)
            hnr = np.mean(harmonic**2, axis=0) / (np.mean(y**2, axis=0) - np.mean(harmonic**2, axis=0) + 1e-10)
            hnr = np.nan_to_num(hnr, nan=0.0, posinf=0.0, neginf=0.0)
        except Exception:
            hnr = np.zeros(y.shape[0] // self.hop_length + 1)
        parts.append(hnr[np.newaxis, :])   # (1, T)

        # 6. Tonnetz (6)
        tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr, hop_length=self.hop_length)
        parts.append(tonnetz)   # (6, T)

        # 7. Spectral contrast (7)
        contrast = librosa.feature.spectral_contrast(
            y=y, sr=sr, n_fft=self.n_fft, hop_length=self.hop_length
        )
        parts.append(contrast)   # (7, T)

        # Stack → (77, T)
        stacked = np.vstack(parts)
        return stacked  # shape (77, T_audio)

    def _align_to_words(
        self,
        features: np.ndarray,
        y: np.ndarray,
        sr: int,
        word_timestamps: list[dict],
    ) -> np.ndarray:
        """Align audio frames to word timestamps via mean-pooling.

        Each word gets one feature vector = mean of audio frames within [word_start, word_end].
        Result has shape (len(word_timestamps), 77).
        """
        if not word_timestamps:
            return self._uniform_resample(features, MAX_SEQ_LEN)

        audio_frames_per_sec = sr / self.hop_length   # 100 fps
        num_audio_frames = features.shape[1]
        duration = num_audio_frames / audio_frames_per_sec

        aligned: list[np.ndarray] = []
        for w in word_timestamps:
            start_s = max(0.0, float(w.get("start", 0)))
            end_s = min(duration, float(w.get("end", start_s)))
            start_frame = int(start_s * audio_frames_per_sec)
            end_frame = int(end_s * audio_frames_per_sec)
            start_frame = min(start_frame, num_audio_frames)
            end_frame = min(end_frame, num_audio_frames)
            if end_frame <= start_frame:
                aligned.append(np.zeros(features.shape[0]))
            else:
                aligned.append(np.mean(features[:, start_frame:end_frame], axis=1))

        return np.stack(aligned, axis=0)  # (num_words, 77)

    def _uniform_resample(self, features: np.ndarray, target_len: int) -> np.ndarray:
        """Resample features to target_len via linear interpolation."""
        if features.shape[1] == target_len:
            return features
        indices = np.linspace(0, features.shape[1] - 1, target_len)
        return np.stack([np.interp(indices, np.arange(features.shape[1]), features[i]) for i in range(features.shape[0])], axis=0)

    def _pad_or_truncate(self, features: np.ndarray, max_len: int) -> np.ndarray:
        """Pad with zeros or truncate to exactly (max_len, 74)."""
        dim = features.shape[1] if features.ndim > 1 else AUDIO_DIM
        if features.shape[0] >= max_len:
            return features[:max_len, :dim]
        pad_len = max_len - features.shape[0]
        pad = np.zeros((pad_len, dim), dtype=features.dtype)
        return np.vstack([features[:, :dim], pad])

    def _empty_result(self, audio_path: str, warning: str = "") -> dict[str, Any]:
        zeros = np.zeros((MAX_SEQ_LEN, AUDIO_DIM), dtype=np.float64)
        return {
            "features": zeros,
            "feature_path": "",
            "shape": f"({MAX_SEQ_LEN}, {AUDIO_DIM})",
            "num_frames": MAX_SEQ_LEN,
            "duration_sec": 0.0,
            "aligned": False,
            "warning": warning,
        }
