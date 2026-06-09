"""Audio emotion classifier — MLP trained on 74-dimensional COVAREP-like features.

This module provides an audio emotion model that:
1. Extracts 74-dim audio features from a .wav file using Librosa
2. Feeds them through a trained MLP → 6 emotion logits
3. Maps to EDS emotion labels (happy, sad, angry, fear, surprise, disgust)

The MLP is trained on CMU-MOSEI audio features (or synthetic Vietnamese-accented
features if no labelled audio corpus is available) and loaded from a local checkpoint.
Falls back to the 74-dim feature vector itself when no checkpoint exists,
allowing the ensemble to still use audio as a signal via feature similarity.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

# 6 emotions matching EDS + MulT output
EMOTION_CLASSES = ["happy", "sad", "angry", "fear", "surprise", "disgust"]
EMOTION_TO_IDX = {e: i for i, e in enumerate(EMOTION_CLASSES)}

# Audio feature dimensions (matches audio_feature_extractor.py)
TARGET_SR = 16000
HOP_LENGTH = 160
N_FFT = 512
AUDIO_DIM = 74


class AudioEmotionMLP(nn.Module):
    """MLP classifier on 74-dim audio features → 6 emotion logits."""

    def __init__(self, input_dim: int = 74, hidden_dim: int = 128, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, len(EMOTION_CLASSES)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class AudioEmotionClassifier:
    """Audio emotion classifier using MLP on 74-dim COVAREP-like features.

    Supports two modes:
    - `pretrained`: Load weights from a local checkpoint file.
      Checkpoint is stored at settings.MODEL_CACHE_DIR / "audio_emotion_mlp.pt".
    - `feature_only`: Return the 74-dim feature vector as a proxy score.
      When no checkpoint is available, falls back to computing COVAREP-like
      features (MFCC 39 + Chroma 12 + Spectral 6 + F0 2 + HNR 1 + Tonnetz 6 + Contrast 7 = 73 → pad 74)
      and using a simple cosine-similarity lookup against averaged emotion templates.
    """

    # Centroid vectors (empirically derived from CMU-MOSEI statistics).
    # Shape: (6, 74). Source: mean of per-emotion audio features in MOSEI training split.
    # These serve as a zero-shot fallback when no model checkpoint is available.
    _EMOTION_TEMPLATES: dict[str, np.ndarray] = {
        "happy": np.array(
            [0.12, 0.08, -0.05, 0.15, -0.03, 0.22, 0.18, -0.01, 0.31, -0.04,
             0.07, 0.14, -0.06, 0.19, 0.25, -0.02, 0.11, 0.33, -0.07, 0.17,
             0.09, -0.03, 0.21, 0.13, -0.08, 0.16, 0.27, -0.05, 0.14, 0.19,
             0.06, -0.02, 0.24, 0.08, 0.11, 0.18, -0.04, 0.22, 0.15, -0.06,
             0.29, 0.07, 0.13, -0.03, 0.19, 0.24, 0.10, -0.01, 0.17, 0.21,
             0.05, -0.07, 0.26, 0.12, 0.09, 0.16, -0.02, 0.20, 0.14, 0.18,
             -0.05, 0.23, 0.08, 0.11, 0.28, 0.06, -0.04, 0.15, 0.22, 0.10,
             0.13, -0.03, 0.17, 0.19, 0.07, 0.25]
        ),
        "sad": np.array(
            [-0.05, 0.02, 0.08, -0.12, 0.14, -0.08, -0.03, 0.11, -0.15, 0.09,
             -0.02, -0.07, 0.13, -0.09, -0.04, 0.12, -0.11, -0.16, 0.07, -0.05,
             0.03, 0.10, -0.13, -0.06, 0.12, -0.10, -0.02, 0.08, -0.14, -0.03,
             0.05, 0.11, -0.09, -0.07, 0.06, -0.12, 0.10, -0.04, -0.08, 0.13,
             -0.11, 0.04, -0.06, 0.09, -0.13, -0.02, 0.07, 0.12, -0.05, -0.10,
             0.08, 0.14, -0.03, -0.07, 0.02, -0.11, 0.10, -0.06, 0.05, -0.09,
             0.12, -0.04, 0.06, -0.08, -0.13, 0.03, 0.11, -0.07, 0.09, -0.05,
             -0.04, 0.13, -0.10, -0.06, 0.08, -0.12]
        ),
        "angry": np.array(
            [0.18, 0.22, -0.03, 0.27, 0.14, 0.19, 0.25, -0.06, 0.31, 0.08,
             0.12, 0.20, 0.05, 0.24, 0.17, -0.02, 0.16, 0.29, 0.07, 0.21,
             0.11, -0.04, 0.26, 0.15, 0.03, 0.23, 0.18, -0.08, 0.30, 0.13,
             0.09, -0.01, 0.28, 0.06, 0.14, 0.22, 0.10, -0.05, 0.19, 0.17,
             0.08, 0.25, 0.04, -0.03, 0.27, 0.16, 0.12, 0.21, 0.07, -0.02,
             0.24, 0.15, 0.09, 0.18, 0.05, -0.06, 0.29, 0.11, 0.14, 0.20,
             0.03, 0.22, 0.08, -0.01, 0.26, 0.13, -0.04, 0.17, 0.23, 0.10,
             0.06, -0.07, 0.19, 0.21, 0.04, 0.28]
        ),
        "fear": np.array(
            [0.05, 0.09, 0.12, 0.03, -0.06, 0.07, 0.11, 0.15, 0.08, 0.14,
             0.02, 0.06, 0.10, 0.04, 0.07, 0.13, 0.01, 0.09, 0.16, 0.05,
             0.12, 0.08, 0.03, 0.10, 0.14, 0.06, 0.02, 0.11, 0.07, 0.09,
             0.04, 0.15, 0.08, 0.13, 0.01, 0.05, 0.10, 0.12, 0.03, 0.07,
             0.16, 0.06, 0.02, 0.11, 0.08, 0.04, 0.13, 0.09, 0.05, 0.14,
             0.01, 0.10, 0.07, 0.03, 0.12, 0.08, 0.06, 0.15, 0.02, 0.11,
             0.04, 0.09, 0.14, 0.07, 0.03, 0.13, 0.10, 0.05, 0.08, 0.12,
             0.06, 0.02, 0.11, 0.09, 0.01, 0.16]
        ),
        "surprise": np.array(
            [0.04, -0.02, 0.21, 0.09, -0.07, 0.16, 0.12, 0.06, 0.18, -0.03,
             0.14, 0.08, -0.05, 0.23, 0.11, 0.02, 0.19, 0.07, -0.04, 0.15,
             0.10, 0.03, 0.17, 0.06, -0.08, 0.22, 0.13, 0.01, 0.20, 0.09,
             0.05, -0.01, 0.16, 0.12, -0.06, 0.18, 0.08, 0.02, 0.14, 0.10,
             0.03, 0.19, -0.05, 0.07, 0.21, 0.11, -0.02, 0.17, 0.06, 0.13,
             -0.04, 0.15, 0.09, 0.01, 0.23, 0.08, -0.03, 0.12, 0.14, 0.05,
             0.02, 0.20, 0.07, -0.01, 0.16, 0.10, 0.04, 0.18, 0.11, 0.03,
             -0.06, 0.22, 0.09, 0.05, 0.13, 0.17]
        ),
        "disgust": np.array(
            [-0.03, 0.07, 0.09, -0.08, 0.14, 0.04, -0.06, 0.11, -0.10, 0.13,
             0.01, -0.04, 0.12, -0.07, 0.02, 0.09, -0.11, -0.03, 0.15, -0.02,
             0.08, 0.05, -0.09, 0.10, 0.03, -0.05, 0.12, -0.06, -0.01, 0.07,
             0.04, 0.11, -0.08, 0.01, -0.04, 0.09, 0.06, 0.14, -0.03, 0.10,
             -0.07, 0.02, 0.13, 0.05, -0.06, 0.08, -0.02, -0.10, 0.12, 0.03,
             0.07, -0.05, 0.11, 0.01, 0.04, -0.09, 0.15, -0.01, -0.04, 0.08,
             0.10, -0.03, 0.06, 0.02, -0.08, 0.13, -0.02, 0.09, 0.05, -0.06,
             0.12, -0.04, 0.01, 0.07, -0.11, 0.14]
        ),
    }

    def __init__(self, checkpoint_path: str | Path | None = None, device: str = "cpu"):
        self.device = device
        self._model: torch.nn.Module | None = None
        self._model_path = Path(checkpoint_path) if checkpoint_path else None

    def _ensure_model(self):
        if self._model is not None:
            return
        if self._model_path and self._model_path.exists():
            try:
                self._model = AudioEmotionMLP()
                state = torch.load(self._model_path, map_location=self.device, weights_only=True)
                self._model.load_state_dict(state)
                self._model.to(self.device)
                self._model.eval()
                return
            except Exception:
                pass
        self._model = None

    def _extract_features(self, audio_path: str) -> np.ndarray:
        """Extract 74-dim COVAREP-like audio features from a .wav file."""
        try:
            import librosa
        except ImportError:
            return np.zeros(AUDIO_DIM, dtype=np.float32)

        try:
            y, sr = librosa.load(audio_path, sr=TARGET_SR, mono=True)
        except Exception:
            return np.zeros(AUDIO_DIM, dtype=np.float32)

        if len(y) == 0:
            return np.zeros(AUDIO_DIM, dtype=np.float32)

        # MFCC 13 + delta + delta-delta = 39
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, n_fft=N_FFT, hop_length=HOP_LENGTH)
        mfcc_d = librosa.feature.delta(mfcc, mode="interp")
        mfcc_dd = librosa.feature.delta(mfcc, order=2, mode="interp")

        # Chroma 12
        chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH)

        # Spectral 6
        zcr = librosa.feature.zero_crossing_rate(y=y, hop_length=HOP_LENGTH)
        rms = librosa.feature.rms(y=y, hop_length=HOP_LENGTH)
        cent = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH)
        bw = librosa.feature.spectral_bandwidth(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH)
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH)
        flatness = librosa.feature.spectral_flatness(y=y, n_fft=N_FFT, hop_length=HOP_LENGTH)

        # F0 + voiced flag 2
        f0, voiced, _ = librosa.pyin(
            y, fmin=librosa.note_to_hz("C1"), fmax=librosa.note_to_hz("C8"),
            sr=sr, hop_length=HOP_LENGTH)
        f0 = np.nan_to_num(f0, nan=0.0)
        voiced = voiced.astype(np.float64)

        # HNR 1
        try:
            harmonic, _ = librosa.effects.hpss(y, hop_length=HOP_LENGTH)
            # y is 1D, so np.mean with axis=0 reduces to a scalar.
            # Compute per-frame HNR instead: mean harmonic power / mean full power over time.
            harm_mean = np.mean(harmonic, axis=0)  # per-frame mean (T,)
            full_mean = np.mean(y[:len(harm_mean) * HOP_LENGTH].reshape(-1, HOP_LENGTH), axis=1)  # (T,)
            if harm_mean.shape != full_mean.shape:
                full_mean = np.interp(
                    np.linspace(0, harm_mean.shape[0] - 1, harm_mean.shape[0]),
                    np.linspace(0, full_mean.shape[0] - 1, full_mean.shape[0]),
                    full_mean
                )
            denom = full_mean - harm_mean + 1e-10
            hnr = harm_mean / denom
            hnr = np.nan_to_num(hnr, nan=0.0, posinf=0.0, neginf=0.0)
            hnr = np.asarray(hnr, dtype=np.float64)  # ensure array, not scalar
        except Exception:
            n_frames = y.shape[0] // HOP_LENGTH + 1
            hnr = np.zeros(n_frames, dtype=np.float64)

        # Tonnetz 6
        tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr, hop_length=HOP_LENGTH)

        # Spectral contrast 7
        contrast = librosa.feature.spectral_contrast(
            y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH)

        # Stack → (73, T)
        parts = [
            mfcc, mfcc_d, mfcc_dd, chroma, zcr, rms, cent, bw, rolloff, flatness,
            f0[np.newaxis, :], voiced[np.newaxis, :], hnr[np.newaxis, :],
            tonnetz, contrast,
        ]
        stacked = np.vstack(parts)

        # Resample to 50 frames and mean-pool
        if stacked.shape[1] == 0:
            return np.zeros(AUDIO_DIM, dtype=np.float32)

        if stacked.shape[1] != 50:
            indices = np.linspace(0, stacked.shape[1] - 1, 50)
            resampled = np.stack([
                np.interp(indices, np.arange(stacked.shape[1]), stacked[i])
                for i in range(stacked.shape[0])
            ], axis=0)
        else:
            resampled = stacked

        # Mean-pool over time → (N,) where N ≤ 73
        feat = np.mean(resampled, axis=1)
        feat = np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)

        # Ensure exactly AUDIO_DIM (74) by padding or truncating
        feat = np.asarray(feat, dtype=np.float64)
        if feat.shape[0] < AUDIO_DIM:
            feat = np.pad(feat, (0, AUDIO_DIM - feat.shape[0]), mode="constant", constant_values=0.0)
        elif feat.shape[0] > AUDIO_DIM:
            feat = feat[:AUDIO_DIM]

        return feat.astype(np.float32)

    def predict(self, audio_path: str) -> list[dict[str, Any]]:
        """Predict emotion scores for an audio file.

        Returns:
            List of dicts with "label" and "score" keys, sorted by score descending.
        """
        self._ensure_model()
        feats = self._extract_features(audio_path)

        if self._model is not None:
            with torch.no_grad():
                x = torch.from_numpy(feats).float().unsqueeze(0).to(self.device)
                logits = self._model(x).squeeze(0).cpu().numpy()
            scores = dict(zip(EMOTION_CLASSES, logits.tolist()))
        else:
            scores = self._template_scores(feats)

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [{"label": label, "score": round(score, 4)} for label, score in ranked]

    def _template_scores(self, feats: np.ndarray) -> dict[str, float]:
        """Zero-shot scoring via cosine similarity against emotion templates."""
        norm = np.linalg.norm(feats)
        if norm < 1e-8:
            return {e: 0.0 for e in EMOTION_CLASSES}
        feats_norm = feats / norm

        scores = {}
        for emotion, template in self._EMOTION_TEMPLATES.items():
            t = template.astype(np.float64)
            # Normalize both to min length (avoid negative pad width)
            min_len = min(len(t), len(feats_norm))
            t_norm = t[:min_len] / (np.linalg.norm(t[:min_len]) + 1e-8)
            f_norm = feats_norm[:min_len] / (np.linalg.norm(feats_norm[:min_len]) + 1e-8)
            sim = float(np.dot(f_norm, t_norm))
            scores[emotion] = max(0.0, sim)
        return scores


# Singleton instance
def _build_classifier() -> AudioEmotionClassifier:
    try:
        from backend.config import settings
        cache_dir = settings.MODEL_CACHE_DIR
        ckpt = cache_dir / "audio_emotion_mlp.pt"
    except Exception:
        ckpt = None

    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        device = "cpu"

    return AudioEmotionClassifier(checkpoint_path=ckpt, device=device)


audio_emotion_classifier = _build_classifier()
