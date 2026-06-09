"""Alignment engine — synchronises audio, text, and vision features to a shared timeline.

The reference timeline is derived from Whisper word-level timestamps:
  - Audio frames are mean-pooled per word interval [word_start, word_end]
  - Text embeddings are already per-word
  - Vision frames are mean-pooled per word interval

Each word corresponds to one time step in the final aligned feature sequence.
Output shape for all modalities: (MAX_SEQ_LEN, D) where D = 768, 74, 35.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

MAX_SEQ_LEN = 50


class AlignmentEngine:
    def __init__(self, max_seq_len: int = MAX_SEQ_LEN):
        self.max_seq_len = max_seq_len

    def align(
        self,
        text_features: np.ndarray,   # (T_words, 768) or (T, 768)
        audio_features: np.ndarray,   # (T_audio, 74)  — already word-aligned from audio extractor
        vision_features: np.ndarray,   # (T_vision, 35) — already word-aligned from visual extractor
        word_timestamps: list[dict] | None = None,
    ) -> dict[str, Any]:
        """Align three modality features to shared timeline.

        If audio_features and vision_features are already word-aligned (from their
        respective extractors), this method simply ensures uniform shape (MAX_SEQ_LEN, D)
        and normalises ranges.

        Args:
            text_features:  Word-level PhoBERT embeddings (T_words, 768).
            audio_features: Word-aligned audio features (T_words, 74) OR raw (T_raw, 74).
            vision_features: Word-aligned vision features (T_words, 35) OR raw (T_raw, 35).
            word_timestamps: List of {"word": str, "start": float, "end": float}.

        Returns:
            {
                "text_aligned":   np.ndarray (MAX_SEQ_LEN, 768),
                "audio_aligned":  np.ndarray (MAX_SEQ_LEN, 74),
                "vision_aligned": np.ndarray (MAX_SEQ_LEN, 35),
                "shape": str,
                "aligned": bool,
            }
        """
        T_words = len(word_timestamps) if word_timestamps else 0

        # Text — already per-word, just pad/truncate
        text_aligned = self._normalize_shape(text_features, self.max_seq_len, 768)

        # Audio — if raw (T_raw != T_words), re-align via word timestamps
        if word_timestamps and audio_features.shape[0] != T_words and T_words > 0:
            audio_aligned = self._align_raw_to_words(
                audio_features, word_timestamps, dim=74
            )
        else:
            audio_aligned = self._normalize_shape(audio_features, self.max_seq_len, 74)

        # Vision — same logic
        if word_timestamps and vision_features.shape[0] != T_words and T_words > 0:
            vision_aligned = self._align_raw_to_words(
                vision_features, word_timestamps, dim=35
            )
        else:
            vision_aligned = self._normalize_shape(vision_features, self.max_seq_len, 35)

        # Replace inf / nan
        for arr in (text_aligned, audio_aligned, vision_aligned):
            arr[np.isnan(arr)] = 0.0
            arr[np.isinf(arr)] = 0.0

        return {
            "text_aligned": text_aligned,
            "audio_aligned": audio_aligned,
            "vision_aligned": vision_aligned,
            "shape": f"({self.max_seq_len}, text:768 / audio:74 / vision:35)",
            "aligned": bool(word_timestamps),
        }

    def _normalize_shape(self, arr: np.ndarray, max_len: int, dim: int) -> np.ndarray:
        """Pad/truncate to (max_len, dim) and ensure float type."""
        if arr.ndim == 1:
            arr = arr[np.newaxis, :]
        if arr.shape[1] < dim:
            pad = np.zeros((arr.shape[0], dim - arr.shape[1]), dtype=arr.dtype)
            arr = np.hstack([arr, pad])
        elif arr.shape[1] > dim:
            arr = arr[:, :dim]

        if arr.shape[0] > max_len:
            arr = arr[:max_len]
        elif arr.shape[0] < max_len:
            pad = np.zeros((max_len - arr.shape[0], dim), dtype=arr.dtype)
            arr = np.vstack([arr, pad])

        return arr.astype(np.float64)

    def _align_raw_to_words(
        self,
        raw_features: np.ndarray,
        word_timestamps: list[dict],
        dim: int,
    ) -> np.ndarray:
        """Mean-pool raw frame-level features into word-level buckets.

        Each word gets mean of raw frames whose timestamps fall within [word_start, word_end].
        The raw features are assumed to be uniformly sampled at 100 fps.
        """
        num_words = len(word_timestamps)
        if num_words == 0:
            return np.zeros((self.max_seq_len, dim), dtype=np.float64)

        # Assume raw features are at 100 fps (hop_length=160 @ 16kHz)
        fps = 100.0
        num_raw_frames = raw_features.shape[0]
        clip_duration = num_raw_frames / fps

        aligned = np.zeros((num_words, dim), dtype=np.float64)
        for i, w in enumerate(word_timestamps):
            start_s = max(0.0, float(w.get("start", 0)))
            end_s = min(clip_duration, float(w.get("end", start_s)))
            start_f = int(start_s * fps)
            end_f = int(end_s * fps)
            start_f = min(start_f, num_raw_frames)
            end_f = min(end_f, num_raw_frames)
            if end_f > start_f:
                aligned[i] = np.mean(raw_features[start_f:end_f], axis=0)
            else:
                aligned[i] = np.zeros(dim)

        return self._normalize_shape(aligned, self.max_seq_len, dim)

    def save_aligned_features(
        self,
        output_dir: Path,
        clip_id: str,
        text_features: np.ndarray,
        audio_features: np.ndarray,
        vision_features: np.ndarray,
    ) -> dict[str, str]:
        """Save aligned features to .npy files."""
        output_dir.mkdir(parents=True, exist_ok=True)
        paths = {}
        for name, feats in [("text", text_features), ("audio", audio_features), ("vision", vision_features)]:
            p = output_dir / f"{clip_id}_{name}_aligned.npy"
            np.save(str(p), feats.astype(np.float64))
            paths[f"{name}_path"] = str(p.resolve())
        return paths
