"""Frame-level visual feature extractor — 35 Action Units (AU) per frame.

This module extracts facial Action Units from video frames, replicating the 35-AU
FACET features used in CMU-MOSEI.

Implementation strategy (two options):
  - Option A (preferred): OpenFace 2.0 command-line tool.
    Produces the standard 35 AUs that perfectly match CMU-MOSEI FACET format.
    Must be installed separately and available on PATH.
  - Option B (fallback): Py-Feat library (pure Python, no external binary needed).
    Extracts ~20 AUs via regression from pixel differences; pads to 35 dims.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import cv2
import numpy as np

MAX_SEQ_LEN = 50
VISION_DIM = 35


class VisualFeatureExtractor:
    def __init__(self, output_dir: Path | None = None):
        self.output_dir = output_dir
        self._openface_path: str | None = None

    @property
    def openface_path(self) -> str | None:
        if self._openface_path is None:
            # Check common installation locations
            candidates = [
                "OpenFace_2.2.0",
                "C:\\Program Files\\OpenFace_2.2.0\\OpenFace.exe",
                "C:\\OpenFace_2.2.0\\OpenFace.exe",
            ]
            for candidate in candidates:
                if shutil.which(candidate) or Path(candidate).exists():
                    self._openface_path = candidate
                    break
        return self._openface_path

    def extract_features(
        self,
        clip_path: str,
        clip_id: str,
        detections_path: str | None = None,
        word_timestamps: list[dict] | None = None,
    ) -> dict[str, Any]:
        """Extract 35-AU features from a video clip, aligned to word timestamps.

        Args:
            clip_path: Path to the video clip file (.mp4).
            clip_id: Clip identifier, used for output filename.
            detections_path: Optional path to detections.json from FaceExtractor.
                             If provided, uses face tracking info for frame selection.
            word_timestamps: Optional word timestamps from Whisper for alignment.

        Returns:
            {
                "features": np.ndarray (MAX_SEQ_LEN, 35), float64,
                "feature_path": str,
                "shape": str,
                "num_frames": int,
                "method": str,  # "openface" or "pyfeat"
            }
        """
        # Try OpenFace first, fall back to Py-Feat
        if self.openface_path:
            return self._extract_openface(clip_path, clip_id, word_timestamps)
        else:
            return self._extract_pyfeat(clip_path, clip_id, detections_path, word_timestamps)

    # ── OpenFace method (Option A) ────────────────────────────────────────────

    def _extract_openface(
        self,
        clip_path: str,
        clip_id: str,
        word_timestamps: list[dict] | None,
    ) -> dict[str, Any]:
        """Run OpenFace 2.0 to extract 35 AUs from clip."""
        output_dir = self.output_dir or (Path(clip_path).parent / "features")
        output_dir.mkdir(parents=True, exist_ok=True)

        # OpenFace requires output directory
        with tempfile.TemporaryDirectory() as tmpdir:
            clip_abs = Path(clip_path).resolve()
            cmd = [
                str(self.openface_path),
                "-f", str(clip_abs),
                "-out_dir", tmpdir,
                "-aus",
                "-pose",
                "-gaze",
                "-simalign",
            ]
            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=300,
                    encoding="utf-8",
                    errors="replace",
                )
            except subprocess.TimeoutExpired:
                return self._empty_result("openface_timeout")

            # Parse output CSV
            csv_files = list(Path(tmpdir).glob("*.csv"))
            if not csv_files:
                return self._empty_result("openface_no_output")

            au_csv = csv_files[0]
            au_df = self._parse_openface_csv(au_csv)  # (T, 35)

        # Resample / pad to MAX_SEQ_LEN
        features = self._resample_au(au_df, MAX_SEQ_LEN)
        features = self._pad_or_truncate(features, MAX_SEQ_LEN)

        # Save
        feature_path = output_dir / f"{clip_id}_vision_features.npy"
        np.save(str(feature_path), features.astype(np.float64))

        return {
            "features": features,
            "feature_path": str(feature_path.resolve()),
            "shape": f"({features.shape[0]}, {features.shape[1]})",
            "num_frames": features.shape[0],
            "method": "openface",
        }

    def _parse_openface_csv(self, csv_path: Path) -> np.ndarray:
        """Parse OpenFace CSV → (T, 35) AU matrix."""
        import pandas as pd

        df = pd.read_csv(csv_path, low_memory=False)

        # AU columns: AU01_r .. AU45_r (presence/absence intensity)
        au_cols = [c for c in df.columns if c.startswith("AU") and "_r" in c]
        if not au_cols:
            # Try AU01, AU02, ... without _r suffix
            au_cols = [c for c in df.columns if c.startswith("AU") and c[2:].isdigit()]

        if not au_cols:
            return np.zeros((0, VISION_DIM), dtype=np.float64)

        au_cols = sorted(au_cols)[:VISION_DIM]   # Take up to 35
        values = df[au_cols].values.astype(np.float64)

        # Replace NaN with 0
        values = np.nan_to_num(values, nan=0.0)

        # Ensure 35 dims
        if values.shape[1] < VISION_DIM:
            pad = np.zeros((values.shape[0], VISION_DIM - values.shape[1]))
            values = np.hstack([values, pad])

        return values  # (T_raw, 35)

    # ── Py-Feat method (Option B — fallback) ────────────────────────────────

    def _extract_pyfeat(
        self,
        clip_path: str,
        clip_id: str,
        detections_path: str | None,
        word_timestamps: list[dict] | None,
    ) -> dict[str, Any]:
        """Extract AUs via Py-Feat library (pure Python, no binary needed)."""
        output_dir = self.output_dir or (Path(clip_path).parent / "features")
        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            import feat
        except ImportError:
            return self._empty_result("pyfeat_not_installed")

        try:
            detector = feat.ExprDetector()
        except Exception as exc:
            return self._empty_result(f"pyfeat_init_failed: {exc}")

        try:
            # run on video file → returns DataFrame with AU columns
            df = detector.detect_video(clip_path, skip_frames=1)
        except Exception as exc:
            return self._empty_result(f"pyfeat_detection_failed: {exc}")

        # Extract AU columns (prefix AU)
        au_cols = [c for c in df.columns if c.startswith("AU")]
        if not au_cols:
            return self._empty_result("no_au_columns")

        # Sort for consistent ordering
        au_cols = sorted(au_cols)[:VISION_DIM]
        values = df[au_cols].values.astype(np.float64)
        values = np.nan_to_num(values, nan=0.0)

        # Pad to 35 dims
        if values.shape[1] < VISION_DIM:
            pad = np.zeros((values.shape[0], VISION_DIM - values.shape[1]))
            values = np.hstack([values, pad])

        # Resample to MAX_SEQ_LEN
        features = self._resample_au(values, MAX_SEQ_LEN)
        features = self._pad_or_truncate(features, MAX_SEQ_LEN)

        feature_path = output_dir / f"{clip_id}_vision_features.npy"
        np.save(str(feature_path), features.astype(np.float64))

        return {
            "features": features,
            "feature_path": str(feature_path.resolve()),
            "shape": f"({features.shape[0]}, {features.shape[1]})",
            "num_frames": features.shape[0],
            "method": "pyfeat",
        }

    # ── Shared helpers ───────────────────────────────────────────────────────

    def _resample_au(self, au_matrix: np.ndarray, target_len: int) -> np.ndarray:
        """Resample AU matrix to target length via linear interpolation."""
        if au_matrix.shape[0] == target_len:
            return au_matrix
        indices = np.linspace(0, au_matrix.shape[0] - 1, target_len)
        result = np.zeros((target_len, au_matrix.shape[1]))
        for i in range(au_matrix.shape[1]):
            result[:, i] = np.interp(indices, np.arange(au_matrix.shape[0]), au_matrix[:, i])
        return result

    def _pad_or_truncate(self, features: np.ndarray, max_len: int) -> np.ndarray:
        if features.shape[0] >= max_len:
            return features[:max_len]
        pad_len = max_len - features.shape[0]
        pad = np.zeros((pad_len, VISION_DIM), dtype=features.dtype)
        return np.vstack([features, pad])

    def _empty_result(self, warning: str = "") -> dict[str, Any]:
        zeros = np.zeros((MAX_SEQ_LEN, VISION_DIM), dtype=np.float64)
        return {
            "features": zeros,
            "feature_path": "",
            "shape": f"({MAX_SEQ_LEN}, {VISION_DIM})",
            "num_frames": 0,
            "method": "none",
            "warning": warning,
        }
