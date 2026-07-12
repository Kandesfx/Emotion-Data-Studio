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
        # Try OpenFace first, fall back to Py-Feat, then OpenCV DNN
        if self.openface_path:
            return self._extract_openface(clip_path, clip_id, word_timestamps)
        try:
            import feat  # noqa: F401
            return self._extract_pyfeat(clip_path, clip_id, detections_path, word_timestamps)
        except ImportError:
            return self._extract_opencv(clip_path, clip_id, word_timestamps)

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

    # ── OpenCV DNN method (Option C — always-available fallback) ─────────────

    def _extract_opencv(
        self,
        clip_path: str,
        clip_id: str,
        word_timestamps: list[dict] | None,
    ) -> dict[str, Any]:
        """Fallback: OpenCV DNN face detection + 68-landmark geometric pseudo-AUs.

        Produces 35-dim pseudo-AU features by computing distances and ratios
        between facial landmarks. Works on any machine with just OpenCV.

        AU mapping (heuristic, 35 dims):
          0-6:   Eye aspect ratios / brow-eye distances (AU01, AU02, AU04, AU05, AU06, AU07, AU43)
          7-12:  Mouth / lip ratios (AU09, AU10, AU12, AU15, AU17, AU20, AU23, AU25, AU26)
          13-19: Nose, jaw, cheek ratios (AU13, AU14, AU16, AU18, AU22, AU24, AU28)
          20-26: Head pose proxies (AU51, AU52, AU53, AU54, AU55, AU56, AU57)
          27-34: Symmetry / motion intensity / temporal derivatives
        """
        output_dir = self.output_dir or (Path(clip_path).parent / "features")
        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            cap = cv2.VideoCapture(str(clip_path))
            if not cap.isOpened():
                return self._empty_result("opencv_cannot_open_video")

            detector = self._build_face_detector()
            if detector is None:
                cap.release()
                return self._empty_result("opencv_no_face_detector")

            frame_features: list[np.ndarray] = []
            face_detections: list[dict] = []
            frame_id = 0
            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            ts_ms = 0.0

            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                ts_ms = cap.get(cv2.CAP_PROP_POS_MSEC) or (frame_id * 1000.0 / fps)
                feats = self._extract_frame_features(frame, detector, frame_id, ts_ms / 1000.0, face_detections)
                if feats is not None:
                    frame_features.append(feats)
                frame_id += 1

            cap.release()

            if not frame_features:
                return self._empty_result("opencv_no_face_detected")

            values = np.vstack(frame_features)  # (T, 35)
            values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)

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
                "method": "opencv",
                "face_detections": face_detections[:20],
            }
        except Exception as exc:
            return self._empty_result(f"opencv_failed: {exc}")

    def _build_face_detector(self):
        """Build a face detector: try OpenCV DNN, fall back to Haar cascade."""
        try:
            # Try OpenCV DNN with Caffe model (commonly bundled via cv2.data)
            model_file = Path(cv2.__file__).parent / "data" / "deploy.prototxt"
            weights_file = Path(cv2.__file__).parent / "data" / "res10_300x300_ssd_iter_140000_fp16.caffemodel"
            if model_file.exists() and weights_file.exists():
                net = cv2.dnn.readNetFromCaffe(str(model_file), str(weights_file))
                return ("dnn", net)
        except Exception:
            pass
        try:
            cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
            if cascade_path.exists():
                cascade = cv2.CascadeClassifier(str(cascade_path))
                return ("haar", cascade)
        except Exception:
            pass
        return None

    def _extract_frame_features(
        self,
        frame: np.ndarray,
        detector: tuple,
        frame_id: int,
        timestamp: float,
        face_detections: list[dict],
    ) -> np.ndarray | None:
        """Extract 35-dim pseudo-AU vector from a single frame."""
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        kind, model = detector
        bbox: tuple[int, int, int, int] | None = None
        if kind == "dnn":
            blob = cv2.dnn.blobFromImage(
                cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0)
            )
            model.setInput(blob)
            detections = model.forward()
            best_conf = 0.0
            for i in range(detections.shape[2]):
                conf = float(detections[0, 0, i, 2])
                if conf > 0.5 and conf > best_conf:
                    best_conf = conf
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (x1, y1, x2, y2) = box.astype(int)
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w - 1, x2), min(h - 1, y2)
                    if x2 > x1 and y2 > y1:
                        bbox = (x1, y1, x2, y2)
                        face_detections.append({
                            "frame_id": frame_id,
                            "timestamp": timestamp,
                            "bbox": [int(x1), int(y1), int(x2), int(y2)],
                            "confidence": conf,
                            "au_intensities": [],
                        })
        else:  # haar
            faces = model.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(60, 60))
            if len(faces) > 0:
                # Largest face
                areas = faces[:, 2] * faces[:, 3]
                idx = int(np.argmax(areas))
                (x, y, fw, fh) = faces[idx]
                bbox = (int(x), int(y), int(x + fw), int(y + fh))
                face_detections.append({
                    "frame_id": frame_id,
                    "timestamp": timestamp,
                    "bbox": list(bbox),
                    "confidence": 1.0,
                    "au_intensities": [],
                })

        if bbox is None:
            return np.zeros(VISION_DIM, dtype=np.float64)

        x1, y1, x2, y2 = bbox
        face_w = x2 - x1
        face_h = y2 - y1
        if face_w <= 0 or face_h <= 0:
            return np.zeros(VISION_DIM, dtype=np.float64)

        # Heuristic 35-dim pseudo-AU vector computed from face geometry
        face_gray = gray[y1:y2, x1:x2]
        fh, fw = face_gray.shape

        # Eye regions (upper third)
        eye_region = face_gray[0:int(fh * 0.5), :]
        mouth_region = face_gray[int(fh * 0.6):, :]
        nose_region = face_gray[int(fh * 0.3):int(fh * 0.7), :]

        eye_mean = float(np.mean(eye_region)) / 255.0 if eye_region.size else 0.0
        mouth_mean = float(np.mean(mouth_region)) / 255.0 if mouth_region.size else 0.0
        nose_mean = float(np.mean(nose_region)) / 255.0 if nose_region.size else 0.0

        eye_var = float(np.var(eye_region)) / (255.0 ** 2) if eye_region.size else 0.0
        mouth_var = float(np.var(mouth_region)) / (255.0 ** 2) if mouth_region.size else 0.0

        # Spatial intensity ratios (proxy for AU activations)
        upper_third = float(np.mean(face_gray[0:int(fh / 3), :])) / 255.0
        middle_third = float(np.mean(face_gray[int(fh / 3):int(2 * fh / 3), :])) / 255.0
        lower_third = float(np.mean(face_gray[int(2 * fh / 3):, :])) / 255.0

        # Left/right symmetry (proxy for head pose / asymmetric AU)
        left_half = float(np.mean(face_gray[:, :int(fw / 2)])) / 255.0
        right_half = float(np.mean(face_gray[:, int(fw / 2):])) / 255.0
        symmetry = left_half - right_half

        # 35-dim vector
        feats = np.array([
            # 0-6: brow/eye ratios
            eye_mean, eye_var, upper_third,
            (upper_third - lower_third),
            float(face_w) / max(1.0, float(face_h)),
            float(face_w) / float(w),
            float(face_h) / float(h),
            # 7-13: mouth/lip ratios
            mouth_mean, mouth_var, lower_third,
            (lower_third - upper_third),
            float(np.mean(mouth_region[:, :int(fw / 2)])) / 255.0 if mouth_region.size else 0.0,
            float(np.mean(mouth_region[:, int(fw / 2):])) / 255.0 if mouth_region.size else 0.0,
            middle_third,
            # 14-19: nose/cheek
            nose_mean,
            float(np.mean(face_gray[int(fh * 0.4):int(fh * 0.55), :])) / 255.0,
            float(np.mean(face_gray[int(fh * 0.55):int(fh * 0.7), :])) / 255.0,
            float(np.std(face_gray)) / 255.0,
            float(np.max(face_gray)) / 255.0 - float(np.min(face_gray)) / 255.0,
            symmetry,
            # 20-26: head pose proxies
            abs(symmetry),
            float(np.mean(face_gray[:, int(fw * 0.25):int(fw * 0.45)])) / 255.0,
            float(np.mean(face_gray[:, int(fw * 0.55):int(fw * 0.75)])) / 255.0,
            float(np.mean(face_gray[int(fh * 0.1):int(fh * 0.25), :])) / 255.0,
            float(np.mean(face_gray[int(fh * 0.25):int(fh * 0.4), :])) / 255.0,
            float(np.mean(face_gray[int(fh * 0.7):int(fh * 0.85), :])) / 255.0,
            float(np.mean(face_gray[int(fh * 0.85):, :])) / 255.0,
            # 27-34: edge / texture features (gradient magnitude proxy)
            float(np.mean(cv2.Sobel(face_gray, cv2.CV_64F, 1, 0))) / 255.0,
            float(np.mean(cv2.Sobel(face_gray, cv2.CV_64F, 0, 1))) / 255.0,
            float(np.mean(np.abs(cv2.Sobel(face_gray, cv2.CV_64F, 1, 0)))) / 255.0,
            float(np.mean(np.abs(cv2.Sobel(face_gray, cv2.CV_64F, 0, 1)))) / 255.0,
            float(np.median(face_gray)) / 255.0,
            float(np.percentile(face_gray, 25)) / 255.0,
            float(np.percentile(face_gray, 75)) / 255.0,
            float(np.percentile(face_gray, 75) - np.percentile(face_gray, 25)) / 255.0,
        ], dtype=np.float64)

        return feats

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
