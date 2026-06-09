"""Face extraction service for Vietnamese drama/movie clips.

Primary detector: facenet-pytorch MTCNN when available.
Fallback: OpenCV Haar cascade so clips remain processable offline.
Also writes detections.json for review overlay bounding boxes with stable-ish
track_id assignment using lightweight IoU tracking.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import cv2

from backend.config import settings


class FaceExtractor:
    def __init__(self, output_dir: Path | None = None, sample_frames: int = 24):
        self.output_dir = output_dir or (settings.DATA_DIR / "faces")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.sample_frames = sample_frames

    def extract_faces_from_clip(self, clip_path: str, clip_id: str) -> Dict[str, Any]:
        clip = Path(clip_path)
        if not clip.exists():
            raise FileNotFoundError(f"Clip not found: {clip_path}")

        clip_dir = self.output_dir / clip_id
        clip_dir.mkdir(parents=True, exist_ok=True)
        frames = self._sample_frames(str(clip))
        detector_name = "mtcnn"
        try:
            face_paths, detections = self._detect_mtcnn(frames, clip_dir)
        except Exception:
            detector_name = "opencv_haar"
            face_paths, detections = self._detect_opencv(frames, clip_dir)

        detections = self._assign_track_ids(detections)
        detections_path = clip_dir / "detections.json"
        detections_path.write_text(json.dumps(detections, ensure_ascii=False, indent=2), encoding="utf-8")

        return {
            "cropped_face_paths": face_paths,
            "detections_path": str(detections_path),
            "detections": detections,
            "num_frames": len(frames),
            "num_faces": len(face_paths),
            "main_track_len": len(face_paths),
            "detector": detector_name,
            "track_count": self._count_tracks(detections),
        }

    def _sample_frames(self, clip_path: str) -> list[tuple[float, Any]]:
        cap = cv2.VideoCapture(clip_path)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open clip: {clip_path}")
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
        step = max(1, total // max(1, self.sample_frames)) if total else 1
        frames: list[tuple[float, Any]] = []
        idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % step == 0:
                frames.append((round(idx / fps, 3), frame))
                if len(frames) >= self.sample_frames:
                    break
            idx += 1
        cap.release()
        return frames

    def _detect_mtcnn(self, frames: list[tuple[float, Any]], clip_dir: Path) -> tuple[list[str], list[dict]]:
        from PIL import Image
        from backend.ai_models.model_manager import model_manager
        mtcnn = model_manager.load_model("mtcnn")
        paths: list[str] = []
        detections: list[dict] = []
        for i, (timestamp, frame) in enumerate(frames):
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(rgb)
            boxes, probs = mtcnn.detect(pil)
            frame_faces = []
            if boxes is not None:
                for j, box in enumerate(boxes[:5]):
                    conf = float(probs[j]) if probs is not None and probs[j] is not None else 0.0
                    if conf < 0.85:
                        continue
                    x1, y1, x2, y2 = [int(max(0, v)) for v in box]
                    crop = frame[y1:y2, x1:x2]
                    if crop.size == 0:
                        continue
                    out = clip_dir / f"face_{i:03d}_{j}.jpg"
                    cv2.imwrite(str(out), crop)
                    paths.append(str(out))
                    frame_faces.append({
                        "face_id": j,
                        "bbox": [x1, y1, x2, y2],
                        "confidence": conf,
                        "crop_path": str(out),
                        "detector": "mtcnn",
                    })
            detections.append({
                "timestamp": timestamp,
                "frame_size": [int(frame.shape[1]), int(frame.shape[0])],
                "faces": frame_faces,
            })
        return paths, detections

    def _detect_opencv(self, frames: list[tuple[float, Any]], clip_dir: Path) -> tuple[list[str], list[dict]]:
        cascade = self._load_haar_cascade()
        if cascade is None or cascade.empty():
            return [], [
                {"timestamp": timestamp, "frame_size": [int(frame.shape[1]), int(frame.shape[0])], "faces": []}
                for timestamp, frame in frames
            ]
        paths: list[str] = []
        detections: list[dict] = []
        for i, (timestamp, frame) in enumerate(frames):
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(40, 40))
            frame_faces = []
            for j, (x, y, w, h) in enumerate(faces[:5]):
                crop = frame[y:y + h, x:x + w]
                out = clip_dir / f"face_{i:03d}_{j}.jpg"
                cv2.imwrite(str(out), crop)
                paths.append(str(out))
                frame_faces.append({
                    "face_id": j,
                    "bbox": [int(x), int(y), int(x + w), int(y + h)],
                    "confidence": 0.60,
                    "crop_path": str(out),
                    "detector": "opencv_haar",
                })
            detections.append({
                "timestamp": timestamp,
                "frame_size": [int(frame.shape[1]), int(frame.shape[0])],
                "faces": frame_faces,
            })
        return paths, detections

    def _load_haar_cascade(self):
        candidates = [
            Path(getattr(cv2.data, "haarcascades", "")) / "haarcascade_frontalface_default.xml",
            Path(cv2.__file__).resolve().parent / "data" / "haarcascade_frontalface_default.xml",
        ]
        for candidate in candidates:
            try:
                cascade = cv2.CascadeClassifier(str(candidate))
                if not cascade.empty():
                    return cascade
            except Exception:
                continue
        return None

    def _assign_track_ids(self, detections: list[dict]) -> list[dict]:
        next_track_id = 0
        active_tracks: dict[int, dict] = {}
        max_age_frames = 2
        for frame in detections:
            assigned_tracks: set[int] = set()
            for face in frame.get("faces", []):
                bbox = face.get("bbox") or []
                best_track = None
                best_iou = 0.0
                for track_id, track in active_tracks.items():
                    if track_id in assigned_tracks or track.get("age", 0) > max_age_frames:
                        continue
                    score = self._bbox_iou(bbox, track.get("bbox", []))
                    if score > best_iou:
                        best_iou = score
                        best_track = track_id
                if best_track is not None and best_iou >= 0.25:
                    track_id = best_track
                else:
                    track_id = next_track_id
                    next_track_id += 1
                face["track_id"] = track_id
                active_tracks[track_id] = {"bbox": bbox, "age": 0}
                assigned_tracks.add(track_id)
            for track_id in list(active_tracks.keys()):
                if track_id not in assigned_tracks:
                    active_tracks[track_id]["age"] = active_tracks[track_id].get("age", 0) + 1
                    if active_tracks[track_id]["age"] > max_age_frames:
                        active_tracks.pop(track_id, None)
        return detections

    @staticmethod
    def _bbox_iou(a: list, b: list) -> float:
        if len(a) != 4 or len(b) != 4:
            return 0.0
        ax1, ay1, ax2, ay2 = [float(v) for v in a]
        bx1, by1, bx2, by2 = [float(v) for v in b]
        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)
        inter = max(0.0, inter_x2 - inter_x1) * max(0.0, inter_y2 - inter_y1)
        area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        denom = area_a + area_b - inter
        return inter / denom if denom > 0 else 0.0

    @staticmethod
    def _count_tracks(detections: list[dict]) -> int:
        return len({face.get("track_id") for frame in detections for face in frame.get("faces", []) if face.get("track_id") is not None})
