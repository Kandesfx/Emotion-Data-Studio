"""Smart face/dialogue-aware segmentation.

This service turns coarse scenes into data-mining friendly segments:
- detect face presence at low FPS inside each scene
- build continuous face runs with gap smoothing
- detect dialogue/silence intervals with FFmpeg silencedetect
- refine cuts so clips follow visible faces and utterance boundaries

It is intentionally conservative: if face/dialogue analysis fails, callers can
fall back to scene-only segmentation.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List

import cv2
import numpy as np

from backend.config import settings


@dataclass
class FaceSample:
    timestamp: float
    has_face: bool
    faces: list[dict] = field(default_factory=list)


@dataclass
class SegmentCandidate:
    scene_index: int
    start_time: float
    end_time: float
    duration: float
    source: str
    face_coverage: float
    speech_coverage: float
    num_faces_avg: float
    has_dialogue: bool
    cut_reason: str
    quality_hint: str
    metadata: dict = field(default_factory=dict)

    def to_scene_dict(self) -> dict:
        data = asdict(self)
        data["duration"] = max(0.0, self.end_time - self.start_time)
        return data


class SmartSegmenter:
    def __init__(
        self,
        face_scan_fps: float = 2.0,
        face_confidence: float = 0.55,
        max_missing_face_gap: float = 1.0,
        min_duration: float = 2.0,
        max_duration: float = 12.0,
        target_duration: float = 6.0,
        silence_threshold_db: str = "-35dB",
        silence_min_duration: float = 0.45,
        max_dialogue_extension: float = 1.5,
        vad_mode: str = "energy",
        metadata_dir: Path | None = None,
        min_face_coverage_in_scene: float | None = None,
    ):
        self.face_scan_fps = face_scan_fps
        self.face_confidence = face_confidence
        self.max_missing_face_gap = max_missing_face_gap
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.target_duration = target_duration
        self.silence_threshold_db = silence_threshold_db
        self.silence_min_duration = silence_min_duration
        self.max_dialogue_extension = max_dialogue_extension
        self.vad_mode = vad_mode
        self.metadata_dir = metadata_dir or (settings.DATA_DIR / "annotations" / "segments")
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        self._cascade = self._load_haar_cascade()
        self._mp_detector = self._init_mediapipe_detector()
        # Sprint 2: nguong Smart pre-cut. Neu None thi lay tu settings.AI_AUTOCUT_MIN_FACE_COVERAGE_IN_SCENE
        if min_face_coverage_in_scene is None:
            min_face_coverage_in_scene = float(getattr(settings, "AI_AUTOCUT_MIN_FACE_COVERAGE_IN_SCENE", 0.70))
        self.min_face_coverage_in_scene = float(min_face_coverage_in_scene)

    def build_segments(self, video_path: str, scenes: list[dict], video_id: str | None = None) -> list[dict]:
        if not Path(video_path).exists():
            raise FileNotFoundError(f"Video not found: {video_path}")
        if not scenes:
            scenes = [self._whole_video_scene(video_path)]

        speech_intervals = self.detect_speech_intervals(video_path)
        candidates: list[SegmentCandidate] = []
        face_samples_by_scene: dict[str, list[dict]] = {}

        for scene in scenes:
            scene_index = int(scene.get("scene_index", len(face_samples_by_scene)))
            scene_start = float(scene.get("start_time", 0.0))
            scene_end = float(scene.get("end_time", scene_start))
            if scene_end <= scene_start:
                continue

            samples = self.scan_faces(video_path, scene_start, scene_end)
            face_samples_by_scene[str(scene_index)] = [asdict(sample) for sample in samples]
            runs = self._build_face_runs(samples, scene_start, scene_end)
            if not runs:
                # Keep short dialogue-only scene as low-priority candidate.
                scene_speech = self._intersections((scene_start, scene_end), speech_intervals)
                if scene_speech and scene_end - scene_start >= self.min_duration:
                    runs = [(scene_start, scene_end, 0.0, 0.0)]
                else:
                    continue

            for run_start, run_end, face_coverage, num_faces_avg in runs:
                refined_start, refined_end = self._refine_with_dialogue(run_start, run_end, speech_intervals)
                refined_start = max(scene_start, refined_start)
                refined_end = min(scene_end, refined_end)
                pieces = self._enforce_duration(refined_start, refined_end, speech_intervals)
                for start, end in pieces:
                    duration = end - start
                    if duration < self.min_duration:
                        continue
                    speech_cov = self._coverage((start, end), speech_intervals)
                    face_cov = self._coverage_from_samples((start, end), samples)
                    avg_faces = self._avg_faces((start, end), samples)
                    quality_hint = self._quality_hint(face_cov, speech_cov, duration)

                    # Sprint 2 — Smart pre-cut: loc segment face_coverage thap (<0.70 mac dinh)
                    # Tru khi dialogue rat tot (speech_cov >= 0.6) → giu lai lam review
                    skip_face_coverage = (
                        face_cov < self.min_face_coverage_in_scene
                        and speech_cov < 0.6
                    )
                    if skip_face_coverage:
                        logger.debug(
                            f"[SmartCut] skip {start:.2f}-{end:.2f}: face_cov={face_cov:.2f} < "
                            f"{self.min_face_coverage_in_scene:.2f} (speech_cov={speech_cov:.2f})"
                        )
                        continue

                    candidates.append(SegmentCandidate(
                        scene_index=scene_index,
                        start_time=round(start, 3),
                        end_time=round(end, 3),
                        duration=round(duration, 3),
                        source="face_dialogue" if speech_cov > 0 else "face_only",
                        face_coverage=round(face_cov, 3),
                        speech_coverage=round(speech_cov, 3),
                        num_faces_avg=round(avg_faces, 3),
                        has_dialogue=speech_cov > 0.15,
                        cut_reason="face_run_refined_by_dialogue" if speech_cov > 0 else "face_run",
                        quality_hint=quality_hint,
                        metadata={
                            "raw_face_coverage": face_coverage,
                            "raw_num_faces_avg": num_faces_avg,
                            "min_face_coverage_threshold": self.min_face_coverage_in_scene,
                            "skipped_strict": skip_face_coverage,
                        },
                    ))

        merged = self._dedupe_and_sort(candidates)
        if video_id:
            self.write_metadata(video_id, merged, face_samples_by_scene, speech_intervals)
        return [candidate.to_scene_dict() for candidate in merged]

    def scan_faces(self, video_path: str, start: float, end: float) -> list[FaceSample]:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        step_sec = 1.0 / max(0.1, self.face_scan_fps)
        samples: list[FaceSample] = []
        t = start
        while t <= end:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps))
            ok, frame = cap.read()
            if not ok:
                break
            faces = self._detect_faces_opencv(frame)
            samples.append(FaceSample(timestamp=round(t, 3), has_face=bool(faces), faces=faces))
            t += step_sec
        cap.release()
        return samples

    def detect_speech_intervals(self, video_path: str) -> list[tuple[float, float]]:
        if self.vad_mode in {"auto", "energy"}:
            intervals = self._detect_speech_energy_vad(video_path)
            if intervals:
                return intervals
        if self.vad_mode == "disabled":
            return []
        return self._detect_speech_ffmpeg_silence(video_path)

    def _detect_speech_energy_vad(self, video_path: str) -> list[tuple[float, float]]:
        ffmpeg_path = settings.FFMPEG_PATH or "ffmpeg"
        if shutil.which(ffmpeg_path) is None and not Path(ffmpeg_path).exists():
            return []
        sample_rate = 16000
        cmd = [
            ffmpeg_path,
            "-i", video_path,
            "-vn",
            "-ac", "1",
            "-ar", str(sample_rate),
            "-f", "s16le",
            "-",
        ]
        startupinfo = None
        if os.name == "nt":
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        try:
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, startupinfo=startupinfo, timeout=120)
            if result.returncode != 0 or not result.stdout:
                return []
            audio = np.frombuffer(result.stdout, dtype=np.int16).astype(np.float32) / 32768.0
            if audio.size < sample_rate * 0.25:
                return []
            frame_len = int(sample_rate * 0.03)
            hop = int(sample_rate * 0.015)
            rms = []
            times = []
            for start in range(0, max(1, audio.size - frame_len), hop):
                frame = audio[start:start + frame_len]
                if frame.size < frame_len:
                    break
                rms.append(float(np.sqrt(np.mean(frame * frame))))
                times.append(start / sample_rate)
            if not rms:
                return []
            arr = np.array(rms)
            noise_floor = float(np.percentile(arr, 35))
            threshold = max(0.008, noise_floor * 2.3, float(np.percentile(arr, 70)) * 0.45)
            voiced = arr >= threshold
            intervals = self._voiced_flags_to_intervals(voiced, times, hop / sample_rate)
            return self._merge_short_gaps(intervals, max_gap=0.35, min_duration=0.25)
        except Exception:
            return []

    def _detect_speech_ffmpeg_silence(self, video_path: str) -> list[tuple[float, float]]:
        ffmpeg_path = settings.FFMPEG_PATH or "ffmpeg"
        if shutil.which(ffmpeg_path) is None and not Path(ffmpeg_path).exists():
            return []
        cmd = [
            ffmpeg_path,
            "-i", video_path,
            "-af", f"silencedetect=noise={self.silence_threshold_db}:d={self.silence_min_duration}",
            "-f", "null", "-",
        ]
        startupinfo = None
        if os.name == "nt":
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, startupinfo=startupinfo)
        text = result.stderr.decode("utf-8", errors="ignore")
        duration = self._probe_duration(video_path)
        silences = self._parse_silences(text)
        return self._speech_from_silences(silences, duration)

    def write_metadata(self, video_id: str, candidates: list[SegmentCandidate], face_samples_by_scene: dict, speech_intervals: list[tuple[float, float]]):
        payload = {
            "video_id": video_id,
            "segments": [candidate.to_scene_dict() for candidate in candidates],
            "face_samples_by_scene": face_samples_by_scene,
            "speech_intervals": [{"start": s, "end": e} for s, e in speech_intervals],
            "parameters": {
                "face_scan_fps": self.face_scan_fps,
                "max_missing_face_gap": self.max_missing_face_gap,
                "min_duration": self.min_duration,
                "max_duration": self.max_duration,
                "target_duration": self.target_duration,
                "silence_threshold_db": self.silence_threshold_db,
                "silence_min_duration": self.silence_min_duration,
            },
        }
        path = self.metadata_dir / f"{video_id}_smart_segments.json"
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

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

    def _init_mediapipe_detector(self):
        try:
            import mediapipe as mp
            return mp.solutions.face_detection.FaceDetection(
                model_selection=1,
                min_detection_confidence=self.face_confidence,
            )
        except Exception:
            return None

    def _detect_faces_opencv(self, frame) -> list[dict]:
        if self._mp_detector is not None:
            try:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = self._mp_detector.process(rgb)
                h, w = frame.shape[:2]
                faces = []
                for det in result.detections or []:
                    score = float(det.score[0]) if det.score else 0.0
                    box = det.location_data.relative_bounding_box
                    x1 = max(0, int(box.xmin * w))
                    y1 = max(0, int(box.ymin * h))
                    x2 = min(w, int((box.xmin + box.width) * w))
                    y2 = min(h, int((box.ymin + box.height) * h))
                    if x2 > x1 and y2 > y1:
                        faces.append({"bbox": [x1, y1, x2, y2], "confidence": score, "detector": "mediapipe"})
                return faces
            except Exception:
                pass

        if self._cascade is None or self._cascade.empty():
            return []
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self._cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(40, 40))
        return [
            {"bbox": [int(x), int(y), int(x + w), int(y + h)], "confidence": 0.60, "detector": "opencv_haar"}
            for x, y, w, h in faces
        ]

    def _build_face_runs(self, samples: list[FaceSample], scene_start: float, scene_end: float) -> list[tuple[float, float, float, float]]:
        runs = []
        run_start = None
        last_face_t = None
        face_count = 0
        total_count = 0
        faces_sum = 0
        step = 1.0 / max(0.1, self.face_scan_fps)
        for sample in samples:
            total_count += 1
            if sample.has_face:
                faces_sum += len(sample.faces)
                face_count += 1
                if run_start is None:
                    run_start = sample.timestamp
                last_face_t = sample.timestamp
            elif run_start is not None and last_face_t is not None:
                if sample.timestamp - last_face_t > self.max_missing_face_gap:
                    end = min(scene_end, last_face_t + step)
                    denom = max(1, total_count)
                    runs.append((run_start, end, face_count / denom, faces_sum / max(1, face_count)))
                    run_start = None
                    last_face_t = None
                    face_count = 0
                    total_count = 0
                    faces_sum = 0
        if run_start is not None and last_face_t is not None:
            end = min(scene_end, last_face_t + step)
            runs.append((run_start, end, face_count / max(1, total_count), faces_sum / max(1, face_count)))
        return [(s, e, fc, nf) for s, e, fc, nf in runs if e - s >= self.min_duration * 0.6]

    def _refine_with_dialogue(self, start: float, end: float, speech: list[tuple[float, float]]) -> tuple[float, float]:
        new_start, new_end = start, end
        for s, e in speech:
            if s <= start <= e and start - s <= self.max_dialogue_extension:
                new_start = s
            if s <= end <= e and e - end <= self.max_dialogue_extension:
                new_end = e
        return new_start, new_end

    def _enforce_duration(self, start: float, end: float, speech: list[tuple[float, float]]) -> list[tuple[float, float]]:
        duration = end - start
        if duration <= self.max_duration:
            return [(start, end)]
        pieces = self._split_by_silence(start, end, speech)
        final = []
        for p_start, p_end in pieces:
            if p_end - p_start <= self.max_duration:
                final.append((p_start, p_end))
            else:
                cursor = p_start
                while cursor + self.min_duration <= p_end:
                    chunk_end = min(cursor + self.target_duration, p_end)
                    if chunk_end - cursor >= self.min_duration:
                        final.append((cursor, chunk_end))
                    cursor = chunk_end
        return final

    def _split_by_silence(self, start: float, end: float, speech: list[tuple[float, float]]) -> list[tuple[float, float]]:
        overlaps = self._intersections((start, end), speech)
        if not overlaps:
            return [(start, end)]
        pieces = []
        cursor = start
        for s, e in overlaps:
            if s - cursor >= self.min_duration and s - cursor <= self.max_duration:
                pieces.append((cursor, s))
                cursor = s
            if e - cursor >= self.min_duration and e - cursor <= self.max_duration:
                pieces.append((cursor, e))
                cursor = e
        if end - cursor >= self.min_duration:
            pieces.append((cursor, end))
        return pieces or [(start, end)]

    def _voiced_flags_to_intervals(self, voiced, times: list[float], frame_step: float) -> list[tuple[float, float]]:
        intervals: list[tuple[float, float]] = []
        start = None
        for flag, timestamp in zip(voiced, times):
            if flag and start is None:
                start = timestamp
            elif not flag and start is not None:
                intervals.append((start, timestamp + frame_step))
                start = None
        if start is not None and times:
            intervals.append((start, times[-1] + frame_step))
        return intervals

    def _merge_short_gaps(self, intervals: list[tuple[float, float]], max_gap: float, min_duration: float) -> list[tuple[float, float]]:
        if not intervals:
            return []
        merged: list[tuple[float, float]] = []
        cur_start, cur_end = intervals[0]
        for start, end in intervals[1:]:
            if start - cur_end <= max_gap:
                cur_end = max(cur_end, end)
            else:
                if cur_end - cur_start >= min_duration:
                    merged.append((cur_start, cur_end))
                cur_start, cur_end = start, end
        if cur_end - cur_start >= min_duration:
            merged.append((cur_start, cur_end))
        return merged

    def _coverage(self, segment: tuple[float, float], intervals: list[tuple[float, float]]) -> float:
        start, end = segment
        if end <= start:
            return 0.0
        total = sum(max(0.0, min(end, e) - max(start, s)) for s, e in intervals)
        return max(0.0, min(1.0, total / (end - start)))

    def _coverage_from_samples(self, segment: tuple[float, float], samples: list[FaceSample]) -> float:
        selected = [sample for sample in samples if segment[0] <= sample.timestamp <= segment[1]]
        if not selected:
            return 0.0
        return sum(1 for sample in selected if sample.has_face) / len(selected)

    def _avg_faces(self, segment: tuple[float, float], samples: list[FaceSample]) -> float:
        selected = [sample for sample in samples if segment[0] <= sample.timestamp <= segment[1] and sample.has_face]
        if not selected:
            return 0.0
        return sum(len(sample.faces) for sample in selected) / len(selected)

    def _intersections(self, segment: tuple[float, float], intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
        start, end = segment
        return [(max(start, s), min(end, e)) for s, e in intervals if min(end, e) > max(start, s)]

    def _quality_hint(self, face_cov: float, speech_cov: float, duration: float) -> str:
        if face_cov >= 0.7 and 2.0 <= duration <= self.max_duration and speech_cov >= 0.2:
            return "good"
        if face_cov >= 0.4 and duration <= self.max_duration:
            return "review"
        return "weak"

    def _dedupe_and_sort(self, candidates: list[SegmentCandidate]) -> list[SegmentCandidate]:
        ordered = sorted(candidates, key=lambda c: (c.start_time, c.end_time))
        result: list[SegmentCandidate] = []
        for candidate in ordered:
            if result and candidate.start_time < result[-1].end_time - 0.25:
                prev = result[-1]
                if self._rank(candidate) > self._rank(prev):
                    result[-1] = candidate
            else:
                result.append(candidate)
        return result

    def _rank(self, candidate: SegmentCandidate) -> float:
        return candidate.face_coverage * 0.6 + candidate.speech_coverage * 0.3 + min(1.0, candidate.duration / self.target_duration) * 0.1

    def _parse_silences(self, ffmpeg_text: str) -> list[tuple[float, float]]:
        starts = [float(x) for x in re.findall(r"silence_start: ([0-9.]+)", ffmpeg_text)]
        ends = [float(x) for x in re.findall(r"silence_end: ([0-9.]+)", ffmpeg_text)]
        return list(zip(starts, ends))

    def _speech_from_silences(self, silences: list[tuple[float, float]], duration: float) -> list[tuple[float, float]]:
        intervals = []
        cursor = 0.0
        for start, end in sorted(silences):
            if start - cursor >= 0.25:
                intervals.append((cursor, start))
            cursor = max(cursor, end)
        if duration - cursor >= 0.25:
            intervals.append((cursor, duration))
        return intervals

    def _probe_duration(self, video_path: str) -> float:
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 0
        frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
        cap.release()
        return float(frames / fps) if fps else 0.0

    def _whole_video_scene(self, video_path: str) -> dict:
        duration = self._probe_duration(video_path)
        return {"scene_index": 0, "start_time": 0.0, "end_time": duration, "duration": duration}
