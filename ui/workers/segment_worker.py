"""
Emotion Data Studio - Segment Worker.

Cuts manually authored video segments with FFmpeg, creates Clip records,
and optionally runs the same AI stages used by the automatic pipeline.
"""

from __future__ import annotations

import os
import subprocess
import traceback
import uuid
from pathlib import Path

from PySide6.QtCore import QThread, Signal


class SegmentWorker(QThread):
    progress_updated = Signal(str, int, int)
    log_message = Signal(str)
    pipeline_finished = Signal(dict)
    error_occurred = Signal(str)
    stage_completed = Signal(str)

    def __init__(self, video_id: str, video_path: str, segments: list, processing_mode: str = "semi_auto"):
        super().__init__()
        self.video_id = video_id
        self.video_path = video_path
        self.segments = segments
        self.processing_mode = processing_mode
        self._is_cancelled = False

    def run(self):
        session = None
        try:
            from backend.config import settings
            from backend.database.local_db import get_session
            from backend.database.models import Clip, Video

            session = get_session()
            video = session.query(Video).filter(Video.id == self.video_id).first()
            if not video:
                raise RuntimeError(f"Video not found: {self.video_id}")
            if not self.video_path or not Path(self.video_path).exists():
                raise FileNotFoundError(f"Video file not found: {self.video_path}")
            if not self.segments:
                raise ValueError("No segments to process")

            ffmpeg_path = settings.FFMPEG_PATH
            import shutil
            if shutil.which(ffmpeg_path) is None:
                raise RuntimeError("FFmpeg is not available. Configure FFmpeg in Settings.")

            video.status = "processing"
            session.commit()

            clips_dir = settings.DATA_DIR / "clips"
            clips_dir.mkdir(parents=True, exist_ok=True)
            total = len(self.segments)
            created_clips = []
            self.log_message.emit(f"[START] Cutting {total} manual segments")

            for idx, segment in enumerate(self._normalized_segments()):
                if self._is_cancelled:
                    video.status = "cancelled"
                    session.commit()
                    self.log_message.emit("[CANCELLED] Segment processing cancelled")
                    return

                start_time = float(segment["start_time"])
                duration = float(segment["duration"])
                clip_id = f"{self.video_id}_manual_{idx}_{uuid.uuid4().hex[:8]}"
                clip_path = clips_dir / f"{clip_id}.mp4"
                self.progress_updated.emit("scene_split", idx, total)
                self.log_message.emit(
                    f"[CUT] {idx + 1}/{total}: {start_time:.2f}s + {duration:.2f}s -> {clip_path.name}"
                )

                self._cut_with_ffmpeg(ffmpeg_path, start_time, duration, clip_path)

                db_clip = Clip(
                    id=clip_id,
                    video_id=self.video_id,
                    clip_index=idx,
                    start_time=start_time,
                    end_time=start_time + duration,
                    duration=duration,
                    clip_path=str(clip_path.resolve()),
                    is_manual_segment=True,
                    status="needs_review" if self.processing_mode == "manual" else "pending",
                )
                session.add(db_clip)
                session.commit()
                created_clips.append(db_clip)

            self.progress_updated.emit("scene_split", total, total)
            self.stage_completed.emit("scene_split")
            video.total_clips = len(created_clips)
            session.commit()
            self.log_message.emit(f"[DONE] Created {len(created_clips)} clips")

            if self.processing_mode == "semi_auto":
                self._run_ai_on_clips(created_clips, session)

            video.status = "completed"
            session.commit()
            self.pipeline_finished.emit({
                "status": "completed",
                "video_id": self.video_id,
                "total_clips": len(created_clips),
                "processing_mode": self.processing_mode,
            })
        except Exception as exc:
            if session is not None:
                try:
                    from backend.database.models import Video
                    video = session.query(Video).filter(Video.id == self.video_id).first()
                    if video:
                        video.status = "error"
                        session.commit()
                except Exception:
                    session.rollback()
            self.log_message.emit(f"[ERROR] {exc}")
            self.log_message.emit(traceback.format_exc())
            self.error_occurred.emit(str(exc))
        finally:
            if session is not None:
                session.close()

    def cancel(self):
        self._is_cancelled = True
        self.log_message.emit("[INFO] Cancellation requested")

    def _normalized_segments(self) -> list[dict]:
        normalized = []
        for seg in sorted(self.segments, key=lambda s: float(s["start_time"])):
            start = float(seg["start_time"])
            end = float(seg["end_time"])
            duration = float(seg.get("duration", end - start))
            if end <= start or duration <= 0:
                continue
            normalized.append({"start_time": start, "end_time": end, "duration": duration})
        return normalized

    def _cut_with_ffmpeg(self, ffmpeg_path: str, start: float, duration: float, output: Path):
        startupinfo = None
        if os.name == "nt":
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW

        copy_cmd = [
            ffmpeg_path, "-y", "-ss", str(start), "-t", str(duration), "-i", self.video_path,
            "-vcodec", "copy", "-acodec", "copy", "-avoid_negative_ts", "1", str(output.resolve())
        ]
        try:
            subprocess.run(copy_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True, startupinfo=startupinfo)
        except subprocess.CalledProcessError as copy_error:
            self.log_message.emit("[WARN] Stream copy failed; re-encoding this segment")
            reencode_cmd = [
                ffmpeg_path, "-y", "-ss", str(start), "-t", str(duration), "-i", self.video_path,
                "-c:v", "libx264", "-preset", "veryfast", "-c:a", "aac", str(output.resolve())
            ]
            try:
                subprocess.run(reencode_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True, startupinfo=startupinfo)
            except subprocess.CalledProcessError as reencode_error:
                stderr = reencode_error.stderr.decode("utf-8", errors="replace") if reencode_error.stderr else ""
                raise RuntimeError(f"FFmpeg failed to cut segment: {stderr[-1000:]}") from copy_error

    def _run_ai_on_clips(self, clips, session):
        self.log_message.emit("[INFO] Semi-auto mode: running AI stages on manual clips")
        from backend.ai_models.model_manager import model_manager
        from backend.services.audio_extractor import AudioExtractor
        from backend.services.emotion_analyzer import EmotionAnalyzer
        from backend.services.face_extractor import FaceExtractor
        from backend.services.quality_scorer import QualityScorer
        from backend.services.transcriber import SpeechTranscriber

        self.progress_updated.emit("prewarm", 0, 100)
        loaded, failed = model_manager.prewarm_models()
        self.progress_updated.emit("prewarm", 100, 100)
        self.stage_completed.emit("prewarm")
        self.log_message.emit(f"[INFO] Models loaded={loaded}, failed={failed}")

        face_extractor = FaceExtractor()
        audio_extractor = AudioExtractor()
        transcriber = SpeechTranscriber()
        analyzer = EmotionAnalyzer()
        quality_scorer = QualityScorer()
        total = len(clips)

        for idx, clip in enumerate(clips):
            if self._is_cancelled:
                return
            try:
                self.progress_updated.emit("face_detect", idx, total)
                face_res = face_extractor.extract_faces_from_clip(clip.clip_path, clip.id)
                self.progress_updated.emit("face_detect", idx + 1, total)

                self.progress_updated.emit("audio_extract", idx, total)
                audio_res = audio_extractor.extract_audio_from_clip(clip.clip_path, clip.id)
                self.progress_updated.emit("audio_extract", idx + 1, total)

                self.progress_updated.emit("transcribe", idx, total)
                text_res = transcriber.transcribe_audio_clip(audio_res["audio_path"], clip.id)
                self.progress_updated.emit("transcribe", idx + 1, total)

                self.progress_updated.emit("emotion_label", idx, total)
                ai_res = analyzer.analyze_clip(
                    face_images=face_res["cropped_face_paths"],
                    transcript=text_res["transcript"],
                    audio_path=audio_res["audio_path"],
                )
                self.progress_updated.emit("emotion_label", idx + 1, total)

                self.progress_updated.emit("quality_score", idx, total)
                quality_res = quality_scorer.calculate_score(
                    confidence=ai_res["confidence"],
                    agreement_str=ai_res["agreement"],
                    sampled_frames_count=face_res["num_frames"],
                    cropped_faces_count=face_res["main_track_len"],
                    audio_clarity=audio_res["audio_clarity"],
                )
                self.progress_updated.emit("quality_score", idx + 1, total)

                clip.num_frames = face_res["num_frames"]
                clip.num_faces = face_res["num_faces"]
                clip.transcript = text_res["transcript"]
                clip.speaker_id = text_res["main_speaker"]
                clip.quality_score = quality_res["quality_score"]
                clip.status = quality_res["status"]
                clip.predicted_emotion = ai_res["predicted_emotion"]
                clip.confidence = ai_res["confidence"]
                clip.agreement = ai_res["agreement"]
                clip.has_incongruity = ai_res["has_incongruity"]
                clip.all_scores = ai_res["all_scores"]
                clip.per_model_scores = ai_res["per_model_scores"]
                session.commit()
                self.log_message.emit(f"[AI] Clip {idx + 1}/{total}: {clip.predicted_emotion} ({clip.confidence:.0%})")
            except Exception as exc:
                clip.status = "failed"
                clip.reviewer_notes = f"AI stage failed: {exc}"
                session.commit()
                self.log_message.emit(f"[WARN] AI failed for clip {idx + 1}: {exc}")
                continue
