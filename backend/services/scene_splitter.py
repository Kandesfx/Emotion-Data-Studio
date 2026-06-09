import os
from pathlib import Path
from typing import List, Dict, Any, Optional
import subprocess

from scenedetect import SceneManager, open_video
from scenedetect.detectors import ContentDetector
from backend.config import settings


class SceneSplitter:
    """Detect scenes and split videos into bounded clips for emotion labeling."""

    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = output_dir or (settings.DATA_DIR / "clips")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def detect_scenes(self, video_path: str, threshold: float = 30.0) -> List[Dict[str, float]]:
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Không tìm thấy video gốc tại: {video_path}")

        try:
            video = open_video(video_path)
            scene_manager = SceneManager()
            scene_manager.add_detector(ContentDetector(threshold=threshold))
            scene_manager.detect_scenes(video)
            scene_list = scene_manager.get_scene_list()

            scenes = []
            for i, scene in enumerate(scene_list):
                start_sec = scene[0].get_seconds()
                end_sec = scene[1].get_seconds()
                duration = max(0.0, end_sec - start_sec)
                if duration > 0:
                    scenes.append({
                        "scene_index": i,
                        "start_time": start_sec,
                        "end_time": end_sec,
                        "duration": duration,
                    })
            return scenes
        except Exception as e:
            raise RuntimeError(f"Lỗi detect_scenes của PySceneDetect: {e}") from e

    def split_video(
        self,
        video_path: str,
        scenes: List[Dict[str, Any]],
        video_id: str,
        min_duration: float = 3.0,
        max_duration: float = 15.0,
    ) -> List[Dict[str, Any]]:
        """Split video with FFmpeg.

        Long scenes are chunked into <= max_duration clips instead of being dropped.
        Very short scenes are skipped unless they are the only available segment.
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Không tìm thấy video gốc tại: {video_path}")

        import shutil as _shutil
        from backend.utils.resource_manager import resource_manager
        ffmpeg_path = settings.FFMPEG_PATH
        plan = resource_manager.apply()
        if _shutil.which(ffmpeg_path) is None:
            raise RuntimeError(
                f"FFmpeg không khả dụng tại '{ffmpeg_path}'. Cài FFmpeg hoặc cấu hình FFMPEG_PATH."
            )

        normalized_segments = self._normalize_segments(scenes, min_duration, max_duration)
        if not normalized_segments:
            raise RuntimeError(
                "Không tạo được clip hợp lệ. Hãy giảm min_duration, tăng max_duration hoặc dùng Manual Segment Editor."
            )

        split_clips = []
        for valid_clip_idx, scene in enumerate(normalized_segments):
            duration = scene["duration"]
            start_time = scene["start_time"]
            clip_name = f"{video_id}_clip_{valid_clip_idx}.mp4"
            clip_path = self.output_dir / clip_name

            cmd = [
                ffmpeg_path,
                "-y",
                "-threads", str(plan.ffmpeg_threads),
                "-ss", str(start_time),
                "-t", str(duration),
                "-i", video_path,
                "-vcodec", "copy",
                "-acodec", "copy",
                "-avoid_negative_ts", "1",
                str(clip_path.resolve()),
            ]

            try:
                startupinfo = None
                if os.name == "nt":
                    startupinfo = subprocess.STARTUPINFO()
                    startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW

                subprocess.run(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    check=True,
                    startupinfo=startupinfo,
                )

                split_clips.append({
                    "clip_index": valid_clip_idx,
                    "start_time": start_time,
                    "end_time": scene["end_time"],
                    "duration": duration,
                    "clip_path": str(clip_path.resolve()),
                    "segment_metadata": {
                        key: value for key, value in scene.items()
                        if key not in {"start_time", "end_time", "duration"}
                    },
                })
            except subprocess.CalledProcessError as e:
                stderr = e.stderr.decode("utf-8", errors="replace") if e.stderr else ""
                print(f"❌ FFmpeg lỗi khi cắt {clip_name}: {stderr[-800:]}")
                continue

        if not split_clips:
            raise RuntimeError("FFmpeg không tạo được clip nào. Kiểm tra codec/file nguồn hoặc thử re-encode video.")
        return split_clips

    def _normalize_segments(
        self,
        scenes: List[Dict[str, Any]],
        min_duration: float,
        max_duration: float,
    ) -> List[Dict[str, Any]]:
        segments: List[Dict[str, Any]] = []
        for scene in scenes:
            start = float(scene.get("start_time", 0.0))
            end = float(scene.get("end_time", start))
            duration = max(0.0, float(scene.get("duration", end - start)))
            if duration <= 0:
                continue
            if duration < min_duration:
                if len(scenes) == 1:
                    segment = dict(scene)
                    segment.update({"start_time": start, "end_time": end, "duration": duration})
                    segments.append(segment)
                continue
            if duration <= max_duration:
                segment = dict(scene)
                segment.update({"start_time": start, "end_time": end, "duration": duration})
                segments.append(segment)
                continue

            cursor = start
            while cursor + min_duration <= end:
                chunk_end = min(cursor + max_duration, end)
                chunk_duration = chunk_end - cursor
                if chunk_duration >= min_duration:
                    segment = dict(scene)
                    segment.update({
                        "start_time": cursor,
                        "end_time": chunk_end,
                        "duration": chunk_duration,
                    })
                    segment["parent_segment_start"] = start
                    segment["parent_segment_end"] = end
                    segments.append(segment)
                cursor = chunk_end
        return segments
