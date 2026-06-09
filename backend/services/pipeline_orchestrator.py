import os
import json
import traceback
import cv2
from sqlalchemy.orm import Session
from backend.database.models import Video, Clip, Feature
from backend.config import settings

# Import tất cả các services của AI Pipeline
from backend.services.downloader import VideoDownloader
from backend.services.scene_splitter import SceneSplitter
from backend.services.smart_segmenter import SmartSegmenter
from backend.services.face_extractor import FaceExtractor
from backend.services.audio_extractor import AudioExtractor
from backend.services.transcriber import SpeechTranscriber
from backend.services.emotion_analyzer import EmotionAnalyzer
from backend.services.quality_scorer import QualityScorer
from backend.services.feature_extractors.audio_feature_extractor import AudioFeatureExtractor
from backend.services.feature_extractors.text_feature_extractor import TextFeatureExtractor
from backend.services.feature_extractors.visual_feature_extractor import VisualFeatureExtractor
from backend.services.feature_extractors.alignment_engine import AlignmentEngine

class PipelineOrchestrator:
    """Bộ điều phối trung tâm quản lý chạy tuần tự toàn bộ AI Pipeline từ đầu đến cuối."""
    
    # Singleton pattern — tránh tạo lại các service objects mỗi lần chạy
    _instance = None
    
    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super().__new__(cls, *args, **kwargs)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        self.downloader = VideoDownloader()
        self.splitter = SceneSplitter()
        self.smart_segmenter = SmartSegmenter(
            face_scan_fps=settings.SMART_FACE_SCAN_FPS,
            face_confidence=settings.SMART_FACE_CONFIDENCE,
            max_missing_face_gap=settings.SMART_MAX_MISSING_FACE_GAP,
            min_duration=settings.MIN_CLIP_DURATION,
            max_duration=settings.MAX_CLIP_DURATION,
            target_duration=settings.SMART_TARGET_CLIP_DURATION,
            silence_threshold_db=settings.SMART_SILENCE_THRESHOLD_DB,
            silence_min_duration=settings.SMART_SILENCE_MIN_DURATION,
            max_dialogue_extension=settings.SMART_MAX_DIALOGUE_EXTENSION,
            vad_mode=settings.SMART_VAD_MODE,
        )
        self.face_extractor = FaceExtractor()
        self.audio_extractor = AudioExtractor()
        self.transcriber = SpeechTranscriber()
        self.analyzer = EmotionAnalyzer()
        self.quality_scorer = QualityScorer()
        from backend.services.auto_decision import AutoDecisionEngine
        self.auto_decision = AutoDecisionEngine()
        # Feature extractors (run after pipeline, only on approved clips)
        self.audio_feat_ext = AudioFeatureExtractor()
        self.text_feat_ext = TextFeatureExtractor()
        self.visual_feat_ext = VisualFeatureExtractor()
        self.alignment_engine = AlignmentEngine()
        from backend.utils.resource_manager import resource_manager
        self.resource_plan = resource_manager.apply()
        print(f"⚙️ [Resource] device={self.resource_plan.device}, CPU threads={self.resource_plan.cpu_threads}, FFmpeg threads={self.resource_plan.ffmpeg_threads}, GPU={self.resource_plan.gpu_name or 'none'}")
        self._initialized = True

    def process_video(self, video_id: int | str, db: Session | None = None, progress_callback=None):
        """Alias for run_pipeline — used by web/main.py pipeline worker."""
        if db is None:
            from backend.database.local_db import SessionLocal as _Session
            db = _Session()
            _should_close = True
        else:
            _should_close = False
        try:
            self.run_pipeline(str(video_id), db, progress_callback)
        finally:
            if _should_close:
                db.close()

    def run_pipeline(self, video_id: str, db: Session, progress_callback=None):
        """Khởi chạy toàn bộ Pipeline xử lý video gốc và lưu trữ các clips phân cảnh."""
        print(f"🎬 [Pipeline] Bắt đầu chạy Orchestrator cho Video ID: {video_id}")
        
        video = db.query(Video).filter(Video.id == video_id).first()
        if not video:
            print(f"❌ [Pipeline] Không tìm thấy video ID {video_id} trong DB")
            return
            
        try:
            # --- STAGE 1: DOWNLOAD VIDEO (nếu là YouTube URL) ---
            video.status = "processing"
            db.commit()
            
            video_file_path = video.file_path
            
            if video.source_url and not video_file_path:
                print(f"📥 [Pipeline] Stage 1: Đang tải video từ YouTube URL: {video.source_url}...")
                
                last_pct = -1
                def download_hook(d):
                    nonlocal last_pct
                    if d['status'] == 'downloading':
                        total = d.get('total_bytes') or d.get('total_bytes_estimate') or 0
                        downloaded = d.get('downloaded_bytes', 0)
                        if total > 0:
                            pct = int(downloaded / total * 100)
                            if pct != last_pct:
                                last_pct = pct
                                if progress_callback:
                                    progress_callback("download", downloaded, total, f"📥 [Download] Đang tải: {pct}% ({downloaded / 1024 / 1024:.1f}MB / {total / 1024 / 1024:.1f}MB)")
                        else:
                            megabytes_downloaded = int(downloaded / 1024 / 1024)
                            if megabytes_downloaded % 5 == 0 and megabytes_downloaded != last_pct:
                                last_pct = megabytes_downloaded
                                if progress_callback:
                                    progress_callback("download", downloaded, 0, f"📥 [Download] Đang tải: {downloaded / 1024 / 1024:.1f}MB (chưa rõ dung lượng)")
                    elif d['status'] == 'finished':
                        if progress_callback:
                            progress_callback("download", 100, 100, "📥 [Download] Đã tải xong video, đang hoàn thiện tệp...")

                if progress_callback:
                    progress_callback("download", 0, 100, f"📥 [Download] Bắt đầu tải video từ YouTube URL...")

                download_res = self.downloader.download(video.source_url, progress_hook=download_hook)
                
                # Cập nhật thông tin video sau tải
                video.file_path = download_res["file_path"]
                video.duration_sec = download_res["duration_sec"]
                video.resolution = download_res["resolution"]
                if download_res.get("title") and (not video.title or video.title == "Unknown" or video.title == "Untitled"):
                    video.title = download_res["title"]
                video_file_path = download_res["file_path"]
                db.commit()
                print(f"✅ [Pipeline] Stage 1: Tải thành công video. Độ dài: {video.duration_sec}s")
                if progress_callback:
                    progress_callback("title_retrieved", 0, 0, video.title)
                    progress_callback("download", 100, 100, f"📥 [Download] Tải thành công video. Độ dài: {video.duration_sec}s")
            else:
                if progress_callback:
                    # Đảm bảo UI cập nhật tên file cục bộ nếu nó là Unknown/Untitled
                    if video.title and progress_callback:
                        progress_callback("title_retrieved", 0, 0, video.title)
                    progress_callback("download", 100, 100, "📥 [Download] Sử dụng file video cục bộ có sẵn")
                
            if not video_file_path or not os.path.exists(video_file_path):
                raise FileNotFoundError(f"Không tìm thấy file video cục bộ tại: {video_file_path}")
                
            # --- STAGE 2: DETECT & SPLIT SCENES ---
            print("✂️ [Pipeline] Stage 2: Đang tự động phát hiện chuyển cảnh và cắt video...")
            if progress_callback:
                progress_callback("scene_split", 0, 100, "✂️ [Scene Split] Bắt đầu phát hiện chuyển cảnh...")
            scenes = self.splitter.detect_scenes(video_file_path, threshold=settings.SCENE_THRESHOLD)
            
            # Logic Fallback: Nếu không phát hiện điểm chuyển cảnh nào, coi cả video là 1 scene duy nhất
            if len(scenes) == 0:
                print("⚠️ [Pipeline] Không phát hiện điểm chuyển cảnh. Tự động tạo 1 scene duy nhất cho toàn bộ video.")
                duration = video.duration_sec
                if not duration:
                    # Lấy duration bằng OpenCV
                    cap = cv2.VideoCapture(video_file_path)
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                    duration = total_frames / fps if fps > 0 else 5.0
                    cap.release()
                
                scenes = [{
                    "scene_index": 0,
                    "start_time": 0.0,
                    "end_time": duration,
                    "duration": duration
                }]
                
            print(f"ℹ️ [Pipeline] Tổng số {len(scenes)} cảnh thô. Bắt đầu smart segmentation theo mặt người/hội thoại...")
            if progress_callback:
                progress_callback("scene_split", 35, 100, f"✂️ [Smart Segment] Phân tích mặt người và vùng hội thoại trong {len(scenes)} cảnh...")

            try:
                smart_scenes = self.smart_segmenter.build_segments(video_file_path, scenes, video_id=video_id)
                if smart_scenes:
                    scenes_for_split = smart_scenes
                    print(f"✅ [Pipeline] SmartSegmenter tạo {len(scenes_for_split)} candidate clips face/dialogue-aware")
                    if progress_callback:
                        progress_callback("scene_split", 70, 100, f"✂️ [Smart Segment] Tạo {len(scenes_for_split)} đoạn có mặt/hội thoại")
                else:
                    scenes_for_split = scenes
                    print("⚠️ [Pipeline] SmartSegmenter không tạo candidate nào, fallback scene-only")
                    if progress_callback:
                        progress_callback("scene_split", 70, 100, "✂️ [Smart Segment] Không thấy đoạn có mặt rõ, fallback cắt theo scene")
            except Exception as smart_err:
                scenes_for_split = scenes
                print(f"⚠️ [Pipeline] SmartSegmenter lỗi, fallback scene-only: {smart_err}")
                if progress_callback:
                    progress_callback("scene_split", 70, 100, f"✂️ [Smart Segment] Fallback scene-only: {smart_err}")
            
            # Cắt video theo smart candidates và lọc theo thời lượng
            clips_metadata = self.splitter.split_video(
                video_file_path,
                scenes_for_split,
                video_id,
                min_duration=settings.MIN_CLIP_DURATION,
                max_duration=settings.MAX_CLIP_DURATION,
            )
            total_clips = len(clips_metadata)
            
            video.total_clips = total_clips
            db.commit()
            print(f"✅ [Pipeline] Stage 2: Hoàn tất. Đã tạo ra {total_clips} clips hợp lệ để xử lý AI")
            if progress_callback:
                progress_callback("scene_split", 100, 100, f"✂️ [Scene Split] Hoàn tất. Đã tạo ra {total_clips} clips hợp lệ")

            # ── GEMINI PRE-FILTERING (Vertex AI) ─────────────────────────────────
            # Dùng Gemini để scan toàn video, tìm các đoạn có cảm xúc mạnh
            # trước khi chạy pipeline AI tốn GPU. Giảm số clips cần xử lý.
            if settings.GCP_PROJECT_ID and video_file_path and os.path.exists(video_file_path):
                print("🤖 [Gemini] Bắt đầu pre-filtering với Gemini 2.5 Flash...")
                if progress_callback:
                    progress_callback("gemini_prefilter", 0, 100, "🤖 [Gemini] Đang phân tích video bằng Gemini...")
                try:
                    from backend.services.gemini_auto_labeler import GeminiAutoLabeler
                    labeler = GeminiAutoLabeler()
                    gem_result = labeler.analyze_video(
                        video_path=video_file_path,
                        intensity_threshold=settings.GEMINI_INTENSITY_THRESHOLD,
                        max_segments=30,
                    )
                    gem_segments = gem_result.get("segments", [])
                    print(f"🤖 [Gemini] Tìm thấy {len(gem_segments)} đoạn cảm xúc mạnh (intensity ≥ {settings.GEMINI_INTENSITY_THRESHOLD})")
                    if progress_callback:
                        progress_callback("gemini_prefilter", 80, 100,
                                         f"🤖 [Gemini] Tìm thấy {len(gem_segments)} đoạn cảm xúc mạnh")
                    # Tag clips that fall within emotional segments
                    for cm in clips_metadata:
                        in_gem_segment = any(
                            cm["start_time"] >= seg["start_time"] and cm["end_time"] <= seg["end_time"]
                            for seg in gem_segments
                        )
                        cm["gem_intensity"] = max(
                            (seg["intensity"] for seg in gem_segments
                             if cm["start_time"] >= seg["start_time"] and cm["end_time"] <= seg["end_time"]),
                            default=0.0
                        )
                        cm["in_gem_segment"] = in_gem_segment
                    if progress_callback:
                        progress_callback("gemini_prefilter", 100, 100,
                                         f"🤖 [Gemini] Done. {len(gem_segments)} emotional segments found.")
                except Exception as gem_err:
                    print(f"⚠️ [Gemini] Pre-filtering bị lỗi, bỏ qua: {gem_err}")
                    if progress_callback:
                        progress_callback("gemini_prefilter", 100, 100,
                                         f"⚠️ [Gemini] Bỏ qua: {gem_err}")
                    for cm in clips_metadata:
                        cm["gem_intensity"] = 0.0
                        cm["in_gem_segment"] = False
            else:
                for cm in clips_metadata:
                    cm["gem_intensity"] = 0.0
                    cm["in_gem_segment"] = False

            # --- STAGES 3 & 4: MULTI-TRACK PROCESSING & MULTI-MODEL LABELING ---
            
            # Pre-warm: Load TẤT CẢ AI models một lần trước khi bắt đầu xử lý clips
            # Tránh việc load rời rạc gây chậm cho clip đầu tiên
            if total_clips > 0:
                print("🔥 [Pipeline] Pre-warming AI models trước khi xử lý clips...")
                if progress_callback:
                    progress_callback("prewarm", 0, 100, "🔥 [Pre-warm] Đang nạp AI models lên bộ nhớ...")
                from backend.ai_models.model_manager import model_manager
                loaded, failed = model_manager.prewarm_models()
                if failed:
                    for status in model_manager.status():
                        if status.error:
                            print(f"⚠️ [Pre-warm] Model {status.key} lỗi: {status.error[:300]}")
                if progress_callback:
                    progress_callback("prewarm", 100, 100, 
                                      f"🔥 [Pre-warm] Hoàn tất: {loaded} models loaded, {failed} failed")
            
            import time as _time
            
            for idx, clip_meta in enumerate(clips_metadata):
                clip_index = clip_meta["clip_index"]
                clip_path = clip_meta["clip_path"]
                clip_id = f"{video_id}_clip_{clip_index}"
                
                print(f"\n⚡ [Pipeline] Xử lý Clip {clip_index + 1}/{total_clips}: {clip_id}")
                
                if progress_callback and not progress_callback("check_cancel", 0, 0, ""):
                    raise Exception("Pipeline bị hủy bởi người dùng")
                
                try:
                    # 1. Trích xuất hình ảnh (Face Extraction & Tracking)
                    print(f"  [Visual] Detect khuôn mặt...")
                    if progress_callback:
                        progress_callback("face_detect", idx, total_clips, f"👤 [Visual] Detect khuôn mặt clip {idx + 1}/{total_clips}...")
                    face_res = self.face_extractor.extract_faces_from_clip(clip_path, clip_id)
                    if progress_callback:
                        progress_callback("face_detect", idx + 1, total_clips, "")
                    
                    if progress_callback and not progress_callback("check_cancel", 0, 0, ""):
                        raise Exception("Pipeline bị hủy bởi người dùng")

                    # 2. Trích xuất âm thanh (Audio extraction & MFCC)
                    print(f"  [Audio] Tách WAV & tính MFCC...")
                    if progress_callback:
                        progress_callback("audio_extract", idx, total_clips, f"🔊 [Audio] Tách WAV & tính MFCC clip {idx + 1}/{total_clips}...")
                    audio_res = self.audio_extractor.extract_audio_from_clip(clip_path, clip_id)
                    if progress_callback:
                        progress_callback("audio_extract", idx + 1, total_clips, "")

                    if progress_callback and not progress_callback("check_cancel", 0, 0, ""):
                        raise Exception("Pipeline bị hủy bởi người dùng")
                    
                    # 3. Chuyển đổi giọng nói thành văn bản (Speech-to-Text)
                    print(f"  [Text] Chạy Whisper Speech-to-Text tiếng Việt...")
                    if progress_callback:
                        progress_callback("transcribe", idx, total_clips, f"📝 [Text] Chạy Whisper Speech-to-Text clip {idx + 1}/{total_clips}...")
                    text_res = self.transcriber.transcribe_audio_clip(audio_res["audio_path"], clip_id)
                    if text_res.get("warning"):
                        print(f"  ⚠️ [Text] {text_res.get('warning')}")
                    elif not text_res.get("transcript"):
                        print("  ⚠️ [Text] Không nhận diện được lời thoại trong clip này")
                    if progress_callback:
                        progress_callback("transcribe", idx + 1, total_clips, "")

                    if progress_callback and not progress_callback("check_cancel", 0, 0, ""):
                        raise Exception("Pipeline bị hủy bởi người dùng")
                    
                    # 4. Phân tích đa phương thức kết hợp (Multi-model Ensemble Voting)
                    print(f"  [AI Ensemble] Chạy Ensemble Voting cảm xúc...")
                    if progress_callback:
                        progress_callback("emotion_label", idx, total_clips, f"🎭 [AI Ensemble] Chạy Ensemble Voting cảm xúc clip {idx + 1}/{total_clips}...")
                    ai_res = self.analyzer.analyze_clip(
                        face_images=face_res["cropped_face_paths"],
                        transcript=text_res["transcript"],
                        audio_path=audio_res["audio_path"]
                    )
                    if progress_callback:
                        progress_callback("emotion_label", idx + 1, total_clips, "")

                    if progress_callback and not progress_callback("check_cancel", 0, 0, ""):
                        raise Exception("Pipeline bị hủy bởi người dùng")
                    
                    # 5. Chấm điểm chất lượng và Định tuyến trạng thái duyệt tự động
                    print(f"  [Quality] Chấm điểm chất lượng...")
                    if progress_callback:
                        progress_callback("quality_score", idx, total_clips, f"⭐ [Quality] Chấm điểm chất lượng clip {idx + 1}/{total_clips}...")
                    quality_res = self.quality_scorer.calculate_score(
                        confidence=ai_res["confidence"],
                        agreement_str=ai_res["agreement"],
                        sampled_frames_count=face_res["num_frames"],
                        cropped_faces_count=face_res["main_track_len"],
                        audio_clarity=audio_res["audio_clarity"]
                    )
                    if progress_callback:
                        progress_callback("quality_score", idx + 1, total_clips, "")
                    
                    segment_meta = clip_meta.get("segment_metadata", {}) or {}
                    audio_clarity = float(audio_res.get("audio_clarity") or 0.0)
                    snr_db = audio_res.get("snr_db")
                    if snr_db is None:
                        snr_db = max(0.0, min(30.0, audio_clarity * 30.0))
                    face_ratio = segment_meta.get("face_ratio")
                    if face_ratio is None:
                        face_ratio = (face_res.get("main_track_len", 0) / max(face_res.get("num_frames", 1), 1))
                    frontal_ratio = segment_meta.get("frontal_ratio", segment_meta.get("head_pose_ok", 1.0))
                    face_quality = segment_meta.get("face_quality", quality_res.get("quality_score", 0.0))
                    decision = self.auto_decision.decide({
                        "duration": clip_meta["duration"],
                        "num_faces": face_res["num_faces"],
                        "transcript": text_res["transcript"],
                        "quality_score": quality_res["quality_score"],
                        "confidence": ai_res["confidence"],
                        "agreement": ai_res["agreement"],
                        "has_incongruity": ai_res["has_incongruity"],
                        "predicted_emotion": ai_res["predicted_emotion"],
                        "snr_db": snr_db,
                        "frontal_ratio": frontal_ratio,
                        "face_quality": face_quality,
                    })
                    model_scores = ai_res.get("per_model_scores") or {}
                    face_scores = model_scores.get("visual_deepface") or {}
                    voice_scores = model_scores.get("audio_wav2vec") or {}
                    text_scores = model_scores.get("text_vi_lexicon") or {}
                    face_winner = max(face_scores, key=face_scores.get) if face_scores else None
                    voice_winner = max(voice_scores, key=voice_scores.get) if voice_scores else None
                    text_winner = max(text_scores, key=text_scores.get) if text_scores else None

                    # 6. Lưu thông tin clip chi tiết vào cơ sở dữ liệu
                    db_clip = Clip(
                        id=clip_id,
                        video_id=video_id,
                        clip_index=clip_index,
                        start_time=clip_meta["start_time"],
                        end_time=clip_meta["end_time"],
                        duration=clip_meta["duration"],
                        clip_path=clip_path,
                        num_frames=face_res["num_frames"],
                        num_faces=face_res["num_faces"],
                        face_ratio=segment_meta.get("face_coverage", face_ratio),
                        frontal_ratio=segment_meta.get("frontal_ratio", frontal_ratio),
                        avg_yaw=segment_meta.get("avg_yaw"),
                        avg_face_size=segment_meta.get("avg_face_size"),
                        face_quality=face_quality,
                        transcript=text_res["transcript"],
                        transcript_conf=text_res.get("confidence"),
                        speaker_id=text_res["main_speaker"],
                        audio_path=audio_res.get("audio_path"),
                        snr_db=snr_db,
                        num_speakers=text_res.get("num_speakers", 1 if text_res.get("transcript") else 0),
                        has_speech=bool(text_res.get("transcript") or audio_res.get("has_speech_energy")),
                        quality_score=quality_res["quality_score"],
                        status=decision.status,
                        predicted_emotion=ai_res["predicted_emotion"],
                        emotion_face=face_winner,
                        emotion_face_conf=face_scores.get(face_winner) if face_scores else None,
                        emotion_voice=voice_winner,
                        emotion_voice_conf=voice_scores.get(voice_winner) if voice_scores else None,
                        emotion_text=text_winner,
                        emotion_text_conf=text_scores.get(text_winner) if text_scores else None,
                        confidence=ai_res["confidence"],
                        agreement=ai_res["agreement"],
                        has_incongruity=ai_res["has_incongruity"],
                        decision_by=decision.decision_by,
                        reject_reason=decision.reject_reason,
                        pipeline_stage="stage_4_done",
                        all_scores=ai_res["all_scores"],
                        per_model_scores={
                            **(ai_res.get("per_model_scores") or {}),
                            "gemini_prefilter": {
                                "gem_intensity": clip_meta.get("gem_intensity", 0.0),
                                "in_emotional_segment": clip_meta.get("in_gem_segment", False),
                                "face_coverage": segment_meta.get("face_coverage"),
                                "speech_coverage": segment_meta.get("speech_coverage"),
                            },
                            "auto_decision": {
                                "status": decision.status,
                                "decision_by": decision.decision_by,
                                "reject_reason": decision.reject_reason,
                            },
                            "segment": segment_meta,
                            "face_extraction": {
                                "detector": face_res.get("detector"),
                                "face_paths": face_res.get("cropped_face_paths", []),
                                "detections_path": face_res.get("detections_path"),
                                "track_count": face_res.get("track_count", 0),
                            },
                            "audio_features": {
                                "audio_path": audio_res.get("audio_path"),
                                "audio_clarity": audio_res.get("audio_clarity"),
                                "has_speech_energy": audio_res.get("has_speech_energy"),
                            },
                            "transcriber": {
                                "transcript": text_res.get("transcript"),
                                "segments": text_res.get("segments", []),
                                "language": text_res.get("language"),
                                "main_speaker": text_res.get("main_speaker"),
                            },
                        }
                    )
                    db.add(db_clip)
                    db.commit()
                    print(f"✅ [Pipeline] Xử lý thành công Clip #{clip_index}. Trạng thái định tuyến: {quality_res['status'].upper()}")
                    
                except Exception as clip_err:
                    print(f"❌ [Pipeline] Lỗi khi xử lý Clip #{clip_index}: {clip_err}")
                    traceback.print_exc()
                    if "bị hủy bởi người dùng" in str(clip_err):
                        raise clip_err
                    # Vẫn tiếp tục xử lý các clips khác nếu 1 clip gặp lỗi
                    continue
            
            # Cập nhật video hoàn tất
            video.status = "completed"
            # Cập nhật số lượng approved_clips tự động
            approved_count = db.query(Clip).filter(
                Clip.video_id == video_id,
                Clip.status.in_(["approved", "auto_approved"])
            ).count()
            video.approved_clips = approved_count
            video.num_clips_raw = total_clips
            video.num_clips_ok = approved_count
            db.commit()
            print(f"\n🎉 [Pipeline] HOÀN TẤT THÀNH CÔNG toàn bộ quy trình cho Video ID: {video_id}")

            # ── STAGE 6: FEATURE EXTRACTION (chỉ chạy cho approved clips) ───────────
            if approved_count > 0:
                self._run_feature_extraction(
                    video_id=video_id,
                    db=db,
                    video_file_path=video_file_path,
                    progress_callback=progress_callback,
                )

            # Đảm bảo tất cả các stage kết thúc đều hiển thị 100%
            if progress_callback:
                for stage_key in ["download", "scene_split", "face_detect", "audio_extract", "transcribe", "emotion_label", "quality_score", "feature_extract"]:
                    progress_callback(stage_key, 100, 100, "")

        except Exception as e:
            print(f"❌ [Pipeline] Gặp lỗi nghiêm trọng dừng pipeline của Video {video_id}: {e}")
            traceback.print_exc()
            if "bị hủy bởi người dùng" in str(e) or "cancelled" in str(e).lower():
                video.status = "cancelled"
            else:
                video.status = "error"
            db.commit()
            raise e

    def _run_feature_extraction(
        self,
        video_id: str,
        db: Session,
        video_file_path: str,
        progress_callback=None,
    ):
        """Stage 6: Extract MMSA-compatible features for all approved clips.

        Runs text, audio, and vision feature extraction, then alignment,
        and saves results to the Feature table.
        """
        print(f"\n🔧 [Feature] Stage 6: Extracting MMSA features for approved clips...")
        if progress_callback:
            progress_callback("feature_extract", 0, 100, "🔧 [Feature] Bắt đầu trích xuất features cho approved clips...")

        approved_clips = db.query(Clip).filter(
            Clip.video_id == video_id,
            Clip.status.in_(["approved", "auto_approved"])
        ).all()

        total = len(approved_clips)
        success_count = 0
        error_count = 0

        # Output directory for aligned features
        feature_base_dir = settings.DATA_DIR / "features" / video_id
        feature_base_dir.mkdir(parents=True, exist_ok=True)

        for idx, clip in enumerate(approved_clips):
            if progress_callback and not progress_callback("check_cancel", 0, 0, ""):
                break

            clip_id = clip.id
            try:
                # Get word timestamps from Whisper segments
                segments = clip.per_model_scores.get("transcriber", {}).get("segments", [])
                word_timestamps = []
                for seg in segments:
                    if isinstance(seg, dict):
                        for w in seg.get("words", []):
                            if isinstance(w, dict):
                                word_timestamps.append({
                                    "word": w.get("word", ""),
                                    "start": w.get("start", 0.0),
                                    "end": w.get("end", 0.0),
                                })
                    elif hasattr(seg, "words"):
                        for w in getattr(seg, "words", []):
                            if hasattr(w, "word"):
                                word_timestamps.append({
                                    "word": getattr(w, "word", ""),
                                    "start": getattr(w, "start", 0.0),
                                    "end": getattr(w, "end", 0.0),
                                })

                # 1. Text features (PhoBERT)
                text_result = self.text_feat_ext.extract_features(
                    transcript=clip.transcript or "",
                    clip_id=clip_id,
                    word_timestamps=word_timestamps or None,
                )

                # 2. Audio features (Librosa 74-dim)
                audio_result = self.audio_feat_ext.extract_features(
                    audio_path=clip.audio_path,
                    clip_id=clip_id,
                    word_timestamps=word_timestamps or None,
                )

                # 3. Vision features (OpenFace or Py-Feat 35-AU)
                detections_path = clip.per_model_scores.get("face_extraction", {}).get("detections_path")
                vision_result = self.visual_feat_ext.extract_features(
                    clip_path=clip.clip_path,
                    clip_id=clip_id,
                    detections_path=detections_path,
                    word_timestamps=word_timestamps or None,
                )

                # 4. Alignment (normalize shapes to (50, D))
                alignment_result = self.alignment_engine.align(
                    text_features=text_result.get("features"),
                    audio_features=audio_result.get("features"),
                    vision_features=vision_result.get("features"),
                    word_timestamps=word_timestamps or None,
                )

                # 5. Save aligned .npy files
                saved_paths = self.alignment_engine.save_aligned_features(
                    output_dir=feature_base_dir,
                    clip_id=clip_id,
                    text_features=alignment_result["text_aligned"],
                    audio_features=alignment_result["audio_aligned"],
                    vision_features=alignment_result["vision_aligned"],
                )

                # 6. Upsert Feature record
                existing = db.query(Feature).filter(Feature.clip_id == clip_id).first()
                if existing:
                    existing.text_path = saved_paths.get("text_path", "")
                    existing.audio_path = saved_paths.get("audio_path", "")
                    existing.vision_path = saved_paths.get("vision_path", "")
                    existing.text_shape = "(50, 768)"
                    existing.audio_shape = "(50, 74)"
                    existing.vision_shape = "(50, 35)"
                    existing.aligned = bool(word_timestamps)
                else:
                    feat_record = Feature(
                        clip_id=clip_id,
                        text_path=saved_paths.get("text_path", ""),
                        audio_path=saved_paths.get("audio_path", ""),
                        vision_path=saved_paths.get("vision_path", ""),
                        text_shape="(50, 768)",
                        audio_shape="(50, 74)",
                        vision_shape="(50, 35)",
                        aligned=bool(word_timestamps),
                    )
                    db.add(feat_record)

                db.commit()
                success_count += 1

            except Exception as feat_err:
                error_count += 1
                print(f"  ⚠️ [Feature] Clip {clip_id} lỗi feature extraction: {feat_err}")
                traceback.print_exc()
                db.rollback()
                continue

            if progress_callback:
                progress_callback(
                    "feature_extract",
                    idx + 1,
                    total,
                    f"🔧 [Feature] {idx + 1}/{total}: {clip_id}",
                )

        print(f"✅ [Feature] Hoàn tất: {success_count} thành công, {error_count} lỗi")
        if progress_callback:
            progress_callback(
                "feature_extract",
                total,
                total,
                f"🔧 [Feature] Hoàn tất: {success_count}/{total} clips có features",
            )
