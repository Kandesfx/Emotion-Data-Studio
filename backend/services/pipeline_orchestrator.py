import os
import json
import traceback
import cv2
from sqlalchemy.orm import Session
from backend.database.models import Video, Clip
from backend.config import settings

# Import tất cả các services của AI Pipeline
from backend.services.downloader import VideoDownloader
from backend.services.scene_splitter import SceneSplitter
from backend.services.face_extractor import FaceExtractor
from backend.services.audio_extractor import AudioExtractor
from backend.services.transcriber import SpeechTranscriber
from backend.services.emotion_analyzer import EmotionAnalyzer
from backend.services.quality_scorer import QualityScorer

class PipelineOrchestrator:
    """Bộ điều phối trung tâm quản lý chạy tuần tự toàn bộ AI Pipeline từ đầu đến cuối."""
    
    def __init__(self):
        self.downloader = VideoDownloader()
        self.splitter = SceneSplitter()
        self.face_extractor = FaceExtractor()
        self.audio_extractor = AudioExtractor()
        self.transcriber = SpeechTranscriber()
        self.analyzer = EmotionAnalyzer()
        self.quality_scorer = QualityScorer()

    def run_pipeline(self, video_id: str, db: Session):
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
                download_res = self.downloader.download(video.source_url)
                
                # Cập nhật thông tin video sau tải
                video.file_path = download_res["file_path"]
                video.duration_sec = download_res["duration_sec"]
                video.resolution = download_res["resolution"]
                video_file_path = download_res["file_path"]
                db.commit()
                print(f"✅ [Pipeline] Stage 1: Tải thành công video. Độ dài: {video.duration_sec}s")
                
            if not video_file_path or not os.path.exists(video_file_path):
                raise FileNotFoundError(f"Không tìm thấy file video cục bộ tại: {video_file_path}")
                
            # --- STAGE 2: DETECT & SPLIT SCENES ---
            print("✂️ [Pipeline] Stage 2: Đang tự động phát hiện chuyển cảnh và cắt video...")
            scenes = self.splitter.detect_scenes(video_file_path)
            
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
                
            print(f"ℹ️ [Pipeline] Tổng số {len(scenes)} cảnh được đưa vào xử lý. Bắt đầu lọc và cắt FFmpeg...")
            
            # Cắt video và lọc theo thời lượng (3s-15s)
            clips_metadata = self.splitter.split_video(video_file_path, scenes, video_id)
            total_clips = len(clips_metadata)
            
            video.total_clips = total_clips
            db.commit()
            print(f"✅ [Pipeline] Stage 2: Hoàn tất. Đã tạo ra {total_clips} clips hợp lệ để xử lý AI")
            
            # --- STAGES 3 & 4: MULTI-TRACK PROCESSING & MULTI-MODEL LABELING ---
            for idx, clip_meta in enumerate(clips_metadata):
                clip_index = clip_meta["clip_index"]
                clip_path = clip_meta["clip_path"]
                clip_id = f"{video_id}_clip_{clip_index}"
                
                print(f"\n⚡ [Pipeline] Xử lý Clip {clip_index + 1}/{total_clips}: {clip_id}")
                
                try:
                    # 1. Trích xuất hình ảnh (Face Extraction & Tracking)
                    print(f"  [Visual] Detect khuôn mặt...")
                    face_res = self.face_extractor.extract_faces_from_clip(clip_path, clip_id)
                    
                    # 2. Trích xuất âm thanh (Audio extraction & MFCC)
                    print(f"  [Audio] Tách WAV & tính MFCC...")
                    audio_res = self.audio_extractor.extract_audio_from_clip(clip_path, clip_id)
                    
                    # 3. Chuyển đổi giọng nói thành văn bản (Speech-to-Text)
                    print(f"  [Text] Chạy Whisper Speech-to-Text tiếng Việt...")
                    text_res = self.transcriber.transcribe_audio_clip(audio_res["audio_path"], clip_id)
                    
                    # 4. Phân tích đa phương thức kết hợp (Multi-model Ensemble Voting)
                    print(f"  [AI Ensemble] Chạy Ensemble Voting cảm xúc...")
                    ai_res = self.analyzer.analyze_clip(
                        face_images=face_res["cropped_face_paths"],
                        transcript=text_res["transcript"],
                        audio_path=audio_res["audio_path"]
                    )
                    
                    # 5. Chấm điểm chất lượng và Định tuyến trạng thái duyệt tự động
                    print(f"  [Quality] Chấm điểm chất lượng...")
                    quality_res = self.quality_scorer.calculate_score(
                        confidence=ai_res["confidence"],
                        agreement_str=ai_res["agreement"],
                        sampled_frames_count=face_res["num_frames"],
                        cropped_faces_count=face_res["main_track_len"],
                        audio_clarity=audio_res["audio_clarity"]
                    )
                    
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
                        transcript=text_res["transcript"],
                        speaker_id=text_res["main_speaker"],
                        quality_score=quality_res["quality_score"],
                        status=quality_res["status"],
                        predicted_emotion=ai_res["predicted_emotion"],
                        confidence=ai_res["confidence"],
                        agreement=ai_res["agreement"],
                        has_incongruity=ai_res["has_incongruity"],
                        all_scores=ai_res["all_scores"],
                        per_model_scores=ai_res["per_model_scores"]
                    )
                    db.add(db_clip)
                    db.commit()
                    print(f"✅ [Pipeline] Xử lý thành công Clip #{clip_index}. Trạng thái định tuyến: {quality_res['status'].upper()}")
                    
                except Exception as clip_err:
                    print(f"❌ [Pipeline] Lỗi khi xử lý Clip #{clip_index}: {clip_err}")
                    traceback.print_exc()
                    # Vẫn tiếp tục xử lý các clips khác nếu 1 clip gặp lỗi
                    continue
            
            # Cập nhật video hoàn tất
            video.status = "completed"
            # Cập nhật số lượng approved_clips tự động
            approved_count = db.query(Clip).filter(
                Clip.video_id == video_id,
                Clip.status == "approved"
            ).count()
            video.approved_clips = approved_count
            db.commit()
            print(f"\n🎉 [Pipeline] HOÀN TẤT THÀNH CÔNG toàn bộ quy trình cho Video ID: {video_id}")
            
        except Exception as e:
            print(f"❌ [Pipeline] Gặp lỗi nghiêm trọng dừng pipeline của Video {video_id}: {e}")
            traceback.print_exc()
            video.status = "error"
            db.commit()
