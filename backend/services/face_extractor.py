import os
import cv2
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from backend.config import settings
from backend.ai_models.model_manager import model_manager

class FaceExtractor:
    """Lớp xử lý trích xuất khuôn mặt, tracking nhân vật chính và crop ảnh 224x224."""
    
    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = output_dir or (settings.DATA_DIR / "frames")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.fps_sample_rate = 2.0  # Lấy mẫu 2 frames mỗi giây (cách nhau 0.5s)

    def extract_faces_from_clip(self, clip_path: str, clip_id: str) -> Dict[str, Any]:
        """Phát hiện và tracking khuôn mặt trong clip, crop khuôn mặt nhân vật chính."""
        if not os.path.exists(clip_path):
            raise FileNotFoundError(f"Không tìm thấy clip tại: {clip_path}")
            
        clip_frames_dir = self.output_dir / clip_id
        clip_frames_dir.mkdir(parents=True, exist_ok=True)
        
        # Mở clip bằng OpenCV
        cap = cv2.VideoCapture(clip_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if fps <= 0:
            fps = 25.0
            
        # Tính khoảng cách giữa các frames cần lấy mẫu
        frame_interval = max(1, int(fps / self.fps_sample_rate))
        
        frame_idx = 0
        sampled_frames = []
        
        # Cố gắng nạp InsightFace thông qua ModelManager
        detector = None
        try:
            detector = model_manager.load_model("insightface")
        except Exception as e:
            print(f"Không thể load InsightFace detector ({e}), thử fallback sang OpenCV Haar Cascades...")
            # Fallback sang Haar Cascade mặc định của OpenCV nếu không có GPU/InsightFace
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            detector = cv2.CascadeClassifier(cascade_path)
            
        detected_tracks: Dict[int, List[Dict[str, Any]]] = {} # track_id -> list of bounding boxes
        frame_detections = []
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_idx % frame_interval == 0:
                h, w, _ = frame.shape
                # Detect khuôn mặt trong frame
                bboxes = [] # Danh sách bounding boxes trong frame này: [x1, y1, x2, y2]
                
                if isinstance(detector, cv2.CascadeClassifier):
                    # Sử dụng OpenCV Haar Cascade detector
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
                    for (x, y, fw, fh) in faces:
                        bboxes.append([x, y, x + fw, y + fh, 0.9]) # mock confidence
                else:
                    # Sử dụng InsightFace detector
                    try:
                        faces = detector.get(frame)
                        for face in faces:
                            box = face.bbox.astype(int)
                            bboxes.append([box[0], box[1], box[2], box[3], face.det_score])
                    except Exception as ex:
                        print(f"Lỗi khi chạy InsightFace: {ex}")
                        
                frame_detections.append({
                    "frame_time": frame_idx / fps,
                    "frame_data": frame,
                    "bboxes": bboxes
                })
                
            frame_idx += 1
            
        cap.release()
        
        # --- TRACKING GIẢ LẬP (BYTE-TRACK SIMPLIFIED) ---
        # gom các bounding box của cùng một người qua các frame bằng độ phủ hình học (IoU)
        next_track_id = 0
        active_tracks: Dict[int, List[float]] = {} # track_id -> last bounding box [x1, y1, x2, y2]
        
        def calculate_iou(boxA, boxB):
            xA = max(boxA[0], boxB[0])
            yA = max(boxA[1], boxB[1])
            xB = min(boxA[2], boxB[2])
            yB = min(boxA[3], boxB[3])
            interArea = max(0, xB - xA) * max(0, yB - yA)
            boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
            boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
            if boxAArea + boxBArea - interArea == 0:
                return 0
            iou = interArea / float(boxAArea + boxBArea - interArea)
            return iou
            
        for f_det in frame_detections:
            bboxes = f_det["bboxes"]
            frame_img = f_det["frame_data"]
            frame_time = f_det["frame_time"]
            
            matched_bboxes = set()
            
            # Khớp bboxes mới với các tracks đang hoạt động
            for track_id, last_box in list(active_tracks.items()):
                best_iou = 0.0
                best_bbox_idx = -1
                
                for idx, bbox in enumerate(bboxes):
                    if idx in matched_bboxes:
                        continue
                    iou = calculate_iou(last_box, bbox[:4])
                    if iou > best_iou:
                        best_iou = iou
                        best_bbox_idx = idx
                        
                if best_iou > 0.3: # Ngưỡng IoU cho tracking
                    matched_bboxes.add(best_bbox_idx)
                    matched_box = bboxes[best_bbox_idx]
                    active_tracks[track_id] = matched_box[:4]
                    
                    if track_id not in detected_tracks:
                        detected_tracks[track_id] = []
                    detected_tracks[track_id].append({
                        "bbox": matched_box[:4],
                        "frame_time": frame_time,
                        "frame_img": frame_img
                    })
                else:
                    # Hủy active track nếu không còn tìm thấy trong frame này
                    active_tracks.pop(track_id)
                    
            # Các bboxes không được khớp -> Tạo track mới
            for idx, bbox in enumerate(bboxes):
                if idx not in matched_bboxes:
                    track_id = next_track_id
                    next_track_id += 1
                    active_tracks[track_id] = bbox[:4]
                    
                    detected_tracks[track_id] = [{
                        "bbox": bbox[:4],
                        "frame_time": frame_time,
                        "frame_img": frame_img
                    }]
                    
        # --- CHỌN NHÂN VẬT CHÍNH ---
        # Nhân vật chính là track xuất hiện nhiều nhất và có kích thước bounding box lớn nhất
        main_track_id = -1
        max_score = -1.0
        
        for track_id, track_data in detected_tracks.items():
            num_occurrences = len(track_data)
            # Tính kích thước trung bình của bbox
            avg_area = np.mean([
                (d["bbox"][2] - d["bbox"][0]) * (d["bbox"][3] - d["bbox"][1])
                for d in track_data
            ])
            # Điểm đánh giá mức độ chính = số frame xuất hiện * diện tích
            score = num_occurrences * avg_area
            if score > max_score:
                max_score = score
                main_track_id = track_id
                
        # --- CROP & LƯU KHUÔN MẶT ---
        cropped_face_paths = []
        if main_track_id != -1:
            main_track_data = detected_tracks[main_track_id]
            for i, data in enumerate(main_track_data):
                bbox = data["bbox"]
                img = data["frame_img"]
                
                h, w, _ = img.shape
                # Giới hạn bounding box trong khung hình
                x1 = max(0, int(bbox[0]))
                y1 = max(0, int(bbox[1]))
                x2 = min(w, int(bbox[2]))
                y2 = min(h, int(bbox[3]))
                
                if x2 > x1 and y2 > y1:
                    face_crop = img[y1:y2, x1:x2]
                    # Resize về 224x224 chuẩn cho các mô hình CNN/ViT
                    face_crop_resized = cv2.resize(face_crop, (224, 224))
                    
                    # Lưu file ảnh
                    face_filename = f"face_{i:04d}.jpg"
                    face_path = clip_frames_dir / face_filename
                    cv2.imwrite(str(face_path), face_crop_resized)
                    cropped_face_paths.append(str(face_path.resolve()))
                    
        num_faces_detected = sum(len(f["bboxes"]) for f in frame_detections)
        
        return {
            "num_frames": len(frame_detections),
            "num_faces": num_faces_detected,
            "main_track_len": len(cropped_face_paths),
            "frames_dir": str(clip_frames_dir.resolve()),
            "cropped_face_paths": cropped_face_paths
        }
