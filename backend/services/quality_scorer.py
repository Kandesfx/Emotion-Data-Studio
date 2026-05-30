from typing import Dict, Any

class QualityScorer:
    """Lớp xử lý tính toán điểm chất lượng (Quality Score) và định tuyến trạng thái tự động cho clips."""
    
    @staticmethod
    def calculate_score(
        confidence: float,
        agreement_str: str,
        sampled_frames_count: int,
        cropped_faces_count: int,
        audio_clarity: float
    ) -> Dict[str, Any]:
        """Tính toán Quality Score dựa trên 4 chỉ số có trọng số từ dữ liệu xử lý."""
        
        # 1. Điểm Đồng thuận (Model Agreement Score)
        # Parse chuỗi e.g. "3/4" -> 0.75
        agreement_score = 0.5 # Default fallback
        if agreement_str and "/" in agreement_str:
            try:
                parts = agreement_str.split("/")
                numerator = float(parts[0])
                denominator = float(parts[1])
                if denominator > 0:
                    agreement_score = numerator / denominator
            except Exception:
                pass
                
        # 2. Chất lượng khuôn mặt (Face Detection Quality)
        # Tỉ lệ số frame crop được mặt trên tổng số frame đã lấy mẫu phân tích
        face_quality = 0.0
        if sampled_frames_count > 0:
            face_quality = min(1.0, cropped_faces_count / sampled_frames_count)
            
        # 3. Chuẩn hóa độ rõ của âm thanh (Audio Clarity Score)
        # RMS energy thường từ 0.0 đến 0.5. Ta scale và clip để có giá trị từ 0.0 đến 1.0
        norm_audio_clarity = min(1.0, audio_clarity * 5.0)
        
        # 4. Công thức tính điểm chất lượng có trọng số chuẩn chỉnh
        # Trọng số: 40% Confidence + 30% Agreement + 20% Face Quality + 10% Audio Clarity
        quality_score = (
            0.40 * confidence +
            0.30 * agreement_score +
            0.20 * face_quality +
            0.10 * norm_audio_clarity
        )
        
        # Đảm bảo điểm nằm trong khoảng 0.0 - 1.0
        quality_score = max(0.0, min(1.0, quality_score))
        
        # --- ĐỊNH TUYẾN TRẠNG THÁI (ROUTING RULES) ---
        # ├── score ≥ 0.85 → ✅ approved (Auto-approved)
        # ├── 0.60 ≤ score < 0.85 → ⚠️ pending (Needs Review)
        # └── score < 0.60 → ❌ rejected (Auto-rejected)
        
        status = "pending"
        if quality_score >= 0.85:
            status = "approved"
        elif quality_score < 0.60:
            status = "rejected"
            
        return {
            "quality_score": round(quality_score, 4),
            "status": status,
            "components": {
                "confidence": round(confidence, 4),
                "agreement_score": round(agreement_score, 4),
                "face_quality": round(face_quality, 4),
                "audio_clarity_normalized": round(norm_audio_clarity, 4)
            }
        }
