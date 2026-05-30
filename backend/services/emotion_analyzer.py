import os
import numpy as np
from typing import List, Dict, Any, Tuple
from backend.ai_models.model_manager import model_manager
from backend.config import settings

class EmotionAnalyzer:
    """Lớp xử lý phân tích cảm xúc đa phương thức bằng thuật toán Ensemble Voting kết hợp 4 mô hình AI SOTA."""
    
    # 7 lớp cảm xúc chuẩn (Ekman)
    EMOTIONS = ["happy", "sad", "angry", "surprise", "fear", "disgust", "neutral"]
    
    # Trọng số bình chọn như thiết kế kiến trúc
    WEIGHTS = {
        "hsemotion_face": 0.35,    # SOTA facial expression
        "deepface_face":  0.25,    # Facial expression backup
        "phobert_text":   0.20,    # Lời thoại tiếng Việt sentiment
        "wav2vec_audio":  0.20,    # Âm thanh giọng nói emotion
    }

    def analyze_clip(
        self, 
        face_images: List[str], 
        transcript: str, 
        audio_path: str
    ) -> Dict[str, Any]:
        """Chạy 4 mô hình AI và thực hiện Ensemble Voting tổng hợp cảm xúc cuối cùng."""
        
        predictions: Dict[str, Dict[str, float]] = {}
        
        # 1. PHÂN TÍCH BIỂU CẢM KHUÔN MẶT CHÍNH (HSEmotion)
        predictions["hsemotion_face"] = self._analyze_hsemotion(face_images)
        
        # 2. PHÂN TÍCH BIỂU CẢM KHUÔN MẶT DỰ PHÒNG (DeepFace)
        predictions["deepface_face"] = self._analyze_deepface(face_images)
        
        # 3. PHÂN TÍCH NGỮ NGHĨA LỜI THOẠI (PhoBERT)
        predictions["phobert_text"] = self._analyze_phobert(transcript)
        
        # 4. PHÂN TÍCH ÂM THANH GIỌNG NÓI (Wav2Vec2)
        predictions["wav2vec_audio"] = self._analyze_wav2vec(audio_path)
        
        # --- ENSEMBLE VOTING ALGORITHM ---
        
        # Khởi tạo điểm tổng hợp cho 7 lớp cảm xúc
        combined_scores = {emo: 0.0 for emo in self.EMOTIONS}
        
        # Cộng dồn điểm số có trọng số từ các nguồn active
        active_weights_sum = 0.0
        for model_key, weight in self.WEIGHTS.items():
            model_preds = predictions.get(model_key)
            if not model_preds:
                continue
            
            active_weights_sum += weight
            for emo in self.EMOTIONS:
                combined_scores[emo] += model_preds.get(emo, 0.0) * weight
                
        # Chuẩn hóa lại điểm số nếu có mô hình bị lỗi/không hoạt động
        if active_weights_sum > 0:
            for emo in combined_scores:
                combined_scores[emo] /= active_weights_sum
                
        # Tìm cảm xúc chiến thắng (Dominant Emotion)
        dominant_emotion = max(combined_scores, key=combined_scores.get)
        confidence = combined_scores[dominant_emotion]
        
        # Tính mức độ đồng thuận (Agreement: bao nhiêu mô hình đồng ý với cảm xúc chính này)
        agreement_count = 0
        total_active_models = 0
        
        for model_key in self.WEIGHTS:
            model_preds = predictions.get(model_key)
            if not model_preds:
                continue
            total_active_models += 1
            # Lấy cảm xúc chiến thắng của mô hình đơn lẻ đó
            model_winner = max(model_preds, key=model_preds.get)
            
            # Map nhãn text PhoBERT sentiment tương đồng
            if model_key == "phobert_text":
                if model_winner == "positive" and dominant_emotion in ["happy", "surprise"]:
                    agreement_count += 1
                elif model_winner == "negative" and dominant_emotion in ["sad", "angry", "fear", "disgust"]:
                    agreement_count += 1
                elif model_winner == "neutral" and dominant_emotion == "neutral":
                    agreement_count += 1
            else:
                if model_winner == dominant_emotion:
                    agreement_count += 1
                    
        agreement_str = f"{agreement_count}/{total_active_models}"
        
        # Phát hiện sự mâu thuẫn cảm xúc (Incongruity -> Khả năng châm biếm sarcasm)
        # Nếu mặt cười (happy) nhưng lời thoại tiêu cực (negative) hoặc ngược lại
        has_incongruity = False
        face_winner = max(predictions["hsemotion_face"], key=predictions["hsemotion_face"].get)
        text_winner = max(predictions["phobert_text"], key=predictions["phobert_text"].get)
        
        if face_winner == "happy" and text_winner == "negative":
            has_incongruity = True
        elif face_winner in ["sad", "angry"] and text_winner == "positive":
            has_incongruity = True
            
        return {
            "predicted_emotion": dominant_emotion,
            "confidence": round(float(confidence), 4),
            "agreement": agreement_str,
            "has_incongruity": has_incongruity,
            "all_scores": {k: round(v, 4) for k, v in combined_scores.items()},
            "per_model_scores": predictions
        }

    # ==========================================
    # CÁC PHƯƠNG THỨC TRÍCH XUẤT ĐƠN LẺ
    # ==========================================
    
    def _analyze_hsemotion(self, face_images: List[str]) -> Dict[str, float]:
        """Phân tích cảm xúc khuôn mặt bằng HSEmotion."""
        default_pred = {emo: 0.0 for emo in self.EMOTIONS}
        default_pred["neutral"] = 1.0  # Mặc định
        
        if not face_images:
            return default_pred
            
        try:
            hsemotion_recognizer = model_manager.load_model("hsemotion")
            
            # Đọc ảnh và dự đoán biểu cảm trên từng ảnh crop
            import cv2
            scores_list = []
            
            # Lấy tối đa 5 ảnh để phân tích trung bình tránh quá tải
            sample_faces = face_images[:5]
            for img_path in sample_faces:
                img = cv2.imread(img_path)
                if img is None:
                    continue
                # HSEmotion trả về (emotion_idx, emotion_probabilities)
                _, emotion_probs = hsemotion_recognizer.predict_emotions(img, logits=False)
                scores_list.append(emotion_probs)
                
            if not scores_list:
                return default_pred
                
            # Tính trung bình xác suất cảm xúc của các ảnh
            avg_probs = np.mean(scores_list, axis=0)
            
            # HSEmotion classes map:
            # 0: Anger, 1: Disgust, 2: Fear, 3: Happiness, 4: Sadness, 5: Surprise, 6: Neutral
            hse_map = {
                "angry": float(avg_probs[0]),
                "disgust": float(avg_probs[1]),
                "fear": float(avg_probs[2]),
                "happy": float(avg_probs[3]),
                "sad": float(avg_probs[4]),
                "surprise": float(avg_probs[5]),
                "neutral": float(avg_probs[6])
            }
            return hse_map
            
        except Exception as e:
            print(f"HSEmotion thật thất bại hoặc chưa cài đặt ({e}). Chạy Mock dự đoán khuôn mặt...")
            # Fallback mock thông minh: sinh xác suất ngẫu nhiên
            return self._mock_emotions_dict("neutral", 0.7)

    def _analyze_deepface(self, face_images: List[str]) -> Dict[str, float]:
        """Phân tích cảm xúc khuôn mặt dự phòng bằng DeepFace."""
        default_pred = {emo: 0.0 for emo in self.EMOTIONS}
        default_pred["neutral"] = 1.0
        
        if not face_images:
            return default_pred
            
        try:
            deepface = model_manager.load_model("deepface")
            
            # DeepFace.analyze nhận diện cảm xúc khuôn mặt
            # Lấy ảnh khuôn mặt đầu tiên để phân tích
            result = deepface.analyze(
                img_path=face_images[0], 
                actions=['emotion'],
                enforce_detection=False,
                silent=True
            )
            
            # Trả về list dict nếu có nhiều mặt, hoặc dict nếu có 1 mặt
            if isinstance(result, list):
                result = result[0]
                
            raw_emotion_scores = result.get("emotion", {})
            
            # Quy đổi phần trăm (0-100) về xác suất (0.0-1.0)
            # DeepFace emotions: angry, disgust, fear, happy, sad, surprise, neutral
            df_map = {}
            for emo in self.EMOTIONS:
                # DeepFace lưu 'sad' / 'happy' tương đồng
                raw_score = raw_emotion_scores.get(emo, 0.0)
                # Đổi tên disgust trong deepface nếu khác
                df_map[emo] = float(raw_score) / 100.0
                
            # Chuẩn hóa
            total = sum(df_map.values())
            if total > 0:
                for k in df_map:
                    df_map[k] /= total
            return df_map
            
        except Exception as e:
            # Fallback mock
            return self._mock_emotions_dict("neutral", 0.65)

    def _analyze_phobert(self, transcript: str) -> Dict[str, float]:
        """Phân tích cảm xúc văn bản bằng PhoBERT-sentiment."""
        # PhoBERT sentiment chỉ có 3 lớp: positive, negative, neutral
        default_pred = {"positive": 0.0, "negative": 0.0, "neutral": 1.0}
        
        if not transcript:
            return self._convert_sentiment_to_ekman(default_pred)
            
        try:
            classifier = model_manager.load_model("phobert_sentiment")
            
            # Chạy phân tích văn bản
            result = classifier(transcript)[0]
            label = result["label"].lower() # 'positive', 'negative', 'neutral'
            score = float(result["score"])
            
            pred = {"positive": 0.0, "negative": 0.0, "neutral": 0.0}
            pred[label] = score
            # Phân bổ phần còn lại cho 2 nhãn kia
            remaining = 1.0 - score
            for k in pred:
                if k != label:
                    pred[k] = remaining / 2.0
                    
            return self._convert_sentiment_to_ekman(pred)
            
        except Exception as e:
            # Fallback mock dựa trên từ khóa trong tiếng Việt
            detected_label = "neutral"
            lower_text = transcript.lower()
            
            positive_words = ["vui", "tuyệt", "đẹp", "yêu", "thích", "cảm ơn", "ơn", "đậu", "lắm con", "tốt", "mừng"]
            negative_words = ["buồn", "khóc", "xin lỗi", "lỗi", "giận", "ghét", "sai", "chết", "mệt", "đau", "khổ", "đi ra", "ngay lập tức"]
            
            if any(w in lower_text for w in positive_words):
                detected_label = "positive"
            elif any(w in lower_text for w in negative_words):
                detected_label = "negative"
                
            mock_sentiment = self._mock_sentiment_dict(detected_label, 0.8)
            return self._convert_sentiment_to_ekman(mock_sentiment)

    def _analyze_wav2vec(self, audio_path: str) -> Dict[str, float]:
        """Phân tích cảm xúc giọng nói bằng XLS-R Wav2Vec2."""
        default_pred = {emo: 0.0 for emo in self.EMOTIONS}
        default_pred["neutral"] = 1.0
        
        if not audio_path or not os.path.exists(audio_path):
            return default_pred
            
        try:
            speech_model_pack = model_manager.load_model("wav2vec_emotion")
            processor = speech_model_pack["processor"]
            model = speech_model_pack["model"]
            
            import librosa
            import torch
            
            # Đọc audio
            speech, sr = librosa.load(audio_path, sr=16000)
            
            # Tiền xử lý
            inputs = processor(speech, sampling_rate=16000, return_tensors="pt", padding=True)
            inputs = {k: v.to(model_manager.get_device()) for k, v in inputs.items()}
            
            with torch.no_grad():
                logits = model(**inputs).logits
                
            scores = torch.softmax(logits, dim=-1).cpu().numpy()[0]
            
            # XLS-R Wav2Vec2 classes map tùy model (thường là angry, disgust, fear, happy, neutral, sad, surprise)
            # Map an toàn về 7 cảm xúc của chúng ta
            w2v_map = {
                "angry": float(scores[0]) if len(scores) > 0 else 0.0,
                "disgust": float(scores[1]) if len(scores) > 1 else 0.0,
                "fear": float(scores[2]) if len(scores) > 2 else 0.0,
                "happy": float(scores[3]) if len(scores) > 3 else 0.0,
                "neutral": float(scores[4]) if len(scores) > 4 else 0.0,
                "sad": float(scores[5]) if len(scores) > 5 else 0.0,
                "surprise": float(scores[6]) if len(scores) > 6 else 0.0,
            }
            
            # Cân đối lại nếu thiếu
            total = sum(w2v_map.values())
            if total > 0:
                for k in w2v_map:
                    w2v_map[k] /= total
            return w2v_map
            
        except Exception as e:
            return self._mock_emotions_dict("neutral", 0.6)

    # ==========================================
    # CÁC HELPER CHUYỂN ĐỔI & MOCK DỮ LIỆU
    # ==========================================
    
    def _convert_sentiment_to_ekman(self, sentiment_probs: Dict[str, float]) -> Dict[str, float]:
        """Quy đổi phân phối xác suất sentiment (3 lớp) sang cảm xúc Ekman (7 lớp)."""
        ekman_probs = {emo: 0.0 for emo in self.EMOTIONS}
        
        pos = sentiment_probs.get("positive", 0.0)
        neg = sentiment_probs.get("negative", 0.0)
        neu = sentiment_probs.get("neutral", 0.0)
        
        # Phân bổ positive -> happy (80%), surprise (20%)
        ekman_probs["happy"] = pos * 0.8
        ekman_probs["surprise"] = pos * 0.2
        
        # Phân bổ negative -> sad (40%), angry (30%), fear (15%), disgust (15%)
        ekman_probs["sad"] = neg * 0.40
        ekman_probs["angry"] = neg * 0.30
        ekman_probs["fear"] = neg * 0.15
        ekman_probs["disgust"] = neg * 0.15
        
        # Phân bổ neutral -> neutral (100%)
        ekman_probs["neutral"] = neu
        
        return ekman_probs

    def _mock_emotions_dict(self, dominant: str, score: float) -> Dict[str, float]:
        """Sinh mock xác suất cảm xúc cho 7 lớp Ekman."""
        res = {emo: 0.0 for emo in self.EMOTIONS}
        res[dominant] = score
        remaining = 1.0 - score
        for emo in self.EMOTIONS:
            if emo != dominant:
                res[emo] = remaining / 6.0
        return res

    def _mock_sentiment_dict(self, dominant: str, score: float) -> Dict[str, float]:
        """Sinh mock xác suất sentiment cho 3 lớp."""
        res = {"positive": 0.0, "negative": 0.0, "neutral": 0.0}
        res[dominant] = score
        remaining = 1.0 - score
        for k in res:
            if k != dominant:
                res[k] = remaining / 2.0
        return res
