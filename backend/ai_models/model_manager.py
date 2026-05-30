import os
from pathlib import Path
from typing import Dict, Any, Optional
import torch
from backend.config import settings

class AIModelManager:
    """Quản lý nạp động (Lazy Loading) các mô hình AI lên GPU hoặc CPU, tối ưu hóa bộ nhớ."""
    
    _instance = None
    
    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(AIModelManager, cls).__new__(cls, *args, **kwargs)
            cls._instance._initialized = False
        return cls._instance
        
    def __init__(self):
        if self._initialized:
            return
            
        self.device = torch.device("cuda" if torch.cuda.is_available() and settings.USE_GPU else "cpu")
        self.cache_dir = settings.MODEL_CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Dictionary lưu giữ các instances mô hình đã được load vào RAM/VRAM
        self._loaded_models: Dict[str, Any] = {}
        self._initialized = True
        print(f"🤖 Khởi tạo AIModelManager sử dụng thiết bị: {self.device.type.upper()}")

    def get_device(self) -> torch.device:
        """Trả về thiết bị xử lý hiện tại (cuda hoặc cpu)."""
        return self.device

    def load_model(self, model_name: str) -> Any:
        """Tải mô hình lên bộ nhớ theo cơ chế Lazy Loading."""
        if model_name in self._loaded_models:
            return self._loaded_models[model_name]
            
        print(f"🧠 Đang nạp mô hình '{model_name}' lên {self.device.type.upper()}...")
        
        model_instance = None
        
        if model_name == "hsemotion":
            # Lazy load thư viện HSEmotion
            from hsemotion.facial_emotions import HSEmotionRecognizer
            # hsemotion tự động download weights và cache
            # Ở đây ta sử dụng model ViT_b_AffectNet (SOTA của hsemotion)
            model_instance = HSEmotionRecognizer(
                model_name='mtcnn_pro',  # dùng MTCNN để align khuôn mặt trước khi nhận diện
                device=self.device.type
            )
            
        elif model_name == "deepface":
            # DeepFace quản lý weights rất tốt, chúng ta import và load
            from deepface import DeepFace
            # Chúng ta chỉ verify bằng cách gọi dummy building để trigger downloading weights
            # Model mặc định là VGG-Face cho nhận dạng khuôn mặt, Emotion model cho phân tích
            # Trả về đối tượng thư viện gốc để wrap sử dụng
            model_instance = DeepFace
            
        elif model_name == "phobert_sentiment":
            # Load PhoBERT sentiment SOTA của tiếng Việt
            from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
            
            model_path = "wonrax/phobert-base-vietnamese-sentiment"
            tokenizer = AutoTokenizer.from_pretrained(model_path, cache_dir=str(self.cache_dir))
            model = AutoModelForSequenceClassification.from_pretrained(model_path, cache_dir=str(self.cache_dir))
            model.to(self.device)
            
            # Đóng gói pipeline
            model_instance = pipeline(
                "sentiment-analysis",
                model=model,
                tokenizer=tokenizer,
                device=0 if self.device.type == "cuda" else -1
            )
            
        elif model_name == "wav2vec_emotion":
            # Phân tích cảm xúc âm thanh qua Wav2Vec2-emotion
            from transformers import Wav2Vec2Processor, AutoModelForAudioClassification
            
            # Model nhận diện cảm xúc giọng nói đa ngôn ngữ / tiếng Việt tốt
            model_path = "harshit345/xlsr-wav2vec-speech-emotion-recognition"
            processor = Wav2Vec2Processor.from_pretrained(model_path, cache_dir=str(self.cache_dir))
            model = AutoModelForAudioClassification.from_pretrained(model_path, cache_dir=str(self.cache_dir))
            model.to(self.device)
            
            model_instance = {
                "processor": processor,
                "model": model
            }
            
        elif model_name == "whisper":
            # Tải Whisper model cho Speech-to-Text
            import whisper
            # Chúng ta mặc định dùng model "base" hoặc "small" cho môi trường local để tiết kiệm tài nguyên
            # Người dùng có thể nâng cấp lên "large-v3" thông qua settings
            model_size = os.getenv("EDS_WHISPER_SIZE", "base")
            model_instance = whisper.load_model(
                model_size, 
                device=self.device,
                download_root=str(self.cache_dir / "whisper")
            )
            
        elif model_name == "insightface":
            # Load insightface face analysis
            import insightface
            # Sử dụng model pack buffalo_l
            app = insightface.app.FaceAnalysis(
                name='buffalo_l',
                root=str(self.cache_dir / "insightface"),
                allowed_modules=['detection'] # Chỉ cần detection cho tác vụ crop face
            )
            app.prepare(ctx_id=0 if self.device.type == "cuda" else -1, det_size=(640, 640))
            model_instance = app
            
        else:
            raise ValueError(f"Không nhận dạng được tên mô hình AI: {model_name}")
            
        self._loaded_models[model_name] = model_instance
        print(f"✅ Đã nạp thành công mô hình '{model_name}'.")
        return model_instance

    def unload_model(self, model_name: str):
        """Giải phóng bộ nhớ GPU/RAM của một mô hình khi không còn sử dụng."""
        if model_name in self._loaded_models:
            del self._loaded_models[model_name]
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
            print(f"🧹 Đã giải phóng bộ nhớ cho mô hình '{model_name}'.")

    def unload_all_models(self):
        """Giải phóng bộ nhớ cho tất cả các mô hình."""
        self._loaded_models.clear()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        print("🧹 Đã dọn sạch tất cả các mô hình AI khỏi VRAM/RAM.")

# Khởi tạo instance duy nhất
model_manager = AIModelManager()
