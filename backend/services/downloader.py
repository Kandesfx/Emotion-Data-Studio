import os
import re
from pathlib import Path
from typing import Dict, Any, Optional
import yt_dlp
from backend.config import settings

class VideoDownloader:
    """Lớp xử lý tải video từ YouTube hoặc các nền tảng được hỗ trợ bởi yt-dlp."""
    
    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = output_dir or (settings.DATA_DIR / "videos")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def is_valid_url(url: str) -> bool:
        """Kiểm tra xem URL đầu vào có phải là URL video hợp lệ không."""
        if not url:
            return False
        # Kiểm tra URL youtube cơ bản
        youtube_regex = (
            r'(https?://)?(www\.)?'
            '(youtube|youtu|youtube-nocookie)\.(com|be)/'
            '(watch\?v=|embed/|v/|.+\?v=)?([^&=%\?]{11})'
        )
        return bool(re.match(youtube_regex, url)) or url.startswith("http")

    def get_video_info(self, url: str) -> Dict[str, Any]:
        """Trích xuất thông tin chi tiết của video mà không tải về."""
        ydl_opts = {
            'quiet': True,
            'no_warnings': True,
            'skip_download': True,
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            try:
                info = ydl.extract_info(url, download=False)
                return {
                    "title": info.get("title", "Unknown Title"),
                    "duration_sec": float(info.get("duration", 0)),
                    "resolution": f"{info.get('width', 1280)}x{info.get('height', 720)}",
                    "ext": info.get("ext", "mp4"),
                    "id": info.get("id", "unknown_id")
                }
            except Exception as e:
                raise Exception(f"Không thể lấy thông tin video từ URL: {str(e)}")

    def download(self, url: str, progress_hook=None) -> Dict[str, Any]:
        """Tải video chất lượng tối đa là 720p để tiết kiệm dung lượng và tài nguyên xử lý."""
        # Lấy thông tin video trước
        info = self.get_video_info(url)
        video_id = info["id"]
        
        # Định nghĩa file path output
        output_template = str(self.output_dir / f"{video_id}.%(ext)s")
        
        # Cấu hình yt-dlp
        # Chọn định dạng mp4, độ phân giải tối đa 720p
        ydl_opts = {
            'format': 'bestvideo[height<=720][ext=mp4]+bestaudio[ext=m4a]/best[height<=720][ext=mp4]/best',
            'outtmpl': output_template,
            'quiet': True,
            'no_warnings': True,
            'merge_output_format': 'mp4',
        }
        
        if progress_hook:
            ydl_opts['progress_hooks'] = [progress_hook]
            
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            try:
                ydl.download([url])
                
                # Tìm file mp4 đã tải thực tế
                expected_path = self.output_dir / f"{video_id}.mp4"
                if not expected_path.exists():
                    # Fallback tìm kiếm bất kỳ file nào có tên là video_id trong folder
                    downloaded_files = list(self.output_dir.glob(f"{video_id}.*"))
                    if downloaded_files:
                        expected_path = downloaded_files[0]
                    else:
                        raise FileNotFoundError("Không tìm thấy file video sau khi tải thành công")
                
                return {
                    "title": info["title"],
                    "duration_sec": info["duration_sec"],
                    "resolution": info["resolution"],
                    "file_path": str(expected_path.resolve()),
                    "video_id": video_id
                }
            except Exception as e:
                raise Exception(f"Lỗi trong quá trình tải video: {str(e)}")
