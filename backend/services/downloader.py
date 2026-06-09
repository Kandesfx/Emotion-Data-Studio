import os
import re
import shutil
from pathlib import Path
from typing import Dict, Any, Optional

import yt_dlp

from backend.config import settings


class VideoDownloader:
    """Lớp xử lý tải video từ YouTube hoặc các nền tảng được hỗ trợ bởi yt-dlp."""

    # Class-level cache to share video info across instances (e.g., worker and orchestrator)
    _info_cache = {}

    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = output_dir or (settings.DATA_DIR / "videos")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def is_valid_url(url: str) -> bool:
        """Kiểm tra xem URL đầu vào có phải là URL video hợp lệ không."""
        if not url:
            return False
        youtube_regex = (
            r'(https?://)?(www\.)?'
            r'(youtube|youtu|youtube-nocookie)\.(com|be)/'
            r'(watch\?v=|embed/|v/|.+\?v=)?([^&=%\?]{11})'
        )
        return bool(re.match(youtube_regex, url)) or url.startswith("http")

    def get_video_info(self, url: str) -> Dict[str, Any]:
        """Trích xuất thông tin chi tiết của video mà không tải về."""
        if url in self._info_cache:
            return self._info_cache[url]

        ydl_opts = {
            "quiet": True,
            "no_warnings": True,
            "skip_download": True,
            "nocheckcertificate": True,
            "extract_flat": False,
            "retries": 3,
            "extractor_retries": 2,
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            try:
                info = ydl.extract_info(url, download=False)
                res = {
                    "title": info.get("title", "Unknown Title"),
                    "duration_sec": float(info.get("duration", 0)),
                    "resolution": f"{info.get('width', 1280)}x{info.get('height', 720)}",
                    "ext": info.get("ext", "mp4"),
                    "id": info.get("id", "unknown_id"),
                }
                self._info_cache[url] = res
                return res
            except Exception as e:
                raise Exception(self._friendly_error_message(e))

    def download(self, url: str, progress_hook=None) -> Dict[str, Any]:
        """Tải video với fallback nhiều tầng để tăng ổn định và khả năng vượt chặn."""
        info = self.get_video_info(url)
        video_id = info["id"]
        output_template = str(self.output_dir / f"{video_id}.%(ext)s")
        ffmpeg_available = shutil.which(settings.FFMPEG_PATH) is not None
        download_mode = settings.EDS_DOWNLOAD_MODE or self._download_mode()
        profiles = self._build_profiles(ffmpeg_available, download_mode)

        last_error: Exception | None = None
        for attempt, profile in enumerate(profiles, start=1):
            ydl_opts = self._build_ydl_opts(output_template, ffmpeg_available, profile)
            if progress_hook:
                ydl_opts["progress_hooks"] = [progress_hook]
            self._log_profile_attempt(attempt, profile, ffmpeg_available)
            try:
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    ydl.download([url])
                expected_path = self._resolve_downloaded_file(video_id)
                if not expected_path.exists():
                    raise FileNotFoundError("Không tìm thấy file video sau khi tải thành công")
                return {
                    "title": info["title"],
                    "duration_sec": info["duration_sec"],
                    "resolution": info["resolution"],
                    "file_path": str(expected_path.resolve()),
                    "video_id": video_id,
                    "download_profile": profile["name"],
                }
            except Exception as e:
                last_error = e
                self._log_download_error(attempt, profile, e)
                if self._should_skip_cookie_profile(e, profile):
                    print("⚠️ [Downloader] Bỏ qua profile cookie và thử lại không dùng cookies...")
                    continue
                continue

        raise Exception(self._friendly_error_message(last_error))

    def _download_mode(self) -> str:
        return (os.getenv("DOWNLOAD_MODE") or os.getenv("EDS_DOWNLOAD_MODE") or "balanced").strip().lower()

    def _build_profiles(self, ffmpeg_available: bool, download_mode: str) -> list[dict]:
        base = [
            {
                "name": "balanced",
                "concurrent_fragment_downloads": max(1, int(settings.EDS_DOWNLOAD_CONCURRENT_FRAGMENTS or 5)),
                "buffersize": 262144,
                "socket_timeout": 20,
                "retries": 8,
                "fragment_retries": 8,
                "extractor_retries": 3,
                "file_access_retries": 3,
                "throttled_rate": max(0, int(settings.EDS_DOWNLOAD_THROTTLED_RATE_KBPS or 100)) * 1024,
                "player_clients": ["android", "web_embedded"],
            },
            {
                "name": "safe",
                "concurrent_fragment_downloads": max(1, min(4, int(settings.EDS_DOWNLOAD_CONCURRENT_FRAGMENTS or 5))),
                "buffersize": 131072,
                "socket_timeout": 30,
                "retries": 12,
                "fragment_retries": 12,
                "extractor_retries": 5,
                "file_access_retries": 5,
                "throttled_rate": max(50, int(settings.EDS_DOWNLOAD_THROTTLED_RATE_KBPS or 100) - 25) * 1024,
                "player_clients": ["web_embedded", "android"],
            },
            {
                "name": "turbo",
                "concurrent_fragment_downloads": max(6, int(settings.EDS_DOWNLOAD_CONCURRENT_FRAGMENTS or 5)),
                "buffersize": 524288,
                "socket_timeout": 15,
                "retries": 6,
                "fragment_retries": 6,
                "extractor_retries": 2,
                "file_access_retries": 2,
                "throttled_rate": max(100, int(settings.EDS_DOWNLOAD_THROTTLED_RATE_KBPS or 100) + 50) * 1024,
                "player_clients": ["android", "web_embedded", "tv"],
            },
        ]

        if download_mode == "safe":
            ordered = [base[1], base[0], base[2]]
        elif download_mode == "turbo":
            ordered = [base[2], base[0], base[1]]
        else:
            ordered = [base[0], base[1], base[2]]

        if self._has_cookie_strategy():
            ordered.append({
                "name": "cookies",
                "concurrent_fragment_downloads": max(1, int(settings.EDS_DOWNLOAD_CONCURRENT_FRAGMENTS or 5)),
                "buffersize": 262144,
                "socket_timeout": 20,
                "retries": 8,
                "fragment_retries": 8,
                "extractor_retries": 3,
                "file_access_retries": 3,
                "throttled_rate": max(0, int(settings.EDS_DOWNLOAD_THROTTLED_RATE_KBPS or 100)) * 1024,
                "player_clients": ["android", "web_embedded", "tv"],
                "use_cookies": True,
            })
            ordered.append({
                "name": "nocookies_fallback",
                "concurrent_fragment_downloads": max(1, int(settings.EDS_DOWNLOAD_CONCURRENT_FRAGMENTS or 5)),
                "buffersize": 262144,
                "socket_timeout": 20,
                "retries": 8,
                "fragment_retries": 8,
                "extractor_retries": 3,
                "file_access_retries": 3,
                "throttled_rate": max(0, int(settings.EDS_DOWNLOAD_THROTTLED_RATE_KBPS or 100)) * 1024,
                "player_clients": ["android", "web_embedded", "tv"],
            })

        return ordered

    def _build_ydl_opts(self, output_template: str, ffmpeg_available: bool, profile: dict) -> dict:
        max_height = int(settings.EDS_DOWNLOAD_MAX_HEIGHT or 720)
        use_aria2 = bool(settings.EDS_DOWNLOAD_USE_ARIA2)
        aria2_ok = shutil.which("aria2c") is not None
        format_spec = self._build_format_spec(ffmpeg_available, max_height)
        ydl_opts = {
            "format": format_spec,
            "outtmpl": output_template,
            "quiet": True,
            "no_warnings": True,
            "nocheckcertificate": True,
            "concurrent_fragment_downloads": profile["concurrent_fragment_downloads"],
            "buffersize": profile["buffersize"],
            "socket_timeout": profile["socket_timeout"],
            "retries": profile["retries"],
            "fragment_retries": profile["fragment_retries"],
            "extractor_retries": profile["extractor_retries"],
            "file_access_retries": profile["file_access_retries"],
            "throttled_rate": profile["throttled_rate"],
            "extractor_args": {
                "youtube": {
                    "player_client": profile["player_clients"],
                    "player_js_version": ["actual"],
                }
            },
        }

        if ffmpeg_available:
            ydl_opts["merge_output_format"] = "mp4"

        self._apply_cookie_opts(ydl_opts, profile)

        if use_aria2 and aria2_ok:
            ydl_opts["external_downloader"] = "aria2c"
            ydl_opts["external_downloader_args"] = {"aria2c": ["-x", "16", "-s", "16", "-k", "1M"]}

        return ydl_opts

    @staticmethod
    def _build_format_spec(ffmpeg_available: bool, max_height: int) -> str:
        if ffmpeg_available:
            return f"bestvideo[height<={max_height}][ext=mp4]+bestaudio[ext=m4a]/best[height<={max_height}][ext=mp4]/best"
        return f"best[height<={max_height}][ext=mp4]/best"

    def _apply_cookie_opts(self, ydl_opts: dict, profile: dict):
        cookie_file = (settings.EDS_DOWNLOAD_COOKIE_FILE or os.getenv("DOWNLOAD_COOKIE_FILE") or "").strip()
        browser = (settings.EDS_DOWNLOAD_COOKIES_BROWSER or os.getenv("DOWNLOAD_COOKIES_BROWSER") or "").strip().lower()
        if profile.get("use_cookies") or cookie_file or browser:
            if cookie_file:
                ydl_opts["cookiefile"] = cookie_file
            elif browser:
                ydl_opts["cookiesfrombrowser"] = (browser,)
                ydl_opts["ignore_no_formats_error"] = True

    def _friendly_error_message(self, error: Exception | None) -> str:
        raw = str(error or "")
        lower = raw.lower()
        if "could not copy chrome cookie database" in lower:
            return (
                "Lỗi tải video: không thể đọc cookie từ Chrome. "
                "Hãy tắt Cookies from browser trong Settings hoặc đóng hẳn Chrome rồi thử lại."
            )
        if "failed to decrypt with dpapi" in lower:
            return (
                "Lỗi tải video: cookie Chrome không giải mã được bằng DPAPI trên máy này. "
                "Hãy tắt Cookies from browser, hoặc xuất cookies.txt thủ công rồi chọn file đó."
            )
        if "403" in lower:
            return "Lỗi tải video: YouTube từ chối truy cập (403). Hãy thử bật cookie file hoặc đổi profile tải."
        if "429" in lower:
            return "Lỗi tải video: YouTube đang giới hạn tần suất yêu cầu (429). Hãy chờ một lúc rồi thử lại."
        if "ffmpeg" in lower and "not found" in lower:
            return "Lỗi tải video: không tìm thấy FFmpeg. Hãy kiểm tra đường dẫn FFmpeg trong Settings."
        return f"Lỗi tải video: {raw or 'không xác định'}"

    @staticmethod
    def _has_cookie_strategy() -> bool:
        cookie_file = (settings.EDS_DOWNLOAD_COOKIE_FILE or os.getenv("DOWNLOAD_COOKIE_FILE") or "").strip()
        browser = (settings.EDS_DOWNLOAD_COOKIES_BROWSER or os.getenv("DOWNLOAD_COOKIES_BROWSER") or "").strip()
        return bool(cookie_file or browser)

    @staticmethod
    def _env_bool(key: str, default: bool = False) -> bool:
        value = os.getenv(key)
        if value is None:
            return default
        return value.strip().lower() in {"1", "true", "yes", "on"}

    def _resolve_downloaded_file(self, video_id: str) -> Path:
        expected_path = self.output_dir / f"{video_id}.mp4"
        if expected_path.exists():
            return expected_path
        downloaded_files = sorted(self.output_dir.glob(f"{video_id}.*"), key=lambda p: p.stat().st_mtime, reverse=True)
        return downloaded_files[0] if downloaded_files else expected_path

    @staticmethod
    def _friendly_error_message(exc: Exception | None) -> str:
        message = str(exc or "Không rõ lỗi")
        lowered = message.lower()
        if "http error 403" in lowered or "forbidden" in lowered:
            return "YouTube từ chối truy cập (403). Hãy thử bật cookies trình duyệt, dùng cookies.txt hoặc chuyển sang chế độ tải an toàn."
        if "http error 429" in lowered or "too many requests" in lowered:
            return "YouTube tạm chặn do quá nhiều yêu cầu (429). Hãy chờ vài phút, đổi mạng hoặc bật cookies rồi thử lại."
        if "sign in to confirm your age" in lowered or "age-restricted" in lowered:
            return "Video yêu cầu đăng nhập hoặc xác minh độ tuổi. Hãy cấu hình cookies trình duyệt hoặc cookies.txt trong Cài đặt."
        if "requested format is not available" in lowered:
            return "Không tìm thấy định dạng video phù hợp. Hãy giảm độ phân giải tối đa hoặc thử lại với chế độ tải an toàn."
        if "ffmpeg" in lowered and "not found" in lowered:
            return "Không tìm thấy FFmpeg. Hãy cài FFmpeg hoặc cấu hình đúng đường dẫn trong Cài đặt."
        if "signature" in lowered or "cipher" in lowered or "nsig" in lowered:
            return "YouTube thay đổi cơ chế chữ ký phát video. Hãy cập nhật yt-dlp và cân nhắc cài Deno rồi thử lại."
        if "cookies" in lowered and ("invalid" in lowered or "expired" in lowered):
            return "Cookies không hợp lệ hoặc đã hết hạn. Hãy xuất lại cookies.txt hoặc chọn lại trình duyệt lấy cookies."
        if "unable to download webpage" in lowered or "timed out" in lowered or "connection" in lowered:
            return "Lỗi kết nối mạng khi truy cập nguồn video. Hãy kiểm tra Internet hoặc thử lại với chế độ tải an toàn."
        return f"Lỗi tải video: {message}"

    @staticmethod
    def _log_profile_attempt(attempt: int, profile: dict, ffmpeg_available: bool):
        print(
            f"📡 [Downloader] Attempt {attempt} profile={profile['name']} "
            f"ffmpeg={'yes' if ffmpeg_available else 'no'} "
            f"clients={','.join(profile.get('player_clients', []))}"
        )

    @staticmethod
    def _should_skip_cookie_profile(error: Exception, profile: dict) -> bool:
        if not profile.get("use_cookies"):
            return False
        msg = str(error).lower()
        return any(token in msg for token in ["chrome cookie database", "dpapi", "cookie"])

    @staticmethod
    def _log_download_error(attempt: int, profile: dict, exc: Exception):
        msg = str(exc)
        print(f"⚠️ [Downloader] Attempt {attempt} profile={profile['name']} failed: {msg[:500]}")
