"""
Emotion Data Studio — Colab GPU Worker
=====================================
Chạy trên Google Colab Pro (GPU T4/V100/A100).

Đặt trong Colab cell hoặc chạy như script:
    !python /content/BCDA/tools/emotion-data-studio/colab_worker.py

Luồng:
  1. Kết nối ngrok tunnel đến local backend (port 8765)
  2. Đăng ký worker với backend
  3. Loop vô hạn: claim → process → complete
  4. Heartbeat mỗi 30s
  5. Cleanup khi Colab disconnect

Yêu cầu:
  - Local backend đang chạy + có ngrok token
  - Colab mount Drive (để truy cập videos)
  - GPU runtime được bật
"""

# !pip install pyngrok httpx requests  # chạy trước nếu cần

import os
import sys
import json
import time
import socket
import uuid
import traceback
import logging
from datetime import datetime
from pathlib import Path

# ── Configuration ──────────────────────────────────────────────────────────────

# Thay đổi các giá trị này:
NGROK_TOKEN = "YOUR_NGROK_TOKEN"          # Token từ https://dashboard.ngrok.com
LOCAL_BACKEND_PORT = 8765                  # Port local backend
REPO_DIR = "/content/BCDA"                 # Đường dẫn repo trên Drive
DATA_DIR = "/content/drive/MyDrive/EDS/data"
GPU_WORKER_ID = f"colab-{socket.gethostname()[:8]}-{uuid.uuid4().hex[:6]}"

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("ColabWorker")


# ── GPU Detection ─────────────────────────────────────────────────────────────

def get_gpu_info():
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            mem = torch.cuda.get_device_properties(0).total_memory / 1e9
            return name, round(mem, 1)
    except Exception:
        pass
    return "CPU", 0.0


# ── Ngrok Tunnel ──────────────────────────────────────────────────────────────

def start_ngrok_tunnel(port: int, token: str) -> str:
    """
    Khởi động ngrok tunnel đến local backend.
    Trả về public URL để gọi API.
    """
    try:
        from pyngrok import ngrok
    except ImportError:
        raise ImportError(
            "pyngrok chưa cài. Chạy: !pip install pyngrok"
        )

    # Kill existing tunnels
    try:
        ngrok.kill()
    except Exception:
        pass

    ngrok.set_auth_token(token)

    # Create HTTPS tunnel
    tunnel = ngrok.connect(
        addr=str(port),
        proto="http",
        bind_tls=True,
    )
    public_url = tunnel.public_url
    log.info(f"🌐 Ngrok tunnel: {public_url}")
    return public_url


def get_local_ngrok_url(token: str) -> str:
    """Lấy URL của tunnel đang chạy, hoặc tạo mới."""
    try:
        from pyngrok import ngrok
        tunnels = ngrok.get_tunnels()
        for t in tunnels:
            if str(LOCAL_BACKEND_PORT) in t.config.get("addr", ""):
                return t.public_url
    except Exception:
        pass
    return start_ngrok_tunnel(LOCAL_BACKEND_PORT, token)


# ── API Client ─────────────────────────────────────────────────────────────────

class WorkerAPIClient:
    """HTTP client cho local backend API."""

    def __init__(self, base_url: str):
        # base_url: địa chỉ local backend qua ngrok, ví dụ https://abc123.ngrok.io
        self.base = base_url.rstrip("/")
        self.session = None

    def _get_session(self):
        if self.session is None:
            import httpx
            self.session = httpx.Client(timeout=60.0)
        return self.session

    def _request(self, method: str, path: str, **kwargs):
        url = f"{self.base}{path}"
        try:
            resp = self._get_session().request(method, url, **kwargs)
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:
            log.error(f"Request failed {method} {url}: {exc}")
            raise

    def register(self, worker_id: str, gpu_name: str, gpu_memory_gb: float) -> dict:
        return self._request("POST", "/api/worker/register", json={
            "worker_id": worker_id,
            "gpu_name": gpu_name,
            "gpu_memory_gb": gpu_memory_gb,
            "capabilities": ["gpu", "pipeline", "gemini"],
        })

    def heartbeat(self, worker_id: str, gpu_util: float = None,
                  gpu_mem_used: float = None, video_id: str = None) -> dict:
        return self._request("POST", "/api/worker/heartbeat", json={
            "worker_id": worker_id,
            "gpu_utilization": gpu_util,
            "gpu_memory_used_gb": gpu_mem_used,
            "processing_video_id": video_id,
        })

    def claim(self, worker_id: str) -> dict:
        return self._request("GET", f"/api/worker/claim?worker_id={worker_id}")

    def complete(self, worker_id: str, queue_item_id: int, video_id: str,
                 status: str, total_clips: int = 0, approved_clips: int = 0,
                 error_msg: str = None, gemini_segments: list = None) -> dict:
        return self._request("POST", "/api/worker/complete", json={
            "worker_id": worker_id,
            "queue_item_id": queue_item_id,
            "video_id": video_id,
            "status": status,
            "total_clips": total_clips,
            "approved_clips": approved_clips,
            "error_msg": error_msg,
            "gemini_segments": gemini_segments,
        })

    def skip(self, worker_id: str, queue_item_id: int) -> dict:
        return self._request("POST", "/api/worker/skip", params={
            "worker_id": worker_id,
            "queue_item_id": queue_item_id,
        })

    def status(self) -> dict:
        return self._request("GET", "/api/worker/status")


# ── Pipeline Runner ─────────────────────────────────────────────────────────────

def run_pipeline_on_colab(
    video_id: str,
    video_path: str | None,
    video_url: str | None,
    repo_dir: str,
    data_dir: str,
    progress_callback=None,
) -> dict:
    """
    Chạy pipeline trên Colab.
    Giả lập progress_callback để log ra console.
    """
    import sys
    sys.path.insert(0, f"{repo_dir}/tools/emotion-data-studio")
    os.environ["EDS_DATA_DIR"] = data_dir

    from backend.services.pipeline_orchestrator import PipelineOrchestrator

    def log_progress(stage, current, total, message=""):
        if message:
            log.info(f"  [{stage}] {message}")

    orchestrator = PipelineOrchestrator()
    orchestrator.run_pipeline(video_id, db=None, progress_callback=log_progress)

    return {"status": "ok"}


# ── Main Worker Loop ───────────────────────────────────────────────────────────

def run_worker_loop(
    backend_base_url: str,
    worker_id: str,
    gpu_name: str,
    gpu_memory_gb: float,
    repo_dir: str,
    data_dir: str,
    poll_interval: int = 10,
    heartbeat_interval: int = 30,
):
    """
    Main loop: đăng ký → loop claim/process/complete/heartbeat.
    """
    client = WorkerAPIClient(backend_base_url)

    # ── Register ──────────────────────────────────────────────────────────────
    try:
        reg = client.register(worker_id, gpu_name, gpu_memory_gb)
        log.info(f"✅ Registered as worker: {reg}")
    except Exception as exc:
        log.error(f"❌ Registration failed: {exc}")
        log.error("Kiểm tra local backend có đang chạy không (port 8765)")
        return

    log.info(f"🚀 Worker loop started. Polling every {poll_interval}s, heartbeat every {heartbeat_interval}s")
    log.info(f"   Backend: {backend_base_url}")

    last_heartbeat = time.time()
    current_job = None  # (queue_item_id, video_id)

    # ── Main Loop ─────────────────────────────────────────────────────────────
    consecutive_errors = 0
    MAX_CONSECUTIVE_ERRORS = 5

    while True:
        try:
            # Heartbeat
            now = time.time()
            if now - last_heartbeat >= heartbeat_interval:
                video_id = current_job[1] if current_job else None
                try:
                    client.heartbeat(worker_id, video_id=video_id)
                    last_heartbeat = now
                    if not current_job:
                        log.debug("❤️ Heartbeat sent (idle)")
                except Exception as exc:
                    log.warning(f"Heartbeat failed: {exc}")

            # ── Claim job ────────────────────────────────────────────────────
            if current_job is None:
                try:
                    result = client.claim(worker_id)
                    if result.get("status") == "ok":
                        qid = result["queue_item_id"]
                        vid = result["video_id"]
                        title = result.get("video_title", "?")
                        log.info(f"🎬 Job claimed: [{vid}] {title}")
                        current_job = (qid, vid, result)
                    else:
                        # no_jobs — chờ
                        log.debug(f"⏳ No jobs ({result.get('message', '')})")
                except Exception as exc:
                    log.warning(f"Claim failed: {exc}")

            # ── Process job ──────────────────────────────────────────────────
            if current_job is not None:
                qid, vid, job_info = current_job

                try:
                    result = run_pipeline_on_colab(
                        video_id=vid,
                        video_path=job_info.get("video_path"),
                        video_url=job_info.get("video_url"),
                        repo_dir=repo_dir,
                        data_dir=data_dir,
                    )

                    # Query clip counts
                    try:
                        status_resp = client._request("GET", f"/api/videos/{vid}")
                        total_clips = status_resp.get("total_clips", 0)
                        approved_clips = status_resp.get("approved_clips", 0)
                    except Exception:
                        total_clips = 0
                        approved_clips = 0

                    client.complete(
                        worker_id=worker_id,
                        queue_item_id=qid,
                        video_id=vid,
                        status="done",
                        total_clips=total_clips,
                        approved_clips=approved_clips,
                    )
                    log.info(f"✅ Job done: [{vid}] clips={total_clips}")

                except Exception as exc:
                    err_msg = traceback.format_exc()
                    log.error(f"❌ Pipeline failed for {vid}: {exc}\n{err_msg}")
                    try:
                        client.complete(
                            worker_id=worker_id,
                            queue_item_id=qid,
                            video_id=vid,
                            status="error",
                            error_msg=str(exc)[:500],
                        )
                    except Exception:
                        pass

                current_job = None
                consecutive_errors = 0

            # ── Sleep before next poll ─────────────────────────────────────────
            time.sleep(poll_interval)

        except KeyboardInterrupt:
            log.info("Keyboard interrupt — shutting down")
            try:
                client._request("POST", "/api/worker/unregister",
                                json={"worker_id": worker_id})
            except Exception:
                pass
            break
        except Exception as exc:
            consecutive_errors += 1
            log.error(f"Loop error ({consecutive_errors}/{MAX_CONSECUTIVE_ERRORS}): {exc}")
            if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                log.critical("Too many consecutive errors — exiting")
                break
            time.sleep(30)


# ── Launcher ──────────────────────────────────────────────────────────────────

def main():
    import argparse

    parser = argparse.ArgumentParser(description="EDS Colab GPU Worker")
    parser.add_argument("--ngrok-token", default=NGROK_TOKEN,
                        help="Ngrok auth token")
    parser.add_argument("--backend-port", type=int, default=LOCAL_BACKEND_PORT,
                        help="Local backend port")
    parser.add_argument("--worker-id", default=GPU_WORKER_ID,
                        help="Unique worker ID")
    parser.add_argument("--poll-interval", type=int, default=10,
                        help="Seconds between claim polls")
    parser.add_argument("--heartbeat-interval", type=int, default=30,
                        help="Seconds between heartbeats")
    parser.add_argument("--repo-dir", default=REPO_DIR,
                        help="Path to EDS repo")
    parser.add_argument("--data-dir", default=DATA_DIR,
                        help="EDS data directory on Drive")
    args = parser.parse_args()

    if "YOUR_NGROK_TOKEN" in args.ngrok_token or not args.ngrok_token:
        log.error("❌ Chưa đặt NGROK_TOKEN!")
        log.error("   Lấy token tại: https://dashboard.ngrok.com/get-started/your-authtoken")
        return

    # Detect GPU
    gpu_name, gpu_mem = get_gpu_info()
    log.info(f"🖥️  GPU: {gpu_name} ({gpu_mem}GB)")

    # Start tunnel
    tunnel_url = start_ngrok_tunnel(args.backend_port, args.ngrok_token)

    # Run worker
    run_worker_loop(
        backend_base_url=tunnel_url,
        worker_id=args.worker_id,
        gpu_name=gpu_name,
        gpu_memory_gb=gpu_mem,
        repo_dir=args.repo_dir,
        data_dir=args.data_dir,
        poll_interval=args.poll_interval,
        heartbeat_interval=args.heartbeat_interval,
    )


if __name__ == "__main__":
    main()
