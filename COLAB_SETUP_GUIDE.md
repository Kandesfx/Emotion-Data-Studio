# Hướng Dẫn Thiết Lập — Hybrid Local + Colab GPU Worker

## Tổng Quan Kiến Trúc

```
┌─────────────────────────┐   Cloudflare   ┌──────────────────────────┐
│  PC (Local Backend)     │ ◄───────────► │  Google Colab (GPU)     │
│  ─────────────────────  │   Tunnel       │  ─────────────────────  │
│  Port: 8765             │                │  Tesla T4 / V100 / A100 │
│  SQLite DB              │                │  Pipeline + Gemini       │
│  Dashboard Web UI        │                │                          │
│  Desktop Electron UI     │                │  Worker script           │
│  File storage: local     │                │  Colab Worker API client │
└────────┬────────────────┘                └───────────┬──────────────┘
         │                                             │
         │   videos/ clips/ frames on local disk       │
         │                                             │
         │  User → Dashboard → add video              │
         │  → Local backend queues job                 │
         │  → Colab worker polls & claims             │
         │  → Colab processes with GPU                 │
         │  → Results stored in local SQLite           │
         └─────────────────────────────────────────────┘
```

**Ưu điểm:**
- Dashboard & review chạy 24/7 trên local PC (không tốn tiền)
- Xử lý GPU chỉ bật khi cần (tiết kiệm chi phí)
- Colab Pro: ~$10/tháng cho GPU T4 (hoặc dùng free tier)
- Không phụ thuộc internet — Colab chỉ cần khi đang xử lý

---

## Bước 1 — Cài Đặt Google Cloud

### 1.1 Tạo GCP Project

1. Vào [console.cloud.google.com](https://console.cloud.google.com)
2. Tạo project mới: `emotion-data-studio`
3. Bật **Vertex AI API**: APIs & Services → Enable API → tìm "Vertex AI API"

### 1.2 Tạo Service Account

1. IAM & Admin → Service Accounts → Create
2. Tên: `eds-gpu-worker`
3. Roles cần thiết:
   - **Storage Admin** (upload video lên GCS)
   - **Vertex AI User** (gọi Gemini)
4. Keys → Create Key → JSON → download
5. Upload file JSON lên Google Drive: `/content/drive/MyDrive/EDS/credentials/service-account.json`

### 1.3 Tạo GCS Bucket

1. Cloud Storage → Create Bucket
2. Tên: `your-name-eds-gems` (unique)
3. Location: `us-central1`
4. Access: Uniform

### 1.4 Bật Gemini trong Vertex AI

1. Vertex AI → Model Garden
2. Tìm "Gemini 2.5 Flash"
3. Request access nếu cần (thường đã có sẵn)

---

## Bước 2 — Cài Đặt Cloudflare Tunnel (thay thế ngrok)

### 2.1 Cài cloudflared

```powershell
winget install --id Cloudflare.cloudflared --silent --accept-package-agreements
```

### 2.2 Khởi động tunnel

```powershell
cloudflared tunnel --url http://localhost:8765
```

URL sẽ hiện dạng `https://xxxx.trycloudflare.com` — paste URL này vào Colab worker config.

### 2.3 (Tùy chọn) Tạo named tunnel cho URL cố định

1. Vào https://dash.cloudflare.com → Networks → Tunnels → Create a tunnel
2. Chọn "Cloudflared" connector
3. Đặt tên tunnel, copy token
4. Tạo config file `~/.cloudflared/config.yml`:
   ```yaml
   tunnel: <TUNNEL_ID>
   credentials-file: C:\Users\<USER>\.cloudflared\<TUNNEL_ID>.json
   ingress:
     - service: http://localhost:8765
   ```
5. Chạy: `cloudflared service install <TOKEN>`

---

## Bước 3 — Cấu Hình Local Backend

### 3.1 Tạo file cấu hình

Tạo file `user_settings.json` trong thư mục `data/` của EDS:

```json
{
  "data_dir": "D:/Hai/study/DeepLerning/BCDA/tools/emotion-data-studio/data",
  "gemini_model": "gemini-2.5-flash",
  "gemini_intensity_threshold": 0.6,
  "runtime_mode": "auto"
}
```

### 3.2 Thiết lập biến môi trường (tùy chọn)

Tạo file `.env` trong `tools/emotion-data-studio/`:

```bash
# Google Cloud
GOOGLE_APPLICATION_CREDENTIALS=D:/Hai/Drive/credentials/service-account.json
GCP_PROJECT_ID=aura-social-vn
GCS_BUCKET_NAME=eds-data-bucket-aura

# Cloud SQL (kết nối với Cloud Run API)
CLOUD_SQL_CONNECTION_NAME=aura-social-vn:asia-southeast1:eds-postgres
CLOUD_SQL_USER=eds_user
CLOUD_SQL_PASSWORD=EdsPassword2026!@#
CLOUD_SQL_DB=emotion_studio

# Gemini
GEMINI_MODEL=gemini-2.5-flash
GEMINI_TEMPERATURE=0.2
GEMINI_MAX_TOKENS=8192
GEMINI_INTENSITY_THRESHOLD=0.6

# Ngrok
NGROK_TOKEN=your_ngrok_token_here
```

---

## Bước 4 — Khởi Động Hệ Thống

### 4.1 PC: Chạy Local Backend

```bash
cd D:/Hai/study/DeepLerning/BCDA/tools/emotion-data-studio
python backend/main.py
```

Backend sẽ chạy tại `http://localhost:8765`

### 4.2 PC: Mở Cloudflare Tunnel (terminal riêng)

```bash
cloudflared tunnel --url http://localhost:8765
```

Copy URL (ví dụ: `https://abc123.trycloudflare.com`) — dùng cho Colab worker.

### 4.3 Colab: Chạy GPU Worker

1. Mở notebook: `eds_colab.ipynb`
2. Chạy lần lượt các cells 1-4 (cài đặt môi trường)
3. Cell 5: paste NGROK_TOKEN (từ bước 4.2)
4. Chạy cell 5 → Colab đăng ký worker với local backend

### 4.4 Dashboard: Kiểm tra worker status

Mở `http://localhost:8765` → tab Processing → xem panel **GPU Workers (Colab)**
```
✅ Tesla T4 | Queue: 0 queued, 0 running
```

---

## Bước 5 — Sử Dụng Hệ Thống

### Thêm video để xử lý

1. Dashboard → tab **Thu Hoạch**
2. Paste YouTube URL hoặc đường dẫn local
3. Nhấn **Thêm vào hàng đợi**

### Video được xử lý như thế nào

```
User thêm video
    ↓
Local backend: tạo Video + ProcessQueue (status="queued")
    ↓
Colab worker poll: GET /api/worker/claim
    ↓
Colab: nhận job → chạy PipelineOrchestrator
    ↓
Colab: upload clips/frames lên local backend
    ↓
Colab: POST /api/worker/complete
    ↓
Local backend: cập nhật Video status="completed"
    ↓
Dashboard: clip hiển thị trong tab Review
```

### Xem tiến trình

- **Dashboard**: http://localhost:8765/processing
- **API status**: http://localhost:8765/api/worker/status
- **Logs**: Console của Colab notebook

---

## Chi Phí Ước Tính

| Thành phần | Chi phí | Ghi chú |
|:---|:---|:---|
| Colab Pro (GPU T4) | ~$10/tháng | Unlimited GPU time với Pro |
| Colab Pro+ (A100) | ~$50/tháng | Heavy workloads |
| Cloudflare Tunnel | Miễn phí | 1 tunnel, URL đổi mỗi lần (dùng named tunnel để có URL cố định) |
| **Tổng ước tính** | **~$10-15/tháng** | Rẻ hơn nhiều so với server GPU |

Với ngân sách GCP $1000 (27 triệu VND):
- Có thể phân tích **~100,000 video clips** với Gemini

---

## Troubleshooting

### "Worker not registered"

Colab chưa kết nối được. Kiểm tra:
1. Local backend có đang chạy không? (`python backend/main.py`)
2. Cloudflare Tunnel có đang mở không? (`cloudflared tunnel --url http://localhost:8765`)
3. Tunnel URL có đúng không?
4. Colab có internet không?

### "No module named backend"

```python
import sys
sys.path.insert(0, "/content/BCDA/tools/emotion-data-studio")
sys.path.insert(0, "/content/BCDA")
```

### "Gemini not configured"

```bash
# Kiểm tra credentials
ls -la /content/drive/MyDrive/EDS/credentials/
# Nếu file JSON tên khác, cập nhật đường dẫn trong cell 4

# Verify
from backend.services.gemini_auto_labeler import GeminiAutoLabeler
labeler = GeminiAutoLabeler()
print(labeler.status())
```

### Colab bị disconnect khi đang xử lý

- Dùng Colab Pro với High-RAM runtime
- Chạy cell keep-alive:
  ```javascript
  // Nhấn Ctrl+Shift+i → Console → paste:
  function KeepClicking(){
    document.querySelector("#connect > div").click();
  }
  setInterval(KeepClicking, 60000);
  ```
- Hoặc dùng Google Cloud VM với GPU persistent

---

## File Cấu Hình Quan Trọng

| File | Mô tả |
|:---|:---|
| `backend/config.py` | Cấu hình chính (port, data dir, API keys) |
| `backend/api/colab_worker.py` | API cho Colab worker đăng ký/nhận việc |
| `backend/api/gemini_api.py` | API cho Gemini auto-labeling |
| `backend/services/gemini_auto_labeler.py` | Service gọi Gemini |
| `colab/colab_worker.py` | Script chạy trên Colab |
| `colab/eds_colab.ipynb` | Notebook chính |
