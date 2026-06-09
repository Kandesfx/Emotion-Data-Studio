# Cloud Run Deployment Guide — Emotion Data Studio API

## Overview

Cloud Run chỉ deploy phần **REST API** (FastAPI backend). Desktop app giao tiếp qua API này để sync data lên Cloud SQL.

**Lưu ý quan trọng:** Cloud Run chạy `app_cloud.py` — nó **KHÔNG** chạy pipeline AI (Whisper, PhoBERT, DeepFace). Pipeline chạy trên **desktop app** hoặc **Colab worker**. Cloud Run chỉ phục vụ:
- CRUD videos/clips/labels
- Sync metadata (Desktop ↔ Cloud SQL)
- Health checks

---

## Prerequisites

### 1. Google Cloud SDK

```powershell
# Kiểm tra đã cài chưa
gcloud --version

# Nếu chưa cài, tải tại:
# https://dl.google.com/dl/cloudsdk/channels/rapid/google-cloud-sdk.zip
# Hoặc dùng installer: https://cloud.google.com/sdk/docs/install
```

### 2. Docker Desktop

```powershell
docker --version

# Nếu chưa cài:
# https://www.docker.com/products/docker-desktop/
# Sau khi cài, chạy Docker Desktop app trước
```

### 3. Enable Cloud APIs

```powershell
gcloud init
gcloud auth login

# Enable APIs cần thiết
gcloud services enable \
  run.googleapis.com \
  artifactregistry.googleapis.com \
  cloudbuild.googleapis.com \
  compute.googleapis.com \
  servicenetworking.googleapis.com \
  sqladmin.googleapis.com \
  storage.googleapis.com
```

### 4. Tạo Cloud SQL PostgreSQL

```powershell
# Tạo instance (chạy 1 lần)
gcloud sql instances create eds-postgres \
  --database-version=POSTGRES_15 \
  --region=asia-southeast1 \
  --tier=db-g1-small \
  --storage-type=SSD \
  --storage-size=10GB \
  --availability-type=ZONAL

# Tạo database
gcloud sql databases create eds_db --instance=eds-postgres

# Tạo user
gcloud sql users create eds_user \
  --instance=eds-postgres \
  --password=YOUR_SECURE_PASSWORD
```

### 5. Cập nhật `cloudbuild.yaml`

Sửa substitution default:

```yaml
# deploy/cloudbuild.yaml — thay thế dòng này:
substitutions:
  _CLOUD_SQL_INSTANCE: 'your-project:asia-southeast1:eds-postgres'
```

Thành:

```yaml
substitutions:
  _CLOUD_SQL_INSTANCE: 'YOUR_PROJECT_ID:asia-southeast1:eds-postgres'
```

### 6. Cập nhật `.env` (Cloud SQL credentials)

Thêm vào `.env`:

```
CLOUD_SQL_CONNECTION_NAME=YOUR_PROJECT_ID:asia-southeast1:eds-postgres
CLOUD_SQL_USER=eds_user
CLOUD_SQL_PASSWORD=YOUR_SECURE_PASSWORD
CLOUD_SQL_DB=eds_db
ENV=production
```

---

## Build & Deploy

### Cách 1: Dùng Cloud Build (recommended)

```powershell
cd d:\Hai\study\DeepLerning\BCDA\tools\emotion-data-studio

# Build + Push + Deploy tự động
gcloud builds submit --config=deploy/cloudbuild.yaml --region=asia-southeast1 .
```

Sau khi deploy thành công, Cloud Run sẽ trả URL:

```
https://emotion-data-studio-api-xxxxx-as.a.run.app
```

### Cách 2: Build Docker local + Deploy

```powershell
cd d:\Hai\study\DeepLerning\BCDA\tools\emotion-data-studio

# Build image
docker build -t eds-api -f deploy/Dockerfile .

# Tag cho Artifact Registry
docker tag eds-api gcr.io/YOUR_PROJECT_ID/emotion-data-studio-api:latest

# Push lên GCR
docker push gcr.io/YOUR_PROJECT_ID/emotion-data-studio-api:latest

# Deploy lên Cloud Run
gcloud run deploy emotion-data-studio-api \
  --image=gcr.io/YOUR_PROJECT_ID/emotion-data-studio-api:latest \
  --region=asia-southeast1 \
  --platform=managed \
  --memory=2Gi \
  --cpu=2 \
  --timeout=300 \
  --max-instances=3 \
  --allow-unauthenticated \
  --add-cloudsql-instances=YOUR_PROJECT_ID:asia-southeast1:eds-postgres
```

---

## Verify Deployment

```powershell
# Kiểm tra service
curl https://YOUR_CLOUD_RUN_URL/health

# Kết quả mong đợi:
# {"status":"healthy","version":"1.0.0"}

# Kiểm tra status
curl https://YOUR_CLOUD_RUN_URL/api/status
```

---

## Update Deployment

Khi code thay đổi, chỉ cần chạy lại:

```powershell
gcloud builds submit --config=deploy/cloudbuild.yaml --region=asia-southeast1 .
```

---

## Tắt/Delete Service

```powershell
# Tắt (service vẫn còn, không tính phí)
gcloud run services update emotion-data-studio-api --region=asia-southeast1 --no-traffic

# Xóa hoàn toàn
gcloud run services delete emotion-data-studio-api --region=asia-southeast1
```

---

## Chi phí ước tính (tháng)

| Resource | Spec | Chi phí ước tính |
|----------|------|-------------------|
| Cloud Run | 2 CPU, 2Gi, 3 max instances | ~$5-15/tháng |
| Cloud SQL | db-g1-small, ZONAL | ~$25-30/tháng |
| Cloud Storage | Artifact Registry images | ~$1-5/tháng |
| **Tổng** | | **~$35-50/tháng** |

---

## Troubleshooting

### Lỗi `docker: command not found`
Docker chưa cài. Cài Docker Desktop từ https://docker.com và khởi động app trước.

### Lỗi `gcloud: command not found`
Google Cloud SDK chưa cài. Tải và cài từ https://cloud.google.com/sdk/docs/install

### Lỗi `AccessDeniedException`
```powershell
# Kiểm tra quyền
gcloud auth list
gcloud projects get-iam-policy YOUR_PROJECT_ID

# Thường cần roles: Cloud Run Admin, Cloud Build Editor, Service Account User
gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
  --member="user:YOUR_EMAIL" \
  --role="roles/run.admin"
```

### Lỗi Cloud SQL connection
Kiểm tra `CLOUD_SQL_CONNECTION_NAME` đúng format:
`project:region:instance`

### Lỗi 500 khi gọi API
Kiểm tra Cloud Run logs:
```powershell
gcloud run services logs read emotion-data-studio-api --region=asia-southeast1 --limit=50
```
