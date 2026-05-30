"""
Emotion Data Studio — Google Cloud Storage Client
===================================================
Upload/download files to/from Google Cloud Storage.
Used for syncing videos, clips, and model weights.
"""

import os
import logging
from pathlib import Path
from typing import Optional, List

logger = logging.getLogger(__name__)


class GCSClient:
    """
    Google Cloud Storage client wrapper.
    Handles file upload/download with progress tracking.
    """

    def __init__(self):
        from backend.config import settings
        self.bucket_name = settings.GCS_BUCKET_NAME
        self.credentials_path = settings.GOOGLE_APPLICATION_CREDENTIALS
        self._client = None
        self._bucket = None

    @property
    def is_configured(self) -> bool:
        """Check if GCS is properly configured"""
        return bool(self.bucket_name and self.credentials_path)

    def _get_client(self):
        """Lazy-init Google Cloud Storage client"""
        if self._client is None:
            try:
                from google.cloud import storage

                if self.credentials_path:
                    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = self.credentials_path

                self._client = storage.Client()
                self._bucket = self._client.bucket(self.bucket_name)
                logger.info(f"GCS client initialized: bucket={self.bucket_name}")
            except ImportError:
                logger.error("google-cloud-storage not installed. Run: pip install google-cloud-storage")
                raise
            except Exception as e:
                logger.error(f"GCS client init failed: {e}")
                raise
        return self._client

    def _get_bucket(self):
        """Get bucket reference"""
        if self._bucket is None:
            self._get_client()
        return self._bucket

    # ================================================================
    # UPLOAD
    # ================================================================

    def upload_file(self, local_path: str, gcs_path: str,
                    content_type: Optional[str] = None) -> str:
        """
        Upload a local file to GCS.
        
        Args:
            local_path: Path to local file
            gcs_path: Destination path in GCS bucket
            content_type: MIME type (auto-detected if None)
            
        Returns:
            GCS URI (gs://bucket/path)
        """
        bucket = self._get_bucket()
        blob = bucket.blob(gcs_path)

        if content_type:
            blob.content_type = content_type

        logger.info(f"Uploading: {local_path} → gs://{self.bucket_name}/{gcs_path}")
        blob.upload_from_filename(local_path)

        gcs_uri = f"gs://{self.bucket_name}/{gcs_path}"
        logger.info(f"Upload complete: {gcs_uri}")
        return gcs_uri

    def upload_directory(self, local_dir: str, gcs_prefix: str,
                         extensions: Optional[List[str]] = None,
                         on_progress=None) -> List[str]:
        """
        Upload entire directory to GCS.
        
        Args:
            local_dir: Local directory path
            gcs_prefix: Prefix in GCS bucket
            extensions: File extensions to include (e.g., ['.mp4', '.wav'])
            on_progress: Callback(uploaded, total)
            
        Returns:
            List of uploaded GCS URIs
        """
        uploaded = []
        local_path = Path(local_dir)

        files = list(local_path.rglob("*"))
        files = [f for f in files if f.is_file()]

        if extensions:
            files = [f for f in files if f.suffix.lower() in extensions]

        total = len(files)
        for i, file in enumerate(files):
            relative = file.relative_to(local_path)
            gcs_path = f"{gcs_prefix}/{relative.as_posix()}"

            uri = self.upload_file(str(file), gcs_path)
            uploaded.append(uri)

            if on_progress:
                on_progress(i + 1, total)

        return uploaded

    # ================================================================
    # DOWNLOAD
    # ================================================================

    def download_file(self, gcs_path: str, local_path: str) -> str:
        """
        Download a file from GCS to local.
        
        Args:
            gcs_path: Path in GCS bucket
            local_path: Destination local path
            
        Returns:
            Local file path
        """
        bucket = self._get_bucket()
        blob = bucket.blob(gcs_path)

        os.makedirs(os.path.dirname(local_path), exist_ok=True)

        logger.info(f"Downloading: gs://{self.bucket_name}/{gcs_path} → {local_path}")
        blob.download_to_filename(local_path)

        logger.info(f"Download complete: {local_path}")
        return local_path

    def download_directory(self, gcs_prefix: str, local_dir: str,
                           on_progress=None) -> List[str]:
        """
        Download all files under a GCS prefix.
        
        Args:
            gcs_prefix: Prefix in GCS bucket
            local_dir: Destination local directory
            on_progress: Callback(downloaded, total)
            
        Returns:
            List of downloaded local paths
        """
        bucket = self._get_bucket()
        blobs = list(bucket.list_blobs(prefix=gcs_prefix))
        total = len(blobs)
        downloaded = []

        for i, blob in enumerate(blobs):
            relative = blob.name[len(gcs_prefix):].lstrip("/")
            local_path = os.path.join(local_dir, relative)

            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            blob.download_to_filename(local_path)
            downloaded.append(local_path)

            if on_progress:
                on_progress(i + 1, total)

        return downloaded

    # ================================================================
    # UTILITY
    # ================================================================

    def file_exists(self, gcs_path: str) -> bool:
        """Check if a file exists in GCS"""
        bucket = self._get_bucket()
        blob = bucket.blob(gcs_path)
        return blob.exists()

    def delete_file(self, gcs_path: str):
        """Delete a file from GCS"""
        bucket = self._get_bucket()
        blob = bucket.blob(gcs_path)
        blob.delete()
        logger.info(f"Deleted: gs://{self.bucket_name}/{gcs_path}")

    def list_files(self, prefix: str = "", max_results: int = 1000) -> List[str]:
        """List files in GCS bucket under a prefix"""
        bucket = self._get_bucket()
        blobs = bucket.list_blobs(prefix=prefix, max_results=max_results)
        return [blob.name for blob in blobs]

    def get_signed_url(self, gcs_path: str, expiration_minutes: int = 60) -> str:
        """Generate a signed URL for temporary access"""
        from datetime import timedelta
        bucket = self._get_bucket()
        blob = bucket.blob(gcs_path)
        url = blob.generate_signed_url(
            expiration=timedelta(minutes=expiration_minutes),
            method="GET"
        )
        return url
