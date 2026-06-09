import os
import sys
import boto3
from botocore.client import Config
from pathlib import Path

# Add project root to path to import settings
sys.path.append(str(Path(__file__).resolve().parent.parent))

from backend.config import settings

def upload_to_r2(file_path: str, bucket_name: str, object_name: str, content_type: str = None):
    # Load credentials from settings/env
    r2_endpoint = settings.R2_ENDPOINT
    access_key = settings.AWS_ACCESS_KEY_ID
    secret_key = settings.AWS_SECRET_ACCESS_KEY
    
    if not all([r2_endpoint, access_key, secret_key]):
        print("Error: R2 credentials are not set in .env file.")
        print(f"R2_ENDPOINT: {r2_endpoint}")
        print(f"AWS_ACCESS_KEY_ID: {access_key}")
        print("Please check your .env file.")
        sys.exit(1)
        
    print(f"Connecting to Cloudflare R2 endpoint: {r2_endpoint}...")
    
    # Initialize S3 client for R2
    s3 = boto3.client(
        's3',
        endpoint_url=r2_endpoint,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        config=Config(signature_version='s3v4'),
        region_name='auto'  # R2 requires region_name='auto'
    )
    
    extra_args = {}
    if content_type:
        extra_args['ContentType'] = content_type
        
    print(f"Uploading {file_path} to bucket '{bucket_name}' as '{object_name}'...")
    try:
        s3.upload_file(
            file_path,
            bucket_name,
            object_name,
            ExtraArgs=extra_args
        )
        print("Upload successful!")
    except Exception as e:
        print(f"Upload failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python upload_r2.py <file_path> <bucket_name> <object_name> [content_type]")
        sys.exit(1)
        
    file_p = sys.argv[1]
    bucket_n = sys.argv[2]
    obj_n = sys.argv[3]
    content_t = sys.argv[4] if len(sys.argv) > 4 else None
    
    upload_to_r2(file_p, bucket_n, obj_n, content_t)
