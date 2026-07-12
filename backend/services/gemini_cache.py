"""Gemini response cache (Sprint 3).

Cache JSON response cho Stage 2 (segments detect) va Stage 4 (Verify pass).
Key duoc hash tu (video_path, stage, prompt_hash, params_hash).

Khi user run lai cung video → khong goi Gemini lai, tiet kiem cost.
Cache TTL mac dinh 7 ngay (config duoc).

Luu y: Cache chi hieu qua neu prompt + params khong doi. Neu doi prompt
(AUTOCUT_SYSTEM_PROMPT) → xoa cache.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

from backend.config import settings


_CACHE_TTL_SEC = 7 * 24 * 60 * 60   # 7 ngay


def _cache_dir() -> Path:
    """Thu muc cache Gemini responses."""
    d = settings.DATA_DIR / "cache" / "gemini"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _key_hash(*parts: Any) -> str:
    """Tao SHA256 hex tu cac phan (video_path, stage, prompt_hash, params)."""
    blob = json.dumps(parts, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:32]


def _cache_path(key: str) -> Path:
    return _cache_dir() / f"{key}.json"


def get(key: str) -> dict[str, Any] | None:
    """Doc cache, tra None neu khong co / het han / loi."""
    path = _cache_path(key)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    ts = data.get("_cached_at", 0)
    if time.time() - ts > _CACHE_TTL_SEC:
        try:
            path.unlink()
        except Exception:
            pass
        return None
    return data.get("payload")


def put(key: str, payload: Any) -> None:
    """Luu payload vao cache."""
    path = _cache_path(key)
    blob = {
        "_cached_at": time.time(),
        "payload": payload,
    }
    try:
        path.write_text(json.dumps(blob, ensure_ascii=False), encoding="utf-8")
    except Exception as exc:
        # Cache that bai khong anh huong pipeline
        print(f"[GeminiCache] put failed: {exc}")


def make_key(
    video_path: str,
    stage: str,
    prompt: str,
    params: dict[str, Any] | None = None,
) -> str:
    """Tao cache key tu video_path + stage + prompt + params."""
    prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]
    params_hash = hashlib.sha256(
        json.dumps(params or {}, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()[:16]
    return _key_hash(video_path, stage, prompt_hash, params_hash)


def clear_all() -> int:
    """Xoa toan bo cache (dung khi doi prompt)."""
    d = _cache_dir()
    count = 0
    for p in d.glob("*.json"):
        try:
            p.unlink()
            count += 1
        except Exception:
            pass
    return count


def stats() -> dict[str, Any]:
    """Thong ke cache: so entry, tong size, entry cu nhat."""
    d = _cache_dir()
    entries = list(d.glob("*.json"))
    total_bytes = sum(p.stat().st_size for p in entries if p.exists())
    oldest_ts = None
    for p in entries[:50]:  # sample 50
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            ts = data.get("_cached_at", 0)
            if oldest_ts is None or ts < oldest_ts:
                oldest_ts = ts
        except Exception:
            pass
    return {
        "entries": len(entries),
        "total_bytes": total_bytes,
        "oldest_cached_at": oldest_ts,
        "ttl_seconds": _CACHE_TTL_SEC,
        "cache_dir": str(d),
    }