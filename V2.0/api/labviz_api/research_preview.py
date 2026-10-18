"""Bounded private screen previews from immutable publication PNG bytes."""

from __future__ import annotations

import hashlib
import io
import threading
from collections import OrderedDict

from PIL import Image

_lock = threading.RLock()
_resize_locks = tuple(threading.Lock() for _ in range(32))
_cache: OrderedDict[tuple[str, str, int, str], bytes] = OrderedDict()
_cache_generation = 0
_MAX_BYTES = 8 * 2**20
_MAX_ENTRIES = 32


def preview_png(project_id: str, artifact_id: str, payload: bytes, width: int) -> bytes:
    key = (project_id, artifact_id, width, hashlib.sha256(payload).hexdigest())
    # Fixed lock stripes bound synchronization state and coalesce a warm-up with
    # an arriving image request. Do not hold the global LRU lock during resizing.
    with _resize_locks[hash(key) % len(_resize_locks)]:
        with _lock:
            if key in _cache:
                _cache.move_to_end(key)
                return _cache[key]
            generation = _cache_generation
        with Image.open(io.BytesIO(payload)) as image:
            image.thumbnail((width, 4000), Image.Resampling.LANCZOS)
            stream = io.BytesIO()
            image.save(stream, format="PNG")
        encoded = stream.getvalue()
        with _lock:
            # A project may be deleted while its optional warm-up is running.
            # One scalar also bounds invalidation state for in-flight resizes.
            if generation != _cache_generation:
                return encoded
            _cache[key] = encoded
            _cache.move_to_end(key)
            while len(_cache) > _MAX_ENTRIES or sum(len(v) for v in _cache.values()) > _MAX_BYTES:
                _cache.popitem(last=False)
        return encoded


def clear_previews(project_id: str) -> None:
    global _cache_generation
    with _lock:
        _cache_generation += 1
        for key in list(_cache):
            if key[0] == project_id:
                del _cache[key]
