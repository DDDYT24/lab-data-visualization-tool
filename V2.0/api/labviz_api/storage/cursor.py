"""Opaque, scope-bound continuation cursors for provider inventory."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
from datetime import UTC, datetime, timedelta
from typing import Any

from .base import InvalidStorageCursor

CURSOR_VERSION = 1
CURSOR_DOMAIN = b"labviz-storage-inventory-cursor-v1\0"


def encode_cursor(
    *,
    backend_name: str,
    inventory_scope: str,
    state: dict[str, Any],
    issued_at: datetime,
    ttl_seconds: int,
) -> str:
    payload = {
        "v": CURSOR_VERSION,
        "backend": backend_name,
        "scope": inventory_scope,
        "issuedAt": issued_at.astimezone(UTC).isoformat(),
        "expiresAt": (issued_at + timedelta(seconds=ttl_seconds)).astimezone(UTC).isoformat(),
        "state": state,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    envelope = {
        "payload": base64.urlsafe_b64encode(encoded).decode("ascii").rstrip("="),
        "checksum": hashlib.sha256(CURSOR_DOMAIN + encoded).hexdigest(),
    }
    raw = json.dumps(envelope, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def decode_cursor(
    cursor: str,
    *,
    backend_name: str,
    inventory_scope: str,
    now: datetime,
) -> dict[str, Any]:
    try:
        raw = _urlsafe_decode(cursor)
        envelope = json.loads(raw)
        encoded = _urlsafe_decode(envelope["payload"])
        checksum = hashlib.sha256(CURSOR_DOMAIN + encoded).hexdigest()
        if not hmac.compare_digest(checksum, envelope["checksum"]):
            raise InvalidStorageCursor("Storage inventory cursor checksum is invalid.")
        payload = json.loads(encoded)
        expires_at = datetime.fromisoformat(payload["expiresAt"])
    except InvalidStorageCursor:
        raise
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise InvalidStorageCursor("Storage inventory cursor is malformed.") from exc
    if payload.get("v") != CURSOR_VERSION:
        raise InvalidStorageCursor("Storage inventory cursor version is unsupported.")
    if payload.get("backend") != backend_name or payload.get("scope") != inventory_scope:
        raise InvalidStorageCursor("Storage inventory cursor belongs to another provider scope.")
    if expires_at.tzinfo is None or expires_at <= now.astimezone(UTC):
        raise InvalidStorageCursor("Storage inventory cursor has expired.")
    state = payload.get("state")
    if not isinstance(state, dict):
        raise InvalidStorageCursor("Storage inventory cursor state is malformed.")
    return state


def _urlsafe_decode(value: str) -> bytes:
    if not isinstance(value, str) or not value or len(value) > 16_384:
        raise InvalidStorageCursor("Storage inventory cursor is malformed.")
    padding = "=" * (-len(value) % 4)
    try:
        return base64.b64decode(value + padding, altchars=b"-_", validate=True)
    except ValueError as exc:
        raise InvalidStorageCursor("Storage inventory cursor is malformed.") from exc
