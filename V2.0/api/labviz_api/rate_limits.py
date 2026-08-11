"""Shared fixed-window authentication limiter invariants."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta


class AuthRateLimitUnavailable(RuntimeError):
    """The authoritative limiter could not safely decide a request."""

    code = "auth-rate-limit-unavailable"


def auth_bucket_specs(
    *,
    client_key: str,
    email: str,
    client_limit: int,
    email_limit: int,
    window_seconds: int,
) -> tuple[tuple[str, str, int], tuple[str, str, int]]:
    if len(client_key) != 64 or any(
        character not in "0123456789abcdef" for character in client_key
    ):
        raise ValueError("Authentication client keys must be lowercase SHA-256 digests.")
    if email != email.strip().lower() or not 3 <= len(email) <= 320:
        raise ValueError("Authentication limiter emails must be normalized.")
    if min(client_limit, email_limit, window_seconds) < 1:
        raise ValueError("Authentication limiter values must be positive.")
    return (
        ("client", client_key, client_limit),
        ("email", email, email_limit),
    )


def fixed_window(database_now: datetime, window_seconds: int) -> tuple[datetime, datetime]:
    if database_now.tzinfo is None:
        raise ValueError("Authentication limiter time must be timezone-aware.")
    start_epoch = int(database_now.timestamp()) // window_seconds * window_seconds
    window_started_at = datetime.fromtimestamp(start_epoch, tz=UTC)
    return window_started_at, window_started_at + timedelta(seconds=window_seconds)
