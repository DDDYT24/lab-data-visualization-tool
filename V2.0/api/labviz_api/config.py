"""Environment-backed configuration for the LabViz API."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

DEFAULT_SHARE_TOKEN_KEYS = ((1, "labviz-development-share-token-key-v1"),)


def _as_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _share_token_keys(value: str | None) -> tuple[tuple[int, str], ...]:
    if not value:
        return DEFAULT_SHARE_TOKEN_KEYS
    pairs: list[tuple[int, str]] = []
    for item in value.split(","):
        version, separator, secret = item.strip().partition("=")
        if not separator or not version.isdigit() or not secret:
            raise ValueError("LABVIZ_SHARE_TOKEN_KEYS must use version=secret pairs.")
        pairs.append((int(version), secret))
    return tuple(pairs)


@dataclass(frozen=True)
class Settings:
    database_path: Path
    allowed_origins: tuple[str, ...]
    public_web_url: str
    environment: str = "development"
    project_ttl_seconds: int = 7_200
    export_ttl_seconds: int = 7_200
    session_ttl_seconds: int = 604_800
    max_upload_bytes: int = 50 * 1024 * 1024
    auth_mode: str = "console"
    smtp_host: str | None = None
    smtp_port: int = 587
    smtp_username: str | None = None
    smtp_password: str | None = None
    smtp_from: str = "LabViz <noreply@localhost>"
    smtp_starttls: bool = True
    cookie_secure: bool = False
    postgres_url: str | None = None
    postgres_echo: bool = False
    object_storage_root: Path = Path(".labviz/objects")
    object_storage_backend: str = "local"
    object_storage_cursor_ttl_seconds: int = 86_400
    s3_bucket: str | None = None
    s3_prefix: str = ""
    s3_region: str | None = None
    s3_endpoint_url: str | None = None
    s3_multipart_threshold_bytes: int = 16 * 1024 * 1024
    s3_multipart_part_size_bytes: int = 8 * 1024 * 1024
    s3_connect_timeout_seconds: int = 5
    s3_read_timeout_seconds: int = 60
    persistence_backend: str = "sqlite"
    share_token_key_version: int = 1
    share_token_keys: tuple[tuple[int, str], ...] = DEFAULT_SHARE_TOKEN_KEYS
    worker_batch_size: int = 25
    worker_lease_seconds: int = 60
    worker_heartbeat_seconds: int = 20
    worker_poll_seconds: int = 5
    worker_backoff_base_seconds: int = 30
    worker_backoff_max_seconds: int = 3_600
    worker_max_retries: int = 5
    worker_destructive_maintenance: bool = False
    worker_dry_run: bool = True
    worker_delete_enabled: bool = False
    worker_gc_orphan_age_seconds: int = 86_400
    worker_orphan_staging_grace_seconds: int = 3_600

    def __post_init__(self) -> None:
        if self.environment not in {"development", "test", "production"}:
            raise ValueError("LABVIZ_ENVIRONMENT must be 'development', 'test', or 'production'.")
        if self.auth_mode not in {"console", "smtp"}:
            raise ValueError("LABVIZ_AUTH_MODE must be 'console' or 'smtp'.")
        if self.environment == "production" and self.auth_mode == "console":
            raise ValueError("Production must use SMTP authentication delivery.")
        if self.environment == "production" and not self.cookie_secure:
            raise ValueError("Production requires secure cookies.")
        if self.persistence_backend not in {"sqlite", "postgresql"}:
            raise ValueError("LABVIZ_PERSISTENCE_BACKEND must be 'sqlite' or 'postgresql'.")
        if self.persistence_backend == "postgresql" and not self.postgres_url:
            raise ValueError("LABVIZ_POSTGRES_URL is required for PostgreSQL persistence.")
        if self.object_storage_backend not in {"local", "s3"}:
            raise ValueError("LABVIZ_OBJECT_STORAGE_BACKEND must be 'local' or 's3'.")
        if self.object_storage_backend == "s3" and not self.s3_bucket:
            raise ValueError("LABVIZ_S3_BUCKET is required for S3 object storage.")
        if self.object_storage_cursor_ttl_seconds < 1:
            raise ValueError("Object storage cursor TTL must be positive.")
        if self.s3_multipart_part_size_bytes < 5 * 1024 * 1024:
            raise ValueError("S3 multipart part size must be at least 5 MiB.")
        if self.s3_multipart_threshold_bytes < self.s3_multipart_part_size_bytes:
            raise ValueError("S3 multipart threshold must be at least the part size.")
        if min(self.s3_connect_timeout_seconds, self.s3_read_timeout_seconds) < 1:
            raise ValueError("S3 connection and read timeouts must be positive.")
        versions = [version for version, _secret in self.share_token_keys]
        if self.share_token_key_version not in versions:
            raise ValueError(
                "LABVIZ_SHARE_TOKEN_KEY_VERSION is missing from the configured key ring."
            )
        if len(set(versions)) != len(versions) or any(version < 1 for version in versions):
            raise ValueError("Share token key versions must be unique positive integers.")
        if any(len(secret.encode("utf-8")) < 32 for _version, secret in self.share_token_keys):
            raise ValueError("Share token keys must contain at least 32 UTF-8 bytes.")
        if self.environment == "production" and self.share_token_keys == DEFAULT_SHARE_TOKEN_KEYS:
            raise ValueError("Production requires an explicit LABVIZ_SHARE_TOKEN_KEYS key ring.")
        if self.worker_batch_size < 1 or self.worker_poll_seconds < 1:
            raise ValueError("Worker batch and poll settings must be positive.")
        if self.worker_lease_seconds < 2:
            raise ValueError("Worker lease must be at least two seconds.")
        if not 0 < self.worker_heartbeat_seconds < self.worker_lease_seconds:
            raise ValueError("Worker heartbeat must be positive and shorter than the lease.")
        if self.worker_max_retries < 1:
            raise ValueError("Worker retries must be at least one.")
        if self.worker_gc_orphan_age_seconds < 1:
            raise ValueError("Worker GC orphan age must be positive.")
        if self.worker_orphan_staging_grace_seconds < 1:
            raise ValueError("Orphan staging grace must be positive.")
        if (
            self.worker_backoff_base_seconds < 1
            or self.worker_backoff_max_seconds < self.worker_backoff_base_seconds
        ):
            raise ValueError("Worker backoff settings must be positive and bounded.")

    @classmethod
    def from_env(cls, base_dir: Path | None = None) -> Settings:
        root = base_dir or Path(__file__).resolve().parents[1]
        db_path = Path(
            os.environ.get("LABVIZ_DATABASE_PATH", root / ".labviz" / "labviz-v2.db")
        ).expanduser()
        origins = tuple(
            origin.strip()
            for origin in os.environ.get(
                "LABVIZ_ALLOWED_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000"
            ).split(",")
            if origin.strip()
        )
        auth_mode = os.environ.get("LABVIZ_AUTH_MODE", "console").strip().lower()
        if auth_mode not in {"console", "smtp"}:
            raise ValueError("LABVIZ_AUTH_MODE must be 'console' or 'smtp'.")

        return cls(
            database_path=db_path,
            allowed_origins=origins,
            public_web_url=os.environ.get("LABVIZ_PUBLIC_WEB_URL", "http://localhost:3000").rstrip(
                "/"
            ),
            environment=os.environ.get("LABVIZ_ENVIRONMENT", "development").strip().lower(),
            project_ttl_seconds=int(os.environ.get("LABVIZ_PROJECT_TTL_SECONDS", "7200")),
            export_ttl_seconds=int(os.environ.get("LABVIZ_EXPORT_TTL_SECONDS", "7200")),
            session_ttl_seconds=int(os.environ.get("LABVIZ_SESSION_TTL_SECONDS", "604800")),
            max_upload_bytes=int(os.environ.get("LABVIZ_MAX_UPLOAD_BYTES", str(50 * 1024 * 1024))),
            auth_mode=auth_mode,
            smtp_host=os.environ.get("LABVIZ_SMTP_HOST"),
            smtp_port=int(os.environ.get("LABVIZ_SMTP_PORT", "587")),
            smtp_username=os.environ.get("LABVIZ_SMTP_USERNAME"),
            smtp_password=os.environ.get("LABVIZ_SMTP_PASSWORD"),
            smtp_from=os.environ.get("LABVIZ_SMTP_FROM", "LabViz <noreply@localhost>"),
            smtp_starttls=_as_bool(os.environ.get("LABVIZ_SMTP_STARTTLS"), True),
            cookie_secure=_as_bool(os.environ.get("LABVIZ_COOKIE_SECURE")),
            postgres_url=os.environ.get("LABVIZ_POSTGRES_URL"),
            postgres_echo=_as_bool(os.environ.get("LABVIZ_POSTGRES_ECHO")),
            object_storage_root=Path(
                os.environ.get("LABVIZ_OBJECT_STORAGE_ROOT", root / ".labviz" / "objects")
            ).expanduser(),
            object_storage_backend=os.environ.get("LABVIZ_OBJECT_STORAGE_BACKEND", "local")
            .strip()
            .lower(),
            object_storage_cursor_ttl_seconds=int(
                os.environ.get("LABVIZ_OBJECT_STORAGE_CURSOR_TTL_SECONDS", "86400")
            ),
            s3_bucket=os.environ.get("LABVIZ_S3_BUCKET"),
            s3_prefix=os.environ.get("LABVIZ_S3_PREFIX", ""),
            s3_region=os.environ.get("LABVIZ_S3_REGION"),
            s3_endpoint_url=os.environ.get("LABVIZ_S3_ENDPOINT_URL"),
            s3_multipart_threshold_bytes=int(
                os.environ.get("LABVIZ_S3_MULTIPART_THRESHOLD_BYTES", str(16 * 1024 * 1024))
            ),
            s3_multipart_part_size_bytes=int(
                os.environ.get("LABVIZ_S3_MULTIPART_PART_SIZE_BYTES", str(8 * 1024 * 1024))
            ),
            s3_connect_timeout_seconds=int(
                os.environ.get("LABVIZ_S3_CONNECT_TIMEOUT_SECONDS", "5")
            ),
            s3_read_timeout_seconds=int(os.environ.get("LABVIZ_S3_READ_TIMEOUT_SECONDS", "60")),
            persistence_backend=os.environ.get("LABVIZ_PERSISTENCE_BACKEND", "sqlite")
            .strip()
            .lower(),
            share_token_key_version=int(os.environ.get("LABVIZ_SHARE_TOKEN_KEY_VERSION", "1")),
            share_token_keys=_share_token_keys(os.environ.get("LABVIZ_SHARE_TOKEN_KEYS")),
            worker_batch_size=int(os.environ.get("LABVIZ_WORKER_BATCH_SIZE", "25")),
            worker_lease_seconds=int(os.environ.get("LABVIZ_WORKER_LEASE_SECONDS", "60")),
            worker_heartbeat_seconds=int(os.environ.get("LABVIZ_WORKER_HEARTBEAT_SECONDS", "20")),
            worker_poll_seconds=int(os.environ.get("LABVIZ_WORKER_POLL_SECONDS", "5")),
            worker_backoff_base_seconds=int(
                os.environ.get("LABVIZ_WORKER_BACKOFF_BASE_SECONDS", "30")
            ),
            worker_backoff_max_seconds=int(
                os.environ.get("LABVIZ_WORKER_BACKOFF_MAX_SECONDS", "3600")
            ),
            worker_max_retries=int(os.environ.get("LABVIZ_WORKER_MAX_RETRIES", "5")),
            worker_destructive_maintenance=_as_bool(
                os.environ.get("LABVIZ_WORKER_DESTRUCTIVE_MAINTENANCE")
            ),
            worker_dry_run=_as_bool(os.environ.get("LABVIZ_WORKER_DRY_RUN"), True),
            worker_delete_enabled=_as_bool(os.environ.get("LABVIZ_WORKER_DELETE_ENABLED"), False),
            worker_gc_orphan_age_seconds=int(
                os.environ.get("LABVIZ_WORKER_GC_ORPHAN_AGE_SECONDS", "86400")
            ),
            worker_orphan_staging_grace_seconds=int(
                os.environ.get("LABVIZ_WORKER_ORPHAN_STAGING_GRACE_SECONDS", "3600")
            ),
        )
