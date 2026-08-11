"""Environment-backed configuration for the LabViz API."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from email.utils import parseaddr
from ipaddress import ip_network
from pathlib import Path
from urllib.parse import parse_qs, urlparse

DEFAULT_SHARE_TOKEN_KEYS = ((1, "labviz-development-share-token-key-v1"),)
DEFAULT_CLIENT_IDENTITY_KEY = "labviz-development-client-identity-key-v1"
MAX_CLOUD_UPLOAD_BYTES = 50 * 1024 * 1024
MAX_AUTH_RATE_LIMIT_WINDOW_SECONDS = 86_400
MAX_AUTH_REQUEST_LIMIT = 10_000
SES_CONFIGURATION_SET_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


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


def _is_https_origin(value: str) -> bool:
    parsed = urlparse(value)
    return bool(
        parsed.scheme == "https"
        and parsed.netloc
        and parsed.path in {"", "/"}
        and not parsed.params
        and not parsed.query
        and not parsed.fragment
        and parsed.hostname not in {"localhost", "127.0.0.1", "::1"}
    )


def _postgres_requires_tls(value: str | None) -> bool:
    if not value:
        return False
    parameters = parse_qs(urlparse(value).query)
    return parameters.get("sslmode", [""])[-1] in {"require", "verify-ca", "verify-full"}


def _valid_email_sender(value: str | None) -> bool:
    if not value:
        return False
    _display_name, address = parseaddr(value)
    local, separator, domain = address.rpartition("@")
    return bool(
        separator
        and local
        and "." in domain
        and not domain.startswith(".")
        and not domain.endswith(".")
        and domain.lower() != "localhost"
    )


@dataclass(frozen=True)
class Settings:
    database_path: Path
    allowed_origins: tuple[str, ...]
    public_web_url: str
    environment: str = "development"
    runtime_role: str = "api"
    project_ttl_seconds: int = 7_200
    export_ttl_seconds: int = 7_200
    session_ttl_seconds: int = 604_800
    max_upload_bytes: int = MAX_CLOUD_UPLOAD_BYTES
    auth_mode: str = "console"
    smtp_host: str | None = None
    smtp_port: int = 587
    smtp_username: str | None = None
    smtp_password: str | None = None
    smtp_from: str = "LabViz <noreply@localhost>"
    smtp_starttls: bool = True
    ses_region: str | None = None
    ses_from: str | None = None
    ses_configuration_set: str | None = None
    ses_connect_timeout_seconds: int = 5
    ses_read_timeout_seconds: int = 15
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
    trusted_proxy_cidrs: tuple[str, ...] = ()
    trusted_proxy_hops: int = 0
    client_identity_key: str = DEFAULT_CLIENT_IDENTITY_KEY
    auth_rate_limit_window_seconds: int = 3_600
    auth_client_request_limit: int = 30
    auth_email_request_limit: int = 10
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
        if self.runtime_role not in {"api", "worker"}:
            raise ValueError("LABVIZ_RUNTIME_ROLE must be 'api' or 'worker'.")
        if self.auth_mode not in {"console", "smtp", "ses"}:
            raise ValueError("LABVIZ_AUTH_MODE must be 'console', 'smtp', or 'ses'.")
        if (
            self.environment == "production"
            and self.runtime_role == "api"
            and self.auth_mode != "ses"
        ):
            raise ValueError("Production API must use Amazon SES v2 authentication delivery.")
        if (
            self.environment == "production"
            and self.runtime_role == "api"
            and not self.cookie_secure
        ):
            raise ValueError("Production requires secure cookies.")
        if self.persistence_backend not in {"sqlite", "postgresql"}:
            raise ValueError("LABVIZ_PERSISTENCE_BACKEND must be 'sqlite' or 'postgresql'.")
        if self.persistence_backend == "postgresql" and not self.postgres_url:
            raise ValueError("LABVIZ_POSTGRES_URL is required for PostgreSQL persistence.")
        if self.object_storage_backend not in {"local", "s3"}:
            raise ValueError("LABVIZ_OBJECT_STORAGE_BACKEND must be 'local' or 's3'.")
        if self.object_storage_backend == "s3" and not self.s3_bucket:
            raise ValueError("LABVIZ_S3_BUCKET is required for S3 object storage.")
        if self.environment == "production":
            if self.persistence_backend != "postgresql":
                raise ValueError("Production requires PostgreSQL persistence.")
            if not _postgres_requires_tls(self.postgres_url):
                raise ValueError("Production PostgreSQL requires sslmode=require or stronger.")
            if self.object_storage_backend != "s3":
                raise ValueError("Production requires S3 object storage.")
            if not self.s3_region:
                raise ValueError("Production requires an explicit LABVIZ_S3_REGION.")
            if self.s3_endpoint_url:
                raise ValueError("Production AWS S3 must not use LABVIZ_S3_ENDPOINT_URL.")
            if self.runtime_role == "api":
                if not _is_https_origin(self.public_web_url):
                    raise ValueError(
                        "Production LABVIZ_PUBLIC_WEB_URL must be a public HTTPS origin."
                    )
                if not self.allowed_origins or any(
                    not _is_https_origin(origin) for origin in self.allowed_origins
                ):
                    raise ValueError(
                        "Production CORS origins must be non-empty public HTTPS origins."
                    )
                if not self.ses_region:
                    raise ValueError("Production requires LABVIZ_SES_REGION.")
                if self.ses_region != self.s3_region:
                    raise ValueError("Production SES and S3 must use the same AWS Region.")
                if not _valid_email_sender(self.ses_from):
                    raise ValueError("Production requires a valid LABVIZ_SES_FROM address.")
                if not self.ses_configuration_set:
                    raise ValueError("Production requires LABVIZ_SES_CONFIGURATION_SET.")
                if not SES_CONFIGURATION_SET_PATTERN.fullmatch(self.ses_configuration_set):
                    raise ValueError("LABVIZ_SES_CONFIGURATION_SET has an invalid name.")
                if any((self.smtp_host, self.smtp_username, self.smtp_password)):
                    raise ValueError("Production SES must not configure SMTP credentials.")
            if self.max_upload_bytes > MAX_CLOUD_UPLOAD_BYTES:
                raise ValueError("Production uploads cannot exceed the approved 50 MB limit.")
        if self.max_upload_bytes < 1:
            raise ValueError("Upload size limit must be positive.")
        if self.object_storage_cursor_ttl_seconds < 1:
            raise ValueError("Object storage cursor TTL must be positive.")
        if self.s3_multipart_part_size_bytes < 5 * 1024 * 1024:
            raise ValueError("S3 multipart part size must be at least 5 MiB.")
        if self.s3_multipart_threshold_bytes < self.s3_multipart_part_size_bytes:
            raise ValueError("S3 multipart threshold must be at least the part size.")
        if min(self.s3_connect_timeout_seconds, self.s3_read_timeout_seconds) < 1:
            raise ValueError("S3 connection and read timeouts must be positive.")
        if min(self.ses_connect_timeout_seconds, self.ses_read_timeout_seconds) < 1:
            raise ValueError("SES connection and read timeouts must be positive.")
        versions = [version for version, _secret in self.share_token_keys]
        if self.share_token_key_version not in versions:
            raise ValueError(
                "LABVIZ_SHARE_TOKEN_KEY_VERSION is missing from the configured key ring."
            )
        if len(set(versions)) != len(versions) or any(version < 1 for version in versions):
            raise ValueError("Share token key versions must be unique positive integers.")
        if any(len(secret.encode("utf-8")) < 32 for _version, secret in self.share_token_keys):
            raise ValueError("Share token keys must contain at least 32 UTF-8 bytes.")
        if (
            self.environment == "production"
            and self.runtime_role == "api"
            and self.share_token_keys == DEFAULT_SHARE_TOKEN_KEYS
        ):
            raise ValueError("Production requires an explicit LABVIZ_SHARE_TOKEN_KEYS key ring.")
        if not 0 <= self.trusted_proxy_hops <= 8:
            raise ValueError("LABVIZ_TRUSTED_PROXY_HOPS must be between zero and eight.")
        try:
            trusted_proxy_networks = tuple(
                ip_network(value, strict=False) for value in self.trusted_proxy_cidrs
            )
        except ValueError as exc:
            raise ValueError("LABVIZ_TRUSTED_PROXY_CIDRS contains an invalid network.") from exc
        if len(self.client_identity_key.encode("utf-8")) < 32:
            raise ValueError("LABVIZ_CLIENT_IDENTITY_KEY must contain at least 32 UTF-8 bytes.")
        if self.environment == "production" and self.runtime_role == "api":
            if not self.trusted_proxy_cidrs or self.trusted_proxy_hops < 1:
                raise ValueError(
                    "Production API requires trusted proxy CIDRs and a positive proxy-hop count."
                )
            if any(
                network.prefixlen == 0 or network.is_global for network in trusted_proxy_networks
            ):
                raise ValueError("Production trusted proxy CIDRs must be private network ranges.")
            if self.client_identity_key == DEFAULT_CLIENT_IDENTITY_KEY:
                raise ValueError("Production requires an explicit LABVIZ_CLIENT_IDENTITY_KEY.")
            if self.client_identity_key in {secret for _version, secret in self.share_token_keys}:
                raise ValueError("Client identity and share-token keys must be different.")
        if not 60 <= self.auth_rate_limit_window_seconds <= MAX_AUTH_RATE_LIMIT_WINDOW_SECONDS:
            raise ValueError("LABVIZ_AUTH_RATE_LIMIT_WINDOW_SECONDS must be between 60 and 86400.")
        if not 1 <= self.auth_client_request_limit <= MAX_AUTH_REQUEST_LIMIT:
            raise ValueError("LABVIZ_AUTH_CLIENT_REQUEST_LIMIT must be between 1 and 10000.")
        if not 1 <= self.auth_email_request_limit <= MAX_AUTH_REQUEST_LIMIT:
            raise ValueError("LABVIZ_AUTH_EMAIL_REQUEST_LIMIT must be between 1 and 10000.")
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
        if auth_mode not in {"console", "smtp", "ses"}:
            raise ValueError("LABVIZ_AUTH_MODE must be 'console', 'smtp', or 'ses'.")

        return cls(
            database_path=db_path,
            allowed_origins=origins,
            public_web_url=os.environ.get("LABVIZ_PUBLIC_WEB_URL", "http://localhost:3000").rstrip(
                "/"
            ),
            environment=os.environ.get("LABVIZ_ENVIRONMENT", "development").strip().lower(),
            runtime_role=os.environ.get("LABVIZ_RUNTIME_ROLE", "api").strip().lower(),
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
            ses_region=os.environ.get("LABVIZ_SES_REGION"),
            ses_from=os.environ.get("LABVIZ_SES_FROM"),
            ses_configuration_set=os.environ.get("LABVIZ_SES_CONFIGURATION_SET"),
            ses_connect_timeout_seconds=int(
                os.environ.get("LABVIZ_SES_CONNECT_TIMEOUT_SECONDS", "5")
            ),
            ses_read_timeout_seconds=int(os.environ.get("LABVIZ_SES_READ_TIMEOUT_SECONDS", "15")),
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
            trusted_proxy_cidrs=tuple(
                value.strip()
                for value in os.environ.get("LABVIZ_TRUSTED_PROXY_CIDRS", "").split(",")
                if value.strip()
            ),
            trusted_proxy_hops=int(os.environ.get("LABVIZ_TRUSTED_PROXY_HOPS", "0")),
            client_identity_key=os.environ.get(
                "LABVIZ_CLIENT_IDENTITY_KEY", DEFAULT_CLIENT_IDENTITY_KEY
            ),
            auth_rate_limit_window_seconds=int(
                os.environ.get("LABVIZ_AUTH_RATE_LIMIT_WINDOW_SECONDS", "3600")
            ),
            auth_client_request_limit=int(os.environ.get("LABVIZ_AUTH_CLIENT_REQUEST_LIMIT", "30")),
            auth_email_request_limit=int(os.environ.get("LABVIZ_AUTH_EMAIL_REQUEST_LIMIT", "10")),
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
