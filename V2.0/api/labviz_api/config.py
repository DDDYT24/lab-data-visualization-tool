"""Environment-backed configuration for the LabViz API."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _as_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


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
    persistence_backend: str = "sqlite"

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
            persistence_backend=os.environ.get("LABVIZ_PERSISTENCE_BACKEND", "sqlite")
            .strip()
            .lower(),
        )
