"""Build exactly one project persistence backend from application configuration."""

from __future__ import annotations

from labviz_api.config import Settings
from labviz_api.db.session import Database
from labviz_api.repository import ProjectRepository as SqliteReferenceRepository
from labviz_api.share_tokens import ShareTokenCodec
from labviz_api.storage.factory import build_object_storage

from .contracts import ProjectStore
from .postgres import PostgresProjectStore
from .sqlite import SqliteProjectStore


def build_project_store(
    settings: Settings,
    sqlite_repository: SqliteReferenceRepository,
) -> ProjectStore:
    share_tokens = ShareTokenCodec.from_strings(
        settings.share_token_keys,
        settings.share_token_key_version,
    )
    if settings.persistence_backend == "sqlite":
        return SqliteProjectStore(
            sqlite_repository,
            share_tokens,
            export_ttl_seconds=settings.export_ttl_seconds,
        )
    if settings.postgres_url is None:
        raise ValueError("PostgreSQL persistence requires LABVIZ_POSTGRES_URL.")
    database = Database(settings.postgres_url, echo=settings.postgres_echo)
    storage = build_object_storage(settings)
    return PostgresProjectStore(
        database,
        storage,
        settings.project_ttl_seconds,
        guest_session_ttl_seconds=settings.session_ttl_seconds,
        share_tokens=share_tokens,
    )
