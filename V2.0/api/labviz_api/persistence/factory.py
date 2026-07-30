"""Build exactly one project persistence backend from application configuration."""

from __future__ import annotations

from labviz_api.config import Settings
from labviz_api.db.session import Database
from labviz_api.repository import ProjectRepository as SqliteReferenceRepository
from labviz_api.storage import LocalObjectStorage

from .contracts import ProjectStore
from .postgres import PostgresProjectStore
from .sqlite import SqliteProjectStore


def build_project_store(
    settings: Settings,
    sqlite_repository: SqliteReferenceRepository,
) -> ProjectStore:
    if settings.persistence_backend == "sqlite":
        return SqliteProjectStore(sqlite_repository)
    if settings.postgres_url is None:
        raise ValueError("PostgreSQL persistence requires LABVIZ_POSTGRES_URL.")
    database = Database(settings.postgres_url, echo=settings.postgres_echo)
    storage = LocalObjectStorage(settings.object_storage_root)
    return PostgresProjectStore(database, storage, settings.project_ttl_seconds)
