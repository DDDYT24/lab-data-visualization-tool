"""Container readiness probe for an independent LabViz Worker process."""

from __future__ import annotations

import json

from labviz_api.config import Settings
from labviz_api.db.session import Database
from labviz_api.storage.base import ObjectStorage
from labviz_api.storage.factory import build_object_storage


def dependency_error(database: Database, storage: ObjectStorage) -> str | None:
    if not database.health().ready:
        return "database-unavailable"
    if not storage.ping():
        return "object-storage-unavailable"
    return None


def main() -> int:
    database: Database | None = None
    error: str | None
    try:
        settings = Settings.from_env()
        if settings.persistence_backend != "postgresql" or settings.postgres_url is None:
            error = "postgresql-required"
        else:
            database = Database(settings.postgres_url, echo=settings.postgres_echo)
            storage = build_object_storage(settings)
            error = dependency_error(database, storage)
    except (OSError, ValueError):
        error = "dependency-configuration-unavailable"
    finally:
        if database is not None:
            database.dispose()
    print(json.dumps({"status": "ok" if error is None else "error", "code": error}))
    return 0 if error is None else 1


if __name__ == "__main__":
    raise SystemExit(main())
