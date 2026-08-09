from __future__ import annotations

from typing import cast

from labviz_api.db.session import Database, DatabaseHealth
from labviz_api.runtime_health import dependency_error
from labviz_api.storage import ObjectStorage


class FakeDatabase:
    def __init__(self, ready: bool) -> None:
        self.ready = ready

    def health(self) -> DatabaseHealth:
        return DatabaseHealth(ready=self.ready, detail="ok" if self.ready else "unavailable")


class FakeStorage:
    def __init__(self, ready: bool) -> None:
        self.ready = ready

    def ping(self) -> bool:
        return self.ready


def test_worker_dependency_probe_reports_each_failed_boundary() -> None:
    ready_database = cast(Database, FakeDatabase(True))
    ready_storage = cast(ObjectStorage, FakeStorage(True))

    assert dependency_error(ready_database, ready_storage) is None
    assert (
        dependency_error(cast(Database, FakeDatabase(False)), ready_storage)
        == "database-unavailable"
    )
    assert (
        dependency_error(ready_database, cast(ObjectStorage, FakeStorage(False)))
        == "object-storage-unavailable"
    )
