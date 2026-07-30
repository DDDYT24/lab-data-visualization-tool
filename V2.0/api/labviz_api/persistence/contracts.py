"""Repository and Unit of Work boundaries for the migrated project slice."""

from __future__ import annotations

from collections.abc import Iterable
from types import TracebackType
from typing import Any, Protocol

import pandas as pd

from labviz_api.db.models import ProcessingRun, Project, ProjectRevision, SourceFile, StoredObject


class ProjectReader(Protocol):
    def get_job_for_project(self, project_id: str) -> dict[str, Any] | None: ...

    def get_project(self, project_id: str, *, touch: bool = True) -> dict[str, Any] | None: ...


class ProjectStore(ProjectReader, Protocol):
    """Application-facing persistence contract shared by SQLite and PostgreSQL."""

    def ping(self) -> bool: ...

    def cleanup_expired(self) -> None: ...

    def recover_stale_jobs(self, stale_after_seconds: int = 900) -> int: ...

    def create_project(
        self,
        *,
        project_id: str,
        job_id: str,
        title: str,
        source: dict[str, Any],
        source_sha256: str,
        guest_token_digest: str,
    ) -> None: ...

    def update_job(
        self,
        job_id: str,
        *,
        stage: str,
        progress: float,
        message: str,
        error_code: str | None = None,
    ) -> None: ...

    def get_job(self, job_id: str) -> dict[str, Any] | None: ...

    def complete_project(
        self,
        *,
        project_id: str,
        source: dict[str, Any],
        frame: pd.DataFrame,
        preview: dict[str, Any],
        quality: dict[str, Any],
        chart: dict[str, Any],
    ) -> None: ...

    def save_chart(self, project_id: str, chart: dict[str, Any]) -> str: ...


class ProjectRepository(Protocol):
    """Database-only aggregate access used inside one Unit of Work."""

    def get_project(
        self,
        project_id: str,
        *,
        for_update: bool = False,
        include_deleted: bool = False,
    ) -> Project | None: ...

    def get_source_file(self, project_id: str) -> SourceFile | None: ...

    def get_run(self, job_id: str) -> ProcessingRun | None: ...

    def get_run_for_project(self, project_id: str) -> ProcessingRun | None: ...

    def get_revision(self, project_id: str, revision_number: int) -> ProjectRevision | None: ...

    def pending_objects(self) -> Iterable[StoredObject]: ...


class UnitOfWork(Protocol):
    projects: ProjectRepository

    def __enter__(self) -> UnitOfWork: ...

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None: ...

    def commit(self) -> None: ...

    def rollback(self) -> None: ...
