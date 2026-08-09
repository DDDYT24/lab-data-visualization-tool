"""Repository and Unit of Work boundaries for the migrated project slice."""

from __future__ import annotations

from collections.abc import Iterable
from types import TracebackType
from typing import Any, Protocol

import pandas as pd

from labviz_api.db.models import (
    CleaningDecisionSet,
    ProcessingRun,
    Project,
    ProjectRevision,
    QualityReportRecord,
    SourceFile,
    StoredObject,
)
from labviz_api.persistence_types import ProjectCreation


class ProjectReader(Protocol):
    def get_job_for_project(self, project_id: str) -> dict[str, Any] | None: ...

    def get_project(self, project_id: str, *, touch: bool = True) -> dict[str, Any] | None: ...


class ProjectStore(ProjectReader, Protocol):
    """Application-facing persistence contract shared by SQLite and PostgreSQL."""

    def ping(self) -> bool: ...

    def readiness_error(self) -> str | None: ...

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
        idempotency_key: str | None = None,
        request_sha256: str | None = None,
    ) -> ProjectCreation: ...

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

    def load_quality_dataframe(self, project_id: str) -> pd.DataFrame: ...

    def save_quality_report(
        self,
        project_id: str,
        quality: dict[str, Any],
        *,
        parameters: dict[str, Any],
    ) -> str: ...

    def save_decisions(self, project_id: str, decisions: list[dict[str, str]]) -> str: ...

    def get_decisions(self, project_id: str) -> list[dict[str, str]]: ...

    def load_cleaned_dataframe(self, project_id: str) -> pd.DataFrame: ...

    def load_chart_dataframe(self, project_id: str) -> pd.DataFrame: ...

    def save_project(
        self,
        project_id: str,
        owner_user_id: str,
        *,
        guest_token_digest: str | None,
    ) -> str: ...

    def duplicate_project(
        self,
        *,
        source_project_id: str,
        project_id: str,
        job_id: str,
        owner_user_id: str,
        guest_token_digest: str | None,
        idempotency_key: str | None = None,
    ) -> str: ...

    def delete_project(
        self,
        project_id: str,
        *,
        owner_user_id: str,
        guest_token_digest: str | None,
    ) -> bool: ...

    def restore_deleted_project(self, project_id: str, owner_user_id: str | None = None) -> str: ...

    def restore_project_revision(
        self,
        project_id: str,
        revision_number: int,
        owner_user_id: str | None = None,
    ) -> str: ...

    def list_projects(self, owner_user_id: str | None) -> list[dict[str, Any]]: ...

    def list_deleted_projects(self, owner_user_id: str) -> list[dict[str, Any]]: ...

    def get_workspace(self, project_id: str) -> dict[str, Any]: ...

    def create_share(
        self,
        *,
        project_id: str,
        owner_user_id: str,
        downloads_enabled: bool,
    ) -> dict[str, Any]: ...

    def update_share(
        self,
        *,
        token: str,
        project_id: str,
        owner_user_id: str,
        downloads_enabled: bool,
    ) -> dict[str, Any] | None: ...

    def revoke_share(
        self,
        *,
        token: str,
        project_id: str,
        owner_user_id: str,
    ) -> bool: ...

    def get_shared_project(self, token: str) -> tuple[dict[str, Any], pd.DataFrame] | None: ...

    def create_publication_export(
        self,
        *,
        project_id: str,
        expected_revision_id: str | None,
        chart: dict[str, Any],
        payload: bytes,
        owner_user_id: str | None,
        guest_token_digest: str | None,
        idempotency_key: str | None = None,
    ) -> dict[str, Any]: ...

    def get_export_metadata(self, export_id: str) -> dict[str, Any] | None: ...

    def get_export(self, export_id: str) -> dict[str, Any] | None: ...

    def get_shared_export(self, token: str, format_name: str) -> dict[str, Any] | None: ...


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

    def current_quality_report(self, project: Project) -> QualityReportRecord | None: ...

    def current_decision_set(self, project: Project) -> CleaningDecisionSet | None: ...

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
