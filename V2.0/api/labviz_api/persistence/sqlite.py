"""Adapter preserving the existing SQLite reference repository unchanged."""

from __future__ import annotations

from typing import Any

import pandas as pd

from labviz_api.processing import serialize_dataframe
from labviz_api.repository import ProjectRepository as SqliteReferenceRepository


class SqliteProjectStore:
    def __init__(self, repository: SqliteReferenceRepository) -> None:
        self.repository = repository

    def ping(self) -> bool:
        return self.repository.ping()

    def cleanup_expired(self) -> None:
        self.repository.cleanup_expired()

    def recover_stale_jobs(self, stale_after_seconds: int = 900) -> int:
        return self.repository.recover_stale_jobs(stale_after_seconds)

    def create_project(
        self,
        *,
        project_id: str,
        job_id: str,
        title: str,
        source: dict[str, Any],
        source_sha256: str,
        guest_token_digest: str,
    ) -> None:
        del source_sha256
        self.repository.create_project(
            project_id=project_id,
            job_id=job_id,
            title=title,
            source=source,
            guest_token_digest=guest_token_digest,
        )

    def update_job(
        self,
        job_id: str,
        *,
        stage: str,
        progress: float,
        message: str,
        error_code: str | None = None,
    ) -> None:
        self.repository.update_job(
            job_id,
            stage=stage,
            progress=progress,
            message=message,
            error_code=error_code,
        )

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        return self.repository.get_job(job_id)

    def get_job_for_project(self, project_id: str) -> dict[str, Any] | None:
        return self.repository.get_job_for_project(project_id)

    def complete_project(
        self,
        *,
        project_id: str,
        source: dict[str, Any],
        frame: pd.DataFrame,
        preview: dict[str, Any],
        quality: dict[str, Any],
        chart: dict[str, Any],
    ) -> None:
        self.repository.complete_project(
            project_id=project_id,
            source=source,
            data_blob=serialize_dataframe(frame),
            preview=preview,
            quality=quality,
            chart=chart,
        )

    def get_project(self, project_id: str, *, touch: bool = True) -> dict[str, Any] | None:
        project = self.repository.get_project(project_id, touch=touch)
        if project is not None:
            project["ready"] = bool(
                project.get("preview_json")
                and project.get("quality_json")
                and project.get("data_blob")
            )
        return project

    def save_chart(self, project_id: str, chart: dict[str, Any]) -> str:
        return self.repository.save_chart(project_id, chart)
