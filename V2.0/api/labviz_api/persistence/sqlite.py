"""Adapter preserving the existing SQLite reference repository unchanged."""

from __future__ import annotations

import json
from typing import Any

import pandas as pd

from labviz_api.persistence.exceptions import PersistenceNotFound
from labviz_api.processing import (
    apply_chart_decisions,
    deserialize_dataframe,
    serialize_dataframe,
)
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

    def load_quality_dataframe(self, project_id: str) -> pd.DataFrame:
        payload = self.repository.get_data_blob(project_id)
        if payload is None:
            raise ValueError("Ready project data does not exist.")
        return deserialize_dataframe(payload)

    def save_quality_report(
        self,
        project_id: str,
        quality: dict[str, Any],
        *,
        parameters: dict[str, Any],
    ) -> str:
        del parameters
        return self.repository.replace_quality(project_id, quality)

    def save_decisions(self, project_id: str, decisions: list[dict[str, str]]) -> str:
        return self.repository.save_decisions(project_id, decisions)

    def get_decisions(self, project_id: str) -> list[dict[str, str]]:
        return self.repository.get_decisions(project_id)

    def load_cleaned_dataframe(self, project_id: str) -> pd.DataFrame:
        frame = self.load_quality_dataframe(project_id)
        project = self.repository.get_project(project_id, touch=False)
        if project is None or project["quality_json"] is None:
            raise ValueError("Ready project quality report does not exist.")
        return apply_chart_decisions(
            frame,
            json.loads(project["quality_json"]),
            self.get_decisions(project_id),
            actions={"remove"},
        )

    def load_chart_dataframe(self, project_id: str) -> pd.DataFrame:
        frame = self.load_quality_dataframe(project_id)
        project = self.repository.get_project(project_id, touch=False)
        if project is None or project["quality_json"] is None:
            raise ValueError("Ready project quality report does not exist.")
        return apply_chart_decisions(
            frame,
            json.loads(project["quality_json"]),
            self.get_decisions(project_id),
        )

    def save_project(
        self,
        project_id: str,
        owner_user_id: str,
        *,
        guest_token_digest: str | None,
    ) -> str:
        del guest_token_digest
        return self.repository.save_project(project_id, owner_user_id)

    def duplicate_project(
        self,
        *,
        source_project_id: str,
        project_id: str,
        job_id: str,
        owner_user_id: str,
        guest_token_digest: str | None,
        idempotency_key: str | None = None,
    ) -> str:
        del guest_token_digest, idempotency_key
        self.repository.duplicate_project(
            source_project_id=source_project_id,
            project_id=project_id,
            job_id=job_id,
            owner_user_id=owner_user_id,
        )
        return project_id

    def delete_project(
        self,
        project_id: str,
        *,
        owner_user_id: str,
        guest_token_digest: str | None,
    ) -> bool:
        del owner_user_id, guest_token_digest
        return self.repository.delete_project(project_id)

    def restore_deleted_project(self, project_id: str, owner_user_id: str | None = None) -> str:
        del project_id, owner_user_id
        raise PersistenceNotFound("The SQLite reference repository has no recovery area.")

    def restore_project_revision(
        self,
        project_id: str,
        revision_number: int,
        owner_user_id: str | None = None,
    ) -> str:
        del project_id, revision_number, owner_user_id
        raise PersistenceNotFound("The SQLite reference repository has no immutable history.")

    def list_projects(self, owner_user_id: str | None) -> list[dict[str, Any]]:
        return self.repository.list_projects(owner_user_id)

    def list_deleted_projects(self, owner_user_id: str) -> list[dict[str, Any]]:
        del owner_user_id
        return []

    def get_workspace(self, project_id: str) -> dict[str, Any]:
        project = self.repository.get_project(project_id, touch=False)
        if project is None:
            raise PersistenceNotFound("Project does not exist.")
        return {
            "project": project,
            "preview": json.loads(project["preview_json"]),
            "quality": json.loads(project["quality_json"]),
            "decisions": self.repository.get_decisions(project_id),
            "chart": json.loads(project["chart_json"]),
            "shares": self.repository.list_shares(project_id),
        }
