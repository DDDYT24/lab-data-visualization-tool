"""Adapter preserving the existing SQLite reference repository unchanged."""

from __future__ import annotations

import json
import sqlite3
from typing import Any
from uuid import UUID, uuid4

import pandas as pd

from labviz_api.persistence.exceptions import (
    IdempotencyConflict,
    PersistenceConflict,
    PersistenceNotFound,
    ProjectRevisionConflict,
)
from labviz_api.persistence_types import ProjectCreation
from labviz_api.processing import (
    apply_chart_decisions,
    deserialize_dataframe,
    serialize_dataframe,
)
from labviz_api.repository import ProjectRepository as SqliteReferenceRepository
from labviz_api.repository import expires_in, iso_now
from labviz_api.share_tokens import ShareTokenCodec


class SqliteProjectStore:
    def __init__(
        self,
        repository: SqliteReferenceRepository,
        share_tokens: ShareTokenCodec | None = None,
        export_ttl_seconds: int = 7_200,
    ) -> None:
        self.repository = repository
        self.share_tokens = share_tokens or ShareTokenCodec.from_strings(
            ((1, "labviz-development-share-token-key-v1"),), 1
        )
        self.export_ttl_seconds = export_ttl_seconds

    def ping(self) -> bool:
        return self.repository.ping()

    def readiness_error(self) -> str | None:
        return None if self.ping() else "database-unavailable"

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
        idempotency_key: str | None = None,
        request_sha256: str | None = None,
    ) -> ProjectCreation:
        try:
            return self.repository.create_project(
                project_id=project_id,
                job_id=job_id,
                title=title,
                source=source,
                guest_token_digest=guest_token_digest,
                idempotency_key=idempotency_key,
                request_sha256=request_sha256,
            )
        except ValueError as exc:
            message = str(exc)
            if message.startswith("idempotency-key-reused:"):
                raise IdempotencyConflict(message.split(":", 1)[1].strip()) from exc
            raise PersistenceConflict(message) from exc

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

    def update_project_description(
        self,
        *,
        project_id: str,
        owner_user_id: str,
        description: str,
        expected_revision_id: str,
    ) -> dict[str, Any]:
        expected_id = UUID(expected_revision_id).hex
        with self.repository._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            project = connection.execute(
                """
                SELECT id, description, current_revision_id, updated_at
                FROM projects
                WHERE id = ? AND storage_mode = 'saved-cloud' AND owner_user_id = ?
                """,
                (project_id, owner_user_id),
            ).fetchone()
            if project is None or project["current_revision_id"] is None:
                raise PersistenceNotFound("Active saved project does not exist.")
            if project["current_revision_id"] != expected_id:
                raise ProjectRevisionConflict("The project description changed in another session.")
            current = connection.execute(
                """
                SELECT revision_number FROM project_description_revisions
                WHERE id = ? AND project_id = ?
                """,
                (expected_id, project_id),
            ).fetchone()
            if current is None:
                raise ProjectRevisionConflict(
                    "The project changed before its description could be saved."
                )
            if description == project["description"]:
                return {
                    "projectId": project_id,
                    "description": description,
                    "revisionId": UUID(expected_id),
                    "revisionNumber": int(current["revision_number"]),
                    "updatedAt": project["updated_at"],
                }
            revision_id = uuid4()
            revision_number = int(current["revision_number"]) + 1
            updated_at = iso_now()
            connection.execute(
                """
                INSERT INTO project_description_revisions (
                    id, project_id, revision_number, description, created_at
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (revision_id.hex, project_id, revision_number, description, updated_at),
            )
            updated = connection.execute(
                """
                UPDATE projects
                SET description = ?, current_revision_id = ?, updated_at = ?
                WHERE id = ? AND current_revision_id = ?
                """,
                (description, revision_id.hex, updated_at, project_id, expected_id),
            )
            if updated.rowcount != 1:
                raise ProjectRevisionConflict("The project description changed in another session.")
        return {
            "projectId": project_id,
            "description": description,
            "revisionId": revision_id,
            "revisionNumber": revision_number,
            "updatedAt": updated_at,
        }

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

    def create_share(
        self,
        *,
        project_id: str,
        owner_user_id: str,
        downloads_enabled: bool,
    ) -> dict[str, Any]:
        token = self.share_tokens.issue(uuid4())
        created_at = self.repository.create_share(
            token=token,
            project_id=project_id,
            owner_user_id=owner_user_id,
            downloads_enabled=downloads_enabled,
        )
        return {
            "token": token,
            "downloads_enabled": downloads_enabled,
            "created_at": created_at,
        }

    def update_share(
        self,
        *,
        token: str,
        project_id: str,
        owner_user_id: str,
        downloads_enabled: bool,
    ) -> dict[str, Any] | None:
        project = self.repository.get_project(project_id, touch=False)
        share = self.repository.get_share(token)
        if (
            project is None
            or project.get("owner_user_id") != owner_user_id
            or share is None
            or share["project_id"] != project_id
        ):
            return None
        self.repository.update_share(token, project_id, downloads_enabled)
        return {
            "token": token,
            "downloads_enabled": downloads_enabled,
            "created_at": share["created_at"],
        }

    def revoke_share(
        self,
        *,
        token: str,
        project_id: str,
        owner_user_id: str,
    ) -> bool:
        project = self.repository.get_project(project_id, touch=False)
        if project is None or project.get("owner_user_id") != owner_user_id:
            return False
        share = self.repository.get_share(token)
        if share is None:
            return True
        if share["project_id"] != project_id:
            return False
        return self.repository.revoke_share(token, project_id)

    def get_shared_project(self, token: str) -> tuple[dict[str, Any], pd.DataFrame] | None:
        share = self.repository.get_share(token)
        if share is None:
            return None
        project = self.repository.get_project(share["project_id"], touch=False)
        if project is None:
            return None
        chart = json.loads(project["chart_json"])
        frame = apply_chart_decisions(
            deserialize_dataframe(bytes(project["data_blob"])),
            json.loads(project["quality_json"]),
            self.repository.get_decisions(project["id"]),
        )
        formats = self.repository.latest_exports(project["id"])
        return (
            {
                "token": token,
                "project_id": project["id"],
                "title": project["title"],
                "description": str(share["description_snapshot"]),
                "updated_at": project["updated_at"],
                "chart": chart,
                "preview": json.loads(project["preview_json"]),
                "downloads_enabled": bool(share["downloads_enabled"]),
                "download_formats": sorted(formats),
            },
            frame,
        )

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
    ) -> dict[str, Any]:
        del expected_revision_id, owner_user_id, guest_token_digest, idempotency_key
        export_id = uuid4().hex
        format_name = str(chart["export"]["format"])
        expires_at = expires_in(self.export_ttl_seconds)
        self.repository.save_chart(project_id, chart)
        self.repository.save_export(
            export_id=export_id,
            project_id=project_id,
            format_name=format_name,
            payload=payload,
            expires_at=expires_at,
            message="Your publication-ready figure is ready to download.",
        )
        return {
            "id": export_id,
            "project_id": project_id,
            "status": "ready",
            "format": format_name,
            "expires_at": expires_at,
            "message": "Your publication-ready figure is ready to download.",
        }

    def get_export(self, export_id: str) -> dict[str, Any] | None:
        return self.repository.get_export(export_id)

    def get_export_metadata(self, export_id: str) -> dict[str, Any] | None:
        self.repository.cleanup_expired()
        with sqlite3.connect(self.repository.database_path, timeout=30) as connection:
            connection.row_factory = sqlite3.Row
            row = connection.execute(
                "SELECT id, project_id, format FROM exports WHERE id = ?",
                (export_id,),
            ).fetchone()
        if row is None:
            return None
        return {
            "id": row["id"],
            "project_id": row["project_id"],
            "format": row["format"],
        }

    def get_shared_export(self, token: str, format_name: str) -> dict[str, Any] | None:
        share = self.repository.get_share(token)
        if share is None:
            return None
        if not bool(share["downloads_enabled"]):
            return {"downloads_enabled": False, "export": None}
        export_id = self.repository.latest_exports(share["project_id"]).get(format_name)
        return {
            "downloads_enabled": True,
            "export": self.repository.get_export(export_id) if export_id else None,
        }
