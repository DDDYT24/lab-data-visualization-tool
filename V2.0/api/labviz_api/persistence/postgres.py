"""PostgreSQL repository, Unit of Work, and object-compensated project store."""

from __future__ import annotations

import io
import json
from datetime import UTC, datetime, timedelta
from types import TracebackType
from typing import Any
from uuid import UUID, uuid4

import pandas as pd
from sqlalchemy import func, or_, select
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.orm import Session

from labviz_api.db.models import (
    ChartSpecRevision,
    Dataset,
    DatasetVersion,
    ProcessingRun,
    Project,
    ProjectRevision,
    SourceFile,
    StoredObject,
)
from labviz_api.db.session import Database
from labviz_api.models import ChartSpec
from labviz_api.parquet import PARQUET_SCHEMA_VERSION, read_parquet, write_parquet
from labviz_api.project_spec import ProjectSourceSpec, ProjectSpecV1
from labviz_api.repository import iso_at
from labviz_api.storage import ObjectStorage, StagedObject

from .contracts import ProjectRepository
from .exceptions import (
    ObjectConfirmationPending,
    PersistenceConflict,
    PersistenceError,
    PersistenceNotFound,
    PersistenceUnavailable,
)


def _uuid(value: str | UUID) -> UUID:
    return value if isinstance(value, UUID) else UUID(value)


def _id(value: UUID) -> str:
    return value.hex


def _now() -> datetime:
    return datetime.now(UTC)


def _api_job(
    *,
    stage: str,
    progress: float,
    message: str,
    error_code: str | None = None,
) -> dict[str, Any]:
    return {
        "stage": stage,
        "progress": progress,
        "message": message,
        "errorCode": error_code,
        "updatedAt": iso_at(_now()),
    }


def _translate_database_error(exc: Exception) -> PersistenceError:
    if isinstance(exc, PersistenceError):
        return exc
    if isinstance(exc, IntegrityError):
        return PersistenceConflict("The persistence operation conflicts with existing data.")
    if isinstance(exc, SQLAlchemyError):
        return PersistenceUnavailable("PostgreSQL could not complete the operation.")
    return PersistenceUnavailable("The persistence operation could not be completed.")


class SqlAlchemyProjectRepository(ProjectRepository):
    """Database-only aggregate queries scoped to a caller-owned Session."""

    def __init__(self, session: Session) -> None:
        self.session = session

    def get_project(
        self,
        project_id: str,
        *,
        for_update: bool = False,
        include_deleted: bool = False,
    ) -> Project | None:
        statement = select(Project).where(Project.id == _uuid(project_id))
        if not include_deleted:
            statement = statement.where(
                Project.deleted_at.is_(None),
                or_(
                    Project.storage_mode != "temporary-cloud",
                    Project.expires_at > _now(),
                ),
            )
        if for_update:
            statement = statement.with_for_update()
        return self.session.scalar(statement)

    def get_source_file(self, project_id: str) -> SourceFile | None:
        return self.session.scalar(
            select(SourceFile)
            .where(SourceFile.project_id == _uuid(project_id))
            .order_by(SourceFile.created_at)
            .limit(1)
        )

    def get_run(self, job_id: str) -> ProcessingRun | None:
        return self.session.get(ProcessingRun, _uuid(job_id))

    def get_run_for_project(self, project_id: str) -> ProcessingRun | None:
        return self.session.scalar(
            select(ProcessingRun)
            .where(ProcessingRun.project_id == _uuid(project_id))
            .order_by(ProcessingRun.created_at.desc())
            .limit(1)
        )

    def get_revision(self, project_id: str, revision_number: int) -> ProjectRevision | None:
        return self.session.scalar(
            select(ProjectRevision).where(
                ProjectRevision.project_id == _uuid(project_id),
                ProjectRevision.revision_number == revision_number,
            )
        )

    def pending_objects(self) -> list[StoredObject]:
        return list(
            self.session.scalars(
                select(StoredObject).where(
                    StoredObject.status == "pending",
                    StoredObject.staging_key.is_not(None),
                )
            )
        )

    def current_dataset_version(self, project: Project) -> DatasetVersion | None:
        if project.current_revision_id is None:
            return None
        revision = self.session.get(ProjectRevision, project.current_revision_id)
        if revision is None:
            return None
        return self.session.get(DatasetVersion, revision.active_dataset_version_id)

    def current_chart_revision(self, project: Project) -> ChartSpecRevision | None:
        if project.current_revision_id is None:
            return None
        revision = self.session.get(ProjectRevision, project.current_revision_id)
        if revision is None:
            return None
        return self.session.get(ChartSpecRevision, revision.chart_spec_revision_id)


class SqlAlchemyUnitOfWork:
    """One PostgreSQL transaction; object storage is deliberately outside this boundary."""

    def __init__(self, database: Database) -> None:
        self.database = database
        self.session: Session | None = None
        self.projects: SqlAlchemyProjectRepository
        self._committed = False

    def __enter__(self) -> SqlAlchemyUnitOfWork:
        self.session = self.database.session_factory()
        self.projects = SqlAlchemyProjectRepository(self.session)
        self._committed = False
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if self.session is None:
            return
        if exc_type is not None or not self._committed:
            self.session.rollback()
        self.session.close()
        self.session = None

    def commit(self) -> None:
        if self.session is None:
            raise RuntimeError("Unit of Work is not active.")
        self.session.commit()
        self._committed = True

    def rollback(self) -> None:
        if self.session is not None:
            self.session.rollback()


class PostgresProjectStore:
    """Migrated project slice with explicit storage compensation and recovery."""

    def __init__(
        self,
        database: Database,
        storage: ObjectStorage,
        project_ttl_seconds: int,
    ) -> None:
        self.database = database
        self.storage = storage
        self.project_ttl_seconds = project_ttl_seconds

    def _uow(self) -> SqlAlchemyUnitOfWork:
        return SqlAlchemyUnitOfWork(self.database)

    def dispose(self) -> None:
        self.database.dispose()

    def ping(self) -> bool:
        return self.database.health().ready

    def cleanup_expired(self) -> None:
        # Query boundaries already hide expired/deleted projects. Physical deletion and
        # object cleanup remain a Phase 3 worker responsibility.
        return None

    def recover_stale_jobs(self, stale_after_seconds: int = 900) -> int:
        recovered = self.recover_pending_objects()
        cutoff = _now() - timedelta(seconds=stale_after_seconds)
        try:
            with self._uow() as uow:
                assert uow.session is not None
                runs = list(
                    uow.session.scalars(
                        select(ProcessingRun).where(ProcessingRun.status.in_(("queued", "running")))
                    )
                )
                changed = 0
                for run in runs:
                    job = dict(run.parameters.get("apiJob", {}))
                    updated = str(job.get("updatedAt", ""))
                    try:
                        updated_at = datetime.fromisoformat(updated.replace("Z", "+00:00"))
                    except ValueError:
                        updated_at = run.created_at
                    if updated_at < cutoff:
                        run.status = "failed"
                        run.error_code = "processing-interrupted"
                        run.error_message = "Processing was interrupted. Upload the file again."
                        run.parameters = {
                            **run.parameters,
                            "apiJob": _api_job(
                                stage="failed",
                                progress=100,
                                message=run.error_message,
                                error_code=run.error_code,
                            ),
                        }
                        changed += 1
                uow.commit()
            return recovered + changed
        except Exception as exc:
            raise _translate_database_error(exc) from exc

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
        now = _now()
        project = Project(
            id=_uuid(project_id),
            storage_mode="temporary-cloud",
            title=title,
            description="",
            guest_token_digest=guest_token_digest,
            expires_at=now + timedelta(seconds=self.project_ttl_seconds),
            last_activity_at=now,
            created_at=now,
            updated_at=now,
        )
        source_file = SourceFile(
            id=uuid4(),
            project=project,
            original_name=str(source["name"]),
            media_type=str(source["mediaType"]),
            size_bytes=int(source["size"]),
            sha256=source_sha256,
            sheet_name=source.get("sheetName"),
            available_sheets=list(source.get("availableSheets", [])),
            header_row=source.get("headerRow"),
            parser_name="pending",
            parser_version="pending",
            created_at=now,
        )
        run = ProcessingRun(
            id=_uuid(job_id),
            project=project,
            operation="parse",
            status="queued",
            parameters={
                "apiJob": _api_job(
                    stage="queued",
                    progress=0,
                    message="Waiting to process the uploaded file.",
                )
            },
            algorithm_version="pending",
            code_version="v2-phase2",
            created_at=now,
        )
        try:
            with self._uow() as uow:
                assert uow.session is not None
                uow.session.add_all([project, source_file, run])
                uow.commit()
        except Exception as exc:
            raise _translate_database_error(exc) from exc

    def update_job(
        self,
        job_id: str,
        *,
        stage: str,
        progress: float,
        message: str,
        error_code: str | None = None,
    ) -> None:
        try:
            with self._uow() as uow:
                run = uow.projects.get_run(job_id)
                if run is None:
                    raise PersistenceNotFound("Processing job does not exist.")
                if stage == "ready":
                    run.status = "succeeded"
                elif stage == "failed":
                    run.status = "failed"
                elif stage == "queued":
                    run.status = "queued"
                else:
                    run.status = "running"
                run.error_code = error_code
                run.error_message = message if error_code else None
                transition_time = _now()
                run.parameters = {
                    **run.parameters,
                    "apiJob": _api_job(
                        stage=stage,
                        progress=progress,
                        message=message,
                        error_code=error_code,
                    ),
                }
                if stage != "queued" and run.started_at is None:
                    run.started_at = transition_time
                if stage in {"ready", "failed"}:
                    run.finished_at = transition_time
                uow.commit()
        except Exception as exc:
            raise _translate_database_error(exc) from exc

    @staticmethod
    def _job_row(run: ProcessingRun) -> dict[str, Any]:
        job = dict(run.parameters.get("apiJob", {}))
        return {
            "id": _id(run.id),
            "project_id": _id(run.project_id),
            "stage": job.get("stage", "queued"),
            "progress": float(job.get("progress", 0)),
            "message": str(job.get("message", "")),
            "error_code": job.get("errorCode"),
        }

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        try:
            with self._uow() as uow:
                run = uow.projects.get_run(job_id)
                return self._job_row(run) if run is not None else None
        except Exception as exc:
            raise _translate_database_error(exc) from exc

    def get_job_for_project(self, project_id: str) -> dict[str, Any] | None:
        try:
            with self._uow() as uow:
                run = uow.projects.get_run_for_project(project_id)
                return self._job_row(run) if run is not None else None
        except Exception as exc:
            raise _translate_database_error(exc) from exc

    def _project_row(
        self,
        repository: SqlAlchemyProjectRepository,
        project: Project,
    ) -> dict[str, Any]:
        source = repository.get_source_file(_id(project.id))
        if source is None:
            raise PersistenceNotFound("Project source metadata does not exist.")
        version = repository.current_dataset_version(project)
        chart_revision = repository.current_chart_revision(project)
        source_document = {
            "name": source.original_name,
            "size": source.size_bytes,
            "mediaType": source.media_type,
            "sheetName": source.sheet_name,
            "availableSheets": list(source.available_sheets),
            "headerRow": source.header_row,
        }
        ready = bool(
            version is not None
            and version.stored_object is not None
            and version.stored_object.status == "available"
            and chart_revision is not None
        )
        return {
            "id": _id(project.id),
            "title": project.title,
            "source_json": json.dumps(source_document, ensure_ascii=False),
            "storage_mode": project.storage_mode,
            "owner_user_id": _id(project.owner_user_id) if project.owner_user_id else None,
            "guest_token_digest": project.guest_token_digest,
            "expires_at": iso_at(project.expires_at) if project.expires_at else None,
            "updated_at": iso_at(project.updated_at),
            "chart_json": (
                json.dumps(chart_revision.spec_document, ensure_ascii=False)
                if chart_revision is not None
                else None
            ),
            "preview_json": (
                json.dumps(version.preview_document, ensure_ascii=False)
                if version is not None
                else None
            ),
            "quality_json": (
                json.dumps(version.quality_document, ensure_ascii=False)
                if version is not None
                else None
            ),
            "data_blob": None,
            "ready": ready,
        }

    def get_project(self, project_id: str, *, touch: bool = True) -> dict[str, Any] | None:
        try:
            with self._uow() as uow:
                project = uow.projects.get_project(project_id, for_update=touch)
                if project is None:
                    return None
                if touch:
                    now = _now()
                    project.last_activity_at = now
                    project.updated_at = now
                    if project.storage_mode == "temporary-cloud":
                        project.expires_at = now + timedelta(seconds=self.project_ttl_seconds)
                    row = self._project_row(uow.projects, project)
                    row["expires_at"] = iso_at(project.expires_at) if project.expires_at else None
                    row["updated_at"] = iso_at(project.updated_at)
                    uow.commit()
                    return row
                return self._project_row(uow.projects, project)
        except Exception as exc:
            raise _translate_database_error(exc) from exc

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
        existing = self.get_project(project_id, touch=False)
        if existing is None:
            raise PersistenceNotFound("Project does not exist.")
        if existing["ready"]:
            return
        if existing["preview_json"] is not None:
            self.recover_pending_objects()
            recovered = self.get_project(project_id, touch=False)
            if recovered is not None and recovered["ready"]:
                return
            raise ObjectConfirmationPending(
                "The existing DatasetVersion still requires object confirmation."
            )
        units = {str(item["field"]): item.get("unit") for item in preview["columns"]}
        artifact = write_parquet(frame, units=units)
        version_id = uuid4()
        final_key = f"datasets/{project_id}/{version_id.hex}-{artifact.sha256[:16]}.parquet"
        staged = self.storage.stage(
            final_key,
            io.BytesIO(artifact.payload),
            expected_sha256=artifact.sha256,
        )
        try:
            with self.storage.open_staged(staged) as staged_stream:
                read_parquet(staged_stream.read(), expected_sha256=artifact.sha256)
            persisted = self._persist_processed_project(
                project_id=project_id,
                source=source,
                version_id=version_id,
                staged=staged,
                artifact_schema=artifact.schema_document,
                row_count=artifact.row_count,
                column_count=artifact.column_count,
                preview=preview,
                quality=quality,
                chart=chart,
            )
        except Exception as exc:
            self.storage.discard(staged)
            raise _translate_database_error(exc) from exc
        if not persisted:
            self.storage.discard(staged)
            return
        try:
            self.storage.confirm(staged)
            self._mark_object_confirmed(staged)
        except Exception as exc:
            raise ObjectConfirmationPending(
                "Dataset bytes are durable but require confirmation recovery."
            ) from exc

    def _persist_processed_project(
        self,
        *,
        project_id: str,
        source: dict[str, Any],
        version_id: UUID,
        staged: StagedObject,
        artifact_schema: dict[str, Any],
        row_count: int,
        column_count: int,
        preview: dict[str, Any],
        quality: dict[str, Any],
        chart: dict[str, Any],
    ) -> bool:
        now = _now()
        with self._uow() as uow:
            assert uow.session is not None
            project = uow.projects.get_project(project_id, for_update=True)
            if project is None:
                raise PersistenceNotFound("Project does not exist.")
            if project.current_revision_id is not None:
                return False
            source_file = uow.projects.get_source_file(project_id)
            run = uow.projects.get_run_for_project(project_id)
            if source_file is None or run is None:
                raise PersistenceNotFound("Pending project metadata is incomplete.")

            stored_object = StoredObject(
                id=uuid4(),
                storage_backend="local",
                object_key=staged.key,
                staging_key=staged.staging_key,
                purpose="dataset",
                status="pending",
                media_type="application/vnd.apache.parquet",
                size_bytes=staged.size_bytes,
                sha256=staged.sha256,
                created_at=now,
                updated_at=now,
            )
            dataset = Dataset(
                id=uuid4(),
                project=project,
                source_file=source_file,
                name=str(source.get("sheetName") or source["name"]),
                sheet_name=source.get("sheetName"),
                header_row=source.get("headerRow"),
                created_at=now,
            )
            version = DatasetVersion(
                id=version_id,
                project=project,
                dataset=dataset,
                stored_object=stored_object,
                version_number=1,
                kind="parsed",
                schema_document=artifact_schema,
                preview_document=preview,
                quality_document=quality,
                parquet_schema_version=PARQUET_SCHEMA_VERSION,
                content_sha256=staged.sha256,
                row_count=row_count,
                column_count=column_count,
                created_at=now,
            )
            chart_model = ChartSpec.model_validate(chart)
            chart_revision = ChartSpecRevision(
                id=uuid4(),
                project=project,
                dataset_version=version,
                revision_number=1,
                schema_version=chart_model.schema_version,
                spec_document=chart_model.model_dump(mode="json", by_alias=True),
                created_at=now,
            )
            project_spec = ProjectSpecV1(
                project_id=project.id,
                title=project.title,
                description=project.description,
                source=ProjectSourceSpec(
                    source_file_id=source_file.id,
                    dataset_id=dataset.id,
                    dataset_version_id=version.id,
                    sheet_name=source.get("sheetName"),
                    header_row=source.get("headerRow"),
                ),
                cleaning=None,
                chart=chart_model,
            )
            project_revision = ProjectRevision(
                id=uuid4(),
                project=project,
                active_dataset_version=version,
                chart_spec_revision=chart_revision,
                revision_number=1,
                spec_schema_version=1,
                spec_document=project_spec.model_dump(mode="json", by_alias=True),
                created_at=now,
            )
            source_file.sheet_name = source.get("sheetName")
            source_file.available_sheets = list(source.get("availableSheets", []))
            source_file.header_row = source.get("headerRow")
            source_file.parser_name = "pandas"
            source_file.parser_version = pd.__version__
            source_file.binary_deleted_at = now
            source_file.parsed_at = now
            run.status = "running"
            run.started_at = run.started_at or now
            run.algorithm_version = "parquet-v1"
            run.parameters = {
                **run.parameters,
                "pendingDatasetVersionId": version.id.hex,
                "apiJob": _api_job(
                    stage="profiling",
                    progress=95,
                    message="The dataset is durable and awaiting final object confirmation.",
                ),
            }
            project.current_revision = project_revision
            project.last_activity_at = now
            project.updated_at = now
            if project.storage_mode == "temporary-cloud":
                project.expires_at = now + timedelta(seconds=self.project_ttl_seconds)
            uow.session.add_all([stored_object, dataset, version, chart_revision, project_revision])
            uow.commit()
        return True

    def _mark_object_confirmed(self, staged: StagedObject) -> None:
        now = _now()
        with self._uow() as uow:
            assert uow.session is not None
            stored = uow.session.scalar(
                select(StoredObject)
                .where(
                    StoredObject.object_key == staged.key,
                    StoredObject.sha256 == staged.sha256,
                    StoredObject.status == "pending",
                )
                .with_for_update()
            )
            if stored is None:
                return
            version = uow.session.scalar(
                select(DatasetVersion).where(DatasetVersion.stored_object_id == stored.id)
            )
            if version is None:
                raise PersistenceNotFound("Pending object has no DatasetVersion.")
            run = uow.projects.get_run_for_project(_id(version.project_id))
            if run is None:
                raise PersistenceNotFound("Pending object has no ProcessingRun.")
            stored.status = "available"
            stored.staging_key = None
            stored.updated_at = now
            run.output_dataset_version = version
            run.status = "succeeded"
            run.finished_at = now
            run.error_code = None
            run.error_message = None
            run.parameters = {
                **run.parameters,
                "apiJob": _api_job(
                    stage="ready",
                    progress=100,
                    message="Your data is ready to inspect.",
                ),
            }
            uow.commit()

    def recover_pending_objects(self) -> int:
        recovered = 0
        try:
            with self._uow() as uow:
                pending = [
                    StagedObject(
                        key=item.object_key,
                        staging_key=str(item.staging_key),
                        size_bytes=item.size_bytes,
                        sha256=item.sha256,
                    )
                    for item in uow.projects.pending_objects()
                ]
        except Exception as exc:
            raise _translate_database_error(exc) from exc
        for staged in pending:
            try:
                self.storage.confirm(staged)
                self._mark_object_confirmed(staged)
                recovered += 1
            except Exception:
                continue
        return recovered

    def save_chart(self, project_id: str, chart: dict[str, Any]) -> str:
        now = _now()
        chart_model = ChartSpec.model_validate(chart)
        chart_document = chart_model.model_dump(mode="json", by_alias=True)
        try:
            with self._uow() as uow:
                assert uow.session is not None
                project = uow.projects.get_project(project_id, for_update=True)
                if project is None or project.current_revision_id is None:
                    raise PersistenceNotFound("Ready project does not exist.")
                current = uow.session.get(ProjectRevision, project.current_revision_id)
                if current is None:
                    raise PersistenceNotFound("Current ProjectRevision does not exist.")
                current_chart = uow.session.get(ChartSpecRevision, current.chart_spec_revision_id)
                if current_chart is None:
                    raise PersistenceNotFound("Current ChartSpecRevision does not exist.")
                project.title = chart_model.title or "Untitled figure"
                project.last_activity_at = now
                project.updated_at = now
                if current_chart.spec_document == chart_document:
                    uow.commit()
                    return iso_at(now)
                chart_number = (
                    int(
                        uow.session.scalar(
                            select(func.max(ChartSpecRevision.revision_number)).where(
                                ChartSpecRevision.project_id == project.id
                            )
                        )
                        or 0
                    )
                    + 1
                )
                project_number = (
                    int(
                        uow.session.scalar(
                            select(func.max(ProjectRevision.revision_number)).where(
                                ProjectRevision.project_id == project.id
                            )
                        )
                        or 0
                    )
                    + 1
                )
                chart_revision = ChartSpecRevision(
                    id=uuid4(),
                    project=project,
                    dataset_version_id=current.active_dataset_version_id,
                    revision_number=chart_number,
                    schema_version=chart_model.schema_version,
                    spec_document=chart_document,
                    created_at=now,
                )
                previous_spec = ProjectSpecV1.model_validate(current.spec_document)
                next_spec = previous_spec.model_copy(
                    update={"title": project.title, "chart": chart_model}
                )
                next_revision = ProjectRevision(
                    id=uuid4(),
                    project=project,
                    active_dataset_version_id=current.active_dataset_version_id,
                    chart_spec_revision=chart_revision,
                    revision_number=project_number,
                    spec_schema_version=1,
                    spec_document=next_spec.model_dump(mode="json", by_alias=True),
                    created_at=now,
                )
                project.current_revision = next_revision
                uow.session.add_all([chart_revision, next_revision])
                uow.commit()
            return iso_at(now)
        except Exception as exc:
            raise _translate_database_error(exc) from exc

    def restore_project_revision(self, project_id: str, revision_number: int) -> str:
        now = _now()
        try:
            with self._uow() as uow:
                assert uow.session is not None
                project = uow.projects.get_project(project_id, for_update=True)
                revision = uow.projects.get_revision(project_id, revision_number)
                if project is None or revision is None:
                    raise PersistenceNotFound("ProjectRevision does not exist.")
                version = uow.session.get(DatasetVersion, revision.active_dataset_version_id)
                if (
                    version is None
                    or version.stored_object is None
                    or version.stored_object.status != "available"
                ):
                    raise PersistenceConflict("ProjectRevision data is not available.")
                spec = ProjectSpecV1.model_validate(revision.spec_document)
                project.current_revision = revision
                project.title = spec.title
                project.last_activity_at = now
                project.updated_at = now
                uow.commit()
            return iso_at(now)
        except Exception as exc:
            raise _translate_database_error(exc) from exc

    def restore_deleted_project(self, project_id: str) -> str:
        now = _now()
        try:
            with self._uow() as uow:
                project = uow.projects.get_project(
                    project_id, for_update=True, include_deleted=True
                )
                if project is None or project.deleted_at is None or project.purge_after is None:
                    raise PersistenceNotFound("Deleted project does not exist.")
                if project.storage_mode != "saved-cloud" or project.purge_after <= now:
                    raise PersistenceNotFound("The project recovery window has expired.")
                project.deleted_at = None
                project.purge_after = None
                project.updated_at = now
                project.last_activity_at = now
                uow.commit()
            return iso_at(now)
        except Exception as exc:
            raise _translate_database_error(exc) from exc

    def load_dataframe(self, project_id: str) -> pd.DataFrame:
        try:
            with self._uow() as uow:
                project = uow.projects.get_project(project_id)
                if project is None:
                    raise PersistenceNotFound("Project does not exist.")
                version = uow.projects.current_dataset_version(project)
                if (
                    version is None
                    or version.stored_object is None
                    or version.stored_object.status != "available"
                ):
                    raise PersistenceConflict("DatasetVersion is not available.")
                object_key = version.stored_object.object_key
                object_sha = version.stored_object.sha256
                content_sha = version.content_sha256
            with self.storage.open(object_key) as stream:
                payload = stream.read()
            frame = read_parquet(payload, expected_sha256=object_sha)
            if object_sha != content_sha:
                raise PersistenceConflict("DatasetVersion content hash metadata differs.")
            return frame
        except PersistenceError:
            raise
        except Exception as exc:
            raise PersistenceUnavailable("DatasetVersion could not be reopened.") from exc
