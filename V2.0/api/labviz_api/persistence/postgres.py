"""PostgreSQL repository, Unit of Work, and object-compensated project store."""

from __future__ import annotations

import hashlib
import io
import json
import secrets
from copy import deepcopy
from datetime import UTC, datetime, timedelta
from types import TracebackType
from typing import Any
from uuid import UUID, uuid4

import pandas as pd
from sqlalchemy import delete as sql_delete
from sqlalchemy import func, or_, select
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.orm import Session

from labviz_api.db.models import (
    AuthChallenge,
    AuthRequest,
    AuthSession,
    ChartSpecRevision,
    CleaningDecisionRecord,
    CleaningDecisionSet,
    Dataset,
    DatasetVersion,
    GuestSession,
    IdempotencyRecord,
    ProcessingRun,
    Project,
    ProjectClaim,
    ProjectLifecycleEvent,
    ProjectOrigin,
    ProjectRevision,
    QualityFindingRecord,
    QualityReportRecord,
    SourceFile,
    StoredObject,
    User,
)
from labviz_api.db.session import Database
from labviz_api.models import ChartSpec
from labviz_api.parquet import PARQUET_SCHEMA_VERSION, read_parquet, write_parquet
from labviz_api.processing import apply_chart_decisions, build_preview
from labviz_api.project_spec import (
    ProjectCleaningSpec,
    ProjectSourceSpec,
    ProjectSpecV1,
)
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

QUALITY_PROFILER_NAME = "labviz-quality"
QUALITY_PROFILER_VERSION = "1"
QUALITY_ALGORITHM_VERSION = "quality-v1"
CLEANING_ALGORITHM_VERSION = "cleaning-decisions-v1"
PHASE3_CODE_VERSION = "v2-phase3"
PHASE4_CODE_VERSION = "v2-phase4"
MAX_PERSISTED_FINDING_REFS = 100


def _uuid(value: str | UUID) -> UUID:
    return value if isinstance(value, UUID) else UUID(value)


def _id(value: UUID) -> str:
    return value.hex


def _now() -> datetime:
    return datetime.now(UTC)


def _parse_iso(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _scope_storage_key(dedup_scope: str, sha256: str) -> str:
    scope_hash = hashlib.sha256(dedup_scope.encode("utf-8")).hexdigest()[:32]
    return f"datasets/parquet-v1/{scope_hash}/{sha256}.parquet"


def _canonical_decisions(decisions: list[dict[str, str]]) -> list[dict[str, str]]:
    """Match the reference repository's last-write-wins behavior, then sort stably."""

    by_finding = {str(item["findingId"]): str(item["action"]) for item in decisions}
    return [
        {"findingId": finding_id, "action": by_finding[finding_id]}
        for finding_id in sorted(by_finding)
    ]


def _decisions_hash(decisions: list[dict[str, str]]) -> str:
    encoded = json.dumps(
        decisions, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _json_value(value: Any) -> Any:
    if pd.isna(value):
        return None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if hasattr(value, "item"):
        return _json_value(value.item())
    if isinstance(value, str | int | float | bool) or value is None:
        return value
    return str(value)


def _source_record_refs(
    frame: pd.DataFrame,
    dataset_version_id: UUID,
    row_ids: list[str | int],
) -> list[dict[str, Any]]:
    references: list[dict[str, Any]] = []
    columns = [str(column) for column in frame.columns]
    for raw_row_id in row_ids[:MAX_PERSISTED_FINDING_REFS]:
        try:
            ordinal = int(raw_row_id)
        except (TypeError, ValueError):
            continue
        if ordinal < 1 or ordinal > len(frame):
            continue
        values = [_json_value(value) for value in frame.iloc[ordinal - 1].tolist()]
        fingerprint_payload = json.dumps(
            {"columns": columns, "values": values},
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        references.append(
            {
                "datasetVersionId": dataset_version_id.hex,
                "rowOrdinal": ordinal,
                "rowFingerprint": hashlib.sha256(fingerprint_payload).hexdigest(),
            }
        )
    return references


def _quality_findings(
    *,
    project_id: UUID,
    report_id: UUID,
    dataset_version_id: UUID,
    frame: pd.DataFrame,
    schema_document: dict[str, Any],
    quality: dict[str, Any],
    created_at: datetime,
) -> list[QualityFindingRecord]:
    schema_columns = {
        str(item.get("name")): {**item, "ordinal": index}
        for index, item in enumerate(schema_document.get("columns", []))
    }
    records: list[QualityFindingRecord] = []
    for finding in quality.get("findings", []):
        column_name = finding.get("column")
        evidence = {
            key: finding[key]
            for key in ("rowIdsTruncated", "validMinimum", "validMaximum")
            if key in finding
        }
        records.append(
            QualityFindingRecord(
                id=uuid4(),
                project_id=project_id,
                quality_report_id=report_id,
                external_id=str(finding["id"]),
                kind=str(finding["kind"]),
                severity=str(finding["severity"]),
                column_name=str(column_name) if column_name is not None else None,
                column_identity=(
                    schema_columns.get(str(column_name), {"name": str(column_name)})
                    if column_name is not None
                    else None
                ),
                source_record_refs=_source_record_refs(
                    frame,
                    dataset_version_id,
                    list(finding.get("rowIds", [])),
                ),
                affected_count=int(finding.get("affectedCount", 0)),
                evidence_document=evidence,
                summary=str(finding["summary"]),
                reason=str(finding["reason"]),
                created_at=created_at,
            )
        )
    return records


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
            .where(
                ProcessingRun.project_id == _uuid(project_id),
                ProcessingRun.operation == "parse",
            )
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

    def current_revision(self, project: Project) -> ProjectRevision | None:
        if project.current_revision_id is None:
            return None
        return self.session.get(ProjectRevision, project.current_revision_id)

    def current_dataset_version(self, project: Project) -> DatasetVersion | None:
        revision = self.current_revision(project)
        if revision is None:
            return None
        return self.session.get(DatasetVersion, revision.active_dataset_version_id)

    def current_chart_revision(self, project: Project) -> ChartSpecRevision | None:
        revision = self.current_revision(project)
        if revision is None:
            return None
        return self.session.get(ChartSpecRevision, revision.chart_spec_revision_id)

    def current_quality_report(self, project: Project) -> QualityReportRecord | None:
        revision = self.current_revision(project)
        if revision is None or revision.quality_report_id is None:
            return None
        return self.session.get(QualityReportRecord, revision.quality_report_id)

    def current_decision_set(self, project: Project) -> CleaningDecisionSet | None:
        revision = self.current_revision(project)
        if revision is None or revision.cleaning_decision_set_id is None:
            return None
        return self.session.get(CleaningDecisionSet, revision.cleaning_decision_set_id)


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
        *,
        guest_session_ttl_seconds: int = 604_800,
    ) -> None:
        self.database = database
        self.storage = storage
        self.project_ttl_seconds = project_ttl_seconds
        self.guest_session_ttl_seconds = guest_session_ttl_seconds

    def _uow(self) -> SqlAlchemyUnitOfWork:
        return SqlAlchemyUnitOfWork(self.database)

    def dispose(self) -> None:
        self.database.dispose()

    def ping(self) -> bool:
        return self.database.health().ready

    def allow_auth_request(
        self,
        *,
        client_key: str,
        email: str,
        ip_limit: int = 30,
        email_limit: int = 10,
    ) -> bool:
        now = _now()
        cutoff = now - timedelta(hours=1)
        try:
            with self._uow() as uow:
                assert uow.session is not None
                ip_count = int(
                    uow.session.scalar(
                        select(func.count())
                        .select_from(AuthRequest)
                        .where(
                            AuthRequest.client_key == client_key,
                            AuthRequest.requested_at >= cutoff,
                        )
                    )
                    or 0
                )
                email_count = int(
                    uow.session.scalar(
                        select(func.count())
                        .select_from(AuthRequest)
                        .where(
                            AuthRequest.email == email,
                            AuthRequest.requested_at >= cutoff,
                        )
                    )
                    or 0
                )
                if ip_count >= ip_limit or email_count >= email_limit:
                    uow.commit()
                    return False
                uow.session.add(
                    AuthRequest(id=uuid4(), client_key=client_key, email=email, requested_at=now)
                )
                uow.commit()
                return True
        except Exception as exc:
            raise _translate_database_error(exc) from exc

    def latest_auth_challenge(self, email: str) -> dict[str, Any] | None:
        with self._uow() as uow:
            assert uow.session is not None
            challenge = uow.session.scalar(
                select(AuthChallenge)
                .where(AuthChallenge.email == email, AuthChallenge.expires_at > _now())
                .order_by(AuthChallenge.created_at.desc())
                .limit(1)
            )
            return self._auth_challenge_row(challenge) if challenge is not None else None

    @staticmethod
    def _auth_challenge_row(challenge: AuthChallenge) -> dict[str, Any]:
        return {
            "id": challenge.id.hex,
            "email": challenge.email,
            "salt": challenge.salt,
            "code_digest": challenge.code_digest,
            "expires_at": iso_at(challenge.expires_at),
            "resend_at": iso_at(challenge.resend_at),
            "failed_attempts": challenge.failed_attempts,
            "created_at": iso_at(challenge.created_at),
        }

    def create_auth_challenge(
        self,
        *,
        challenge_id: str,
        email: str,
        salt: str,
        code_digest: str,
        expires_at: str,
        resend_at: str,
    ) -> None:
        try:
            with self._uow() as uow:
                assert uow.session is not None
                uow.session.add(
                    AuthChallenge(
                        id=_uuid(challenge_id),
                        email=email,
                        salt=salt,
                        code_digest=code_digest,
                        expires_at=_parse_iso(expires_at),
                        resend_at=_parse_iso(resend_at),
                        failed_attempts=0,
                        created_at=_now(),
                    )
                )
                uow.commit()
        except Exception as exc:
            raise _translate_database_error(exc) from exc

    def get_auth_challenge(self, challenge_id: str) -> dict[str, Any] | None:
        with self._uow() as uow:
            assert uow.session is not None
            challenge = uow.session.get(AuthChallenge, _uuid(challenge_id))
            if challenge is None or challenge.expires_at <= _now():
                return None
            return self._auth_challenge_row(challenge)

    def increment_auth_challenge_attempts(self, challenge_id: str) -> None:
        try:
            with self._uow() as uow:
                assert uow.session is not None
                challenge = uow.session.scalar(
                    select(AuthChallenge)
                    .where(AuthChallenge.id == _uuid(challenge_id))
                    .with_for_update()
                )
                if challenge is not None:
                    challenge.failed_attempts += 1
                uow.commit()
        except Exception as exc:
            raise _translate_database_error(exc) from exc

    def delete_auth_challenge(self, challenge_id: str) -> None:
        try:
            with self._uow() as uow:
                assert uow.session is not None
                uow.session.execute(
                    sql_delete(AuthChallenge).where(AuthChallenge.id == _uuid(challenge_id))
                )
                uow.commit()
        except Exception as exc:
            raise _translate_database_error(exc) from exc

    def create_auth_session(
        self,
        *,
        token_digest: str,
        user_id: str,
        email: str,
        expires_at: str,
    ) -> None:
        try:
            with self._uow() as uow:
                assert uow.session is not None
                user = uow.session.scalar(select(User).where(User.email == email).with_for_update())
                if user is None:
                    user = User(
                        id=_uuid(user_id),
                        email=email,
                        created_at=_now(),
                        updated_at=_now(),
                    )
                    uow.session.add(user)
                uow.session.add(
                    AuthSession(
                        token_digest=token_digest,
                        user=user,
                        expires_at=_parse_iso(expires_at),
                        created_at=_now(),
                    )
                )
                uow.commit()
        except Exception as exc:
            raise _translate_database_error(exc) from exc

    def get_auth_session(self, token_digest: str) -> dict[str, Any] | None:
        with self._uow() as uow:
            assert uow.session is not None
            auth_session = uow.session.get(AuthSession, token_digest)
            if auth_session is None or auth_session.expires_at <= _now():
                return None
            return {
                "token_digest": auth_session.token_digest,
                "user_id": auth_session.user_id.hex,
                "email": auth_session.user.email,
                "expires_at": iso_at(auth_session.expires_at),
                "created_at": iso_at(auth_session.created_at),
            }

    def delete_auth_session(self, token_digest: str) -> None:
        try:
            with self._uow() as uow:
                assert uow.session is not None
                uow.session.execute(
                    sql_delete(AuthSession).where(AuthSession.token_digest == token_digest)
                )
                uow.commit()
        except Exception as exc:
            raise _translate_database_error(exc) from exc

    def cleanup_expired(self) -> None:
        now = _now()
        try:
            with self._uow() as uow:
                assert uow.session is not None
                project_ids = list(
                    uow.session.scalars(
                        select(Project.id).where(
                            or_(
                                (
                                    (Project.storage_mode == "temporary-cloud")
                                    & (Project.expires_at <= now)
                                ),
                                (Project.deleted_at.is_not(None) & (Project.purge_after <= now)),
                            )
                        )
                    )
                )
                uow.session.execute(
                    sql_delete(AuthChallenge).where(AuthChallenge.expires_at <= now)
                )
                uow.session.execute(sql_delete(AuthSession).where(AuthSession.expires_at <= now))
                uow.session.execute(
                    sql_delete(AuthRequest).where(
                        AuthRequest.requested_at < now - timedelta(hours=1)
                    )
                )
                uow.commit()
            for project_id in project_ids:
                self._purge_project(project_id)
            self._collect_garbage()
            with self._uow() as uow:
                assert uow.session is not None
                expired_guests = list(
                    uow.session.scalars(
                        select(GuestSession).where(GuestSession.expires_at <= now).with_for_update()
                    )
                )
                for guest in expired_guests:
                    reference_count = int(
                        uow.session.scalar(
                            select(func.count())
                            .select_from(Project)
                            .where(Project.guest_session_id == guest.id)
                        )
                        or 0
                    )
                    if reference_count == 0:
                        uow.session.delete(guest)
                uow.commit()
        except Exception as exc:
            raise _translate_database_error(exc) from exc

    @staticmethod
    def _object_reference_count(session: Session, stored_object_id: UUID) -> int:
        dataset_refs = int(
            session.scalar(
                select(func.count())
                .select_from(DatasetVersion)
                .where(DatasetVersion.stored_object_id == stored_object_id)
            )
            or 0
        )
        source_refs = int(
            session.scalar(
                select(func.count())
                .select_from(SourceFile)
                .where(SourceFile.stored_object_id == stored_object_id)
            )
            or 0
        )
        return dataset_refs + source_refs

    def _mark_gc_candidates(self, stored_object_ids: list[UUID]) -> None:
        if not stored_object_ids:
            return
        now = _now()
        with self._uow() as uow:
            assert uow.session is not None
            objects = list(
                uow.session.scalars(
                    select(StoredObject)
                    .where(StoredObject.id.in_(set(stored_object_ids)))
                    .with_for_update()
                )
            )
            for stored_object in objects:
                if self._object_reference_count(uow.session, stored_object.id) == 0:
                    stored_object.gc_candidate_at = now
            uow.commit()

    def _purge_project(self, project_id: UUID) -> None:
        object_ids: list[UUID] = []
        with self._uow() as uow:
            assert uow.session is not None
            project = uow.projects.get_project(
                project_id.hex, for_update=True, include_deleted=True
            )
            now = _now()
            if project is None:
                uow.commit()
                return
            eligible = (
                project.storage_mode == "temporary-cloud"
                and project.expires_at is not None
                and project.expires_at <= now
            ) or (
                project.deleted_at is not None
                and project.purge_after is not None
                and project.purge_after <= now
            )
            if not eligible:
                uow.commit()
                return
            object_ids = list(
                uow.session.scalars(
                    select(DatasetVersion.stored_object_id).where(
                        DatasetVersion.project_id == project.id
                    )
                )
            )
            uow.session.add(
                ProjectLifecycleEvent(
                    id=uuid4(),
                    project_id=project.id,
                    project_uuid_snapshot=project.id,
                    actor_user_id=project.owner_user_id,
                    event_type="purge",
                    details={},
                    created_at=now,
                )
            )
            uow.session.delete(project)
            uow.commit()
        self._mark_gc_candidates(object_ids)

    def _collect_garbage(self) -> int:
        with self._uow() as uow:
            assert uow.session is not None
            candidate_ids = list(
                uow.session.scalars(
                    select(StoredObject.id).where(
                        or_(
                            StoredObject.gc_candidate_at.is_not(None),
                            StoredObject.status == "deleting",
                        )
                    )
                )
            )
        deleted = 0
        for stored_object_id in candidate_ids:
            object_key: str | None = None
            with self._uow() as uow:
                assert uow.session is not None
                stored_object = uow.session.scalar(
                    select(StoredObject)
                    .where(StoredObject.id == stored_object_id)
                    .with_for_update()
                )
                if stored_object is None or stored_object.status == "deleted":
                    uow.commit()
                    continue
                if self._object_reference_count(uow.session, stored_object.id) != 0:
                    stored_object.gc_candidate_at = None
                    uow.commit()
                    continue
                if stored_object.status == "available":
                    stored_object.status = "deleting"
                    stored_object.updated_at = _now()
                if stored_object.status != "deleting":
                    uow.commit()
                    continue
                object_key = stored_object.object_key
                uow.commit()
            if object_key is None:
                continue
            try:
                self.storage.delete(object_key)
            except Exception:
                continue
            with self._uow() as uow:
                assert uow.session is not None
                stored_object = uow.session.scalar(
                    select(StoredObject)
                    .where(StoredObject.id == stored_object_id)
                    .with_for_update()
                )
                if (
                    stored_object is not None
                    and stored_object.status == "deleting"
                    and self._object_reference_count(uow.session, stored_object.id) == 0
                ):
                    stored_object.status = "deleted"
                    stored_object.deleted_at = _now()
                    stored_object.gc_candidate_at = None
                    stored_object.updated_at = _now()
                    deleted += 1
                uow.commit()
        return deleted

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
        try:
            with self._uow() as uow:
                assert uow.session is not None
                guest_session = uow.session.scalar(
                    select(GuestSession)
                    .where(GuestSession.token_digest == guest_token_digest)
                    .with_for_update()
                )
                guest_expiry = now + timedelta(seconds=self.guest_session_ttl_seconds)
                if guest_session is None:
                    guest_session = GuestSession(
                        id=uuid4(),
                        token_digest=guest_token_digest,
                        status="active",
                        expires_at=guest_expiry,
                        last_seen_at=now,
                        created_at=now,
                    )
                else:
                    guest_session.status = "active"
                    guest_session.revoked_at = None
                    guest_session.expires_at = guest_expiry
                    guest_session.last_seen_at = now
                project = Project(
                    id=_uuid(project_id),
                    guest_session=guest_session,
                    storage_mode="temporary-cloud",
                    title=title,
                    description="",
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
        quality_report = repository.current_quality_report(project)
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
            "guest_token_digest": (
                project.guest_session.token_digest if project.guest_session is not None else None
            ),
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
                json.dumps(quality_report.report_document, ensure_ascii=False)
                if quality_report is not None
                else json.dumps(version.quality_document, ensure_ascii=False)
                if version is not None
                else None
            ),
            "data_blob": None,
            "ready": ready,
        }

    @staticmethod
    def _dedup_scope(project: Project) -> str:
        if project.storage_mode == "saved-cloud" and project.owner_user_id is not None:
            return f"user:{project.owner_user_id.hex}"
        if project.guest_session_id is not None:
            return f"guest:{project.guest_session_id.hex}"
        if project.owner_user_id is not None:
            return f"user:{project.owner_user_id.hex}"
        raise PersistenceConflict("Project has no deduplication security scope.")

    def _project_dedup_scope(self, project_id: str) -> str:
        with self._uow() as uow:
            project = uow.projects.get_project(project_id)
            if project is None:
                raise PersistenceNotFound("Project does not exist.")
            return self._dedup_scope(project)

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
                        if project.guest_session is not None:
                            project.guest_session.last_seen_at = now
                            project.guest_session.expires_at = now + timedelta(
                                seconds=self.guest_session_ttl_seconds
                            )
                    row = self._project_row(uow.projects, project)
                    row["expires_at"] = iso_at(project.expires_at) if project.expires_at else None
                    row["updated_at"] = iso_at(project.updated_at)
                    uow.commit()
                    return row
                return self._project_row(uow.projects, project)
        except Exception as exc:
            raise _translate_database_error(exc) from exc

    @staticmethod
    def _guest_matches(project: Project, guest_token_digest: str | None, now: datetime) -> bool:
        guest = project.guest_session
        return bool(
            guest_token_digest
            and guest is not None
            and guest.status == "active"
            and guest.expires_at > now
            and secrets.compare_digest(guest.token_digest, guest_token_digest)
        )

    def save_project(
        self,
        project_id: str,
        owner_user_id: str,
        *,
        guest_token_digest: str | None,
    ) -> str:
        now = _now()
        owner_id = _uuid(owner_user_id)
        try:
            with self._uow() as uow:
                assert uow.session is not None
                project = uow.projects.get_project(project_id, for_update=True)
                owner = uow.session.get(User, owner_id)
                if project is None or project.current_revision_id is None:
                    raise PersistenceNotFound("Ready project does not exist.")
                if owner is None:
                    raise PersistenceNotFound("Authenticated user does not exist.")
                if project.storage_mode == "saved-cloud":
                    if project.owner_user_id != owner_id:
                        raise PersistenceConflict("Project belongs to another user.")
                    uow.commit()
                    return iso_at(project.updated_at)
                if project.storage_mode != "temporary-cloud" or not self._guest_matches(
                    project, guest_token_digest, now
                ):
                    raise PersistenceConflict("GuestSession cannot claim this project.")
                guest = project.guest_session
                if guest is None:
                    raise PersistenceConflict("Temporary project has no GuestSession owner.")
                existing_claim = uow.session.get(ProjectClaim, project.id)
                if existing_claim is not None:
                    if existing_claim.user_uuid_snapshot != owner_id:
                        raise PersistenceConflict("Project was already claimed by another user.")
                else:
                    uow.session.add(
                        ProjectClaim(
                            project_id=project.id,
                            guest_session_id=guest.id,
                            user_id=owner.id,
                            guest_session_uuid_snapshot=guest.id,
                            user_uuid_snapshot=owner.id,
                            claimed_at=now,
                        )
                    )
                    uow.session.add(
                        ProjectLifecycleEvent(
                            id=uuid4(),
                            project_id=project.id,
                            project_uuid_snapshot=project.id,
                            actor_user_id=owner.id,
                            event_type="claim",
                            details={"guestSessionId": guest.id.hex},
                            created_at=now,
                        )
                    )
                project.owner = owner
                project.guest_session = None
                project.storage_mode = "saved-cloud"
                project.expires_at = None
                project.saved_at = project.saved_at or now
                project.updated_at = now
                project.last_activity_at = now
                project.lock_version += 1
                uow.session.add(
                    ProjectLifecycleEvent(
                        id=uuid4(),
                        project_id=project.id,
                        project_uuid_snapshot=project.id,
                        actor_user_id=owner.id,
                        event_type="save",
                        details={"objectLifecycleReconciliationRequired": True},
                        created_at=now,
                    )
                )
                uow.commit()
                return iso_at(now)
        except Exception as exc:
            raise _translate_database_error(exc) from exc

    @staticmethod
    def _duplicate_access_allowed(
        project: Project,
        owner_id: UUID,
        guest_token_digest: str | None,
        now: datetime,
    ) -> bool:
        if project.storage_mode == "saved-cloud":
            return project.owner_user_id == owner_id
        return PostgresProjectStore._guest_matches(project, guest_token_digest, now)

    @staticmethod
    def _version_closure(
        session: Session,
        active_version: DatasetVersion,
        report: QualityReportRecord | None,
    ) -> list[DatasetVersion]:
        ordered: list[DatasetVersion] = []
        visited: set[UUID] = set()

        def visit(version: DatasetVersion) -> None:
            if version.id in visited:
                return
            if version.parent_version_id is not None:
                parent = session.get(DatasetVersion, version.parent_version_id)
                if parent is None:
                    raise PersistenceConflict("DatasetVersion parent lineage is incomplete.")
                visit(parent)
            visited.add(version.id)
            ordered.append(version)

        visit(active_version)
        if report is not None:
            visit(report.dataset_version)
        return ordered

    def _prepare_duplicate_objects(
        self,
        *,
        source_project_id: str,
        owner_id: UUID,
        guest_token_digest: str | None,
    ) -> tuple[UUID, dict[UUID, dict[str, Any]]]:
        with self._uow() as uow:
            assert uow.session is not None
            project = uow.projects.get_project(source_project_id)
            if project is None or project.current_revision_id is None:
                raise PersistenceNotFound("Ready source project does not exist.")
            if not self._duplicate_access_allowed(project, owner_id, guest_token_digest, _now()):
                raise PersistenceConflict("Source project access changed.")
            revision = uow.projects.current_revision(project)
            active_version = uow.projects.current_dataset_version(project)
            report = uow.projects.current_quality_report(project)
            if revision is None or active_version is None:
                raise PersistenceConflict("Source ProjectRevision is incomplete.")
            versions = self._version_closure(uow.session, active_version, report)
            source_revision_id = revision.id
            same_owner_saved_project = (
                project.storage_mode == "saved-cloud" and project.owner_user_id == owner_id
            )
            source_objects = {
                version.stored_object.id: {
                    "id": version.stored_object.id,
                    "object_key": version.stored_object.object_key,
                    "storage_backend": version.stored_object.storage_backend,
                    "purpose": version.stored_object.purpose,
                    "media_type": version.stored_object.media_type,
                    "size_bytes": version.stored_object.size_bytes,
                    "sha256": version.stored_object.sha256,
                    "encryption_key_id": version.stored_object.encryption_key_id,
                    "dedup_scope": version.stored_object.dedup_scope,
                    "format_contract_version": version.stored_object.format_contract_version,
                    "status": version.stored_object.status,
                }
                for version in versions
            }
        target_scope = f"user:{owner_id.hex}"
        prepared: dict[UUID, dict[str, Any]] = {}
        for source_object_id, source_object in source_objects.items():
            if source_object["status"] != "available":
                raise PersistenceConflict("Source DatasetVersion object is not available.")
            # A claimed Project keeps its immutable object graph in place. Once the
            # source Project is owner-authorized, another Project of that same owner
            # can safely add an FK reference even if the object's creation scope still
            # records the former GuestSession. The source Project authorization—not
            # a project id embedded in the key—is the security boundary here.
            if source_object["dedup_scope"] == target_scope or same_owner_saved_project:
                prepared[source_object_id] = {"reuse_source": True, **source_object}
                continue
            target_key = _scope_storage_key(target_scope, str(source_object["sha256"]))
            preexisting = self.storage.exists(target_key)
            with self.storage.open(str(source_object["object_key"])) as source_stream:
                staged = self.storage.stage(
                    target_key,
                    source_stream,
                    expected_sha256=str(source_object["sha256"]),
                )
            try:
                self.storage.confirm(staged)
            except Exception:
                self.storage.discard(staged)
                raise
            prepared[source_object_id] = {
                **source_object,
                "reuse_source": False,
                "target_key": target_key,
                "target_scope": target_scope,
                "preexisting": preexisting,
            }
        return source_revision_id, prepared

    def _compensate_duplicate_objects(self, prepared: dict[UUID, dict[str, Any]]) -> None:
        for item in prepared.values():
            if item.get("reuse_source") or item.get("preexisting"):
                continue
            target_key = str(item["target_key"])
            with self._uow() as uow:
                assert uow.session is not None
                persisted = uow.session.scalar(
                    select(StoredObject.id).where(StoredObject.object_key == target_key)
                )
            if persisted is None:
                self.storage.delete(target_key)

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
        owner_id = _uuid(owner_user_id)
        if idempotency_key is not None and not 1 <= len(idempotency_key) <= 255:
            raise PersistenceConflict("Idempotency-Key must contain 1 to 255 characters.")
        if idempotency_key is not None:
            with self._uow() as uow:
                assert uow.session is not None
                existing = uow.session.scalar(
                    select(IdempotencyRecord).where(
                        IdempotencyRecord.actor_user_id == owner_id,
                        IdempotencyRecord.operation == "duplicate-project",
                        IdempotencyRecord.idempotency_key == idempotency_key,
                    )
                )
                if existing is not None:
                    return existing.resource_id.hex

        prepared: dict[UUID, dict[str, Any]] = {}
        try:
            source_revision_id, prepared = self._prepare_duplicate_objects(
                source_project_id=source_project_id,
                owner_id=owner_id,
                guest_token_digest=guest_token_digest,
            )
            now = _now()
            with self._uow() as uow:
                assert uow.session is not None
                session = uow.session
                source_project = uow.projects.get_project(source_project_id, for_update=True)
                owner = session.get(User, owner_id)
                if source_project is None or owner is None:
                    raise PersistenceNotFound("Source project or owner does not exist.")
                if not self._duplicate_access_allowed(
                    source_project, owner_id, guest_token_digest, now
                ):
                    raise PersistenceConflict("Source project access changed.")
                source_revision = uow.projects.current_revision(source_project)
                active_version = uow.projects.current_dataset_version(source_project)
                source_report = uow.projects.current_quality_report(source_project)
                source_decision_set = uow.projects.current_decision_set(source_project)
                source_chart = uow.projects.current_chart_revision(source_project)
                source_file = uow.projects.get_source_file(source_project_id)
                if (
                    source_revision is None
                    or source_revision.id != source_revision_id
                    or active_version is None
                    or source_chart is None
                    or source_file is None
                ):
                    raise PersistenceConflict("Source ProjectRevision changed during duplicate.")
                source_dataset = active_version.dataset
                versions = self._version_closure(session, active_version, source_report)
                target_scope = f"user:{owner.id.hex}"

                stored_object_map: dict[UUID, StoredObject] = {}
                target_object_by_signature: dict[tuple[str, int], StoredObject] = {}
                for source_version in versions:
                    source_object = source_version.stored_object
                    if source_object.id in stored_object_map:
                        continue
                    item = prepared[source_object.id]
                    signature = (source_object.sha256, source_object.size_bytes)
                    if not item["reuse_source"] and signature in target_object_by_signature:
                        stored_object_map[source_object.id] = target_object_by_signature[signature]
                        continue
                    if item["reuse_source"]:
                        target_object = session.scalar(
                            select(StoredObject)
                            .where(StoredObject.id == source_object.id)
                            .with_for_update()
                        )
                    else:
                        target_object = session.scalar(
                            select(StoredObject)
                            .where(
                                StoredObject.storage_backend == source_object.storage_backend,
                                StoredObject.dedup_scope == target_scope,
                                StoredObject.purpose == source_object.purpose,
                                StoredObject.media_type == source_object.media_type,
                                StoredObject.format_contract_version
                                == source_object.format_contract_version,
                                StoredObject.sha256 == source_object.sha256,
                                StoredObject.size_bytes == source_object.size_bytes,
                                (
                                    StoredObject.encryption_key_id.is_(None)
                                    if source_object.encryption_key_id is None
                                    else StoredObject.encryption_key_id
                                    == source_object.encryption_key_id
                                ),
                            )
                            .with_for_update()
                        )
                        if target_object is None:
                            target_object = StoredObject(
                                id=uuid4(),
                                storage_backend=source_object.storage_backend,
                                object_key=str(item["target_key"]),
                                staging_key=None,
                                purpose=source_object.purpose,
                                status="available",
                                media_type=source_object.media_type,
                                size_bytes=source_object.size_bytes,
                                sha256=source_object.sha256,
                                encryption_key_id=source_object.encryption_key_id,
                                dedup_scope=target_scope,
                                format_contract_version=source_object.format_contract_version,
                                created_at=now,
                                updated_at=now,
                            )
                            session.add(target_object)
                    if target_object is None or target_object.status != "available":
                        raise PersistenceConflict("StoredObject is not available for duplicate.")
                    stored_object_map[source_object.id] = target_object
                    if not item["reuse_source"]:
                        target_object_by_signature[signature] = target_object

                copy_title = f"{source_project.title} copy"[:200]
                target_project = Project(
                    id=_uuid(project_id),
                    owner=owner,
                    storage_mode="saved-cloud",
                    title=copy_title,
                    description=source_project.description,
                    saved_at=now,
                    last_activity_at=now,
                    created_at=now,
                    updated_at=now,
                )
                target_source = SourceFile(
                    id=uuid4(),
                    project=target_project,
                    original_name=source_file.original_name,
                    media_type=source_file.media_type,
                    size_bytes=source_file.size_bytes,
                    sha256=source_file.sha256,
                    sheet_name=source_file.sheet_name,
                    available_sheets=deepcopy(source_file.available_sheets),
                    header_row=source_file.header_row,
                    parser_name=source_file.parser_name,
                    parser_version=source_file.parser_version,
                    binary_deleted_at=source_file.binary_deleted_at,
                    parsed_at=source_file.parsed_at,
                    created_at=now,
                )
                target_dataset = Dataset(
                    id=uuid4(),
                    project=target_project,
                    source_file=target_source,
                    name=source_dataset.name,
                    sheet_name=source_dataset.sheet_name,
                    header_row=source_dataset.header_row,
                    created_at=now,
                )
                session.add_all([target_project, target_source, target_dataset])

                version_map: dict[UUID, DatasetVersion] = {}
                deferred_versions: list[DatasetVersion] = []
                for source_version in versions:
                    if source_version.cleaning_decision_set_id is not None:
                        deferred_versions.append(source_version)
                        continue
                    parent = (
                        version_map.get(source_version.parent_version_id)
                        if source_version.parent_version_id is not None
                        else None
                    )
                    cloned_version = DatasetVersion(
                        id=uuid4(),
                        project=target_project,
                        dataset=target_dataset,
                        parent_version=parent,
                        stored_object=stored_object_map[source_version.stored_object_id],
                        version_number=len(version_map) + 1,
                        kind=source_version.kind,
                        schema_document=deepcopy(source_version.schema_document),
                        preview_document={
                            **deepcopy(source_version.preview_document),
                            "projectId": target_project.id.hex,
                        },
                        quality_document={
                            **deepcopy(source_version.quality_document),
                            "projectId": target_project.id.hex,
                        },
                        parquet_schema_version=source_version.parquet_schema_version,
                        content_sha256=source_version.content_sha256,
                        row_count=source_version.row_count,
                        column_count=source_version.column_count,
                        created_at=now,
                    )
                    version_map[source_version.id] = cloned_version
                    session.add(cloned_version)

                target_report: QualityReportRecord | None = None
                finding_map: dict[UUID, QualityFindingRecord] = {}
                if source_report is not None:
                    report_version = version_map.get(source_report.dataset_version_id)
                    if report_version is None:
                        raise PersistenceConflict(
                            "QualityReport input is not in duplicate closure."
                        )
                    source_run = source_report.processing_run
                    report_run = ProcessingRun(
                        id=uuid4(),
                        project=target_project,
                        input_dataset_version=report_version,
                        operation=source_run.operation,
                        status="succeeded",
                        parameters={
                            **deepcopy(source_run.parameters),
                            "reusedFromRunId": source_run.id.hex,
                        },
                        algorithm_version=source_run.algorithm_version,
                        code_version=source_run.code_version,
                        execution_mode="reused-result",
                        origin_processing_run_id=source_run.id,
                        origin_run_uuid_snapshot=source_run.id,
                        started_at=now,
                        finished_at=now,
                        created_at=now,
                    )
                    target_report = QualityReportRecord(
                        id=uuid4(),
                        project=target_project,
                        dataset_version=report_version,
                        processing_run=report_run,
                        revision_number=1,
                        status=source_report.status,
                        profiler_name=source_report.profiler_name,
                        profiler_version=source_report.profiler_version,
                        algorithm_version=source_report.algorithm_version,
                        code_version=source_report.code_version,
                        parameters=deepcopy(source_report.parameters),
                        report_document={
                            **deepcopy(source_report.report_document),
                            "projectId": target_project.id.hex,
                        },
                        completed_at=now,
                        created_at=now,
                    )
                    session.add_all([report_run, target_report])
                    for source_finding in source_report.findings:
                        refs = deepcopy(source_finding.source_record_refs)
                        for reference in refs:
                            source_version_text = str(reference.get("datasetVersionId", ""))
                            for old_id, new_version in version_map.items():
                                if source_version_text.replace("-", "") == old_id.hex:
                                    reference["datasetVersionId"] = new_version.id.hex
                        target_finding = QualityFindingRecord(
                            id=uuid4(),
                            project_id=target_project.id,
                            quality_report=target_report,
                            external_id=source_finding.external_id,
                            kind=source_finding.kind,
                            severity=source_finding.severity,
                            column_name=source_finding.column_name,
                            column_identity=deepcopy(source_finding.column_identity),
                            source_record_refs=refs,
                            affected_count=source_finding.affected_count,
                            evidence_document=deepcopy(source_finding.evidence_document),
                            summary=source_finding.summary,
                            reason=source_finding.reason,
                            created_at=now,
                        )
                        finding_map[source_finding.id] = target_finding
                        session.add(target_finding)

                target_decision_set: CleaningDecisionSet | None = None
                if source_decision_set is not None:
                    if target_report is None:
                        raise PersistenceConflict("DecisionSet has no QualityReport in closure.")
                    decision_input = version_map.get(source_decision_set.input_dataset_version_id)
                    if decision_input is None:
                        raise PersistenceConflict("DecisionSet input is not in duplicate closure.")
                    target_decision_set = CleaningDecisionSet(
                        id=uuid4(),
                        project=target_project,
                        quality_report=target_report,
                        input_dataset_version=decision_input,
                        created_by=owner,
                        revision_number=1,
                        decisions_hash=source_decision_set.decisions_hash,
                        created_at=now,
                    )
                    session.add(target_decision_set)
                    for source_decision in source_decision_set.decisions:
                        decision_finding = finding_map.get(source_decision.quality_finding_id)
                        if decision_finding is None:
                            raise PersistenceConflict(
                                "Decision finding is not in duplicate closure."
                            )
                        session.add(
                            CleaningDecisionRecord(
                                id=uuid4(),
                                project_id=target_project.id,
                                decision_set=target_decision_set,
                                quality_finding=decision_finding,
                                action=source_decision.action,
                                created_at=now,
                            )
                        )

                for source_version in deferred_versions:
                    if (
                        source_decision_set is None
                        or source_version.cleaning_decision_set_id != source_decision_set.id
                        or target_decision_set is None
                    ):
                        raise PersistenceConflict(
                            "Duplicate closure would require an unrelated DecisionSet history."
                        )
                    parent = (
                        version_map.get(source_version.parent_version_id)
                        if source_version.parent_version_id is not None
                        else None
                    )
                    cloned_version = DatasetVersion(
                        id=uuid4(),
                        project=target_project,
                        dataset=target_dataset,
                        parent_version=parent,
                        stored_object=stored_object_map[source_version.stored_object_id],
                        version_number=len(version_map) + 1,
                        kind=source_version.kind,
                        schema_document=deepcopy(source_version.schema_document),
                        preview_document={
                            **deepcopy(source_version.preview_document),
                            "projectId": target_project.id.hex,
                        },
                        quality_document={
                            **deepcopy(source_version.quality_document),
                            "projectId": target_project.id.hex,
                        },
                        parquet_schema_version=source_version.parquet_schema_version,
                        content_sha256=source_version.content_sha256,
                        cleaning_decision_set=target_decision_set,
                        row_count=source_version.row_count,
                        column_count=source_version.column_count,
                        created_at=now,
                    )
                    version_map[source_version.id] = cloned_version
                    session.add(cloned_version)

                target_active = version_map.get(active_version.id)
                if target_active is None:
                    raise PersistenceConflict("Active DatasetVersion was not duplicated.")
                source_parse_run = uow.projects.get_run_for_project(source_project_id)
                target_parse_run = ProcessingRun(
                    id=_uuid(job_id),
                    project=target_project,
                    output_dataset_version=next(iter(version_map.values())),
                    operation="parse",
                    status="succeeded",
                    parameters={
                        "apiJob": _api_job(
                            stage="ready", progress=100, message="Copied project is ready."
                        ),
                        "reusedFromRunId": source_parse_run.id.hex if source_parse_run else None,
                    },
                    algorithm_version=(
                        source_parse_run.algorithm_version if source_parse_run else "duplicate-v1"
                    ),
                    code_version=PHASE4_CODE_VERSION,
                    execution_mode="reused-result",
                    origin_processing_run_id=source_parse_run.id if source_parse_run else None,
                    origin_run_uuid_snapshot=(
                        source_parse_run.id if source_parse_run else source_revision.id
                    ),
                    started_at=now,
                    finished_at=now,
                    created_at=now,
                )
                session.add(target_parse_run)
                if target_decision_set is not None:
                    source_clean_run = session.scalar(
                        select(ProcessingRun).where(
                            ProcessingRun.output_dataset_version_id == active_version.id
                        )
                    )
                    session.add(
                        ProcessingRun(
                            id=uuid4(),
                            project=target_project,
                            input_dataset_version=target_decision_set.input_dataset_version,
                            output_dataset_version=target_active,
                            operation="clean",
                            status="succeeded",
                            parameters={
                                "cleaningDecisionSetId": target_decision_set.id.hex,
                                "reusedFromRunId": (
                                    source_clean_run.id.hex if source_clean_run else None
                                ),
                            },
                            algorithm_version=(
                                source_clean_run.algorithm_version
                                if source_clean_run
                                else CLEANING_ALGORITHM_VERSION
                            ),
                            code_version=PHASE4_CODE_VERSION,
                            execution_mode="reused-result",
                            origin_processing_run_id=(
                                source_clean_run.id if source_clean_run else None
                            ),
                            origin_run_uuid_snapshot=(
                                source_clean_run.id if source_clean_run else source_revision.id
                            ),
                            started_at=now,
                            finished_at=now,
                            created_at=now,
                        )
                    )

                target_chart = ChartSpecRevision(
                    id=uuid4(),
                    project=target_project,
                    dataset_version=target_active,
                    created_by=owner,
                    revision_number=1,
                    schema_version=source_chart.schema_version,
                    decision_set_revision=1 if target_decision_set is not None else None,
                    cleaning_decision_set=target_decision_set,
                    spec_document=deepcopy(source_chart.spec_document),
                    created_at=now,
                )
                source_spec = ProjectSpecV1.model_validate(source_revision.spec_document)
                target_spec = source_spec.model_copy(
                    update={
                        "project_id": target_project.id,
                        "title": copy_title,
                        "source": ProjectSourceSpec(
                            source_file_id=target_source.id,
                            dataset_id=target_dataset.id,
                            dataset_version_id=target_active.id,
                            sheet_name=target_source.sheet_name,
                            header_row=target_source.header_row,
                        ),
                        "cleaning": (
                            ProjectCleaningSpec(decision_set_id=target_decision_set.id, revision=1)
                            if target_decision_set is not None
                            else None
                        ),
                        "chart": ChartSpec.model_validate(target_chart.spec_document),
                    }
                )
                target_revision = ProjectRevision(
                    id=uuid4(),
                    project=target_project,
                    active_dataset_version=target_active,
                    chart_spec_revision=target_chart,
                    quality_report=target_report,
                    cleaning_decision_set=target_decision_set,
                    created_by=owner,
                    revision_number=1,
                    spec_schema_version=1,
                    spec_document=target_spec.model_dump(mode="json", by_alias=True),
                    created_at=now,
                )
                target_project.current_revision = target_revision
                session.add_all([target_chart, target_revision])
                # ProjectOrigin and lifecycle rows intentionally carry scalar FKs
                # rather than aggregate relationships. Flush the complete target
                # graph first so PostgreSQL never observes provenance before its
                # target Project exists.
                session.flush()
                session.add(
                    ProjectOrigin(
                        target_project_id=target_project.id,
                        source_project_id=source_project.id,
                        source_revision_id=source_revision.id,
                        source_project_uuid_snapshot=source_project.id,
                        source_revision_uuid_snapshot=source_revision.id,
                        origin_kind="duplicate",
                        created_at=now,
                    )
                )
                session.add(
                    ProjectLifecycleEvent(
                        id=uuid4(),
                        project_id=target_project.id,
                        project_uuid_snapshot=target_project.id,
                        actor_user_id=owner.id,
                        event_type="duplicate",
                        details={
                            "sourceProjectId": source_project.id.hex,
                            "sourceRevisionId": source_revision.id.hex,
                        },
                        created_at=now,
                    )
                )
                if idempotency_key is not None:
                    session.add(
                        IdempotencyRecord(
                            id=uuid4(),
                            actor_user_id=owner.id,
                            operation="duplicate-project",
                            idempotency_key=idempotency_key,
                            resource_id=target_project.id,
                            response_document={"projectId": target_project.id.hex},
                            created_at=now,
                        )
                    )
                uow.commit()
            return _uuid(project_id).hex
        except Exception as exc:
            if prepared:
                self._compensate_duplicate_objects(prepared)
            if idempotency_key is not None:
                with self._uow() as uow:
                    assert uow.session is not None
                    existing = uow.session.scalar(
                        select(IdempotencyRecord).where(
                            IdempotencyRecord.actor_user_id == owner_id,
                            IdempotencyRecord.operation == "duplicate-project",
                            IdempotencyRecord.idempotency_key == idempotency_key,
                        )
                    )
                    if existing is not None:
                        return existing.resource_id.hex
            raise _translate_database_error(exc) from exc

    def list_projects(self, owner_user_id: str | None) -> list[dict[str, Any]]:
        if owner_user_id is None:
            return []
        try:
            with self._uow() as uow:
                assert uow.session is not None
                projects = list(
                    uow.session.scalars(
                        select(Project)
                        .where(
                            Project.owner_user_id == _uuid(owner_user_id),
                            Project.storage_mode == "saved-cloud",
                            Project.deleted_at.is_(None),
                        )
                        .order_by(Project.updated_at.desc())
                    )
                )
                return [self._project_row(uow.projects, project) for project in projects]
        except Exception as exc:
            raise _translate_database_error(exc) from exc

    def list_deleted_projects(self, owner_user_id: str) -> list[dict[str, Any]]:
        try:
            with self._uow() as uow:
                assert uow.session is not None
                projects = list(
                    uow.session.scalars(
                        select(Project)
                        .where(
                            Project.owner_user_id == _uuid(owner_user_id),
                            Project.storage_mode == "saved-cloud",
                            Project.deleted_at.is_not(None),
                            Project.purge_after > _now(),
                        )
                        .order_by(Project.deleted_at.desc())
                    )
                )
                return [self._project_row(uow.projects, project) for project in projects]
        except Exception as exc:
            raise _translate_database_error(exc) from exc

    def get_workspace(self, project_id: str) -> dict[str, Any]:
        try:
            with self._uow() as uow:
                assert uow.session is not None
                project = uow.projects.get_project(project_id)
                if project is None or project.current_revision_id is None:
                    raise PersistenceNotFound("Ready project does not exist.")
                revision = uow.projects.current_revision(project)
                version = uow.projects.current_dataset_version(project)
                chart = uow.projects.current_chart_revision(project)
                report = uow.projects.current_quality_report(project)
                decision_set = uow.projects.current_decision_set(project)
                if revision is None or version is None or chart is None or report is None:
                    raise PersistenceConflict("ProjectRevision workspace graph is incomplete.")
                if chart.dataset_version_id != version.id:
                    raise PersistenceConflict("ChartSpecRevision does not bind the active data.")
                if decision_set is not None and (
                    revision.cleaning_decision_set_id != decision_set.id
                    or chart.cleaning_decision_set_id != decision_set.id
                    or decision_set.quality_report_id != report.id
                ):
                    raise PersistenceConflict("Cleaning lineage is inconsistent.")
                ProjectSpecV1.model_validate(revision.spec_document)
                decisions: list[dict[str, str]] = []
                if decision_set is not None:
                    decisions = [
                        {
                            "findingId": decision.quality_finding.external_id,
                            "action": decision.action,
                        }
                        for decision in sorted(
                            decision_set.decisions,
                            key=lambda item: item.quality_finding.external_id,
                        )
                    ]
                return {
                    "project": self._project_row(uow.projects, project),
                    "preview": version.preview_document,
                    "quality": report.report_document,
                    "decisions": decisions,
                    "chart": chart.spec_document,
                    "shares": [],
                }
        except Exception as exc:
            raise _translate_database_error(exc) from exc

    def delete_project(
        self,
        project_id: str,
        *,
        owner_user_id: str,
        guest_token_digest: str | None,
    ) -> bool:
        now = _now()
        object_ids: list[UUID] = []
        try:
            with self._uow() as uow:
                assert uow.session is not None
                project = uow.projects.get_project(project_id, for_update=True)
                if project is None:
                    return False
                actor_id = _uuid(owner_user_id)
                if project.storage_mode == "saved-cloud":
                    if project.owner_user_id != actor_id:
                        raise PersistenceConflict("Project belongs to another user.")
                    project.deleted_at = now
                    project.purge_after = now + timedelta(hours=24)
                    project.updated_at = now
                    project.lock_version += 1
                    uow.session.add(
                        ProjectLifecycleEvent(
                            id=uuid4(),
                            project_id=project.id,
                            project_uuid_snapshot=project.id,
                            actor_user_id=actor_id,
                            event_type="delete",
                            details={"purgeAfter": iso_at(project.purge_after)},
                            created_at=now,
                        )
                    )
                else:
                    if not self._guest_matches(project, guest_token_digest, now):
                        raise PersistenceConflict("GuestSession cannot delete this project.")
                    object_ids = list(
                        uow.session.scalars(
                            select(DatasetVersion.stored_object_id).where(
                                DatasetVersion.project_id == project.id
                            )
                        )
                    )
                    uow.session.delete(project)
                uow.commit()
            if object_ids:
                self._mark_gc_candidates(object_ids)
            return True
        except Exception as exc:
            raise _translate_database_error(exc) from exc

    def restore_deleted_project(self, project_id: str, owner_user_id: str | None = None) -> str:
        now = _now()
        try:
            with self._uow() as uow:
                assert uow.session is not None
                project = uow.projects.get_project(
                    project_id, for_update=True, include_deleted=True
                )
                if project is None or project.deleted_at is None or project.purge_after is None:
                    raise PersistenceNotFound("Deleted project does not exist.")
                if owner_user_id is not None and project.owner_user_id != _uuid(owner_user_id):
                    raise PersistenceNotFound("Deleted project does not exist.")
                if project.storage_mode != "saved-cloud" or project.purge_after <= now:
                    raise PersistenceNotFound("The project recovery window has expired.")
                project.deleted_at = None
                project.purge_after = None
                project.updated_at = now
                project.last_activity_at = now
                project.lock_version += 1
                uow.session.add(
                    ProjectLifecycleEvent(
                        id=uuid4(),
                        project_id=project.id,
                        project_uuid_snapshot=project.id,
                        actor_user_id=project.owner_user_id,
                        event_type="restore",
                        details={},
                        created_at=now,
                    )
                )
                uow.commit()
            return iso_at(now)
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
        dedup_scope = self._project_dedup_scope(project_id)
        final_key = _scope_storage_key(dedup_scope, artifact.sha256)
        staged = self.storage.stage(
            final_key,
            io.BytesIO(artifact.payload),
            expected_sha256=artifact.sha256,
        )
        try:
            with self.storage.open_staged(staged) as staged_stream:
                read_parquet(staged_stream.read(), expected_sha256=artifact.sha256)
            persistence_result = self._persist_processed_project(
                project_id=project_id,
                source=source,
                version_id=version_id,
                staged=staged,
                artifact_schema=artifact.schema_document,
                frame=frame,
                row_count=artifact.row_count,
                column_count=artifact.column_count,
                preview=preview,
                quality=quality,
                chart=chart,
            )
        except Exception as exc:
            self.storage.discard(staged)
            raise _translate_database_error(exc) from exc
        if persistence_result == "noop":
            self.storage.discard(staged)
            return
        if persistence_result == "reused":
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
        frame: pd.DataFrame,
        row_count: int,
        column_count: int,
        preview: dict[str, Any],
        quality: dict[str, Any],
        chart: dict[str, Any],
    ) -> str:
        now = _now()
        with self._uow() as uow:
            assert uow.session is not None
            project = uow.projects.get_project(project_id, for_update=True)
            if project is None:
                raise PersistenceNotFound("Project does not exist.")
            if project.current_revision_id is not None:
                return "noop"
            source_file = uow.projects.get_source_file(project_id)
            parse_run = uow.projects.get_run_for_project(project_id)
            if source_file is None or parse_run is None:
                raise PersistenceNotFound("Pending project metadata is incomplete.")

            dedup_scope = self._dedup_scope(project)
            stored_object = uow.session.scalar(
                select(StoredObject)
                .where(
                    StoredObject.storage_backend == "local",
                    StoredObject.dedup_scope == dedup_scope,
                    StoredObject.purpose == "dataset",
                    StoredObject.media_type == "application/vnd.apache.parquet",
                    StoredObject.format_contract_version == "parquet-v1",
                    StoredObject.sha256 == staged.sha256,
                    StoredObject.size_bytes == staged.size_bytes,
                    StoredObject.encryption_key_id.is_(None),
                )
                .with_for_update()
            )
            reused = stored_object is not None
            if stored_object is not None and stored_object.status != "available":
                raise PersistenceConflict("Matching StoredObject is not available for reuse.")
            if stored_object is None:
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
                    dedup_scope=dedup_scope,
                    format_contract_version="parquet-v1",
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
            profile_run = ProcessingRun(
                id=uuid4(),
                project=project,
                input_dataset_version=version,
                operation="profile",
                status="succeeded",
                parameters={"validRanges": []},
                algorithm_version=QUALITY_ALGORITHM_VERSION,
                code_version=PHASE3_CODE_VERSION,
                started_at=now,
                finished_at=now,
                created_at=now,
            )
            quality_report = QualityReportRecord(
                id=uuid4(),
                project=project,
                dataset_version=version,
                processing_run=profile_run,
                revision_number=1,
                status="completed",
                profiler_name=QUALITY_PROFILER_NAME,
                profiler_version=QUALITY_PROFILER_VERSION,
                algorithm_version=QUALITY_ALGORITHM_VERSION,
                code_version=PHASE3_CODE_VERSION,
                parameters={"validRanges": []},
                report_document=quality,
                completed_at=now,
                created_at=now,
            )
            findings = _quality_findings(
                project_id=project.id,
                report_id=quality_report.id,
                dataset_version_id=version.id,
                frame=frame,
                schema_document=artifact_schema,
                quality=quality,
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
                quality_report=quality_report,
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
            parse_run.status = "running"
            parse_run.started_at = parse_run.started_at or now
            parse_run.algorithm_version = "parquet-v1"
            parse_run.parameters = {
                **parse_run.parameters,
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
            uow.session.add_all(
                [
                    stored_object,
                    dataset,
                    version,
                    profile_run,
                    quality_report,
                    *findings,
                    chart_revision,
                    project_revision,
                ]
            )
            uow.commit()
        return "reused" if reused else "created"

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
            run = uow.session.scalar(
                select(ProcessingRun)
                .where(
                    ProcessingRun.project_id == version.project_id,
                    ProcessingRun.parameters["pendingDatasetVersionId"].as_string()
                    == version.id.hex,
                )
                .with_for_update()
            )
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
            if run.operation == "parse":
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

    def _load_version_dataframe(self, version_id: UUID) -> pd.DataFrame:
        try:
            with self._uow() as uow:
                assert uow.session is not None
                version = uow.session.get(DatasetVersion, version_id)
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

    def load_quality_dataframe(self, project_id: str) -> pd.DataFrame:
        try:
            with self._uow() as uow:
                project = uow.projects.get_project(project_id)
                if project is None:
                    raise PersistenceNotFound("Project does not exist.")
                report = uow.projects.current_quality_report(project)
                version = (
                    report.dataset_version
                    if report is not None
                    else uow.projects.current_dataset_version(project)
                )
                if version is None:
                    raise PersistenceConflict("Project has no quality input DatasetVersion.")
                version_id = version.id
            return self._load_version_dataframe(version_id)
        except PersistenceError:
            raise
        except Exception as exc:
            raise _translate_database_error(exc) from exc

    def get_decisions(self, project_id: str) -> list[dict[str, str]]:
        try:
            with self._uow() as uow:
                assert uow.session is not None
                project = uow.projects.get_project(project_id)
                if project is None:
                    raise PersistenceNotFound("Project does not exist.")
                decision_set = uow.projects.current_decision_set(project)
                if decision_set is None:
                    return []
                rows = uow.session.execute(
                    select(CleaningDecisionRecord, QualityFindingRecord)
                    .join(
                        QualityFindingRecord,
                        CleaningDecisionRecord.quality_finding_id == QualityFindingRecord.id,
                    )
                    .where(CleaningDecisionRecord.decision_set_id == decision_set.id)
                    .order_by(QualityFindingRecord.external_id)
                ).all()
                return [
                    {"findingId": finding.external_id, "action": decision.action}
                    for decision, finding in rows
                ]
        except Exception as exc:
            raise _translate_database_error(exc) from exc

    def load_cleaned_dataframe(self, project_id: str) -> pd.DataFrame:
        return self.load_dataframe(project_id)

    def load_chart_dataframe(self, project_id: str) -> pd.DataFrame:
        frame = self.load_quality_dataframe(project_id)
        try:
            with self._uow() as uow:
                project = uow.projects.get_project(project_id)
                if project is None:
                    raise PersistenceNotFound("Project does not exist.")
                report = uow.projects.current_quality_report(project)
                if report is None:
                    raise PersistenceConflict("Project has no persistent QualityReport.")
                quality = report.report_document
            return apply_chart_decisions(frame, quality, self.get_decisions(project_id))
        except PersistenceError:
            raise
        except Exception as exc:
            raise _translate_database_error(exc) from exc

    def save_quality_report(
        self,
        project_id: str,
        quality: dict[str, Any],
        *,
        parameters: dict[str, Any],
    ) -> str:
        try:
            with self._uow() as uow:
                project = uow.projects.get_project(project_id)
                if project is None:
                    raise PersistenceNotFound("Project does not exist.")
                source_report = uow.projects.current_quality_report(project)
                source_version = (
                    source_report.dataset_version
                    if source_report is not None
                    else uow.projects.current_dataset_version(project)
                )
                if source_version is None:
                    raise PersistenceConflict("Project has no quality input DatasetVersion.")
                source_report_id = source_report.id if source_report is not None else None
                source_version_id = source_version.id
            frame = self._load_version_dataframe(source_version_id)
        except PersistenceError:
            raise
        except Exception as exc:
            raise _translate_database_error(exc) from exc
        now = _now()
        try:
            with self._uow() as uow:
                assert uow.session is not None
                project = uow.projects.get_project(project_id, for_update=True)
                if project is None:
                    raise PersistenceNotFound("Project does not exist.")
                current = uow.projects.current_revision(project)
                current_chart = uow.projects.current_chart_revision(project)
                previous_report = uow.projects.current_quality_report(project)
                input_version = (
                    previous_report.dataset_version
                    if previous_report is not None
                    else uow.projects.current_dataset_version(project)
                )
                if current is None or current_chart is None or input_version is None:
                    raise PersistenceConflict("Ready project lineage is incomplete.")
                if (
                    input_version.id != source_version_id
                    or (previous_report.id if previous_report is not None else None)
                    != source_report_id
                ):
                    raise PersistenceConflict("Quality input changed while profiling was running.")

                profile_run = ProcessingRun(
                    id=uuid4(),
                    project_id=project.id,
                    input_dataset_version_id=input_version.id,
                    operation="profile",
                    status="succeeded",
                    parameters=parameters,
                    algorithm_version=QUALITY_ALGORITHM_VERSION,
                    code_version=PHASE3_CODE_VERSION,
                    started_at=now,
                    finished_at=now,
                    created_at=now,
                )
                report_number = (
                    int(
                        uow.session.scalar(
                            select(func.max(QualityReportRecord.revision_number)).where(
                                QualityReportRecord.dataset_version_id == input_version.id
                            )
                        )
                        or 0
                    )
                    + 1
                )
                report = QualityReportRecord(
                    id=uuid4(),
                    project_id=project.id,
                    dataset_version_id=input_version.id,
                    processing_run=profile_run,
                    revision_number=report_number,
                    status="completed",
                    profiler_name=QUALITY_PROFILER_NAME,
                    profiler_version=QUALITY_PROFILER_VERSION,
                    algorithm_version=QUALITY_ALGORITHM_VERSION,
                    code_version=PHASE3_CODE_VERSION,
                    parameters=parameters,
                    report_document=quality,
                    completed_at=now,
                    created_at=now,
                )
                findings = _quality_findings(
                    project_id=project.id,
                    report_id=report.id,
                    dataset_version_id=input_version.id,
                    frame=frame,
                    schema_document=input_version.schema_document,
                    quality=quality,
                    created_at=now,
                )

                if (
                    current_chart.dataset_version_id == input_version.id
                    and current.cleaning_decision_set_id is None
                ):
                    next_chart = current_chart
                else:
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
                    next_chart = ChartSpecRevision(
                        id=uuid4(),
                        project_id=project.id,
                        dataset_version_id=input_version.id,
                        created_by_user_id=project.owner_user_id,
                        revision_number=chart_number,
                        schema_version=current_chart.schema_version,
                        decision_set_revision=None,
                        cleaning_decision_set_id=None,
                        spec_document=current_chart.spec_document,
                        created_at=now,
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
                previous_spec = ProjectSpecV1.model_validate(current.spec_document)
                next_spec = previous_spec.model_copy(
                    update={
                        "source": previous_spec.source.model_copy(
                            update={"dataset_version_id": input_version.id}
                        ),
                        "cleaning": None,
                        "chart": ChartSpec.model_validate(current_chart.spec_document),
                    }
                )
                next_revision = ProjectRevision(
                    id=uuid4(),
                    project_id=project.id,
                    active_dataset_version_id=input_version.id,
                    chart_spec_revision=next_chart,
                    quality_report=report,
                    cleaning_decision_set_id=None,
                    created_by_user_id=project.owner_user_id,
                    revision_number=project_number,
                    spec_schema_version=1,
                    spec_document=next_spec.model_dump(mode="json", by_alias=True),
                    created_at=now,
                )
                project.current_revision = next_revision
                project.last_activity_at = now
                project.updated_at = now
                if project.storage_mode == "temporary-cloud":
                    project.expires_at = now + timedelta(seconds=self.project_ttl_seconds)
                uow.session.add_all([profile_run, report, *findings, next_chart, next_revision])
                uow.commit()
            return iso_at(now)
        except Exception as exc:
            raise _translate_database_error(exc) from exc

    def save_decisions(self, project_id: str, decisions: list[dict[str, str]]) -> str:
        canonical = _canonical_decisions(decisions)
        decision_hash = _decisions_hash(canonical)
        pending_retry_at: str | None = None
        try:
            with self._uow() as uow:
                project = uow.projects.get_project(project_id)
                if project is None:
                    raise PersistenceNotFound("Project does not exist.")
                report = uow.projects.current_quality_report(project)
                current_set = uow.projects.current_decision_set(project)
                if report is None:
                    raise PersistenceConflict("Project has no persistent QualityReport.")
                if (
                    current_set is not None
                    and current_set.quality_report_id == report.id
                    and current_set.decisions_hash == decision_hash
                ):
                    output = current_set.output_dataset_version
                    if (
                        output is not None
                        and output.stored_object is not None
                        and output.stored_object.status == "available"
                    ):
                        return iso_at(current_set.created_at)
                    pending_retry_at = iso_at(current_set.created_at)
                report_id = report.id
                input_version_id = report.dataset_version_id
                quality = report.report_document
                schema_document = report.dataset_version.schema_document
            if pending_retry_at is not None:
                self.recover_pending_objects()
                recovered_project = self.get_project(project_id, touch=False)
                if recovered_project is not None and recovered_project["ready"]:
                    return pending_retry_at
                raise ObjectConfirmationPending(
                    "Cleaned DatasetVersion still requires object confirmation recovery."
                )
            frame = self._load_version_dataframe(input_version_id)
            cleaned = apply_chart_decisions(
                frame,
                quality,
                canonical,
                actions={"remove"},
            )
            preview = build_preview(project_id, cleaned)
            units = {
                str(item.get("name")): item.get("unit")
                for item in schema_document.get("columns", [])
            }
            artifact = write_parquet(cleaned, units=units)
            version_id = uuid4()
            dedup_scope = self._project_dedup_scope(project_id)
            final_key = _scope_storage_key(dedup_scope, artifact.sha256)
            staged = self.storage.stage(
                final_key,
                io.BytesIO(artifact.payload),
                expected_sha256=artifact.sha256,
            )
            try:
                with self.storage.open_staged(staged) as staged_stream:
                    read_parquet(staged_stream.read(), expected_sha256=artifact.sha256)
                persistence_result = self._persist_cleaning_result(
                    project_id=project_id,
                    quality_report_id=report_id,
                    decisions=canonical,
                    decisions_hash=decision_hash,
                    version_id=version_id,
                    staged=staged,
                    schema_document=artifact.schema_document,
                    preview=preview,
                    quality=quality,
                    row_count=artifact.row_count,
                    column_count=artifact.column_count,
                )
            except Exception as exc:
                self.storage.discard(staged)
                raise _translate_database_error(exc) from exc
            if persistence_result == "noop":
                self.storage.discard(staged)
                existing = self.get_decisions(project_id)
                if _decisions_hash(existing) != decision_hash:
                    raise PersistenceConflict("Cleaning decisions changed concurrently.")
                project_row = self.get_project(project_id, touch=False)
                if project_row is None:
                    raise PersistenceNotFound("Project does not exist.")
                return str(project_row["updated_at"])
            if persistence_result == "reused":
                self.storage.discard(staged)
                project_row = self.get_project(project_id, touch=False)
                if project_row is None:
                    raise PersistenceNotFound("Project does not exist.")
                return str(project_row["updated_at"])
            try:
                self.storage.confirm(staged)
                self._mark_object_confirmed(staged)
            except Exception as exc:
                raise ObjectConfirmationPending(
                    "Cleaned DatasetVersion is durable but requires confirmation recovery."
                ) from exc
            project_row = self.get_project(project_id, touch=False)
            if project_row is None:
                raise PersistenceNotFound("Project does not exist.")
            return str(project_row["updated_at"])
        except PersistenceError:
            raise
        except Exception as exc:
            raise _translate_database_error(exc) from exc

    def _persist_cleaning_result(
        self,
        *,
        project_id: str,
        quality_report_id: UUID,
        decisions: list[dict[str, str]],
        decisions_hash: str,
        version_id: UUID,
        staged: StagedObject,
        schema_document: dict[str, Any],
        preview: dict[str, Any],
        quality: dict[str, Any],
        row_count: int,
        column_count: int,
    ) -> str:
        now = _now()
        with self._uow() as uow:
            assert uow.session is not None
            project = uow.projects.get_project(project_id, for_update=True)
            if project is None:
                raise PersistenceNotFound("Project does not exist.")
            current = uow.projects.current_revision(project)
            report = uow.projects.current_quality_report(project)
            current_set = uow.projects.current_decision_set(project)
            current_chart = uow.projects.current_chart_revision(project)
            if report is None or report.id != quality_report_id:
                raise PersistenceConflict("QualityReport changed while cleaning was running.")
            if (
                current_set is not None
                and current_set.quality_report_id == report.id
                and current_set.decisions_hash == decisions_hash
            ):
                return "noop"
            if current is None or current_chart is None:
                raise PersistenceConflict("Ready project lineage is incomplete.")
            input_version = report.dataset_version
            findings = {
                finding.external_id: finding
                for finding in uow.session.scalars(
                    select(QualityFindingRecord).where(
                        QualityFindingRecord.quality_report_id == report.id
                    )
                )
            }
            if any(item["findingId"] not in findings for item in decisions):
                raise PersistenceConflict("Cleaning decision refers to another QualityReport.")

            decision_number = (
                int(
                    uow.session.scalar(
                        select(func.max(CleaningDecisionSet.revision_number)).where(
                            CleaningDecisionSet.project_id == project.id
                        )
                    )
                    or 0
                )
                + 1
            )
            decision_set = CleaningDecisionSet(
                id=uuid4(),
                project_id=project.id,
                quality_report_id=report.id,
                input_dataset_version_id=input_version.id,
                created_by_user_id=project.owner_user_id,
                revision_number=decision_number,
                decisions_hash=decisions_hash,
                created_at=now,
            )
            decision_rows = [
                CleaningDecisionRecord(
                    id=uuid4(),
                    project_id=project.id,
                    decision_set_id=decision_set.id,
                    quality_finding_id=findings[item["findingId"]].id,
                    action=item["action"],
                    created_at=now,
                )
                for item in decisions
            ]
            dedup_scope = self._dedup_scope(project)
            stored_object = uow.session.scalar(
                select(StoredObject)
                .where(
                    StoredObject.storage_backend == "local",
                    StoredObject.dedup_scope == dedup_scope,
                    StoredObject.purpose == "dataset",
                    StoredObject.media_type == "application/vnd.apache.parquet",
                    StoredObject.format_contract_version == "parquet-v1",
                    StoredObject.sha256 == staged.sha256,
                    StoredObject.size_bytes == staged.size_bytes,
                    StoredObject.encryption_key_id.is_(None),
                )
                .with_for_update()
            )
            reused = stored_object is not None
            if stored_object is not None and stored_object.status != "available":
                raise PersistenceConflict("Matching StoredObject is not available for reuse.")
            if stored_object is None:
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
                    dedup_scope=dedup_scope,
                    format_contract_version="parquet-v1",
                    created_at=now,
                    updated_at=now,
                )
            version_number = (
                int(
                    uow.session.scalar(
                        select(func.max(DatasetVersion.version_number)).where(
                            DatasetVersion.dataset_id == input_version.dataset_id
                        )
                    )
                    or 0
                )
                + 1
            )
            version = DatasetVersion(
                id=version_id,
                project_id=project.id,
                dataset_id=input_version.dataset_id,
                parent_version_id=input_version.id,
                stored_object=stored_object,
                version_number=version_number,
                kind="cleaned",
                schema_document=schema_document,
                preview_document=preview,
                quality_document=quality,
                parquet_schema_version=PARQUET_SCHEMA_VERSION,
                content_sha256=staged.sha256,
                cleaning_decision_set=decision_set,
                row_count=row_count,
                column_count=column_count,
                created_at=now,
            )
            clean_run = ProcessingRun(
                id=uuid4(),
                project_id=project.id,
                input_dataset_version_id=input_version.id,
                operation="clean",
                status="running",
                parameters={
                    "cleaningDecisionSetId": decision_set.id.hex,
                    "decisionSetRevision": decision_number,
                    "decisionsHash": decisions_hash,
                    "actionSemantics": {
                        "ignore": "retain-everywhere",
                        "exclude": "retain-data-exclude-chart",
                        "remove": "remove-cleaned-copy-and-chart",
                    },
                    "pendingDatasetVersionId": version.id.hex,
                },
                algorithm_version=CLEANING_ALGORITHM_VERSION,
                code_version=PHASE3_CODE_VERSION,
                started_at=now,
                created_at=now,
            )
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
            chart_revision = ChartSpecRevision(
                id=uuid4(),
                project_id=project.id,
                dataset_version=version,
                created_by_user_id=project.owner_user_id,
                revision_number=chart_number,
                schema_version=current_chart.schema_version,
                decision_set_revision=decision_number,
                cleaning_decision_set=decision_set,
                spec_document=current_chart.spec_document,
                created_at=now,
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
            previous_spec = ProjectSpecV1.model_validate(current.spec_document)
            next_chart_model = ChartSpec.model_validate(current_chart.spec_document)
            next_spec = previous_spec.model_copy(
                update={
                    "source": previous_spec.source.model_copy(
                        update={"dataset_version_id": version.id}
                    ),
                    "cleaning": ProjectCleaningSpec(
                        decision_set_id=decision_set.id,
                        revision=decision_number,
                    ),
                    "chart": next_chart_model,
                }
            )
            project_revision = ProjectRevision(
                id=uuid4(),
                project_id=project.id,
                active_dataset_version=version,
                chart_spec_revision=chart_revision,
                quality_report=report,
                cleaning_decision_set=decision_set,
                created_by_user_id=project.owner_user_id,
                revision_number=project_number,
                spec_schema_version=1,
                spec_document=next_spec.model_dump(mode="json", by_alias=True),
                created_at=now,
            )
            project.current_revision = project_revision
            project.last_activity_at = now
            project.updated_at = now
            if project.storage_mode == "temporary-cloud":
                project.expires_at = now + timedelta(seconds=self.project_ttl_seconds)
            uow.session.add_all(
                [
                    decision_set,
                    *decision_rows,
                    stored_object,
                    version,
                    clean_run,
                    chart_revision,
                    project_revision,
                ]
            )
            uow.commit()
        return "reused" if reused else "created"

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
                    cleaning_decision_set_id=current.cleaning_decision_set_id,
                    decision_set_revision=(
                        current.cleaning_decision_set.revision_number
                        if current.cleaning_decision_set is not None
                        else None
                    ),
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
                    quality_report_id=current.quality_report_id,
                    cleaning_decision_set_id=current.cleaning_decision_set_id,
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

    def restore_project_revision(
        self,
        project_id: str,
        revision_number: int,
        owner_user_id: str | None = None,
    ) -> str:
        now = _now()
        try:
            with self._uow() as uow:
                assert uow.session is not None
                project = uow.projects.get_project(project_id, for_update=True)
                revision = uow.projects.get_revision(project_id, revision_number)
                if project is None or revision is None:
                    raise PersistenceNotFound("ProjectRevision does not exist.")
                if owner_user_id is not None and project.owner_user_id != _uuid(owner_user_id):
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
                project.lock_version += 1
                uow.session.add(
                    ProjectLifecycleEvent(
                        id=uuid4(),
                        project_id=project.id,
                        project_uuid_snapshot=project.id,
                        actor_user_id=project.owner_user_id,
                        event_type="revision-restore",
                        details={"revisionNumber": revision_number},
                        created_at=now,
                    )
                )
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
                if version is None:
                    raise PersistenceConflict("DatasetVersion is not available.")
                version_id = version.id
            return self._load_version_dataframe(version_id)
        except PersistenceError:
            raise
        except Exception as exc:
            raise PersistenceUnavailable("DatasetVersion could not be reopened.") from exc
