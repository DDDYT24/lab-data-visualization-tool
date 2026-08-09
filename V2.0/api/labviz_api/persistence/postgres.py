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
    ExportJobRecord,
    GuestSession,
    IdempotencyRecord,
    ProcessingRun,
    Project,
    ProjectClaim,
    ProjectLifecycleEvent,
    ProjectOrigin,
    ProjectRevision,
    PublicationExport,
    QualityFindingRecord,
    QualityReportRecord,
    ShareExportBinding,
    ShareLinkEvent,
    ShareLinkRecord,
    SourceFile,
    StoredObject,
    StoredObjectWriteIntent,
    User,
)
from labviz_api.db.session import Database
from labviz_api.models import ChartSpec
from labviz_api.parquet import (
    PARQUET_SCHEMA_VERSION,
    read_parquet,
    storage_provenance_metadata,
    write_parquet,
)
from labviz_api.processing import apply_chart_decisions, build_preview
from labviz_api.project_spec import (
    ProjectCleaningSpec,
    ProjectSourceSpec,
    ProjectSpecV1,
)
from labviz_api.repository import iso_at
from labviz_api.share_tokens import ShareTokenCodec
from labviz_api.storage import ObjectStorage, StagedObject
from labviz_api.workers.references import (
    collect_project_object_ids,
    finalize_purged_project_objects,
    lock_staging_key,
    staging_key_deletion_claimed,
)

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
PHASE5_CODE_VERSION = "v2-phase5a"
EXPORT_RENDERER_NAME = "labviz-matplotlib"
EXPORT_RENDERER_VERSION = "1"
EXPORT_CONTRACT_VERSION = "publication-export-v1"
API_CONTRACT_VERSION = "api-v1"
MAX_PERSISTED_FINDING_REFS = 100
EXPORT_MEDIA_TYPES = {
    "png": "image/png",
    "svg": "image/svg+xml",
    "pdf": "application/pdf",
}


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


def _export_storage_key(dedup_scope: str, sha256: str, format_name: str) -> str:
    scope_hash = hashlib.sha256(dedup_scope.encode("utf-8")).hexdigest()[:32]
    return f"exports/{EXPORT_CONTRACT_VERSION}/{scope_hash}/{sha256}.{format_name}"


def _canonical_sha256(document: dict[str, Any]) -> str:
    payload = json.dumps(
        document,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _export_request_sha256(
    *,
    actor_user_id: UUID | None,
    guest_session_id: UUID | None,
    project_id: UUID,
    project_revision_id: UUID,
    chart_document: dict[str, Any],
) -> str:
    if actor_user_id is not None:
        actor = {"kind": "user", "id": actor_user_id.hex}
    elif guest_session_id is not None:
        actor = {"kind": "guest-session", "id": guest_session_id.hex}
    else:
        raise ValueError("An export request requires exactly one actor.")
    return _canonical_sha256(
        {
            "actor": actor,
            "projectId": project_id.hex,
            "projectRevisionId": project_revision_id.hex,
            "chart": chart_document,
            "apiContractVersion": API_CONTRACT_VERSION,
            "exportContractVersion": EXPORT_CONTRACT_VERSION,
        }
    )


def _validate_export_payload(payload: bytes, format_name: str) -> dict[str, Any]:
    if not payload:
        raise PersistenceConflict("The rendered export is empty.")
    valid = False
    if format_name == "png":
        valid = payload.startswith(b"\x89PNG\r\n\x1a\n")
    elif format_name == "pdf":
        valid = payload.startswith(b"%PDF-")
    elif format_name == "svg":
        try:
            valid = "<svg" in payload[:1024].decode("utf-8").lower()
        except UnicodeDecodeError:
            valid = False
    if not valid:
        raise PersistenceConflict(
            f"Rendered bytes do not match the requested {format_name} format."
        )
    return {"signatureValidated": True, "validatorVersion": "export-signature-v1"}


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
            for key in (
                "rowIdsTruncated",
                "validMinimum",
                "validMaximum",
                "summaryCode",
                "summaryParams",
                "reasonCode",
                "reasonParams",
            )
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
        share_tokens: ShareTokenCodec | None = None,
    ) -> None:
        self.database = database
        self.storage = storage
        self.project_ttl_seconds = project_ttl_seconds
        self.guest_session_ttl_seconds = guest_session_ttl_seconds
        self.share_tokens = share_tokens or ShareTokenCodec.from_strings(
            ((1, "labviz-development-share-token-key-v1"),), 1
        )

    def _uow(self) -> SqlAlchemyUnitOfWork:
        return SqlAlchemyUnitOfWork(self.database)

    def dispose(self) -> None:
        self.database.dispose()

    def ping(self) -> bool:
        return self.database.health().ready

    def _discard_staged_best_effort(self, staged: StagedObject) -> None:
        """Do not turn a committed result or the original failure into a cleanup failure."""
        try:
            self.storage.discard(staged)
        except Exception:
            # Phase 5B's orphan-staging cleanup owns eventual removal.
            return

    def _protect_staged_write(self, session: Session, staged: StagedObject) -> None:
        lock_staging_key(session, self.storage.backend_name, staged.staging_key)
        if staging_key_deletion_claimed(
            session,
            backend_name=self.storage.backend_name,
            inventory_scope=self.storage.inventory_scope,
            staging_key=staged.staging_key,
        ):
            raise PersistenceConflict("Staged object is already claimed for orphan cleanup.")

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
        # Compatibility entry point for explicit callers. PostgreSQL FastAPI lifespan
        # never invokes it; the same leased handlers are used by the independent CLI.
        from labviz_api.workers.garbage_collection import StoredObjectGarbageCollector
        from labviz_api.workers.leases import (
            METADATA_CLEANUP,
            PROJECT_LIFECYCLE,
            STORED_OBJECT_GC,
            LeaseStore,
            RetryPolicy,
        )
        from labviz_api.workers.lifecycle import ProjectLifecycleHandler
        from labviz_api.workers.metadata_cleanup import MetadataCleanupHandler
        from labviz_api.workers.runner import RunnerConfig, WorkerRunner
        from labviz_api.workers.safety import MaintenanceSafety

        leases = LeaseStore(self.database, storage_backend=self.storage.backend_name)
        safety = MaintenanceSafety(dry_run=False, delete_enabled=True)
        config = RunnerConfig(
            batch_size=100,
            lease_seconds=60,
            heartbeat_seconds=20,
            retry_policy=RetryPolicy(),
            dry_run=False,
            delete_enabled=True,
        )
        owner_prefix = f"compat-cleanup-{uuid4()}"
        lifecycle = WorkerRunner(
            task=PROJECT_LIFECYCLE,
            owner=f"{owner_prefix}-projects",
            leases=leases,
            config=config,
            item_handler=ProjectLifecycleHandler(safety),
        )
        while lifecycle.run_once() > 0:
            pass
        gc = WorkerRunner(
            task=STORED_OBJECT_GC,
            owner=f"{owner_prefix}-objects",
            leases=leases,
            config=config,
            item_handler=StoredObjectGarbageCollector(self.storage, safety, config.retry_policy),
        )
        while gc.run_once() > 0:
            pass
        metadata = WorkerRunner(
            task=METADATA_CLEANUP,
            owner=f"{owner_prefix}-metadata",
            leases=leases,
            config=config,
            scanner_handler=MetadataCleanupHandler(safety, batch_size=config.batch_size),
        )
        while metadata.run_once() > 0:
            pass

    def _purge_project(self, project_id: UUID, *, force_temporary: bool = False) -> bool:
        object_ids: list[UUID] = []
        with self._uow() as uow:
            assert uow.session is not None
            project = uow.projects.get_project(
                project_id.hex, for_update=True, include_deleted=True
            )
            now = _now()
            if project is None:
                uow.commit()
                return False
            eligible = (
                project.storage_mode == "temporary-cloud"
                and project.expires_at is not None
                and (force_temporary or project.expires_at <= now)
            ) or (
                project.deleted_at is not None
                and project.purge_after is not None
                and project.purge_after <= now
            )
            if not eligible:
                uow.commit()
                return False
            object_ids = list(collect_project_object_ids(uow.session, project.id))
            stored_objects: list[StoredObject] = []
            if object_ids:
                stored_objects = list(
                    uow.session.scalars(
                        select(StoredObject)
                        .where(StoredObject.id.in_(object_ids))
                        .with_for_update()
                    )
                )
                for stored_object in stored_objects:
                    stored_object.gc_candidate_at = now
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
            finalize_purged_project_objects(uow.session, stored_objects, now)
            uow.commit()
            return True

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
            "current_revision_id": (
                _id(project.current_revision_id) if project.current_revision_id else None
            ),
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
            if source_object["storage_backend"] != self.storage.backend_name:
                raise PersistenceUnavailable(
                    "Source DatasetVersion belongs to another object storage backend."
                )
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
                    metadata={
                        "labviz-format-version": str(source_object["format_contract_version"]),
                        "labviz-media-type": str(source_object["media_type"]),
                    },
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
                    select(StoredObject.id).where(
                        StoredObject.storage_backend == self.storage.backend_name,
                        StoredObject.object_key == target_key,
                    )
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
                    "shares": [
                        self._share_row(share)
                        for share in uow.session.scalars(
                            select(ShareLinkRecord)
                            .where(
                                ShareLinkRecord.project_id == project.id,
                                ShareLinkRecord.status == "active",
                                or_(
                                    ShareLinkRecord.expires_at.is_(None),
                                    ShareLinkRecord.expires_at > _now(),
                                ),
                            )
                            .order_by(ShareLinkRecord.created_at)
                        )
                    ],
                }
        except Exception as exc:
            raise _translate_database_error(exc) from exc

    def _share_row(self, share: ShareLinkRecord) -> dict[str, Any]:
        return {
            "token": self.share_tokens.issue(share.id, share.token_key_version),
            "project_id": _id(share.project_id),
            "project_revision_id": _id(share.project_revision_id),
            "downloads_enabled": share.downloads_enabled,
            "created_at": iso_at(share.created_at),
        }

    def _load_share_by_token(
        self,
        session: Session,
        token: str,
        *,
        for_update: bool = False,
        require_active: bool = True,
    ) -> tuple[ShareLinkRecord, Project] | None:
        parsed_id = self.share_tokens.parse_public_id(token)
        lookup_id = parsed_id or UUID(int=0)
        statement = select(ShareLinkRecord).where(ShareLinkRecord.id == lookup_id)
        share = session.scalar(statement)
        key_version = (
            share.token_key_version if share is not None else self.share_tokens.current_key_version
        )
        stored_digest = share.token_digest if share is not None else "0" * 64
        verified = self.share_tokens.verify(
            token,
            public_id=lookup_id,
            key_version=key_version,
            stored_digest=stored_digest,
        )
        if share is None or not verified:
            return None
        if for_update:
            project = session.scalar(
                select(Project).where(Project.id == share.project_id).with_for_update()
            )
            share = session.scalar(
                select(ShareLinkRecord).where(ShareLinkRecord.id == share.id).with_for_update()
            )
            if share is None or not self.share_tokens.verify(
                token,
                public_id=share.id,
                key_version=share.token_key_version,
                stored_digest=share.token_digest,
            ):
                return None
        else:
            project = session.get(Project, share.project_id)
        now = _now()
        unavailable = bool(
            project is None
            or project.deleted_at is not None
            or (
                project.storage_mode == "temporary-cloud"
                and project.expires_at is not None
                and project.expires_at <= now
            )
            or (share.expires_at is not None and share.expires_at <= now)
            or (require_active and share.status != "active")
        )
        if unavailable or project is None:
            return None
        return share, project

    @staticmethod
    def _share_owner_matches(project: Project, owner_user_id: str) -> bool:
        return bool(
            project.storage_mode == "saved-cloud"
            and project.owner_user_id == _uuid(owner_user_id)
            and project.deleted_at is None
        )

    @staticmethod
    def _bind_existing_exports(
        session: Session,
        share: ShareLinkRecord,
        now: datetime,
    ) -> None:
        if not share.downloads_enabled:
            return
        bound_formats = set(
            session.scalars(
                select(ShareExportBinding.format).where(
                    ShareExportBinding.share_link_id == share.id
                )
            )
        )
        for format_name in EXPORT_MEDIA_TYPES:
            if format_name in bound_formats:
                continue
            publication = session.scalar(
                select(PublicationExport)
                .where(
                    PublicationExport.project_id == share.project_id,
                    PublicationExport.project_revision_id == share.project_revision_id,
                    PublicationExport.format == format_name,
                )
                .order_by(PublicationExport.created_at, PublicationExport.id)
                .limit(1)
            )
            if publication is not None:
                session.add(
                    ShareExportBinding(
                        share_link_id=share.id,
                        format=format_name,
                        project_id=share.project_id,
                        project_revision_id=share.project_revision_id,
                        publication_export_id=publication.id,
                        created_at=now,
                    )
                )

    @staticmethod
    def _bind_export_to_shares(
        session: Session,
        publication: PublicationExport,
        now: datetime,
    ) -> None:
        shares = list(
            session.scalars(
                select(ShareLinkRecord)
                .where(
                    ShareLinkRecord.project_id == publication.project_id,
                    ShareLinkRecord.project_revision_id == publication.project_revision_id,
                    ShareLinkRecord.status == "active",
                    ShareLinkRecord.downloads_enabled.is_(True),
                    or_(
                        ShareLinkRecord.expires_at.is_(None),
                        ShareLinkRecord.expires_at > now,
                    ),
                )
                .with_for_update()
            )
        )
        for share in shares:
            existing = session.get(ShareExportBinding, (share.id, publication.format))
            if existing is None:
                session.add(
                    ShareExportBinding(
                        share_link_id=share.id,
                        format=publication.format,
                        project_id=share.project_id,
                        project_revision_id=share.project_revision_id,
                        publication_export_id=publication.id,
                        created_at=now,
                    )
                )

    def create_share(
        self,
        *,
        project_id: str,
        owner_user_id: str,
        downloads_enabled: bool,
    ) -> dict[str, Any]:
        now = _now()
        try:
            with self._uow() as uow:
                assert uow.session is not None
                project = uow.projects.get_project(project_id, for_update=True)
                if (
                    project is None
                    or project.current_revision_id is None
                    or not self._share_owner_matches(project, owner_user_id)
                ):
                    raise PersistenceNotFound("Shareable project does not exist.")
                public_id = uuid4()
                key_version = self.share_tokens.current_key_version
                token = self.share_tokens.issue(public_id, key_version)
                share = ShareLinkRecord(
                    id=public_id,
                    project_id=project.id,
                    project_revision_id=project.current_revision_id,
                    created_by_user_id=_uuid(owner_user_id),
                    token_digest=self.share_tokens.digest(token),
                    token_key_version=key_version,
                    downloads_enabled=downloads_enabled,
                    status="active",
                    created_at=now,
                    updated_at=now,
                )
                uow.session.add(share)
                uow.session.flush()
                self._bind_existing_exports(uow.session, share, now)
                uow.session.add(
                    ShareLinkEvent(
                        id=uuid4(),
                        share_link_id=share.id,
                        share_uuid_snapshot=share.id,
                        project_uuid_snapshot=project.id,
                        project_revision_uuid_snapshot=project.current_revision_id,
                        actor_user_id=_uuid(owner_user_id),
                        event_type="create",
                        details={"downloadsEnabled": downloads_enabled},
                        created_at=now,
                    )
                )
                uow.commit()
                return self._share_row(share)
        except Exception as exc:
            raise _translate_database_error(exc) from exc

    def update_share(
        self,
        *,
        token: str,
        project_id: str,
        owner_user_id: str,
        downloads_enabled: bool,
    ) -> dict[str, Any] | None:
        now = _now()
        try:
            with self._uow() as uow:
                assert uow.session is not None
                resolved = self._load_share_by_token(
                    uow.session, token, for_update=True, require_active=True
                )
                if resolved is None:
                    return None
                share, project = resolved
                if share.project_id != _uuid(project_id) or not self._share_owner_matches(
                    project, owner_user_id
                ):
                    return None
                share.downloads_enabled = downloads_enabled
                share.updated_at = now
                if downloads_enabled:
                    self._bind_existing_exports(uow.session, share, now)
                uow.session.add(
                    ShareLinkEvent(
                        id=uuid4(),
                        share_link_id=share.id,
                        share_uuid_snapshot=share.id,
                        project_uuid_snapshot=project.id,
                        project_revision_uuid_snapshot=share.project_revision_id,
                        actor_user_id=_uuid(owner_user_id),
                        event_type="downloads-update",
                        details={"downloadsEnabled": downloads_enabled},
                        created_at=now,
                    )
                )
                uow.commit()
                return self._share_row(share)
        except Exception as exc:
            raise _translate_database_error(exc) from exc

    def revoke_share(
        self,
        *,
        token: str,
        project_id: str,
        owner_user_id: str,
    ) -> bool:
        now = _now()
        try:
            with self._uow() as uow:
                assert uow.session is not None
                resolved = self._load_share_by_token(
                    uow.session, token, for_update=True, require_active=False
                )
                if resolved is None:
                    return False
                share, project = resolved
                if share.project_id != _uuid(project_id) or not self._share_owner_matches(
                    project, owner_user_id
                ):
                    return False
                if share.status == "revoked":
                    return True
                share.status = "revoked"
                share.revoked_at = now
                share.revoked_by_user_id = _uuid(owner_user_id)
                share.updated_at = now
                uow.session.add(
                    ShareLinkEvent(
                        id=uuid4(),
                        share_link_id=share.id,
                        share_uuid_snapshot=share.id,
                        project_uuid_snapshot=project.id,
                        project_revision_uuid_snapshot=share.project_revision_id,
                        actor_user_id=_uuid(owner_user_id),
                        event_type="revoke",
                        details={},
                        created_at=now,
                    )
                )
                uow.commit()
                return True
        except Exception as exc:
            raise _translate_database_error(exc) from exc

    def get_shared_project(self, token: str) -> tuple[dict[str, Any], pd.DataFrame] | None:
        try:
            with self._uow() as uow:
                assert uow.session is not None
                resolved = self._load_share_by_token(uow.session, token)
                if resolved is None:
                    return None
                share, project = resolved
                revision = uow.session.get(ProjectRevision, share.project_revision_id)
                if revision is None:
                    return None
                chart = uow.session.get(ChartSpecRevision, revision.chart_spec_revision_id)
                version = uow.session.get(DatasetVersion, revision.active_dataset_version_id)
                report = (
                    uow.session.get(QualityReportRecord, revision.quality_report_id)
                    if revision.quality_report_id is not None
                    else None
                )
                if chart is None or version is None:
                    return None
                decisions: list[dict[str, str]] = []
                if revision.cleaning_decision_set_id is not None:
                    decision_set = uow.session.get(
                        CleaningDecisionSet, revision.cleaning_decision_set_id
                    )
                    if decision_set is None:
                        return None
                    decisions = [
                        {
                            "findingId": decision.quality_finding.external_id,
                            "action": decision.action,
                        }
                        for decision in decision_set.decisions
                    ]
                formats = list(
                    uow.session.scalars(
                        select(ShareExportBinding.format)
                        .where(ShareExportBinding.share_link_id == share.id)
                        .order_by(ShareExportBinding.format)
                    )
                )
                spec = ProjectSpecV1.model_validate(revision.spec_document)
                quality_document: dict[str, Any] = (
                    report.report_document if report is not None else version.quality_document
                )
                context = {
                    "token": token,
                    "project_id": _id(project.id),
                    "title": str(chart.spec_document.get("title") or spec.title),
                    "description": project.description,
                    "updated_at": iso_at(revision.created_at),
                    "chart": chart.spec_document,
                    "preview": version.preview_document,
                    "quality": quality_document,
                    "decisions": decisions,
                    "downloads_enabled": share.downloads_enabled,
                    "download_formats": formats if share.downloads_enabled else [],
                    "dataset_version_id": _id(version.id),
                }
                version_id = version.id
            frame = self._load_version_dataframe(version_id)
            if quality_document:
                frame = apply_chart_decisions(
                    frame,
                    quality_document,
                    decisions,
                )
            return context, frame
        except PersistenceError:
            raise
        except Exception as exc:
            raise _translate_database_error(exc) from exc

    @staticmethod
    def _authorize_export_actor(
        project: Project,
        *,
        owner_user_id: str | None,
        guest_token_digest: str | None,
        now: datetime,
    ) -> tuple[UUID | None, UUID | None]:
        if project.storage_mode == "saved-cloud":
            if owner_user_id is None or project.owner_user_id != _uuid(owner_user_id):
                raise PersistenceNotFound("Exportable project does not exist.")
            return project.owner_user_id, None
        if not PostgresProjectStore._guest_matches(project, guest_token_digest, now):
            raise PersistenceNotFound("Exportable project does not exist.")
        if project.guest_session_id is None:
            raise PersistenceConflict("Temporary project has no GuestSession owner.")
        return None, project.guest_session_id

    @staticmethod
    def _export_job_row(job: ExportJobRecord) -> dict[str, Any]:
        return {
            "id": _id(job.id),
            "project_id": _id(job.project_id),
            "project_revision_id": _id(job.project_revision_id),
            "status": job.status,
            "format": job.format,
            "expires_at": iso_at(job.expires_at) if job.expires_at else None,
            "message": job.message,
        }

    @staticmethod
    def _find_idempotency_record(
        session: Session,
        *,
        actor_user_id: UUID | None,
        guest_session_id: UUID | None,
        idempotency_key: str,
        now: datetime,
    ) -> IdempotencyRecord | None:
        actor_filter = (
            IdempotencyRecord.actor_user_id == actor_user_id
            if actor_user_id is not None
            else IdempotencyRecord.guest_session_id == guest_session_id
        )
        record = session.scalar(
            select(IdempotencyRecord)
            .where(
                actor_filter,
                IdempotencyRecord.operation == "publication-export",
                IdempotencyRecord.idempotency_key == idempotency_key,
            )
            .with_for_update()
        )
        if record is not None and record.expires_at is not None and record.expires_at <= now:
            session.delete(record)
            session.flush()
            return None
        return record

    @staticmethod
    def _revision_for_export(
        session: Session,
        project: Project,
        current: ProjectRevision,
        chart_model: ChartSpec,
        chart_document: dict[str, Any],
        now: datetime,
    ) -> ProjectRevision:
        current_chart = session.get(ChartSpecRevision, current.chart_spec_revision_id)
        if current_chart is None:
            raise PersistenceNotFound("Current ChartSpecRevision does not exist.")
        project.title = chart_model.title or "Untitled figure"
        project.last_activity_at = now
        project.updated_at = now
        current_visual = {
            key: value for key, value in current_chart.spec_document.items() if key != "export"
        }
        requested_visual = {key: value for key, value in chart_document.items() if key != "export"}
        if current_visual == requested_visual:
            return current
        chart_number = (
            int(
                session.scalar(
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
                session.scalar(
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
            created_by_user_id=project.owner_user_id,
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
        next_spec = previous_spec.model_copy(update={"title": project.title, "chart": chart_model})
        revision = ProjectRevision(
            id=uuid4(),
            project=project,
            active_dataset_version_id=current.active_dataset_version_id,
            chart_spec_revision=chart_revision,
            quality_report_id=current.quality_report_id,
            cleaning_decision_set_id=current.cleaning_decision_set_id,
            created_by_user_id=project.owner_user_id,
            revision_number=project_number,
            spec_schema_version=1,
            spec_document=next_spec.model_dump(mode="json", by_alias=True),
            created_at=now,
        )
        project.current_revision = revision
        session.add_all([chart_revision, revision])
        return revision

    @staticmethod
    def _publication_from_job(
        job: ExportJobRecord,
        run: ProcessingRun,
        stored: StoredObject,
        now: datetime,
    ) -> PublicationExport:
        parameters = run.parameters
        if run.input_dataset_version_id is None:
            raise PersistenceConflict("Export ProcessingRun has no input DatasetVersion.")
        render_spec = dict(parameters["renderSpec"])
        decision_id = parameters.get("cleaningDecisionSetId")
        return PublicationExport(
            id=job.id,
            project_id=job.project_id,
            project_revision_id=job.project_revision_id,
            dataset_version_id=run.input_dataset_version_id,
            cleaning_decision_set_id=_uuid(decision_id) if decision_id else None,
            chart_spec_revision_id=_uuid(parameters["chartSpecRevisionId"]),
            processing_run_id=run.id,
            stored_object_id=stored.id,
            format=job.format,
            media_type=parameters["mediaType"],
            renderer_name=parameters["rendererName"],
            renderer_version=parameters["rendererVersion"],
            render_contract_version=parameters["renderContractVersion"],
            render_spec_document=render_spec,
            size_preset=render_spec["sizePreset"],
            width=render_spec.get("width"),
            height=render_spec.get("height"),
            unit=render_spec["unit"],
            dpi=render_spec["dpi"],
            output_sha256=stored.sha256,
            output_size_bytes=stored.size_bytes,
            validation_document=dict(parameters["validation"]),
            created_at=now,
        )

    def _complete_export_intent(
        self,
        session: Session,
        stored: StoredObject,
        now: datetime,
    ) -> ExportJobRecord | None:
        intent = session.scalar(
            select(StoredObjectWriteIntent)
            .where(
                StoredObjectWriteIntent.stored_object_id == stored.id,
                StoredObjectWriteIntent.status == "pending",
            )
            .with_for_update()
        )
        if intent is None:
            return None
        job = session.scalar(
            select(ExportJobRecord)
            .where(ExportJobRecord.id == intent.export_job_id)
            .with_for_update()
        )
        if job is None or job.current_processing_run_id is None:
            raise PersistenceNotFound("Pending export has no ExportJob or ProcessingRun.")
        run = session.get(ProcessingRun, job.current_processing_run_id)
        if run is None:
            raise PersistenceNotFound("Pending export ProcessingRun does not exist.")
        stored.status = "available"
        stored.staging_key = None
        stored.updated_at = now
        session.flush()
        publication = self._publication_from_job(job, run, stored, now)
        session.add(publication)
        session.flush()
        intent.status = "completed"
        intent.completed_at = now
        job.pending_stored_object_id = None
        job.status = "ready"
        job.message = "Your publication-ready figure is ready to download."
        job.finished_at = now
        job.updated_at = now
        run.status = "succeeded"
        run.finished_at = now
        run.error_code = None
        run.error_message = None
        self._bind_export_to_shares(session, publication, now)
        return job

    def _finalize_export_object(self, staged: StagedObject) -> dict[str, Any]:
        now = _now()
        with self._uow() as uow:
            assert uow.session is not None
            stored = uow.session.scalar(
                select(StoredObject)
                .where(
                    StoredObject.storage_backend == self.storage.backend_name,
                    StoredObject.object_key == staged.key,
                    StoredObject.sha256 == staged.sha256,
                    StoredObject.status.in_(("pending", "available")),
                )
                .with_for_update()
            )
            if stored is None:
                raise PersistenceNotFound("Pending export StoredObject does not exist.")
            job = self._complete_export_intent(uow.session, stored, now)
            if job is None:
                publication = uow.session.scalar(
                    select(PublicationExport).where(PublicationExport.stored_object_id == stored.id)
                )
                if publication is None:
                    raise PersistenceNotFound("Pending export WriteIntent does not exist.")
                job = uow.session.get(ExportJobRecord, publication.id)
                if job is None:
                    raise PersistenceNotFound("Completed ExportJob does not exist.")
            uow.commit()
            return self._export_job_row(job)

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
        if idempotency_key is not None and not 1 <= len(idempotency_key) <= 255:
            raise PersistenceConflict("Idempotency-Key must contain between 1 and 255 characters.")
        chart_model = ChartSpec.model_validate(chart)
        chart_document = chart_model.model_dump(mode="json", by_alias=True)
        export_spec = dict(chart_document["export"])
        format_name = str(export_spec["format"])
        media_type = EXPORT_MEDIA_TYPES[format_name]
        validation = _validate_export_payload(payload, format_name)
        output_sha = hashlib.sha256(payload).hexdigest()
        now = _now()
        try:
            with self._uow() as uow:
                assert uow.session is not None
                project = uow.projects.get_project(project_id)
                if project is None or project.current_revision_id is None:
                    raise PersistenceNotFound("Exportable project does not exist.")
                actor_user_id, guest_session_id = self._authorize_export_actor(
                    project,
                    owner_user_id=owner_user_id,
                    guest_token_digest=guest_token_digest,
                    now=now,
                )
                if idempotency_key is not None:
                    record = self._find_idempotency_record(
                        uow.session,
                        actor_user_id=actor_user_id,
                        guest_session_id=guest_session_id,
                        idempotency_key=idempotency_key,
                        now=now,
                    )
                    if record is not None:
                        job = uow.session.get(ExportJobRecord, record.resource_id)
                        if job is None:
                            raise PersistenceConflict(
                                "Idempotency record refers to a missing ExportJob."
                            )
                        request_sha = _export_request_sha256(
                            actor_user_id=actor_user_id,
                            guest_session_id=guest_session_id,
                            project_id=project.id,
                            project_revision_id=job.project_revision_id,
                            chart_document=chart_document,
                        )
                        if record.request_sha256 != request_sha:
                            raise PersistenceConflict(
                                "Idempotency-Key was already used with a different request."
                            )
                        return self._export_job_row(job)
                    uow.commit()
                dedup_scope = self._dedup_scope(project)
            object_key = _export_storage_key(dedup_scope, output_sha, format_name)
            staged = self.storage.stage(
                object_key,
                io.BytesIO(payload),
                expected_sha256=output_sha,
                metadata={
                    "labviz-format-version": EXPORT_CONTRACT_VERSION,
                    "labviz-media-type": media_type,
                },
            )
            pending_committed = False
            try:
                with self._uow() as uow:
                    assert uow.session is not None
                    project = uow.projects.get_project(project_id, for_update=True)
                    if project is None or project.current_revision_id is None:
                        raise PersistenceNotFound("Exportable project does not exist.")
                    actor_user_id, guest_session_id = self._authorize_export_actor(
                        project,
                        owner_user_id=owner_user_id,
                        guest_token_digest=guest_token_digest,
                        now=now,
                    )
                    if idempotency_key is not None:
                        record = self._find_idempotency_record(
                            uow.session,
                            actor_user_id=actor_user_id,
                            guest_session_id=guest_session_id,
                            idempotency_key=idempotency_key,
                            now=now,
                        )
                        if record is not None:
                            job = uow.session.get(ExportJobRecord, record.resource_id)
                            if job is None:
                                raise PersistenceConflict(
                                    "Idempotency record refers to a missing ExportJob."
                                )
                            request_sha = _export_request_sha256(
                                actor_user_id=actor_user_id,
                                guest_session_id=guest_session_id,
                                project_id=project.id,
                                project_revision_id=job.project_revision_id,
                                chart_document=chart_document,
                            )
                            if record.request_sha256 != request_sha:
                                raise PersistenceConflict(
                                    "Idempotency-Key was already used with a different request."
                                )
                            uow.commit()
                            self._discard_staged_best_effort(staged)
                            return self._export_job_row(job)
                    if expected_revision_id is not None and project.current_revision_id != _uuid(
                        expected_revision_id
                    ):
                        raise PersistenceConflict(
                            "Project changed while the export was being rendered. Retry the export."
                        )
                    current = uow.session.get(ProjectRevision, project.current_revision_id)
                    if current is None:
                        raise PersistenceNotFound("Current ProjectRevision does not exist.")
                    revision = self._revision_for_export(
                        uow.session,
                        project,
                        current,
                        chart_model,
                        chart_document,
                        now,
                    )
                    uow.session.flush()
                    request_sha = _export_request_sha256(
                        actor_user_id=actor_user_id,
                        guest_session_id=guest_session_id,
                        project_id=project.id,
                        project_revision_id=revision.id,
                        chart_document=chart_document,
                    )
                    run_id = uuid4()
                    job_id = uuid4()
                    run = ProcessingRun(
                        id=run_id,
                        project_id=project.id,
                        input_dataset_version_id=revision.active_dataset_version_id,
                        operation="export",
                        status="running",
                        parameters={
                            "chartSpecRevisionId": revision.chart_spec_revision_id.hex,
                            "cleaningDecisionSetId": (
                                revision.cleaning_decision_set_id.hex
                                if revision.cleaning_decision_set_id is not None
                                else None
                            ),
                            "renderSpec": export_spec,
                            "rendererName": EXPORT_RENDERER_NAME,
                            "rendererVersion": EXPORT_RENDERER_VERSION,
                            "renderContractVersion": EXPORT_CONTRACT_VERSION,
                            "mediaType": media_type,
                            "validation": validation,
                            "outputSha256": output_sha,
                            "outputSizeBytes": len(payload),
                        },
                        algorithm_version=EXPORT_RENDERER_VERSION,
                        code_version=PHASE5_CODE_VERSION,
                        execution_mode="executed",
                        started_at=now,
                        created_at=now,
                    )
                    uow.session.add(run)
                    uow.session.flush()
                    candidate = uow.session.scalar(
                        select(StoredObject)
                        .where(
                            StoredObject.storage_backend == self.storage.backend_name,
                            StoredObject.dedup_scope == dedup_scope,
                            StoredObject.purpose == "export",
                            StoredObject.media_type == media_type,
                            StoredObject.format_contract_version == EXPORT_CONTRACT_VERSION,
                            StoredObject.sha256 == output_sha,
                            StoredObject.size_bytes == len(payload),
                            StoredObject.encryption_key_id.is_(None),
                            StoredObject.status == "available",
                        )
                        .with_for_update()
                    )
                    job = ExportJobRecord(
                        id=job_id,
                        project_id=project.id,
                        project_revision_id=revision.id,
                        requested_by_user_id=actor_user_id,
                        guest_session_id=guest_session_id,
                        current_processing_run_id=run.id,
                        pending_stored_object_id=None,
                        status="rendering",
                        format=format_name,
                        request_sha256=request_sha,
                        message="Rendering the publication-ready figure.",
                        attempt_count=1,
                        expires_at=(
                            project.expires_at
                            if project.storage_mode == "temporary-cloud"
                            else None
                        ),
                        created_at=now,
                        updated_at=now,
                    )
                    uow.session.add(job)
                    uow.session.flush()
                    if idempotency_key is not None:
                        uow.session.add(
                            IdempotencyRecord(
                                id=uuid4(),
                                actor_user_id=actor_user_id,
                                guest_session_id=guest_session_id,
                                operation="publication-export",
                                idempotency_key=idempotency_key,
                                request_sha256=request_sha,
                                resource_id=job.id,
                                response_document={"exportJobId": job.id.hex},
                                created_at=now,
                                expires_at=now + timedelta(hours=24),
                            )
                        )
                    if candidate is not None:
                        run.status = "succeeded"
                        run.finished_at = now
                        job.status = "ready"
                        job.message = "Your publication-ready figure is ready to download."
                        job.finished_at = now
                        publication = self._publication_from_job(job, run, candidate, now)
                        uow.session.add(publication)
                        uow.session.flush()
                        self._bind_export_to_shares(uow.session, publication, now)
                        uow.commit()
                        self._discard_staged_best_effort(staged)
                        return self._export_job_row(job)
                    self._protect_staged_write(uow.session, staged)
                    existing_key = uow.session.scalar(
                        select(StoredObject.id).where(
                            StoredObject.storage_backend == self.storage.backend_name,
                            StoredObject.object_key == object_key,
                        )
                    )
                    if existing_key is not None:
                        raise PersistenceConflict(
                            "An identical export object is still awaiting confirmation."
                        )
                    stored = StoredObject(
                        id=uuid4(),
                        storage_backend=self.storage.backend_name,
                        object_key=staged.key,
                        staging_key=staged.staging_key,
                        purpose="export",
                        status="pending",
                        media_type=media_type,
                        size_bytes=staged.size_bytes,
                        sha256=staged.sha256,
                        dedup_scope=dedup_scope,
                        format_contract_version=EXPORT_CONTRACT_VERSION,
                        expires_at=job.expires_at,
                        created_at=now,
                        updated_at=now,
                    )
                    uow.session.add(stored)
                    uow.session.flush()
                    job.pending_stored_object_id = stored.id
                    intent = StoredObjectWriteIntent(
                        id=uuid4(),
                        project_id=project.id,
                        export_job_id=job.id,
                        stored_object_id=stored.id,
                        operation="export",
                        status="pending",
                        created_at=now,
                    )
                    uow.session.add(intent)
                    uow.commit()
                    pending_committed = True
            except Exception:
                if not pending_committed:
                    self._discard_staged_best_effort(staged)
                raise
            try:
                self.storage.confirm(staged)
                return self._finalize_export_object(staged)
            except Exception as exc:
                raise ObjectConfirmationPending(
                    "The export is durable and awaiting object confirmation."
                ) from exc
        except PersistenceError:
            raise
        except Exception as exc:
            raise _translate_database_error(exc) from exc

    def get_export_metadata(self, export_id: str) -> dict[str, Any] | None:
        try:
            export_uuid = _uuid(export_id)
        except ValueError:
            return None
        try:
            with self._uow() as uow:
                assert uow.session is not None
                publication = uow.session.get(PublicationExport, export_uuid)
                if publication is None:
                    return None
                return {
                    "id": _id(publication.id),
                    "project_id": _id(publication.project_id),
                    "project_revision_id": _id(publication.project_revision_id),
                    "format": publication.format,
                    "media_type": publication.media_type,
                }
        except Exception as exc:
            raise _translate_database_error(exc) from exc

    def get_export(self, export_id: str) -> dict[str, Any] | None:
        try:
            export_uuid = _uuid(export_id)
        except ValueError:
            return None
        try:
            with self._uow() as uow:
                assert uow.session is not None
                publication = uow.session.get(PublicationExport, export_uuid)
                if publication is None:
                    return None
                stored = uow.session.get(StoredObject, publication.stored_object_id)
                if (
                    stored is None
                    or stored.status != "available"
                    or stored.storage_backend != self.storage.backend_name
                ):
                    return None
                object_key = stored.object_key
                expected_sha = publication.output_sha256
                expected_size = publication.output_size_bytes
                row: dict[str, Any] = {
                    "id": _id(publication.id),
                    "project_id": _id(publication.project_id),
                    "project_revision_id": _id(publication.project_revision_id),
                    "format": publication.format,
                    "media_type": publication.media_type,
                }
            with self.storage.open(object_key) as stream:
                payload = stream.read()
            if len(payload) != expected_size or hashlib.sha256(payload).hexdigest() != expected_sha:
                raise PersistenceUnavailable("Publication export integrity verification failed.")
            row["payload"] = payload
            return row
        except PersistenceError:
            raise
        except Exception as exc:
            raise PersistenceUnavailable("Publication export could not be reopened.") from exc

    def get_shared_export(self, token: str, format_name: str) -> dict[str, Any] | None:
        if format_name not in EXPORT_MEDIA_TYPES:
            return None
        try:
            with self._uow() as uow:
                assert uow.session is not None
                resolved = self._load_share_by_token(
                    uow.session, token, for_update=True, require_active=True
                )
                if resolved is None:
                    return None
                share, _project = resolved
                if not share.downloads_enabled:
                    return {"downloads_enabled": False, "export": None}
                binding = uow.session.get(ShareExportBinding, (share.id, format_name))
                export_id = binding.publication_export_id if binding is not None else None
                uow.commit()
            export = self.get_export(_id(export_id)) if export_id is not None else None
            return {"downloads_enabled": True, "export": export}
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
        immediate_purge_id: UUID | None = None
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
                    immediate_purge_id = project.id
                uow.commit()
            if immediate_purge_id is not None:
                return self._purge_project(immediate_purge_id, force_temporary=True)
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
            self._reconcile_project_dataset(project_id)
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
            metadata={
                "labviz-format-version": "parquet-v1",
                "labviz-media-type": "application/vnd.apache.parquet",
                **storage_provenance_metadata(artifact.provenance),
            },
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
                provenance=artifact.provenance,
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
            if not self._reconcile_staged_dataset(staged):
                raise ObjectConfirmationPending(
                    "The pending DatasetVersion is currently owned by another worker."
                )
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
        provenance: dict[str, str],
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
            parse_run.parameters = {
                **parse_run.parameters,
                "runtimeProvenance": provenance,
            }

            dedup_scope = self._dedup_scope(project)
            stored_object = uow.session.scalar(
                select(StoredObject)
                .where(
                    StoredObject.storage_backend == self.storage.backend_name,
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
                self._protect_staged_write(uow.session, staged)
                stored_object = StoredObject(
                    id=uuid4(),
                    storage_backend=self.storage.backend_name,
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
                parameters={"validRanges": [], "runtimeProvenance": provenance},
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
                parameters={"validRanges": [], "runtimeProvenance": provenance},
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

    def _reconcile_staged_dataset(self, staged: StagedObject) -> bool:
        with self.database.session() as session:
            object_id = session.scalar(
                select(StoredObject.id).where(
                    StoredObject.storage_backend == self.storage.backend_name,
                    StoredObject.object_key == staged.key,
                    StoredObject.sha256 == staged.sha256,
                    StoredObject.status == "pending",
                )
            )
        return object_id is not None and self._reconcile_dataset_object(object_id)

    def _reconcile_project_dataset(self, project_id: str) -> bool:
        with self.database.session() as session:
            object_id = session.scalar(
                select(StoredObject.id)
                .join(DatasetVersion, DatasetVersion.stored_object_id == StoredObject.id)
                .where(
                    DatasetVersion.project_id == _uuid(project_id),
                    StoredObject.status == "pending",
                )
                .order_by(DatasetVersion.created_at.desc())
                .limit(1)
            )
        return object_id is not None and self._reconcile_dataset_object(object_id)

    def _reconcile_dataset_object(self, object_id: UUID) -> bool:
        from labviz_api.workers.leases import LeaseStore
        from labviz_api.workers.reconciliation import PendingObjectReconciler

        leases = LeaseStore(self.database, storage_backend=self.storage.backend_name)
        lease = leases.claim_pending_object(
            f"request-reconciliation-{uuid4().hex}",
            object_id=object_id,
            lease_seconds=60,
        )
        if lease is None:
            return False
        try:
            PendingObjectReconciler(self)(lease, leases)
        finally:
            # Finalization clears the lease atomically. On failure, releasing it
            # allows the independent worker to retry through the same protocol.
            leases.release_item(lease)
        with self.database.session() as session:
            stored = session.get(StoredObject, object_id)
            return bool(
                stored is not None
                and stored.status == "available"
                and stored.lease_owner is None
                and stored.lease_until is None
            )

    def recover_pending_objects(self) -> int:
        from labviz_api.workers.leases import PENDING_RECONCILIATION, LeaseStore
        from labviz_api.workers.reconciliation import PendingObjectReconciler
        from labviz_api.workers.runner import RunnerConfig, WorkerRunner

        runner = WorkerRunner(
            task=PENDING_RECONCILIATION,
            owner=f"compat-reconciliation-{uuid4().hex}",
            leases=LeaseStore(self.database, storage_backend=self.storage.backend_name),
            config=RunnerConfig(batch_size=100, lease_seconds=60, heartbeat_seconds=20),
            item_handler=PendingObjectReconciler(self),
        )
        return runner.run_once()

    def _load_version_dataframe(self, version_id: UUID) -> pd.DataFrame:
        try:
            with self._uow() as uow:
                assert uow.session is not None
                version = uow.session.get(DatasetVersion, version_id)
                if (
                    version is None
                    or version.stored_object is None
                    or version.stored_object.status != "available"
                    or version.stored_object.storage_backend != self.storage.backend_name
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
                self._reconcile_project_dataset(project_id)
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
                metadata={
                    "labviz-format-version": "parquet-v1",
                    "labviz-media-type": "application/vnd.apache.parquet",
                    **storage_provenance_metadata(artifact.provenance),
                },
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
                    provenance=artifact.provenance,
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
                if not self._reconcile_staged_dataset(staged):
                    raise ObjectConfirmationPending(
                        "The pending cleaned DatasetVersion is owned by another worker."
                    )
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
        provenance: dict[str, str],
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
                    StoredObject.storage_backend == self.storage.backend_name,
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
                self._protect_staged_write(uow.session, staged)
                stored_object = StoredObject(
                    id=uuid4(),
                    storage_backend=self.storage.backend_name,
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
                    "runtimeProvenance": provenance,
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
