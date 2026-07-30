"""PostgreSQL-time task and work-item leases with fencing guarantees."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Literal, cast
from uuid import UUID

from sqlalchemy import and_, func, or_, select
from sqlalchemy.orm import Session

from labviz_api.db.models import Project, StoredObject, StoredObjectWriteIntent, WorkerLease
from labviz_api.db.session import Database

PENDING_RECONCILIATION = "pending-reconciliation"
PROJECT_LIFECYCLE = "project-lifecycle"
STORED_OBJECT_GC = "stored-object-gc"
ORPHAN_STAGING_INVENTORY = "orphan-staging-inventory"
METADATA_CLEANUP = "metadata-cleanup"

TASKS = (
    PENDING_RECONCILIATION,
    PROJECT_LIFECYCLE,
    STORED_OBJECT_GC,
    ORPHAN_STAGING_INVENTORY,
    METADATA_CLEANUP,
)

WorkItemKind = Literal["write-intent", "project", "stored-object"]


@dataclass(frozen=True)
class RetryPolicy:
    """Bounded exponential retry policy applied using the database timestamp."""

    max_retries: int = 5
    base_seconds: int = 30
    max_seconds: int = 3_600

    def __post_init__(self) -> None:
        if self.max_retries < 1:
            raise ValueError("max_retries must be at least 1")
        if self.base_seconds < 1 or self.max_seconds < self.base_seconds:
            raise ValueError("retry delays must be positive and bounded")

    def delay_after(self, retry_count: int) -> timedelta:
        seconds = min(self.base_seconds * (2 ** max(retry_count - 1, 0)), self.max_seconds)
        return timedelta(seconds=seconds)


@dataclass(frozen=True)
class TaskLease:
    task: str
    owner: str
    fencing_token: int
    lease_until: datetime


@dataclass(frozen=True)
class WorkItemLease:
    kind: WorkItemKind
    item_id: UUID
    owner: str
    fencing_token: int
    lease_until: datetime
    expected_state: str


class LeaseStore:
    """Acquire short-transaction leases and fence every completion write."""

    def __init__(self, database: Database) -> None:
        self.database = database

    @staticmethod
    def _database_now(session: Session) -> datetime:
        return cast(datetime, session.execute(select(func.clock_timestamp())).scalar_one())

    @staticmethod
    def _validate_owner(owner: str) -> None:
        if not owner or len(owner) > 255:
            raise ValueError("lease owner must contain 1 to 255 characters")

    @staticmethod
    def _validate_lease_seconds(lease_seconds: int) -> None:
        if lease_seconds < 1:
            raise ValueError("lease_seconds must be positive")

    def acquire_task(self, task: str, owner: str, lease_seconds: int) -> TaskLease | None:
        if task not in TASKS:
            raise ValueError(f"Unknown worker task: {task}")
        self._validate_owner(owner)
        self._validate_lease_seconds(lease_seconds)
        with self.database.session() as session:
            now = self._database_now(session)
            row = session.scalar(
                select(WorkerLease)
                .where(
                    WorkerLease.task == task,
                    or_(WorkerLease.lease_owner.is_(None), WorkerLease.lease_until <= now),
                )
                .with_for_update(skip_locked=True)
            )
            if row is None:
                return None
            row.lease_owner = owner
            row.lease_until = now + timedelta(seconds=lease_seconds)
            row.fencing_token += 1
            row.heartbeat_at = now
            row.updated_at = now
            session.flush()
            return TaskLease(task, owner, row.fencing_token, row.lease_until)

    def heartbeat_task(self, lease: TaskLease, lease_seconds: int) -> TaskLease | None:
        self._validate_lease_seconds(lease_seconds)
        with self.database.session() as session:
            now = self._database_now(session)
            row = session.scalar(
                select(WorkerLease)
                .where(
                    WorkerLease.task == lease.task,
                    WorkerLease.lease_owner == lease.owner,
                    WorkerLease.fencing_token == lease.fencing_token,
                    WorkerLease.lease_until > now,
                )
                .with_for_update()
            )
            if row is None:
                return None
            row.lease_until = now + timedelta(seconds=lease_seconds)
            row.heartbeat_at = now
            row.updated_at = now
            session.flush()
            return TaskLease(lease.task, lease.owner, lease.fencing_token, row.lease_until)

    def release_task(self, lease: TaskLease) -> bool:
        with self.database.session() as session:
            now = self._database_now(session)
            row = session.scalar(
                select(WorkerLease)
                .where(
                    WorkerLease.task == lease.task,
                    WorkerLease.lease_owner == lease.owner,
                    WorkerLease.fencing_token == lease.fencing_token,
                    WorkerLease.lease_until > now,
                )
                .with_for_update()
            )
            if row is None:
                return False
            row.lease_owner = None
            row.lease_until = None
            row.updated_at = now
            return True

    def claim_write_intents(
        self, owner: str, *, batch_size: int, lease_seconds: int
    ) -> list[WorkItemLease]:
        return self._claim(
            "write-intent",
            owner,
            batch_size=batch_size,
            lease_seconds=lease_seconds,
        )

    def claim_projects(
        self, owner: str, *, batch_size: int, lease_seconds: int
    ) -> list[WorkItemLease]:
        return self._claim("project", owner, batch_size=batch_size, lease_seconds=lease_seconds)

    def claim_stored_objects(
        self, owner: str, *, batch_size: int, lease_seconds: int
    ) -> list[WorkItemLease]:
        return self._claim(
            "stored-object", owner, batch_size=batch_size, lease_seconds=lease_seconds
        )

    def _claim(
        self,
        kind: WorkItemKind,
        owner: str,
        *,
        batch_size: int,
        lease_seconds: int,
    ) -> list[WorkItemLease]:
        self._validate_owner(owner)
        self._validate_lease_seconds(lease_seconds)
        if batch_size < 1:
            raise ValueError("batch_size must be positive")

        model = self._model(kind)
        with self.database.session() as session:
            now = self._database_now(session)
            statement = (
                select(model)
                .where(
                    model.quarantined_at.is_(None),
                    or_(model.next_attempt_at.is_(None), model.next_attempt_at <= now),
                    or_(model.lease_owner.is_(None), model.lease_until <= now),
                    self._eligible(kind, now),
                )
                .order_by(model.next_attempt_at.asc().nulls_first(), model.id)
                .limit(batch_size)
                .with_for_update(skip_locked=True)
            )
            rows = list(session.scalars(statement))
            claimed: list[WorkItemLease] = []
            for row in rows:
                row.lease_owner = owner
                row.lease_until = now + timedelta(seconds=lease_seconds)
                row.fencing_token += 1
                row.last_attempt_at = now
                state = self._state(kind, row)
                claimed.append(
                    WorkItemLease(
                        kind=kind,
                        item_id=row.id,
                        owner=owner,
                        fencing_token=row.fencing_token,
                        lease_until=row.lease_until,
                        expected_state=state,
                    )
                )
            session.flush()
            return claimed

    def heartbeat_item(self, lease: WorkItemLease, lease_seconds: int) -> WorkItemLease | None:
        self._validate_lease_seconds(lease_seconds)
        with self.database.session() as session:
            now = self._database_now(session)
            row = self._locked_owned_row(session, lease, now)
            if row is None:
                return None
            row.lease_until = now + timedelta(seconds=lease_seconds)
            session.flush()
            return WorkItemLease(
                lease.kind,
                lease.item_id,
                lease.owner,
                lease.fencing_token,
                row.lease_until,
                lease.expected_state,
            )

    def release_item(self, lease: WorkItemLease) -> bool:
        with self.database.session() as session:
            now = self._database_now(session)
            row = self._locked_owned_row(session, lease, now)
            if row is None:
                return False
            row.lease_owner = None
            row.lease_until = None
            return True

    def record_failure(
        self,
        lease: WorkItemLease,
        *,
        error_code: str,
        error_message: str,
        policy: RetryPolicy,
    ) -> bool:
        with self.database.session() as session:
            now = self._database_now(session)
            row = self._locked_owned_row(session, lease, now)
            if row is None:
                return False
            row.retry_count += 1
            row.last_error_code = (error_code or "worker-error")[:128]
            row.last_error_message = (error_message or "Worker operation failed.")[:1024]
            row.lease_owner = None
            row.lease_until = None
            if row.retry_count >= policy.max_retries:
                row.next_attempt_at = None
                row.quarantined_at = now
            else:
                row.next_attempt_at = now + policy.delay_after(row.retry_count)
            return True

    def _locked_owned_row(
        self, session: Session, lease: WorkItemLease, now: datetime
    ) -> Any | None:
        model = self._model(lease.kind)
        return session.scalar(
            select(model)
            .where(
                model.id == lease.item_id,
                model.lease_owner == lease.owner,
                model.fencing_token == lease.fencing_token,
                model.lease_until > now,
                self._eligible(lease.kind, now),
                self._state_expression(lease.kind, lease.expected_state),
            )
            .with_for_update()
        )

    @staticmethod
    def _model(kind: WorkItemKind) -> Any:
        if kind == "write-intent":
            return StoredObjectWriteIntent
        if kind == "project":
            return Project
        return StoredObject

    @staticmethod
    def _eligible(kind: WorkItemKind, now: datetime) -> Any:
        if kind == "write-intent":
            return StoredObjectWriteIntent.status == "pending"
        if kind == "project":
            return or_(
                and_(
                    Project.storage_mode == "temporary-cloud",
                    Project.deleted_at.is_(None),
                    Project.expires_at <= now,
                ),
                and_(
                    Project.storage_mode == "saved-cloud",
                    Project.deleted_at.is_not(None),
                    Project.purge_after <= now,
                ),
            )
        return or_(
            and_(StoredObject.status == "available", StoredObject.gc_candidate_at.is_not(None)),
            StoredObject.status == "deleting",
        )

    @staticmethod
    def _state(kind: WorkItemKind, row: Any) -> str:
        if kind == "project":
            return "temporary-expired" if row.storage_mode == "temporary-cloud" else "saved-purge"
        return str(row.status)

    @staticmethod
    def _state_expression(kind: WorkItemKind, state: str) -> Any:
        if kind == "write-intent":
            return StoredObjectWriteIntent.status == state
        if kind == "stored-object":
            return StoredObject.status == state
        if state == "temporary-expired":
            return Project.storage_mode == "temporary-cloud"
        return and_(Project.storage_mode == "saved-cloud", Project.deleted_at.is_not(None))
