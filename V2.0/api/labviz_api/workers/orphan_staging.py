"""Paged, resumable, task-fenced inventory and cleanup of orphan staging objects."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import cast
from uuid import UUID, uuid4

from sqlalchemy import func, select

from labviz_api.db.models import OrphanStagingCandidate, StorageInventoryCheckpoint
from labviz_api.storage import (
    InvalidStorageCursor,
    ObjectInfo,
    ObjectIntegrityError,
    ObjectStorage,
)

from .leases import LeaseStore, RetryPolicy, TaskLease
from .references import lock_staging_key, staging_key_has_database_owner
from .safety import MaintenanceSafety, PermanentWorkerFailure

DELETION_TOMBSTONE_SECONDS = 86_400


@dataclass(frozen=True)
class InventoryPageClaim:
    generation_id: UUID
    cursor: str | None


class OrphanStagingHandler:
    def __init__(
        self,
        storage: ObjectStorage,
        safety: MaintenanceSafety,
        retry_policy: RetryPolicy,
        *,
        grace_seconds: int,
        batch_size: int = 25,
    ) -> None:
        if grace_seconds < 1 or batch_size < 1:
            raise ValueError("orphan staging grace and batch size must be positive")
        self.storage = storage
        self.safety = safety
        self.retry_policy = retry_policy
        self.grace_seconds = grace_seconds
        self.batch_size = batch_size

    def __call__(self, lease: TaskLease, leases: LeaseStore) -> int:
        claim = self._claim_page(lease, leases)
        try:
            page = self.storage.list_staged(page_size=self.batch_size, cursor=claim.cursor)
        except InvalidStorageCursor as exc:
            self._fail_checkpoint(lease, leases, claim.generation_id, exc)
            raise PermanentWorkerFailure(
                "Provider rejected the persisted inventory cursor."
            ) from exc
        except ObjectIntegrityError as exc:
            self._fail_checkpoint(lease, leases, claim.generation_id, exc)
            raise PermanentWorkerFailure(
                "Provider staging metadata failed integrity validation."
            ) from exc
        try:
            completed, candidate_count = self._persist_page(
                lease,
                leases,
                claim.generation_id,
                page.items,
                page.next_cursor,
                page.has_more,
            )
        except PermanentWorkerFailure as exc:
            self._fail_checkpoint(lease, leases, claim.generation_id, exc)
            raise
        if not completed:
            return 0
        if not self.safety.may_delete:
            return candidate_count
        return self._delete_eligible(lease, leases)

    def _claim_page(self, lease: TaskLease, leases: LeaseStore) -> InventoryPageClaim:
        identity = (self.storage.backend_name, self.storage.inventory_scope)
        with leases.database.session() as session:
            if leases.lock_owned_task(session, lease) is None:
                raise PermanentWorkerFailure("Inventory task lease is no longer current.")
            now = cast(datetime, session.scalar(select(func.clock_timestamp())))
            checkpoint = session.get(StorageInventoryCheckpoint, identity, with_for_update=True)
            if checkpoint is not None and checkpoint.status == "failed":
                raise PermanentWorkerFailure(
                    "Inventory checkpoint is failed and requires explicit operator diagnosis."
                )
            if checkpoint is None:
                checkpoint = StorageInventoryCheckpoint(
                    backend_name=identity[0],
                    inventory_scope=identity[1],
                    generation_id=uuid4(),
                    status="running",
                    cursor=None,
                    started_at=now,
                    last_checkpoint_at=now,
                    completed_at=None,
                    lease_owner=lease.owner,
                    task_fencing_token=lease.fencing_token,
                )
                session.add(checkpoint)
            elif checkpoint.status == "completed":
                checkpoint.generation_id = uuid4()
                checkpoint.status = "running"
                checkpoint.cursor = None
                checkpoint.started_at = now
                checkpoint.last_checkpoint_at = now
                checkpoint.completed_at = None
                checkpoint.page_count = 0
                checkpoint.item_count = 0
                checkpoint.last_error_code = None
                checkpoint.last_error_message = None
            checkpoint.lease_owner = lease.owner
            checkpoint.task_fencing_token = lease.fencing_token
            session.flush()
            return InventoryPageClaim(checkpoint.generation_id, checkpoint.cursor)

    def _persist_page(
        self,
        lease: TaskLease,
        leases: LeaseStore,
        generation_id: UUID,
        items: tuple[ObjectInfo, ...],
        next_cursor: str | None,
        has_more: bool,
    ) -> tuple[bool, int]:
        identity = (self.storage.backend_name, self.storage.inventory_scope)
        with leases.database.session() as session:
            if leases.lock_owned_task(session, lease) is None:
                return False, 0
            checkpoint = session.get(StorageInventoryCheckpoint, identity, with_for_update=True)
            if not self._checkpoint_owned(checkpoint, lease, generation_id):
                return False, 0
            assert checkpoint is not None
            now = cast(datetime, session.scalar(select(func.clock_timestamp())))
            for item in items:
                if item.last_modified is None:
                    raise PermanentWorkerFailure(
                        "Provider staging inventory omitted last_modified metadata."
                    )
                candidate_id = (*identity, item.key)
                candidate = session.get(
                    OrphanStagingCandidate,
                    candidate_id,
                    with_for_update=True,
                )
                if staging_key_has_database_owner(
                    session,
                    backend_name=self.storage.backend_name,
                    staging_key=item.key,
                ):
                    if candidate is not None:
                        session.delete(candidate)
                    continue
                if candidate is None:
                    candidate = OrphanStagingCandidate(
                        backend_name=identity[0],
                        inventory_scope=identity[1],
                        staging_key=item.key,
                        size_bytes=item.size_bytes,
                        sha256=item.sha256,
                        provider_last_modified=item.last_modified,
                        provider_etag=item.etag,
                        first_seen_at=now,
                        last_seen_at=now,
                        observation_count=0,
                    )
                    session.add(candidate)
                elif candidate.deletion_completed_at is not None or not self._same_provider_object(
                    candidate, item
                ):
                    candidate.size_bytes = item.size_bytes
                    candidate.sha256 = item.sha256
                    candidate.provider_last_modified = item.last_modified
                    candidate.provider_etag = item.etag
                    candidate.first_seen_at = now
                    candidate.observation_count = 0
                    candidate.retry_count = 0
                    candidate.next_attempt_at = None
                    candidate.quarantined_at = None
                    candidate.deletion_started_at = None
                    candidate.deletion_completed_at = None
                candidate.seen_generation_id = generation_id
                candidate.last_seen_at = now
            checkpoint.cursor = next_cursor if has_more else None
            checkpoint.last_checkpoint_at = now
            checkpoint.page_count += 1
            checkpoint.item_count += len(items)
            if has_more:
                return False, 0
            session.flush()
            candidates = list(
                session.scalars(
                    select(OrphanStagingCandidate).where(
                        OrphanStagingCandidate.backend_name == identity[0],
                        OrphanStagingCandidate.inventory_scope == identity[1],
                    )
                )
            )
            candidate_count = 0
            for candidate in candidates:
                if candidate.seen_generation_id != generation_id:
                    if candidate.deletion_completed_at is not None:
                        if candidate.deletion_completed_at <= now - timedelta(
                            seconds=DELETION_TOMBSTONE_SECONDS
                        ):
                            session.delete(candidate)
                        continue
                    if candidate.deletion_started_at is not None:
                        continue
                    session.delete(candidate)
                    continue
                candidate.observation_count += 1
                candidate_count += 1
            checkpoint.status = "completed"
            checkpoint.completed_at = now
            return True, candidate_count

    def _delete_eligible(self, lease: TaskLease, leases: LeaseStore) -> int:
        identity = (self.storage.backend_name, self.storage.inventory_scope)
        with leases.database.session() as session:
            now = cast(datetime, session.scalar(select(func.clock_timestamp())))
            keys = list(
                session.scalars(
                    select(OrphanStagingCandidate.staging_key)
                    .where(
                        OrphanStagingCandidate.backend_name == identity[0],
                        OrphanStagingCandidate.inventory_scope == identity[1],
                        OrphanStagingCandidate.observation_count >= 2,
                        OrphanStagingCandidate.first_seen_at
                        <= now - timedelta(seconds=self.grace_seconds),
                        OrphanStagingCandidate.quarantined_at.is_(None),
                        OrphanStagingCandidate.deletion_completed_at.is_(None),
                        (
                            OrphanStagingCandidate.next_attempt_at.is_(None)
                            | (OrphanStagingCandidate.next_attempt_at <= now)
                        ),
                    )
                    .order_by(OrphanStagingCandidate.staging_key)
                    .limit(self.batch_size)
                )
            )
        deleted = 0
        for staging_key in keys:
            current = self.storage.head(staging_key)
            if current is None:
                if self._finish_missing(lease, leases, staging_key):
                    deleted += 1
                continue
            if not self._claim_deletion(lease, leases, current):
                continue
            try:
                self.storage.delete(staging_key, expected=current)
            except PermissionError as exc:
                self._record_failure(lease, leases, staging_key, exc, permanent=True)
                continue
            except (OSError, ValueError) as exc:
                self._record_failure(lease, leases, staging_key, exc, permanent=False)
                continue
            if self._finish_deleted(lease, leases, staging_key):
                deleted += 1
        return deleted

    def _claim_deletion(
        self,
        lease: TaskLease,
        leases: LeaseStore,
        current: ObjectInfo,
    ) -> bool:
        identity = (self.storage.backend_name, self.storage.inventory_scope)
        with leases.database.session() as session:
            if leases.lock_owned_task(session, lease) is None:
                return False
            lock_staging_key(session, self.storage.backend_name, current.key)
            candidate = session.get(
                OrphanStagingCandidate,
                (*identity, current.key),
                with_for_update=True,
            )
            if candidate is None or not self._same_provider_object(candidate, current):
                return False
            if staging_key_has_database_owner(
                session,
                backend_name=self.storage.backend_name,
                staging_key=current.key,
            ):
                session.delete(candidate)
                return False
            now = cast(datetime, session.scalar(select(func.clock_timestamp())))
            if candidate.deletion_completed_at is not None:
                return False
            if candidate.deletion_started_at is None and (
                candidate.observation_count < 2
                or candidate.first_seen_at > now - timedelta(seconds=self.grace_seconds)
                or candidate.quarantined_at is not None
            ):
                return False
            if candidate.deletion_started_at is None:
                candidate.deletion_started_at = now
            candidate.last_attempt_at = now
            return True

    def _finish_missing(self, lease: TaskLease, leases: LeaseStore, staging_key: str) -> bool:
        return self._finish_deleted(lease, leases, staging_key)

    def _finish_deleted(self, lease: TaskLease, leases: LeaseStore, staging_key: str) -> bool:
        identity = (self.storage.backend_name, self.storage.inventory_scope)
        with leases.database.session() as session:
            if leases.lock_owned_task(session, lease) is None:
                return False
            lock_staging_key(session, self.storage.backend_name, staging_key)
            candidate = session.get(
                OrphanStagingCandidate,
                (*identity, staging_key),
                with_for_update=True,
            )
            if candidate is None:
                return True
            if candidate.deletion_completed_at is not None:
                return True
            if staging_key_has_database_owner(
                session,
                backend_name=self.storage.backend_name,
                staging_key=staging_key,
            ):
                candidate.deletion_started_at = None
                candidate.deletion_completed_at = None
                return False
            now = cast(datetime, session.scalar(select(func.clock_timestamp())))
            if candidate.deletion_started_at is None:
                candidate.deletion_started_at = now
            candidate.deletion_completed_at = now
            candidate.next_attempt_at = None
            candidate.last_error_code = None
            candidate.last_error_message = None
            return True

    def _record_failure(
        self,
        lease: TaskLease,
        leases: LeaseStore,
        staging_key: str,
        error: Exception,
        *,
        permanent: bool,
    ) -> None:
        identity = (self.storage.backend_name, self.storage.inventory_scope)
        with leases.database.session() as session:
            if leases.lock_owned_task(session, lease) is None:
                return
            candidate = session.get(
                OrphanStagingCandidate,
                (*identity, staging_key),
                with_for_update=True,
            )
            if candidate is None:
                return
            if candidate.deletion_completed_at is not None:
                return
            now = cast(datetime, session.scalar(select(func.clock_timestamp())))
            candidate.deletion_started_at = None
            candidate.deletion_completed_at = None
            candidate.retry_count += 1
            candidate.last_attempt_at = now
            candidate.last_error_code = type(error).__name__[:128]
            candidate.last_error_message = str(error)[:1024]
            if permanent or candidate.retry_count >= self.retry_policy.max_retries:
                candidate.next_attempt_at = None
                candidate.quarantined_at = now
            else:
                candidate.next_attempt_at = now + self.retry_policy.delay_after(
                    candidate.retry_count
                )

    def _fail_checkpoint(
        self,
        lease: TaskLease,
        leases: LeaseStore,
        generation_id: UUID,
        error: Exception,
    ) -> None:
        identity = (self.storage.backend_name, self.storage.inventory_scope)
        with leases.database.session() as session:
            if leases.lock_owned_task(session, lease) is None:
                return
            checkpoint = session.get(StorageInventoryCheckpoint, identity, with_for_update=True)
            if not self._checkpoint_owned(checkpoint, lease, generation_id):
                return
            assert checkpoint is not None
            checkpoint.status = "failed"
            checkpoint.last_error_code = type(error).__name__[:128]
            checkpoint.last_error_message = str(error)[:1024]
            checkpoint.last_checkpoint_at = cast(
                datetime,
                session.scalar(select(func.clock_timestamp())),
            )

    @staticmethod
    def _checkpoint_owned(
        checkpoint: StorageInventoryCheckpoint | None,
        lease: TaskLease,
        generation_id: UUID,
    ) -> bool:
        return bool(
            checkpoint is not None
            and checkpoint.status == "running"
            and checkpoint.generation_id == generation_id
            and checkpoint.lease_owner == lease.owner
            and checkpoint.task_fencing_token == lease.fencing_token
        )

    @staticmethod
    def _same_provider_object(
        candidate: OrphanStagingCandidate,
        item: ObjectInfo,
    ) -> bool:
        return bool(
            item.last_modified is not None
            and candidate.size_bytes == item.size_bytes
            and candidate.sha256 == item.sha256
            and candidate.provider_last_modified == item.last_modified
            and candidate.provider_etag == item.etag
        )
