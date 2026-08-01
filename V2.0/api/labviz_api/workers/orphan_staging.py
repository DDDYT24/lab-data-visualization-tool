"""Two-pass, task-fenced inventory and cleanup of orphan staging objects."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import cast

from sqlalchemy import func, select

from labviz_api.db.models import OrphanStagingCandidate
from labviz_api.storage import ObjectInfo, ObjectStorage

from .leases import LeaseStore, RetryPolicy, TaskLease
from .references import staging_key_has_database_owner
from .safety import MaintenanceSafety


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
        inventory = self.storage.list_staged()
        eligible: list[ObjectInfo] = []
        candidate_count = 0
        seen = {item.key for item in inventory}
        with leases.database.session() as session:
            if leases.lock_owned_task(session, lease) is None:
                return 0
            now = cast(datetime, session.scalar(select(func.clock_timestamp())))
            for item in inventory:
                candidate = session.get(OrphanStagingCandidate, item.key)
                if staging_key_has_database_owner(session, item.key):
                    if candidate is not None:
                        session.delete(candidate)
                    continue
                candidate_count += 1
                if candidate is None:
                    session.add(
                        OrphanStagingCandidate(
                            staging_key=item.key,
                            size_bytes=item.size_bytes,
                            sha256=item.sha256,
                            first_seen_at=now,
                            last_seen_at=now,
                            observation_count=1,
                        )
                    )
                    continue
                if candidate.sha256 != item.sha256 or candidate.size_bytes != item.size_bytes:
                    candidate.sha256 = item.sha256
                    candidate.size_bytes = item.size_bytes
                    candidate.first_seen_at = now
                    candidate.observation_count = 1
                    candidate.quarantined_at = None
                    candidate.retry_count = 0
                else:
                    candidate.observation_count += 1
                candidate.last_seen_at = now
                if (
                    candidate.observation_count >= 2
                    and candidate.first_seen_at <= now - timedelta(seconds=self.grace_seconds)
                    and candidate.quarantined_at is None
                    and (candidate.next_attempt_at is None or candidate.next_attempt_at <= now)
                ):
                    eligible.append(item)
            missing = [
                candidate
                for candidate in session.scalars(select(OrphanStagingCandidate))
                if candidate.staging_key not in seen
            ]
            for candidate in missing:
                session.delete(candidate)

        if not self.safety.may_delete:
            return candidate_count
        deleted = 0
        for item in eligible[: self.batch_size]:
            try:
                self.storage.delete(item.key)
            except PermissionError as exc:
                self._record_failure(lease, leases, item.key, exc, permanent=True)
                continue
            except OSError as exc:
                self._record_failure(lease, leases, item.key, exc, permanent=False)
                continue
            with leases.database.session() as session:
                if leases.lock_owned_task(session, lease) is None:
                    return deleted
                candidate = session.get(OrphanStagingCandidate, item.key)
                if candidate is not None and not staging_key_has_database_owner(session, item.key):
                    session.delete(candidate)
                    deleted += 1
        return deleted

    def _record_failure(
        self,
        lease: TaskLease,
        leases: LeaseStore,
        staging_key: str,
        error: OSError,
        *,
        permanent: bool,
    ) -> None:
        with leases.database.session() as session:
            if leases.lock_owned_task(session, lease) is None:
                return
            candidate = session.get(OrphanStagingCandidate, staging_key)
            if candidate is None:
                return
            now = cast(datetime, session.scalar(select(func.clock_timestamp())))
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
