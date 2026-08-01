"""Fenced StoredObject deletion with authoritative reachability rechecks."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime
from typing import cast

from sqlalchemy import func, select

from labviz_api.storage import ObjectStorage

from .leases import LeaseStore, RetryPolicy, WorkItemLease
from .references import has_object_references
from .safety import MaintenanceSafety


class StoredObjectGarbageCollector:
    def __init__(
        self,
        storage: ObjectStorage,
        safety: MaintenanceSafety,
        retry_policy: RetryPolicy,
    ) -> None:
        self.storage = storage
        self.safety = safety
        self.retry_policy = retry_policy

    def __call__(self, lease: WorkItemLease, leases: LeaseStore) -> None:
        if not self.safety.may_delete:
            return
        deleting_lease = replace(lease, expected_state="deleting")
        with leases.database.session() as session:
            stored = leases.lock_owned_item(session, lease)
            if stored is None:
                return
            if has_object_references(session, stored.id):
                if stored.status == "deleting":
                    stored.status = "available"
                    stored.deleted_at = None
                stored.gc_candidate_at = None
                stored.lease_owner = None
                stored.lease_until = None
                return
            if stored.status == "available":
                stored.status = "deleting"
            if stored.status != "deleting":
                stored.lease_owner = None
                stored.lease_until = None
                return
            object_key = stored.object_key

        try:
            self.storage.delete(object_key)
        except PermissionError as exc:
            leases.quarantine_item(
                deleting_lease,
                error_code=type(exc).__name__,
                error_message=str(exc),
            )
            return
        except OSError as exc:
            leases.record_failure(
                deleting_lease,
                error_code=type(exc).__name__,
                error_message=str(exc),
                policy=self.retry_policy,
            )
            return

        with leases.database.session() as session:
            stored = leases.lock_owned_item(session, deleting_lease)
            if stored is None:
                return
            if has_object_references(session, stored.id):
                stored.status = "available"
                stored.gc_candidate_at = None
                stored.lease_owner = None
                stored.lease_until = None
                return
            now = cast(datetime, session.scalar(select(func.clock_timestamp())))
            stored.status = "deleted"
            stored.deleted_at = now
            stored.gc_candidate_at = None
            stored.lease_owner = None
            stored.lease_until = None
            stored.next_attempt_at = None
            stored.last_error_code = None
            stored.last_error_message = None
