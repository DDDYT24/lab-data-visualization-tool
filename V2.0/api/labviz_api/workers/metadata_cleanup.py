"""Metadata cleanup boundary for Phase 5B-2 activation."""

from .leases import LeaseStore, TaskLease


def inspect_metadata(_lease: TaskLease, _store: LeaseStore) -> int:
    """Do not mutate authentication or idempotency metadata in Phase 5B-1."""

    return 0
