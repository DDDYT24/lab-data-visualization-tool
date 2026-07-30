"""Orphan staging inventory boundary for Phase 5B-2 activation."""

from .leases import LeaseStore, TaskLease


def inventory_only(_lease: TaskLease, _store: LeaseStore) -> int:
    """Return an empty inventory without reading or deleting staging objects."""

    return 0
