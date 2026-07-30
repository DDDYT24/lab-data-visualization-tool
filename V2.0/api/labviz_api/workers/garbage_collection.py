"""Stored-object GC boundary for Phase 5B-2 activation."""

from .leases import LeaseStore, WorkItemLease


def inspect_gc_candidate(_lease: WorkItemLease, _store: LeaseStore) -> None:
    """Claim and release only; physical object deletion remains disabled in Phase 5B-1."""
