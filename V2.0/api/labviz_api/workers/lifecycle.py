"""Project lifecycle boundary for Phase 5B-2 activation."""

from .leases import LeaseStore, WorkItemLease


def inspect_expired_project(_lease: WorkItemLease, _store: LeaseStore) -> None:
    """Claim and release only; project purge remains disabled in Phase 5B-1."""
