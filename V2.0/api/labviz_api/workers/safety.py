"""Explicit destructive-maintenance safety policy shared by worker handlers."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MaintenanceSafety:
    """Require two deliberate settings before any destructive external effect."""

    dry_run: bool = True
    delete_enabled: bool = False

    @property
    def may_delete(self) -> bool:
        return not self.dry_run and self.delete_enabled


class PermanentWorkerFailure(RuntimeError):
    """A bounded diagnostic for an item that should be quarantined, not retried."""
