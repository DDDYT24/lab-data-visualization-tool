"""Small value objects shared by both reference and migrated persistence paths."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ProjectCreation:
    """The authoritative project/job identity returned by an upload attempt."""

    project_id: str
    job_id: str
    replayed: bool
