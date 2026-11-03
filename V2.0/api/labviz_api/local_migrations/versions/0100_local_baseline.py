"""Mark the pre-Alembic local SQLite schema as the point where Alembic takes over.

Revision ID: 0100_local_baseline
Revises: None

The local profile's baseline tables are still created by the idempotent
``CREATE TABLE IF NOT EXISTS`` blocks in ``labviz_api.repository`` and the research
sidecar installers. Re-expressing that DDL here would risk breaking every existing
profile for no behavioural gain, so this revision only records where the hand-written
``PRAGMA table_info`` patching stops.

From ``0101`` onward every local schema change is a migration, and the baseline DDL
blocks stay frozen at this shape. A brand-new profile therefore runs the baseline DDL,
is stamped at this revision, and then upgrades -- the same path an existing profile takes.
"""

from __future__ import annotations

from collections.abc import Sequence

revision: str = "0100_local_baseline"
down_revision: str | Sequence[str] | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
