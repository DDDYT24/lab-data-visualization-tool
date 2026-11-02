"""Keep the rows each cleaning decision removes as an explicit set.

Revision ID: 0105_decision_row_sets
Revises: 0104_research_rows

A decision stored only its finding and action, so every chart, cleaned download and share
recomputed which rows the finding covers. A change to a detector, or to the data it ran on,
silently changed what a saved decision removed. A decision that excludes or removes rows now
records them once, as row positions in the dataset version it was made against, and every
reader applies that set. Decisions saved before this revision keep the recomputation until
they are frozen.

The startup backup taken before an upgrade is the way back, so there is no downgrade.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Sequence
from typing import cast

from alembic import op

revision: str = "0105_decision_row_sets"
down_revision: str | Sequence[str] | None = "0104_research_rows"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

STATEMENTS = (
    """
    CREATE TABLE IF NOT EXISTS decision_row_sets (
        project_id TEXT NOT NULL,
        finding_id TEXT NOT NULL,
        dataset_version_id TEXT NOT NULL REFERENCES dataset_versions(id) ON DELETE CASCADE,
        row_count INTEGER NOT NULL,
        frozen_at TEXT NOT NULL,
        PRIMARY KEY (project_id, finding_id),
        FOREIGN KEY (project_id, finding_id)
            REFERENCES decisions(project_id, finding_id) ON DELETE CASCADE
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS decision_rows (
        project_id TEXT NOT NULL,
        finding_id TEXT NOT NULL,
        row_number INTEGER NOT NULL,
        PRIMARY KEY (project_id, finding_id, row_number),
        FOREIGN KEY (project_id, finding_id)
            REFERENCES decision_row_sets(project_id, finding_id) ON DELETE CASCADE
    ) WITHOUT ROWID
    """,
)


def upgrade() -> None:
    connection = cast(sqlite3.Connection, op.get_bind().connection.driver_connection)
    for statement in STATEMENTS:
        connection.execute(statement)


def downgrade() -> None:
    raise RuntimeError("Restore the backup taken before upgrade to return to 0104.")
