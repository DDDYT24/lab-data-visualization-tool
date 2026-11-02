"""Freeze the rows of every cleaning decision saved before row sets existed.

Revision ID: 0106_freeze_legacy_decisions
Revises: 0105_decision_row_sets

Until 0105 a decision kept only its finding and action, and each read detected the finding's
rows again. Each such decision that excludes or removes rows is now detected once more, on its
project's current dataset version, and frozen like a decision saved today, so readers no longer
detect anything. The detection is this build's, which is what every read of the decision used
until now, so a migrated project draws and downloads the same rows it did before the upgrade.

``decision_row_sets.migrated_from`` names this revision on the sets it froze and stays NULL on
sets a save froze, so a migrated decision stays distinguishable from one a person made against
row sets.

The startup backup taken before an upgrade is the way back, so there is no downgrade.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Sequence
from pathlib import Path
from typing import cast

from alembic import op

from labviz_api.artifacts import local_artifact_store
from labviz_api.repository import iso_now

revision: str = "0106_freeze_legacy_decisions"
down_revision: str | Sequence[str] | None = "0105_decision_row_sets"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

EXCLUDING = ("exclude", "remove")


def upgrade() -> None:
    bind = op.get_bind()
    connection = cast(sqlite3.Connection, bind.connection.driver_connection)
    connection.execute("ALTER TABLE decision_row_sets ADD COLUMN migrated_from TEXT")
    pending: dict[str, list[str]] = {}
    for project_id, finding_id in connection.execute(
        """
        SELECT d.project_id, d.finding_id FROM decisions d
        JOIN projects p ON p.id = d.project_id
        WHERE d.action IN (?, ?)
          AND p.dataset_version_id IS NOT NULL AND p.quality_json IS NOT NULL
          AND NOT EXISTS (
              SELECT 1 FROM decision_row_sets s
              WHERE s.project_id = d.project_id AND s.finding_id = d.finding_id
                AND s.dataset_version_id = p.dataset_version_id
          )
        ORDER BY d.project_id, d.finding_id
        """,
        EXCLUDING,
    ):
        pending.setdefault(str(project_id), []).append(str(finding_id))
    if not pending:
        return

    # Detection drags in the scientific stack, so only a profile with decisions pays for it.
    from labviz_api.parquet import read_parquet
    from labviz_api.processing import freeze_decision_rows

    store = local_artifact_store(Path(str(bind.engine.url.database)))
    frozen_at = iso_now()
    for project_id, finding_ids in pending.items():
        version_id, digest, quality_json = connection.execute(
            """
            SELECT p.dataset_version_id, v.content_sha256, p.quality_json
            FROM projects p JOIN dataset_versions v ON v.id = p.dataset_version_id
            WHERE p.id = ?
            """,
            (project_id,),
        ).fetchone()
        frame = read_parquet(store.read(digest), expected_sha256=digest)
        row_sets = freeze_decision_rows(frame, json.loads(quality_json), finding_ids)
        for finding_id, rows in row_sets.items():
            # A set frozen against an earlier version describes other rows.
            connection.execute(
                "DELETE FROM decision_row_sets WHERE project_id = ? AND finding_id = ?",
                (project_id, finding_id),
            )
            connection.execute(
                """
                INSERT INTO decision_row_sets (
                    project_id, finding_id, dataset_version_id, row_count, frozen_at,
                    migrated_from
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (project_id, finding_id, version_id, len(rows), frozen_at, revision),
            )
            connection.executemany(
                "INSERT INTO decision_rows (project_id, finding_id, row_number) VALUES (?, ?, ?)",
                [(project_id, finding_id, row) for row in rows],
            )


def downgrade() -> None:
    raise RuntimeError("Restore the backup taken before upgrade to return to 0105.")
