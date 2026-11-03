"""Pack each decision's frozen rows into one compressed value.

Revision ID: 0108_pack_decision_rows
Revises: 0107_cleaned_dataset_versions

0105 kept one ``decision_rows`` row per frozen row, so a decision covering half a million
rows cost half a million rows to write, and every chart read them all back and hashed them
again to check the recipe. Each set now keeps its rows as zlib-compressed uint32 gaps
(``labviz_api.cleaning_recipe.FrozenRows``), with the ``rows_sha256`` its recipe step
records, and ``decision_rows`` is dropped. The rows are index metadata of a few bytes each,
smaller than the table they replace, so they stay in SQLite rather than the artifact store.

Recipes hash the same positions the same way, so cleaned versions keep their recipe hashes.

The startup backup taken before an upgrade is the way back, so there is no downgrade.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Sequence
from typing import cast

from alembic import op

from labviz_api.cleaning_recipe import FrozenRows

revision: str = "0108_pack_decision_rows"
down_revision: str | Sequence[str] | None = "0107_cleaned_dataset_versions"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    connection = cast(sqlite3.Connection, op.get_bind().connection.driver_connection)
    connection.execute("ALTER TABLE decision_row_sets ADD COLUMN rows_sha256 TEXT")
    connection.execute("ALTER TABLE decision_row_sets ADD COLUMN rows_packed BLOB")
    sets = connection.execute(
        "SELECT project_id, finding_id, row_count FROM decision_row_sets "
        "ORDER BY project_id, finding_id"
    ).fetchall()
    for project_id, finding_id, row_count in sets:
        rows = FrozenRows.of(
            int(row[0])
            for row in connection.execute(
                "SELECT row_number FROM decision_rows WHERE project_id = ? AND finding_id = ?",
                (project_id, finding_id),
            )
        )
        if rows.row_count != row_count:
            raise RuntimeError(f"Frozen rows of {finding_id} do not match their recorded count.")
        connection.execute(
            "UPDATE decision_row_sets SET rows_sha256 = ?, rows_packed = ? "
            "WHERE project_id = ? AND finding_id = ?",
            (rows.sha256, rows.packed, project_id, finding_id),
        )
    connection.execute("DROP TABLE decision_rows")


def downgrade() -> None:
    raise RuntimeError("Restore the backup taken before upgrade to return to 0107.")
