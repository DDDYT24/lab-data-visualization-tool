"""Store each project's cleaning result as a cleaned dataset version.

Revision ID: 0107_cleaned_dataset_versions
Revises: 0106_freeze_legacy_decisions

Charts and the cleaned download sliced the parsed table with the frozen decision rows on
every read, so the cleaned table had no identity a later step could cache against. Saving
decisions now writes a ``cleaned`` version, the parsed one minus the rows ``remove``
decisions froze, whose provenance holds the recipe (``labviz_api.cleaning_recipe``) and
whose ``recipe_sha256`` names it; ``projects.cleaned_version_id`` points at it. Projects
with decisions that drop rows are backfilled from the rows 0106 froze, so nothing is
detected here.

The startup backup taken before an upgrade is the way back, so there is no downgrade.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Sequence
from pathlib import Path
from typing import cast
from uuid import uuid4

from alembic import op

from labviz_api.artifacts import MEDIA_TYPES, local_artifact_store, remember
from labviz_api.cleaning_recipe import (
    EXCLUDING_ACTIONS,
    build_recipe,
    decided_rows,
    recipe_sha256,
)
from labviz_api.repository import iso_now

revision: str = "0107_cleaned_dataset_versions"
down_revision: str | Sequence[str] | None = "0106_freeze_legacy_decisions"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    bind = op.get_bind()
    connection = cast(sqlite3.Connection, bind.connection.driver_connection)
    connection.execute("ALTER TABLE dataset_versions ADD COLUMN recipe_sha256 TEXT")
    connection.execute(
        "ALTER TABLE projects ADD COLUMN cleaned_version_id TEXT REFERENCES dataset_versions(id)"
    )
    connection.execute(
        "CREATE INDEX IF NOT EXISTS dataset_versions_recipe "
        "ON dataset_versions(project_id, recipe_sha256)"
    )
    placeholders = ", ".join("?" for _ in EXCLUDING_ACTIONS)
    projects = [
        str(row[0])
        for row in connection.execute(
            f"""
            SELECT DISTINCT d.project_id FROM decisions d
            JOIN projects p ON p.id = d.project_id
            WHERE d.action IN ({placeholders}) AND p.dataset_version_id IS NOT NULL
            ORDER BY d.project_id
            """,
            tuple(EXCLUDING_ACTIONS),
        )
    ]
    if not projects:
        return

    from labviz_api.parquet import read_parquet, write_parquet
    from labviz_api.processing import cleaned_frame

    store = local_artifact_store(Path(str(bind.engine.url.database)))
    for project_id in projects:
        parent_id, digest, schema_json, provenance_json, row_count, column_count = (
            connection.execute(
                "SELECT v.id, v.content_sha256, v.schema_json, v.provenance_json, v.row_count, "
                "v.column_count FROM dataset_versions v "
                "JOIN projects p ON p.dataset_version_id = v.id WHERE p.id = ?",
                (project_id,),
            ).fetchone()
        )
        provenance = json.loads(provenance_json)
        decisions = [
            {"findingId": str(row[0]), "action": str(row[1])}
            for row in connection.execute(
                "SELECT finding_id, action FROM decisions WHERE project_id = ?", (project_id,)
            )
        ]
        frozen: dict[str, list[int]] = {}
        for finding_id, row_number in connection.execute(
            "SELECT r.finding_id, r.row_number FROM decision_rows r "
            "JOIN decision_row_sets s "
            "ON s.project_id = r.project_id AND s.finding_id = r.finding_id "
            "WHERE r.project_id = ? AND s.dataset_version_id = ? ORDER BY r.row_number",
            (project_id, parent_id),
        ):
            frozen.setdefault(str(finding_id), []).append(int(row_number))
        for (finding_id,) in connection.execute(
            "SELECT finding_id FROM decision_row_sets WHERE project_id = ? "
            "AND dataset_version_id = ?",
            (project_id, parent_id),
        ):
            frozen.setdefault(str(finding_id), [])
        recipe = build_recipe(digest, decisions, frozen)
        removed = decided_rows(decisions, frozen, {"remove"})
        if removed:
            schema = json.loads(schema_json)
            units = {str(column["name"]): column.get("unit") for column in schema["columns"]}
            frame = read_parquet(store.read(digest), expected_sha256=digest)
            dataset = write_parquet(cleaned_frame(frame, removed), units=units)
            remember(connection, store, dataset.payload, media_type=MEDIA_TYPES["parquet"])
            digest = dataset.sha256
            schema_json = json.dumps(dataset.schema_document, sort_keys=True)
            provenance = dict(dataset.provenance)
            row_count = dataset.row_count
            column_count = dataset.column_count
        version_id = uuid4().hex
        connection.execute(
            """
            INSERT INTO dataset_versions (
                id, project_id, parent_version_id, version_number, kind, content_sha256,
                schema_json, provenance_json, row_count, column_count, created_at,
                recipe_sha256
            )
            SELECT ?, ?, ?, COALESCE(MAX(version_number), 0) + 1, 'cleaned',
                   ?, ?, ?, ?, ?, ?, ?
            FROM dataset_versions WHERE project_id = ?
            """,
            (
                version_id,
                project_id,
                parent_id,
                digest,
                schema_json,
                json.dumps({**provenance, "recipe": recipe}, sort_keys=True),
                row_count,
                column_count,
                iso_now(),
                recipe_sha256(recipe),
                project_id,
            ),
        )
        connection.execute(
            "UPDATE projects SET cleaned_version_id = ? WHERE id = ?", (version_id, project_id)
        )


def downgrade() -> None:
    raise RuntimeError("Restore the backup taken before upgrade to return to 0106.")
