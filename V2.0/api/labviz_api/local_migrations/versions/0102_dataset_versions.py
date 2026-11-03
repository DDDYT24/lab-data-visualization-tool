"""Store each project's table as an immutable Parquet dataset version.

Revision ID: 0102_dataset_versions
Revises: 0101_artifacts_out_of_sqlite

A classic project kept its parsed table in ``projects.data_blob`` as zlib-compressed
pandas JSON: a format with no schema, no units and no digest, and the last large BLOB in
the database. Each table now becomes a ``dataset_versions`` row whose bytes are LabViz
Parquet v1 (the same contract the PostgreSQL profile uses) in the artifact store, and the
project points at its current version.

Projects are converted one at a time to keep memory flat. The startup backup taken before
an upgrade is the way back, so there is no downgrade.
"""

from __future__ import annotations

import json
import sqlite3
import zlib
from collections.abc import Sequence
from pathlib import Path
from typing import cast
from uuid import uuid4

import pandas as pd
from alembic import op

from labviz_api.artifacts import MEDIA_TYPES, local_artifact_store, remember
from labviz_api.parquet import write_parquet

revision: str = "0102_dataset_versions"
down_revision: str | Sequence[str] | None = "0101_artifacts_out_of_sqlite"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def legacy_frame(payload: bytes) -> pd.DataFrame:
    """Decode the pre-0102 ``data_blob`` format, frozen here so the app can drop it."""
    parsed = json.loads(zlib.decompress(payload).decode("utf-8"))
    return pd.DataFrame(data=parsed["data"], columns=parsed["columns"], index=parsed["index"])


def upgrade() -> None:
    bind = op.get_bind()
    connection = cast(sqlite3.Connection, bind.connection.driver_connection)
    store = local_artifact_store(Path(str(bind.engine.url.database)))
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS dataset_versions (
            id TEXT PRIMARY KEY,
            project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
            parent_version_id TEXT REFERENCES dataset_versions(id),
            version_number INTEGER NOT NULL CHECK (version_number >= 1),
            kind TEXT NOT NULL CHECK (kind IN ('parsed', 'cleaned', 'derived')),
            content_sha256 TEXT NOT NULL,
            schema_json TEXT NOT NULL,
            provenance_json TEXT NOT NULL,
            row_count INTEGER NOT NULL CHECK (row_count >= 0),
            column_count INTEGER NOT NULL CHECK (column_count >= 1),
            created_at TEXT NOT NULL,
            UNIQUE (project_id, version_number)
        )
        """
    )
    connection.execute(
        "CREATE INDEX IF NOT EXISTS dataset_versions_project ON dataset_versions(project_id)"
    )
    columns = {str(row[1]) for row in connection.execute("PRAGMA table_info(projects)")}
    if "dataset_version_id" not in columns:
        connection.execute(
            "ALTER TABLE projects ADD COLUMN dataset_version_id TEXT "
            "REFERENCES dataset_versions(id)"
        )
    if "data_blob" not in columns:
        return
    project_ids = [
        str(row[0])
        for row in connection.execute(
            "SELECT id FROM projects WHERE data_blob IS NOT NULL ORDER BY rowid"
        )
    ]
    for project_id in project_ids:
        payload, preview_json, updated_at = connection.execute(
            "SELECT data_blob, preview_json, updated_at FROM projects WHERE id = ?",
            (project_id,),
        ).fetchone()
        preview = json.loads(preview_json) if preview_json else {}
        units = {str(item["field"]): item.get("unit") for item in preview.get("columns", [])}
        artifact = write_parquet(legacy_frame(bytes(payload)), units=units)
        remember(connection, store, artifact.payload, media_type=MEDIA_TYPES["parquet"])
        version_id = uuid4().hex
        connection.execute(
            "INSERT INTO dataset_versions (id, project_id, parent_version_id, version_number, "
            "kind, content_sha256, schema_json, provenance_json, row_count, column_count, "
            "created_at) VALUES (?, ?, NULL, 1, 'parsed', ?, ?, ?, ?, ?, ?)",
            (
                version_id,
                project_id,
                artifact.sha256,
                json.dumps(artifact.schema_document, sort_keys=True),
                json.dumps(artifact.provenance, sort_keys=True),
                artifact.row_count,
                artifact.column_count,
                updated_at,
            ),
        )
        connection.execute(
            "UPDATE projects SET dataset_version_id = ? WHERE id = ?", (version_id, project_id)
        )
    connection.execute("ALTER TABLE projects DROP COLUMN data_blob")


def downgrade() -> None:
    raise RuntimeError(
        "Dataset versions cannot be folded back into SQLite; restore the backup taken "
        "before upgrade."
    )
