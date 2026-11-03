"""Read research revisions from dataset versions and keep their snapshots as a cache.

Revision ID: 0103_research_revision_datasets
Revises: 0102_dataset_versions

Each applied research revision stored the parsed sources (``raw_json``) and the prepared
snapshot (``prepared_json``) inline, so undoing and reapplying a plan copied the whole
input every time. A revision now points at the dataset version that holds its sources
(shared by every revision of the same input) and keeps only its plan. The prepared
snapshot moves to ``research_prepared``, a cache that is rebuilt from the sources and
the plan when it is missing; the stored snapshots are kept there as they are, so applied
results do not change.

A revision whose sources cannot be stored losslessly stops the upgrade instead of being
rewritten. The startup backup taken before an upgrade is the way back, so there is no
downgrade.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast
from uuid import uuid4

from alembic import op

from labviz_api.artifacts import MEDIA_TYPES, local_artifact_store, remember
from labviz_api.research_dataset import decode_raw, encode_raw

revision: str = "0103_research_revision_datasets"
down_revision: str | Sequence[str] | None = "0102_dataset_versions"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def columns(connection: sqlite3.Connection, table: str) -> set[str]:
    return {str(row[1]) for row in connection.execute(f"PRAGMA table_info({table})")}


def sources_of(connection: sqlite3.Connection, version_id: str) -> list[dict[str, Any]]:
    return [
        {"id": row[0], "name": row[1], "sha256": row[2], "bytes": row[3]}
        for row in connection.execute(
            "SELECT source_id, name, sha256, byte_size FROM dataset_sources "
            "WHERE dataset_version_id = ? ORDER BY ordinal",
            (version_id,),
        )
    ]


def upgrade() -> None:
    bind = op.get_bind()
    connection = cast(sqlite3.Connection, bind.connection.driver_connection)
    store = local_artifact_store(Path(str(bind.engine.url.database)))
    if "namespace" not in columns(connection, "dataset_versions"):
        connection.execute("ALTER TABLE dataset_versions ADD COLUMN namespace TEXT")
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS dataset_sources (
            dataset_version_id TEXT NOT NULL REFERENCES dataset_versions(id) ON DELETE CASCADE,
            ordinal INTEGER NOT NULL CHECK (ordinal >= 0),
            source_id TEXT NOT NULL,
            name TEXT NOT NULL,
            sha256 TEXT NOT NULL,
            byte_size INTEGER NOT NULL CHECK (byte_size >= 0),
            PRIMARY KEY (dataset_version_id, ordinal)
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS research_prepared (
            revision_id TEXT PRIMARY KEY REFERENCES research_revisions(id) ON DELETE CASCADE,
            prepared_json TEXT NOT NULL
        )
        """
    )
    existing = columns(connection, "research_revisions")
    if "dataset_version_id" not in existing:
        connection.execute(
            "ALTER TABLE research_revisions ADD COLUMN dataset_version_id TEXT "
            "REFERENCES dataset_versions(id)"
        )
    if "plan_json" not in existing:
        connection.execute("ALTER TABLE research_revisions ADD COLUMN plan_json TEXT")
    if "raw_json" not in existing:
        return
    revision_ids = [
        str(row[0])
        for row in connection.execute("SELECT id FROM research_revisions ORDER BY rowid")
    ]
    for revision_id in revision_ids:
        project_id, raw_json, prepared_json, created_at = connection.execute(
            "SELECT project_id, raw_json, prepared_json, created_at FROM research_revisions "
            "WHERE id = ?",
            (revision_id,),
        ).fetchone()
        raw = json.loads(raw_json)
        dataset = encode_raw(raw)
        if (
            decode_raw(
                dataset.payload, dataset.sha256, namespace=raw["namespace"], sources=raw["sources"]
            )
            != raw
        ):
            raise RuntimeError(f"Research revision {revision_id} cannot be stored losslessly.")
        version_id = next(
            (
                str(row[0])
                for row in connection.execute(
                    "SELECT id FROM dataset_versions WHERE project_id = ? "
                    "AND content_sha256 = ? AND namespace = ?",
                    (project_id, dataset.sha256, raw["namespace"]),
                ).fetchall()
                if sources_of(connection, str(row[0])) == raw["sources"]
            ),
            None,
        )
        if version_id is None:
            remember(connection, store, dataset.payload, media_type=MEDIA_TYPES["parquet"])
            version_id = uuid4().hex
            connection.execute(
                """
                INSERT INTO dataset_versions (
                    id, project_id, parent_version_id, version_number, kind, content_sha256,
                    schema_json, provenance_json, row_count, column_count, created_at,
                    namespace
                )
                SELECT ?, ?, NULL, COALESCE(MAX(version_number), 0) + 1, 'parsed',
                       ?, ?, ?, ?, ?, ?, ?
                FROM dataset_versions WHERE project_id = ?
                """,
                (
                    version_id,
                    project_id,
                    dataset.sha256,
                    json.dumps(dataset.schema_document, sort_keys=True),
                    json.dumps(dataset.provenance, sort_keys=True),
                    dataset.row_count,
                    dataset.column_count,
                    created_at,
                    raw["namespace"],
                    project_id,
                ),
            )
            connection.executemany(
                "INSERT INTO dataset_sources (dataset_version_id, ordinal, source_id, name, "
                "sha256, byte_size) VALUES (?, ?, ?, ?, ?, ?)",
                [
                    (version_id, ordinal, item["id"], item["name"], item["sha256"], item["bytes"])
                    for ordinal, item in enumerate(raw["sources"])
                ],
            )
        connection.execute(
            "UPDATE research_revisions SET dataset_version_id = ?, plan_json = ? WHERE id = ?",
            (
                version_id,
                json.dumps(json.loads(prepared_json)["plan"], sort_keys=True),
                revision_id,
            ),
        )
        connection.execute(
            "INSERT OR REPLACE INTO research_prepared (revision_id, prepared_json) VALUES (?, ?)",
            (revision_id, prepared_json),
        )
    connection.execute("ALTER TABLE research_revisions DROP COLUMN raw_json")
    connection.execute("ALTER TABLE research_revisions DROP COLUMN prepared_json")


def downgrade() -> None:
    raise RuntimeError(
        "Research revisions cannot be folded back into SQLite; restore the backup taken "
        "before upgrade."
    )
