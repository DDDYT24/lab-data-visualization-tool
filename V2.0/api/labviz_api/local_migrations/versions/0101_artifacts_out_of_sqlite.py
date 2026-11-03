"""Move rendered figures and export archives out of SQLite into the artifact store.

Revision ID: 0101_artifacts_out_of_sqlite
Revises: 0100_local_baseline

Every saved figure kept its SVG, PNG and PDF bytes (and every export its archive) in a
BLOB column, so the database grew with each render and identical renders were stored
twice. Each BLOB moves into ``labviz_api.artifacts`` under its SHA-256, the row keeps the
digest, and the BLOB column is dropped. ``figure_snapshots.sha256`` and
``research_outputs.sha256`` already record the digest of their payload, so they become
the reference instead of gaining a second column.

Bytes are moved one row at a time to keep memory flat on large profiles. A failed
upgrade rolls the database back; files it already wrote are content-addressed and reused
by the retry. The startup backup taken before an upgrade is the way back, so there is no
downgrade.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Sequence
from pathlib import Path
from typing import cast

from alembic import op

from labviz_api.artifacts import MEDIA_TYPES, REGISTRY_TABLE, local_artifact_store, remember

revision: str = "0101_artifacts_out_of_sqlite"
down_revision: str | Sequence[str] | None = "0100_local_baseline"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

#: (table, BLOB column, digest column, artifact kind). A kind of None means the row's own
#: ``format`` column names it.
MOVES = (
    ("exports", "payload", "payload_sha256", None),
    ("figure_snapshots", "payload", "sha256", None),
    ("figure_snapshots", "thumbnail_png", "thumbnail_sha256", "png"),
    ("research_charts", "svg", "svg_sha256", "svg"),
    ("research_charts", "png", "png_sha256", "png"),
    ("research_charts", "pdf", "pdf_sha256", "pdf"),
    ("research_figure_artifacts", "svg", "svg_sha256", "svg"),
    ("research_figure_artifacts", "png", "png_sha256", "png"),
    ("research_figure_artifacts", "pdf", "pdf_sha256", "pdf"),
    ("research_outputs", "zip_blob", "sha256", "zip"),
)


def upgrade() -> None:
    bind = op.get_bind()
    connection = cast(sqlite3.Connection, bind.connection.driver_connection)
    store = local_artifact_store(Path(str(bind.engine.url.database)))
    connection.execute(
        f"CREATE TABLE IF NOT EXISTS {REGISTRY_TABLE} ("
        "sha256 TEXT PRIMARY KEY, media_type TEXT NOT NULL, "
        "byte_size INTEGER NOT NULL CHECK (byte_size >= 0), created_at TEXT NOT NULL)"
    )
    tables = {
        str(row[0])
        for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")
    }
    for table, source, digest_column, kind in MOVES:
        if table not in tables:
            continue
        columns = {str(row[1]) for row in connection.execute(f"PRAGMA table_info({table})")}
        if source not in columns:
            continue
        if digest_column not in columns:
            connection.execute(f"ALTER TABLE {table} ADD COLUMN {digest_column} TEXT")
        format_column = "format" if kind is None else "NULL"
        rowids = [
            row[0]
            for row in connection.execute(
                f"SELECT rowid FROM {table} WHERE {source} IS NOT NULL ORDER BY rowid"
            )
        ]
        for rowid in rowids:
            payload, format_name = connection.execute(
                f"SELECT {source}, {format_column} FROM {table} WHERE rowid = ?", (rowid,)
            ).fetchone()
            # The recorded digest of a snapshot or output must describe the bytes that are
            # now its only copy, so it is rewritten from the bytes rather than trusted.
            digest = remember(
                connection,
                store,
                bytes(payload),
                media_type=MEDIA_TYPES[kind or str(format_name)],
            )
            connection.execute(
                f"UPDATE {table} SET {digest_column} = ? WHERE rowid = ?", (digest, rowid)
            )
        connection.execute(f"ALTER TABLE {table} DROP COLUMN {source}")


def downgrade() -> None:
    raise RuntimeError(
        "Artifacts cannot be moved back into SQLite; restore the backup taken before upgrade."
    )
