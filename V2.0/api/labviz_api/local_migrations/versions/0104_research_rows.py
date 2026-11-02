"""Write research observations, lineage, steps and analyses as rows.

Revision ID: 0104_research_rows
Revises: 0103_research_revision_datasets

A prepared revision was only a JSON snapshot, so paging observations, filtering them by
state or sample, or tracing one source row meant parsing all of it. Each revision now
also stores its contents as indexed rows (``labviz_api.research_rows``); the snapshot
remains the cache charts read whole. Existing revisions are backfilled from their cached
snapshots, which 0103 kept unchanged.

The startup backup taken before an upgrade is the way back, so there is no downgrade.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Sequence
from typing import cast

from alembic import op

from labviz_api.research_rows import write_rows

revision: str = "0104_research_rows"
down_revision: str | Sequence[str] | None = "0103_research_revision_datasets"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

STATEMENTS = (
    """
    CREATE TABLE IF NOT EXISTS research_observations (
        revision_id TEXT NOT NULL REFERENCES research_revisions(id) ON DELETE CASCADE,
        ordinal INTEGER NOT NULL,
        observation_id TEXT NOT NULL,
        sample_id TEXT,
        sample_label TEXT,
        experimental_unit TEXT,
        group_label TEXT,
        time_label TEXT,
        batch TEXT,
        technical_repeat INTEGER,
        pair_key TEXT,
        unit TEXT,
        state TEXT NOT NULL,
        value REAL,
        limit_value REAL,
        reported_text TEXT NOT NULL,
        reported_unit TEXT,
        measurement_type TEXT,
        role TEXT NOT NULL,
        notes TEXT,
        record_link TEXT,
        exclusion_reason TEXT,
        PRIMARY KEY (revision_id, ordinal)
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS research_observations_id
        ON research_observations(revision_id, observation_id)
    """,
    """
    CREATE INDEX IF NOT EXISTS research_observations_state
        ON research_observations(revision_id, state)
    """,
    """
    CREATE INDEX IF NOT EXISTS research_observations_sample
        ON research_observations(revision_id, sample_id)
    """,
    """
    CREATE TABLE IF NOT EXISTS research_samples (
        revision_id TEXT NOT NULL REFERENCES research_revisions(id) ON DELETE CASCADE,
        ordinal INTEGER NOT NULL,
        sample_id TEXT NOT NULL,
        label TEXT NOT NULL,
        experimental_unit TEXT,
        parent_sample TEXT,
        group_label TEXT,
        PRIMARY KEY (revision_id, ordinal)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS research_lineage (
        revision_id TEXT NOT NULL REFERENCES research_revisions(id) ON DELETE CASCADE,
        observation_ordinal INTEGER NOT NULL,
        ordinal INTEGER NOT NULL,
        source_id TEXT NOT NULL,
        sheet TEXT,
        source_row INTEGER NOT NULL,
        source_column TEXT NOT NULL,
        PRIMARY KEY (revision_id, observation_ordinal, ordinal)
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS research_lineage_source
        ON research_lineage(revision_id, source_id, source_row)
    """,
    """
    CREATE TABLE IF NOT EXISTS research_steps (
        revision_id TEXT NOT NULL REFERENCES research_revisions(id) ON DELETE CASCADE,
        ordinal INTEGER NOT NULL,
        kind TEXT NOT NULL,
        step_json TEXT NOT NULL,
        input_rows INTEGER NOT NULL,
        output_rows INTEGER NOT NULL,
        algorithm_version TEXT NOT NULL,
        PRIMARY KEY (revision_id, ordinal)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS research_analyses (
        revision_id TEXT PRIMARY KEY REFERENCES research_revisions(id) ON DELETE CASCADE,
        analysis_unit TEXT NOT NULL,
        observation_count INTEGER NOT NULL,
        sample_count INTEGER NOT NULL,
        independent_unit_count INTEGER,
        included_observed INTEGER NOT NULL,
        excluded INTEGER NOT NULL,
        analysis_n INTEGER,
        mean REAL,
        summary_json TEXT NOT NULL
    )
    """,
)


def upgrade() -> None:
    connection = cast(sqlite3.Connection, op.get_bind().connection.driver_connection)
    for statement in STATEMENTS:
        connection.execute(statement)
    pending = [
        str(row[0])
        for row in connection.execute(
            "SELECT p.revision_id FROM research_prepared p WHERE NOT EXISTS "
            "(SELECT 1 FROM research_analyses a WHERE a.revision_id = p.revision_id) "
            "ORDER BY p.rowid"
        )
    ]
    for revision_id in pending:
        (prepared_json,) = connection.execute(
            "SELECT prepared_json FROM research_prepared WHERE revision_id = ?", (revision_id,)
        ).fetchone()
        write_rows(connection, revision_id, json.loads(prepared_json))


def downgrade() -> None:
    raise RuntimeError("Restore the backup taken before upgrade to return to 0103.")
