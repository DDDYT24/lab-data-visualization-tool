"""Research observations, lineage, processing steps and analyses as queryable rows.

A prepared revision is one JSON snapshot, so finding the observations of one sample, the
ones in one state, or every observation that came from one source row meant parsing all
of it. Each applied revision now also writes its contents as rows, indexed by state,
sample and source row. The snapshot stays as the cache that charts read whole; the rows
are what pages, filters and provenance queries read.

Columns mirror ``research_preparation.prepare`` field for field, so a row rebuilds the
exact object the snapshot holds.
"""

from __future__ import annotations

import json
import sqlite3
from typing import Any

#: (snapshot key, column) in the order ``prepare`` writes an observation.
OBSERVATION_FIELDS = (
    ("id", "observation_id"),
    ("sampleId", "sample_id"),
    ("sampleLabel", "sample_label"),
    ("experimentalUnit", "experimental_unit"),
    ("group", "group_label"),
    ("time", "time_label"),
    ("batch", "batch"),
    ("technicalRepeat", "technical_repeat"),
    ("pairKey", "pair_key"),
    ("unit", "unit"),
    ("state", "state"),
    ("value", "value"),
    ("limit", "limit_value"),
    ("reportedText", "reported_text"),
    ("reportedUnit", "reported_unit"),
    ("measurementType", "measurement_type"),
    ("role", "role"),
    ("notes", "notes"),
    ("recordLink", "record_link"),
    ("exclusionReason", "exclusion_reason"),
)
SAMPLE_FIELDS = (
    ("id", "sample_id"),
    ("label", "label"),
    ("experimentalUnit", "experimental_unit"),
    ("parentSample", "parent_sample"),
    ("group", "group_label"),
)
LINEAGE_FIELDS = (
    ("sourceId", "source_id"),
    ("sheet", "sheet"),
    ("row", "source_row"),
    ("column", "source_column"),
)
TABLES = (
    "research_observations",
    "research_samples",
    "research_lineage",
    "research_steps",
    "research_analyses",
)


def write_rows(connection: sqlite3.Connection, revision_id: str, prepared: dict[str, Any]) -> None:
    """Record a prepared revision's contents; a revision is written once.

    Migration 0104 backfills existing revisions through this function, so a later change
    to these tables needs a migration that stops relying on it.
    """
    columns = ",".join(column for _, column in OBSERVATION_FIELDS)
    marks = ",".join("?" for _ in OBSERVATION_FIELDS)
    connection.executemany(
        f"INSERT INTO research_observations (revision_id,ordinal,{columns}) VALUES (?,?,{marks})",
        (
            (revision_id, ordinal, *(item[key] for key, _ in OBSERVATION_FIELDS))
            for ordinal, item in enumerate(prepared["observations"])
        ),
    )
    connection.executemany(
        "INSERT INTO research_lineage (revision_id,observation_ordinal,ordinal,source_id,sheet,"
        "source_row,source_column) VALUES (?,?,?,?,?,?,?)",
        (
            (revision_id, ordinal, index, *(ref[key] for key, _ in LINEAGE_FIELDS))
            for ordinal, item in enumerate(prepared["observations"])
            for index, ref in enumerate(item["lineage"])
        ),
    )
    sample_columns = ",".join(column for _, column in SAMPLE_FIELDS)
    connection.executemany(
        f"INSERT INTO research_samples (revision_id,ordinal,{sample_columns}) "
        "VALUES (?,?,?,?,?,?,?)",
        (
            (revision_id, ordinal, *(item[key] for key, _ in SAMPLE_FIELDS))
            for ordinal, item in enumerate(prepared["samples"])
        ),
    )
    connection.executemany(
        "INSERT INTO research_steps (revision_id,ordinal,kind,step_json,input_rows,output_rows,"
        "algorithm_version) VALUES (?,?,?,?,?,?,?)",
        (
            (
                revision_id,
                ordinal,
                item["step"]["kind"],
                json.dumps(item["step"], sort_keys=True),
                item["inputRows"],
                item["outputRows"],
                item["algorithmVersion"],
            )
            for ordinal, item in enumerate(prepared["history"])
        ),
    )
    summary = prepared["summary"]
    connection.execute(
        "INSERT INTO research_analyses (revision_id,analysis_unit,observation_count,sample_count,"
        "independent_unit_count,included_observed,excluded,analysis_n,mean,summary_json) "
        "VALUES (?,?,?,?,?,?,?,?,?,?)",
        (
            revision_id,
            summary["analysisUnit"],
            summary["observations"],
            summary["samples"],
            summary["independentUnits"],
            summary["includedObserved"],
            summary["excluded"],
            summary["analysisN"],
            summary["mean"],
            json.dumps(summary, allow_nan=False, sort_keys=True),
        ),
    )


def copy_rows(connection: sqlite3.Connection, source_revision: str, revision_id: str) -> None:
    """Give a copied revision its own rows, for a project duplicated with its history."""
    for table in TABLES:
        names = [
            str(row[1])
            for row in connection.execute(f"PRAGMA table_info({table})")
            if row[1] != "revision_id"
        ]
        listed = ",".join(names)
        connection.execute(
            f"INSERT INTO {table} (revision_id,{listed}) "
            f"SELECT ?,{listed} FROM {table} WHERE revision_id=?",
            (revision_id, source_revision),
        )


def observation_page(
    connection: sqlite3.Connection, revision_id: str, offset: int, limit: int
) -> list[dict[str, Any]]:
    columns = ",".join(column for _, column in OBSERVATION_FIELDS)
    rows = connection.execute(
        f"SELECT ordinal,{columns} FROM research_observations WHERE revision_id=? "
        "ORDER BY ordinal LIMIT ? OFFSET ?",
        (revision_id, limit, offset),
    ).fetchall()
    if not rows:
        return []
    lineage: dict[int, list[dict[str, Any]]] = {row[0]: [] for row in rows}
    for item in connection.execute(
        "SELECT observation_ordinal,source_id,sheet,source_row,source_column "
        "FROM research_lineage WHERE revision_id=? AND observation_ordinal BETWEEN ? AND ? "
        "ORDER BY observation_ordinal,ordinal",
        (revision_id, rows[0][0], rows[-1][0]),
    ):
        lineage[item[0]].append(
            {key: item[index + 1] for index, (key, _) in enumerate(LINEAGE_FIELDS)}
        )
    page = []
    for row in rows:
        values = dict(zip((key for key, _ in OBSERVATION_FIELDS), row[1:], strict=True))
        reason = values.pop("exclusionReason")
        page.append(values | {"lineage": lineage[row[0]], "exclusionReason": reason})
    return page


def sample_page(
    connection: sqlite3.Connection, revision_id: str, offset: int, limit: int
) -> list[dict[str, Any]]:
    columns = ",".join(column for _, column in SAMPLE_FIELDS)
    return [
        dict(zip((key for key, _ in SAMPLE_FIELDS), row, strict=True))
        for row in connection.execute(
            f"SELECT {columns} FROM research_samples WHERE revision_id=? "
            "ORDER BY ordinal LIMIT ? OFFSET ?",
            (revision_id, limit, offset),
        )
    ]


def count(connection: sqlite3.Connection, revision_id: str, kind: str) -> int:
    table = "research_observations" if kind == "observations" else "research_samples"
    return int(
        connection.execute(
            f"SELECT COUNT(*) FROM {table} WHERE revision_id=?", (revision_id,)
        ).fetchone()[0]
    )
