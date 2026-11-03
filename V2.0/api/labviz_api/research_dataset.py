"""Research source tables as immutable Parquet dataset versions.

``parse_sources`` returns every stacked row with the source and physical row it came
from. A revision used to store that document as JSON, so each applied revision carried
its own copy of the same input. The rows now become one dataset version per distinct
input: the cells, plus three reserved columns that record each row's identity and origin,
in LabViz Parquet v1. Revisions of the same input share that version.

Every origin ``parse_sources`` writes points at one source row and names its own column,
so the reserved columns are enough to rebuild the document exactly.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from .artifacts import MEDIA_TYPES, ArtifactStore, remember

if TYPE_CHECKING:
    from .parquet import ParquetArtifact

ROW_ID = "__labviz_row_id"
SOURCE_ID = "__labviz_source_id"
SOURCE_ROW = "__labviz_source_row"
RESERVED = (ROW_ID, SOURCE_ID, SOURCE_ROW)
#: Labels ``parse_sources`` appends to every row; they have no origin of their own.
LABELS = ("__labviz_source", "__labviz_batch")


def origins(fields: list[str], source_id: str, row: int) -> dict[str, list[dict[str, Any]]]:
    return {
        field: [{"sourceId": source_id, "sheet": None, "row": row, "column": field}]
        for field in fields
    }


def encode_raw(raw: dict[str, Any]) -> ParquetArtifact:
    """Write parsed research sources as Parquet, refusing anything it could not rebuild."""
    import pandas as pd

    from .parquet import write_parquet

    columns = list(raw["columns"])
    if raw["schemaVersion"] != 1 or tuple(columns[-2:]) != LABELS:
        raise ValueError("Research sources have an unsupported layout.")
    fields = columns[:-2]
    data: dict[str, list[Any]] = {name: [] for name in (*RESERVED, *columns)}
    for row in raw["rows"]:
        reference = row["origins"][fields[0]][0]
        source_id, source_row = reference["sourceId"], reference["row"]
        if (
            row["origins"] != origins(fields, source_id, source_row)
            or list(row["cells"]) != columns
        ):
            raise ValueError("A research row has origins the dataset layout cannot record.")
        data[ROW_ID].append(row["id"])
        data[SOURCE_ID].append(source_id)
        data[SOURCE_ROW].append(source_row)
        for name in columns:
            data[name].append(row["cells"][name])
    return write_parquet(pd.DataFrame(data))


def decode_raw(
    payload: bytes, sha256: str, *, namespace: str, sources: list[dict[str, Any]]
) -> dict[str, Any]:
    from .parquet import read_parquet

    frame = read_parquet(payload, expected_sha256=sha256)
    columns = [str(name) for name in frame.columns if name not in RESERVED]
    fields = columns[:-2]
    values = {name: frame[name].tolist() for name in columns}
    ids = frame[ROW_ID].tolist()
    source_ids = frame[SOURCE_ID].tolist()
    source_rows = [int(value) for value in frame[SOURCE_ROW].tolist()]
    return {
        "schemaVersion": 1,
        "namespace": namespace,
        "sources": sources,
        "columns": columns,
        "rows": [
            {
                "id": ids[index],
                "cells": {name: values[name][index] for name in columns},
                "origins": origins(fields, source_ids[index], source_rows[index]),
            }
            for index in range(len(ids))
        ],
    }


def version_sources(connection: sqlite3.Connection, version_id: str) -> list[dict[str, Any]]:
    return [
        {"id": row[0], "name": row[1], "sha256": row[2], "bytes": row[3]}
        for row in connection.execute(
            "SELECT source_id, name, sha256, byte_size FROM dataset_sources "
            "WHERE dataset_version_id = ? ORDER BY ordinal",
            (version_id,),
        )
    ]


def freeze_raw(
    connection: sqlite3.Connection,
    store: ArtifactStore,
    project_id: str,
    raw: dict[str, Any],
    dataset: ParquetArtifact,
) -> str:
    """Return the project's version holding these sources, creating it if it is new."""
    for (version_id,) in connection.execute(
        "SELECT id FROM dataset_versions WHERE project_id = ? AND content_sha256 = ? "
        "AND namespace = ?",
        (project_id, dataset.sha256, raw["namespace"]),
    ).fetchall():
        if version_sources(connection, version_id) == raw["sources"]:
            return str(version_id)
    remember(connection, store, dataset.payload, media_type=MEDIA_TYPES["parquet"])
    version_id = uuid4().hex
    connection.execute(
        """
        INSERT INTO dataset_versions (
            id, project_id, parent_version_id, version_number, kind, content_sha256,
            schema_json, provenance_json, row_count, column_count, created_at, namespace
        )
        SELECT ?, ?, NULL, COALESCE(MAX(version_number), 0) + 1, 'parsed', ?, ?, ?, ?, ?, ?, ?
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
            datetime.now(UTC).isoformat(),
            raw["namespace"],
            project_id,
        ),
    )
    connection.executemany(
        "INSERT INTO dataset_sources (dataset_version_id, ordinal, source_id, name, sha256, "
        "byte_size) VALUES (?, ?, ?, ?, ?, ?)",
        [
            (version_id, ordinal, item["id"], item["name"], item["sha256"], item["bytes"])
            for ordinal, item in enumerate(raw["sources"])
        ],
    )
    return version_id


def load_raw(
    connection: sqlite3.Connection, store: ArtifactStore, version_id: str
) -> dict[str, Any]:
    row = connection.execute(
        "SELECT content_sha256, namespace FROM dataset_versions WHERE id = ?", (version_id,)
    ).fetchone()
    return decode_raw(
        store.read(row[0]),
        row[0],
        namespace=row[1],
        sources=version_sources(connection, version_id),
    )
