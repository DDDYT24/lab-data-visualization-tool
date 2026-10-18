"""Immutable computed chart revisions and their three frozen artifacts."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any, Literal
from uuid import uuid4

from pydantic_core import to_json

from .research_panel_headers import install_headers, remember_header, scientific_header
from .research_preparation import PreparationError
from .research_store import ResearchStore


def install_charts(store: ResearchStore) -> None:
    with store.connect() as connection:
        connection.executescript("""
            BEGIN IMMEDIATE;
            CREATE TABLE IF NOT EXISTS research_charts (
                id TEXT PRIMARY KEY,
                project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
                revision_id TEXT NOT NULL REFERENCES research_revisions(id) ON DELETE CASCADE,
                result_json TEXT NOT NULL,
                svg BLOB NOT NULL,
                png BLOB NOT NULL,
                pdf BLOB NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS research_charts_project ON research_charts(project_id);
            CREATE INDEX IF NOT EXISTS research_charts_catalog
                ON research_charts(project_id, created_at DESC, id DESC, revision_id);
            COMMIT;
        """)
        install_headers(connection)


def save_chart(
    store: ResearchStore,
    project_id: str,
    revision: str,
    result: dict[str, Any],
    artifacts: dict[str, bytes],
) -> str:
    identifier = uuid4().hex
    created_at = datetime.now(UTC).isoformat()
    with store.connect() as connection:
        connection.execute("BEGIN IMMEDIATE")
        store.check_head(connection, project_id, revision)
        connection.execute(
            "INSERT INTO research_charts VALUES (?,?,?,?,?,?,?,?)",
            (
                identifier,
                project_id,
                revision,
                to_json(result, ensure_ascii=False, round_trip=True).decode("utf-8"),
                artifacts["svg"],
                artifacts["png"],
                artifacts["pdf"],
                created_at,
            ),
        )
        header = scientific_header(result, revision)
        header["firstPage"] = {
            key: value
            for key, value in chart_view(
                {"id": identifier, "revision_id": revision, "created_at": created_at}, result=result
            ).items()
            if key not in {"id", "revisionId", "createdAt"}
        }
        remember_header(connection, project_id, identifier, header)
    return identifier


def chart_row(
    store: ResearchStore,
    project_id: str,
    chart_id: str,
    field: Literal["svg", "png", "pdf", "json"] | None = None,
    *,
    header_only: bool = False,
) -> dict[str, Any]:
    if field not in {None, "svg", "png", "pdf", "json"}:
        raise ValueError("Unknown chart artifact")
    columns = (
        "c.id, c.revision_id, c.created_at, c.result_json"
        if field in {None, "json"}
        else (
            f"c.{field}, coalesce(json_extract(h.header_json, '$.resultSha256'), "
            "json_extract(c.result_json, '$.resultSha256')) AS result_sha256"
        )
    )
    if header_only:
        columns = "c.id, c.revision_id, c.created_at"
    with store.connect() as connection:
        row = connection.execute(
            f"SELECT {columns} FROM research_charts c LEFT JOIN research_panel_headers h "
            "ON h.id=c.id AND h.project_id=c.project_id WHERE c.id=? AND c.project_id=?",
            (chart_id, project_id),
        ).fetchone()
    if row is None:
        raise PreparationError("chart-not-found", "This chart is not available in this project.")
    return dict(row)


def read_chart_page(
    store: ResearchStore, project_id: str, chart_id: str, offset: int = 0
) -> dict[str, Any]:
    if offset == 0:
        with store.connect() as connection:
            row = connection.execute(
                "SELECT c.id,c.revision_id,c.created_at,"
                "json_extract(h.header_json, '$.firstPage') AS first_page "
                "FROM research_charts c LEFT JOIN research_panel_headers h "
                "ON h.id=c.id AND h.project_id=c.project_id WHERE c.id=? AND c.project_id=?",
                (chart_id, project_id),
            ).fetchone()
        if row is None:
            raise PreparationError(
                "chart-not-found", "This chart is not available in this project."
            )
        if row["first_page"] is not None:
            # IDs/timestamp are read from this chart, so duplicated projects retain
            # their remapped identity while the frozen rows and method remain exact.
            return {
                **json.loads(row["first_page"]),
                "id": row["id"],
                "revisionId": row["revision_id"],
                "createdAt": row["created_at"],
            }
    return chart_view(chart_row(store, project_id, chart_id), offset)


def chart_view(
    row: dict[str, Any], offset: int = 0, *, result: dict[str, Any] | None = None
) -> dict[str, Any]:
    if result is None:
        result = json.loads(row["result_json"])
    observations = [p["observation"] for p in result["points"]] + result["excluded"]
    return {
        "id": row["id"],
        "revisionId": row["revision_id"],
        "createdAt": row["created_at"],
        "spec": result["spec"],
        "resultSha256": result["resultSha256"],
        "groups": result["groups"],
        "sources": result["sources"],
        "observations": len(result["points"]),
        "excluded": len(result["excluded"]),
        "totalRows": len(observations),
        "offset": offset,
        "rows": observations[offset : offset + 100],
        "pairCount": len(result["pairs"]),
        "pairs": result["pairs"][offset : offset + 100],
        "densityMethods": [
            {
                k: v
                for k, v in d.items()
                if k not in {"grid", "density", "halfWidths", "observationIds"}
            }
            for d in result["densities"]
        ],
    }
