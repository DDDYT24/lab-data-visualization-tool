"""Additive SQLite sidecars with preview/apply and optimistic revision control."""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from .research_preparation import PreparationError


class ResearchStore:
    def __init__(self, database_path: Path) -> None:
        self.path = database_path
        with self.connect() as connection:
            connection.executescript("""
                BEGIN IMMEDIATE;
                CREATE TABLE IF NOT EXISTS research_revisions (
                    id TEXT PRIMARY KEY,
                    project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
                    parent_id TEXT,
                    raw_json TEXT NOT NULL,
                    prepared_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS research_revision_project
                    ON research_revisions(project_id);
                CREATE TABLE IF NOT EXISTS research_heads (
                    project_id TEXT PRIMARY KEY REFERENCES projects(id) ON DELETE CASCADE,
                    revision_id TEXT NOT NULL REFERENCES research_revisions(id),
                    future_json TEXT NOT NULL DEFAULT '[]'
                );
                CREATE TABLE IF NOT EXISTS research_drafts (
                    project_id TEXT PRIMARY KEY REFERENCES projects(id) ON DELETE CASCADE,
                    token TEXT NOT NULL UNIQUE,
                    expected_revision TEXT,
                    raw_json TEXT NOT NULL,
                    prepared_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS research_public_views (
                    id TEXT PRIMARY KEY,
                    project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
                    data_json TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS research_public_views_project
                    ON research_public_views(project_id);
                COMMIT;
            """)

    @contextmanager
    def connect(self) -> Any:
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def current(self, project_id: str, *, prepared_only: bool = False) -> dict[str, Any] | None:
        columns = "r.id,r.prepared_json" if prepared_only else "r.*,h.future_json"
        with self.connect() as connection:
            row = connection.execute(
                f"""
                SELECT {columns} FROM research_heads h
                JOIN research_revisions r ON r.id=h.revision_id WHERE h.project_id=?
            """,
                (project_id,),
            ).fetchone()
        return dict(row) if row else None

    @staticmethod
    def check_head(connection: sqlite3.Connection, project_id: str, expected: str | None) -> None:
        row = connection.execute(
            "SELECT revision_id FROM research_heads WHERE project_id=?",
            (project_id,),
        ).fetchone()
        actual = row["revision_id"] if row else None
        if actual != expected:
            raise PreparationError(
                "research-conflict", "The research revision changed. Reload and preview again."
            )

    def stage(
        self, project_id: str, expected: str | None, raw: dict[str, Any], prepared: dict[str, Any]
    ) -> str:
        token = uuid4().hex
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            self.check_head(connection, project_id, expected)
            old = connection.execute(
                "SELECT token FROM research_drafts WHERE project_id=?", (project_id,)
            ).fetchone()
            if old:
                connection.execute("DELETE FROM research_public_views WHERE id=?", (old["token"],))
            connection.execute(
                """
                INSERT INTO research_drafts VALUES (?,?,?,?,?)
                ON CONFLICT(project_id) DO UPDATE SET token=excluded.token,
                expected_revision=excluded.expected_revision,raw_json=excluded.raw_json,
                prepared_json=excluded.prepared_json
            """,
                (
                    project_id,
                    token,
                    expected,
                    json.dumps(raw, allow_nan=False, separators=(",", ":")),
                    json.dumps(prepared, allow_nan=False, separators=(",", ":"), sort_keys=True),
                ),
            )
            connection.execute(
                "INSERT INTO research_public_views VALUES (?,?,?)",
                (token, project_id, json.dumps(present(prepared), allow_nan=False)),
            )
        return token

    def apply(self, project_id: str, token: str, expected: str | None) -> str:
        revision = uuid4().hex
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            self.check_head(connection, project_id, expected)
            draft = connection.execute(
                "SELECT expected_revision FROM research_drafts WHERE project_id=? AND token=?",
                (project_id, token),
            ).fetchone()
            if draft is None or draft["expected_revision"] != expected:
                raise PreparationError(
                    "preview-expired", "Preview this plan again before applying."
                )
            connection.execute(
                "INSERT INTO research_revisions "
                "SELECT ?,?,?,raw_json,prepared_json,? FROM research_drafts "
                "WHERE project_id=? AND token=?",
                (
                    revision,
                    project_id,
                    expected,
                    datetime.now(UTC).isoformat(),
                    project_id,
                    token,
                ),
            )
            connection.execute(
                """
                INSERT INTO research_heads VALUES (?,?,'[]')
                ON CONFLICT(project_id) DO UPDATE
                SET revision_id=excluded.revision_id,future_json='[]'
            """,
                (project_id, revision),
            )
            connection.execute(
                "INSERT INTO research_public_views SELECT ?,project_id,data_json "
                "FROM research_public_views WHERE id=? AND project_id=?",
                (revision, token, project_id),
            )
            connection.execute("DELETE FROM research_public_views WHERE id=?", (token,))
            connection.execute("DELETE FROM research_drafts WHERE project_id=?", (project_id,))
        return revision

    def move(self, project_id: str, expected: str, direction: str) -> None:
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            self.check_head(connection, project_id, expected)
            row = connection.execute(
                """
                SELECT r.parent_id,h.future_json FROM research_heads h
                JOIN research_revisions r ON h.revision_id=r.id WHERE h.project_id=?
            """,
                (project_id,),
            ).fetchone()
            future = json.loads(row["future_json"])
            if direction == "undo":
                target = row["parent_id"]
                if target is None:
                    raise PreparationError(
                        "no-undo", "There is no earlier applied research revision."
                    )
                future.insert(0, expected)
            else:
                if not future:
                    raise PreparationError("no-redo", "There is no research revision to redo.")
                target = future.pop(0)
            connection.execute(
                "UPDATE research_heads SET revision_id=?,future_json=? WHERE project_id=?",
                (target, json.dumps(future), project_id),
            )
            connection.execute("DELETE FROM research_drafts WHERE project_id=?", (project_id,))


def present(prepared: dict[str, Any], *, revision: str | None = None) -> dict[str, Any]:
    return {
        "schemaVersion": 1,
        "supported": True,
        "revisionId": revision,
        "sources": prepared["sources"],
        "columns": prepared["columns"],
        "summary": prepared["summary"],
        "plan": prepared["plan"],
        "history": prepared["history"],
        "samples": prepared["samples"][:200],
        "observations": prepared["observations"][:100],
        "displayedObservations": min(100, len(prepared["observations"])),
        "displayedSamples": min(200, len(prepared["samples"])),
        "previewSampled": len(prepared["observations"]) > 100,
    }


def current_view(store: ResearchStore, project_id: str) -> dict[str, Any]:
    with store.connect() as connection:
        compact = connection.execute(
            "SELECT r.id,r.parent_id,h.future_json,v.data_json "
            "FROM research_heads h JOIN research_revisions r ON r.id=h.revision_id "
            "JOIN research_public_views v ON v.id=r.id AND v.project_id=h.project_id "
            "WHERE h.project_id=?",
            (project_id,),
        ).fetchone()
    if compact is not None:
        data = json.loads(compact["data_json"])
        data["revisionId"] = compact["id"]
        return {
            "schemaVersion": 1,
            "supported": True,
            "revisionId": compact["id"],
            "canUndo": compact["parent_id"] is not None,
            "canRedo": bool(json.loads(compact["future_json"])),
            "data": data,
        }
    # JSONB skips repeated parsing of the large snapshot. Older SQLite keeps the text path.
    binary = (
        "jsonb(r.prepared_json)" if sqlite3.sqlite_version_info >= (3, 45) else "r.prepared_json"
    )
    with store.connect() as connection:
        row = connection.execute(
            f"""
            WITH snapshot AS MATERIALIZED (
                SELECT r.id,r.parent_id,h.future_json,{binary} AS prepared
                FROM research_heads h JOIN research_revisions r ON r.id=h.revision_id
                WHERE h.project_id=?
            )
            SELECT id,parent_id,future_json,
                json_extract(prepared,'$.sources') AS sources,
                json_extract(prepared,'$.columns') AS columns,
                json_extract(prepared,'$.summary') AS summary,
                json_extract(prepared,'$.plan') AS plan,
                json_extract(prepared,'$.history') AS history,
                json_array_length(prepared,'$.samples') AS sample_count,
                json_array_length(prepared,'$.observations') AS observation_count,
                (SELECT json_group_array(json(value)) FROM
                    (SELECT value FROM json_each(prepared,'$.samples') LIMIT 200)) AS samples,
                (SELECT json_group_array(json(value)) FROM
                    (SELECT value FROM json_each(prepared,'$.observations') LIMIT 100))
                    AS observations
            FROM snapshot
        """,
            (project_id,),
        ).fetchone()
    if row is None:
        return {
            "schemaVersion": 1,
            "supported": True,
            "revisionId": None,
            "canUndo": False,
            "canRedo": False,
            "data": None,
        }
    return {
        "schemaVersion": 1,
        "supported": True,
        "revisionId": row["id"],
        "canUndo": row["parent_id"] is not None,
        "canRedo": bool(json.loads(row["future_json"])),
        "data": {
            "schemaVersion": 1,
            "supported": True,
            "revisionId": row["id"],
            **{
                key: json.loads(row[key])
                for key in (
                    "sources",
                    "columns",
                    "summary",
                    "plan",
                    "history",
                    "samples",
                    "observations",
                )
            },
            "displayedObservations": min(100, row["observation_count"]),
            "displayedSamples": min(200, row["sample_count"]),
            "previewSampled": row["observation_count"] > 100,
        },
    }
