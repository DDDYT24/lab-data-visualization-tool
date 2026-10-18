"""Phase 3A: immutable layouts referencing already-frozen local scientific charts."""

from __future__ import annotations

import gc
import json
import sqlite3
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from time import perf_counter
from typing import Annotated, Any, Literal
from uuid import uuid4

from pydantic import Field, model_validator
from pydantic_core import to_json

from .research_contract import Identifier, ResearchModel
from .research_figure_render import render_board
from .research_figure_sources import source
from .research_panel_headers import install_headers
from .research_preparation import PreparationError
from .research_store import ResearchStore


class FigureBoardSpec(ResearchModel):
    schema_version: Literal[1] = 1
    point_rendering: Literal["auto", "vector", "raster"] = "auto"
    title: str = Field(default="", max_length=160)
    axis_mode: Literal["independent", "shared_x", "shared_y", "shared_xy"] = "independent"
    palette: Literal["blue_teal", "grayscale"] = "blue_teal"
    font_size: int = Field(default=10, ge=8, le=16)
    grid_visible: bool = True
    panel_titles: list[Annotated[str, Field(max_length=160)]] = Field(
        default_factory=list, max_length=4
    )
    panel_chart_ids: list[Identifier] = Field(min_length=1, max_length=4)

    @model_validator(mode="after")
    def valid_layout(self) -> FigureBoardSpec:
        if self.panel_titles and len(self.panel_titles) != len(self.panel_chart_ids):
            raise ValueError("Panel titles must match the visible panel count.")
        if len(self.panel_chart_ids) not in {1, 2, 4}:
            raise ValueError("Choose exactly one, two or four panels.")
        if any(not value.strip() or len(value) > 128 for value in self.panel_chart_ids):
            raise ValueError("Every panel needs a saved chart identifier.")
        return self


def install_boards(store: ResearchStore) -> None:
    with store.connect() as connection:
        connection.executescript("""
            BEGIN IMMEDIATE;
            CREATE TABLE IF NOT EXISTS research_figure_boards (
                id TEXT PRIMARY KEY,
                project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
                spec_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS research_figure_boards_project
                ON research_figure_boards(project_id);
            CREATE TABLE IF NOT EXISTS research_figure_sources (
                id TEXT PRIMARY KEY,
                project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
                result_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS research_figure_sources_catalog
                ON research_figure_sources(project_id, created_at DESC, id DESC);
            CREATE TABLE IF NOT EXISTS research_figure_artifacts (
                board_id TEXT PRIMARY KEY REFERENCES research_figure_boards(id) ON DELETE CASCADE,
                svg BLOB NOT NULL, png BLOB NOT NULL, pdf BLOB NOT NULL, scene_json TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS research_figure_jobs (
                board_id TEXT PRIMARY KEY REFERENCES research_figure_boards(id) ON DELETE CASCADE,
                state TEXT NOT NULL CHECK(state IN ('queued','running','ready','failed')),
                panels_json TEXT NOT NULL,
                error TEXT,
                attempt TEXT,
                timings_json TEXT NOT NULL DEFAULT '{}'
            );
            COMMIT;
        """)
        install_headers(connection)


def resolve_panels(
    connection: sqlite3.Connection, project_id: str, spec: FigureBoardSpec
) -> list[dict[str, Any]]:
    loaded = {
        identifier: source(connection, project_id, identifier)
        for identifier in dict.fromkeys(spec.panel_chart_ids)
    }
    return panel_metadata([loaded[identifier] for identifier in spec.panel_chart_ids], spec)


def panel_metadata(sources: list[dict[str, Any]], spec: FigureBoardSpec) -> list[dict[str, Any]]:
    panels = []
    for index, (identifier, panel) in enumerate(zip(spec.panel_chart_ids, sources, strict=True)):
        observations = panel["rows"]
        units = {item.get("experimentalUnit") for item in observations}
        panels.append(
            {
                "label": "ABCD"[index],
                "chartId": identifier,
                "revisionId": panel["revisionId"],
                "kind": panel["kind"],
                "title": (spec.panel_titles[index] if spec.panel_titles else "") or panel["title"],
                "unit": panel["unit"],
                "observations": len(observations),
                "independentUnits": None if None in units else len(units),
                "resultSha256": panel["resultSha256"],
            }
        )
    return panels


def board_view(connection: sqlite3.Connection, row: Any) -> dict[str, Any]:
    spec = FigureBoardSpec.model_validate_json(row["spec_json"])
    job = connection.execute(
        "SELECT state,panels_json,error,timings_json FROM research_figure_jobs WHERE board_id=?",
        (row["id"],),
    ).fetchone()
    if job:
        return {
            "id": row["id"],
            "createdAt": row["created_at"],
            "spec": spec.model_dump(),
            "panels": json.loads(job["panels_json"]),
            "composed": job["state"] == "ready",
            "artifactState": job["state"],
            "artifactError": job["error"],
            "artifactTimings": json.loads(job["timings_json"]),
        }
    artifact = connection.execute(
        "SELECT json_extract(scene_json, '$.panelMetadata') AS panels "
        "FROM research_figure_artifacts WHERE board_id=?",
        (row["id"],),
    ).fetchone()
    panels = (
        json.loads(artifact["panels"])
        if artifact and artifact["panels"]
        else resolve_panels(connection, row["project_id"], spec)
    )
    # Duplicates remap chart/revision IDs but retain artifact bytes and original provenance.
    if [panel["chartId"] for panel in panels] != spec.panel_chart_ids:
        panels = resolve_panels(connection, row["project_id"], spec)
    return {
        "id": row["id"],
        "createdAt": row["created_at"],
        "composed": connection.execute(
            "SELECT 1 FROM research_figure_artifacts WHERE board_id=?", (row["id"],)
        ).fetchone()
        is not None,
        "spec": spec.model_dump(),
        "panels": panels,
        "artifactState": "ready" if artifact else "unavailable",
        "artifactError": None,
    }


def save_board(
    store: ResearchStore,
    project_id: str,
    spec: FigureBoardSpec,
    on_scene: Callable[[str, dict[str, Any]], None] | None = None,
    timings: dict[str, float] | None = None,
    *,
    batch_sources: bool = False,
) -> dict[str, Any]:
    started = perf_counter()
    row = {
        "id": uuid4().hex,
        "project_id": project_id,
        "spec_json": spec.model_dump_json(),
        "created_at": datetime.now(UTC).isoformat(),
    }
    with store.connect() as connection:
        connection.execute("BEGIN IMMEDIATE")
        # Only decoded JSON and derived row metadata are constructed here.
        # Route callers opt in under the shared research execution lock;
        # cyclic drawing objects are created after restoring the caller state.
        restore_gc = batch_sources and gc.isenabled()
        if restore_gc:
            gc.disable()
        try:
            # Resolve each immutable source once, even when it is placed in several panels.
            loaded = {
                identifier: source(connection, project_id, identifier)
                for identifier in dict.fromkeys(spec.panel_chart_ids)
            }
            sources = [loaded[identifier] for identifier in spec.panel_chart_ids]
            view: dict[str, Any] = {
                "id": row["id"],
                "createdAt": row["created_at"],
                "composed": False,
                "spec": spec.model_dump(),
                "panels": panel_metadata(sources, spec),
            }
        finally:
            if restore_gc:
                gc.enable()
        rendered_start = perf_counter()
        artifacts, scene = render_board(sources, spec)
        rendered_end = perf_counter()
        # Small frozen panel summaries allow reopen without decoding complete chart results.
        scene["panelMetadata"] = view["panels"]
        # The scene owns complete source rows. Release render-only snapshots before
        # allocating serialized scene bytes and SQLite's transaction buffers.
        del loaded, sources
        if sum(panel["observations"] for panel in view["panels"]) > 20_000:
            gc.collect()
        connection.execute(
            "INSERT INTO research_figure_boards VALUES (?,?,?,?)",
            (row["id"], project_id, row["spec_json"], row["created_at"]),
        )
        connection.execute(
            "INSERT INTO research_figure_artifacts VALUES (?,?,?,?,?)",
            (
                row["id"],
                artifacts["svg"],
                artifacts["png"],
                artifacts["pdf"],
                to_json(scene, ensure_ascii=False, round_trip=True).decode("utf-8"),
            ),
        )
        view["composed"] = True
        view["artifactState"] = "ready"
        view["artifactError"] = None
    persisted_end = perf_counter()
    if on_scene is not None:
        on_scene(row["id"], scene)
    if timings is not None:
        timings["load"] = (rendered_start - started) * 1000
        timings["render"] = (rendered_end - rendered_start) * 1000
        timings["persist"] = (persisted_end - rendered_end) * 1000
        timings["index"] = (perf_counter() - persisted_end) * 1000
    return view


def read_board(store: ResearchStore, project_id: str, identifier: str) -> dict[str, Any]:
    with store.connect() as connection:
        row = connection.execute(
            "SELECT * FROM research_figure_boards WHERE project_id=? AND id=?",
            (project_id, identifier),
        ).fetchone()
        if row is None:
            raise PreparationError(
                "figure-not-found", "This layout is not available in this project."
            )
        return board_view(connection, row)


def list_boards(store: ResearchStore, project_id: str) -> dict[str, Any]:
    with store.connect() as connection:
        rows = connection.execute(
            "SELECT id,spec_json,created_at FROM research_figure_boards "
            "WHERE project_id=? ORDER BY created_at DESC,id DESC",
            (project_id,),
        ).fetchall()
    return {
        "boards": [
            {"id": row["id"], "spec": json.loads(row["spec_json"]), "createdAt": row["created_at"]}
            for row in rows
        ]
    }


def artifact_row(
    store: ResearchStore,
    project_id: str,
    identifier: str,
    field: Literal["svg", "png", "pdf", "scene_json"] | None = None,
) -> dict[str, Any]:
    # Inspecting one row must not load every multi-megabyte export into memory.
    if field is not None and field not in {"svg", "png", "pdf", "scene_json"}:
        raise ValueError("Unknown figure artifact field")
    columns = "a.*" if field is None else "a." + field
    if field == "scene_json":
        # Read frozen UTF-8 directly for the JSON parser. Persistence remains TEXT;
        # avoid decoding a large Unicode string only to re-encode it in from_json.
        columns = "CAST(a.scene_json AS BLOB) AS scene_json"
    with store.connect() as connection:
        row = connection.execute(
            f"SELECT {columns} FROM research_figure_artifacts a JOIN research_figure_boards b "
            "ON b.id=a.board_id WHERE b.project_id=? AND b.id=?",
            (project_id, identifier),
        ).fetchone()
    if row is None:
        with store.connect() as connection:
            pending = connection.execute(
                "SELECT j.state FROM research_figure_jobs j JOIN research_figure_boards b "
                "ON b.id=j.board_id WHERE b.project_id=? AND b.id=?",
                (project_id, identifier),
            ).fetchone()
        if pending:
            raise PreparationError("figure-not-ready", "The saved figure artifacts are not ready.")
        raise PreparationError(
            "figure-not-found", "Save a new layout revision to create composed artifacts."
        )
    return dict(row)


def scene_row_references(scene: dict[str, Any]) -> list[tuple[dict[str, Any], str]]:
    """Read both original flat scenes and compact v2 scenes without copying observations."""
    if "rowSets" not in scene:
        return [(row, row["panel"]) for row in scene["rows"]]
    return [
        (row, label)
        for label, key in scene["panelRowSets"].items()
        for row in scene["rowSets"][key]
    ]


@dataclass
class SceneIndex:
    references: list[tuple[dict[str, Any], str]]
    by_id: dict[str, list[tuple[dict[str, Any], str]]]
    samples_by_id: dict[str, tuple[str, ...]]
    ids_by_sample: dict[str, tuple[str, ...]]
    duplicate_keys: bool = False

    @classmethod
    def build(cls, scene: dict[str, Any]) -> SceneIndex:
        references = scene_row_references(scene)
        # Stable scalar sorts preserve the exact (row ID, panel) order and equal-key
        # row order without allocating one compound key tuple per panel reference.
        references.sort(key=lambda item: item[1])
        references.sort(key=lambda item: item[0]["id"])
        by_id: dict[str, list[tuple[dict[str, Any], str]]] = defaultdict(list)
        # Independent samples usually have one member. Retain a scalar until a
        # second distinct member occurs, avoiding two singleton sets per row.
        samples_by_id: dict[str, str | set[str]] = {}
        ids_by_sample: dict[str, str | set[str]] = {}

        def add(groups: dict[str, str | set[str]], key: str, value: str) -> None:
            previous = groups.get(key)
            if previous is None:
                groups[key] = value
            elif isinstance(previous, str):
                if previous != value:
                    groups[key] = {previous, value}
            else:
                previous.add(value)

        def freeze(groups: dict[str, str | set[str]]) -> dict[str, tuple[str, ...]]:
            return {
                key: (value,) if isinstance(value, str) else tuple(value)
                for key, value in groups.items()
            }

        duplicate_keys = False
        for reference in references:
            row, label = reference
            identifier = row["id"]
            # Reuse the row/panel pair already owned by the sorted table index.
            group = by_id[identifier]
            if group and group[-1][1] == label:
                duplicate_keys = True
            group.append(reference)
            if row.get("sampleId"):
                add(samples_by_id, identifier, row["sampleId"])
                add(ids_by_sample, row["sampleId"], identifier)
        return cls(references, by_id, freeze(samples_by_id), freeze(ids_by_sample), duplicate_keys)


def interaction(
    scene: dict[str, Any],
    mark_id: str | None,
    row_id: str | None,
    offset: int,
    query: str,
    descending: bool,
    index: SceneIndex | None = None,
    *,
    highlights_only: bool = False,
) -> dict[str, Any]:
    marks = scene["marks"]
    references = index.references if index else scene_row_references(scene)
    selected = next((m for m in marks if m["id"] == mark_id), None)
    ids = set(selected["rowIds"] if selected else [row_id] if row_id else [])
    if mark_id and selected is None:
        raise PreparationError("figure-mark", "This mark is not part of the saved figure.")
    if index:
        if row_id and row_id not in index.by_id:
            raise PreparationError("figure-row", "This row is not part of the saved figure.")
        samples = {
            sample for identifier in ids for sample in index.samples_by_id.get(identifier, ())
        }
        linked = set(ids)
        for sample in samples:
            linked.update(index.ids_by_sample[sample])
        rows = (
            [ref for identifier in linked for ref in index.by_id.get(identifier, [])]
            if ids
            else references
        )
    else:
        if row_id and not any(r["id"] == row_id for r, _ in references):
            raise PreparationError("figure-row", "This row is not part of the saved figure.")
        samples = {r["sampleId"] for r, _ in references if r["id"] in ids and r.get("sampleId")}
        linked = {r["id"] for r, _ in references if r["id"] in ids or r.get("sampleId") in samples}
        rows = [(r, label) for r, label in references if not ids or r["id"] in linked]
    if query:
        rows = [
            (r, label)
            for r, label in rows
            if query.casefold() in json.dumps({**r, "panel": label}, ensure_ascii=False).casefold()
        ]
    if index is not None and not ids and not query and (not descending or not index.duplicate_keys):
        # The immutable index already owns the ascending order. Slice before
        # reversing so pagination allocates at most 100 row references.
        if descending:
            end = max(0, len(rows) - offset)
            page = rows[max(0, end - 100) : end][::-1]
        else:
            page = rows[offset : offset + 100]
    else:
        if rows is references:
            rows = list(rows)
        rows.sort(key=lambda item: (item[0]["id"], item[1]), reverse=descending)
        page = rows[offset : offset + 100]
    highlighted = [bool(linked.intersection(mark["rowIds"])) for mark in marks]
    return {
        "marks": [
            {k: v for k, v in m.items() if k != "rowIds"}
            | {
                "inputCount": len(m["rowIds"]),
                "highlighted": is_highlighted,
            }
            for m, is_highlighted in zip(marks, highlighted, strict=True)
            if not highlights_only or is_highlighted
        ],
        "rows": [{**r, "panel": label} for r, label in page],
        "totalRows": len(rows),
        "offset": offset,
        "sampled": scene["sampled"],
        "selectedKind": selected["kind"] if selected else "row" if row_id else None,
        "selectedInputs": len(ids),
        "panels": scene["panels"],
    }
