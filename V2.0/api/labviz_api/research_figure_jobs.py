"""Durable layout revisions and a bounded, restartable local artifact queue."""

from __future__ import annotations

import gc
import json
import logging
import threading
from collections.abc import Callable
from datetime import UTC, datetime
from time import perf_counter
from typing import Any
from uuid import uuid4

from pydantic_core import to_json

from .research_figure_board import FigureBoardSpec, SceneIndex, read_board
from .research_figure_render import render_board
from .research_figure_sources import source
from .research_panel_headers import remember_header, source_header
from .research_preparation import PreparationError
from .research_store import ResearchStore

LOGGER = logging.getLogger(__name__)


class FigureJobs:
    """One source API process, one executing job, at most one waiting job.

    The source launcher owns the database/profile mutex. This is not a multi-host worker.
    Existing synchronous research requests share the same slots and execution lock.
    """

    def __init__(
        self,
        store: ResearchStore,
        slots: Any,
        execution: Any,
        on_ready: Callable[[str, str, dict[str, Any], SceneIndex | None], None] | None = None,
        on_start: Callable[[], None] | None = None,
        on_preview: Callable[[str, str, bytes], None] | None = None,
    ) -> None:
        self.store, self.slots, self.execution = store, slots, execution
        self.on_ready = on_ready
        self.on_start = on_start
        self.on_preview = on_preview
        self.lock = threading.Lock()
        self.wake = threading.Event()
        self.thread: threading.Thread | None = None
        self.stopping = False
        self.owned: dict[str, None] = {}

    def start(self) -> None:
        with self.lock:
            if self.thread is not None:
                return
            self.stopping = False
            with self.store.connect() as connection:
                # A crash cannot publish half of an artifact set. Interrupted immutable
                # revisions restart from their saved bindings; committed sets remain untouched.
                connection.execute(
                    "UPDATE research_figure_jobs SET state='queued', attempt=NULL "
                    "WHERE state='running'"
                )
                for row in connection.execute(
                    "SELECT board_id FROM research_figure_jobs WHERE state='queued' ORDER BY rowid"
                ):
                    if self.slots.acquire(blocking=False):
                        self.owned[row["board_id"]] = None
            self.thread = threading.Thread(
                target=self._run, name="labviz-figure-build", daemon=True
            )
            self.thread.start()
            self.wake.set()

    def close(self) -> None:
        with self.lock:
            self.stopping = True
            thread = self.thread
            self.wake.set()
        if thread is not None:
            thread.join()
        with self.lock:
            for _ in self.owned:
                self.slots.release()
            self.owned.clear()
            self.thread = None

    def save(self, project_id: str, spec: FigureBoardSpec) -> dict[str, Any]:
        self.start()
        if not self.slots.acquire(blocking=False):
            raise PreparationError("research-busy", "One job is running and one is waiting.")
        identifier = uuid4().hex
        try:
            with self.store.connect() as connection:
                connection.execute("BEGIN IMMEDIATE")
                headers = {}
                for chart_id in dict.fromkeys(spec.panel_chart_ids):
                    cached = connection.execute(
                        "SELECT header_json FROM research_panel_headers "
                        "WHERE id=? AND project_id=?",
                        (chart_id, project_id),
                    ).fetchone()
                    exists = connection.execute(
                        "SELECT 1 FROM research_charts WHERE id=? AND project_id=? "
                        "UNION ALL SELECT 1 FROM research_figure_sources "
                        "WHERE id=? AND project_id=?",
                        (chart_id, project_id, chart_id, project_id),
                    ).fetchone()
                    if exists is None:
                        raise PreparationError(
                            "figure-chart-unavailable",
                            "Choose saved panels belonging to this project.",
                        )
                    if cached:
                        headers[chart_id] = json.loads(cached["header_json"])
                    else:
                        # Older charts are upgraded once, without rewriting their frozen bytes.
                        headers[chart_id] = source_header(source(connection, project_id, chart_id))
                        remember_header(connection, project_id, chart_id, headers[chart_id])
                panels = []
                for index, chart_id in enumerate(spec.panel_chart_ids):
                    header = headers[chart_id]
                    panels.append(
                        {
                            **{
                                key: value
                                for key, value in header.items()
                                if key not in {"xSignature", "ySignature", "spec", "firstPage"}
                            },
                            "label": "ABCD"[index],
                            "chartId": chart_id,
                            "title": (spec.panel_titles[index] if spec.panel_titles else "")
                            or header["title"],
                        }
                    )
                for dimension in ("x", "y"):
                    if spec.axis_mode in {"shared_" + dimension, "shared_xy"}:
                        signatures = {
                            json.dumps(headers[key][dimension + "Signature"])
                            for key in spec.panel_chart_ids
                        }
                        if len(signatures) != 1:
                            raise PreparationError(
                                "figure-incompatible-axis",
                                f"Shared {dimension} axes need matching meanings, "
                                "units and categories.",
                            )
                if (
                    connection.execute(
                        "SELECT count(*) FROM research_figure_jobs "
                        "WHERE state IN ('queued','running')"
                    ).fetchone()[0]
                    >= 2
                ):
                    raise PreparationError("research-busy", "The figure queue is full.")
                connection.execute(
                    "INSERT INTO research_figure_boards VALUES (?,?,?,?)",
                    (identifier, project_id, spec.model_dump_json(), datetime.now(UTC).isoformat()),
                )
                connection.execute(
                    "INSERT INTO research_figure_jobs "
                    "(board_id,state,panels_json) VALUES (?,'queued',?)",
                    (identifier, json.dumps(panels, ensure_ascii=False, allow_nan=False)),
                )
            # Success is returned only after the revision and the pending job commit together.
            view = read_board(self.store, project_id, identifier)
            with self.lock:
                self.owned[identifier] = None
                self.wake.set()
            return view
        except BaseException:
            self.slots.release()
            raise

    def retry(self, project_id: str, identifier: str) -> dict[str, Any]:
        self.start()
        if not self.slots.acquire(blocking=False):
            raise PreparationError("research-busy", "Research processing is busy. Retry shortly.")
        try:
            with self.store.connect() as connection:
                connection.execute("BEGIN IMMEDIATE")
                row = connection.execute(
                    "SELECT j.state FROM research_figure_jobs j JOIN research_figure_boards b "
                    "ON b.id=j.board_id WHERE b.id=? AND b.project_id=?",
                    (identifier, project_id),
                ).fetchone()
                if row is None:
                    raise PreparationError(
                        "figure-not-found", "This generated revision is unavailable."
                    )
                if row["state"] != "failed":
                    raise PreparationError(
                        "figure-build-state", "Only a failed generation can be retried."
                    )
                connection.execute(
                    "UPDATE research_figure_jobs SET state='queued',error=NULL,attempt=NULL "
                    "WHERE board_id=? AND state='failed'",
                    (identifier,),
                )
            view = read_board(self.store, project_id, identifier)
            with self.lock:
                self.owned[identifier] = None
                self.wake.set()
            return view
        except BaseException:
            self.slots.release()
            raise

    def _run(self) -> None:
        while True:
            self.wake.wait(0.5)
            self.wake.clear()
            with self.lock:
                if self.stopping:
                    return
                identifiers = tuple(self.owned)
            for identifier in identifiers:
                with self.lock:
                    if self.stopping:
                        return
                try:
                    with self.execution:
                        self._build(identifier)
                except Exception:
                    LOGGER.exception("Figure generation failed for %s", identifier)
                    with self.store.connect() as connection:
                        connection.execute(
                            "UPDATE research_figure_jobs SET state='failed', error=? "
                            "WHERE board_id=? AND state IN ('queued','running')",
                            (
                                "Generation failed; layout retained. Retry generation.",
                                identifier,
                            ),
                        )
                finally:
                    with self.lock:
                        if identifier in self.owned:
                            self.owned.pop(identifier)
                            self.slots.release()

    def _build(self, identifier: str) -> None:
        attempt = uuid4().hex
        started = perf_counter()
        with self.store.connect() as connection:
            row = connection.execute(
                "SELECT b.*,j.panels_json FROM research_figure_boards b "
                "JOIN research_figure_jobs j ON j.board_id=b.id "
                "WHERE b.id=? AND j.state='queued'",
                (identifier,),
            ).fetchone()
            if row is None:
                return
            connection.execute(
                "UPDATE research_figure_jobs SET state='running',attempt=? "
                "WHERE board_id=? AND state='queued'",
                (attempt, identifier),
            )
        if self.on_start:
            self.on_start()
        spec = FigureBoardSpec.model_validate_json(row["spec_json"])
        # One bounded build holds the shared research execution lock. Avoid
        # automatic scans of the large live snapshots while Matplotlib allocates
        # its small objects; collect its cycles once after releasing the sources.
        restore_gc = gc.isenabled()
        if restore_gc:
            gc.disable()
        try:
            with self.store.connect() as connection:
                loaded = {
                    key: source(connection, row["project_id"], key)
                    for key in dict.fromkeys(spec.panel_chart_ids)
                }
            panels = json.loads(row["panels_json"])
            for panel in panels:
                if loaded[panel["chartId"]]["resultSha256"] != panel["resultSha256"]:
                    raise PreparationError(
                        "figure-source-changed", "The frozen source fingerprint changed."
                    )
            load_end = perf_counter()
            # Rendering runs without a SQLite connection or write transaction.
            formats: dict[str, float] = {}
            artifacts, scene = render_board(
                [loaded[key] for key in spec.panel_chart_ids], spec, timings=formats
            )
            render_end = perf_counter()
            scene["panelMetadata"] = panels
            del loaded
            gc.collect()
            cleanup_end = perf_counter()
        finally:
            if restore_gc:
                gc.enable()
        scene_json = to_json(scene, ensure_ascii=False, round_trip=True).decode("utf-8")
        serialized = perf_counter()
        # Prepare the bounded interaction cache before publication, without a writer lock.
        # First inspection reuses these exact rows instead of decoding the large JSON again.
        distinct = sum(len(rows) for rows in scene["rowSets"].values())
        restore_index_gc = gc.isenabled()
        if restore_index_gc:
            gc.disable()
        try:
            index = SceneIndex.build(scene) if self.on_ready and distinct <= 200_000 else None
        finally:
            if restore_index_gc:
                gc.enable()
        indexed = perf_counter()
        with self.store.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            current = connection.execute(
                "SELECT attempt,state FROM research_figure_jobs WHERE board_id=?",
                (identifier,),
            ).fetchone()
            if current is None or current["attempt"] != attempt or current["state"] != "running":
                return
            connection.execute(
                "INSERT INTO research_figure_artifacts VALUES (?,?,?,?,?)",
                (identifier, artifacts["svg"], artifacts["png"], artifacts["pdf"], scene_json),
            )
            timings = {
                **formats,
                "load": (load_end - started) * 1000,
                "render": (render_end - load_end) * 1000,
                "index": (indexed - serialized) * 1000,
                "cleanup": (cleanup_end - render_end) * 1000,
                "serialization": (serialized - cleanup_end) * 1000,
                "serializeAndWriteBeforeCommit": (
                    serialized - render_end + perf_counter() - indexed
                )
                * 1000,
            }
            connection.execute(
                "UPDATE research_figure_jobs SET state='ready',error=NULL,timings_json=? "
                "WHERE board_id=? AND attempt=?",
                (json.dumps(timings), identifier, attempt),
            )
            # Install the already-built cache before commit makes ready visible.
            # Inspect checks the committed artifact row first; failed commits cannot
            # expose this cache. This avoids a first-read race with post-commit publication.
            if self.on_ready:
                self.on_ready(row["project_id"], identifier, scene, index)
            commit_started = perf_counter()
        LOGGER.info(
            "Figure artifacts committed: commit=%.2fms total=%.2fms",
            (perf_counter() - commit_started) * 1000,
            (perf_counter() - started) * 1000,
        )
        preview_payload = artifacts["png"] if self.on_preview else None
        del artifacts, scene, scene_json
        # Optional screen work follows the atomic publication of all three
        # immutable exports. A failed preview never invalidates a ready board.
        if self.on_preview and preview_payload is not None:
            try:
                self.on_preview(row["project_id"], identifier, preview_payload)
            except Exception:
                LOGGER.debug("Screen preview warm-up failed; retry on demand", exc_info=True)
