"""Private local figure layouts, frozen artifacts and restartable generation."""

from __future__ import annotations

import gc
import json
from collections.abc import Callable
from time import perf_counter
from typing import Any, Literal

from fastapi import Cookie, FastAPI, Query
from fastapi.responses import Response
from pydantic_core import from_json, to_json

from .processing import ProcessingError
from .research_figure_board import (
    FigureBoardSpec,
    SceneIndex,
    artifact_row,
    install_boards,
    interaction,
    list_boards,
    read_board,
    save_board,
)
from .research_figure_jobs import FigureJobs
from .research_figure_render import render_board
from .research_figure_sources import legacy_snapshot, save_legacy
from .research_preparation import PreparationError
from .research_preview import clear_previews, preview_png
from .research_store import ResearchStore


def register_figure_routes(
    app: FastAPI,
    store: ResearchStore | None,
    available: Callable[[], ResearchStore],
    user_dep: Any,
    repository_dep: Any,
    require_access: Callable[..., Any],
    guest_cookie: str,
    slots: Any,
    execution: Any,
) -> None:
    jobs = None
    if store is not None:
        install_boards(store)

    # One immutable board per application, at most 200,000 distinct observations.
    # Access and existence checks always precede reuse. No export bytes are retained.
    cached: tuple[str, str, dict[str, Any], SceneIndex] | None = None

    def remember(project_id: str, board_id: str, scene: dict[str, Any]) -> None:
        nonlocal cached
        cached = None
        distinct = (
            sum(len(rows) for rows in scene.get("rowSets", {}).values())
            if "rowSets" in scene
            else len(scene["rows"])
        )
        if distinct <= 200_000:
            cached = (project_id, board_id, scene, SceneIndex.build(scene))

    def publish_scene(
        project_id: str, board_id: str, scene: dict[str, Any], index: SceneIndex | None
    ) -> None:
        nonlocal cached
        cached = (project_id, board_id, scene, index) if index is not None else None

    def evict_scene() -> None:
        nonlocal cached
        if cached is not None:
            cached = None
            gc.collect()

    app.state.evict_research_scene = evict_scene

    def warm_preview(project_id: str, board_id: str, payload: bytes) -> None:
        preview_png(project_id, board_id, payload, 1400)
        # Deletion can precede the start of this optional work as well as occur
        # during resizing. Never leave a late warm-up cached for a deleted board.
        with available().connect() as connection:
            exists = connection.execute(
                "SELECT 1 FROM research_figure_artifacts a JOIN research_figure_boards b "
                "ON b.id=a.board_id WHERE b.project_id=? AND b.id=?",
                (project_id, board_id),
            ).fetchone()
        if exists is None:
            clear_previews(project_id)

    if store is not None:
        jobs = FigureJobs(store, slots, execution, publish_scene, evict_scene, warm_preview)
        app.state.figure_jobs = jobs

    def clear_project(project_id: str) -> None:
        nonlocal cached
        if cached is not None and cached[0] == project_id:
            cached = None
        clear_previews(project_id)

    app.state.clear_research_caches = clear_project

    def interaction_response(data: dict[str, Any]) -> Response:
        # Immutable JSON primitives bypass recursive FastAPI jsonable_encoder work.
        return Response(
            to_json(data),
            media_type="application/json",
            headers={"Cache-Control": "private, no-store"},
        )

    @app.get("/api/v1/projects/{project_id}/research/figure-boards")
    def history(
        project_id: str,
        user: Any = user_dep,
        guest_token: str | None = Cookie(default=None, alias=guest_cookie),
        repository: Any = repository_dep,
    ) -> dict[str, Any]:
        require_access(repository, project_id, user, guest_token, ready=True)
        return list_boards(available(), project_id)

    @app.post("/api/v1/projects/{project_id}/research/figure-boards")
    def save(
        project_id: str,
        body: FigureBoardSpec,
        response: Response,
        defer_artifacts: bool = Query(default=False, alias="deferArtifacts"),
        user: Any = user_dep,
        guest_token: str | None = Cookie(default=None, alias=guest_cookie),
        repository: Any = repository_dep,
    ) -> dict[str, Any]:
        require_access(repository, project_id, user, guest_token, ready=True)
        if defer_artifacts:
            available()
            assert jobs is not None
            started = perf_counter()
            view = jobs.save(project_id, body)
            response.status_code = 202
            response.headers["Server-Timing"] = (
                f"durableLayout;dur={(perf_counter() - started) * 1000:.2f}"
            )
            return view
        if not slots.acquire(blocking=False):
            raise PreparationError("research-busy", "Research processing is busy. Retry shortly.")
        large = False
        try:
            with execution:
                evict_scene()
                timings: dict[str, float] = {}
                view = save_board(
                    available(),
                    project_id,
                    body,
                    lambda identifier, scene: remember(project_id, identifier, scene),
                    timings,
                    batch_sources=True,
                )
                large = sum(panel["observations"] for panel in view["panels"]) > 20_000
                response.headers["Server-Timing"] = ", ".join(
                    f"{key};dur={value:.2f}" for key, value in timings.items()
                )
                return view
        finally:
            slots.release()
            if large:
                gc.collect()

    @app.post(
        "/api/v1/projects/{project_id}/research/figure-boards/{board_id}/retry", status_code=202
    )
    def retry(
        project_id: str,
        board_id: str,
        user: Any = user_dep,
        guest_token: str | None = Cookie(default=None, alias=guest_cookie),
        repository: Any = repository_dep,
    ) -> dict[str, Any]:
        require_access(repository, project_id, user, guest_token, ready=True)
        available()
        assert jobs is not None
        return jobs.retry(project_id, board_id)

    @app.get("/api/v1/projects/{project_id}/research/figure-boards/{board_id}")
    def read(
        project_id: str,
        board_id: str,
        user: Any = user_dep,
        guest_token: str | None = Cookie(default=None, alias=guest_cookie),
        repository: Any = repository_dep,
    ) -> dict[str, Any]:
        require_access(repository, project_id, user, guest_token, ready=True)
        return read_board(available(), project_id, board_id)

    @app.get("/api/v1/projects/{project_id}/research/figure-sources")
    def catalog(
        project_id: str,
        user: Any = user_dep,
        guest_token: str | None = Cookie(default=None, alias=guest_cookie),
        repository: Any = repository_dep,
    ) -> dict[str, Any]:
        require_access(repository, project_id, user, guest_token, ready=True)
        with available().connect() as connection:
            charts = [
                {
                    "id": r["id"],
                    "revisionId": r["revision_id"],
                    "createdAt": r["created_at"],
                    "spec": json.loads(r["spec_json"]),
                    "legacy": False,
                }
                for r in connection.execute(
                    "SELECT c.id,c.revision_id, "
                    "coalesce(json_extract(h.header_json, '$.spec'), "
                    "json_extract(c.result_json, '$.spec')) AS spec_json, "
                    "c.created_at FROM research_charts c LEFT JOIN research_panel_headers h "
                    "ON h.id=c.id AND h.project_id=c.project_id "
                    "WHERE c.project_id=? ORDER BY c.created_at DESC",
                    (project_id,),
                )
            ]
            for r in connection.execute(
                "SELECT c.id,c.created_at, "
                "coalesce(json_extract(h.header_json, '$.revisionId'), "
                "json_extract(c.result_json, '$.revisionId')) AS revision_id, "
                "coalesce(json_extract(h.header_json, '$.kind'), "
                "json_extract(c.result_json, '$.kind')) AS kind, "
                "coalesce(json_extract(h.header_json, '$.title'), "
                "json_extract(c.result_json, '$.title')) AS title "
                "FROM research_figure_sources c LEFT JOIN research_panel_headers h "
                "ON h.id=c.id AND h.project_id=c.project_id "
                "WHERE c.project_id=? ORDER BY c.created_at DESC",
                (project_id,),
            ):
                charts.append(
                    {
                        "id": r["id"],
                        "revisionId": r["revision_id"],
                        "createdAt": r["created_at"],
                        "spec": {"kind": r["kind"], "title": r["title"]},
                        "legacy": True,
                    }
                )
        return {"charts": charts}

    @app.post("/api/v1/projects/{project_id}/research/figure-sources")
    def capture(
        project_id: str,
        panel: int = Query(default=1, ge=1, le=4),
        sample_field: str | None = Query(default=None, alias="sampleField", max_length=128),
        user: Any = user_dep,
        guest_token: str | None = Cookie(default=None, alias=guest_cookie),
        repository: Any = repository_dep,
    ) -> dict[str, str]:
        require_access(repository, project_id, user, guest_token, ready=True)
        active = available()
        if not slots.acquire(blocking=False):
            raise PreparationError("research-busy", "Research processing is busy. Retry shortly.")
        try:
            with execution:
                evict_scene()
                try:
                    # Sample linkage needs one scalar, not both complete revision snapshots.
                    with active.connect() as connection:
                        current = connection.execute(
                            "SELECT json_extract(r.raw_json, '$.namespace') AS namespace "
                            "FROM research_heads h JOIN research_revisions r ON r.id=h.revision_id "
                            "WHERE h.project_id=?",
                            (project_id,),
                        ).fetchone()
                    namespace = current["namespace"] if current else None
                    result = legacy_snapshot(repository, project_id, panel, sample_field, namespace)
                    # Validate renderability before retaining the immutable source.
                    render_board([result], FigureBoardSpec(panel_chart_ids=["validation"]))
                except ProcessingError as error:
                    raise PreparationError(error.code, str(error)) from error
                identifier = save_legacy(active, project_id, result)
                large = len(result["rows"]) > 20_000
                del result
                if large:
                    gc.collect()
                return {"id": identifier}
        finally:
            slots.release()

    @app.get("/api/v1/projects/{project_id}/research/figure-boards/{board_id}/artifact/{format}")
    def artifact(
        project_id: str,
        board_id: str,
        format: Literal["svg", "png", "pdf"],
        download: bool = False,
        preview: bool = False,
        user: Any = user_dep,
        guest_token: str | None = Cookie(default=None, alias=guest_cookie),
        repository: Any = repository_dep,
    ) -> Response:
        require_access(repository, project_id, user, guest_token, ready=True)
        row = artifact_row(available(), project_id, board_id, format)
        payload = row[format]
        if preview and format == "png" and not download:
            payload = preview_png(project_id, board_id, payload, 1400)
        return Response(
            payload,
            media_type={"svg": "image/svg+xml", "png": "image/png", "pdf": "application/pdf"}[
                format
            ],
            headers={
                "Cache-Control": "private, no-store",
                "Content-Security-Policy": "default-src 'none'; style-src 'unsafe-inline'; sandbox",
                "Content-Disposition": (
                    f"{'attachment' if download else 'inline'}; "
                    f'filename="figure-{board_id}.{format}"'
                ),
            },
        )

    @app.get("/api/v1/projects/{project_id}/research/figure-boards/{board_id}/interaction")
    def inspect(
        project_id: str,
        board_id: str,
        mark: str | None = None,
        row: str | None = None,
        offset: int = Query(default=0, ge=0),
        query: str = Query(default="", max_length=200),
        descending: bool = False,
        highlights_only: bool = Query(default=False, alias="highlightsOnly"),
        user: Any = user_dep,
        guest_token: str | None = Cookie(default=None, alias=guest_cookie),
        repository: Any = repository_dep,
    ) -> Response:
        require_access(repository, project_id, user, guest_token, ready=True)
        active = available()
        with active.connect() as connection:
            exists = connection.execute(
                "SELECT 1 FROM research_figure_boards b "
                "JOIN research_figure_artifacts a ON a.board_id=b.id "
                "WHERE b.project_id=? AND b.id=?",
                (project_id, board_id),
            ).fetchone()
        if exists is None:
            raise PreparationError("figure-not-found", "This composed figure is not available.")
        snapshot = cached
        if snapshot is None or snapshot[:2] != (project_id, board_id):
            # Acyclic frozen JSON/index preparation shares the research execution lock.
            # One cold build at a time; do not run cyclic collection on every row allocation.
            with execution:
                snapshot = cached
                if snapshot is None or snapshot[:2] != (project_id, board_id):
                    restore_gc = gc.isenabled()
                    if restore_gc:
                        gc.disable()
                    try:
                        scene = from_json(
                            artifact_row(active, project_id, board_id, "scene_json")["scene_json"]
                        )
                        remember(project_id, board_id, scene)
                    finally:
                        if restore_gc:
                            gc.enable()
                    snapshot = cached
                    if snapshot is None or snapshot[:2] != (project_id, board_id):
                        return interaction_response(
                            interaction(
                                scene,
                                mark,
                                row,
                                offset,
                                query,
                                descending,
                                highlights_only=highlights_only,
                            )
                        )
        return interaction_response(
            interaction(
                snapshot[2],
                mark,
                row,
                offset,
                query,
                descending,
                snapshot[3],
                highlights_only=highlights_only,
            )
        )
