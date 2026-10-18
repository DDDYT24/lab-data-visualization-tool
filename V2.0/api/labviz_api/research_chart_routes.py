"""Owner-protected local scientific chart history, preview, data and exports."""

from __future__ import annotations

import gc
import json
from collections.abc import Callable
from time import perf_counter
from typing import Any, Literal

from fastapi import Cookie, FastAPI, Query
from fastapi.responses import Response
from pydantic_core import from_json

from .research_chart_render import render_chart
from .research_chart_store import chart_row, chart_view, install_charts, read_chart_page, save_chart
from .research_charts import ScientificChartSpec, compute_chart
from .research_contract import ResearchModel
from .research_preparation import PreparationError
from .research_preview import preview_png
from .research_store import ResearchStore


class CreateScientificChart(ResearchModel):
    revision_id: str
    spec: ScientificChartSpec


def register_chart_routes(
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
    if store is not None:
        install_charts(store)

    @app.get("/api/v1/projects/{project_id}/research/charts")
    def list_charts(
        project_id: str,
        user: Any = user_dep,
        guest_token: str | None = Cookie(default=None, alias=guest_cookie),
        repository: Any = repository_dep,
    ) -> dict[str, Any]:
        require_access(repository, project_id, user, guest_token, ready=True)
        active = available()
        with active.connect() as connection:
            rows = connection.execute(
                "SELECT c.id, c.revision_id, "
                "coalesce(json_extract(h.header_json, '$.spec'), "
                "json_extract(c.result_json, '$.spec')) AS spec_json, "
                "c.created_at FROM research_charts c LEFT JOIN research_panel_headers h "
                "ON h.id=c.id AND h.project_id=c.project_id "
                "WHERE c.project_id=? ORDER BY c.created_at DESC, c.id DESC",
                (project_id,),
            ).fetchall()
        return {
            "charts": [
                {
                    "id": row["id"],
                    "revisionId": row["revision_id"],
                    "createdAt": row["created_at"],
                    "spec": json.loads(row["spec_json"]),
                }
                for row in rows
            ]
        }

    @app.post("/api/v1/projects/{project_id}/research/charts")
    def create_chart(
        project_id: str,
        body: CreateScientificChart,
        response: Response,
        user: Any = user_dep,
        guest_token: str | None = Cookie(default=None, alias=guest_cookie),
        repository: Any = repository_dep,
    ) -> dict[str, Any]:
        require_access(repository, project_id, user, guest_token, ready=True)
        active = available()
        if not slots.acquire(blocking=False):
            raise PreparationError("research-busy", "Research processing is busy. Retry shortly.")
        large = False
        try:
            with execution:
                started = perf_counter()
                app.state.evict_research_scene()
                current = active.current(project_id, prepared_only=True)
                if current is None or current["id"] != body.revision_id:
                    raise PreparationError(
                        "research-conflict", "Reload the applied research revision."
                    )
                # Batch one bounded compute/render/save under the execution lock.
                # Release the large acyclic snapshot before collecting the
                # renderer's cycles, instead of repeatedly scanning live data.
                cyclic_enabled = gc.isenabled()
                if cyclic_enabled:
                    gc.disable()
                try:
                    prepared = from_json(
                        current["prepared_json"], cache_strings="all", allow_inf_nan=False
                    )
                    del current
                    result = compute_chart(prepared, body.spec)
                    del prepared
                    large = len(result["points"]) > 20_000
                    computed = perf_counter()
                    artifacts = render_chart(result)
                    rendered = perf_counter()
                    identifier = save_chart(active, project_id, body.revision_id, result, artifacts)
                    stored = perf_counter()
                    view = chart_view(
                        chart_row(active, project_id, identifier, header_only=True), result=result
                    )
                    # The response owns only its bounded first page. Release the full
                    # snapshot and exports before collecting Matplotlib's cycles.
                    del result, artifacts
                finally:
                    if cyclic_enabled:
                        gc.enable()
                response.headers["Server-Timing"] = (
                    f"compute;dur={(computed - started) * 1000:.2f}, "
                    f"render;dur={(rendered - computed) * 1000:.2f}, "
                    f"persist;dur={(stored - rendered) * 1000:.2f}"
                )
                return view
        finally:
            slots.release()
            if large:
                gc.collect()

    @app.get("/api/v1/projects/{project_id}/research/charts/{chart_id}")
    def read_chart(
        project_id: str,
        chart_id: str,
        offset: int = Query(default=0, ge=0),
        user: Any = user_dep,
        guest_token: str | None = Cookie(default=None, alias=guest_cookie),
        repository: Any = repository_dep,
    ) -> dict[str, Any]:
        require_access(repository, project_id, user, guest_token, ready=True)
        return read_chart_page(available(), project_id, chart_id, offset)

    @app.get("/api/v1/projects/{project_id}/research/charts/{chart_id}/artifact/{format}")
    def chart_artifact(
        project_id: str,
        chart_id: str,
        format: Literal["svg", "png", "pdf", "json"],
        download: bool = False,
        preview: bool = False,
        preview_width: int = Query(default=960, ge=160, le=1400, alias="previewWidth"),
        user: Any = user_dep,
        guest_token: str | None = Cookie(default=None, alias=guest_cookie),
        repository: Any = repository_dep,
    ) -> Response:
        require_access(repository, project_id, user, guest_token, ready=True)
        row = chart_row(available(), project_id, chart_id, format)
        result = json.loads(row["result_json"]) if format == "json" else None
        payload = (
            row[format]
            if format != "json"
            else json.dumps(
                {"chartId": chart_id, "revisionId": row["revision_id"], "result": result},
                ensure_ascii=False,
                allow_nan=False,
            ).encode("utf-8")
        )
        if preview and format == "png" and not download:
            payload = preview_png(project_id, chart_id, payload, preview_width)
        mime = {
            "svg": "image/svg+xml",
            "png": "image/png",
            "pdf": "application/pdf",
            "json": "application/json",
        }[format]
        headers = {
            "Cache-Control": "private, no-store",
            "X-LabViz-Result-SHA256": result["resultSha256"] if result else row["result_sha256"],
            "Content-Security-Policy": "default-src 'none'; style-src 'unsafe-inline'; sandbox",
            "Content-Disposition": (
                f'{"attachment" if download else "inline"}; filename="research-{chart_id}.{format}"'
            ),
        }
        return Response(payload, media_type=mime, headers=headers)
