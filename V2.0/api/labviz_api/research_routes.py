"""Local research API. Existing project ownership checks run before every operation."""

from __future__ import annotations

import json
import threading
from collections.abc import Callable
from typing import Any, Literal

from fastapi import Cookie, Depends, FastAPI, File, Form, Query, UploadFile
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from .config import Settings
from .research_chart_routes import register_chart_routes
from .research_contract import ResearchModel
from .research_figure_routes import register_figure_routes
from .research_preparation import LIMITS, PreparationError, PreparationPlan, parse_sources, prepare
from .research_reuse_routes import register_reuse_routes
from .research_store import ResearchStore, current_view, present


class ApplyResearch(ResearchModel):
    token: str
    expected_revision: str | None = None


class MoveResearch(ResearchModel):
    direction: Literal["undo", "redo"]
    expected_revision: str


def register_research_routes(
    app: FastAPI,
    settings: Settings,
    repository_dependency: Callable[..., Any],
    user_dependency: Callable[..., Any],
    require_access: Callable[..., Any],
    guest_cookie: str,
) -> None:
    # This first local feature uses additive SQLite sidecars; PostgreSQL routes stay unchanged.
    store = (
        ResearchStore(settings.database_path) if settings.persistence_backend == "sqlite" else None
    )
    user_dep = Depends(user_dependency)
    repository_dep = Depends(repository_dependency)
    upload_dep = File(default=None)
    slots = threading.BoundedSemaphore(2)
    execution = threading.Lock()

    @app.exception_handler(PreparationError)
    async def research_error(_request: Any, error: PreparationError) -> JSONResponse:
        status = (
            409
            if error.code
            in {"research-conflict", "preview-expired", "figure-not-ready", "figure-build-state"}
            else 422
        )
        if error.code in {
            "chart-not-found",
            "figure-not-found",
            "template-not-found",
            "output-not-found",
        }:
            status = 404
        if error.code == "research-busy":
            status = 429
        if error.code == "research-local-only":
            status = 501
        return JSONResponse(status_code=status, content={"code": error.code, "message": str(error)})

    def available() -> ResearchStore:
        if store is None:
            raise PreparationError(
                "research-local-only",
                "Research preparation is currently available in the local SQLite profile.",
            )
        return store

    @app.get("/api/v1/projects/{project_id}/research")
    def read_research(
        project_id: str,
        user: Any = user_dep,
        guest_token: str | None = Cookie(default=None, alias=guest_cookie),
        repository: Any = repository_dep,
    ) -> dict[str, Any]:
        require_access(repository, project_id, user, guest_token, ready=True)
        if store is None:
            return {
                "schemaVersion": 1,
                "supported": False,
                "revisionId": None,
                "canUndo": False,
                "canRedo": False,
                "data": None,
            }
        return current_view(store, project_id)

    @app.get("/api/v1/projects/{project_id}/research/page")
    def read_research_page(
        project_id: str,
        kind: Literal["observations", "samples"] = "observations",
        offset: int = Query(default=0, ge=0),
        limit: int = Query(default=100, ge=1, le=200),
        token: str | None = None,
        revision: str | None = None,
        user: Any = user_dep,
        guest_token: str | None = Cookie(default=None, alias=guest_cookie),
        repository: Any = repository_dep,
    ) -> dict[str, Any]:
        require_access(repository, project_id, user, guest_token, ready=True)
        active_store = available()
        if token:
            with active_store.connect() as connection:
                row = connection.execute(
                    "SELECT prepared_json FROM research_drafts WHERE project_id=? AND token=?",
                    (project_id, token),
                ).fetchone()
            if row is None:
                raise PreparationError(
                    "preview-expired", "This preview has expired. Preview again."
                )
        else:
            row = active_store.current(project_id)
            if row is None or row["id"] != revision:
                raise PreparationError(
                    "research-conflict", "The revision changed. Reload research."
                )
        prepared = json.loads(row["prepared_json"])
        return {
            "kind": kind,
            "offset": offset,
            "total": len(prepared[kind]),
            "observations": prepared[kind][offset : offset + limit]
            if kind == "observations"
            else [],
            "samples": prepared[kind][offset : offset + limit] if kind == "samples" else [],
        }

    @app.post("/api/v1/projects/{project_id}/research/inspect")
    def inspect_sources(
        project_id: str,
        files: list[UploadFile] = upload_dep,
        user: Any = user_dep,
        guest_token: str | None = Cookie(default=None, alias=guest_cookie),
        repository: Any = repository_dep,
    ) -> dict[str, Any]:
        require_access(repository, project_id, user, guest_token, ready=True)
        available()
        if not files or len(files) > LIMITS["filesPerImport"]:
            raise PreparationError("file-count-limit", "Choose between one and five files.")
        if not slots.acquire(blocking=False):
            raise PreparationError("research-busy", "Research processing is busy. Retry shortly.")
        try:
            with execution:
                inputs = []
                total = 0
                for file in files:
                    payload = file.file.read(LIMITS["singleFileBytes"] + 1)
                    total += len(payload)
                    if total > LIMITS["aggregateBytes"]:
                        raise PreparationError(
                            "aggregate-byte-limit", "Combined inputs exceed 100 MiB."
                        )
                    inputs.append((file.filename or "data.csv", payload))
                raw = parse_sources(project_id, inputs)
                return {
                    "columns": raw["columns"],
                    "sources": raw["sources"],
                    "rows": len(raw["rows"]),
                }
        finally:
            slots.release()

    @app.post("/api/v1/projects/{project_id}/research/preview")
    def preview_research(
        project_id: str,
        plan_json: str = Form(alias="plan"),
        expected_revision: str | None = Form(default=None, alias="expectedRevision"),
        files: list[UploadFile] | None = upload_dep,
        user: Any = user_dep,
        guest_token: str | None = Cookie(default=None, alias=guest_cookie),
        repository: Any = repository_dep,
    ) -> dict[str, Any]:
        require_access(repository, project_id, user, guest_token, ready=True)
        active_store = available()
        try:
            plan = PreparationPlan.model_validate_json(plan_json)
        except ValidationError as exc:
            raise PreparationError("invalid-research-plan", str(exc)) from exc
        if not slots.acquire(blocking=False):
            raise PreparationError(
                "research-busy", "One research job is running and one is pending. Retry shortly."
            )
        try:
            with execution:
                if files:
                    if len(files) > LIMITS["filesPerImport"]:
                        raise PreparationError("file-count-limit", "Choose at most five files.")
                    inputs = []
                    total = 0
                    for file in files:
                        payload = file.file.read(LIMITS["singleFileBytes"] + 1)
                        total += len(payload)
                        if total > LIMITS["aggregateBytes"]:
                            raise PreparationError(
                                "aggregate-byte-limit", "The combined input exceeds 100 MiB."
                            )
                        inputs.append((file.filename or "data.csv", payload))
                    raw = parse_sources(project_id, inputs)
                else:
                    current = active_store.current(project_id)
                    if current is None:
                        raise PreparationError(
                            "research-input-required",
                            "Choose source files for the first research revision.",
                        )
                    raw = json.loads(current["raw_json"])
                prepared = prepare(raw, plan)
                token = active_store.stage(project_id, expected_revision, raw, prepared)
                return {
                    "token": token,
                    "expectedRevision": expected_revision,
                    "data": present(prepared),
                }
        finally:
            slots.release()

    @app.post("/api/v1/projects/{project_id}/research/apply")
    def apply_research(
        project_id: str,
        body: ApplyResearch,
        user: Any = user_dep,
        guest_token: str | None = Cookie(default=None, alias=guest_cookie),
        repository: Any = repository_dep,
    ) -> dict[str, Any]:
        require_access(repository, project_id, user, guest_token, ready=True)
        active_store = available()
        active_store.apply(project_id, body.token, body.expected_revision)
        return current_view(active_store, project_id)

    @app.post("/api/v1/projects/{project_id}/research/history")
    def move_research(
        project_id: str,
        body: MoveResearch,
        user: Any = user_dep,
        guest_token: str | None = Cookie(default=None, alias=guest_cookie),
        repository: Any = repository_dep,
    ) -> dict[str, Any]:
        require_access(repository, project_id, user, guest_token, ready=True)
        active_store = available()
        active_store.move(project_id, body.expected_revision, body.direction)
        return current_view(active_store, project_id)

    register_chart_routes(
        app,
        store,
        available,
        user_dep,
        repository_dep,
        require_access,
        guest_cookie,
        slots,
        execution,
    )

    register_figure_routes(
        app,
        store,
        available,
        user_dep,
        repository_dep,
        require_access,
        guest_cookie,
        slots,
        execution,
    )

    register_reuse_routes(
        app,
        store,
        available,
        user_dep,
        repository_dep,
        require_access,
        guest_cookie,
        slots,
        execution,
    )
