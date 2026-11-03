"""Installed desktop admission gate; holds activity through response and background work."""

from __future__ import annotations

import secrets
import sqlite3
import threading
from collections.abc import Iterator
from contextlib import closing, contextmanager
from pathlib import Path
from typing import Any
from uuid import UUID

from fastapi import FastAPI, Header, HTTPException
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send

from .installed_profile import InstalledProfile

CONTROL = frozenset(
    "/api/v1/" + suffix
    for suffix in (
        "health",
        "ready",
        "local/session",
        "local/instance",
        "local/shutdown",
        "local/activity",
        "local/drain",
        "local/cancel-drain",
        "local/draft-bases",
        "local/draft-figures",
    )
)


class Activity:
    def __init__(self, database: Path, app: FastAPI) -> None:
        self.database, self.app = database, app
        self.lock = threading.RLock()
        self.active = 0
        self.draining = False
        self.generation = 0

    def begin(self) -> bool:
        with self.lock:
            if self.draining:
                return False
            self.active += 1
            return True

    def end(self) -> None:
        with self.lock:
            self.active -= 1
            assert self.active >= 0

    @contextmanager
    def operation(self) -> Iterator[bool]:
        entered = self.begin()
        try:
            yield entered
        finally:
            if entered:
                self.end()

    def status(self) -> dict[str, Any]:
        with self.lock:
            with closing(
                sqlite3.connect(self.database.as_uri() + "?mode=ro", uri=True, timeout=2)
            ) as db:
                imports = db.execute(
                    "SELECT count(*) FROM jobs "
                    "WHERE stage IN ('queued','uploading','parsing','profiling')"
                ).fetchone()[0]
                figures = db.execute(
                    "SELECT count(*) FROM research_figure_jobs WHERE state IN ('queued','running')"
                ).fetchone()[0]
            jobs = getattr(self.app.state, "figure_jobs", None)
            owned = 0
            if jobs is not None:
                with jobs.lock:
                    owned = len(jobs.owned)
            return {
                "format": "labviz-desktop-activity-v1",
                "draining": self.draining,
                "generation": self.generation,
                "requests": self.active,
                "imports": imports,
                "figures": figures,
                "ownedFigureTasks": owned,
                "idle": self.active == imports == figures == owned == 0,
            }

    def drain(self) -> dict[str, Any]:
        with self.lock:
            self.draining = True
            self.generation += 1
            return self.status()

    def cancel(self) -> dict[str, Any]:
        with self.lock:
            self.draining = False
            self.generation += 1
            return self.status()


class ActivityMiddleware:
    def __init__(self, app: ASGIApp, activity: Activity) -> None:
        self.app, self.activity = app, activity

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope["path"] in CONTROL:
            return await self.app(scope, receive, send)
        if not self.activity.begin():
            return await JSONResponse(
                {
                    "code": "desktop-draining",
                    "message": "The local workspace is pausing. Reopen LabViz to continue.",
                },
                status_code=503,
                headers={"Retry-After": "1"},
            )(scope, receive, send)
        try:
            # Includes upload receive, final streaming send and Starlette BackgroundTasks.
            return await self.app(scope, receive, send)
        finally:
            self.activity.end()


def install(app: FastAPI, profile: InstalledProfile, key: str) -> None:
    activity = Activity(app.state.settings.database_path, app)
    app.state.desktop_activity = activity
    app.add_middleware(ActivityMiddleware, activity=activity)

    def authorized(
        local_key: str | None = Header(default=None, alias="X-LabViz-Local-Key"),
    ) -> None:
        if not local_key or not secrets.compare_digest(local_key, key):
            raise HTTPException(status_code=403, detail="Invalid local launch credential.")

    from fastapi import Depends

    gate = Depends(authorized)

    @app.get("/api/v1/local/activity", dependencies=[gate])
    def status() -> dict[str, Any]:
        return {**activity.status(), "profileId": profile.identifier}

    @app.post("/api/v1/local/drain", dependencies=[gate])
    def drain() -> dict[str, Any]:
        return {**activity.drain(), "profileId": profile.identifier}

    @app.post("/api/v1/local/cancel-drain", dependencies=[gate])
    def cancel() -> dict[str, Any]:
        return {**activity.cancel(), "profileId": profile.identifier}

    @app.post("/api/v1/local/draft-bases", dependencies=[gate])
    def bases(projects: list[str]) -> dict[str, Any]:
        if len(projects) > 100 or any(len(p) > 128 for p in projects):
            raise HTTPException(400, "Too many checkpoint projects")
        result = {}
        with closing(
            sqlite3.connect(activity.database.as_uri() + "?mode=ro", uri=True, timeout=2)
        ) as db:
            for project in projects:
                row = db.execute(
                    "SELECT current_revision_id FROM projects WHERE id=?", (project,)
                ).fetchone()
                research = db.execute(
                    "SELECT revision_id FROM research_heads WHERE project_id=?", (project,)
                ).fetchone()
                figure = db.execute(
                    "SELECT id FROM research_figure_boards WHERE project_id=? "
                    "ORDER BY created_at DESC,id DESC LIMIT 1",
                    (project,),
                ).fetchone()
                result[project] = (
                    None
                    if row is None
                    else {
                        "project": str(UUID(row[0])) if row[0] else None,
                        "research": research[0] if research else None,
                        "figure": figure[0] if figure else None,
                    }
                )
        return {"profileId": profile.identifier, "projects": result}

    @app.post("/api/v1/local/draft-figures", dependencies=[gate])
    def draft_figures(bindings: list[dict[str, str]]) -> dict[str, Any]:
        if len(bindings) > 100 or any(
            set(item) != {"projectId", "figureId"} or any(len(v) > 128 for v in item.values())
            for item in bindings
        ):
            raise HTTPException(400, "Invalid draft figure bindings")
        with closing(sqlite3.connect(activity.database.as_uri() + "?mode=ro", uri=True)) as db:
            valid = all(
                db.execute(
                    "SELECT 1 FROM research_figure_boards WHERE id=? AND project_id=?",
                    (item["figureId"], item["projectId"]),
                ).fetchone()
                is not None
                for item in bindings
            )
        return {"profileId": profile.identifier, "valid": valid}
