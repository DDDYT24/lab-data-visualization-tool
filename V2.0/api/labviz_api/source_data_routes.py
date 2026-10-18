"""Authenticated, read-only commands for the private source workspace."""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, Response

from .config import Settings
from .installed_profile import APP_ID, environment_profile, unlinked

WORKSPACE_ROOT = Path(__file__).resolve().parents[3]


def register_source_data_routes(
    app: FastAPI, settings: Settings, user_dependency: Callable[..., Any], local_user: str
) -> None:
    user_dep = Depends(user_dependency)

    @app.get("/api/v1/local/source-data")
    def source_data(response: Response, user: Any = user_dep) -> dict[str, Any]:
        response.headers["Cache-Control"] = "no-store"
        root = WORKSPACE_ROOT
        data = settings.database_path.resolve().parent
        installed = environment_profile()
        if installed:
            if (
                not settings.local_access_key
                or not user
                or user.get("id") != local_user
                or settings.database_path.resolve() != installed.data / "labviz.db"
                or settings.object_storage_root.resolve() != installed.data / "objects"
            ):
                return {"available": False, "reason": "installed-local-only"}
            program = unlinked(Path(os.environ["LABVIZ_INSTALL_ROOT"]))
            import json

            marker = json.loads((program / "installation.json").read_text(encoding="utf-8-sig"))
            if marker.get("appId") != APP_ID:
                return {"available": False, "reason": "installed-identity-invalid"}
            return {
                "available": True,
                "mode": "installed",
                "profileRoot": str(installed.root),
                "database": str(settings.database_path.resolve()),
                "objects": str(settings.object_storage_root.resolve()),
                "dataDirectory": str(installed.data),
                "backupDirectory": str(installed.root / "backups"),
                "tool": str(program / "bin/maintain-v23.ps1"),
            }
        permitted = [
            root / "V2.0/api/.labviz/v23-development",
            root / "V2.0/api/.labviz/v23-restored",
            root / "outputs/v23-phase5",
        ]
        if (
            os.environ.get("LABVIZ_DEVELOPMENT_LINE") != "2.3"
            or not settings.local_access_key
            or not user
            or user.get("id") != local_user
            or settings.persistence_backend != "sqlite"
            or settings.object_storage_backend != "local"
            or settings.database_path.name != "labviz.db"
            or settings.object_storage_root.resolve() != data / "objects"
            or not data.is_relative_to(root.resolve())
            or not any(data.is_relative_to(p.resolve()) for p in permitted)
        ):
            return {"available": False, "reason": "source-local-only"}
        return {
            "available": True,
            "database": str(settings.database_path.resolve()),
            "objects": str(settings.object_storage_root.resolve()),
            "dataDirectory": str(data),
            "backupDirectory": str(root / "outputs/v23-backups"),
            "tool": str(root / "source-data-v23.ps1"),
        }
