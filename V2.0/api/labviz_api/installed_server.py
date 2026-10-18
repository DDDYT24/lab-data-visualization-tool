"""Loopback-only installed server with authenticated graceful shutdown."""

from __future__ import annotations

import argparse
import secrets

import uvicorn
from fastapi import Header, HTTPException

from .installed_profile import APP_ID, environment_profile


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()
    profile = environment_profile()
    if profile is None:
        raise ValueError("An installed profile is required.")
    from .main import app

    key = app.state.settings.local_access_key
    server = uvicorn.Server(
        uvicorn.Config(
            app,
            host="127.0.0.1",
            port=args.port,
            access_log=False,
            timeout_graceful_shutdown=30,
        )
    )

    @app.post("/api/v1/local/shutdown")
    def shutdown(
        local_key: str | None = Header(default=None, alias="X-LabViz-Local-Key"),
    ) -> dict[str, str | bool]:
        if not local_key or not key or not secrets.compare_digest(local_key, key):
            raise HTTPException(status_code=403, detail="Invalid local launch credential.")
        server.should_exit = True
        return {"accepted": True, "appId": APP_ID, "profileId": profile.identifier}

    server.run()


if __name__ == "__main__":
    main()
