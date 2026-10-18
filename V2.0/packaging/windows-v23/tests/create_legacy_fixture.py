"""Create synthetic saved projects and frozen exports using the released V2.2 API."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    data = args.data_root.resolve()
    assert data.is_relative_to((Path.home() / "LabVizV23Acceptance").resolve())
    data.mkdir(parents=True, exist_ok=False)
    for name in list(os.environ):
        if name.startswith("LABVIZ_"):
            del os.environ[name]
    key = "synthetic-closed-v22-fixture-" + "x" * 40
    os.environ.update(
        LABVIZ_DATABASE_PATH=str(data / "labviz-v2.db"),
        LABVIZ_OBJECT_STORAGE_ROOT=str(data / "objects"),
        LABVIZ_PERSISTENCE_BACKEND="sqlite",
        LABVIZ_OBJECT_STORAGE_BACKEND="local",
        LABVIZ_ENVIRONMENT="development",
        LABVIZ_AUTH_MODE="console",
        LABVIZ_LOCAL_ACCESS_KEY=key,
        MPLCONFIGDIR=str(data.parent / "legacy-font-cache"),
    )
    sys.path.insert(0, str(args.source_root / "V2.0/api"))
    from fastapi.testclient import TestClient
    from labviz_api import __version__
    from labviz_api.main import app

    assert __version__ == "2.2.0"
    with TestClient(app) as client:
        assert (
            client.post(
                "/api/v1/local/session", headers={"X-LabViz-Local-Key": key}
            ).status_code
            == 200
        )
        uploaded = client.post(
            "/api/v1/projects",
            files={"file": ("v22-synthetic.csv", b"x,y\n0,1\n1,3\n2,5\n", "text/csv")},
        )
        assert uploaded.status_code == 202
        identifier = uploaded.json()["projectId"]
        workspace = client.get(f"/api/v1/projects/{identifier}/workspace").json()
        assert workspace["preview"]["totalRows"] == 3
        chart = workspace["chart"]
        for format_name in ("png", "svg", "pdf"):
            chart["title"] = "V2.2 frozen synthetic figure"
            chart["export"]["format"] = format_name
            exported = client.post(
                f"/api/v1/projects/{identifier}/exports", json={"chart": chart}
            )
            assert exported.status_code == 200
        snapshots = client.get(f"/api/v1/projects/{identifier}/figures").json()[
            "snapshots"
        ]
        assert len(snapshots) == 3
        frozen = {}
        for snapshot in snapshots:
            payload = client.get(
                f"/api/v1/projects/{identifier}/figures/{snapshot['id']}/download"
            ).content
            digest = hashlib.sha256(payload).hexdigest()
            assert digest == snapshot["sha256"]
            frozen[snapshot["format"]] = {"id": snapshot["id"], "sha256": digest}
    args.report.write_text(
        json.dumps(
            {
                "legacyVersion": __version__,
                "projectId": identifier,
                "rows": 3,
                "snapshots": frozen,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
