"""Disposable rehearsal; deliberately not a production migration implementation."""

from __future__ import annotations

import hashlib
import shutil
import sqlite3
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from labviz_api.config import Settings
from labviz_api.main import create_app


def app_at(root: Path) -> FastAPI:
    return create_app(
        Settings(
            database_path=root / "labviz.db",
            object_storage_root=root / "objects",
            allowed_origins=("http://127.0.0.1:3001",),
            public_web_url="http://127.0.0.1:3001",
            local_access_key="r" * 43,
        )
    )


def unlock(client: TestClient) -> None:
    assert (
        client.post(
            "/api/v1/local/session",
            headers={"X-LabViz-Local-Key": "r" * 43},
        ).status_code
        == 200
    )


def hashes(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in root.rglob("*")
        if path.is_file()
    }


def test_legacy_backup_failed_migration_and_separate_restore(tmp_path: Path) -> None:
    original, backup, restored = [tmp_path / name for name in ("original", "backup", "restored")]
    with TestClient(app_at(original)) as client:
        unlock(client)
        uploaded = client.post(
            "/api/v1/projects",
            files={"file": ("legacy.csv", b"x,y\n0,1\n1,3\n2,5\n", "text/csv")},
        )
        assert uploaded.status_code == 202
        project_id = uploaded.json()["projectId"]
        chart = client.get(f"/api/v1/projects/{project_id}/workspace").json()["chart"]
        chart["title"] = "Legacy recovery fixture"
        assert (
            client.put(
                f"/api/v1/projects/{project_id}/chart",
                json={"chart": chart},
            ).status_code
            == 200
        )
        exported = client.post(f"/api/v1/projects/{project_id}/exports", json={"chart": chart})
        assert exported.status_code == 200
        snapshot = client.get(f"/api/v1/projects/{project_id}/figures").json()["snapshots"][0]
        snapshot_url = f"/api/v1/projects/{project_id}/figures/{snapshot['id']}/download"
        expected_bytes = client.get(snapshot_url).content
        expected_chart = client.get(f"/api/v1/projects/{project_id}/workspace").json()["chart"]
    backup.mkdir()
    with sqlite3.connect(original / "labviz.db") as source:
        expected_dump = "\n".join(source.iterdump())
        with sqlite3.connect(backup / "labviz.db") as target:
            source.backup(target)
        with pytest.raises(sqlite3.OperationalError), source:
            source.execute("BEGIN IMMEDIATE")
            source.execute("CREATE TABLE v23_rehearsal_only (id TEXT)")
            source.execute("UPDATE projects SET owner_user_id = 'must-roll-back'")
            source.execute("INSERT INTO deliberately_missing_table VALUES (1)")
        assert "\n".join(source.iterdump()) == expected_dump
        assert source.execute("PRAGMA integrity_check").fetchone() == ("ok",)
    # Copy all object bytes, including future non-database payloads, while writers are stopped.
    (original / "objects").mkdir(exist_ok=True)
    shutil.copytree(original / "objects", backup / "objects")
    assert hashes(original / "objects") == hashes(backup / "objects")
    shutil.copytree(backup, restored)
    with sqlite3.connect(restored / "labviz.db") as connection:
        assert "\n".join(connection.iterdump()) == expected_dump
        assert connection.execute("PRAGMA integrity_check").fetchone() == ("ok",)
    with TestClient(app_at(restored)) as client:
        assert client.get(f"/api/v1/projects/{project_id}/workspace").status_code == 403
        unlock(client)
        workspace = client.get(f"/api/v1/projects/{project_id}/workspace")
        assert workspace.status_code == 200
        assert workspace.json()["chart"] == expected_chart
        assert workspace.json()["preview"]["totalRows"] == 3
        recovered = client.get(snapshot_url)
        assert recovered.status_code == 200
        assert recovered.content == expected_bytes
        assert hashlib.sha256(recovered.content).hexdigest() == snapshot["sha256"]
