from __future__ import annotations

import importlib.util
import sqlite3
from contextlib import closing
from pathlib import Path
from types import ModuleType

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from labviz_api.config import Settings
from labviz_api.main import create_app
from labviz_api.repository import ProjectRepository


@pytest.fixture
def data_tool() -> ModuleType:
    path = Path(__file__).resolve().parents[2] / "packaging/windows/bin/local-data.py"
    spec = importlib.util.spec_from_file_location("local_data", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def old_data(tmp_path: Path) -> Path:
    root = tmp_path / "旧版 路径" / ".labviz"
    ProjectRepository(root / "labviz-v2.db", project_ttl_seconds=7200)
    with closing(sqlite3.connect(root / "labviz-v2.db")) as db, db:
        db.execute("CREATE TABLE migration_answer (value TEXT)")
        db.execute("INSERT INTO migration_answer VALUES ('合成实验')")
    objects = root / "objects"
    objects.mkdir()
    (objects / "synthetic.bin").write_bytes(b"synthetic-only")
    return root


def test_snapshot_import_and_rollback_preserve_data(
    data_tool: ModuleType, old_data: Path, tmp_path: Path
) -> None:
    backup, installed = tmp_path / "backup", tmp_path / "installed"
    data_tool.snapshot(old_data, backup)
    data_tool.restore(backup, installed)
    with closing(sqlite3.connect(installed / "labviz-v2.db")) as db, db:
        assert db.execute("SELECT value FROM migration_answer").fetchone() == ("合成实验",)
        db.execute("UPDATE migration_answer SET value='new-version'")
    (installed / "objects/synthetic.bin").write_bytes(b"new-version")
    data_tool.restore(backup, installed, replace=True)
    with closing(sqlite3.connect(installed / "labviz-v2.db")) as db, db:
        assert db.execute("SELECT value FROM migration_answer").fetchone() == ("合成实验",)
    assert (installed / "objects/synthetic.bin").read_bytes() == b"synthetic-only"
    assert (old_data / "objects/synthetic.bin").read_bytes() == b"synthetic-only"
    assert list(tmp_path.glob("installed.retained-*"))


def test_import_never_overwrites_existing_data(
    data_tool: ModuleType, old_data: Path, tmp_path: Path
) -> None:
    backup = tmp_path / "backup"
    data_tool.snapshot(old_data, backup)
    with pytest.raises(ValueError, match="already contains"):
        data_tool.restore(backup, old_data)


def test_import_resets_browser_sessions_but_preserves_guest_projects(
    data_tool: ModuleType, old_data: Path, tmp_path: Path
) -> None:
    with closing(sqlite3.connect(old_data / "labviz-v2.db")) as db, db:
        db.execute(
            """
            INSERT INTO projects
                (id, title, source_json, storage_mode, guest_token_digest, updated_at)
            VALUES (
                'legacy-guest-project', 'Legacy guest project', '{}', 'inline', ?,
                '2026-01-01T00:00:00Z'
            )
            """,
            ("guest-token-digest",),
        )
        db.execute(
            """
            INSERT INTO auth_sessions
                (token_digest, user_id, email, expires_at, created_at)
            VALUES (
                'old-session', 'old-user', 'old@example.test', '2099-01-01T00:00:00Z',
                '2026-01-01T00:00:00Z'
            )
            """
        )
    installed = tmp_path / "installed"
    data_tool.import_data(old_data, installed)
    with closing(sqlite3.connect(installed / "labviz-v2.db")) as db:
        assert db.execute(
            "SELECT guest_token_digest FROM projects WHERE id='legacy-guest-project'"
        ).fetchone() == ("guest-token-digest",)
        assert db.execute("SELECT COUNT(*) FROM auth_sessions").fetchone() == (0,)
    with closing(sqlite3.connect(old_data / "labviz-v2.db")) as db:
        assert db.execute("SELECT COUNT(*) FROM auth_sessions").fetchone() == (1,)
    # Import refreshes its manifest after sanitizing the copied database.
    data_tool.restore(installed, tmp_path / "reopened")


def test_corrupt_snapshot_is_rejected_before_replacement(
    data_tool: ModuleType, old_data: Path, tmp_path: Path
) -> None:
    backup = tmp_path / "backup"
    data_tool.snapshot(old_data, backup)
    (backup / "objects/synthetic.bin").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="changed"):
        data_tool.restore(backup, old_data, replace=True)
    assert (old_data / "objects/synthetic.bin").read_bytes() == b"synthetic-only"


def test_live_writer_blocks_snapshot(data_tool: ModuleType, old_data: Path, tmp_path: Path) -> None:
    with sqlite3.connect(old_data / "labviz-v2.db") as writer:
        writer.execute("BEGIN IMMEDIATE")
        with pytest.raises(sqlite3.OperationalError, match="locked"):
            data_tool.snapshot(old_data, tmp_path / "backup")
        writer.rollback()
    assert not (tmp_path / "backup").exists()
    assert not list(tmp_path.glob("backup.partial-*"))


def test_unrelated_database_and_nested_destination_rejected(
    data_tool: ModuleType, old_data: Path, tmp_path: Path
) -> None:
    with pytest.raises(ValueError, match="outside"):
        data_tool.snapshot(old_data, old_data / "backup")
    invalid = tmp_path / "invalid"
    invalid.mkdir()
    with sqlite3.connect(invalid / "labviz-v2.db") as db:
        db.execute("CREATE TABLE unrelated (x)")
    with pytest.raises(ValueError, match="supported LabViz"):
        data_tool.snapshot(invalid, tmp_path / "backup")


def test_offline_backup_recovers_a_deleted_local_project(
    data_tool: ModuleType, tmp_path: Path
) -> None:
    profile = tmp_path / "Windows 用户" / "data"
    database = profile / "labviz-v2.db"

    def local_app(key: str) -> FastAPI:
        return create_app(
            Settings(
                database_path=database,
                allowed_origins=("http://127.0.0.1:3000",),
                public_web_url="http://127.0.0.1:3000",
                local_access_key=key,
            )
        )

    with TestClient(local_app("f" * 43)) as client:
        assert (
            client.post(
                "/api/v1/local/session", headers={"X-LabViz-Local-Key": "f" * 43}
            ).status_code
            == 200
        )
        uploaded = client.post(
            "/api/v1/projects",
            files={"file": ("synthetic.csv", b"x,y\n0,1\n1,2\n", "text/csv")},
        )
        assert uploaded.status_code == 202
        project_id = uploaded.json()["projectId"]
        assert client.get("/api/v1/projects").json()["projects"][0]["id"] == project_id

    backup = tmp_path / "backup"
    data_tool.snapshot(profile, backup)
    with TestClient(local_app("g" * 43)) as client:
        assert (
            client.post(
                "/api/v1/local/session", headers={"X-LabViz-Local-Key": "g" * 43}
            ).status_code
            == 200
        )
        assert client.delete(f"/api/v1/projects/{project_id}").status_code == 204
        assert client.get("/api/v1/projects").json()["projects"] == []

    data_tool.restore(backup, profile, replace=True)
    with TestClient(local_app("h" * 43)) as client:
        assert (
            client.post(
                "/api/v1/local/session", headers={"X-LabViz-Local-Key": "h" * 43}
            ).status_code
            == 200
        )
        assert client.get("/api/v1/projects").json()["projects"][0]["id"] == project_id
        assert client.get(f"/api/v1/projects/{project_id}/workspace").status_code == 200
    assert list(profile.parent.glob("data.retained-*"))


def test_import_retains_account_owned_records_without_exposing_them_locally(
    data_tool: ModuleType, old_data: Path, tmp_path: Path
) -> None:
    with closing(sqlite3.connect(old_data / "labviz-v2.db")) as db, db:
        db.execute(
            """INSERT INTO projects
            (id, title, source_json, storage_mode, owner_user_id, updated_at)
            VALUES ('legacy-account-project', 'Private old account', '{}',
                    'saved-cloud', 'other-owner', '2026-01-01T00:00:00Z')"""
        )
    installed = tmp_path / "installed"
    data_tool.import_data(old_data, installed)
    database = installed / "labviz-v2.db"
    with TestClient(
        create_app(
            Settings(
                database_path=database,
                allowed_origins=("http://127.0.0.1:3000",),
                public_web_url="http://127.0.0.1:3000",
                local_access_key="i" * 43,
            )
        )
    ) as client:
        assert (
            client.post(
                "/api/v1/local/session", headers={"X-LabViz-Local-Key": "i" * 43}
            ).status_code
            == 200
        )
        assert client.get("/api/v1/projects").json()["projects"] == []
        assert client.get("/api/v1/projects/legacy-account-project").status_code == 403
    with closing(sqlite3.connect(database)) as db:
        assert db.execute(
            "SELECT owner_user_id FROM projects WHERE id='legacy-account-project'"
        ).fetchone() == ("other-owner",)
