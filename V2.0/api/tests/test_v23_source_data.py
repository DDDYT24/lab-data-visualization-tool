from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from test_v23_recovery_rehearsal import app_at, hashes, unlock

from labviz_api import source_data_routes
from labviz_api.source_data import backup_data, file_sha, restore_copy, verify_backup


def make_data(root: Path) -> None:
    root.mkdir()
    (root / "objects").mkdir()
    (root / "objects/manifest.json").write_bytes(b"future object named manifest")
    (root / "objects/raw.csv").write_bytes(b"v\n1\n")
    with sqlite3.connect(root / "labviz.db") as db:
        db.execute("CREATE TABLE projects (id TEXT PRIMARY KEY)")
        db.execute("INSERT INTO projects VALUES ('preserve-me')")


def test_complete_backup_separate_restore_and_existing_destination(tmp_path: Path) -> None:
    original, backup, restored = (tmp_path / n for n in ("original", "backup", "restored"))
    make_data(original)
    before = hashes(original)
    manifest = backup_data(original, backup)
    assert len(manifest["files"]) == 3
    restore_copy(backup, restored)
    assert hashes(original) == before
    assert hashes(original / "objects") == hashes(restored / "objects")
    assert file_sha(backup / "labviz.db") == file_sha(restored / "labviz.db")
    with pytest.raises(FileExistsError):
        restore_copy(backup, original)
    assert hashes(original) == before


@pytest.mark.parametrize(
    "damage", ["bytes", "extra", "missing", "traversal", "duplicate", "noncanonical", "invalid-db"]
)
def test_corrupt_backup_rejected_before_destination_created(tmp_path: Path, damage: str) -> None:
    original, backup, restored = (tmp_path / n for n in ("original", "backup", "restored"))
    make_data(original)
    manifest = backup_data(original, backup)
    if damage == "bytes":
        (backup / "objects/raw.csv").write_bytes(b"tampered")
    elif damage == "extra":
        (backup / "unexpected").write_bytes(b"x")
    elif damage == "missing":
        (backup / "objects/raw.csv").unlink()
    elif damage == "invalid-db":
        (backup / "labviz.db").write_bytes(b"not a SQLite database")
        record = next(r for r in manifest["files"] if r["path"] == "labviz.db")
        record.update(
            size=(backup / "labviz.db").stat().st_size, sha256=file_sha(backup / "labviz.db")
        )
    else:
        record = dict(manifest["files"][0])
        if damage == "traversal":
            record["path"] = "objects/../../outside"
        elif damage == "noncanonical":
            record["path"] = "objects//raw.csv"
        manifest["files"].append(record)
    (backup / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises((ValueError, sqlite3.DatabaseError)):
        restore_copy(backup, restored)
    assert not restored.exists()


def test_nested_destinations_are_rejected(tmp_path: Path) -> None:
    original = tmp_path / "original"
    make_data(original)
    with pytest.raises(ValueError):
        backup_data(original, original / "backup")
    backup = tmp_path / "backup"
    backup_data(original, backup)
    with pytest.raises(ValueError):
        restore_copy(backup, backup / "restored")
    verify_backup(backup)


def test_real_project_prepared_revision_and_frozen_chart_survive_restore(tmp_path: Path) -> None:
    original, backup, restored = (tmp_path / n for n in ("original", "backup", "restored"))
    with TestClient(app_at(original)) as client:
        unlock(client)
        uploaded = client.post(
            "/api/v1/projects", files={"file": ("raw.csv", b"v\n1\n2\n3\n", "text/csv")}
        )
        identifier = uploaded.json()["projectId"]
        base = f"/api/v1/projects/{identifier}/research"
        preview = client.post(
            base + "/preview",
            data={"plan": json.dumps({"mapping": {"value": "v"}})},
            files={"files": ("raw.csv", b"v\n1\n2\n3\n", "text/csv")},
        )
        assert preview.status_code == 200, preview.text
        revision = client.post(base + "/apply", json={"token": preview.json()["token"]}).json()[
            "revisionId"
        ]
        generated = client.post(
            base + "/charts", json={"revisionId": revision, "spec": {"title": "Backup fixture"}}
        )
        assert generated.status_code == 200, generated.text
        chart = generated.json()
        artifact = base + f"/charts/{chart['id']}/artifact/svg?download=true"
        expected = client.get(artifact).content
        assert expected.startswith(b"<?xml") or b"<svg" in expected
    before = hashes(original)
    backup_data(original, backup)
    restore_copy(backup, restored)
    assert hashes(original) == before
    with TestClient(app_at(restored)) as client:
        assert client.get(base).status_code == 403
        unlock(client)
        assert client.get(base).json()["revisionId"] == revision
        assert client.get(artifact).content == expected
        assert (
            client.get(f"/api/v1/projects/{identifier}/workspace").json()["preview"]["totalRows"]
            == 3
        )


def test_source_location_requires_local_unlock_and_private_profile(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(source_data_routes, "WORKSPACE_ROOT", tmp_path)
    monkeypatch.setenv("LABVIZ_DEVELOPMENT_LINE", "2.3")
    original = tmp_path / "V2.0/api/.labviz/v23-development"
    with TestClient(app_at(original)) as client:
        response = client.get("/api/v1/local/source-data")
        assert response.status_code == 403
        assert str(original) not in response.text
        unlock(client)
        info = client.get("/api/v1/local/source-data").json()
        assert info["available"] and info["database"] == str(original / "labviz.db")
        monkeypatch.setenv("LABVIZ_DEVELOPMENT_LINE", "2.2")
        assert client.get("/api/v1/local/source-data").json() == {
            "available": False,
            "reason": "source-local-only",
        }
    monkeypatch.setenv("LABVIZ_DEVELOPMENT_LINE", "2.3")
    with TestClient(app_at(tmp_path / "outside-private-roots")) as client:
        unlock(client)
        info = client.get("/api/v1/local/source-data").json()
        assert not info["available"] and "database" not in info


def test_committed_wal_rows_are_included_without_mutating_source(tmp_path: Path) -> None:
    original, backup = tmp_path / "original", tmp_path / "backup"
    make_data(original)
    with sqlite3.connect(original / "labviz.db") as db:
        db.execute("PRAGMA journal_mode=WAL")
        db.execute("INSERT INTO projects VALUES ('committed-in-wal')")
        db.commit()
        assert (original / "labviz.db-wal").stat().st_size > 0
        before = hashes(original)
        backup_data(original, backup)
        assert hashes(original) == before
        with sqlite3.connect(backup / "labviz.db") as restored:
            assert restored.execute("SELECT id FROM projects ORDER BY id").fetchall() == [
                ("committed-in-wal",),
                ("preserve-me",),
            ]
        assert {name.replace("\\", "/") for name in hashes(backup)} == {
            "labviz.db",
            "manifest.json",
            "objects/raw.csv",
            "objects/manifest.json",
        }
