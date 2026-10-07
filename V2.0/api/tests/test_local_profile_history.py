"""Account-free SQLite history must still require a per-launch local credential."""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image

from labviz_api.config import Settings
from labviz_api.main import create_app


def _app(path: Path, key: str) -> FastAPI:
    return create_app(
        Settings(
            database_path=path,
            allowed_origins=("http://127.0.0.1:3000",),
            public_web_url="http://127.0.0.1:3000",
            local_access_key=key,
        )
    )


def _unlock(client: TestClient, key: str) -> None:
    response = client.post("/api/v1/local/session", headers={"X-LabViz-Local-Key": key})
    assert response.status_code == 200
    assert client.cookies.get("labviz_local_session") == key


def test_local_storage_contract_document_is_versioned_and_preserves_ownership() -> None:
    contract_path = Path(__file__).parents[2] / "contracts" / "local-storage-v1.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))

    assert contract["contractVersion"] == "local-storage-v1"
    assert contract["legacyReadMapping"]["temporary-cloud"] == "temporary-local"
    assert (
        "only when owner_user_id is local-profile" in contract["legacyReadMapping"]["saved-cloud"]
    )
    assert "are not re-owned, listed, or exposed" in contract["migrationPolicy"]["accountProjects"]


def test_real_import_survives_new_browser_and_restart_without_email(tmp_path: Path) -> None:
    database = tmp_path / "profile" / "labviz.db"
    key = "a" * 43
    with TestClient(_app(database, key)) as first:
        assert first.get("/api/v1/projects").status_code == 403
        assert (
            first.post("/api/v1/local/session", headers={"X-LabViz-Local-Key": "wrong"}).status_code
            == 403
        )
        _unlock(first, key)
        uploaded = first.post(
            "/api/v1/projects",
            files={"file": ("中文 数据.csv", b"x,y\n0,1\n1,3\n2,5\n", "text/csv")},
        )
        assert uploaded.status_code == 202
        assert uploaded.json()["storageMode"] == "temporary-local"
        assert uploaded.json()["storageContractVersion"] == "local-storage-v1"
        project_id = uploaded.json()["projectId"]
        assert first.get(f"/api/v1/jobs/{uploaded.json()['job']['id']}").json()["stage"] == "ready"
        listed = first.get("/api/v1/projects")
        assert listed.status_code == 200
        assert listed.json()["storageContractVersion"] == "local-storage-v1"
        assert listed.json()["projects"][0]["storageMode"] == "saved-local"
        assert first.get("/api/v1/auth/me").json()["user"]["id"] == "local-profile"

        with TestClient(_app(database, key)) as second:
            assert second.get(f"/api/v1/projects/{project_id}/workspace").status_code == 403
            _unlock(second, key)
            workspace = second.get(f"/api/v1/projects/{project_id}/workspace")
            assert workspace.status_code == 200
            reopened_session = second.get(f"/api/v1/projects/{project_id}").json()
            assert reopened_session["storageMode"] == "saved-local"
            assert reopened_session["storageContractVersion"] == "local-storage-v1"
            assert workspace.json()["preview"]["totalRows"] == 3
            duplicate = second.post(f"/api/v1/projects/{project_id}/duplicate")
            assert duplicate.status_code == 200
            assert duplicate.json()["storageMode"] == "saved-local"
            assert duplicate.json()["storageContractVersion"] == "local-storage-v1"
            assert len(second.get("/api/v1/projects").json()["projects"]) == 2

    with TestClient(_app(database, "b" * 43)) as restarted:
        assert restarted.get("/api/v1/projects").status_code == 403
        _unlock(restarted, "b" * 43)
        assert len(restarted.get("/api/v1/projects").json()["projects"]) == 2


def test_sample_remains_temporary_and_is_not_added_to_history(tmp_path: Path) -> None:
    key = "c" * 43
    with TestClient(_app(tmp_path / "db.sqlite", key)) as client:
        _unlock(client, key)
        sample = client.post("/api/v1/samples/thermal-response/projects")
        assert sample.status_code == 202
        assert sample.json()["storageMode"] == "temporary-local"
        assert sample.json()["storageContractVersion"] == "local-storage-v1"
        assert client.get("/api/v1/projects").json()["projects"] == []


def test_local_chart_cleaning_export_and_delete_survive_restart(tmp_path: Path) -> None:
    database = tmp_path / "profile" / "labviz.db"
    key = "d" * 43
    with TestClient(_app(database, key)) as first:
        _unlock(first, key)
        uploaded = first.post(
            "/api/v1/projects",
            files={"file": ("missing.csv", b"x,y\n0,1\n1,\n2,5\n", "text/csv")},
        )
        assert uploaded.status_code == 202
        project_id = uploaded.json()["projectId"]
        workspace = first.get(f"/api/v1/projects/{project_id}/workspace")
        assert workspace.status_code == 200
        chart = workspace.json()["chart"]
        chart["title"] = "Retained local figure"
        assert (
            first.put(f"/api/v1/projects/{project_id}/chart", json={"chart": chart}).status_code
            == 200
        )
        quality = first.get(f"/api/v1/projects/{project_id}/quality").json()
        finding_id = next(item["id"] for item in quality["findings"] if item["kind"] == "missing")
        assert (
            first.patch(
                f"/api/v1/projects/{project_id}/cleaning-decisions",
                json={"decisions": [{"findingId": finding_id, "action": "remove"}]},
            ).status_code
            == 200
        )

    with TestClient(_app(database, "e" * 43)) as restarted:
        assert restarted.get(f"/api/v1/projects/{project_id}/workspace").status_code == 403
        _unlock(restarted, "e" * 43)
        reopened = restarted.get(f"/api/v1/projects/{project_id}/workspace")
        assert reopened.status_code == 200
        assert reopened.json()["chart"]["title"] == "Retained local figure"
        assert reopened.json()["decisions"][0]["action"] == "remove"
        cleaned = restarted.get(f"/api/v1/projects/{project_id}/exports/cleaned-data.csv")
        assert cleaned.status_code == 200
        assert "1," not in cleaned.text
        assert restarted.delete(f"/api/v1/projects/{project_id}").status_code == 204
        assert restarted.get("/api/v1/projects").json()["projects"] == []
        # The SQLite local profile has permanent deletion, not a soft-delete API.
        # Recovery is through an offline snapshot of the entire profile.
        assert restarted.post(f"/api/v1/projects/{project_id}/restore").status_code == 404


def test_local_figure_gallery_keeps_exact_exports_and_enforces_local_access(tmp_path: Path) -> None:
    database = tmp_path / "profile" / "labviz.db"
    key = "f" * 43
    with TestClient(_app(database, key)) as first:
        _unlock(first, key)
        uploaded = first.post(
            "/api/v1/projects",
            files={"file": ("gallery.csv", b"time,response\n0,1\n1,3\n2,4\n", "text/csv")},
        )
        assert uploaded.status_code == 202
        project_id = uploaded.json()["projectId"]
        chart = first.get(f"/api/v1/projects/{project_id}/workspace").json()["chart"]
        expected_payloads: dict[str, bytes] = {}

        for format_name in ("png", "svg", "pdf"):
            chart["title"] = f"Persisted {format_name.upper()}"
            chart["export"]["format"] = format_name
            created = first.post(
                f"/api/v1/projects/{project_id}/exports",
                json={"chart": chart},
            )
            assert created.status_code == 200
            export_id = created.json()["id"]
            exported = first.get(f"/api/v1/exports/{export_id}/download")
            assert exported.status_code == 200
            expected_payloads[format_name] = exported.content

        listing = first.get(f"/api/v1/projects/{project_id}/figures")
        assert listing.status_code == 200
        body = listing.json()
        assert len(body["snapshots"]) == 3
        assert body["totalBytes"] == sum(map(len, expected_payloads.values()))
        snapshots = {item["format"]: item for item in body["snapshots"]}
        project = first.get("/api/v1/projects").json()["projects"][0]
        assert project["figureSnapshotCount"] == 3
        assert project["figureStorageBytes"] == body["totalBytes"]
        assert project["thumbnailUrl"].endswith("?thumbnail=true")

        for format_name, snapshot in snapshots.items():
            downloaded = first.get(
                f"/api/v1/projects/{project_id}/figures/{snapshot['id']}/download"
            )
            assert downloaded.status_code == 200
            assert downloaded.content == expected_payloads[format_name]
            assert snapshot["sha256"] == hashlib.sha256(downloaded.content).hexdigest()

        png = snapshots["png"]
        thumb = first.get(
            f"/api/v1/projects/{project_id}/figures/{png['id']}/preview?thumbnail=true"
        )
        assert thumb.status_code == 200
        with Image.open(io.BytesIO(thumb.content)) as image:
            assert image.format == "PNG"
            assert image.width <= 640 and image.height <= 400
        svg = first.get(f"/api/v1/projects/{project_id}/figures/{snapshots['svg']['id']}/preview")
        assert svg.status_code == 200
        assert svg.content == expected_payloads["svg"]
        pdf_preview = first.get(
            f"/api/v1/projects/{project_id}/figures/{snapshots['pdf']['id']}/preview"
        )
        assert pdf_preview.status_code == 415
        assert first.get("/api/v1/projects/not-this-project/figures").status_code == 404

        snapshot_to_delete = snapshots["svg"]["id"]
        assert (
            first.delete(f"/api/v1/projects/{project_id}/figures/{snapshot_to_delete}").status_code
            == 204
        )
        after_delete = first.get(f"/api/v1/projects/{project_id}/figures").json()
        assert {item["format"] for item in after_delete["snapshots"]} == {"png", "pdf"}
        assert after_delete["totalBytes"] == len(expected_payloads["png"]) + len(
            expected_payloads["pdf"]
        )

    with TestClient(_app(database, "g" * 43)) as restarted:
        assert restarted.get(f"/api/v1/projects/{project_id}/figures").status_code == 403
        _unlock(restarted, "g" * 43)
        recovered = restarted.get(f"/api/v1/projects/{project_id}/figures")
        assert recovered.status_code == 200
        assert {item["format"] for item in recovered.json()["snapshots"]} == {"png", "pdf"}
        assert restarted.delete(f"/api/v1/projects/{project_id}").status_code == 204
        assert restarted.get(f"/api/v1/projects/{project_id}/figures").status_code == 404
