from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from labviz_api.config import Settings
from labviz_api.main import create_app

KEY = "v" * 43
PLAN = {
    "mapping": {
        "value": "v",
        "sample": "s",
        "experimentalUnit": "s",
        "technicalRepeat": "repeat",
        "unit": "unit",
    },
    "analysisUnit": "experimental_unit",
}
SOURCE = b"s,repeat,v,unit\na,1,1,mV\na,2,3,mV\nb,1,5,mV\nb,2,7,mV\n"


def settings(root: Path) -> Settings:
    return Settings(
        database_path=root / "test.db",
        object_storage_root=root / "objects",
        allowed_origins=("http://127.0.0.1:3001",),
        public_web_url="http://127.0.0.1:3001",
        local_access_key=KEY,
    )


def unlock(client: TestClient) -> None:
    assert (
        client.post("/api/v1/local/session", headers={"X-LabViz-Local-Key": KEY}).status_code == 200
    )


def project(client: TestClient) -> str:
    response = client.post(
        "/api/v1/projects", files={"file": ("original.csv", b"x,y\n1,2\n", "text/csv")}
    )
    assert response.status_code == 202
    return str(response.json()["projectId"])


def test_preview_apply_conflict_restart_undo_redo_and_access(tmp_path: Path) -> None:
    config = settings(tmp_path)
    with TestClient(create_app(config)) as client:
        unlock(client)
        identifier = project(client)
        url = f"/api/v1/projects/{identifier}/research"
        assert client.get(url).json()["data"] is None
        preview = client.post(
            url + "/preview",
            data={"plan": json.dumps(PLAN)},
            files={"files": ("research.csv", SOURCE, "text/csv")},
        )
        assert preview.status_code == 200, preview.text
        assert preview.json()["data"]["summary"]["independentUnits"] == 2
        assert preview.json()["data"]["summary"]["observations"] == 4
        assert preview.json()["data"]["summary"]["mean"] == 4
        assert client.get(url).json()["data"] is None  # preview is not an applied revision
        saved = client.post(url + "/apply", json={"token": preview.json()["token"]})
        assert saved.status_code == 200, saved.text
        first = saved.json()["revisionId"]
        changed = {
            **PLAN,
            "steps": [
                {"kind": "convert_unit", "valueField": "v", "unitField": "unit", "toUnit": "V"}
            ],
        }
        preview = client.post(
            url + "/preview", data={"plan": json.dumps(changed), "expectedRevision": first}
        )
        assert preview.status_code == 200, preview.text
        saved = client.post(
            url + "/apply", json={"token": preview.json()["token"], "expectedRevision": first}
        )
        assert saved.status_code == 200
        second = saved.json()["revisionId"]
        assert saved.json()["data"]["summary"]["mean"] == 0.004
        conflict = client.post(
            url + "/preview", data={"plan": json.dumps(PLAN), "expectedRevision": first}
        )
        assert conflict.status_code == 409
        assert client.get(url).json()["revisionId"] == second
        undo = client.post(url + "/history", json={"direction": "undo", "expectedRevision": second})
        assert undo.status_code == 200
        assert undo.json()["data"]["summary"]["mean"] == 4
        redo = client.post(url + "/history", json={"direction": "redo", "expectedRevision": first})
        assert redo.status_code == 200
        assert redo.json()["revisionId"] == second
        original = client.get(f"/api/v1/projects/{identifier}/workspace").json()
        assert original["preview"]["totalRows"] == 1  # original quick-plot input stays intact
    with TestClient(create_app(config)) as restarted:
        assert restarted.get(url).status_code == 403
        unlock(restarted)
        reopened = restarted.get(url).json()
        assert reopened["revisionId"] == second
        assert reopened["data"]["summary"]["mean"] == 0.004
        assert restarted.delete(f"/api/v1/projects/{identifier}").status_code == 204
    with sqlite3.connect(config.database_path) as connection:
        assert connection.execute("SELECT count(*) FROM research_revisions").fetchone() == (0,)
        assert connection.execute("SELECT count(*) FROM research_heads").fetchone() == (0,)


def test_rejected_plan_does_not_change_current_or_create_revision(tmp_path: Path) -> None:
    with TestClient(create_app(settings(tmp_path))) as client:
        unlock(client)
        url = f"/api/v1/projects/{project(client)}/research"
        bad = client.post(
            url + "/preview",
            data={"plan": json.dumps({"mapping": {"value": "absent"}})},
            files={"files": ("research.csv", SOURCE, "text/csv")},
        )
        assert bad.status_code == 422
        assert client.get(url).json()["revisionId"] is None
        assert client.post(url + "/apply", json={"token": "invented"}).status_code == 409


def test_duplicate_preserves_research_history_under_new_project_ownership(tmp_path: Path) -> None:
    with TestClient(create_app(settings(tmp_path))) as client:
        unlock(client)
        identifier = project(client)
        url = f"/api/v1/projects/{identifier}/research"
        preview = client.post(
            url + "/preview",
            data={"plan": json.dumps(PLAN)},
            files={"files": ("research.csv", SOURCE, "text/csv")},
        ).json()
        original = client.post(url + "/apply", json={"token": preview["token"]}).json()
        duplicate = client.post(f"/api/v1/projects/{identifier}/duplicate")
        assert duplicate.status_code == 200
        copied_id = duplicate.json()["projectId"]
        copied = client.get(f"/api/v1/projects/{copied_id}/research").json()
        assert copied["revisionId"] != original["revisionId"]
        assert copied["data"]["summary"] == original["data"]["summary"]
        assert copied["data"]["observations"] == original["data"]["observations"]
        assert client.delete(f"/api/v1/projects/{identifier}").status_code == 204
        assert client.get(f"/api/v1/projects/{copied_id}/research").json() == copied


def test_paging_covers_full_input_and_rejects_stale_preview(tmp_path: Path) -> None:
    payload = ("v\n" + "\n".join(str(i) for i in range(205)) + "\n").encode()
    with TestClient(create_app(settings(tmp_path))) as client:
        unlock(client)
        identifier = project(client)
        url = f"/api/v1/projects/{identifier}/research"
        preview = client.post(
            url + "/preview",
            data={"plan": json.dumps({"mapping": {"value": "v"}})},
            files={"files": ("large.csv", payload, "text/csv")},
        ).json()
        token = preview["token"]
        assert preview["data"]["summary"]["observations"] == 205
        assert len(preview["data"]["observations"]) == 100
        page = client.get(url + "/page", params={"token": token, "offset": 200}).json()
        assert [item["value"] for item in page["observations"]] == [200, 201, 202, 203, 204]
        saved = client.post(url + "/apply", json={"token": token}).json()
        assert client.get(url + "/page", params={"token": token}).status_code == 409
        assert client.get(url + "/page", params={"revision": "old"}).status_code == 409
        assert (
            client.get(
                url + "/page", params={"revision": saved["revisionId"], "offset": 200}
            ).json()["total"]
            == 205
        )


def test_additive_research_schema_failure_rolls_back(tmp_path: Path) -> None:
    import pytest

    from labviz_api.repository import ProjectRepository
    from labviz_api.research_store import ResearchStore

    database = tmp_path / "legacy.db"
    ProjectRepository(database, 7200)
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE research_revision_project (intentional_blocker TEXT)")
        before = "\n".join(connection.iterdump())
    with pytest.raises(sqlite3.OperationalError):
        ResearchStore(database)
    with sqlite3.connect(database) as connection:
        assert "\n".join(connection.iterdump()) == before
        assert (
            connection.execute(
                "SELECT name FROM sqlite_master WHERE name='research_revisions'"
            ).fetchone()
            is None
        )


def test_research_uses_project_owner_boundary(tmp_path: Path) -> None:
    config = settings(tmp_path)
    with TestClient(create_app(config)) as client:
        unlock(client)
        identifier = project(client)
        with sqlite3.connect(config.database_path) as connection:
            connection.execute(
                "UPDATE projects SET owner_user_id='another-owner' WHERE id=?", (identifier,)
            )
        url = f"/api/v1/projects/{identifier}/research"
        assert client.get(url).status_code == 403
        assert client.post(url + "/apply", json={"token": "anything"}).status_code == 403
        assert client.get(url + "/page", params={"revision": "anything"}).status_code == 403


@pytest.mark.parametrize("sqlite_version", [None, (3, 44, 0), (3, 53, 1)])
def test_projected_view_matches_full_snapshot_including_nulls_and_pagination(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, sqlite_version: tuple[int, int, int] | None
) -> None:
    from labviz_api.research_store import ResearchStore, current_view, present

    config = settings(tmp_path)
    with TestClient(create_app(config)) as client:
        unlock(client)
        identifier = project(client)
        base = f"/api/v1/projects/{identifier}/research"
        source = ("v\n" + "\n".join(str(n) for n in range(205)) + "\n").encode()
        preview = client.post(
            base + "/preview",
            data={"plan": json.dumps({"mapping": {"value": "v"}})},
            files={"files": ("many.csv", source, "text/csv")},
        )
        assert preview.status_code == 200, preview.text
        saved = client.post(base + "/apply", json={"token": preview.json()["token"]}).json()
        store = ResearchStore(config.database_path)
        complete = store.current(identifier)
        assert complete is not None
        expected = present(json.loads(complete["prepared_json"]), revision=saved["revisionId"])
        if sqlite_version is not None:
            with store.connect() as connection:
                connection.execute(
                    "DELETE FROM research_public_views WHERE project_id=?", (identifier,)
                )
            monkeypatch.setattr(sqlite3, "sqlite_version_info", sqlite_version)
        actual = current_view(store, identifier)["data"]
        assert actual == expected
        assert len(actual["observations"]) == 100
        assert actual["observations"][0]["sampleId"] is None
        assert actual["summary"]["observations"] == 205
