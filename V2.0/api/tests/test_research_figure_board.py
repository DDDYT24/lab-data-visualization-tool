from __future__ import annotations

import gc
import json
import sqlite3
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError
from test_research_api import SOURCE, project, settings, unlock
from test_research_chart_api import applied

from labviz_api.main import create_app
from labviz_api.repository import ProjectRepository
from labviz_api.research_figure_board import FigureBoardSpec, install_boards
from labviz_api.research_store import ResearchStore


def charts(client: TestClient, identifier: str) -> list[dict[str, Any]]:
    revision = applied(client, identifier)
    base = f"/api/v1/projects/{identifier}/research/charts"
    first = client.post(base, json={"revisionId": revision, "spec": {"title": "Raw"}})
    assert first.status_code == 200, first.text
    revision = applied(client, identifier, SOURCE.replace(b",7,", b",9,"))
    second = client.post(
        base, json={"revisionId": revision, "spec": {"kind": "violin", "title": "Distribution"}}
    )
    assert second.status_code == 200, second.text
    return [first.json(), second.json()]


def test_one_two_four_panel_history_reordering_and_restart(tmp_path: Path) -> None:
    config = settings(tmp_path)
    with TestClient(create_app(config)) as client:
        unlock(client)
        identifier = project(client)
        saved_charts = charts(client, identifier)
        ids = [item["id"] for item in saved_charts]
        base = f"/api/v1/projects/{identifier}/research/figure-boards"
        saved = []
        for count in (1, 2, 4):
            bindings = (ids * 2)[:count]
            response = client.post(
                base, json={"title": f"Layout {count}", "panelChartIds": bindings}
            )
            assert response.status_code == 200, response.text
            measured = {
                part.split(";dur=")[0]: float(part.split(";dur=")[1])
                for part in response.headers["Server-Timing"].split(", ")
            }
            assert set(measured) == {"load", "render", "persist", "index"}
            assert all(value >= 0 for value in measured.values())
            body = response.json()
            assert [p["label"] for p in body["panels"]] == list("ABCD")[:count]
            assert [p["chartId"] for p in body["panels"]] == bindings
            assert [p["observations"] for p in body["panels"]] == [4] * count
            assert [p["independentUnits"] for p in body["panels"]] == [2] * count
            saved.append(body)
        reordered = client.post(base, json={"panelChartIds": ids[::-1]}).json()
        assert [p["revisionId"] for p in reordered["panels"]] == [
            c["revisionId"] for c in saved_charts[::-1]
        ]
        assert [p["resultSha256"] for p in reordered["panels"]] == [
            c["resultSha256"] for c in saved_charts[::-1]
        ]
        assert client.get(base + "/" + saved[1]["id"]).json() == saved[1]
        assert len(client.get(base).json()["boards"]) == 4
    with TestClient(create_app(config)) as client:
        unlock(client)
        assert client.get(base + "/" + saved[2]["id"]).json() == saved[2]


def test_copy_remaps_board_bindings_and_delete_preserves_copy(tmp_path: Path) -> None:
    config = settings(tmp_path)
    with TestClient(create_app(config)) as client:
        unlock(client)
        identifier = project(client)
        saved_charts = charts(client, identifier)
        base = f"/api/v1/projects/{identifier}/research/figure-boards"
        original = client.post(
            base, json={"title": "Copy", "panelChartIds": [c["id"] for c in saved_charts]}
        ).json()
        duplicate = client.post(f"/api/v1/projects/{identifier}/duplicate").json()["projectId"]
        copied_base = f"/api/v1/projects/{duplicate}/research/figure-boards"
        copied_id = client.get(copied_base).json()["boards"][0]["id"]
        copied = client.get(copied_base + "/" + copied_id).json()
        assert copied_id != original["id"]
        assert [p["chartId"] for p in copied["panels"]] != [
            p["chartId"] for p in original["panels"]
        ]
        assert [p["revisionId"] for p in copied["panels"]] != [
            p["revisionId"] for p in original["panels"]
        ]
        assert [p["resultSha256"] for p in copied["panels"]] == [
            p["resultSha256"] for p in original["panels"]
        ]
        before = []
        for p in copied["panels"]:
            before.append(
                client.get(
                    f"/api/v1/projects/{duplicate}/research/charts/{p['chartId']}/artifact/svg"
                ).content
            )
        assert client.delete(f"/api/v1/projects/{identifier}").status_code == 204
        assert client.get(copied_base + "/" + copied_id).json() == copied
        for index, p in enumerate(copied["panels"]):
            assert (
                client.get(
                    f"/api/v1/projects/{duplicate}/research/charts/{p['chartId']}/artifact/svg"
                ).content
                == before[index]
            )
        assert client.delete(f"/api/v1/projects/{duplicate}").status_code == 204
    with sqlite3.connect(config.database_path) as connection:
        assert connection.execute("SELECT count(*) FROM research_figure_boards").fetchone() == (0,)


def test_wrong_project_invalid_binding_and_owner_boundary(tmp_path: Path) -> None:
    config = settings(tmp_path)
    with TestClient(create_app(config)) as client:
        unlock(client)
        identifier = project(client)
        chart = charts(client, identifier)[0]
        other = project(client)
        base = f"/api/v1/projects/{identifier}/research/figure-boards"
        other_base = f"/api/v1/projects/{other}/research/figure-boards"
        assert client.post(other_base, json={"panelChartIds": [chart["id"]]}).status_code == 422
        assert client.get(other_base).json()["boards"] == []
        assert (
            client.post(base, json={"panelChartIds": [chart["id"], "missing"]}).status_code == 422
        )
        assert client.get(base).json()["boards"] == []
        valid = client.post(base, json={"panelChartIds": [chart["id"]]}).json()
        assert client.get(other_base + "/" + valid["id"]).status_code == 404
        with sqlite3.connect(config.database_path) as connection:
            original_owner = connection.execute(
                "SELECT owner_user_id FROM projects WHERE id=?", (identifier,)
            ).fetchone()[0]
            connection.execute(
                "UPDATE projects SET owner_user_id='other-owner' WHERE id=?", (identifier,)
            )
        assert client.get(base + "/" + valid["id"] + "/artifact/svg").status_code == 403
        assert client.get(base + "/" + valid["id"] + "/interaction").status_code == 403
        source_url = f"/api/v1/projects/{identifier}/research/figure-sources"
        assert client.get(source_url).status_code == 403
        assert client.post(source_url).status_code == 403
        assert client.get(base).status_code == 403
        assert client.get(base + "/" + valid["id"]).status_code == 403
        assert client.post(base, json={"panelChartIds": [chart["id"]]}).status_code == 403
        # A cached immutable scene must still respect deletion after permissions return.
        with sqlite3.connect(config.database_path) as connection:
            connection.execute(
                "UPDATE projects SET owner_user_id=? WHERE id=?", (original_owner, identifier)
            )
            connection.execute("DELETE FROM research_figure_boards WHERE id=?", (valid["id"],))
        assert client.get(base + "/" + valid["id"] + "/interaction").status_code == 404


def test_schema_and_installation_rollback(tmp_path: Path) -> None:
    schema = Path(__file__).resolve().parents[2] / "contracts/figure-board-v1.schema.json"
    assert json.loads(schema.read_text()) == FigureBoardSpec.model_json_schema(by_alias=True)
    database = tmp_path / "rollback.db"
    ProjectRepository(database, 7200)
    store = ResearchStore(database)
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE research_figure_boards_project (blocker TEXT)")
        before = "\n".join(connection.iterdump())
    with pytest.raises(sqlite3.OperationalError):
        install_boards(store)
    with sqlite3.connect(database) as connection:
        assert "\n".join(connection.iterdump()) == before


@pytest.mark.parametrize(
    "body",
    [
        {"panelChartIds": []},
        {"panelChartIds": ["a", "b", "c"]},
        {"panelChartIds": ["a"] * 5},
        {"panelChartIds": [""]},
        {"panelChartIds": ["a"], "axisMode": "shared"},
        {"panelChartIds": ["a"], "script": "x"},
    ],
)
def test_invalid_layout_contract(body: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        FigureBoardSpec.model_validate(body)


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("fail", [False, True])
def test_source_batch_restores_gc_and_rolls_back_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, enabled: bool, fail: bool
) -> None:
    from labviz_api import research_figure_board
    from labviz_api.research_figure_sources import source
    from labviz_api.research_preparation import PreparationError

    original_source = source
    observed = []

    def checked(connection: Any, project_id: str, chart_id: str) -> dict[str, Any]:
        observed.append(gc.isenabled())
        if fail:
            raise PreparationError("figure-test-failure", "Injected source-load failure")
        return original_source(connection, project_id, chart_id)

    with TestClient(create_app(settings(tmp_path))) as client:
        unlock(client)
        identifier = project(client)
        chart_id = charts(client, identifier)[0]["id"]
        base = f"/api/v1/projects/{identifier}/research/figure-boards"
        monkeypatch.setattr(research_figure_board, "source", checked)
        previous = gc.isenabled()
        (gc.enable if enabled else gc.disable)()
        try:
            response = client.post(base, json={"panelChartIds": [chart_id] * 4})
            assert response.status_code == (422 if fail else 200), response.text
            assert observed == [False]
            assert gc.isenabled() == enabled
            assert len(client.get(base).json()["boards"]) == (0 if fail else 1)
        finally:
            (gc.enable if previous else gc.disable)()
