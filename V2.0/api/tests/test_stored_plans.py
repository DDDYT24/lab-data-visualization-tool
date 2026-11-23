"""Saved plans stay readable, and one that is damaged is refused instead of drawn."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from test_figure_layout import figure
from test_research_api import project, settings, unlock
from test_research_chart_api import applied

from labviz_api.artifacts import local_artifact_store, remember
from labviz_api.main import create_app
from labviz_api.render_plan import RenderPlan, check_stored_plan, plan_bytes, plan_hash
from labviz_api.research_preparation import PreparationError

# Plans LabViz 2.4.0 built and stored, three of charts and four of figure panels (a layered one
# with a log axis and annotations, two captured charts). Never regenerate this file.
STORED: dict[str, dict[str, Any]] = json.loads(
    (Path(__file__).resolve().parents[2] / "contracts/fixtures/v24/stored-plans.json").read_text(
        encoding="utf-8"
    )
)["plans"]


@pytest.mark.parametrize("name", STORED)
def test_a_plan_saved_by_v24_is_read_as_it_is(name: str) -> None:
    payload = plan_bytes(STORED[name])
    check_stored_plan(payload)
    plan = RenderPlan.model_validate_json(payload)
    assert plan.plan_hash == plan_hash(STORED[name])


def altered(change: str) -> bytes:
    plan = json.loads(json.dumps(STORED["chart-violin"]))
    mark = plan["panels"][0]["marks"][0]
    if change == "a value":
        mark["halfWidths"][3] *= 2
    elif change == "an unknown mark":
        mark["kind"] = "cloud"
    elif change == "a missing field":
        del mark["grid"]
    elif change == "an unknown field":
        plan["extra"] = 1
    else:
        return plan_bytes(plan)[:-20]
    if change != "a value":
        # Consistent with itself, so only the contract can refuse it.
        plan["planHash"] = plan_hash(plan)
    return plan_bytes(plan)


@pytest.mark.parametrize(
    "change", ["a value", "an unknown mark", "a missing field", "an unknown field", "cut short"]
)
def test_an_altered_plan_is_refused(change: str) -> None:
    with pytest.raises(PreparationError) as caught:
        check_stored_plan(altered(change))
    assert caught.value.code == "plan-invalid"


@pytest.mark.parametrize("damage", ["content", "file", "missing"])
def test_a_damaged_plan_is_refused_and_the_saved_image_is_still_served(
    tmp_path: Path, damage: str
) -> None:
    config = settings(tmp_path)
    with TestClient(create_app(config)) as client:
        unlock(client)
        identifier = project(client)
        revision = applied(client, identifier)
        charts = f"/api/v1/projects/{identifier}/research/charts"
        chart = client.post(charts, json={"revisionId": revision, "spec": {}}).json()
        boards = f"/api/v1/projects/{identifier}/research/figure-boards"
        board = client.post(boards, json=figure([chart["id"]], "A")).json()
        saved = (
            (f"{charts}/{chart['id']}", "/plan", "research_charts"),
            (f"{boards}/{board['id']}", "/plans/A", "research_figure_panel_plans"),
        )
        store = local_artifact_store(config.database_path)
        for url, plan_path, table in saved:
            image = client.get(url + "/artifact/png").content
            assert client.get(url + plan_path).status_code == 200
            with sqlite3.connect(config.database_path) as connection:
                (digest,) = connection.execute(f"SELECT plan_sha256 FROM {table}").fetchone()
                if damage == "content":
                    changed = remember(
                        connection, store, altered("a value"), media_type="application/json"
                    )
                    connection.execute(f"UPDATE {table} SET plan_sha256=?", (changed,))
                elif damage == "missing":
                    store.delete(digest)
            if damage == "file":
                (path,) = (
                    item
                    for item in (config.database_path.parent / "objects").rglob(digest)
                    if item.is_file()
                )
                path.write_bytes(b"{}")
            refused = client.get(url + plan_path)
            assert (refused.status_code, refused.json()["code"]) == (422, "plan-invalid")
            assert client.get(url + "/artifact/png").content == image
