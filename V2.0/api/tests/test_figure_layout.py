"""FigureSpec v2 saved as a board: layouts up to six panels, axis controls and annotations."""

from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter
from typing import Any

import pytest
from fastapi.testclient import TestClient
from test_research_api import project, settings, unlock
from test_research_figure_board import charts

from labviz_api.figure_spec import FigureSpec, panel_cells
from labviz_api.main import create_app
from labviz_api.plan_axes import axis_position
from labviz_api.render_plan import RenderPlan

MOSAIC = ["AAB", "CDE", "CDF"]


def figure(ids: list[str], labels: str, **extra: Any) -> dict[str, Any]:
    return {
        "schemaVersion": 2,
        "panels": [
            {
                "id": f"p{index + 1}",
                "label": label,
                "layers": [{"kind": "chart", "chartId": ids[index % len(ids)]}],
            }
            for index, label in enumerate(labels)
        ],
        **extra,
    }


@pytest.fixture
def saved(tmp_path: Path) -> Any:
    with TestClient(create_app(settings(tmp_path))) as client:
        unlock(client)
        identifier = project(client)
        ids = [item["id"] for item in charts(client, identifier)]
        yield client, f"/api/v1/projects/{identifier}/research/figure-boards", ids


def test_cells_of_grid_and_mosaic() -> None:
    grid = FigureSpec.model_validate(figure(["c"], "ABCDE", layout={"rows": 2, "cols": 3}))
    assert panel_cells(grid) == [
        (0, 0, 1, 1),
        (0, 1, 1, 1),
        (0, 2, 1, 1),
        (1, 0, 1, 1),
        (1, 1, 1, 1),
    ]
    mosaic = FigureSpec.model_validate(
        figure(["c"], "ABCDEF", layout={"kind": "mosaic", "rows": 3, "cols": 3, "mosaic": MOSAIC})
    )
    assert panel_cells(mosaic) == [
        (0, 0, 1, 2),
        (0, 2, 1, 1),
        (1, 0, 2, 1),
        (1, 1, 2, 1),
        (1, 2, 1, 1),
        (2, 2, 1, 1),
    ]


@pytest.mark.parametrize(
    ("labels", "layout"),
    [
        ("ABC", {"rows": 1, "cols": 3}),
        ("ABCDE", {"rows": 2, "cols": 3}),
        ("ABCDEF", {"kind": "mosaic", "rows": 3, "cols": 3, "mosaic": MOSAIC}),
    ],
)
def test_three_five_and_six_panels_save_and_reopen(
    saved: Any, labels: str, layout: dict[str, Any]
) -> None:
    client, base, ids = saved
    response = client.post(base, json=figure(ids, labels, layout=layout, title="Layout"))
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["spec"]["schemaVersion"] == 2 and body["composed"] is True
    assert [panel["label"] for panel in body["panels"]] == list(labels)
    assert [panel["chartId"] for panel in body["panels"]] == [
        ids[index % len(ids)] for index in range(len(labels))
    ]
    url = base + "/" + body["id"]
    assert client.get(url).json() == body
    for label in labels:
        plan = RenderPlan.model_validate_json(client.get(f"{url}/plans/{label}").content)
        assert plan.panels[0].id == label
    assert client.get(url + "/artifact/png").content.startswith(b"\x89PNG")
    listed = client.get(base).json()["boards"]
    assert listed[0]["spec"]["schemaVersion"] == 2


def test_a_v1_board_is_still_stored_as_v1_and_headed_by_its_label(saved: Any) -> None:
    client, base, ids = saved
    plain = client.post(base, json={"panelChartIds": ids}).json()
    assert plain["spec"]["schemaVersion"] == 1
    plan = json.loads(client.get(f"{base}/{plain['id']}/plans/B").content)
    assert plan["panels"][0]["title"].startswith("B · ")


def test_a_log_axis_and_a_bracket_reach_the_stored_plan(saved: Any) -> None:
    client, base, ids = saved

    def stored(spec: dict[str, Any]) -> dict[str, Any]:
        response = client.post(base, json=spec)
        assert response.status_code == 200, response.text
        return dict(json.loads(client.get(f"{base}/{response.json()['id']}/plans/A").content))

    plain = stored(figure(ids, "A"))
    logged_spec = figure(ids, "A")
    logged_spec["panels"][0]["axes"] = {"y": {"scale": "log10"}}
    logged = stored(logged_spec)["panels"][0]
    low, high = logged["axes"]["y"]["domain"]
    middle = (low * high) ** 0.5
    left, right = logged["axes"]["x"]["domain"]
    first, second = left + (right - left) / 4, right - (right - left) / 4
    noted_spec = figure(ids, "A")
    noted_spec["panels"][0]["axes"] = {"y": {"scale": "log10"}}
    noted_spec["panels"][0]["annotations"] = [
        {"kind": "bracket", "x": [first, second], "y": middle, "label": "p < 0.01"},
        {"kind": "hline", "value": middle, "label": "limit"},
    ]
    plan = stored(noted_spec)
    RenderPlan.model_validate_json(json.dumps(plan))
    panel = plan["panels"][0]
    assert panel["axes"] == logged["axes"] and panel["axes"]["y"]["scale"] == "log10"
    assert panel["axes"]["x"] == plain["panels"][0]["axes"]["x"]
    bracket, line = panel["notes"]
    assert bracket["kind"] == "bracket"
    assert line == {"kind": "hline", "value": middle, "label": "limit"}
    assert axis_position("log10", bracket["foot"]) < axis_position("log10", middle)
    assert panel["marks"] == plain["panels"][0]["marks"]
    assert "notes" not in plain["panels"][0] and "notes" not in logged
    assert len({plan["planHash"], plain["planHash"]}) == 2


def test_label_style_and_canvas_reach_the_export(saved: Any) -> None:
    client, base, ids = saved
    spec = figure(
        ids,
        "AB",
        layout={"rows": 1, "cols": 2, "labelStyle": "lower"},
        canvas={"sizePreset": "custom", "width": 180, "height": 80, "unit": "mm", "dpi": 150},
    )
    body = client.post(base, json=spec).json()
    plan = json.loads(client.get(f"{base}/{body['id']}/plans/B").content)
    assert plan["panels"][0]["title"].startswith("b · ") and plan["panels"][0]["id"] == "B"
    png = client.get(f"{base}/{body['id']}/artifact/png").content
    width, height = int.from_bytes(png[16:20], "big"), int.from_bytes(png[20:24], "big")
    # Matplotlib truncates the pixel size: 1062.99 px wide is 1062.
    assert (width, height) == (int(180 / 25.4 * 150), int(80 / 25.4 * 150))


@pytest.mark.parametrize(
    ("change", "code"),
    [
        (
            {"layers": [{"kind": "scatter", "encoding": {"x": "a", "y": "b"}}]},
            "figure-layer-unsupported",
        ),
        ({"annotations": [{"kind": "hline", "value": 1e12}]}, "annotation-outside"),
        ({"axes": {"x": {"scale": "log10"}}}, "axis-control-category"),
    ],
)
def test_refused_figures(saved: Any, change: dict[str, Any], code: str) -> None:
    client, base, ids = saved
    spec = figure(ids, "A")
    spec["panels"][0] |= change
    response = client.post(base, json=spec)
    assert response.status_code == 422, response.text
    assert response.json()["code"] == code
    assert client.get(base).json()["boards"] == []


def test_a_shared_axis_refuses_a_scale_on_it_and_an_unknown_palette_is_refused(saved: Any) -> None:
    client, base, ids = saved
    spec = figure([ids[0]], "AB", layout={"rows": 1, "cols": 2}, axisMode="shared_y")
    spec["panels"][0]["axes"] = {"y": {"scale": "log10"}}
    assert client.post(base, json=spec).json()["code"] == "figure-shared-axis-control"
    other = figure(ids, "A", palette={"categorical": "neon"})
    assert client.post(base, json=other).json()["code"] == "figure-palette-unknown"


def test_the_deferred_path_builds_a_six_panel_figure(saved: Any) -> None:
    client, base, ids = saved
    layout = {"kind": "mosaic", "rows": 3, "cols": 3, "mosaic": MOSAIC}
    response = client.post(
        base, params={"deferArtifacts": True}, json=figure(ids, "ABCDEF", layout=layout)
    )
    assert response.status_code == 202, response.text
    body = response.json()
    assert [panel["label"] for panel in body["panels"]] == list("ABCDEF")
    deadline = perf_counter() + 120
    while perf_counter() < deadline:
        view = client.get(base + "/" + body["id"]).json()
        if view["artifactState"] in ("ready", "failed"):
            break
    assert view["artifactState"] == "ready", view
    assert client.get(f"{base}/{body['id']}/plans/F").status_code == 200


@pytest.mark.parametrize("body", [[], "x", 3, {"schemaVersion": "2"}, {"schemaVersion": 3}])
def test_a_body_that_is_no_figure_is_a_422(saved: Any, body: Any) -> None:
    client, base, _ = saved
    assert client.post(base, json=body).status_code == 422


@pytest.mark.parametrize(("style", "first"), [("lower", "a: "), ("none", "Panel 1: ")])
def test_output_notes_name_panels_as_the_figure_draws_them(
    saved: Any, style: str, first: str
) -> None:
    client, base, ids = saved
    spec = figure(ids, "AB", layout={"rows": 1, "cols": 2, "labelStyle": style})
    board = client.post(base, json=spec).json()["id"]
    project_url = base.rsplit("/", 1)[0]
    preview = client.post(project_url + "/outputs/preview", json={"boardId": board})
    assert preview.status_code == 200, preview.text
    notes = preview.json()["notes"]
    assert notes["legendEn"].startswith(first) and notes["methodsEn"].startswith(first)


def test_the_deferred_save_refuses_what_the_figure_alone_decides(saved: Any) -> None:
    client, base, ids = saved
    spec = figure([ids[0]], "AB", layout={"rows": 1, "cols": 2}, axisMode="shared_y")
    spec["panels"][0]["axes"] = {"y": {"scale": "log10"}}
    response = client.post(base, params={"deferArtifacts": True}, json=spec)
    assert response.status_code == 422, response.text
    assert response.json()["code"] == "figure-shared-axis-control"
    assert client.get(base).json()["boards"] == []


def test_a_deferred_build_the_plan_refuses_keeps_its_reason_and_is_not_retried(saved: Any) -> None:
    client, base, ids = saved
    spec = figure(ids, "A")
    spec["panels"][0]["annotations"] = [{"kind": "hline", "value": 1e12, "label": "far"}]
    response = client.post(base, params={"deferArtifacts": True}, json=spec)
    assert response.status_code == 202, response.text
    url = base + "/" + response.json()["id"]
    deadline = perf_counter() + 120
    while perf_counter() < deadline:
        view = client.get(url).json()
        if view["artifactState"] in ("ready", "failed"):
            break
    assert view["artifactState"] == "failed", view
    assert view["artifactRefusal"] == "annotation-outside"
    assert "Retry" not in view["artifactError"]
    retried = client.post(url + "/retry")
    assert retried.status_code == 422, retried.text
    assert retried.json()["code"] == "figure-build-refused"


def test_duplicating_a_project_copies_a_v2_figure_onto_the_copied_charts(saved: Any) -> None:
    client, base, ids = saved
    client.post(base, json=figure(ids, "ABC", layout={"rows": 1, "cols": 3}))
    identifier = base.split("/")[4]
    response = client.post(f"/api/v1/projects/{identifier}/duplicate")
    assert response.status_code == 200, response.text
    copy = response.json()["projectId"]
    copied_base = base.replace(identifier, copy)
    (copied,) = client.get(copied_base).json()["boards"]
    copied_charts = {
        chart["id"]
        for chart in client.get(f"/api/v1/projects/{copy}/research/charts").json()["charts"]
    }
    layers = [panel["layers"][0]["chartId"] for panel in copied["spec"]["panels"]]
    assert copied["spec"]["schemaVersion"] == 2 and len(layers) == 3
    assert set(layers) <= copied_charts and not set(layers) & set(ids)
    png = client.get(f"{copied_base}/{copied['id']}/artifact/png")
    assert png.content.startswith(b"\x89PNG")


def test_a_v2_figure_that_v1_can_express_makes_a_template(saved: Any) -> None:
    client, base, ids = saved
    body = client.post(base, json=figure(ids, "AB", layout={"rows": 1, "cols": 2})).json()
    assert body["spec"]["schemaVersion"] == 2
    project_url = base.rsplit("/", 1)[0]
    response = client.post(project_url + "/templates", json={"boardId": body["id"], "name": "T"})
    assert response.status_code == 200, response.text


def test_a_template_is_refused_for_a_v2_figure(saved: Any) -> None:
    client, base, ids = saved
    body = client.post(base, json=figure(ids, "ABC", layout={"rows": 1, "cols": 3})).json()
    project_url = base.rsplit("/", 1)[0]
    response = client.post(project_url + "/templates", json={"boardId": body["id"], "name": "T"})
    assert response.status_code == 422, response.text
    assert response.json()["code"] == "reuse-figure-unsupported"
