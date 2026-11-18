"""S4e: several saved research charts of the same data overlaid in one panel."""

from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter
from typing import Any

import pytest
from fastapi.testclient import TestClient
from test_figure_layout import figure, saved  # noqa: F401  (`saved` is a fixture)
from test_research_api import project, settings, unlock
from test_research_chart_api import applied
from test_research_charts import prepared_csv
from test_research_figure_board import charts
from test_research_figure_composition import legacy

from labviz_api.figure_spec import FigureSpec, layer_ids
from labviz_api.main import create_app
from labviz_api.render_plan import RenderPlan, merge_layers
from labviz_api.research_charts import ScientificChartSpec, compute_chart
from labviz_api.research_figure_render import check_layers, render_board
from labviz_api.research_figure_sources import scientific_source
from labviz_api.research_preparation import PreparationError


def same_revision_points(client: Any, base: str, chart_id: str) -> str:
    """Raw points of the revision the violin shows (charts are made on the current revision)."""
    charts_url = base.replace("figure-boards", "charts")
    revision = next(
        chart["revisionId"]
        for chart in client.get(charts_url).json()["charts"]
        if chart["id"] == chart_id
    )
    response = client.post(charts_url, json={"revisionId": revision, "spec": {"title": "Points"}})
    assert response.status_code == 200, response.text
    return str(response.json()["id"])


def layered(own: str, other: str, **style: Any) -> dict[str, Any]:
    """Two layers; by default the second listed is drawn under the panel's own chart."""
    spec = figure([own], "A")
    spec["panels"][0]["layers"] = [
        {"kind": "chart", "chartId": own, "style": {"zIndex": style.get("points", 1)}},
        {"kind": "chart", "chartId": other, "style": {"zIndex": style.get("violin", 0)}},
    ]
    return spec


def stored_plan(client: Any, url: str) -> dict[str, Any]:
    return dict(json.loads(client.get(url + "/plans/A").content))


def saved_board(client: Any, base: str, spec: dict[str, Any], deferred: bool) -> str:
    """Saves a figure on either path and waits until its files are drawn."""
    response = client.post(base, params={"deferArtifacts": deferred}, json=spec)
    assert response.status_code == (202 if deferred else 200), response.text
    url = base + "/" + response.json()["id"]
    deadline = perf_counter() + 120
    while deferred and perf_counter() < deadline:
        if client.get(url).json()["artifactState"] in ("ready", "failed"):
            break
    assert client.get(url).json()["artifactState"] == "ready"
    return url


KDE = "Gaussian KDE of observations; density is not extra data."


def test_layers_order_by_z_index_then_place() -> None:
    spec = FigureSpec.model_validate(
        {
            "schemaVersion": 2,
            "panels": [
                {
                    "id": "p1",
                    "label": "A",
                    "layers": [
                        {"kind": "chart", "chartId": "top", "style": {"zIndex": 2}},
                        {"kind": "chart", "chartId": "low"},
                        {"kind": "chart", "chartId": "mid"},
                    ],
                }
            ],
        }
    )
    assert layer_ids(spec) == [["low", "mid", "top"]]


@pytest.mark.parametrize("deferred", [False, True])
def test_a_violin_under_raw_points_saves_and_draws_both(saved: Any, deferred: bool) -> None:  # noqa: F811
    client, base, ids = saved
    violin = ids[1]
    points = same_revision_points(client, base, violin)
    response = client.post(base, params={"deferArtifacts": deferred}, json=layered(points, violin))
    assert response.status_code == (202 if deferred else 200), response.text
    body = response.json()
    url = base + "/" + body["id"]
    deadline = perf_counter() + 120
    while deferred and perf_counter() < deadline:
        if client.get(url).json()["artifactState"] in ("ready", "failed"):
            break
    assert client.get(url).json()["artifactState"] == "ready"
    plan = stored_plan(client, url)
    RenderPlan.model_validate_json(json.dumps(plan))
    marks = plan["panels"][0]["marks"]
    # The violin has zIndex 0, so it is the bottom layer even though it is listed second.
    assert marks[0]["kind"] == "violin" and "layer" not in marks[0]
    top = [mark for mark in marks if mark.get("layer") == 1]
    assert top and all(mark["id"].startswith("L1:") for mark in top)
    assert {mark["kind"] for mark in top} >= {"points"}
    # The panel stands for its first listed chart: the raw points.
    assert body["panels"][0]["chartId"] == points
    assert client.get(url + "/artifact/png").content.startswith(b"\x89PNG")
    # Both layers show the same observations, so the panel's rows are one set.
    scene = client.get(url + "/interaction").json()
    assert scene["totalRows"] == body["panels"][0]["observations"]
    selected = next(mark for mark in scene["marks"] if mark["kind"] == "point")
    picked = client.get(url + "/interaction", params={"mark": selected["id"]}).json()
    assert picked["selectedInputs"] == 1


def test_a_single_layer_plan_does_not_change(saved: Any) -> None:  # noqa: F811
    client, base, ids = saved
    one = client.post(base, json=figure([ids[0]], "A")).json()
    plan = stored_plan(client, base + "/" + one["id"])
    assert all("layer" not in mark for mark in plan["panels"][0]["marks"])
    assert merge_layers([plan]) is plan


@pytest.mark.parametrize("deferred", [False, True])
def test_layers_of_two_revisions_are_refused(saved: Any, deferred: bool) -> None:  # noqa: F811
    client, base, ids = saved
    response = client.post(base, params={"deferArtifacts": deferred}, json=layered(ids[0], ids[1]))
    assert response.status_code == 422, response.text
    assert response.json()["code"] == "figure-layer-revision"
    assert client.get(base).json()["boards"] == []


HEADER = {"kind": "grouped_strip", "revisionId": "r1", "xSignature": ["x"], "ySignature": ["y"]}


@pytest.mark.parametrize(
    ("change", "code"),
    [
        ({"kind": "line"}, "figure-layer-unplanned"),
        ({"revisionId": "r2"}, "figure-layer-revision"),
        ({"ySignature": ["other"]}, "figure-layer-axis"),
    ],
)
def test_layer_rules(change: dict[str, Any], code: str) -> None:
    check_layers([[HEADER], [HEADER | {"kind": "line"}]])  # a lone captured chart is fine
    with pytest.raises(PreparationError) as caught:
        check_layers([[HEADER, HEADER | change]])
    assert caught.value.code == code


def axis(scale: str, low: float, high: float, ticks: list[float]) -> dict[str, Any]:
    return {
        "scale": scale,
        "domain": [low, high],
        "ticks": [{"value": v, "label": str(v)} for v in ticks],
        "title": "",
        "niceRule": None,
        "grid": True,
    }


def plan(
    y: tuple[float, float], marks: list[dict[str, Any]], density: str | None = None
) -> dict[str, Any]:
    return {
        "planHash": "0" * 64,
        "geometry": {"axis": "a", "density": density},
        "disclosures": ["n=4"],
        "panels": [
            {
                "id": "A",
                "title": "",
                "legend": [{"label": "A", "color": "#000000", "symbol": "marker"}],
                "marks": marks,
                "axes": {
                    "x": axis("category", -0.5, 1.5, [0, 1]),
                    "y": axis("linear", *y, [0]),
                    "y2": None,
                    "z": None,
                },
            }
        ],
    }


def test_merge_spans_every_layer_and_refuses_other_categories() -> None:
    low = plan((0, 10), [{"kind": "points", "id": "points:0"}])
    high = plan((-5, 30), [{"kind": "violin", "id": "density:0"}], density="kde")
    merged = merge_layers([low, high])
    panel = merged["panels"][0]
    assert panel["axes"]["y"]["domain"] == [-5, 30]
    assert [tick["value"] for tick in panel["axes"]["y"]["ticks"]][-1] <= 30
    assert panel["marks"] == [
        {"kind": "points", "id": "points:0"},
        {"kind": "violin", "id": "L1:density:0", "layer": 1},
    ]
    assert panel["legend"] == low["panels"][0]["legend"] and merged["disclosures"] == ["n=4"]
    assert merged["geometry"] == {"axis": "a", "density": "kde"}
    other = plan((0, 10), [])
    other["panels"][0]["axes"]["x"] = axis("category", -0.5, 1.5, [0, 1])
    other["panels"][0]["axes"]["x"]["ticks"][1]["label"] = "Treated"
    with pytest.raises(PreparationError) as caught:
        merge_layers([low, other])
    assert caught.value.code == "figure-layer-axis"


def test_points_under_a_violin_disclose_and_note_the_violin(saved: Any) -> None:  # noqa: F811
    client, base, ids = saved
    violin = ids[1]
    points = same_revision_points(client, base, violin)
    # The panel's own chart (points) is the bottom layer; the violin is drawn over it.
    url = saved_board(client, base, layered(points, violin, points=0, violin=1), False)
    plan = stored_plan(client, url)
    assert plan["source"]["kind"] == "grouped_strip"
    assert plan["geometry"]["density"] is not None
    assert KDE in plan["disclosures"]
    project_url = base.rsplit("/", 1)[0]
    preview = client.post(project_url + "/outputs/preview", json={"boardId": url.rsplit("/", 1)[1]})
    assert preview.status_code == 200, preview.text
    (panel,) = preview.json()["manifest"]["panels"]
    assert panel["chartId"] == points
    (layer,) = panel["layers"]
    assert layer["chartId"] == violin and layer["kind"] == "violin"
    assert layer["method"]["densityMethods"] and layer["resultSha256"]
    notes = preview.json()["notes"]
    assert "A, layer violin: " in notes["methodsEn"] and "Gaussian KDE" in notes["methodsEn"]
    assert "A，图层 violin：" in notes["methodsZh"]


def test_a_single_layer_output_has_no_layer_list(saved: Any) -> None:  # noqa: F811
    client, base, ids = saved
    board = client.post(base, json=figure([ids[0]], "A")).json()["id"]
    preview = client.post(base.rsplit("/", 1)[0] + "/outputs/preview", json={"boardId": board})
    assert "layers" not in preview.json()["manifest"]["panels"][0]


@pytest.mark.parametrize("deferred", [False, True])
def test_a_captured_chart_with_a_layer_is_refused_on_save(tmp_path: Path, deferred: bool) -> None:
    with TestClient(create_app(settings(tmp_path))) as client:
        unlock(client)
        identifier, _, captured = legacy(client, "line")
        research = charts(client, identifier)[0]["id"]
        base = f"/api/v1/projects/{identifier}/research/figure-boards"
        response = client.post(
            base, params={"deferArtifacts": deferred}, json=layered(research, captured)
        )
        assert response.status_code == 422, response.text
        assert response.json()["code"] == "figure-layer-unplanned"
        assert client.get(base).json()["boards"] == []


PAIRED_SOURCE = (
    b"s,p,t,v,g,u\na,p1,before,1,A,mV\na,p1,after,3,A,mV\nb,p2,before,4,B,mV\nb,p2,after,6,B,mV\n"
)
PAIRED_PLAN = {
    "mapping": {
        "value": "v",
        "sample": "s",
        "experimentalUnit": "s",
        "pair": "p",
        "time": "t",
        "group": "g",
        "unit": "u",
    },
    "analysisUnit": "experimental_unit",
}


@pytest.mark.parametrize("deferred", [False, True])
def test_layers_on_other_axes_are_refused_on_save(tmp_path: Path, deferred: bool) -> None:
    with TestClient(create_app(settings(tmp_path))) as client:
        unlock(client)
        identifier = project(client)
        revision = applied(client, identifier, PAIRED_SOURCE, PAIRED_PLAN)
        made = [
            client.post(
                f"/api/v1/projects/{identifier}/research/charts",
                json={"revisionId": revision, "spec": spec},
            )
            for spec in ({"title": "Points"}, {"kind": "paired", "pairTimes": ["before", "after"]})
        ]
        assert all(response.status_code == 200 for response in made), made[1].text
        points, paired = (response.json()["id"] for response in made)
        base = f"/api/v1/projects/{identifier}/research/figure-boards"
        response = client.post(
            base, params={"deferArtifacts": deferred}, json=layered(points, paired)
        )
        assert response.status_code == 422, response.text
        assert response.json()["code"] == "figure-layer-axis"
        assert client.get(base).json()["boards"] == []


@pytest.mark.parametrize("deferred", [False, True])
def test_a_chart_twice_in_one_panel_is_refused(saved: Any, deferred: bool) -> None:  # noqa: F811
    client, base, ids = saved
    response = client.post(base, params={"deferArtifacts": deferred}, json=layered(ids[1], ids[1]))
    assert response.status_code == 422, response.text
    assert response.json()["code"] == "figure-layer-repeated"


def test_duplicating_a_project_copies_every_layer(saved: Any) -> None:  # noqa: F811
    client, base, ids = saved
    violin = ids[1]
    points = same_revision_points(client, base, violin)
    saved_board(client, base, layered(points, violin), False)
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
    layers = [layer["chartId"] for layer in copied["spec"]["panels"][0]["layers"]]
    assert len(layers) == 2 and set(layers) <= copied_charts and not set(layers) & {points, violin}
    url = f"{copied_base}/{copied['id']}"
    assert client.get(url + "/artifact/png").content.startswith(b"\x89PNG")
    assert any(mark.get("layer") == 1 for mark in stored_plan(client, url)["panels"][0]["marks"])


def test_points_of_any_layer_are_noted_as_embedded() -> None:
    data = prepared_csv("v,g\n0,A\n1,A\n2,A\n3,B\n4,B\n6,B\n", group="g")
    violin = compute_chart(data, ScientificChartSpec(kind="violin", show_points=False))
    points = compute_chart(data, ScientificChartSpec())
    sources = [scientific_source(result, "r1") for result in (violin, points)]
    spec = FigureSpec.model_validate(
        {
            "schemaVersion": 2,
            "pointRendering": "raster",
            "panels": [{"id": "p1", "label": "A", "layers": [{"kind": "chart", "chartId": "v"}]}],
        }
    )
    _, alone = render_board(sources[:1], spec)
    assert alone["exportRendering"]["rasterPanels"] == []
    _, both = render_board(sources[:1], spec, layers=[sources])
    assert both["exportRendering"]["rasterPanels"] == ["A"]
