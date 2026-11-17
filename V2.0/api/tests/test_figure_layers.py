"""S4e: several saved research charts of the same data overlaid in one panel."""

from __future__ import annotations

import json
from time import perf_counter
from typing import Any

import pytest
from test_figure_layout import figure, saved  # noqa: F401  (`saved` is a fixture)

from labviz_api.figure_spec import FigureSpec, layer_ids
from labviz_api.render_plan import RenderPlan, merge_layers
from labviz_api.research_figure_render import check_layers
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


def layered(points: str, violin: str, **style: Any) -> dict[str, Any]:
    spec = figure([points], "A")
    spec["panels"][0]["layers"] = [
        {"kind": "chart", "chartId": points, "style": {"zIndex": style.get("points", 1)}},
        {"kind": "chart", "chartId": violin, "style": {"zIndex": style.get("violin", 0)}},
    ]
    return spec


def stored_plan(client: Any, url: str) -> dict[str, Any]:
    return dict(json.loads(client.get(url + "/plans/A").content))


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


def plan(y: tuple[float, float], marks: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "planHash": "0" * 64,
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
    high = plan((-5, 30), [{"kind": "violin", "id": "density:0"}])
    merged = merge_layers([low, high])
    panel = merged["panels"][0]
    assert panel["axes"]["y"]["domain"] == [-5, 30]
    assert [tick["value"] for tick in panel["axes"]["y"]["ticks"]][-1] <= 30
    assert panel["marks"] == [
        {"kind": "points", "id": "points:0"},
        {"kind": "violin", "id": "L1:density:0", "layer": 1},
    ]
    assert panel["legend"] == low["panels"][0]["legend"] and merged["disclosures"] == ["n=4"]
    other = plan((0, 10), [])
    other["panels"][0]["axes"]["x"] = axis("category", -0.5, 1.5, [0, 1])
    other["panels"][0]["axes"]["x"]["ticks"][1]["label"] = "Treated"
    with pytest.raises(PreparationError) as caught:
        merge_layers([low, other])
    assert caught.value.code == "figure-layer-axis"
