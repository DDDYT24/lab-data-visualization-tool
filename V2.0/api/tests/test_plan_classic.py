from __future__ import annotations

import json
from typing import Any

import pandas as pd
import pytest
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

from labviz_api.models import ChartSpec
from labviz_api.plan_classic import build_classic_plan
from labviz_api.processing import analyze_chart, classic_plan, render_chart
from labviz_api.render_plan import RenderPlan


def spec(kind: str, **extra: Any) -> ChartSpec:
    series = extra.pop(
        "series",
        [
            {"field": "a", "label": "Alpha", "color": "#2563EB"},
            {"field": "b", "label": "Beta", "color": "#0F766E"},
        ],
    )
    return ChartSpec.model_validate(
        {
            "schemaVersion": 1,
            "type": kind,
            "title": "Classic",
            "xAxis": {"field": "x", "title": "Dose", "unit": "mg"},
            "yAxis": {"field": "a", "title": "Signal", "unit": "mV"},
            "series": series,
            "panelCount": 1,
            "export": {
                "format": "png",
                "dpi": 300,
                "sizePreset": "single-column",
                "grayscalePreview": False,
            },
            **extra,
        }
    )


FRAME = pd.DataFrame(
    {
        "x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        "name": ["p", "q", "r", "s", "t", "u", "v", "w"],
        "a": [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 40.0],
        "b": [8.0, 7.5, 7.0, 6.5, 6.0, 5.5, 5.0, 4.5],
    }
)


def planned(chart: ChartSpec) -> dict[str, Any]:
    plan = build_classic_plan(chart, analyze_chart(FRAME, chart))
    assert plan is not None
    RenderPlan.model_validate_json(json.dumps(plan))
    return plan


def exported_axes(monkeypatch: pytest.MonkeyPatch, chart: ChartSpec) -> list[dict[str, Any]]:
    original = Figure.savefig
    drawn: list[dict[str, Any]] = []

    def capture(figure: Any, *args: Any, **kwargs: Any) -> Any:
        for axis in figure.axes:
            drawn.append(
                {
                    "xlim": axis.get_xlim(),
                    "ylim": axis.get_ylim(),
                    "yticks": [float(value) for value in axis.get_yticks()],
                    "xlabels": [label.get_text() for label in axis.get_xticklabels()],
                    "rects": [p for p in axis.patches if isinstance(p, Rectangle)],
                }
            )
        return original(figure, *args, **kwargs)

    monkeypatch.setattr(Figure, "savefig", capture)
    assert render_chart(FRAME, chart)
    return drawn


@pytest.mark.parametrize(
    ("kind", "extra"),
    [
        ("histogram", {}),
        ("box", {}),
        ("bar", {}),
        ("bar", {"xAxis": {"field": "name", "title": "Name", "unit": ""}}),
    ],
)
def test_export_draws_the_axes_the_plan_states(
    monkeypatch: pytest.MonkeyPatch, kind: str, extra: dict[str, Any]
) -> None:
    chart = spec(kind, **extra)
    plan = planned(chart)
    assert (
        plan["source"]["kind"] == kind
        and build_classic_plan(chart, analyze_chart(FRAME, chart)) == plan
    )
    axes = plan["panels"][0]["axes"]
    drawn = exported_axes(monkeypatch, chart)[0]
    assert list(drawn["xlim"]) == pytest.approx(axes["x"]["domain"])
    assert list(drawn["ylim"]) == pytest.approx(axes["y"]["domain"])
    assert drawn["yticks"] == [tick["value"] for tick in axes["y"]["ticks"]]
    assert drawn["xlabels"] == [tick["label"] for tick in axes["x"]["ticks"]]


def test_histogram_plan_is_the_api_bins() -> None:
    chart = spec("histogram", series=[{"field": "b", "label": "Beta", "color": "#0F766E"}])
    derived = analyze_chart(FRAME, chart)
    plan = build_classic_plan(chart, derived)
    assert plan is not None
    mark = plan["panels"][0]["marks"][0]
    bins = derived["preview"]["histograms"][0]["bins"]
    assert mark["kind"] == "rects" and mark["shape"] == "stairs"
    assert mark["x0"] == [value["start"] for value in bins]
    assert mark["x1"] == [value["end"] for value in bins]
    assert mark["y1"] == [value["count"] for value in bins]
    assert plan["geometry"]["histogram"] == "numpy-auto-v1"
    assert plan["panels"][0]["axes"]["x"]["title"] == "Signal (mV)"


def test_box_plan_is_the_api_summary_and_keeps_outliers_in_frame() -> None:
    chart = spec("box")
    derived = analyze_chart(FRAME, chart)
    plan = build_classic_plan(chart, derived)
    assert plan is not None
    first, second = plan["panels"][0]["marks"]
    box = derived["preview"]["boxes"][0]
    assert (first["low"], first["median"], first["high"]) == (
        box["minimum"],
        box["median"],
        box["maximum"],
    )
    assert first["outliers"] == [40.0]
    assert (first["position"], second["position"]) == (1.0, 2.0)
    y = plan["panels"][0]["axes"]["y"]
    assert y["domain"][1] > 40
    assert [t["label"] for t in plan["panels"][0]["axes"]["x"]["ticks"]] == ["Alpha", "Beta"]
    assert plan["geometry"]["quantile"] == "linear-v1"


def test_bars_start_from_zero_and_text_categories_stay_in_order() -> None:
    chart = spec("bar", xAxis={"field": "name", "title": "Name", "unit": ""})
    panel = planned(chart)["panels"][0]
    assert panel["axes"]["y"]["domain"][0] == 0
    assert [t["label"] for t in panel["axes"]["x"]["ticks"]][:3] == ["p", "q", "r"]
    assert panel["axes"]["x"]["scale"] == "category"
    assert panel["marks"][0]["x0"][0] == pytest.approx(-0.4)


LONG = pd.DataFrame({"x": [float(i) for i in range(4000)]})
LONG["a"] = 1.0
LONG.loc[1111, "a"] = 97.5
LONG.loc[2222, "a"] = -42.5


def long_chart(kind: str) -> ChartSpec:
    return spec(kind, series=[{"field": "a", "label": "Alpha", "color": "#2563EB"}])


@pytest.mark.parametrize("kind", ["line", "scatter", "bar"])
def test_export_draws_every_row_and_preview_keeps_its_axes(
    monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    chart = long_chart(kind)
    analysis = analyze_chart(LONG, chart)
    full = classic_plan(LONG, chart, analysis)
    preview = classic_plan(LONG, chart, analysis, detail="preview")
    assert full and preview
    field = "y1" if kind == "bar" else "y"
    mark, reduced = full["panels"][0]["marks"][0], preview["panels"][0]["marks"][0]
    assert len(mark[field]) == 4000
    assert mark["decimation"] is None
    assert len(reduced[field]) <= 250
    assert reduced["decimation"] == {
        "rule": "min-max-envelope-v1",
        "input": 4000,
        "drawn": len(reduced[field]),
    }
    assert max(reduced[field]) == 97.5 and min(reduced[field]) == -42.5
    assert full["panels"][0]["axes"] == preview["panels"][0]["axes"]

    original = Figure.savefig
    drawn: list[int] = []

    def capture(figure: Any, *args: Any, **kwargs: Any) -> Any:
        axis = figure.axes[0]
        drawn.extend(len(line.get_xdata()) for line in axis.get_lines())
        # 4,000 bars are over BAR_PATCH_LIMIT, so they are one collection.
        drawn.extend(len(collection.get_paths()) for collection in axis.collections)
        if kind == "bar":
            # A fixed place: the search for the best one would test each of the 4,000 bars.
            assert axis.get_legend()._loc == 1
        return original(figure, *args, **kwargs)

    monkeypatch.setattr(Figure, "savefig", capture)
    assert render_chart(LONG, chart)
    assert drawn == [4000]


def test_dense_exports_rasterize_their_data() -> None:
    frame = pd.DataFrame({"x": [float(i) for i in range(20001)]})
    frame["a"] = frame["x"] * 0.5
    chart = long_chart("line")
    plan = classic_plan(frame, chart, analyze_chart(frame, chart))
    assert plan and plan["rendering"]["rasterizePoints"] is True
    assert build_classic_plan(chart, analyze_chart(LONG, chart)) is not None


def test_fit_band_and_error_bars_are_marks_inside_the_axes() -> None:
    chart = spec(
        "scatter",
        series=[{"field": "b", "label": "Beta", "color": "#0F766E"}],
        fitting={"model": "linear", "confidenceBand": True, "showEquation": True},
        uncertainty={"mode": "standard-deviation"},
    )
    panel = planned(chart)["panels"][0]
    kinds = [mark["kind"] for mark in panel["marks"]]
    assert kinds[0] == "points" and "polyline" in kinds and "band" in kinds
    fit = next(mark for mark in panel["marks"] if mark["kind"] == "polyline")
    assert fit["lineStyle"] == "dashed" and fit["markerSize"] == 0
    assert fit["group"].startswith("Beta fit")
    assert {entry["symbol"] for entry in panel["legend"]} == {"marker", "dashed"}
    low, high = panel["axes"]["y"]["domain"]
    band = next(mark for mark in panel["marks"] if mark["kind"] == "band")
    assert low <= min(band["lower"]) and max(band["upper"]) <= high


def test_secondary_axis_is_stated_and_drawn(monkeypatch: pytest.MonkeyPatch) -> None:
    chart = spec(
        "line",
        secondaryYAxis={"enabled": True, "field": "b", "title": "Other", "unit": "A"},
        series=[
            {"field": "a", "label": "Alpha", "color": "#2563EB"},
            {"field": "b", "label": "Beta", "color": "#0F766E", "yAxis": "secondary"},
        ],
    )
    panel = planned(chart)["panels"][0]
    assert [mark["axis"] for mark in panel["marks"]] == ["y", "y2"]
    y2 = panel["axes"]["y2"]
    assert y2["title"] == "Other (A)" and y2["grid"] is False
    assert y2["domain"][1] < panel["axes"]["y"]["domain"][1]
    original = Figure.savefig
    limits: list[Any] = []

    def capture(figure: Any, *args: Any, **kwargs: Any) -> Any:
        limits.extend(axis.get_ylim() for axis in figure.axes)
        return original(figure, *args, **kwargs)

    monkeypatch.setattr(Figure, "savefig", capture)
    assert render_chart(FRAME, chart)
    assert list(limits[1]) == pytest.approx(y2["domain"])


def test_grouped_series_and_text_categories() -> None:
    frame = FRAME.assign(group=["u", "v"] * 4)
    chart = spec(
        "line",
        groupField="group",
        xAxis={"field": "name", "title": "Name", "unit": ""},
        series=[{"field": "b", "label": "Beta", "color": "#0F766E"}],
    )
    plan = classic_plan(frame, chart, analyze_chart(frame, chart))
    assert plan
    panel = plan["panels"][0]
    assert [mark["group"] for mark in panel["marks"]] == ["Beta · u", "Beta · v"]
    assert panel["axes"]["x"]["scale"] == "category"
    assert [tick["label"] for tick in panel["axes"]["x"]["ticks"]][:2] == ["p", "r"]


def test_category_ticks_stay_readable_with_many_rows() -> None:
    frame = pd.DataFrame({"name": [f"row {i}" for i in range(1000)], "a": [1.0] * 1000})
    chart = spec(
        "bar",
        xAxis={"field": "name", "title": "Name", "unit": ""},
        series=[{"field": "a", "label": "Alpha", "color": "#2563EB"}],
    )
    plan = classic_plan(frame, chart, analyze_chart(frame, chart))
    assert plan
    assert len(plan["panels"][0]["axes"]["x"]["ticks"]) <= 250
    assert len(plan["panels"][0]["marks"][0]["y1"]) == 1000


def test_correlation_and_surface_have_no_plan_yet() -> None:
    chart = spec("heatmap")
    assert build_classic_plan(chart, analyze_chart(FRAME, chart)) is None


def test_panels_follow_their_series() -> None:
    chart = spec(
        "histogram",
        panelCount=2,
        series=[
            {"field": "a", "label": "Alpha", "color": "#2563EB", "panel": 1},
            {"field": "b", "label": "Beta", "color": "#0F766E", "panel": 2},
        ],
    )
    plan = planned(chart)
    assert [p["title"] for p in plan["panels"]] == ["Panel 1", "Panel 2"]
    assert [p["marks"][0]["group"] for p in plan["panels"]] == ["Alpha", "Beta"]


@pytest.mark.parametrize(
    ("kind", "visible", "drawn"),
    [
        ("histogram", True, True),
        ("histogram", False, False),
        ("bar", True, True),
        ("bar", False, False),
        ("box", True, False),
    ],
)
def test_plan_states_the_grid_the_export_draws(
    monkeypatch: pytest.MonkeyPatch, kind: str, visible: bool, drawn: bool
) -> None:
    chart = spec(kind)
    chart.export_settings.grid_visible = visible
    axes = planned(chart)["panels"][0]["axes"]
    assert (axes["x"]["grid"], axes["y"]["grid"]) == (drawn, drawn)
    original = Figure.savefig
    exported: list[bool] = []

    def capture(figure: Any, *args: Any, **kwargs: Any) -> Any:
        axis = figure.axes[0]
        lines = axis.get_xgridlines() + axis.get_ygridlines()
        exported.extend(line.get_visible() for line in lines)
        return original(figure, *args, **kwargs)

    monkeypatch.setattr(Figure, "savefig", capture)
    assert render_chart(FRAME, chart)
    assert exported and all(state == drawn for state in exported)


def test_bar_colour_comes_from_the_series_in_its_own_panel() -> None:
    chart = spec(
        "bar",
        panelCount=2,
        series=[
            {"field": "a", "label": "Left", "color": "#2563EB", "panel": 1},
            {"field": "a", "label": "Right", "color": "#0F766E", "panel": 2},
        ],
    )
    colors = [panel["marks"][0]["color"] for panel in planned(chart)["panels"]]
    assert colors == ["#2563eb", "#0f766e"]


def test_grayscale_changes_the_plan_colours() -> None:
    chart = spec("box")
    chart.export_settings.grayscale_preview = True
    mark = planned(chart)["panels"][0]["marks"][0]
    assert len({mark["medianColor"][i : i + 2] for i in (1, 3, 5)}) == 1
