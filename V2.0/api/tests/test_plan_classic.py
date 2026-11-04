from __future__ import annotations

import json
from typing import Any

import pandas as pd
import pytest
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

from labviz_api.models import ChartSpec
from labviz_api.plan_classic import build_classic_plan
from labviz_api.processing import analyze_chart, render_chart
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


def test_bar_with_a_fit_or_secondary_axis_keeps_the_existing_drawing() -> None:
    fitted = spec("bar", fitting={"model": "linear"})
    assert build_classic_plan(fitted, analyze_chart(FRAME, fitted)) is None
    secondary = spec(
        "bar",
        secondaryYAxis={"enabled": True, "field": "b", "title": "Other", "unit": ""},
        series=[
            {"field": "a", "label": "Alpha", "color": "#2563EB"},
            {"field": "b", "label": "Beta", "color": "#0F766E", "yAxis": "secondary"},
        ],
    )
    assert build_classic_plan(secondary, analyze_chart(FRAME, secondary)) is None
    assert render_chart(FRAME, secondary)


def test_other_kinds_have_no_plan_yet() -> None:
    chart = spec("line")
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


def test_grayscale_changes_the_plan_colours() -> None:
    chart = spec("box")
    chart.export_settings.grayscale_preview = True
    mark = planned(chart)["panels"][0]["marks"][0]
    assert len({mark["medianColor"][i : i + 2] for i in (1, 3, 5)}) == 1
