"""Axis controls: scales, manual limits and ticks on a finished plan."""

from __future__ import annotations

import json
from typing import Any

import pandas as pd
import pytest

from labviz_api.models import ChartSpec
from labviz_api.plan_axes import (
    SYMLOG_THRESHOLD,
    apply_axis_controls,
    axis_position,
    axis_value,
    controlled_axis,
    decade_ticks,
    mark_values,
    scale_label,
)
from labviz_api.processing import analyze_chart, classic_plan
from labviz_api.render_plan import RenderPlan, plan_hash
from labviz_api.research_preparation import PreparationError


def chart(kind: str = "scatter") -> ChartSpec:
    return ChartSpec.model_validate(
        {
            "schemaVersion": 1,
            "type": kind,
            "title": "Axes",
            "xAxis": {"field": "x", "title": "X", "unit": ""},
            "yAxis": {"field": "y", "title": "Y", "unit": ""},
            "series": [{"field": "y", "label": "Y", "color": "#2563EB"}],
            "panelCount": 1,
            "export": {
                "format": "png",
                "dpi": 300,
                "sizePreset": "single-column",
                "grayscalePreview": False,
            },
        }
    )


def plan(values: list[float], kind: str = "scatter", detail: str = "export") -> dict[str, Any]:
    data = pd.DataFrame({"x": [float(i + 1) for i in range(len(values))], "y": values})
    spec = chart(kind)
    derived = analyze_chart(data, spec)
    built = (
        classic_plan(data, spec, derived, detail="preview")
        if detail == "preview"
        else classic_plan(data, spec, derived)
    )
    assert built
    return built


def y_axis(built: dict[str, Any]) -> dict[str, Any]:
    axis: dict[str, Any] = built["panels"][0]["axes"]["y"]
    return axis


@pytest.mark.parametrize("scale", ["linear", "log10", "symlog"])
@pytest.mark.parametrize("value", [1e-6, 0.5, 1.0, 37.0, 1e9])
def test_position_and_value_are_inverse(scale: str, value: float) -> None:
    assert axis_value(scale, axis_position(scale, value)) == pytest.approx(value, rel=1e-9)


def test_symlog_is_odd_and_passes_through_zero() -> None:
    assert axis_position("symlog", 0.0) == 0.0
    assert axis_position("symlog", -9.0) == -axis_position("symlog", 9.0)
    assert axis_position("symlog", 9 * SYMLOG_THRESHOLD) == pytest.approx(1.0)


def test_plain_controls_leave_the_plan_identical() -> None:
    built = plan([1.0, 10.0, 100.0])
    assert apply_axis_controls(built, [None]) == built
    assert apply_axis_controls(built, [{"x": {"scale": "linear"}, "y": {}}]) == built


def test_log_axis_gets_decade_ticks_and_keeps_the_marks() -> None:
    built = plan([1.0, 10.0, 100.0, 1000.0])
    logged = apply_axis_controls(built, [{"y": {"scale": "log10"}}])
    axis = y_axis(logged)
    assert axis["scale"] == "log10" and axis["niceRule"] is None
    assert [tick["value"] for tick in axis["ticks"]] == [1.0, 10.0, 100.0, 1000.0]
    assert [tick["label"] for tick in axis["ticks"]] == ["1", "10", "100", "1000"]
    assert 0 < axis["domain"][0] < 1.0 and axis["domain"][1] > 1000.0
    assert logged["panels"][0]["marks"] == built["panels"][0]["marks"]
    assert logged["panels"][0]["axes"]["x"] == built["panels"][0]["axes"]["x"]
    assert logged["planHash"] == plan_hash(logged) != built["planHash"]
    RenderPlan.model_validate_json(json.dumps(logged))


def test_a_narrow_log_range_adds_2_and_5() -> None:
    assert decade_ticks("log10", 2.0, 60.0) == [2.0, 5.0, 10.0, 20.0, 50.0]


def test_many_decades_are_thinned_to_the_tick_limit() -> None:
    ticks = decade_ticks("log10", 1e-12, 1e12)
    assert len(ticks) <= 9 and ticks[0] == 1e-12


def test_symlog_ticks_straddle_zero() -> None:
    expected = [-100.0, -10.0, -1.0, 0.0, 1.0, 10.0, 100.0, 1000.0]
    assert decade_ticks("symlog", -150.0, 1500.0) == expected


@pytest.mark.parametrize(
    ("value", "label"),
    [
        (0.0, "0"),
        (1.0, "1"),
        (0.001, "0.001"),
        (250000.0, "250000"),
        (1e6, "1e6"),
        (2e-7, "2e−7"),
        (-10.0, "−10"),
    ],
)
def test_scale_labels(value: float, label: str) -> None:
    assert scale_label(value) == label


def test_manual_domain_and_ticks() -> None:
    built = plan([1.0, 2.0, 3.0])
    control = {"y": {"domain": [0.0, 4.0], "ticks": [0.0, 2.0, 4.0]}}
    axis = y_axis(apply_axis_controls(built, [control]))
    assert axis["domain"] == [0.0, 4.0] and axis["niceRule"] is None
    assert [(t["value"], t["label"]) for t in axis["ticks"]] == [(0.0, "0"), (2.0, "2"), (4.0, "4")]
    retick = y_axis(apply_axis_controls(built, [{"y": {"domain": [0.0, 4.0]}}]))
    assert retick["niceRule"] == "nice-1-2-5-v1" and retick["ticks"][0]["value"] == 0.0


@pytest.mark.parametrize(
    ("values", "control", "code"),
    [
        ([0.0, 1.0, 2.0], {"y": {"scale": "log10"}}, "axis-log-nonpositive"),
        ([-1.0, 1.0, 2.0], {"y": {"scale": "log10"}}, "axis-log-nonpositive"),
        (
            [1.0, 2.0, 3.0],
            {"y": {"scale": "log10", "domain": [-1.0, 5.0]}},
            "axis-log-nonpositive",
        ),
        (
            [1.0, 2.0, 3.0],
            {"y": {"domain": [0.0, 4.0], "ticks": [0.0, 5.0]}},
            "axis-ticks-outside",
        ),
    ],
)
def test_refused_controls(values: list[float], control: dict[str, Any], code: str) -> None:
    with pytest.raises(PreparationError) as caught:
        apply_axis_controls(plan(values), [control])
    assert caught.value.code == code


def test_bars_cannot_take_a_log_axis_because_they_start_at_zero() -> None:
    with pytest.raises(PreparationError) as caught:
        apply_axis_controls(plan([1.0, 2.0, 3.0], "bar"), [{"y": {"scale": "log10"}}])
    assert caught.value.code == "axis-log-nonpositive"


def test_category_axis_refuses_a_control() -> None:
    axis = {"scale": "category", "domain": [0.0, 1.0], "ticks": [], "title": "", "niceRule": None}
    with pytest.raises(PreparationError) as caught:
        controlled_axis(axis, {"scale": "log10"}, [1.0])
    assert caught.value.code == "axis-control-category"


def test_errorbars_count_both_ends() -> None:
    mark = {"kind": "errorbar", "y": [5.0, 1.0], "error": [1.0, 2.0], "x": [1.0, 2.0]}
    assert mark_values(mark, "y") == [4.0, 6.0, -1.0, 3.0]
    assert mark_values(mark, "x") == [1.0, 2.0]


@pytest.mark.parametrize("scale", ["log10", "symlog"])
def test_preview_and_export_get_the_same_axes(scale: str) -> None:
    values = [float(1 + (i * 37) % 991) for i in range(3000)]
    control: list[dict[str, Any] | None] = [{"y": {"scale": scale}, "x": {"scale": scale}}]
    preview = apply_axis_controls(plan(values, "line", "preview"), control)
    export = apply_axis_controls(plan(values, "line"), control)
    assert preview["panels"][0]["axes"] == export["panels"][0]["axes"]


@pytest.mark.parametrize("scale", ["log10", "symlog"])
def test_matplotlib_places_points_where_the_plan_says(scale: str) -> None:
    from matplotlib.figure import Figure

    from labviz_api.render_plan_draw import draw_panel

    built = apply_axis_controls(plan([1.0, 10.0, 100.0, 1000.0]), [{"y": {"scale": scale}}])
    panel = built["panels"][0]
    figure = Figure(figsize=(4, 3), dpi=100)
    ax = figure.subplots()
    draw_panel(ax, panel)
    figure.canvas.draw()
    domain = panel["axes"]["y"]["domain"]
    low, high = (axis_position(scale, edge) for edge in domain)
    bottom, top = (ax.transData.transform((1.0, edge))[1] for edge in domain)
    for value in (1.0, 10.0, 100.0, 1000.0):
        expected = bottom + (axis_position(scale, value) - low) / (high - low) * (top - bottom)
        assert ax.transData.transform((1.0, value))[1] == pytest.approx(expected, abs=1e-6)
    drawn = [label.get_text() for label in ax.get_yticklabels()]
    assert drawn == [tick["label"] for tick in panel["axes"]["y"]["ticks"]]
