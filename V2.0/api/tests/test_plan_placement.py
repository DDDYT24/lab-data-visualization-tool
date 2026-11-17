"""The placement fixture the browser tests also read: positions, a noted log plan, pixels."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest

from labviz_api.plan_axes import axis_position
from labviz_api.render_plan import RenderPlan

ROOT = Path(__file__).resolve().parents[1]
FIXTURE: dict[str, Any] = json.loads(
    (ROOT.parent / "contracts/fixtures/v24/plan-placement.json").read_text(encoding="utf-8")
)


def test_the_fixture_is_what_the_code_builds_now() -> None:
    spec = importlib.util.spec_from_file_location(
        "build_plan_placement_fixture", ROOT / "scripts/build_plan_placement_fixture.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert json.loads(json.dumps(module.build())) == FIXTURE
    RenderPlan.model_validate_json(json.dumps(FIXTURE["plan"]))


@pytest.mark.parametrize("sample", FIXTURE["positions"])
def test_positions(sample: dict[str, Any]) -> None:
    assert axis_position(sample["scale"], sample["value"]) == sample["position"]


@pytest.mark.parametrize("drawer", ["classic", "research"])
def test_matplotlib_is_linear_in_the_fixture_positions(drawer: str) -> None:
    from matplotlib.figure import Figure

    from labviz_api.render_plan_draw import draw_panel
    from labviz_api.research_chart_render import draw_plan

    plan = FIXTURE["plan"]
    panel = plan["panels"][0]
    figure = Figure(figsize=(4, 3), dpi=100)
    ax = figure.subplots()
    if drawer == "classic":
        draw_panel(ax, panel)
    else:
        draw_plan(plan, target_figure=figure, target_axis=ax)
    figure.canvas.draw()
    x_low, x_high = (axis_position("log10", edge) for edge in panel["axes"]["x"]["domain"])
    y_low, y_high = (axis_position("log10", edge) for edge in panel["axes"]["y"]["domain"])
    (left, bottom), (right, top) = ax.transData.transform(
        [[axis["domain"][i] for axis in (panel["axes"]["x"], panel["axes"]["y"])] for i in (0, 1)]
    )

    def pixel(position: list[float]) -> tuple[float, float]:
        return (
            left + (position[0] - x_low) / (x_high - x_low) * (right - left),
            bottom + (position[1] - y_low) / (y_high - y_low) * (top - bottom),
        )

    points = next(mark for mark in panel["marks"] if mark["kind"] == "points")
    drawn = ax.transData.transform(list(zip(points["x"], points["y"], strict=True)))
    for got, position in zip(drawn, FIXTURE["placed"]["points"], strict=True):
        assert tuple(got) == pytest.approx(pixel(position), abs=1e-6)
    bracket = panel["notes"][4]
    corners = [
        (bracket["x"][0], bracket["foot"]),
        (bracket["x"][0], bracket["y"]),
        (bracket["x"][1], bracket["y"]),
        (bracket["x"][1], bracket["foot"]),
    ]
    line = next(
        line
        for line in ax.lines
        if [tuple(xy) for xy in line.get_xydata().tolist()] == corners  # type: ignore[union-attr]
    )
    for got, position in zip(
        ax.transData.transform(line.get_xydata()), FIXTURE["placed"]["bracket"], strict=True
    ):
        assert tuple(got) == pytest.approx(pixel(position), abs=1e-6)
