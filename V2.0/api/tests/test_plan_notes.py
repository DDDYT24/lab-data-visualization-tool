"""Annotations on a finished plan and how Matplotlib draws them."""

from __future__ import annotations

import json
from typing import Any

import pytest
from test_plan_axes import plan

from labviz_api.plan_axes import apply_axis_controls, axis_position
from labviz_api.plan_notes import BRACKET_DROP, apply_notes
from labviz_api.render_plan import RenderPlan, plan_hash
from labviz_api.research_preparation import PreparationError

VALUES = [1.0, 10.0, 100.0, 1000.0]
NOTES: list[dict[str, Any]] = [
    {"kind": "text", "x": 2.0, "y": 500.0, "text": "peak"},
    {"kind": "hline", "value": 100.0, "label": "limit"},
    {"kind": "vline", "value": 3.0, "label": ""},
    {"kind": "rect", "x": [1.5, 2.5], "y": [10.0, 400.0], "label": "window"},
    {"kind": "bracket", "x": [2.0, 4.0], "y": 800.0, "label": "p < 0.01"},
]


def noted(scale: str = "linear", notes: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    control: list[dict[str, Any] | None] = [{"x": {"scale": scale}, "y": {"scale": scale}}]
    return apply_notes(apply_axis_controls(plan(VALUES), control), [notes or NOTES])


def test_no_notes_returns_the_plan_itself() -> None:
    built = plan(VALUES)
    assert apply_notes(built, [[]]) is built
    assert "notes" not in built["panels"][0]
    RenderPlan.model_validate_json(json.dumps(built))


@pytest.mark.parametrize("scale", ["linear", "log10", "symlog"])
def test_notes_are_planned_and_the_plan_is_rehashed(scale: str) -> None:
    before = apply_axis_controls(plan(VALUES), [{"x": {"scale": scale}, "y": {"scale": scale}}])
    built = apply_notes(before, [NOTES])
    RenderPlan.model_validate_json(json.dumps(built))
    assert built["planHash"] == plan_hash(built) != before["planHash"]
    assert before["panels"][0].get("notes") is None
    kinds = [note["kind"] for note in built["panels"][0]["notes"]]
    assert kinds == ["text", "hline", "vline", "rect", "bracket"]
    assert built["panels"][0]["marks"] == before["panels"][0]["marks"]


@pytest.mark.parametrize("scale", ["linear", "log10", "symlog"])
def test_bracket_feet_and_label_follow_the_axis_scale(scale: str) -> None:
    panel = noted(scale)["panels"][0]
    bracket = panel["notes"][4]
    low, high = (axis_position(scale, edge) for edge in panel["axes"]["y"]["domain"])
    drop = axis_position(scale, bracket["y"]) - axis_position(scale, bracket["foot"])
    assert drop == pytest.approx((high - low) * BRACKET_DROP)
    middle = (axis_position(scale, 2.0) + axis_position(scale, 4.0)) / 2
    assert axis_position(scale, bracket["labelAt"][0]) == pytest.approx(middle)
    assert bracket["labelAt"][1] == 800.0


def test_rect_label_sits_at_the_top_left_even_on_a_reversed_axis() -> None:
    built = noted()
    assert built["panels"][0]["notes"][3]["labelAt"] == [1.5, 400.0]
    flipped = plan(VALUES)
    axis = flipped["panels"][0]["axes"]["y"]
    axis["domain"] = axis["domain"][::-1]
    rect = apply_notes(flipped, [[NOTES[3]]])["panels"][0]["notes"][0]
    assert rect["labelAt"] == [1.5, 10.0]


@pytest.mark.parametrize(
    "note",
    [
        {"kind": "text", "x": 2.0, "y": 1e9, "text": "far"},
        {"kind": "hline", "value": -1e9, "label": ""},
        {"kind": "vline", "value": 99.0, "label": ""},
        {"kind": "rect", "x": [1.5, 99.0], "y": [10.0, 400.0], "label": ""},
        {"kind": "bracket", "x": [2.0, 4.0], "y": 1e9, "label": ""},
    ],
)
def test_a_note_outside_the_axes_is_refused(note: dict[str, Any]) -> None:
    with pytest.raises(PreparationError) as refused:
        noted(notes=[note])
    assert refused.value.code == "annotation-outside"


def test_a_log_axis_refuses_a_note_at_zero() -> None:
    with pytest.raises(PreparationError) as refused:
        noted("log10", [{"kind": "hline", "value": 0.0, "label": ""}])
    assert refused.value.code == "annotation-outside"


def test_a_3d_panel_takes_no_notes() -> None:
    built = plan(VALUES)
    built["panels"][0]["axes"]["z"] = built["panels"][0]["axes"]["y"]
    with pytest.raises(PreparationError) as refused:
        apply_notes(built, [NOTES])
    assert refused.value.code == "annotation-3d"


def test_one_list_per_panel_is_required() -> None:
    with pytest.raises(ValueError, match="per panel"):
        apply_notes(plan(VALUES), [])


def test_preview_and_export_get_the_same_notes() -> None:
    preview = apply_notes(plan(VALUES, "line", "preview"), [NOTES])
    export = apply_notes(plan(VALUES, "line"), [NOTES])
    assert preview["panels"][0]["notes"] == export["panels"][0]["notes"]


@pytest.mark.parametrize("scale", ["linear", "log10"])
def test_matplotlib_draws_each_note_where_the_plan_says(scale: str) -> None:
    import numpy as np
    from matplotlib.figure import Figure
    from matplotlib.patches import Rectangle

    from labviz_api.render_plan_draw import NOTE, draw_panel

    panel = noted(scale)["panels"][0]
    figure = Figure(figsize=(4, 3), dpi=100)
    ax = figure.subplots()
    before = len(ax.lines)
    draw_panel(ax, panel)
    figure.canvas.draw()
    text, hline, vline, rect, bracket = panel["notes"]

    lines = {
        tuple(map(tuple, np.asarray(line.get_xydata()).tolist())): line
        for line in ax.lines[before:]
    }
    feet = (
        (bracket["x"][0], bracket["foot"]),
        (bracket["x"][0], bracket["y"]),
        (bracket["x"][1], bracket["y"]),
        (bracket["x"][1], bracket["foot"]),
    )
    assert feet in lines
    assert lines[feet].get_color() == NOTE["color"]
    assert any(set(y for _, y in points) == {hline["value"]} for points in lines)
    assert any(set(x for x, _ in points) == {vline["value"]} for points in lines)

    patches = [patch for patch in ax.patches if isinstance(patch, Rectangle)]
    assert len(patches) == 1
    assert patches[0].get_xy() == (rect["x"][0], rect["y"][0])
    assert patches[0].get_width() == pytest.approx(rect["x"][1] - rect["x"][0])
    assert patches[0].get_alpha() == NOTE["rectOpacity"]

    words: dict[str, Any] = {item.get_text(): item for item in ax.texts}
    assert set(words) == {"peak", "limit", "window", "p < 0.01"}
    assert words["peak"].get_position() == (text["x"], text["y"])
    assert words["window"].xy == tuple(rect["labelAt"])
    assert words["p < 0.01"].xy == tuple(bracket["labelAt"])
    assert words["limit"].xy == (1.0, hline["value"])
    # The bracket label is centred between the feet on the drawn axis, not in data values.
    left, right = (ax.transData.transform((x, bracket["y"]))[0] for x in bracket["x"])
    centre = ax.transData.transform(tuple(bracket["labelAt"]))[0]
    assert centre == pytest.approx((left + right) / 2, abs=1e-6)


def test_a_plan_without_notes_draws_as_before() -> None:
    from matplotlib.figure import Figure

    from labviz_api.render_plan_draw import draw_panel

    ax = Figure(figsize=(4, 3), dpi=100).subplots()
    draw_panel(ax, plan(VALUES)["panels"][0])
    assert not ax.texts and not ax.patches
