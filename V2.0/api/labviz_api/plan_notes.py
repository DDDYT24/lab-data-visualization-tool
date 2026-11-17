"""Annotations on a finished plan: text, reference lines, shaded ranges and brackets.

A note is given in data values. The plan adds every other number a renderer would have
to work out (where a bracket's feet end, where a label sits), so preview and export draw
the same thing on any axis scale.
"""

from __future__ import annotations

import copy
from typing import Any

from .plan_axes import axis_position, axis_value
from .render_contract import GEOMETRY
from .render_plan import plan_hash
from .research_preparation import PreparationError

# A bracket's feet drop by this share of the y axis length.
BRACKET_DROP: float = GEOMETRY["annotation"]["bracketDrop"]


def inside(axis: dict[str, Any], *values: float) -> bool:
    low, high = sorted(axis["domain"])
    return all(low <= value <= high for value in values)


def planned_note(note: dict[str, Any], x_axis: dict[str, Any], y_axis: dict[str, Any]) -> Any:
    """The note as the plan states it; refuses a note that leaves the axes."""
    kind = note["kind"]
    if kind == "text":
        xs, ys = [note["x"]], [note["y"]]
    elif kind == "hline":
        xs, ys = [], [note["value"]]
    elif kind == "vline":
        xs, ys = [note["value"]], []
    elif kind == "rect":
        xs, ys = list(note["x"]), list(note["y"])
    else:
        xs, ys = list(note["x"]), [note["y"]]
    if not inside(x_axis, *xs) or not inside(y_axis, *ys):
        raise PreparationError("annotation-outside", "An annotation must lie inside the axes.")
    if kind == "text":
        return {"kind": kind, "x": xs[0], "y": ys[0], "text": note["text"]}
    label = note.get("label", "")
    if kind in ("hline", "vline"):
        return {"kind": kind, "value": note["value"], "label": label}
    if kind == "rect":
        # The corner nearest the start of the x axis and the end of the y axis: top left.
        left = min(xs, key=lambda value: abs(value - x_axis["domain"][0]))
        top = min(ys, key=lambda value: abs(value - y_axis["domain"][1]))
        return {"kind": kind, "x": xs, "y": ys, "label": label, "labelAt": [left, top]}
    scale = y_axis["scale"]
    start, end = (axis_position(scale, edge) for edge in y_axis["domain"])
    foot = axis_value(scale, axis_position(scale, ys[0]) - (end - start) * BRACKET_DROP)
    middle = sum(axis_position(x_axis["scale"], value) for value in xs) / 2
    return {
        "kind": kind,
        "x": xs,
        "y": ys[0],
        "foot": foot,
        "label": label,
        "labelAt": [axis_value(x_axis["scale"], middle), ys[0]],
    }


def apply_notes(plan: dict[str, Any], notes: list[list[dict[str, Any]]]) -> dict[str, Any]:
    """A copy of the plan with each panel's notes added; no notes returns the plan as it is.

    Apply axis controls first: a note is checked against the axes the panel ends up with.
    """
    if len(notes) != len(plan["panels"]):
        raise ValueError("One list of annotations per panel is required.")
    if not any(notes):
        return plan
    changed = copy.deepcopy(plan)
    for panel, wanted in zip(changed["panels"], notes, strict=True):
        if not wanted:
            continue
        if panel["axes"]["z"] is not None:
            raise PreparationError("annotation-3d", "A 3D panel takes no annotations.")
        panel["notes"] = [
            planned_note(note, panel["axes"]["x"], panel["axes"]["y"]) for note in wanted
        ]
    changed["planHash"] = plan_hash(changed)
    return changed
