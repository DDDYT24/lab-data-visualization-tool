"""Axis controls on a finished plan: log and symlog scales, manual domains and ticks.

Marks stay in data values. An axis states its scale, and both renderers place every
coordinate with `axis_position`, so a log panel is the same picture in preview and export.
"""

from __future__ import annotations

import copy
import math
from typing import Any

from .render_contract import GEOMETRY
from .render_plan import (
    AXIS_MARGIN,
    AXIS_MAX_TICKS,
    AXIS_NICE_RULE,
    AXIS_PLAIN_EXPONENTS,
    axis_ticks,
    plan_hash,
    tick_labels,
)
from .research_preparation import PreparationError

# Symlog is linear near zero: position = sign(v) * log10(1 + |v| / threshold).
SYMLOG_THRESHOLD: float = GEOMETRY["axis"]["symlogThreshold"]
PLAIN_CONTROL = {"scale": "linear", "domain": None, "ticks": None}
X_FIELDS = {
    "points": ("x",),
    "polyline": ("x",),
    "band": ("x",),
    "errorbar": ("x",),
    "segments": ("x0", "x1"),
    "summary": ("x0", "x1"),
    "rects": ("x0", "x1"),
    "mesh": ("x",),
}
Y_FIELDS = {
    "points": ("y",),
    "polyline": ("y",),
    "band": ("lower", "upper"),
    "segments": ("y0", "y1"),
    "violin": ("grid",),
    "summary": ("y",),
    "rects": ("y0", "y1"),
    "box": ("low", "q1", "median", "q3", "high", "outliers"),
    "mesh": ("y",),
}


def axis_position(scale: str, value: float) -> float:
    """Where a value sits along an axis, in the units the axis is linear in."""
    if scale == "log10":
        return math.log10(value)
    if scale == "symlog":
        return math.copysign(math.log10(1 + abs(value) / SYMLOG_THRESHOLD), value)
    return value


def axis_value(scale: str, position: float) -> float:
    if scale == "log10":
        return float(10.0**position)
    if scale == "symlog":
        return math.copysign((10.0 ** abs(position) - 1) * SYMLOG_THRESHOLD, position)
    return position


def mark_values(mark: dict[str, Any], dimension: str) -> list[float]:
    """Every value a mark places along x or y."""
    kind = mark["kind"]
    if dimension == "x":
        if kind in ("violin", "box"):
            half = mark.get("halfWidth") or max(mark.get("halfWidths", [0.0]))
            return [mark["position"] - half, mark["position"] + half]
        fields = X_FIELDS[kind]
    elif kind == "errorbar":
        return [
            y + sign * error
            for y, error in zip(mark["y"], mark["error"], strict=True)
            for sign in (-1, 1)
        ]
    else:
        fields = Y_FIELDS[kind]
    found: list[float] = []
    for field in fields:
        value = mark[field]
        found.extend(value if isinstance(value, list) else [value])
    return found


def scale_label(value: float) -> str:
    """One tick label that stands alone: plain digits where they stay short."""
    if value == 0:
        return "0"
    exponent = math.floor(math.log10(abs(value)))
    if exponent in AXIS_PLAIN_EXPONENTS:
        text = f"{value:.10f}".rstrip("0").rstrip(".")
    else:
        mantissa = f"{value / 10.0**exponent:.6f}".rstrip("0").rstrip(".")
        text = f"{mantissa}e{exponent}"
    return text.replace("-", "−")


def decade_ticks(scale: str, low: float, high: float) -> list[float]:
    """Powers of ten inside the domain; 2 and 5 times them when under two decades fit."""
    if scale == "symlog":
        top = math.ceil(math.log10(max(abs(low), abs(high), SYMLOG_THRESHOLD)))
        decades = [10.0**power for power in range(0, top + 1)]
        candidates = [-value for value in reversed(decades)] + [0.0] + decades
    else:
        first, last = math.floor(math.log10(low)), math.ceil(math.log10(high))
        candidates = [10.0**power for power in range(first, last + 1)]
        if sum(low <= value <= high for value in candidates) < 2:
            candidates = sorted(
                multiple * 10.0**power
                for power in range(first - 1, last + 1)
                for multiple in (1, 2, 5)
            )
    ticks = [value for value in candidates if low <= value <= high]
    stride = math.ceil(len(ticks) / AXIS_MAX_TICKS)
    return ticks[::stride]


def controlled_axis(
    axis: dict[str, Any], control: dict[str, Any], values: list[float]
) -> dict[str, Any]:
    """The axis a control asks for; the plain control leaves the axis as it was built."""
    control = {**PLAIN_CONTROL, **control}
    if control == PLAIN_CONTROL:
        return axis
    if axis["scale"] == "category":
        raise PreparationError("axis-control-category", "A category axis has no scale or limits.")
    scale = control["scale"]
    if scale == "log10" and any(value <= 0 for value in values):
        raise PreparationError(
            "axis-log-nonpositive", "A log axis cannot show zero or negative values."
        )
    if control["domain"] is not None:
        low, high = (float(value) for value in control["domain"])
    elif scale == "linear" or not values:
        low, high = axis["domain"]
    else:
        start, end = (axis_position(scale, edge) for edge in (min(values), max(values)))
        margin = (end - start) * AXIS_MARGIN or AXIS_MARGIN
        low, high = axis_value(scale, start - margin), axis_value(scale, end + margin)
    if scale == "log10" and low <= 0:
        raise PreparationError("axis-log-nonpositive", "A log axis needs limits above zero.")
    if control["ticks"] is not None:
        chosen = [float(value) for value in control["ticks"]]
        if any(not low <= value <= high for value in chosen):
            raise PreparationError("axis-ticks-outside", "Every tick must lie inside the axis.")
        if scale == "linear":
            step = min(b - a for a, b in zip(chosen, chosen[1:], strict=False))
            labels = tick_labels(chosen, step)
        else:
            labels = [scale_label(value) for value in chosen]
        ticks = [{"value": v, "label": label} for v, label in zip(chosen, labels, strict=True)]
        rule = None
    elif scale == "linear":
        ticks, rule = axis_ticks(low, high), AXIS_NICE_RULE
    else:
        ticks = [
            {"value": value, "label": scale_label(value)}
            for value in decade_ticks(scale, low, high)
        ]
        rule = None
    return {**axis, "scale": scale, "domain": [low, high], "ticks": ticks, "niceRule": rule}


def apply_axis_controls(
    plan: dict[str, Any], controls: list[dict[str, Any] | None]
) -> dict[str, Any]:
    """A copy of the plan with one `{x, y}` control per panel applied; None leaves a panel."""
    if len(controls) != len(plan["panels"]):
        raise ValueError("One axis control per panel is required.")
    changed = copy.deepcopy(plan)
    for panel, control in zip(changed["panels"], controls, strict=True):
        for dimension in ("x", "y"):
            wanted = (control or {}).get(dimension) or {}
            if {**PLAIN_CONTROL, **wanted} == PLAIN_CONTROL:
                continue
            if panel["axes"]["z"] is not None:
                raise PreparationError("axis-control-3d", "A 3D panel keeps linear axes.")
            values = [
                value
                for mark in panel["marks"]
                if dimension == "x" or mark.get("axis", "y") == "y"
                for value in mark_values(mark, dimension)
            ]
            panel["axes"][dimension] = controlled_axis(panel["axes"][dimension], wanted, values)
    changed["planHash"] = plan_hash(changed)
    return changed
