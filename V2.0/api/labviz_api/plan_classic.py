"""RenderPlans for classic charts, built from the analysis both preview and export use.

Histogram, box and plain bar charts are planned here. A bar chart with a fit, uncertainty or
a secondary axis returns no plan and keeps the existing drawing until those marks exist.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from .models import ChartSpec
from .render_contract import (
    GEOMETRY,
    HISTOGRAM_AXIS_PADDING,
    HISTOGRAM_FILL_OPACITY,
    chart_series_color,
)
from .render_plan import AXIS_NICE_RULE, CONTRACT, axis_ticks, grayscale, nice_axis, plan_hash

# Matplotlib's boxplot colours, kept so classic exports look as they did.
BOX_LINE_COLOR = "#000000"
BOX_MEDIAN_COLOR = "#ff7f0e"
BAR_WIDTH = 0.8
BOX_LINE_WIDTH = 1.0


def axis_label(title: str, unit: str) -> str:
    return f"{title} ({unit})" if unit else title


def value_axis(low: float, high: float, title: str) -> dict[str, Any]:
    return {
        "scale": "linear",
        "domain": [low, high],
        "ticks": axis_ticks(low, high),
        "title": title,
        "niceRule": AXIS_NICE_RULE,
    }


def category_axis(
    labels: list[str], low: float, high: float, title: str, first: int
) -> dict[str, Any]:
    return {
        "scale": "category",
        "domain": [low, high],
        "ticks": [
            {"value": float(index), "label": label} for index, label in enumerate(labels, first)
        ],
        "title": title,
        "niceRule": None,
    }


def legend_entries(marks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{"label": mark["group"], "color": mark["color"], "symbol": "marker"} for mark in marks]


def histogram_panel(
    chart: ChartSpec, items: list[tuple[int, Any]], preview: dict[str, Any]
) -> dict[str, Any]:
    marks = []
    for index, item in items:
        bins = next(
            value["bins"]
            for value in preview["histograms"]
            if value["field"] == item.field and value["label"] == item.label
        )
        marks.append(
            {
                "kind": "rects",
                "id": f"histogram:{index}",
                "group": item.label,
                "color": chart_series_color(
                    configured_color=item.color,
                    grayscale=chart.export_settings.grayscale_preview,
                    grouped=False,
                    index=index,
                ).lower(),
                "opacity": HISTOGRAM_FILL_OPACITY,
                "width": chart.export_settings.line_width,
                "shape": "stairs",
                "x0": [float(value["start"]) for value in bins],
                "x1": [float(value["end"]) for value in bins],
                "y0": [0.0] * len(bins),
                "y1": [float(value["count"]) for value in bins],
            }
        )
    low = min(mark["x0"][0] for mark in marks)
    high = max(mark["x1"][-1] for mark in marks)
    padding = (high - low) * HISTOGRAM_AXIS_PADDING
    top = max(max(mark["y1"]) for mark in marks)
    return {
        "marks": marks,
        "legend": legend_entries(marks),
        "axes": {
            "x": value_axis(
                low - padding, high + padding, axis_label(chart.y_axis.title, chart.y_axis.unit)
            ),
            "y": value_axis(0.0, max(1.0, top * (1 + HISTOGRAM_AXIS_PADDING)), "Frequency"),
        },
    }


def box_panel(
    chart: ChartSpec, items: list[tuple[int, Any]], preview: dict[str, Any]
) -> dict[str, Any]:
    use_grayscale = chart.export_settings.grayscale_preview
    # Matplotlib's default box width for evenly placed boxes.
    half_width = min(0.5, max(0.15, 0.15 * (len(items) - 1))) / 2
    marks = []
    for position, (index, item) in enumerate(items, 1):
        box = next(
            value
            for value in preview["boxes"]
            if value["field"] == item.field and value["label"] == item.label
        )
        marks.append(
            {
                "kind": "box",
                "id": f"box:{index}",
                "group": item.label,
                "color": BOX_LINE_COLOR,
                "medianColor": grayscale(BOX_MEDIAN_COLOR) if use_grayscale else BOX_MEDIAN_COLOR,
                "width": BOX_LINE_WIDTH,
                "position": float(position),
                "halfWidth": half_width,
                "low": float(box["minimum"]),
                "q1": float(box["q1"]),
                "median": float(box["median"]),
                "q3": float(box["q3"]),
                "high": float(box["maximum"]),
                "outliers": [float(value) for value in box["outliers"]],
            }
        )
    values = [v for mark in marks for v in (mark["low"], mark["high"], *mark["outliers"])]
    domain, ticks = nice_axis(min(values), max(values))
    return {
        "marks": marks,
        "legend": [],
        "axes": {
            "x": category_axis(
                [item.label for _, item in items], 0.5, len(items) + 0.5, "", first=1
            ),
            "y": {
                "scale": "linear",
                "domain": list(domain),
                "ticks": ticks,
                "title": axis_label(chart.y_axis.title, chart.y_axis.unit),
                "niceRule": AXIS_NICE_RULE,
            },
        },
    }


def is_number(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def bar_domain(low: float, high: float) -> tuple[float, float]:
    """Bars grow from zero, so the axis starts there and only the far end gets a margin."""
    (start, end), _ = nice_axis(low, high)
    if low < high:
        start = 0.0 if low == 0 else start
        end = 0.0 if high == 0 else end
    return start, end


def bar_panel(
    chart: ChartSpec,
    panel_index: int,
    analyses: list[dict[str, Any]],
    series_colors: dict[str, str] | None,
) -> dict[str, Any] | None:
    listed = [
        (index, analysis)
        for index, analysis in enumerate(analyses)
        if analysis["panel"] == panel_index and analysis["points"]
    ]
    xs = [point["x"] for _, analysis in listed for point in analysis["points"]]
    if not all(is_number(x) or isinstance(x, str) for x in xs):
        return None
    numeric = all(is_number(x) for x in xs)
    # Matplotlib places text categories in order of first appearance.
    categories = [] if numeric else list(dict.fromkeys(str(x) for x in xs))
    positions = {label: float(index) for index, label in enumerate(categories)}
    marks = []
    for order, (index, analysis) in enumerate(listed):
        item = next(
            series
            for series in chart.series
            if series.field == analysis["field"] and series.panel == panel_index
        )
        color = chart_series_color(
            configured_color=item.color,
            grayscale=chart.export_settings.grayscale_preview,
            grouped=bool(chart.group_field),
            index=index,
        )
        centers = [float(p["x"]) if numeric else positions[str(p["x"])] for p in analysis["points"]]
        marks.append(
            {
                "kind": "rects",
                "id": f"bar:{index}",
                "group": analysis["label"],
                "color": (series_colors or {}).get(analysis["label"], color).lower(),
                "opacity": max(0.45, 0.85 - order * 0.08),
                "width": chart.export_settings.line_width,
                "shape": "bars",
                "x0": [center - BAR_WIDTH / 2 for center in centers],
                "x1": [center + BAR_WIDTH / 2 for center in centers],
                "y0": [0.0] * len(centers),
                "y1": [float(p["y"]) for p in analysis["points"]],
            }
        )
    if not marks:
        return None
    x_title = axis_label(chart.x_axis.title, chart.x_axis.unit)
    y_title = axis_label(chart.y_axis.title, chart.y_axis.unit)
    (low, high), _ = nice_axis(
        min(min(mark["x0"]) for mark in marks), max(max(mark["x1"]) for mark in marks)
    )
    y_values = [0.0, *(value for mark in marks for value in mark["y1"])]
    y_low, y_high = bar_domain(min(y_values), max(y_values))
    return {
        "marks": marks,
        "legend": legend_entries(marks),
        "axes": {
            "x": value_axis(low, high, x_title)
            if numeric
            else category_axis(categories, low, high, x_title, first=0),
            "y": value_axis(y_low, y_high, y_title),
        },
    }


def build_classic_plan(
    chart: ChartSpec, derived: dict[str, Any], *, series_colors: dict[str, str] | None = None
) -> dict[str, Any] | None:
    """Plan a classic chart, or return None for kinds and options not yet on plans."""
    if chart.type not in {"histogram", "box", "bar"}:
        return None
    analyses = derived["series"]
    uses_secondary = chart.secondary_y_axis.enabled and any(
        item.y_axis == "secondary" for item in chart.series
    )
    if chart.type == "bar" and (
        uses_secondary or any(a["fit"] or a["uncertainty"] for a in analyses)
    ):
        return None
    # Box exports have never drawn a grid.
    grid = chart.export_settings.grid_visible and chart.type != "box"
    panels = []
    for panel_index in range(1, chart.panel_count + 1):
        items = [(i, s) for i, s in enumerate(chart.series) if s.panel == panel_index]
        if not items:
            continue
        panel = (
            histogram_panel(chart, items, derived["preview"])
            if chart.type == "histogram"
            else box_panel(chart, items, derived["preview"])
            if chart.type == "box"
            else bar_panel(chart, panel_index, analyses, series_colors)
        )
        if panel is None:
            return None
        for axis in panel["axes"].values():
            axis["grid"] = grid
        title = f"{chart.title}\n{chart.subtitle}" if chart.subtitle else chart.title
        panels.append(
            {
                "id": str(panel_index),
                "title": title if chart.panel_count == 1 else f"Panel {panel_index}",
                **panel,
            }
        )
    source = json.dumps(
        [chart.model_dump(mode="json"), analyses, derived["preview"]], sort_keys=True, default=str
    )
    plan: dict[str, Any] = {
        "schemaVersion": 1,
        "contract": CONTRACT,
        "source": {
            "kind": chart.type,
            "resultSha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
            "algorithmVersion": "classic-charts-v1",
        },
        "geometry": {
            "contract": GEOMETRY["contractVersion"],
            "axis": AXIS_NICE_RULE,
            "jitter": None,
            "density": None,
            "histogram": GEOMETRY["histogram"]["rule"] if chart.type == "histogram" else None,
            "quantile": GEOMETRY["quantile"]["rule"] if chart.type == "box" else None,
        },
        "rendering": {"pointRendering": "auto", "rasterizePoints": False},
        "panels": panels,
        "disclosures": [],
    }
    plan = json.loads(json.dumps(plan, allow_nan=False))
    plan["planHash"] = plan_hash(plan)
    return plan
