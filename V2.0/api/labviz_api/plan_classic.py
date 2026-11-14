"""RenderPlans for classic charts, built from the analysis both preview and export use.

An export plan holds every row. A preview plan is the same plan with long marks reduced, so
the two can differ only in those marks and each reduced mark records what it dropped.
Heatmaps and 3D surfaces have no plan yet.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from labviz_api.scientific_imports import scientific_module

if TYPE_CHECKING:
    import numpy as np
else:
    np = scientific_module("numpy")

import hashlib
import json
import math
from typing import Any, Literal

from .models import ChartSpec
from .render_contract import (
    CONFIDENCE_BAND_OPACITY,
    GEOMETRY,
    HISTOGRAM_AXIS_PADDING,
    HISTOGRAM_FILL_OPACITY,
    chart_series_color,
)
from .render_plan import (
    AXIS_NICE_RULE,
    CONTRACT,
    axis_ticks,
    grayscale,
    nice_axis,
    plan_hash,
    rasterize_points,
)

# Matplotlib's boxplot colours, kept so classic exports look as they did.
BOX_LINE_COLOR = "#000000"
BOX_MEDIAN_COLOR = "#ff7f0e"
BAR_WIDTH = 0.8
BOX_LINE_WIDTH = 1.0
SCATTER_OPACITY = 0.8
ERRORBAR_OPACITY = 0.9
MAX_POINTS = GEOMETRY["decimation"]["maxPoints"]
DECIMATION_RULE = GEOMETRY["decimation"]["rule"]
SERIES_KINDS = {"line", "scatter", "bar"}
REDUCED_FIELDS = {
    "polyline": ("x", "y"),
    "points": ("x", "y"),
    "bars": ("x0", "x1", "y0", "y1"),
}


def envelope_positions(
    values: np.ndarray[Any, Any], limit: int = MAX_POINTS
) -> np.ndarray[Any, Any]:
    """Keep every bucket's extremes so a decimated preview still shows real peaks.

    Evenly spaced positions drop narrow peaks and troughs that the full-data
    export still draws, which reads as an export defect rather than a preview
    limit. Selecting each bucket's minimum and maximum preserves the visible
    envelope. It never changes the analysed data or the reported sample size.
    """
    length = len(values)
    if length <= limit:
        return np.arange(length)
    buckets = max(1, (limit - 2) // 2)
    edges = np.linspace(0, length, num=buckets + 1, dtype=int)
    keep = {0, length - 1}
    for start, end in zip(edges[:-1], edges[1:], strict=True):
        window = np.asarray(values[start:end], dtype=float)
        if window.size == 0:
            continue
        if not np.isfinite(window).any():
            keep.add(int(start))
            continue
        masked = np.where(np.isfinite(window), window, np.nan)
        keep.add(int(start) + int(np.nanargmin(masked)))
        keep.add(int(start) + int(np.nanargmax(masked)))
    return np.array(sorted(keep), dtype=int)


def axis_label(title: str, unit: str) -> str:
    return f"{title} ({unit})" if unit else title


def reduced_mark(mark: dict[str, Any]) -> dict[str, Any]:
    """Reduce one long mark for the preview; stairs and fits are short and stay whole."""
    fields = REDUCED_FIELDS.get(mark.get("shape", mark["kind"]))
    if fields is None or len(mark[fields[-1]]) <= MAX_POINTS:
        return mark
    values = mark[fields[-1]]
    keep = envelope_positions(np.asarray(values, dtype=float)).tolist()
    return {
        **mark,
        **{field: [mark[field][index] for index in keep] for field in fields},
        "decimation": {"rule": DECIMATION_RULE, "input": len(values), "drawn": len(keep)},
    }


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
    # A label per row cannot be read and costs far more to draw than the row itself.
    stride = max(1, math.ceil(len(labels) / MAX_POINTS))
    ticks = list(enumerate(labels, first))[::stride]
    return {
        "scale": "category",
        "domain": [low, high],
        "ticks": [{"value": float(index), "label": label} for index, label in ticks],
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
                "axis": "y",
                "x0": [float(value["start"]) for value in bins],
                "x1": [float(value["end"]) for value in bins],
                "y0": [0.0] * len(bins),
                "y1": [float(value["count"]) for value in bins],
                "decimation": None,
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
            "y2": None,
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
            "y2": None,
        },
    }


def bar_domain(low: float, high: float) -> tuple[float, float]:
    """Bars grow from zero, so the axis starts there and only the far end gets a margin."""
    (start, end), _ = nice_axis(low, high)
    if low < high:
        start = 0.0 if low == 0 else start
        end = 0.0 if high == 0 else end
    return start, end


def is_finite(value: Any) -> bool:
    return isinstance(value, int | float) and math.isfinite(value)


def data_mark(
    chart: ChartSpec,
    item: Any,
    index: int,
    order: int,
    group: str,
    color: str,
    axis: str,
    xs: list[float],
    ys: list[float],
) -> dict[str, Any]:
    export = chart.export_settings
    base = {
        "id": f"{chart.type}:{index}",
        "group": group,
        "color": color,
        "axis": axis,
        "decimation": None,
    }
    if chart.type == "line":
        return {
            **base,
            "kind": "polyline",
            "width": export.line_width,
            "lineStyle": item.line_style,
            "markerSize": export.marker_size,
            "x": xs,
            "y": ys,
        }
    if chart.type == "scatter":
        return {
            **base,
            "kind": "points",
            "marker": "o",
            "size": export.marker_size,
            "opacity": SCATTER_OPACITY,
            "x": xs,
            "y": ys,
            "sourceRefs": [],
        }
    return {
        **base,
        "kind": "rects",
        "opacity": max(0.45, 0.85 - order * 0.08),
        "width": export.line_width,
        "shape": "bars",
        "x0": [x - BAR_WIDTH / 2 for x in xs],
        "x1": [x + BAR_WIDTH / 2 for x in xs],
        "y0": [0.0] * len(xs),
        "y1": ys,
    }


def errorbar_mark(
    chart: ChartSpec,
    index: int,
    group: str,
    color: str,
    axis: str,
    place: Any,
    uncertainty: dict[str, Any],
) -> dict[str, Any]:
    export = chart.export_settings
    points = [p for p in uncertainty["points"] if is_finite(p["y"]) and is_finite(p["error"])]
    return {
        "kind": "errorbar",
        "id": f"error:{index}",
        "group": group,
        "color": color,
        "width": export.line_width,
        "capSize": max(2, export.marker_size / 2),
        "opacity": ERRORBAR_OPACITY,
        "axis": axis,
        "x": [place(point["x"]) for point in points],
        "y": [float(point["y"]) for point in points],
        "error": [float(point["error"]) for point in points],
    }


def fit_label(chart: ChartSpec, label: str, fit: dict[str, Any]) -> str:
    text = f"{label} fit"
    if chart.fitting.show_equation:
        text += f" · {fit['equation']}"
    if chart.fitting.show_r_squared:
        text += f" · R²={fit['rSquared']:.4f}"
    return text


def fit_marks(
    chart: ChartSpec, index: int, label: str, color: str, axis: str, fit: dict[str, Any]
) -> list[dict[str, Any]]:
    points = [point for point in fit["points"] if is_finite(point["y"])]
    marks: list[dict[str, Any]] = [
        {
            "kind": "polyline",
            "id": f"fit:{index}",
            "group": fit_label(chart, label, fit),
            "color": color,
            "width": max(chart.export_settings.line_width, 1),
            "lineStyle": "dashed",
            "markerSize": 0.0,
            "axis": axis,
            "x": [float(point["x"]) for point in points],
            "y": [float(point["y"]) for point in points],
            "decimation": None,
        }
    ]
    band = [p for p in points if is_finite(p["lower"]) and is_finite(p["upper"])]
    if chart.fitting.confidence_band and band:
        marks.append(
            {
                "kind": "band",
                "id": f"band:{index}",
                "group": label,
                "color": color,
                "opacity": CONFIDENCE_BAND_OPACITY,
                "axis": axis,
                "x": [float(point["x"]) for point in band],
                "lower": [float(point["lower"]) for point in band],
                "upper": [float(point["upper"]) for point in band],
            }
        )
    return marks


def mark_extent(mark: dict[str, Any]) -> tuple[list[float], list[float]]:
    """The x and y values a mark reaches, which is what frames the axes."""
    if mark["kind"] == "rects":
        return [*mark["x0"], *mark["x1"]], [*mark["y0"], *mark["y1"]]
    if mark["kind"] == "band":
        return mark["x"], [*mark["lower"], *mark["upper"]]
    if mark["kind"] == "errorbar":
        pairs = zip(mark["y"], mark["error"], strict=True)
        return mark["x"], [value for y, error in pairs for value in (y - error, y + error)]
    return mark["x"], mark["y"]


def series_legend(mark: dict[str, Any]) -> dict[str, Any]:
    symbol = "marker"
    if mark["kind"] == "polyline":
        symbol = "dashed" if mark["id"].startswith("fit:") else "line"
    return {"label": mark["group"], "color": mark["color"], "symbol": symbol}


def series_panel(
    chart: ChartSpec,
    panel_index: int,
    analyses: list[dict[str, Any]],
    rows: list[list[dict[str, Any]]] | None,
    series_colors: dict[str, str] | None,
) -> dict[str, Any]:
    """Plan one line, scatter or bar panel with its fits, bands and error bars."""
    listed = [
        (
            index,
            analysis,
            [p for p in (rows[index] if rows else analysis["points"]) if p["x"] is not None],
        )
        for index, analysis in enumerate(analyses)
        if analysis["panel"] == panel_index
    ]
    numeric = all(isinstance(p["x"], int | float) for _, _, points in listed for p in points)
    categories: dict[str, float] = {}

    def place(x: Any) -> float:
        # Matplotlib places text categories in order of first appearance.
        return float(x) if numeric else categories.setdefault(str(x), float(len(categories)))

    secondary = chart.secondary_y_axis.enabled and any(
        item.panel == panel_index and item.y_axis == "secondary" for item in chart.series
    )
    marks: list[dict[str, Any]] = []
    for order, (index, analysis, points) in enumerate(listed):
        if not points:
            continue
        item = next(
            series
            for series in chart.series
            if series.field == analysis["field"] and series.panel == panel_index
        )
        label = analysis["label"]
        color = chart_series_color(
            configured_color=item.color,
            grayscale=chart.export_settings.grayscale_preview,
            grouped=bool(chart.group_field),
            index=index,
        )
        color = (series_colors or {}).get(label, color).lower()
        axis = "y2" if secondary and item.y_axis == "secondary" else "y"
        xs = [place(point["x"]) for point in points]
        ys = [float(point["y"]) for point in points]
        marks.append(data_mark(chart, item, index, order, label, color, axis, xs, ys))
        if analysis["uncertainty"]:
            marks.append(
                errorbar_mark(chart, index, label, color, axis, place, analysis["uncertainty"])
            )
        if analysis["fit"]:
            marks += fit_marks(chart, index, label, color, axis, analysis["fit"])
    x_values = [value for mark in marks for value in mark_extent(mark)[0]] or [0.0, 1.0]
    (low, high), _ = nice_axis(min(x_values), max(x_values))
    x_title = axis_label(chart.x_axis.title, chart.x_axis.unit)

    def y_axis(name: str, title: str) -> dict[str, Any]:
        values = [v for m in marks if m["axis"] == name for v in mark_extent(m)[1]] or [0.0, 1.0]
        if chart.type == "bar":
            return value_axis(*bar_domain(min(0.0, *values), max(0.0, *values)), title)
        (start, end), _ = nice_axis(min(values), max(values))
        return value_axis(start, end, title)

    return {
        "marks": marks,
        "legend": [
            series_legend(mark) for mark in marks if mark["kind"] in {"polyline", "points", "rects"}
        ],
        "axes": {
            "x": category_axis(list(categories), low, high, x_title, first=0)
            if categories
            else value_axis(low, high, x_title),
            "y": y_axis("y", axis_label(chart.y_axis.title, chart.y_axis.unit)),
            "y2": y_axis(
                "y2", axis_label(chart.secondary_y_axis.title, chart.secondary_y_axis.unit)
            )
            if secondary
            else None,
        },
    }


def build_classic_plan(
    chart: ChartSpec,
    derived: dict[str, Any],
    *,
    rows: list[list[dict[str, Any]]] | None = None,
    detail: Literal["export", "preview"] = "export",
    series_colors: dict[str, str] | None = None,
) -> dict[str, Any] | None:
    """Plan a classic chart, or return None for kinds not yet on plans.

    ``rows`` holds every row per analysis series. Without it the analysis's reduced points
    are all there is, which is the case for a capture whose frame lost its value types.
    """
    if chart.type not in {"histogram", "box", *SERIES_KINDS}:
        return None
    analyses = derived["series"]
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
            else series_panel(chart, panel_index, analyses, rows, series_colors)
        )
        for name in ("x", "y"):
            panel["axes"][name]["grid"] = grid
        if panel["axes"]["y2"] is not None:
            panel["axes"]["y2"]["grid"] = False
        title = f"{chart.title}\n{chart.subtitle}" if chart.subtitle else chart.title
        panels.append(
            {
                "id": str(panel_index),
                "title": title if chart.panel_count == 1 else f"Panel {panel_index}",
                **panel,
            }
        )
    drawn = sum(
        len(mark.get("y") or mark["y1"])
        for panel in panels
        for mark in panel["marks"]
        if mark["kind"] in {"polyline", "points"} or mark.get("shape") == "bars"
    )
    if detail == "preview":
        for panel in panels:
            panel["marks"] = [reduced_mark(mark) for mark in panel["marks"]]
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
            "decimation": DECIMATION_RULE if chart.type in SERIES_KINDS else None,
        },
        "rendering": {"pointRendering": "auto", "rasterizePoints": rasterize_points("auto", drawn)},
        "panels": panels,
        "disclosures": [],
    }
    plan = json.loads(json.dumps(plan, allow_nan=False))
    plan["planHash"] = plan_hash(plan)
    return plan
