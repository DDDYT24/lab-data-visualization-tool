"""One Figure and frozen inputs for composed artifacts and selection coordinates."""

from __future__ import annotations

from typing import TYPE_CHECKING

from labviz_api.scientific_imports import scientific_module

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd
else:
    np = scientific_module("numpy")
    pd = scientific_module("pandas")

import io
import json
from time import perf_counter
from typing import Any, cast, get_args

from .figure_spec import (
    DEFAULT_FONT,
    FigureBoardSpec,
    FigurePanel,
    FigureSpec,
    as_figure,
    panel_cells,
    panel_heading,
)
from .models import ChartSpec
from .plan_axes import PLAIN_CONTROL, apply_axis_controls
from .plan_classic import SERIES_KINDS, build_classic_plan
from .plan_notes import apply_notes
from .processing import RENDER_LOCK, _cjk_font_info, _contains_cjk, embed_svg_font, series_rows
from .processing import render_chart as legacy_render
from .render_plan import build_research_plan, merge_axis, merge_layers, plan_bytes, plan_hash
from .research_chart_render import COLORS, VECTOR_TEXT, draw_plan, rasterize_points
from .research_charts import ScientificChartSpec
from .research_preparation import PreparationError


def frame_rows(panel: dict[str, Any], chart: ChartSpec, frame: Any) -> Any:
    """Every row of a captured panel, or None when its frozen frame cannot supply them.

    The frozen frame stores dates as numbers and floats to ten places. A panel whose frame no
    longer holds the x values it was captured with keeps its frozen reduced points.
    """
    if chart.type not in SERIES_KINDS:
        return None

    def key(x: Any) -> Any:
        return round(x, 9) if isinstance(x, float) else x

    rows = series_rows(frame, chart)
    analyses = panel["analysis"]["series"]
    matches = len(rows) == len(analyses) and all(
        {key(p["x"]) for p in analysis["points"]} <= {key(p["x"]) for p in series}
        for analysis, series in zip(analyses, rows, strict=False)
    )
    return rows if matches else None


# A v1 board is 7 by 5.5 inches a cell with these gaps; the v2 gutter scales the gaps.
CELL_INCHES = (7.0, 5.5)
PLAIN_GUTTER = 0.08
GAPS = (0.45, 0.9)
BOTTOM = {1: 0.26, 2: 0.13}
PALETTES = ("blue_teal", "grayscale")
# Matplotlib families per FigureSpec font. The generic name ends each list: Matplotlib falls back
# to it where the font is not installed, and SVG viewers do the same. "sans-serif" alone is what
# every text had before the choice existed (it resolves to DejaVu Sans).
FONT_FAMILIES = {
    "DejaVu Sans": ["sans-serif"],
    "Arial": ["Arial", "sans-serif"],
    "Times New Roman": ["Times New Roman", "serif"],
}
MILLIMETRES_PER_INCH = 25.4


def controlled(panel: FigurePanel, dimension: str) -> bool:
    control = getattr(panel.axes, dimension).model_dump(by_alias=True)
    return bool(control != PLAIN_CONTROL)


# Matplotlib refuses a side past 2**16 pixels, and an RGBA buffer of 200 million pixels is
# about 800 MB; a figure that would need more is refused before anything is stored or drawn.
MAX_FIGURE_SIDE_PIXELS = 32_768
MAX_FIGURE_PIXELS = 200_000_000


def figure_inches(spec: FigureSpec) -> tuple[float, float]:
    """The drawn figure's size: the custom canvas, or one fixed cell per layout cell."""
    canvas = spec.canvas
    if canvas.width is not None and canvas.height is not None:
        per_inch = MILLIMETRES_PER_INCH if canvas.unit == "mm" else 1.0
        return (canvas.width / per_inch, canvas.height / per_inch)
    return (CELL_INCHES[0] * spec.layout.cols, CELL_INCHES[1] * spec.layout.rows)


def check_figure(spec: FigureSpec) -> None:
    """Refusals the figure alone decides, run before a revision is stored or drawn."""
    width, height = (side * spec.canvas.dpi for side in figure_inches(spec))
    if max(width, height) > MAX_FIGURE_SIDE_PIXELS or width * height > MAX_FIGURE_PIXELS:
        raise PreparationError(
            "figure-canvas-too-large", "Choose a smaller canvas or a lower resolution."
        )
    if spec.palette.categorical not in PALETTES:
        raise PreparationError("figure-palette-unknown", "Choose a known figure palette.")
    for dimension in ("x", "y"):
        if spec.axis_mode in {"shared_" + dimension, "shared_xy"} and any(
            controlled(panel, dimension) for panel in spec.panels
        ):
            raise PreparationError(
                "figure-shared-axis-control",
                f"A shared {dimension} axis cannot also take a scale, limits or ticks.",
            )


RESEARCH_KINDS = frozenset(get_args(ScientificChartSpec.model_fields["kind"].annotation))


def check_layers(panels: list[list[dict[str, Any]]]) -> None:
    """Layers of one panel are research charts of one revision with the same axes.

    Each item is a chart's source or its cached header; both carry these keys.
    """
    for layers in panels:
        if len(layers) == 1:
            continue
        first = layers[0]
        for layer in layers:
            if layer["kind"] not in RESEARCH_KINDS:
                raise PreparationError(
                    "figure-layer-unplanned",
                    "A captured chart is drawn on its own; it cannot share a panel with layers.",
                )
            if layer["revisionId"] != first["revisionId"]:
                raise PreparationError(
                    "figure-layer-revision",
                    "Layers in one panel must show the same research revision.",
                )
            if (layer["xSignature"], layer["ySignature"]) != (
                first["xSignature"],
                first["ySignature"],
            ):
                raise PreparationError(
                    "figure-layer-axis",
                    "Layers in one panel need the same axis meanings, units and categories.",
                )


def adjusted(plan: dict[str, Any] | None, panel: FigurePanel) -> dict[str, Any] | None:
    """The panel's plan with its axis controls and annotations applied."""
    if not (controlled(panel, "x") or controlled(panel, "y") or panel.annotations):
        return plan
    if plan is None:
        raise PreparationError(
            "figure-panel-unplanned", "This panel cannot take axis controls or annotations."
        )
    plan = apply_axis_controls(plan, [panel.axes.model_dump(by_alias=True)])
    return apply_notes(plan, [[note.model_dump(by_alias=True) for note in panel.annotations]])


def render_board(
    sources: list[dict[str, Any]],
    board: FigureBoardSpec | FigureSpec,
    *,
    timings: dict[str, float] | None = None,
    layers: list[list[dict[str, Any]]] | None = None,
) -> tuple[dict[str, bytes], dict[str, Any]]:
    """Draw a board. `sources` stand for the panels (their first layer); `layers` holds every
    layer's source per panel, bottom first, and defaults to the panel's one chart."""
    from matplotlib import font_manager, rc_context
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.collections import PathCollection
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D
    from matplotlib.text import Text

    from .research_raster import freeze_raster_layers

    with RENDER_LOCK:
        started = perf_counter()
        spec = as_figure(board)
        count = len(sources)
        check_figure(spec)
        layered = layers or [[panel] for panel in sources]
        check_layers(layered)
        panel_labels = [panel.label for panel in spec.panels]
        rows, columns = spec.layout.rows, spec.layout.cols
        canvas = spec.canvas
        figure = Figure(figsize=figure_inches(spec), dpi=canvas.dpi)
        grid = figure.add_gridspec(rows, columns)
        axes = [
            figure.add_subplot(grid[row : row + tall, column : column + wide])
            for row, column, tall, wide in panel_cells(spec)
        ]
        use_grayscale = canvas.grayscale or spec.palette.categorical == "grayscale"
        palette = ["#20252d", "#5b616b", "#858b94", "#b0b4ba"] if use_grayscale else COLORS
        labels = sorted(
            {g["label"] for s in sources if "scientific" in s for g in s["scientific"]["groups"]},
            key=lambda x: (x is not None, x or ""),
        )
        colors = {label: palette[i % len(palette)] for i, label in enumerate(labels)}
        plans: list[dict[str, Any] | None] = [None] * count
        # Text x values sit at numeric positions, so hit targets need the same mapping.
        x_positions: list[dict[str, float]] = [{} for _ in range(count)]
        for i, (panel, ax) in enumerate(zip(sources, axes, strict=True)):
            if "scientific" in panel:
                planned = merge_layers(
                    [
                        build_research_plan(
                            layer["scientific"],
                            group_colors=colors,
                            use_grayscale=use_grayscale,
                            point_rendering=spec.point_rendering,
                            category_counts=spec.category_counts,
                        )
                        for layer in layered[i]
                    ]
                )
                drawn = adjusted(planned, spec.panels[i])
                assert drawn is not None
                plans[i] = drawn
                draw_plan(drawn, target_figure=figure, target_axis=ax)
            else:
                chart = ChartSpec.model_validate(panel["legacySpec"])
                chart.export_settings.font_size = spec.font_size
                chart.export_settings.grayscale_preview = use_grayscale
                for j, series in enumerate(chart.series):
                    series.color = palette[j % len(palette)]
                frame = pd.DataFrame(**panel["frame"])
                series_colors = None if use_grayscale else panel.get("seriesColors")
                planned_rows = frame_rows(panel, chart, frame)
                # The board stores the reduced plan; its exports draw every row.
                stored = build_classic_plan(
                    chart,
                    panel["analysis"],
                    rows=planned_rows,
                    detail="preview",
                    series_colors=series_colors,
                )
                full = build_classic_plan(
                    chart, panel["analysis"], rows=planned_rows, series_colors=series_colors
                )
                if stored is not None and full is not None:
                    stored["rendering"] = full["rendering"] = {
                        "pointRendering": spec.point_rendering,
                        "rasterizePoints": rasterize_points(
                            spec.point_rendering, len(panel["rows"])
                        ),
                    }
                stored, full = adjusted(stored, spec.panels[i]), adjusted(full, spec.panels[i])
                plans[i] = stored
                if stored is not None and stored["panels"][0]["axes"]["x"]["scale"] == "category":
                    points = planned_rows or [a["points"] for a in panel["analysis"]["series"]]
                    names = [str(p["x"]) for row in points for p in row if p["x"] is not None]
                    x_positions[i] = {x: float(n) for n, x in enumerate(dict.fromkeys(names))}
                legacy_render(
                    frame,
                    chart,
                    _target_figure=figure,
                    _target_axes=[ax],
                    _frozen_analysis=panel["analysis"],
                    _series_colors=series_colors,
                    _plan=full,
                )
                # Dense legacy line/scatter data artists share the same 300 dpi policy.
                if panel["kind"] in {"line", "scatter"} and rasterize_points(
                    spec.point_rendering, len(panel["rows"])
                ):
                    for artist in ax.get_children():
                        if isinstance(artist, (Line2D, PathCollection)):
                            artist.set_rasterized(True)
            title = spec.panels[i].title or panel["title"] or panel["kind"]
            heading = panel_heading(spec, i, title)
            ax.set_title(heading, fontsize=spec.font_size + 1)
            ax.tick_params(labelsize=spec.font_size)
            ax.xaxis.label.set_fontsize(spec.font_size)
            ax.yaxis.label.set_fontsize(spec.font_size)
            if spec.grid_visible:
                ax.grid(True, alpha=0.2)
            else:
                ax.grid(False)
            footer = f"Input rows={len(panel['rows'])}; revision={panel['revisionId'][:16]}"
            if "scientific" in panel:
                data = panel["scientific"]
                units = {r.get("experimentalUnit") for r in panel["rows"]}
                count_text = "?" if None in units else str(len(units))
                footer += f"\nIndependent units={count_text}; excluded={len(data['excluded'])}"
                # Every layer's method is disclosed, the panel's own chart first.
                methods: list[str] = []
                for layer in [panel, *layered[i]]:
                    chart = layer["scientific"]["spec"]
                    if chart["kind"] == "violin":
                        methods.append("Gaussian KDE of observations; density is not extra data.")
                    if chart["summary"] != "none":
                        methods.append(
                            f"{chart['summary']} of {chart['summaryUnit']}; declared repeat means."
                        )
                footer += "".join("\n" + line for line in dict.fromkeys(methods))
            else:
                legacy_spec = panel["legacySpec"]
                if legacy_spec["fitting"]["model"] != "none":
                    method = legacy_spec["fitting"]
                    footer += (
                        f"\nFit: {method['model']}; {method['fitMethod']}; {method['intervalKind']}"
                    )
                    footer += (
                        "\nAssumes representative complete observations; "
                        "no multiplicity correction."
                    )
                if legacy_spec["uncertainty"]["mode"] != "none":
                    footer += f"\nUncertainty: {legacy_spec['uncertainty']['mode']}"
            plan = plans[i]
            if plan is not None:
                plan["panels"][0] |= {"id": panel_labels[i], "title": heading}
                plan["disclosures"] = footer.split("\n")
                for name in ("x", "y"):
                    plan["panels"][0]["axes"][name]["grid"] = spec.grid_visible
            legend = ax.get_legend()
            if legend:
                for text in legend.get_texts():
                    text.set_fontsize(spec.font_size)
            if spec.provenance_on_figure:
                ax.text(
                    0,
                    -0.25,
                    footer,
                    va="top",
                    transform=ax.transAxes,
                    fontsize=min(8, max(6, spec.font_size - 2)),
                )
        for dimension in ["x", "y"]:
            if spec.axis_mode not in {"shared_" + dimension, "shared_xy"}:
                continue
            signatures = [json.dumps(s[dimension + "Signature"], sort_keys=True) for s in sources]
            if len(set(signatures)) != 1:
                raise PreparationError(
                    "figure-incompatible-axis",
                    f"Shared {dimension} axes require identical meanings, units and categories.",
                )
            getter = "get_" + dimension + "lim"
            setter = "set_" + dimension + "lim"
            limits = [getattr(ax, getter)() for ax in axes]
            # Heatmaps use reversed y limits; preserve orientation as well as the domain.
            low, high = min(min(v) for v in limits), max(max(v) for v in limits)
            reverse = limits[0][0] > limits[0][1]
            for ax, plan in zip(axes, plans, strict=True):
                if plan is not None:
                    axis = merge_axis(plan, dimension, low, high)
                    if reverse:
                        axis["domain"] = [high, low]
                    if axis["scale"] == "linear":
                        getattr(ax, f"set_{dimension}ticks")(
                            [tick["value"] for tick in axis["ticks"]],
                            [tick["label"] for tick in axis["ticks"]],
                        )
                getattr(ax, setter)((high, low) if reverse else (low, high))
        figure.suptitle(spec.title, fontsize=spec.font_size + 3)
        figure.subplots_adjust(
            left=0.10,
            right=0.94,
            top=0.88,
            bottom=BOTTOM.get(rows, 0.09),
            wspace=GAPS[0] * (spec.layout.gutter / PLAIN_GUTTER),
            hspace=GAPS[1] * (spec.layout.gutter / PLAIN_GUTTER),
        )
        texts = figure.findobj(Text)
        info = _cjk_font_info() if any(_contains_cjk(t.get_text()) for t in texts) else None
        if any(_contains_cjk(t.get_text()) for t in texts) and info is None:
            raise PreparationError("chart-cjk-font", "A CJK font is required for Chinese labels.")
        *chosen, generic = FONT_FAMILIES[spec.font_family]
        for text in texts:
            text.set_parse_math(False)
            if info and spec.font_family == DEFAULT_FONT:
                # The bundled CJK file draws every text, as before the choice existed; the family
                # name lets SVG text find the glyphs embed_svg_font adds.
                text.set_fontproperties(
                    font_manager.FontProperties(
                        family=[info[0], generic], fname=info[1], size=text.get_fontsize()
                    )
                )
            else:
                # A chosen font draws what it can; Chinese glyphs fall back to the CJK font.
                text.set_fontfamily([*chosen, *([info[0]] if info else []), generic])
        # Transform coordinates after axis sharing/layout, using the same Figure as the exports.
        FigureCanvasAgg(figure)
        # Resolve transforms/text geometry without painting the full dense data twice.
        figure.draw_without_rendering()
        artifacts: dict[str, bytes] = {}
        for label, plan in zip(panel_labels, plans, strict=True):
            if plan is not None:
                plan["planHash"] = plan_hash(plan)
                artifacts["plan:" + label] = plan_bytes(plan)
        scene: dict[str, Any] = {
            "schemaVersion": 3,
            "panels": [],
            "marks": [],
            "rowSets": {},
            "exportRendering": {
                "requested": spec.point_rendering,
                "thresholdObservations": 20_000,
                "imageDpi": 300,
                "rasterPanels": [
                    panel_labels[i]
                    for i, panel in enumerate(sources)
                    if rasterize_points(spec.point_rendering, len(panel["rows"]))
                    and any(
                        layer["kind"] in {"line", "scatter", "paired"}
                        or ("scientific" in layer and layer["scientific"]["spec"]["showPoints"])
                        for layer in layered[i]
                    )
                ],
                "fullSourceDataRetained": True,
            },
            "panelRowSets": {},
            "sampled": False,
        }
        for i, (panel, ax) in enumerate(zip(sources, axes, strict=True)):
            label = panel_labels[i]
            scene["panels"].append(
                {
                    "label": label,
                    "xLimits": list(ax.get_xlim()),
                    "yLimits": list(ax.get_ylim()),
                    "xSignature": panel["xSignature"],
                    "ySignature": panel["ySignature"],
                    "planHash": (plans[i] or {}).get("planHash"),
                    "observations": len(panel["rows"]),
                }
            )
            row_key = (
                "research:" + panel["revisionId"]
                if "scientific" in panel
                else "legacy:" + panel["resultSha256"]
            )
            scene["rowSets"].setdefault(row_key, panel["rows"])
            scene["panelRowSets"][label] = row_key
            raw = panel["marks"]
            # Sample only point hit targets; retain every aggregate and all input/export values.
            point_indices = [j for j, m in enumerate(raw) if m["kind"] == "point"]
            stride = max(1, (len(point_indices) + 499) // 500)
            shown = set(point_indices[::stride])
            scene["sampled"] |= stride > 1 or panel["sampled"]
            for j, mark in enumerate(raw):
                if mark["kind"] == "point" and j not in shown:
                    continue
                x = x_positions[i].get(str(mark["x"]), mark["x"])
                x = cast(Any, ax).convert_xunits(x)
                y = cast(Any, ax).convert_yunits(mark["y"])
                pixel = ax.transData.transform(
                    (float(np.asarray(x).item()), float(np.asarray(y).item()))
                )
                normalized = figure.transFigure.inverted().transform(pixel)
                scene["marks"].append(
                    {
                        **mark,
                        "id": f"{label}:{mark.get('sourceIndex', j)}",
                        "panel": label,
                        "cx": float(normalized[0]),
                        "cy": float(1 - normalized[1]),
                    }
                )
        if timings is not None:
            timings["geometry"] = (perf_counter() - started) * 1000
        with rc_context(VECTOR_TEXT):
            for kind in ["png", "svg", "pdf"]:
                format_started = perf_counter()
                if kind == "svg":
                    # PNG retains its original direct Agg drawing; only vector containers
                    # reuse the approved final-layout 300 dpi data layer.
                    freeze_raster_layers(figure)
                    if timings is not None:
                        timings["rasterPreparation"] = (perf_counter() - format_started) * 1000
                    format_started = perf_counter()
                stream: Any = io.StringIO() if kind == "svg" else io.BytesIO()
                figure.savefig(
                    stream,
                    format=kind,
                    backend="module://labviz_api.research_svg_backend" if kind == "svg" else None,
                    facecolor=canvas.background,
                    transparent=canvas.transparent,
                )
                artifacts[kind] = (
                    stream.getvalue().encode("utf-8") if kind == "svg" else stream.getvalue()
                )
                if timings is not None:
                    timings[kind] = (perf_counter() - format_started) * 1000
        if info is not None:
            artifacts["svg"] = embed_svg_font(artifacts["svg"], figure, info[1], info[0])
        figure.clear()
        return artifacts, scene
