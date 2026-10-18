"""One Figure and frozen inputs for composed artifacts and selection coordinates."""

from __future__ import annotations

import io
import json
from time import perf_counter
from typing import Any, cast

import numpy as np
import pandas as pd
from matplotlib import font_manager
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.collections import PathCollection
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.text import Text

from .models import ChartSpec
from .processing import _cjk_font_info, _contains_cjk
from .processing import render_chart as legacy_render
from .research_chart_render import _RENDER_LOCK, COLORS, rasterize_points, render_chart
from .research_preparation import PreparationError
from .research_raster import freeze_raster_layers


def render_board(
    sources: list[dict[str, Any]], spec: Any, *, timings: dict[str, float] | None = None
) -> tuple[dict[str, bytes], dict[str, Any]]:
    with _RENDER_LOCK:
        started = perf_counter()
        count = len(sources)
        columns = 1 if count == 1 else 2
        rows = 2 if count == 4 else 1
        figure = Figure(figsize=(7 * columns, 5.5 * rows), dpi=300)
        axes = [figure.add_subplot(rows, columns, i + 1) for i in range(count)]
        palette = (
            COLORS if spec.palette == "blue_teal" else ["#20252d", "#5b616b", "#858b94", "#b0b4ba"]
        )
        labels = sorted(
            {g["label"] for s in sources if "scientific" in s for g in s["scientific"]["groups"]},
            key=lambda x: (x is not None, x or ""),
        )
        colors = {label: palette[i % len(palette)] for i, label in enumerate(labels)}
        for i, (panel, ax) in enumerate(zip(sources, axes, strict=True)):
            if "scientific" in panel:
                render_chart(
                    panel["scientific"],
                    target_figure=figure,
                    target_axis=ax,
                    group_colors=colors,
                    grayscale=spec.palette == "grayscale",
                    point_rendering=spec.point_rendering,
                )
            else:
                chart = ChartSpec.model_validate(panel["legacySpec"])
                chart.export_settings.font_size = spec.font_size
                chart.export_settings.grayscale_preview = spec.palette == "grayscale"
                for j, series in enumerate(chart.series):
                    series.color = palette[j % len(palette)]
                frame = pd.DataFrame(**panel["frame"])
                legacy_render(
                    frame,
                    chart,
                    _target_figure=figure,
                    _target_axes=[ax],
                    _frozen_analysis=panel["analysis"],
                    _series_colors=None
                    if spec.palette == "grayscale"
                    else panel.get("seriesColors"),
                )
                # Dense legacy line/scatter data artists share the same 300 dpi policy.
                if panel["kind"] in {"line", "scatter"} and rasterize_points(
                    spec.point_rendering, len(panel["rows"])
                ):
                    for artist in ax.get_children():
                        if isinstance(artist, (Line2D, PathCollection)):
                            artist.set_rasterized(True)
            title = (
                (spec.panel_titles[i] if spec.panel_titles else "")
                or panel["title"]
                or panel["kind"]
            )
            ax.set_title(f"{'ABCD'[i]} · {title}", fontsize=spec.font_size + 1)
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
                if data["spec"]["kind"] == "violin":
                    footer += "\nGaussian KDE of observations; density is not extra data."
                if data["spec"]["summary"] != "none":
                    footer += (
                        f"\n{data['spec']['summary']} of {data['spec']['summaryUnit']}; "
                        "declared repeat means."
                    )
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
            legend = ax.get_legend()
            if legend:
                for text in legend.get_texts():
                    text.set_fontsize(spec.font_size)
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
            for ax in axes:
                getattr(ax, setter)((high, low) if reverse else (low, high))
        figure.suptitle(spec.title, fontsize=spec.font_size + 3)
        figure.subplots_adjust(
            left=0.10,
            right=0.94,
            top=0.88,
            bottom=0.26 if rows == 1 else 0.13,
            wspace=0.45,
            hspace=0.9,
        )
        texts = figure.findobj(Text)
        info = _cjk_font_info() if any(_contains_cjk(t.get_text()) for t in texts) else None
        if any(_contains_cjk(t.get_text()) for t in texts) and info is None:
            raise PreparationError("chart-cjk-font", "A CJK font is required for Chinese labels.")
        for text in texts:
            text.set_parse_math(False)
            if info:
                text.set_fontproperties(
                    font_manager.FontProperties(fname=info[1], size=text.get_fontsize())
                )
        # Transform coordinates after axis sharing/layout, using the same Figure as the exports.
        FigureCanvasAgg(figure)
        # Resolve transforms/text geometry without painting the full dense data twice.
        figure.draw_without_rendering()
        scene: dict[str, Any] = {
            "schemaVersion": 2,
            "panels": [],
            "marks": [],
            "rowSets": {},
            "exportRendering": {
                "requested": spec.point_rendering,
                "thresholdObservations": 20_000,
                "imageDpi": 300,
                "rasterPanels": [
                    "ABCD"[i]
                    for i, panel in enumerate(sources)
                    if rasterize_points(spec.point_rendering, len(panel["rows"]))
                    and (
                        panel["kind"] in {"line", "scatter", "paired"}
                        or ("scientific" in panel and panel["scientific"]["spec"]["showPoints"])
                    )
                ],
                "fullSourceDataRetained": True,
            },
            "panelRowSets": {},
            "sampled": False,
        }
        for i, (panel, ax) in enumerate(zip(sources, axes, strict=True)):
            label = "ABCD"[i]
            scene["panels"].append(
                {
                    "label": label,
                    "xLimits": list(ax.get_xlim()),
                    "yLimits": list(ax.get_ylim()),
                    "xSignature": panel["xSignature"],
                    "ySignature": panel["ySignature"],
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
                x = cast(Any, ax).convert_xunits(mark["x"])
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
        artifacts = {}
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
                facecolor="white",
            )
            artifacts[kind] = (
                stream.getvalue().encode("utf-8") if kind == "svg" else stream.getvalue()
            )
            if timings is not None:
                timings[kind] = (perf_counter() - format_started) * 1000
        figure.clear()
        return artifacts, scene
