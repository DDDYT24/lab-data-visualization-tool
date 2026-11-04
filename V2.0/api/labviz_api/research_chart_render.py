"""One frozen Figure supplies SVG preview and PNG/SVG/PDF artifacts."""

from __future__ import annotations

import io
import threading
from typing import Any

from .processing import _cjk_font_info, _contains_cjk
from .render_plan import build_research_plan
from .render_plan import rasterize_points as rasterize_points
from .research_preparation import PreparationError
from .research_styles import COLORS as COLORS

_RENDER_LOCK = threading.RLock()
# Room left of the axis, in inches: the y title and pads, plus one tick label's digits at 10 pt.
LEFT_TITLE_INCHES = 0.4
LEFT_DIGIT_INCHES = 0.09
LEFT_MARGIN = 0.12


def render_chart(result: dict[str, Any]) -> dict[str, bytes]:
    return draw_plan(build_research_plan(result))


def _draw_mark(ax: Any, mark: dict[str, Any], dense: bool) -> None:
    from matplotlib.collections import LineCollection

    if mark["kind"] == "segments":
        ax.add_collection(
            LineCollection(
                [
                    [[x0, y0], [x1, y1]]
                    for x0, y0, x1, y1 in zip(
                        mark["x0"], mark["y0"], mark["x1"], mark["y1"], strict=True
                    )
                ],
                colors=mark["color"],
                linewidths=mark["width"],
                alpha=mark["opacity"],
                rasterized=dense,
            )
        )
    elif mark["kind"] == "violin":
        center = mark["position"]
        ax.fill_betweenx(
            mark["grid"],
            [center - w for w in mark["halfWidths"]],
            [center + w for w in mark["halfWidths"]],
            alpha=mark["opacity"],
            color=mark["color"],
            linewidth=1,
        )
    elif mark["kind"] == "points":
        ax.plot(
            mark["x"],
            mark["y"],
            linestyle="None",
            markersize=mark["size"],
            markeredgewidth=1,
            marker=mark["marker"],
            color=mark["color"],
            alpha=mark["opacity"],
            zorder=4,
            rasterized=dense,
        )
    else:
        ax.plot(
            [mark["x0"], mark["x1"]],
            [mark["y"]] * 2,
            color=mark["color"],
            linewidth=mark["width"],
            zorder=5,
        )


def draw_plan(
    plan: dict[str, Any], *, target_figure: Any = None, target_axis: Any = None
) -> dict[str, bytes]:
    from matplotlib import font_manager
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D
    from matplotlib.text import Text

    from .research_raster import freeze_raster_layers

    with _RENDER_LOCK:
        panel = plan["panels"][0]
        figure = target_figure or Figure(figsize=(8, 5.5), dpi=300)
        ax = target_axis if target_axis is not None else figure.subplots()
        for mark in panel["marks"]:
            _draw_mark(ax, mark, plan["rendering"]["rasterizePoints"])
        x_axis, y_axis = panel["axes"]["x"], panel["axes"]["y"]
        ax.set_xticks(
            [tick["value"] for tick in x_axis["ticks"]],
            [tick["label"] for tick in x_axis["ticks"]],
            rotation=30 if len(x_axis["ticks"]) > 4 else 0,
        )
        ax.set_xlim(*x_axis["domain"])
        if panel["legend"]:
            ax.legend(
                handles=[
                    Line2D([0], [0], color=entry["color"], label=entry["label"])
                    for entry in panel["legend"]
                ],
                fontsize=8,
                loc="best",
            )
        ax.set_title(panel["title"], pad=14, wrap=True)
        ax.set_ylabel(y_axis["title"])
        ax.grid(axis="y", alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_yticks(
            [tick["value"] for tick in y_axis["ticks"]],
            [tick["label"] for tick in y_axis["ticks"]],
        )
        ax.set_ylim(*y_axis["domain"])
        if target_axis is not None:
            return {}
        figure.text(0.1, 0.02, "\n".join(plan["disclosures"]), fontsize=8)
        widest = max(len(tick["label"]) for tick in y_axis["ticks"])
        left = (LEFT_TITLE_INCHES + widest * LEFT_DIGIT_INCHES) / figure.get_figwidth()
        figure.subplots_adjust(left=max(LEFT_MARGIN, left), right=0.96, top=0.86, bottom=0.24)
        texts = figure.findobj(Text)
        for text in texts:
            text.set_parse_math(False)
        if any(_contains_cjk(text.get_text()) for text in texts):
            info = _cjk_font_info()
            if info is None:
                raise PreparationError(
                    "chart-cjk-font", "A CJK font is required for Chinese labels."
                )
            for text in texts:
                text.set_fontproperties(
                    font_manager.FontProperties(fname=info[1], size=text.get_fontsize())
                )
        artifacts = {}
        for kind in ("png", "svg", "pdf"):
            if kind == "svg":
                # PNG retains its original direct Agg drawing; only vector containers
                # reuse the approved final-layout 300 dpi data layer.
                freeze_raster_layers(figure)
            output: Any = io.StringIO() if kind == "svg" else io.BytesIO()
            figure.savefig(
                output,
                format=kind,
                backend="module://labviz_api.research_svg_backend" if kind == "svg" else None,
                facecolor="white",
                metadata={"Creator": "LabViz scientific-charts-v1"} if kind != "png" else None,
            )
            artifacts[kind] = (
                output.getvalue().encode("utf-8") if kind == "svg" else output.getvalue()
            )
        figure.clear()
        return artifacts
