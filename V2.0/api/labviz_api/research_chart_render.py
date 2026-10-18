"""One frozen Figure supplies SVG preview and PNG/SVG/PDF artifacts."""

from __future__ import annotations

import io
import threading
from typing import Any

from matplotlib import font_manager
from matplotlib.collections import LineCollection
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.text import Text

from .processing import _cjk_font_info, _contains_cjk
from .research_preparation import PreparationError
from .research_raster import freeze_raster_layers
from .research_styles import COLORS as COLORS

_RENDER_LOCK = threading.RLock()
DENSE_POINT_THRESHOLD = 20_000


def rasterize_points(mode: str, count: int) -> bool:
    return mode == "raster" or (mode == "auto" and count > DENSE_POINT_THRESHOLD)


def render_chart(
    result: dict[str, Any],
    *,
    target_figure: Any = None,
    target_axis: Any = None,
    group_colors: dict[Any, str] | None = None,
    grayscale: bool = False,
    point_rendering: str | None = None,
) -> dict[str, bytes]:
    with _RENDER_LOCK:
        spec = result["spec"]
        zh = spec["locale"] == "zh"
        dense = rasterize_points(
            point_rendering or spec.get("pointRendering", "auto"), len(result["points"])
        )
        names = {
            "grouped_strip": ("Grouped raw points", "分组原始点图"),
            "paired": ("Paired change", "配对变化图"),
            "violin": ("Violin distribution", "小提琴分布图"),
        }
        title = spec["title"] or names[spec["kind"]][int(zh)]
        figure = target_figure or Figure(figsize=(8, 5.5), dpi=300)
        ax = target_axis if target_axis is not None else figure.subplots()
        groups = result["groups"]
        colors = (
            dict(group_colors)
            if group_colors is not None
            else {group["label"]: COLORS[index % len(COLORS)] for index, group in enumerate(groups)}
        )
        styles = {item["group"]: item for item in spec.get("categoryStyles", [])}
        colors.update({key: item["color"] for key, item in styles.items()})
        if grayscale:
            from matplotlib.colors import to_hex, to_rgb

            for key, color in colors.items():
                gray = sum(
                    c * w for c, w in zip(to_rgb(color), (0.2126, 0.7152, 0.0722), strict=True)
                )
                colors[key] = to_hex((gray, gray, gray))
        if spec["kind"] == "paired":
            segments = [[[0, pair["first"]], [1, pair["second"]]] for pair in result["pairs"]]
            ax.add_collection(
                LineCollection(
                    segments,
                    colors=[colors[p["group"]] for p in result["pairs"]],
                    linewidths=1.2,
                    alpha=0.65,
                    rasterized=dense,
                )
            )
            ax.set_xticks([0, 1], spec["pairTimes"])
            ax.set_xlim(-0.35, 1.35)
            ax.legend(
                handles=[
                    Line2D(
                        [0],
                        [0],
                        color=colors[g["label"]],
                        label=g["label"] or ("未声明分组" if zh else "Undeclared"),
                    )
                    for g in groups
                ],
                fontsize=8,
                loc="best",
            )
        else:
            for density in result["densities"]:
                center = density["position"]
                widths = density["halfWidths"]
                ax.fill_betweenx(
                    density["grid"],
                    [center - w for w in widths],
                    [center + w for w in widths],
                    alpha=0.3,
                    color=colors[density["group"]],
                    linewidth=1,
                )
            labels = []
            for group in groups:
                label = (
                    group["label"]
                    if group["label"] is not None
                    else ("未声明分组" if zh else "Undeclared")
                )
                count = "?" if group["independentUnits"] is None else str(group["independentUnits"])
                labels.append(f"{label[:36]}\nobs={group['observations']}, units={count}")
                if group["summary"] is not None:
                    center = group["position"]
                    ax.plot(
                        [center - 0.25, center + 0.25],
                        [group["summary"]] * 2,
                        color="#303030" if grayscale else "#192d40",
                        linewidth=2.4,
                        zorder=5,
                    )
            ax.set_xticks(
                [g["position"] for g in groups], labels, rotation=30 if len(groups) > 4 else 0
            )
            ax.set_xlim(-0.65, max(len(groups) - 0.35, 0.65))
        if spec["showPoints"]:
            for group in groups:
                points = [
                    p for p in result["points"] if p["observation"]["group"] == group["label"]
                ]
                ax.plot(
                    [p["x"] for p in points],
                    [p["y"] for p in points],
                    linestyle="None",
                    markersize=4,
                    markeredgewidth=1,
                    marker=styles.get(group["label"], {}).get("marker", "o"),
                    color=colors[group["label"]],
                    alpha=0.7,
                    zorder=4,
                    rasterized=dense,
                )
        ax.set_title(title, pad=14, wrap=True)
        ax.set_ylabel(result["unit"])
        ax.grid(axis="y", alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)
        ax.autoscale_view(scalex=False)
        if target_axis is not None:
            return {}
        unit_count = len({p["observation"]["experimentalUnit"] for p in result["points"]})
        if any(p["observation"]["experimentalUnit"] is None for p in result["points"]):
            units = "?"
        else:
            units = str(unit_count)
        footer = (
            f"Observations={len(result['points'])}; independent units={units}; "
            f"excluded={len(result['excluded'])}"
        )
        if spec["kind"] == "violin":
            footer += "\nGaussian KDE of observations; density is not additional data."
        if spec["summary"] != "none":
            footer += (
                f"\n{spec['summary']} of {spec['summaryUnit']}; "
                "declared repeat means when grouping."
            )
        figure.text(0.1, 0.02, footer, fontsize=8)
        figure.subplots_adjust(left=0.12, right=0.96, top=0.86, bottom=0.24)
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
