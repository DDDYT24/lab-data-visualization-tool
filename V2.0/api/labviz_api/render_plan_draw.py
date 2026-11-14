"""Draw RenderPlan marks on a Matplotlib axis. Nothing here computes a statistic."""

from __future__ import annotations

from typing import Any

from .render_contract import GRID_COLOR, LINE_STYLES

# A mark with more bars than this is drawn as one collection, and the legend then takes a fixed
# place (`dense_bars`). A patch per bar, and a legend that tests every bar each time it looks for
# its best place, cost about 1 ms a bar: 100 s for a 100,000-row export.
BAR_PATCH_LIMIT = 1000


def draw_mark(ax: Any, mark: dict[str, Any], dense: bool) -> None:
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
            label=mark["group"],
            zorder=4,
            rasterized=dense,
        )
    elif mark["kind"] == "polyline":
        ax.plot(
            mark["x"],
            mark["y"],
            label=mark["group"],
            color=mark["color"],
            linestyle=LINE_STYLES[mark["lineStyle"]],
            linewidth=mark["width"],
            marker="o" if mark["markerSize"] else None,
            markersize=mark["markerSize"],
            rasterized=dense,
        )
    elif mark["kind"] == "band":
        ax.fill_between(
            mark["x"], mark["lower"], mark["upper"], color=mark["color"], alpha=mark["opacity"]
        )
    elif mark["kind"] == "errorbar":
        ax.errorbar(
            mark["x"],
            mark["y"],
            yerr=mark["error"],
            fmt="none",
            ecolor=mark["color"],
            elinewidth=mark["width"],
            capsize=mark["capSize"],
            alpha=mark["opacity"],
        )
    elif mark["kind"] == "summary":
        ax.plot(
            [mark["x0"], mark["x1"]],
            [mark["y"]] * 2,
            color=mark["color"],
            linewidth=mark["width"],
            zorder=5,
        )
    elif mark["kind"] == "rects":
        draw_rects(ax, mark, dense)
    else:
        draw_box(ax, mark)


def draw_rects(ax: Any, mark: dict[str, Any], dense: bool) -> None:
    style = {
        "alpha": mark["opacity"],
        "color": mark["color"],
        "label": mark["group"],
        "linewidth": mark["width"],
        "rasterized": dense,
    }
    if mark["shape"] == "stairs":
        ax.stairs(mark["y1"], [*mark["x0"], mark["x1"][-1]], fill=True, **style)
    elif len(mark["x0"]) > BAR_PATCH_LIMIT:
        import numpy as np
        from matplotlib.collections import PolyCollection

        corners = [
            np.asarray(mark[key], dtype=float)
            for key in ("x0", "y0", "x1", "y0", "x1", "y1", "x0", "y1")
        ]
        # One array of all bars: the stubs ask for a sequence, the class takes either.
        bars: Any = np.stack(corners, axis=1).reshape(-1, 4, 2)
        # ax.bar gives a coloured bar no edge, so none here keeps the pixels the same.
        ax.add_collection(
            PolyCollection(
                bars,
                alpha=mark["opacity"],
                facecolor=mark["color"],
                edgecolor="none",
                label=mark["group"],
                rasterized=dense,
            ),
            autolim=False,
        )
    else:
        ax.bar(
            mark["x0"],
            [top - base for base, top in zip(mark["y0"], mark["y1"], strict=True)],
            width=[end - start for start, end in zip(mark["x0"], mark["x1"], strict=True)],
            bottom=mark["y0"],
            align="edge",
            **style,
        )


def dense_bars(ax: Any) -> bool:
    """Whether the axis holds a bar mark that was drawn as one collection."""
    from matplotlib.collections import PolyCollection

    return any(
        isinstance(collection, PolyCollection) and len(collection.get_paths()) > BAR_PATCH_LIMIT
        for collection in ax.collections
    )


def draw_box(ax: Any, mark: dict[str, Any]) -> None:
    line = {"color": mark["color"], "linewidth": mark["width"]}
    ax.bxp(
        [
            {
                "label": mark["group"],
                "whislo": mark["low"],
                "q1": mark["q1"],
                "med": mark["median"],
                "q3": mark["q3"],
                "whishi": mark["high"],
                "fliers": mark["outliers"],
            }
        ],
        positions=[mark["position"]],
        widths=[mark["halfWidth"] * 2],
        manage_ticks=False,
        boxprops=line,
        whiskerprops=line,
        capprops=line,
        medianprops={"color": mark["medianColor"], "linewidth": mark["width"]},
        flierprops={"markeredgecolor": mark["color"]},
    )


def draw_y_axis(ax: Any, axis: dict[str, Any]) -> None:
    ax.set_yticks([t["value"] for t in axis["ticks"]], [t["label"] for t in axis["ticks"]])
    ax.set_ylim(*axis["domain"])
    ax.set_ylabel(axis["title"])


def draw_panel(
    ax: Any, panel: dict[str, Any], *, secondary: Any = None, dense: bool = False
) -> None:
    """Draw one classic panel: its marks, then exactly the axes the plan states."""
    for mark in panel["marks"]:
        draw_mark(secondary if mark.get("axis") == "y2" else ax, mark, dense)
    x_axis, y_axis = panel["axes"]["x"], panel["axes"]["y"]
    ax.set_xticks([t["value"] for t in x_axis["ticks"]], [t["label"] for t in x_axis["ticks"]])
    ax.set_xlim(*x_axis["domain"])
    ax.set_xlabel(x_axis["title"])
    draw_y_axis(ax, y_axis)
    if secondary is not None:
        draw_y_axis(secondary, panel["axes"]["y2"])
    for name, axis in (("x", x_axis), ("y", y_axis)):
        if axis["grid"]:
            ax.grid(True, axis=name, color=GRID_COLOR, linewidth=0.7, alpha=0.8)
        else:
            # Off must also override a grid-enabled Matplotlib theme.
            ax.grid(False, axis=name)
