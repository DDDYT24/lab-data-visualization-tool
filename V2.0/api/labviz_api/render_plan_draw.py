"""Draw RenderPlan marks on a Matplotlib axis. Nothing here computes a statistic."""

from __future__ import annotations

from typing import Any


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
            zorder=4,
            rasterized=dense,
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
        draw_rects(ax, mark)
    else:
        draw_box(ax, mark)


def draw_rects(ax: Any, mark: dict[str, Any]) -> None:
    style = {
        "alpha": mark["opacity"],
        "color": mark["color"],
        "label": mark["group"],
        "linewidth": mark["width"],
    }
    if mark["shape"] == "stairs":
        ax.stairs(mark["y1"], [*mark["x0"], mark["x1"][-1]], fill=True, **style)
    else:
        ax.bar(
            mark["x0"],
            [top - base for base, top in zip(mark["y0"], mark["y1"], strict=True)],
            width=[end - start for start, end in zip(mark["x0"], mark["x1"], strict=True)],
            bottom=mark["y0"],
            align="edge",
            **style,
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


def draw_panel(ax: Any, panel: dict[str, Any]) -> None:
    """Draw one classic panel: its marks, then exactly the axes the plan states."""
    for mark in panel["marks"]:
        draw_mark(ax, mark, False)
    x_axis, y_axis = panel["axes"]["x"], panel["axes"]["y"]
    ax.set_xticks([t["value"] for t in x_axis["ticks"]], [t["label"] for t in x_axis["ticks"]])
    ax.set_yticks([t["value"] for t in y_axis["ticks"]], [t["label"] for t in y_axis["ticks"]])
    ax.set_xlim(*x_axis["domain"])
    ax.set_ylim(*y_axis["domain"])
    ax.set_xlabel(x_axis["title"])
    ax.set_ylabel(y_axis["title"])
