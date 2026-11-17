"""Draw RenderPlan marks on a Matplotlib axis. Nothing here computes a statistic."""

from __future__ import annotations

from typing import Any

from .plan_axes import SYMLOG_THRESHOLD
from .render_contract import GEOMETRY, GRID_COLOR, LINE_STYLES

NOTE = GEOMETRY["annotation"]
LAYER_STEP: int = GEOMETRY["layers"]["step"]

# A mark with more bars than this is drawn as one collection, and the legend then takes a fixed
# place (`dense_bars`). A patch per bar, and a legend that tests every bar each time it looks for
# its best place, cost about 1 ms a bar: 100 s for a 100,000-row export.
BAR_PATCH_LIMIT = 1000


def layer_lift(marks: list[dict[str, Any]]) -> float:
    """How far the panel's top layer is lifted; notes go above it."""
    top: int = max((mark.get("layer") or 0 for mark in marks), default=0)
    return float(top * LAYER_STEP)


def draw_mark(ax: Any, mark: dict[str, Any], dense: bool) -> None:
    """Draw one mark; a mark of layer n sits n layer steps above every mark of lower layers."""
    layer = mark.get("layer")
    if not layer:
        draw_kind(ax, mark, dense)
        return
    before = set(ax.get_children())
    draw_kind(ax, mark, dense)
    for artist in ax.get_children():
        if artist not in before:
            artist.set_zorder(artist.get_zorder() + layer * LAYER_STEP)


def draw_kind(ax: Any, mark: dict[str, Any], dense: bool) -> None:
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
    elif mark["kind"] == "mesh":
        draw_mesh(ax, mark)
    else:
        draw_box(ax, mark)


def draw_mesh(ax: Any, mark: dict[str, Any]) -> None:
    """Heatmap cells as an image, or a surface triangulated through every grid point."""
    from matplotlib.colors import LinearSegmentedColormap

    colormap = LinearSegmentedColormap.from_list(f"labviz-{mark['shape']}", mark["palette"])
    low, high = mark["colorDomain"]
    if mark["shape"] == "cells":
        import numpy as np

        values = np.array(
            [[np.nan if value is None else value for value in row] for row in mark["values"]],
            dtype=float,
        )
        # Cells are one unit wide around their positions, as imshow places them by default.
        extent = (mark["x"][0] - 0.5, mark["x"][-1] + 0.5, mark["y"][-1] + 0.5, mark["y"][0] - 0.5)
        image = ax.imshow(values, cmap=colormap, vmin=low, vmax=high, extent=extent)
        ax.get_figure().colorbar(image, ax=ax, label=mark["colorTitle"])
        return
    points = [
        (x, y, value)
        for y, row in zip(mark["y"], mark["values"], strict=True)
        for x, value in zip(mark["x"], row, strict=True)
        if value is not None
    ]
    xs, ys, zs = (list(values) for values in zip(*points, strict=True))
    surface = ax.plot_trisurf(xs, ys, zs, cmap=colormap, vmin=low, vmax=high)
    ax.get_figure().colorbar(surface, ax=ax, shrink=0.65, pad=0.2, label=mark["colorTitle"])


def draw_rects(ax: Any, mark: dict[str, Any], dense: bool) -> None:
    style = {
        "alpha": mark["opacity"],
        "color": mark["color"],
        "label": mark["group"],
        "linewidth": mark["width"],
        "rasterized": dense,
    }
    if mark["shape"] == "stairs":
        # Bins rise from y0: 0 on a linear axis, the axis floor on a log count axis. A list
        # baseline writes other SVG and PDF bytes than 0, so linear exports pass the number.
        baseline = mark["y0"] if any(mark["y0"]) else 0
        ax.stairs(mark["y1"], [*mark["x0"], mark["x1"][-1]], baseline=baseline, fill=True, **style)
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


def draw_notes(ax: Any, notes: list[dict[str, Any]], lift: float = 0) -> None:
    """Annotations exactly as the plan states them; labels sit a fixed gap from their anchor.

    `lift` puts lines and text above a layered panel's top layer; shaded ranges stay under all.
    """
    from matplotlib.patches import Rectangle

    color, gap = NOTE["color"], NOTE["labelGap"]
    line = {"color": color, "linewidth": NOTE["lineWidth"], "zorder": 6 + lift}
    words = {"color": color, "fontsize": NOTE["fontSize"], "zorder": 7 + lift}

    def label(
        text: str, anchor: tuple[float, float], frame: Any, offset: Any, **align: str
    ) -> None:
        if text:
            ax.annotate(
                text,
                anchor,
                xycoords=frame,
                xytext=offset,
                textcoords="offset points",
                **align,
                **words,
            )

    for note in notes:
        kind = note["kind"]
        if kind == "text":
            ax.text(
                note["x"], note["y"], note["text"], ha="center", va="center", clip_on=True, **words
            )
        elif kind == "hline":
            ax.axhline(note["value"], linestyle=LINE_STYLES[NOTE["ruleStyle"]], **line)
            label(
                note["label"],
                (1.0, note["value"]),
                ("axes fraction", "data"),
                (-gap, gap),
                ha="right",
                va="bottom",
            )
        elif kind == "vline":
            ax.axvline(note["value"], linestyle=LINE_STYLES[NOTE["ruleStyle"]], **line)
            label(
                note["label"],
                (note["value"], 1.0),
                ("data", "axes fraction"),
                (gap, -gap),
                ha="left",
                va="top",
            )
        elif kind == "rect":
            (x0, x1), (y0, y1) = note["x"], note["y"]
            ax.add_patch(
                Rectangle(
                    (x0, y0),
                    x1 - x0,
                    y1 - y0,
                    facecolor=color,
                    edgecolor="none",
                    alpha=NOTE["rectOpacity"],
                    zorder=0.5,
                )
            )
            label(note["label"], tuple(note["labelAt"]), "data", (gap, -gap), ha="left", va="top")
        else:
            (x0, x1), y, foot = note["x"], note["y"], note["foot"]
            ax.plot([x0, x0, x1, x1], [foot, y, y, foot], **line)
            label(note["label"], tuple(note["labelAt"]), "data", (0, gap), ha="center", va="bottom")


def set_scale(ax: Any, name: str, scale: str) -> None:
    """Place values along a log or symlog axis exactly as plan_axes.axis_position does."""
    if scale not in ("log10", "symlog"):
        return
    import numpy as np

    def forward(values: Any) -> Any:
        data = np.asarray(values, dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            if scale == "log10":
                return np.log10(data)
            return np.sign(data) * np.log10(1 + np.abs(data) / SYMLOG_THRESHOLD)

    def inverse(positions: Any) -> Any:
        data = np.asarray(positions, dtype=float)
        if scale == "log10":
            return 10.0**data
        return np.sign(data) * (10.0 ** np.abs(data) - 1) * SYMLOG_THRESHOLD

    getattr(ax, f"set_{name}scale")("function", functions=(forward, inverse))


def draw_y_axis(ax: Any, axis: dict[str, Any]) -> None:
    set_scale(ax, "y", axis["scale"])
    ax.set_yticks([t["value"] for t in axis["ticks"]], [t["label"] for t in axis["ticks"]])
    ax.set_ylim(*axis["domain"])
    ax.set_ylabel(axis["title"])


def draw_panel(
    ax: Any, panel: dict[str, Any], *, secondary: Any = None, dense: bool = False
) -> None:
    """Draw one classic panel: its marks, then exactly the axes the plan states.

    A panel with a z axis needs a 3D Matplotlib axis.
    """
    for mark in panel["marks"]:
        draw_mark(secondary if mark.get("axis") == "y2" else ax, mark, dense)
    x_axis, y_axis, z_axis = panel["axes"]["x"], panel["axes"]["y"], panel["axes"]["z"]
    # Heatmap column names would overlap if they ran along the axis.
    cells = any(mark.get("shape") == "cells" for mark in panel["marks"])
    set_scale(ax, "x", x_axis["scale"])
    ax.set_xticks(
        [t["value"] for t in x_axis["ticks"]],
        [t["label"] for t in x_axis["ticks"]],
        **({"rotation": 45, "ha": "right"} if cells else {}),
    )
    ax.set_xlim(*x_axis["domain"])
    ax.set_xlabel(x_axis["title"])
    draw_y_axis(ax, y_axis)
    if secondary is not None:
        draw_y_axis(secondary, panel["axes"]["y2"])
    if z_axis is not None:
        ax.set_zticks([t["value"] for t in z_axis["ticks"]], [t["label"] for t in z_axis["ticks"]])
        ax.set_zlim(*z_axis["domain"])
        ax.set_zlabel(z_axis["title"])
        # A 3D axis has one grid switch for all three planes.
        ax.grid(x_axis["grid"])
        return
    for name, axis in (("x", x_axis), ("y", y_axis)):
        if axis["grid"]:
            ax.grid(True, axis=name, color=GRID_COLOR, linewidth=0.7, alpha=0.8)
        else:
            # Off must also override a grid-enabled Matplotlib theme.
            ax.grid(False, axis=name)
    draw_notes(ax, panel.get("notes", []), layer_lift(panel["marks"]))
