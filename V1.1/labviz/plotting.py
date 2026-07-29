"""Matplotlib-based scientific plotting with bounded rendering cost."""

from __future__ import annotations

import io
from collections.abc import Sequence
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from .core import validate_columns_exist

matplotlib.use("Agg")

SUPPORTED_PLOTS = ("line", "scatter", "bar", "histogram", "box", "heatmap", "surface3d")
MAX_PLOT_POINTS = 20_000


def sample_dataframe(frame: pd.DataFrame, max_points: int = MAX_PLOT_POINTS) -> pd.DataFrame:
    """Return an evenly spaced, deterministic sample while preserving row order."""

    if max_points < 1:
        raise ValueError("max_points must be at least 1.")
    if len(frame) <= max_points:
        return frame
    positions: np.ndarray = np.linspace(0, len(frame) - 1, num=max_points, dtype=int)
    return frame.iloc[positions]


def _require_numeric(frame: pd.DataFrame, columns: Sequence[str]) -> None:
    non_numeric = [column for column in columns if not pd.api.types.is_numeric_dtype(frame[column])]
    if non_numeric:
        raise ValueError(f"Numeric columns required: {non_numeric}")


def create_figure(
    frame: pd.DataFrame,
    *,
    kind: str,
    x: str | None,
    ys: Sequence[str],
    title: str = "",
    max_points: int = MAX_PLOT_POINTS,
) -> Figure:
    """Create a validated figure for a tabular dataset."""

    kind = kind.lower()
    if kind not in SUPPORTED_PLOTS:
        raise ValueError(f"kind must be one of: {', '.join(SUPPORTED_PLOTS)}")
    if frame.empty:
        raise ValueError("There are no rows to plot.")
    if not ys:
        raise ValueError("Select at least one value column.")

    requested = [*ys]
    if x is not None:
        requested.insert(0, x)
    validate_columns_exist(frame, requested)

    limit = min(max_points, 500) if kind == "bar" else max_points
    plotted = sample_dataframe(frame, limit)

    if kind == "surface3d":
        if x is None or len(ys) != 2:
            raise ValueError("3D surface requires X, Y, and Z columns.")
        _require_numeric(plotted, [x, *ys])
        complete = plotted[[x, *ys]].dropna()
        if len(complete) < 3:
            raise ValueError("3D surface requires at least three complete points.")
        figure = Figure(figsize=(8, 5))
        axis = figure.add_subplot(111, projection="3d")
        try:
            surface = axis.plot_trisurf(
                complete[x], complete[ys[0]], complete[ys[1]], cmap="viridis", linewidth=0.2
            )
        except (RuntimeError, ValueError) as exc:
            raise ValueError("3D points must span a surface rather than a single line.") from exc
        axis.set_xlabel(x)
        axis.set_ylabel(ys[0])
        axis.set_zlabel(ys[1])
        figure.colorbar(surface, ax=axis, shrink=0.65, pad=0.1, label=ys[1])
    else:
        figure = Figure(figsize=(8, 5))
        axis = figure.subplots()

        if kind in {"line", "scatter", "bar"}:
            if x is None:
                raise ValueError(f"{kind} plot requires an X column.")
            _require_numeric(plotted, ys)
            complete = plotted[[x, *ys]].dropna()
            if complete.empty:
                raise ValueError("The selected columns have no complete rows to plot.")
            for y_column in ys:
                if kind == "line":
                    axis.plot(complete[x], complete[y_column], label=y_column)
                elif kind == "scatter":
                    axis.scatter(complete[x], complete[y_column], label=y_column, alpha=0.75)
                else:
                    axis.bar(complete[x], complete[y_column], label=y_column, alpha=0.8)
            axis.set_xlabel(x)
            axis.set_ylabel(", ".join(ys))
            axis.legend()
        elif kind == "histogram":
            _require_numeric(plotted, ys)
            if all(plotted[column].dropna().empty for column in ys):
                raise ValueError("The selected columns have no numeric values to plot.")
            for y_column in ys:
                axis.hist(plotted[y_column].dropna(), bins="auto", alpha=0.6, label=y_column)
            axis.set_xlabel("Value")
            axis.set_ylabel("Frequency")
            axis.legend()
        elif kind == "box":
            _require_numeric(plotted, ys)
            values = [plotted[column].dropna().to_numpy() for column in ys]
            if any(len(value) == 0 for value in values):
                raise ValueError("Every selected box-plot column needs at least one value.")
            axis.boxplot(values, tick_labels=list(ys), showmeans=True)
            axis.set_ylabel("Value")
        else:
            _require_numeric(plotted, ys)
            if len(ys) < 2:
                raise ValueError("Correlation heatmap requires at least two numeric columns.")
            correlations = plotted[list(ys)].corr()
            image = axis.imshow(correlations, cmap="coolwarm", vmin=-1, vmax=1)
            ticks = np.arange(len(ys))
            axis.set_xticks(ticks, ys, rotation=45, ha="right")
            axis.set_yticks(ticks, ys)
            figure.colorbar(image, ax=axis, label="Correlation")
            if len(ys) <= 10:
                for row in range(len(ys)):
                    for column in range(len(ys)):
                        value = correlations.iloc[row, column]
                        axis.text(column, row, f"{value:.2f}", ha="center", va="center")

        if kind not in {"heatmap", "box"}:
            axis.grid(True, alpha=0.25)
        else:
            axis.grid(False)

    if title:
        axis.set_title(title)
    figure.tight_layout()
    return figure


def figure_to_bytes(figure: Figure, *, dpi: int = 160) -> bytes:
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", dpi=dpi, bbox_inches="tight")
    return buffer.getvalue()


def save_figure(figure: Figure, output: Path, *, dpi: int = 160) -> Path:
    target = output if output.suffix.lower() == ".png" else output.with_suffix(".png")
    target.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(target, dpi=dpi, bbox_inches="tight")
    return target
