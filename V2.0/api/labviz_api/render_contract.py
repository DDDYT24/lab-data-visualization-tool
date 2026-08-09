"""Shared semantic styling contract for browser previews and publication exports."""

from __future__ import annotations

from typing import Literal

CHART_RENDER_CONTRACT_VERSION = "chart-render-v1"
GRAYSCALE_PALETTE = ("#20252D", "#5B616B", "#858B94", "#B0B4BA")
GROUP_PALETTE = (
    "#2563EB",
    "#0F766E",
    "#D97706",
    "#7C3AED",
    "#DC2626",
    "#0891B2",
    "#4D7C0F",
    "#C2410C",
    "#4338CA",
    "#BE185D",
    "#0369A1",
    "#3F6212",
)
GRID_COLOR = "#D8DEE8"
CONFIDENCE_BAND_OPACITY = 0.15
LINE_STYLES = {
    "solid": "-",
    "dashed": "--",
    "dotted": ":",
    "dashdot": "-.",
}


def chart_series_color(
    *,
    configured_color: str,
    grayscale: bool,
    grouped: bool,
    index: int,
) -> str:
    if grayscale:
        return GRAYSCALE_PALETTE[index % len(GRAYSCALE_PALETTE)]
    if grouped:
        return GROUP_PALETTE[index % len(GROUP_PALETTE)]
    return configured_color


ChartLineStyle = Literal["solid", "dashed", "dotted", "dashdot"]


def matplotlib_line_style(style: ChartLineStyle) -> str:
    return LINE_STYLES[style]
