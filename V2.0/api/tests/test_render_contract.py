from __future__ import annotations

import json
from pathlib import Path

from labviz_api.render_contract import (
    CHART_RENDER_CONTRACT_VERSION,
    CONFIDENCE_BAND_OPACITY,
    CORRELATION_PALETTE,
    GRAYSCALE_PALETTE,
    GRID_COLOR,
    GROUP_PALETTE,
    HISTOGRAM_AXIS_PADDING,
    HISTOGRAM_FILL_OPACITY,
    LINE_STYLES,
    SURFACE_GRAYSCALE_PALETTE,
    SURFACE_PALETTE,
    chart_series_color,
    matplotlib_line_style,
)

CONTRACT_PATH = Path(__file__).resolve().parents[2] / "contracts" / "chart-render-v1.json"


def test_python_renderer_matches_shared_render_contract() -> None:
    contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))

    assert contract["contractVersion"] == CHART_RENDER_CONTRACT_VERSION
    assert list(GRAYSCALE_PALETTE) == contract["grayscalePalette"]
    assert list(GROUP_PALETTE) == contract["groupPalette"]
    assert list(SURFACE_PALETTE) == contract["surfacePalette"]
    assert list(CORRELATION_PALETTE) == contract["correlationPalette"]
    assert list(SURFACE_GRAYSCALE_PALETTE) == contract["surfaceGrayscalePalette"]
    assert contract["gridColor"] == GRID_COLOR
    assert contract["histogramFillOpacity"] == HISTOGRAM_FILL_OPACITY
    assert contract["histogramAxisPadding"] == HISTOGRAM_AXIS_PADDING
    assert contract["confidenceBandOpacity"] == CONFIDENCE_BAND_OPACITY
    assert {
        name: values["matplotlib"] for name, values in contract["lineStyles"].items()
    } == LINE_STYLES


def test_grayscale_takes_precedence_over_group_colors() -> None:
    assert (
        chart_series_color(configured_color="#FF0000", grayscale=True, grouped=True, index=1)
        == "#5B616B"
    )
    assert (
        chart_series_color(configured_color="#FF0000", grayscale=False, grouped=True, index=1)
        == "#0F766E"
    )
    assert matplotlib_line_style("dashdot") == "-."
