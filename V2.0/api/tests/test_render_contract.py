from __future__ import annotations

import json
import warnings
from pathlib import Path

import matplotlib
import pandas as pd
import pytest
from matplotlib.figure import Figure

from labviz_api.models import ChartSpec
from labviz_api.processing import analyze_chart, render_chart
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


@pytest.mark.parametrize("chart_type", ["line", "scatter", "bar", "histogram"])
@pytest.mark.parametrize(
    "format_name,signature", [("png", b"\x89PNG"), ("svg", b"<?xml"), ("pdf", b"%PDF")]
)
@pytest.mark.parametrize("grid_visible", [False, True])
@pytest.mark.parametrize("panel_count", [1, 4])
def test_grid_visibility_matches_saved_export_in_every_format(
    chart_type: str, format_name: str, signature: bytes, grid_visible: bool, panel_count: int
) -> None:
    frame = pd.DataFrame({"x": [0, 1, 2, 3], "y": [1.0, 3.0, 2.0, 4.0]})
    original_frame = frame.copy(deep=True)
    chart = ChartSpec.model_validate(
        {
            "schemaVersion": 1,
            "type": chart_type,
            "title": "Grid visibility",
            "xAxis": {"field": "x", "title": "X", "unit": "s"},
            "yAxis": {"field": "y", "title": "Y", "unit": "V"},
            "series": [
                {"field": "y", "label": f"Y{panel}", "color": "#2563EB", "panel": panel}
                for panel in range(1, panel_count + 1)
            ],
            "panelCount": panel_count,
            "export": {
                "format": format_name,
                "dpi": 300,
                "sizePreset": "single-column",
                "grayscalePreview": False,
                "gridVisible": grid_visible,
            },
        }
    )
    figure = Figure(figsize=(3.35, 2.6), dpi=300)
    # A saved "off" preference must also override a grid-enabled Matplotlib theme.
    with (
        matplotlib.rc_context({"axes.grid": True}),
        warnings.catch_warnings(record=True) as captured,
    ):
        warnings.simplefilter("always")
        rendered = render_chart(frame, chart, _target_figure=figure)
    assert rendered.startswith(signature)
    assert len(figure.axes) == panel_count
    lines = [line for axis in figure.axes for line in axis.get_xgridlines() + axis.get_ygridlines()]
    assert lines
    assert all(line.get_visible() == grid_visible for line in lines)
    assert not any("First parameter to grid() is false" in str(item.message) for item in captured)
    if format_name == "svg":
        assert (f"stroke: {GRID_COLOR.lower()}".encode() in rendered) == grid_visible
    if grid_visible:
        assert all(line.get_color() == GRID_COLOR for line in lines)
        assert all(line.get_linewidth() == 0.7 and line.get_alpha() == 0.8 for line in lines)
    opposite = chart.model_copy(
        update={
            "export_settings": chart.export_settings.model_copy(
                update={"grid_visible": not grid_visible}
            )
        }
    )
    assert analyze_chart(frame, chart) == analyze_chart(frame, opposite)
    pd.testing.assert_frame_equal(frame, original_frame)
