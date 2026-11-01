from __future__ import annotations

import numpy as np
import pandas as pd

from labviz_api.models import ChartSpec
from labviz_api.processing import MAX_ANALYSIS_POINTS, analyze_chart


def _chart(**overrides: object) -> ChartSpec:
    payload: dict[str, object] = {
        "schemaVersion": 1,
        "type": "line",
        "title": "Preview decimation",
        "xAxis": {"field": "x", "title": "X", "unit": ""},
        "yAxis": {"field": "y", "title": "Y", "unit": ""},
        "series": [{"field": "y", "label": "Response", "color": "#2563EB"}],
        "panelCount": 1,
        "fitting": {"model": "none"},
        "uncertainty": {"mode": "none"},
        "export": {
            "format": "svg",
            "dpi": 300,
            "sizePreset": "single-column",
            "grayscalePreview": False,
        },
    }
    payload.update(overrides)
    return ChartSpec.model_validate(payload)


def _spiked_frame(rows: int = 4_000) -> pd.DataFrame:
    """A flat baseline with one narrow peak and one narrow trough."""
    frame = pd.DataFrame({"x": [float(value) for value in range(rows)]})
    frame["y"] = 1.0
    frame.loc[1_111, "y"] = 97.5
    frame.loc[2_222, "y"] = -42.25
    return frame


def test_preview_keeps_narrow_peaks_that_the_export_still_draws() -> None:
    frame = _spiked_frame()
    chart = _chart()

    series = analyze_chart(frame, chart)["series"][0]
    values = [point["y"] for point in series["points"]]

    assert max(values) == 97.5
    assert min(values) == -42.25
    # Evenly spaced positions are the behaviour this replaces: they miss both.
    spaced = np.linspace(0, len(frame) - 1, num=MAX_ANALYSIS_POINTS, dtype=int)
    assert 1_111 not in set(spaced.tolist())
    assert 2_222 not in set(spaced.tolist())


def test_decimated_preview_stays_bounded_ordered_and_keeps_both_ends() -> None:
    frame = _spiked_frame()
    chart = _chart(type="scatter")

    series = analyze_chart(frame, chart)["series"][0]
    xs = [point["x"] for point in series["points"]]

    assert 0 < len(xs) <= MAX_ANALYSIS_POINTS
    assert xs == sorted(xs)
    assert xs[0] == 0.0
    assert xs[-1] == float(len(frame) - 1)


def test_decimation_does_not_change_the_analysed_sample_size() -> None:
    frame = _spiked_frame()
    chart = _chart(fitting={"model": "linear", "confidenceBand": False})

    series = analyze_chart(frame, chart)["series"][0]
    fit = series["fit"]

    assert fit is not None
    assert fit["sampleSize"] == len(frame)
    assert len(series["points"]) <= MAX_ANALYSIS_POINTS


def test_single_field_distribution_preview_also_keeps_its_extremes() -> None:
    frame = _spiked_frame()
    chart = _chart(
        type="histogram",
        xAxis={"field": "y", "title": "Y", "unit": ""},
    )

    series = analyze_chart(frame, chart)["series"][0]
    values = [point["y"] for point in series["points"]]

    assert max(values) == 97.5
    assert min(values) == -42.25


def test_non_finite_rows_do_not_break_or_dominate_the_envelope() -> None:
    frame = _spiked_frame()
    frame.loc[3_000:3_400, "y"] = np.nan
    chart = _chart()

    series = analyze_chart(frame, chart)["series"][0]
    values = [point["y"] for point in series["points"]]

    assert all(np.isfinite(value) for value in values)
    assert max(values) == 97.5
    assert min(values) == -42.25
