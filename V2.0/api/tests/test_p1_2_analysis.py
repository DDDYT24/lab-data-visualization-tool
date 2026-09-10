from __future__ import annotations

import pandas as pd

from labviz_api.models import ChartSpec
from labviz_api.processing import analyze_chart, render_chart


def _chart(**overrides: object) -> ChartSpec:
    payload: dict[str, object] = {
        "schemaVersion": 1,
        "type": "line",
        "title": "P1-2 answer key",
        "xAxis": {"field": "x", "title": "X", "unit": ""},
        "yAxis": {"field": "y", "title": "Y", "unit": ""},
        "series": [{"field": "y", "label": "Response", "color": "#2563EB"}],
        "panelCount": 1,
        "fitting": {
            "model": "linear",
            "fitMethod": "ordinary-least-squares",
            "confidenceBand": True,
            "confidenceMethod": "bootstrap",
            "confidenceLevel": 95,
        },
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


def _valid_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "x": float(x),
                "y": 4 + 1.5 * x + (replicate - 1) * 0.12,
                "error": 0.12,
            }
            for x in range(8)
            for replicate in range(3)
        ]
    )


def test_bootstrap_and_weighted_fit_are_deterministic_and_disclosed() -> None:
    frame = _valid_frame()
    chart = _chart(
        fitting={
            "model": "linear",
            "fitMethod": "weighted-least-squares",
            "confidenceBand": True,
            "confidenceMethod": "bootstrap",
            "confidenceLevel": 95,
        },
        uncertainty={"mode": "column", "errorField": "error"},
    )

    first = analyze_chart(frame, chart)
    second = analyze_chart(frame, chart)
    first_series = first["series"][0]
    second_series = second["series"][0]
    fit = first_series["fit"]

    assert fit is not None
    assert fit["fitMethod"] == "weighted-least-squares"
    assert fit["confidenceMethod"] == "bootstrap"
    assert fit["sampleSize"] == 24
    assert fit["excludedCount"] == 0
    assert fit["points"] == second_series["fit"]["points"]
    assert all(point["lower"] < point["upper"] for point in fit["points"])
    assert first_series["residualDiagnostic"]["status"] == "supported"
    assert any(
        item["method"] == "prediction-band" and item["status"] == "deferred"
        for item in first_series["disclosures"]
    )
    assert any(
        item["method"] == "weighted-fitting" and item["status"] == "supported"
        for item in first_series["disclosures"]
    )

    exported = render_chart(frame, chart).decode("utf-8")
    assert "n=24" in exported
    assert "excluded=0" in exported
    assert "Deferred:" in exported


def test_insufficient_fit_is_explicitly_limited() -> None:
    frame = pd.DataFrame({"x": [0.0, 1.0, 2.0], "y": [1.0, 2.0, 4.0]})
    chart = _chart(
        fitting={
            "model": "polynomial",
            "polynomialOrder": 3,
            "confidenceBand": False,
        }
    )

    series = analyze_chart(frame, chart)["series"][0]

    assert series["fit"] is None
    assert any("needs more distinct complete points" in warning for warning in series["warnings"])
    assert any(
        item["method"] == "fit" and item["status"] == "insufficient-data"
        for item in series["disclosures"]
    )
    assert series["residualDiagnostic"] is None


def test_misleading_linear_fit_surfaces_residual_structure() -> None:
    frame = pd.DataFrame({"x": [float(value) for value in range(10)]})
    frame["y"] = frame["x"] ** 2
    chart = _chart(
        fitting={
            "model": "linear",
            "confidenceBand": False,
        }
    )

    series = analyze_chart(frame, chart)["series"][0]
    diagnostic = series["residualDiagnostic"]

    assert series["fit"] is not None
    assert diagnostic is not None
    assert diagnostic["residualTrend"] == "possible-trend"
    assert any("miss structure" in limitation for limitation in diagnostic["limitations"])
