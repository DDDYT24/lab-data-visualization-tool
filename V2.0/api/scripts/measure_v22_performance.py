"""Measure local-only V2.2 processing baselines and print JSON to stdout."""

from __future__ import annotations

import csv
import io
import json
import platform
import sys
import time
from collections.abc import Callable
from typing import Any

import pandas as pd

from labviz_api.models import ChartSpec
from labviz_api.processing import (
    analyze_chart,
    build_preview,
    build_quality_report,
    load_dataframe,
    render_chart,
)


def _measure(function: Callable[[], Any]) -> tuple[float, Any]:
    started = time.perf_counter()
    result = function()
    return round((time.perf_counter() - started) * 1000, 2), result


def _csv_payload(row_count: int) -> bytes:
    output = io.StringIO(newline="")
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(["time_min", "response_mV", "temperature_C"])
    for index in range(row_count):
        writer.writerow([index, round(18 + index * 0.02, 4), round(23 + index * 0.001, 4)])
    return output.getvalue().encode("utf-8")


def _line_chart() -> ChartSpec:
    return ChartSpec.model_validate(
        {
            "schemaVersion": 1,
            "type": "line",
            "title": "Performance baseline",
            "xAxis": {"field": "time_min", "title": "Time", "unit": "min"},
            "yAxis": {"field": "response_mV", "title": "Response", "unit": "mV"},
            "series": [{"field": "response_mV", "label": "Response", "color": "#2563EB"}],
            "panelCount": 1,
            "export": {
                "format": "png",
                "dpi": 300,
                "sizePreset": "single-column",
                "grayscalePreview": False,
            },
        }
    )


def _surface_frame(side: int) -> pd.DataFrame:
    rows = [(x, y, round(x * x + y * y, 5)) for y in range(side) for x in range(side)]
    return pd.DataFrame(rows, columns=["x", "y", "z"])


def _surface_chart() -> ChartSpec:
    return ChartSpec.model_validate(
        {
            "schemaVersion": 1,
            "type": "surface3d",
            "title": "Surface baseline",
            "xAxis": {"field": "x", "title": "X", "unit": ""},
            "yAxis": {"field": "z", "title": "Z", "unit": ""},
            "series": [
                {"field": "y", "label": "Y", "color": "#2563EB"},
                {"field": "z", "label": "Z", "color": "#0F766E"},
            ],
            "panelCount": 1,
            "export": {
                "format": "png",
                "dpi": 300,
                "sizePreset": "single-column",
                "grayscalePreview": False,
            },
        }
    )


def _tabular_baseline(row_count: int) -> dict[str, Any]:
    payload = _csv_payload(row_count)
    timings: dict[str, float] = {}
    timings["load_ms"], frame_details = _measure(lambda: load_dataframe(payload, "baseline.csv"))
    frame, _sheet, _sheets, _header = frame_details
    timings["preview_ms"], _ = _measure(lambda: build_preview("baseline", frame))
    timings["quality_ms"], _ = _measure(lambda: build_quality_report("baseline", frame))
    chart = _line_chart()
    timings["analysis_ms"], _ = _measure(lambda: analyze_chart(frame, chart))
    timings["png_render_ms"], rendered = _measure(lambda: render_chart(frame, chart))
    return {
        "rows": row_count,
        "bytes": len(payload),
        "timings": timings,
        "png_bytes": len(rendered),
    }


def _surface_baseline(side: int) -> dict[str, Any]:
    frame = _surface_frame(side)
    chart = _surface_chart()
    timings: dict[str, float] = {}
    timings["quality_ms"], _ = _measure(lambda: build_quality_report("surface", frame))
    timings["analysis_ms"], analysis = _measure(lambda: analyze_chart(frame, chart))
    timings["png_render_ms"], rendered = _measure(lambda: render_chart(frame, chart))
    diagnostic = analysis["preview"]["surfaceDiagnostics"][0]
    return {
        "side": side,
        "points": len(frame),
        "timings": timings,
        "surface_status": diagnostic["status"],
        "png_bytes": len(rendered),
    }


def main() -> None:
    result = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "pandas": pd.__version__,
        "tabular": [_tabular_baseline(size) for size in (24, 1_000, 10_000)],
        "surface": [_surface_baseline(side) for side in (21, 101)],
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
