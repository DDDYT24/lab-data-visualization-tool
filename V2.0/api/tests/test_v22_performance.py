from __future__ import annotations

from typing import Any

from scripts.measure_v22_performance import _aggregate_cases, _budget_violations


def _case(rows: int, timings: dict[str, float], png_bytes: int = 100) -> dict[str, Any]:
    return {
        "rows": rows,
        "bytes": 128,
        "timings": timings,
        "png_bytes": png_bytes,
    }


def test_repeated_performance_runs_use_conservative_maxima_and_publish_variance() -> None:
    runs = [
        [_case(24, {"load_ms": 5.0, "png_render_ms": 20.0}, 100)],
        [_case(24, {"load_ms": 7.0, "png_render_ms": 18.0}, 120)],
        [_case(24, {"load_ms": 6.0, "png_render_ms": 25.0}, 110)],
    ]

    [result] = _aggregate_cases(runs)

    assert result["repetitions"] == 3
    assert result["timings"] == {"load_ms": 7.0, "png_render_ms": 25.0}
    assert result["max_total_ms"] == 31.0
    assert result["png_bytes"] == 120
    assert result["timing_stats"]["load_ms"] == {
        "min_ms": 5.0,
        "mean_ms": 6.0,
        "p95_ms": 7.0,
        "max_ms": 7.0,
        "stdev_ms": 0.82,
    }


def test_performance_budget_checker_reports_stage_total_and_memory_violations() -> None:
    result = {
        "tabular": [
            _case(
                24,
                {
                    "load_ms": 11.0,
                    "preview_ms": 1.0,
                    "quality_ms": 1.0,
                    "analysis_ms": 1.0,
                    "png_render_ms": 1.0,
                },
            )
            | {"max_total_ms": 20.0}
        ],
        "surface": [],
        "process_peak_memory_mib": 12.0,
    }
    budgets = {
        "tabular": {
            "24": {
                "stages": {
                    "load_ms": 10.0,
                    "preview_ms": 10.0,
                    "quality_ms": 10.0,
                    "analysis_ms": 10.0,
                    "png_render_ms": 10.0,
                },
                "total_ms": 15.0,
            }
        },
        "surface": {},
        "process_peak_memory_mib": 8.0,
    }

    violations = _budget_violations(result, budgets)

    assert violations == [
        "tabular/24/load_ms=11.0ms>10.0ms",
        "tabular/24/total=20.00ms>15.0ms",
        "process/peak_memory=12.0MiB>8.0MiB",
    ]
