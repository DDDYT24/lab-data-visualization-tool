"""Measure local-only V2.2 processing baselines and print JSON to stdout."""

from __future__ import annotations

import argparse
import csv
import ctypes
import io
import json
import os
import platform
import sys
import time
from collections.abc import Callable
from ctypes import wintypes
from importlib import import_module
from math import ceil
from pathlib import Path
from statistics import fmean, pstdev
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

BUDGETS_PATH = Path(__file__).resolve().parents[1] / "samples" / "v22" / "performance_budgets.json"


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


def _process_peak_memory_bytes() -> int | None:
    """Return the process peak working set without adding a runtime dependency."""
    if os.name == "nt":

        class ProcessMemoryCounters(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("page_fault_count", wintypes.DWORD),
                ("peak_working_set_size", ctypes.c_size_t),
                ("working_set_size", ctypes.c_size_t),
                ("quota_peak_paged_pool_usage", ctypes.c_size_t),
                ("quota_paged_pool_usage", ctypes.c_size_t),
                ("quota_peak_non_paged_pool_usage", ctypes.c_size_t),
                ("quota_non_paged_pool_usage", ctypes.c_size_t),
                ("pagefile_usage", ctypes.c_size_t),
                ("peak_pagefile_usage", ctypes.c_size_t),
            ]

        counters = ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        try:
            get_current_process = ctypes.windll.kernel32.GetCurrentProcess
            get_process_memory_info = ctypes.windll.psapi.GetProcessMemoryInfo
            get_process_memory_info.argtypes = [
                wintypes.HANDLE,
                ctypes.POINTER(ProcessMemoryCounters),
                wintypes.DWORD,
            ]
            get_process_memory_info.restype = wintypes.BOOL
            if get_process_memory_info(get_current_process(), ctypes.byref(counters), counters.cb):
                return int(counters.peak_working_set_size)
        except (AttributeError, OSError):
            return None
        return None

    try:
        resource_module: Any = import_module("resource")
        value = resource_module.getrusage(resource_module.RUSAGE_SELF).ru_maxrss
        return int(value if platform.system() == "Darwin" else value * 1024)
    except (ImportError, OSError):
        return None


def _timing_stats(runs: list[float]) -> dict[str, float]:
    ordered = sorted(runs)
    percentile_index = max(0, min(len(ordered) - 1, ceil(len(ordered) * 0.95) - 1))
    return {
        "min_ms": min(ordered),
        "mean_ms": round(fmean(ordered), 2),
        "p95_ms": ordered[percentile_index],
        "max_ms": max(ordered),
        "stdev_ms": round(pstdev(ordered), 2),
    }


def _aggregate_cases(runs: list[list[dict[str, Any]]]) -> list[dict[str, Any]]:
    aggregated: list[dict[str, Any]] = []
    for index in range(len(runs[0])):
        case_runs = [run[index] for run in runs]
        stage_names = list(case_runs[0]["timings"])
        stats = {
            stage: _timing_stats([case["timings"][stage] for case in case_runs])
            for stage in stage_names
        }
        aggregated.append(
            {
                key: value
                for key, value in case_runs[0].items()
                if key not in {"timings", "png_bytes"}
            }
            | {
                "repetitions": len(case_runs),
                "timings": {stage: values["max_ms"] for stage, values in stats.items()},
                "timing_stats": stats,
                "max_total_ms": round(max(sum(case["timings"].values()) for case in case_runs), 2),
                "png_bytes": max(case["png_bytes"] for case in case_runs),
            }
        )
    return aggregated


def _budget_violations(result: dict[str, Any], budgets: dict[str, Any]) -> list[str]:
    violations: list[str] = []
    for case in result["tabular"]:
        budget = budgets["tabular"][str(case["rows"])]
        for stage, limit in budget["stages"].items():
            observed = case["timings"][stage]
            if observed > limit:
                violations.append(f"tabular/{case['rows']}/{stage}={observed}ms>{limit}ms")
        total = case.get("max_total_ms", sum(case["timings"].values()))
        if total > budget["total_ms"]:
            violations.append(f"tabular/{case['rows']}/total={total:.2f}ms>{budget['total_ms']}ms")

    for case in result["surface"]:
        budget = budgets["surface"][str(case["side"])]
        for stage, limit in budget["stages"].items():
            observed = case["timings"][stage]
            if observed > limit:
                violations.append(f"surface/{case['side']}/{stage}={observed}ms>{limit}ms")
        total = case.get("max_total_ms", sum(case["timings"].values()))
        if total > budget["total_ms"]:
            violations.append(f"surface/{case['side']}/total={total:.2f}ms>{budget['total_ms']}ms")

    peak = result.get("process_peak_memory_mib")
    if peak is not None and peak > budgets["process_peak_memory_mib"]:
        violations.append(f"process/peak_memory={peak}MiB>{budgets['process_peak_memory_mib']}MiB")
    return violations


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="also write the JSON report to this path")
    parser.add_argument(
        "--repetitions",
        type=int,
        default=3,
        help="number of times to repeat every case (default: 3)",
    )
    arguments = parser.parse_args()
    if not 1 <= arguments.repetitions <= 20:
        parser.error("--repetitions must be between 1 and 20")
    budgets = json.loads(BUDGETS_PATH.read_text(encoding="utf-8"))
    tabular_runs = [
        [_tabular_baseline(size) for size in (24, 1_000, 10_000)]
        for _ in range(arguments.repetitions)
    ]
    surface_runs = [
        [_surface_baseline(side) for side in (21, 101)] for _ in range(arguments.repetitions)
    ]
    result = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "pandas": pd.__version__,
        "repetitions": arguments.repetitions,
        "tabular": _aggregate_cases(tabular_runs),
        "surface": _aggregate_cases(surface_runs),
    }
    peak_bytes = _process_peak_memory_bytes()
    result["process_peak_memory_mib"] = (
        round(peak_bytes / (1024 * 1024), 2) if peak_bytes is not None else None
    )
    violations = _budget_violations(result, budgets)
    result["budget"] = {
        "contractVersion": budgets["contractVersion"],
        "reviewStatus": budgets["status"],
        "status": "fail" if violations else "pass",
        "violations": violations,
    }
    serialized = json.dumps(result, ensure_ascii=False, indent=2)
    if arguments.output:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(serialized + "\n", encoding="utf-8")
    print(serialized)
    if violations:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
