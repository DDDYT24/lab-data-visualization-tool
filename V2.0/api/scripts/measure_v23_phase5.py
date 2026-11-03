"""Measure current scientific computation/composition; do not claim browser acceptance."""

from __future__ import annotations

import argparse
import gc
import hashlib
import io
import json
import platform
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any
from unittest.mock import patch

import pandas as pd
from matplotlib.figure import Figure

from labviz_api import research_figure_render
from labviz_api.research_charts import ScientificChartSpec, compute_chart
from labviz_api.research_figure_board import FigureBoardSpec, SceneIndex, interaction
from labviz_api.research_figure_render import render_board
from labviz_api.research_figure_sources import legacy_snapshot, scientific_source
from labviz_api.research_preparation import PreparationPlan, parse_sources, prepare
from labviz_api.research_raster import freeze_raster_layers
from scripts.measure_v22_performance import _csv_payload, _line_chart, _process_peak_memory_bytes


def measure(rows: int, mixed: bool = False) -> dict[str, Any]:
    payload = _csv_payload(rows)
    plan = PreparationPlan.model_validate(
        {
            "mapping": {
                "value": "response_mV",
                "defaultUnit": "mV",
            }
        }
    )
    runs: list[dict[str, float]] = []
    sizes: dict[str, int] = {}
    for _ in range(3):
        gc.collect()
        times: dict[str, float] = {}
        start = perf_counter()
        raw = parse_sources("phase5-benchmark", [("legacy-linear.csv", payload)])
        prepared = prepare(raw, plan)
        times["parsePrepareMs"] = (perf_counter() - start) * 1000
        results = []
        for kind in ("grouped_strip", "violin"):
            start = perf_counter()
            results.append(
                compute_chart(prepared, ScientificChartSpec.model_validate({"kind": kind}))
            )
            times[kind + "RecomputeMs"] = (perf_counter() - start) * 1000
        # Like the separate compute/composition HTTP calls, release preparation inputs
        # after the immutable results exist. Lifetime peak still covers every stage.
        del raw, prepared
        gc.collect()
        scientific = [scientific_source(r, "benchmark") for r in results]
        sources = scientific * 2
        if mixed:
            frame = pd.read_csv(io.BytesIO(payload))

            class Repository:
                def __init__(self, data: pd.DataFrame) -> None:
                    self.frame = data

                def get_workspace(self, _: str) -> dict[str, Any]:
                    return {
                        "chart": _line_chart().model_dump(),
                        "decisions": [],
                        "quality": {},
                        "preview": {},
                    }

                def load_chart_dataframe(self, _: str) -> Any:
                    return self.frame

                def load_quality_dataframe(self, _: str) -> Any:
                    return self.frame

            start = perf_counter()
            captured = legacy_snapshot(Repository(frame), "benchmark", 1)
            times["legacyCaptureMs"] = (perf_counter() - start) * 1000
            sources = [captured, scientific[0], captured, scientific[1]]
        encode_times: dict[str, float] = {}
        original_savefig = Figure.savefig
        raster_times: list[float] = []
        original_freeze = freeze_raster_layers

        def timed_freeze(
            figure: Figure,
            _freeze: Any = original_freeze,
            _record: list[float] = raster_times,
        ) -> int:
            start = perf_counter()
            result: int = _freeze(figure)
            _record.append((perf_counter() - start) * 1000)
            return result

        def timed_savefig(
            self: Any,
            *args: Any,
            _save: Any = original_savefig,
            _record: dict[str, float] = encode_times,
            **kwargs: Any,
        ) -> Any:
            start = perf_counter()
            value = _save(self, *args, **kwargs)
            _record[kwargs["format"]] = (perf_counter() - start) * 1000
            return value

        start = perf_counter()
        with (
            patch.object(Figure, "savefig", timed_savefig),
            patch.object(research_figure_render, "freeze_raster_layers", timed_freeze),
        ):
            artifacts, scene = render_board(
                sources,
                FigureBoardSpec(
                    title="Four scientific panels", panel_chart_ids=["a", "b", "c", "d"]
                ),
            )
        total = (perf_counter() - start) * 1000
        raster_prepare = sum(raster_times)
        layout = total - sum(encode_times.values()) - raster_prepare
        times["fourPanelAllFormatsMs"] = total
        times["fourPanelGeometryMs"] = layout
        times["rasterLayerPrepareMs"] = raster_prepare
        for kind, duration in encode_times.items():
            # PNG uses its direct Agg render; only vector exports need the cache.
            cold_raster_cost = 0.0 if kind == "png" else raster_prepare
            times["fourPanel" + kind.title() + "Ms"] = layout + cold_raster_cost + duration
        sizes = {key: len(value) for key, value in artifacts.items()}
        assert sum(len(scene["rowSets"][key]) for key in scene["panelRowSets"].values()) == rows * 4
        assert len(scene["marks"]) >= 4
        selected = next(
            mark
            for mark in scene["marks"]
            if mark["kind"] == "point" and (not mixed or mark["panel"] == "B")
        )
        start = perf_counter()
        index = SceneIndex.build(scene)
        times["interactionIndexBuildMs"] = (perf_counter() - start) * 1000
        start = perf_counter()
        highlighted = interaction(scene, selected["id"], None, 0, "", False, index)
        times["highlightCoreMs"] = (perf_counter() - start) * 1000
        assert highlighted["selectedInputs"] == 1 and highlighted["totalRows"] == (
            2 if mixed else 4
        )
        runs.append({key: round(value, 2) for key, value in times.items()})
        del results, sources, scientific, artifacts, scene, index, highlighted
        if mixed:
            del captured, frame, Repository
    peak = _process_peak_memory_bytes()
    return {
        "rows": rows,
        "workload": "legacy-line/raw-point/legacy-line/violin"
        if mixed
        else "raw-point/violin/raw-point/violin",
        "dataset": f"legacy-linear-{rows}",
        "datasetSha256": hashlib.sha256(payload).hexdigest(),
        "runs": runs,
        "maxMs": {key: max(run[key] for run in runs) for key in runs[0]},
        "artifactBytes": sizes,
        "backendPeakMiB": None if peak is None else round(peak / 2**20, 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, choices=[10000, 100000], default=10000)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mixed", action="store_true")
    args = parser.parse_args()
    report: dict[str, Any] = {
        "measuredAt": datetime.now(UTC).isoformat(),
        "scriptSha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "implementationSha256": {
            file.name: hashlib.sha256(file.read_bytes()).hexdigest()
            for file in sorted(
                (Path(__file__).resolve().parents[1] / "labviz_api").glob("research*.py")
            )
        },
        "host": platform.node(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "processor": platform.processor(),
        "statistic": "maximum of three runs, including first; not p95",
        "scope": "Core functions only; no HTTP, disk-save or browser costs. "
        "Per-format times include shared geometry plus that format encoding. "
        "SVG/PDF each include the full cold raster-layer preparation cost; PNG "
        "uses its direct render and excludes that vector-only preparation. "
        "Preparation inputs are released after computation, like separate HTTP stages; "
        "full frozen source results remain. Lifetime peak covers every stage. "
        "All formats are actually rendered; these are proxy timings.",
        "unmeasured": [
            "browser first preview",
            "highlight response",
            "save",
            "reopen",
            "browser memory",
            "human usability",
        ],
        "case": measure(args.rows, args.mixed),
    }
    contract = json.loads(
        (Path(__file__).resolve().parents[2] / "contracts/v23-workflow-v1.json").read_text()
    )
    budget = contract["performance"]
    case = report["case"]
    targets = {
        "highlightCoreMs": budget["targetsMs"]["highlightResponse"],
        "grouped_stripRecomputeMs": budget["targetsMs"]["recompute"],
        "violinRecomputeMs": budget["targetsMs"]["recompute"],
        "fourPanelPngMs": budget["targetsMs"]["fourPanelPng300dpi"],
        "fourPanelSvgMs": budget["targetsMs"]["fourPanelSvg"],
        "fourPanelPdfMs": budget["targetsMs"]["fourPanelPdf"],
    }
    report["proxyBudgetChecks"] = {
        key: {
            "limitMs": limit,
            "actualMs": case["maxMs"][key],
            "passed": case["maxMs"][key] <= limit,
        }
        for key, limit in targets.items()
    }
    report["memoryBudgetMiB"] = budget["memory"]["backendPeakMiB"]
    report["memoryBudgetCheck"] = {
        "limitMiB": report["memoryBudgetMiB"],
        "actualMiB": case["backendPeakMiB"],
        "passed": case["backendPeakMiB"] is not None
        and case["backendPeakMiB"] <= report["memoryBudgetMiB"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {"output": str(args.output), "case": case, "checks": report["proxyBudgetChecks"]},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
