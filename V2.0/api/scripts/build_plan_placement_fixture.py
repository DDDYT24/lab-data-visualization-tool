"""Generate or check the placement fixture both renderers' tests read.

It holds axis positions for sample values on every scale, one plan with log axes and
every kind of note, and where that plan's points and bracket land along the axes; and a
histogram on a log count axis with where its bins land; and raw points over a violin with the
order their marks are drawn in.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from labviz_api.models import ChartSpec
from labviz_api.plan_axes import apply_axis_controls, axis_position
from labviz_api.plan_notes import apply_notes
from labviz_api.processing import analyze_chart, classic_plan

DESTINATION = (
    Path(__file__).resolve().parents[2] / "contracts" / "fixtures" / "v24" / "plan-placement.json"
)
SAMPLES = {
    "linear": [-3.5, 0.0, 42.0],
    "log10": [1e-6, 0.5, 1.0, 37.0, 1e9],
    "symlog": [-1e6, -9.0, -0.25, 0.0, 0.25, 9.0, 1e6],
}
NOTES: list[dict[str, Any]] = [
    {"kind": "text", "x": 2.0, "y": 500.0, "text": "peak"},
    {"kind": "hline", "value": 100.0, "label": "limit"},
    {"kind": "vline", "value": 3.0, "label": "start"},
    {"kind": "rect", "x": [1.5, 2.5], "y": [10.0, 400.0], "label": "window"},
    {"kind": "bracket", "x": [2.0, 4.0], "y": 800.0, "label": "p < 0.01"},
]


def log_histogram() -> dict[str, Any]:
    """A histogram with empty bins on a log count axis, and where its bins land."""
    values = [1.0] * 4 + [1.5] * 30 + [2.0] + [4.0] * 2
    data = pd.DataFrame({"v": values})
    spec = ChartSpec.model_validate(
        {
            "schemaVersion": 1,
            "type": "histogram",
            "title": "Log counts",
            "xAxis": {"field": "v", "title": "V", "unit": ""},
            "yAxis": {"field": "v", "title": "Count", "unit": ""},
            "series": [{"field": "v", "label": "V", "color": "#2563EB"}],
            "panelCount": 1,
            "export": {
                "format": "png",
                "dpi": 300,
                "sizePreset": "single-column",
                "grayscalePreview": False,
            },
        }
    )
    plan = classic_plan(data, spec, analyze_chart(data, spec))
    assert plan
    plan = apply_axis_controls(plan, [{"y": {"scale": "log10"}}])
    bins = next(mark for mark in plan["panels"][0]["marks"] if mark["kind"] == "rects")
    return {
        "plan": plan,
        "placed": [
            [x0, axis_position("log10", y0), x1, axis_position("log10", y1)]
            for x0, y0, x1, y1 in zip(bins["x0"], bins["y0"], bins["x1"], bins["y1"], strict=True)
        ],
    }


def layered() -> dict[str, Any]:
    """Raw points over a violin of the same data, and the order Matplotlib draws the marks in."""
    from matplotlib.figure import Figure

    from labviz_api.render_plan import build_research_plan, merge_layers
    from labviz_api.render_plan_draw import draw_mark
    from labviz_api.research_charts import ScientificChartSpec, compute_chart
    from labviz_api.research_preparation import PreparationPlan, parse_sources, prepare

    rows = ["sample,group,value,unit"]
    rows += [f"s{i},{'Control' if i % 2 else 'Treated'},{i * 3},mV" for i in range(1, 9)]
    plan = PreparationPlan.model_validate(
        {
            "mapping": {
                "value": "value",
                "sample": "sample",
                "experimentalUnit": "sample",
                "group": "group",
                "unit": "unit",
            }
        }
    )
    prepared = prepare(parse_sources("fixture", [("layers.csv", "\n".join(rows).encode())]), plan)
    merged = merge_layers(
        [
            build_research_plan(
                compute_chart(prepared, ScientificChartSpec.model_validate({"kind": kind}))
            )
            for kind in ("violin", "grouped_strip")
        ]
    )
    marks = merged["panels"][0]["marks"]
    ax = Figure().subplots()
    depth = []
    for mark in marks:
        before = set(ax.get_children())
        draw_mark(ax, mark, False)
        depth.append(min(a.get_zorder() for a in ax.get_children() if a not in before))
    order = sorted(range(len(marks)), key=lambda index: (depth[index], index))
    return {"plan": merged, "drawOrder": [marks[index]["id"] for index in order]}


def build() -> dict[str, Any]:
    values = [1.0, 10.0, 100.0, 1000.0]
    data = pd.DataFrame({"x": [float(i + 1) for i in range(len(values))], "y": values})
    spec = ChartSpec.model_validate(
        {
            "schemaVersion": 1,
            "type": "scatter",
            "title": "Placement",
            "xAxis": {"field": "x", "title": "X", "unit": ""},
            "yAxis": {"field": "y", "title": "Y", "unit": ""},
            "series": [{"field": "y", "label": "Y", "color": "#2563EB"}],
            "panelCount": 1,
            "export": {
                "format": "png",
                "dpi": 300,
                "sizePreset": "single-column",
                "grayscalePreview": False,
            },
        }
    )
    plan = classic_plan(data, spec, analyze_chart(data, spec))
    assert plan
    scales = {"x": {"scale": "log10"}, "y": {"scale": "log10"}}
    plan = apply_notes(apply_axis_controls(plan, [scales]), [NOTES])
    panel = plan["panels"][0]
    points = next(mark for mark in panel["marks"] if mark["kind"] == "points")
    bracket = panel["notes"][4]

    def place(x: float, y: float) -> list[float]:
        return [axis_position("log10", x), axis_position("log10", y)]

    return {
        "positions": [
            {"scale": scale, "value": value, "position": axis_position(scale, value)}
            for scale, values_ in SAMPLES.items()
            for value in values_
        ],
        "plan": plan,
        "placed": {
            "points": [place(x, y) for x, y in zip(points["x"], points["y"], strict=True)],
            "bracket": [
                place(bracket["x"][0], bracket["foot"]),
                place(bracket["x"][0], bracket["y"]),
                place(bracket["x"][1], bracket["y"]),
                place(bracket["x"][1], bracket["foot"]),
            ],
            "bracketLabel": place(*bracket["labelAt"]),
            "rectLabel": place(*panel["notes"][3]["labelAt"]),
        },
        "logHistogram": log_histogram(),
        "layered": layered(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    built = json.loads(json.dumps(build()))
    if args.check:
        if json.loads(DESTINATION.read_text(encoding="utf-8")) != built:
            raise SystemExit("plan-placement.json differs from the Python contract")
        print("Plan placement fixture matches.")
    else:
        DESTINATION.write_text(json.dumps(built, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
