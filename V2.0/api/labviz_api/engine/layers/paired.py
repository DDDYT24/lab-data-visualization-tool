"""Two readings of each sample, joined."""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING, Any

from ...refusals import PreparationError
from ..text import Text
from .base import Groups, Layer, LayerPlan, PlanContext, Result

if TYPE_CHECKING:
    from ...research_charts import ScientificChartSpec


def check(spec: ScientificChartSpec) -> None:
    if (
        len(spec.pair_times) != 2
        or len(set(spec.pair_times)) != 2
        or any(not item.strip() for item in spec.pair_times)
    ):
        raise PreparationError(
            "chart-pair-times", "Paired plots require two distinct ordered time labels."
        )


def compute(result: Result, _groups: Groups, spec: ScientificChartSpec) -> None:
    pairs: dict[str, list[dict[str, Any]]] = defaultdict(list)
    key_samples: dict[str, str] = {}
    for item in (point["observation"] for point in result["points"]):
        if not item["sampleId"] or not item["pairKey"] or not item["time"]:
            raise PreparationError(
                "chart-pair-identity", "Pairing requires explicit sample, pair and time fields."
            )
        if item["time"] not in spec.pair_times:
            raise PreparationError("chart-pair-time", "Filter to the two selected time labels.")
        if item["pairKey"] in key_samples and key_samples[item["pairKey"]] != item["sampleId"]:
            raise PreparationError(
                "chart-pair-key-reused", "A pair key cannot identify different samples."
            )
        key_samples[item["pairKey"]] = item["sampleId"]
        pairs[item["sampleId"]].append(item)
    for sample_id, members in sorted(pairs.items()):
        if len(members) != 2 or {item["time"] for item in members} != set(spec.pair_times):
            raise PreparationError(
                "chart-pair-partner", "Each sample requires exactly one reading at each time."
            )
        for key in ("pairKey", "experimentalUnit", "group"):
            if len({item[key] for item in members}) != 1:
                raise PreparationError(
                    "chart-pair-incompatible", "Pair metadata must agree within a sample."
                )
        first, second = sorted(members, key=lambda item: spec.pair_times.index(item["time"]))
        result["pairs"].append(
            {
                "sampleId": sample_id,
                "sampleLabel": first["sampleLabel"],
                "pairKey": first["pairKey"],
                "group": first["group"],
                "firstId": first["id"],
                "secondId": second["id"],
                "first": first["value"],
                "second": second["value"],
                "delta": second["value"] - first["value"],
            }
        )
    for point in result["points"]:
        point["x"] = spec.pair_times.index(point["observation"]["time"])
        point["jitterOffset"] = 0.0


def plan(context: PlanContext) -> LayerPlan:
    result = context.result
    built = LayerPlan(
        x_axis={
            "scale": "category",
            "domain": (-0.35, 1.35),
            "ticks": [
                {"value": float(i), "label": t} for i, t in enumerate(result["spec"]["pairTimes"])
            ],
            "title": "",
            "niceRule": None,
            "grid": False,
        }
    )
    for group in result["groups"]:
        name, color = context.name(group["label"]), context.colors[group["label"]]
        pairs = [p for p in result["pairs"] if p["group"] == group["label"]]
        built.below.append(
            {
                "kind": "segments",
                "id": f"pairs:{group['position']}",
                "group": name,
                "color": color,
                "width": 1.2,
                "opacity": 0.65,
                "x0": [0.0] * len(pairs),
                "y0": [float(p["first"]) for p in pairs],
                "x1": [1.0] * len(pairs),
                "y1": [float(p["second"]) for p in pairs],
                "sourceRefs": [[p["firstId"], p["secondId"]] for p in pairs],
            }
        )
        built.y_values += [float(p[k]) for p in pairs for k in ("first", "second")]
        built.legend.append({"label": name, "color": color, "symbol": "line"})
    return built


LAYER = Layer(
    kind="paired",
    title=Text("Paired change", "配对变化图"),
    short=Text("Paired change", "配对变化图"),
    options=("pair_times",),
    roles=("sample", "pair", "time"),
    x_axis="pairTimes",
    geometry=(),
    check=check,
    compute=compute,
    plan=plan,
)
