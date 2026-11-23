"""The category axis of groups, with their summary marks, shared by the kinds that use it."""

from __future__ import annotations

from .base import LayerPlan, PlanContext


def grouped(context: PlanContext) -> LayerPlan:
    spec = context.result["spec"]
    groups = context.result["groups"]
    built = LayerPlan(
        x_axis={
            "scale": "category",
            "domain": (-0.65, max(len(groups) - 0.35, 0.65)),
            "ticks": [],
            "title": "",
            "niceRule": None,
            "grid": False,
        }
    )
    for group in groups:
        count = "?" if group["independentUnits"] is None else str(group["independentUnits"])
        label = context.name(group["label"])[:36]
        if context.category_counts:
            label += f"\nobs={group['observations']}, units={count}"
        built.x_axis["ticks"].append({"value": float(group["position"]), "label": label})
        if group["summary"] is not None:
            built.above.append(
                {
                    "kind": "summary",
                    "id": f"summary:{group['position']}",
                    "group": context.name(group["label"]),
                    "color": "#303030" if context.use_grayscale else "#192d40",
                    "width": 2.4,
                    "x0": group["position"] - 0.25,
                    "x1": group["position"] + 0.25,
                    "y": float(group["summary"]),
                    "statistic": spec["summary"],
                    "summaryUnit": spec["summaryUnit"],
                    "n": group["summaryN"],
                }
            )
            built.y_values.append(float(group["summary"]))
    return built
