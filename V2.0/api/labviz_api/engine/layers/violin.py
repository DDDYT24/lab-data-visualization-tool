"""A Gaussian density of each group's observations."""

from __future__ import annotations

import math
from statistics import stdev
from typing import TYPE_CHECKING

from ...render_contract import GEOMETRY
from ...research_preparation import PreparationError
from ...scientific_imports import scientific_module
from ..text import Text
from .base import Groups, Layer, LayerPlan, PlanContext, Result
from .grouping import grouped

if TYPE_CHECKING:
    import numpy as np

    from ...research_charts import ScientificChartSpec
else:
    np = scientific_module("numpy")

DENSITY = GEOMETRY["density"]
MIN_OBSERVATIONS = 3


def compute(result: Result, groups: Groups, spec: ScientificChartSpec) -> None:
    for group in result["groups"]:
        items = groups[group["label"]]
        values = [float(item["value"]) for item in items]
        if len(values) < MIN_OBSERVATIONS:
            raise PreparationError(
                "chart-density-small",
                "Violin density needs at least three observations per group.",
            )
        sigma = stdev(values)
        if sigma == 0:
            raise PreparationError(
                "chart-density-degenerate",
                "Constant data have no violin density; use raw points.",
            )
        bandwidth = spec.bandwidth or sigma * len(values) ** DENSITY["scottExponent"]
        tail = DENSITY["tailBandwidths"] * bandwidth
        low, high = min(values) - tail, max(values) + tail
        if not all(math.isfinite(x) for x in (bandwidth, low, high)) or low >= high:
            raise PreparationError(
                "chart-density-range", "Rescale these values before density estimation."
            )
        span_in_bandwidths = (high - low) / bandwidth
        if (
            not math.isfinite(span_in_bandwidths)
            or span_in_bandwidths > DENSITY["maxSpanBandwidths"]
        ):
            raise PreparationError(
                "chart-density-resolution",
                "Increase bandwidth to resolve this density reliably.",
            )
        grid_size = max(
            DENSITY["minGridPoints"],
            math.ceil(span_in_bandwidths * DENSITY["gridPointsPerBandwidth"]) + 1,
        )
        grid = np.linspace(low, high, grid_size)
        density = np.zeros(grid_size)
        # Exact Gaussian sum in bounded chunks; no resampling or hidden data reduction.
        for start in range(0, len(values), 2048):
            with np.errstate(over="ignore", under="ignore"):
                z = (grid[:, None] - np.asarray(values[start : start + 2048])[None, :]) / bandwidth
                density += np.exp(-0.5 * z * z).sum(axis=1)
        density /= len(values) * bandwidth * math.sqrt(2 * math.pi)
        if not np.isfinite(density).all() or float(density.max()) <= 0:
            raise PreparationError(
                "chart-density-numeric",
                "Bandwidth is numerically unsuitable; rescale or change it.",
            )
        result["densities"].append(
            {
                "group": group["label"],
                "position": group["position"],
                "observationIds": [item["id"] for item in items],
                "nObservations": len(values),
                "bandwidth": bandwidth,
                "bandwidthRule": "explicit"
                if spec.bandwidth is not None
                else DENSITY["bandwidthRule"],
                "kernel": DENSITY["kernel"],
                "support": [low, high],
                "grid": grid.tolist(),
                "density": density.tolist(),
                "inputUnit": "observation",
                "boundaryCorrection": "none",
                "gridSize": grid_size,
            }
        )
    peak = max(max(item["density"]) for item in result["densities"])
    for item in result["densities"]:
        denominator = max(item["density"]) if spec.density_scale == "equal_width" else peak
        item["halfWidths"] = [
            DENSITY["maxHalfWidth"] * value / denominator for value in item["density"]
        ]
        item["scale"] = spec.density_scale


def plan(context: PlanContext) -> LayerPlan:
    built = grouped(context)
    for density in context.result["densities"]:
        built.below.append(
            {
                "kind": "violin",
                "id": f"density:{density['position']}",
                "group": context.name(density["group"]),
                "color": context.colors[density["group"]],
                "opacity": 0.3,
                "position": float(density["position"]),
                "grid": density["grid"],
                "halfWidths": density["halfWidths"],
                "bandwidth": density["bandwidth"],
                "bandwidthRule": density["bandwidthRule"],
                "scale": density["scale"],
                "n": density["nObservations"],
                "sourceRefs": density["observationIds"],
            }
        )
        built.y_values += density["support"]
    return built


LAYER = Layer(
    kind="violin",
    title=Text("Violin distribution", "小提琴分布图"),
    short=Text("Violin", "小提琴图"),
    options=("summary", "bandwidth", "show_points"),
    geometry=("jitter", "density"),
    limits={"observationsPerGroup": MIN_OBSERVATIONS},
    compute=compute,
    plan=plan,
    plan_note="Gaussian KDE of observations; density is not additional data.",
    footer_note="Gaussian KDE of observations; density is not extra data.",
    output_note=Text(
        "Gaussian KDE; density is not additional observations.", "高斯核密度不是额外观测。"
    ),
)
