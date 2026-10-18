"""Versioned, renderer-neutral scientific chart results; no renderer recomputes statistics."""

from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from statistics import mean, median, stdev
from typing import Any, Literal

import numpy as np
from pydantic import Field, model_validator
from pydantic_core import to_json

from .research_contract import ResearchModel
from .research_preparation import PreparationError
from .research_strip import StripSpec, build_strip
from .research_styles import category_style


class CategoryStyle(ResearchModel):
    group: str | None
    color: str = Field(pattern=r"^#[0-9A-Fa-f]{6}$")
    marker: Literal["o", "s", "^", "D", "v"] = "o"


class ScientificChartSpec(ResearchModel):
    schema_version: Literal[1] = 1
    point_rendering: Literal["auto", "vector", "raster"] = "auto"
    kind: Literal["grouped_strip", "paired", "violin"] = "grouped_strip"
    title: str = Field(default="", max_length=160)
    locale: Literal["en", "zh"] = "en"
    jitter_seed: str = Field(default="labviz-strip-v1", min_length=1, max_length=128)
    jitter_width: float = Field(default=0.3, ge=0, le=0.45)
    show_points: bool = True
    summary: Literal["none", "mean", "median"] = "none"
    summary_unit: Literal["observation", "sample", "experimental_unit"] = "experimental_unit"
    pair_times: list[str] = Field(default_factory=list, max_length=2)
    bandwidth: float | None = Field(default=None, gt=0)
    density_scale: Literal["equal_width", "common_density"] = "equal_width"
    category_styles: list[CategoryStyle] = Field(default_factory=list, max_length=100)

    @model_validator(mode="after")
    def check_options(self) -> ScientificChartSpec:
        if len({item.group for item in self.category_styles}) != len(self.category_styles):
            raise ValueError("Category style keys must be unique.")
        if self.kind == "paired":
            if (
                len(self.pair_times) != 2
                or len(set(self.pair_times)) != 2
                or any(not item.strip() for item in self.pair_times)
            ):
                raise ValueError("Paired plots require two distinct ordered time labels.")
            if self.summary != "none":
                raise ValueError("Paired plots expose individual deltas, not a pooled summary.")
        elif self.pair_times:
            raise ValueError("Time labels apply only to paired plots.")
        if self.kind != "violin" and (self.bandwidth is not None or not self.show_points):
            raise ValueError("Bandwidth and optional points apply only to violin plots.")
        return self


def reject(code: str, message: str) -> None:
    raise PreparationError(code, message)


def summary_values(items: list[dict[str, Any]], unit: str) -> list[float]:
    if unit == "observation":
        return [float(item["value"]) for item in items]
    key = "sampleId" if unit == "sample" else "experimentalUnit"
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in items:
        if item[key] is None:
            reject("chart-unknown-summary-unit", "Declare all selected summary-unit identities.")
        buckets[item[key]].append(item)
    values = []
    for members in buckets.values():
        if len(members) > 1:
            repeats = [item["technicalRepeat"] for item in members]
            if (
                None in repeats
                or len(set(repeats)) != len(repeats)
                or len({item["time"] for item in members}) != 1
                or len({item["sampleId"] for item in members}) != 1
            ):
                reject(
                    "chart-ambiguous-repeat",
                    "Filter to one time/measurement or declare unique technical repeats.",
                )
        values.append(mean(item["value"] for item in members))
    return values


def _compute_chart(prepared: dict[str, Any], spec: ScientificChartSpec) -> dict[str, Any]:
    result = build_strip(
        prepared, StripSpec(jitter_seed=spec.jitter_seed, jitter_width=spec.jitter_width)
    )
    result["spec"] = spec.model_dump()
    if len(result["groups"]) > 100:
        reject(
            "chart-many-groups", "Filter to at most 100 groups before creating a readable chart."
        )
    styles = {item.group: item.model_dump() for item in spec.category_styles}
    result["spec"]["categoryStyles"] = [
        styles.get(group["label"], category_style(group["label"])) for group in result["groups"]
    ]
    result["sources"] = prepared["sources"]
    result["algorithmVersion"] = "scientific-charts-v1"
    result["densities"] = []
    result["pairs"] = []
    observations = [point["observation"] for point in result["points"]]
    if len({item["measurementType"] for item in observations}) > 1:
        reject("chart-mixed-measurements", "Filter to one explicitly declared measurement type.")
    groups: dict[str | None, list[dict[str, Any]]] = defaultdict(list)
    for item in observations:
        groups[item["group"]].append(item)
    for group in result["groups"]:
        group["summary"] = None
        group["summaryN"] = None
        if spec.summary != "none":
            values = summary_values(groups[group["label"]], spec.summary_unit)
            group["summary"] = (mean if spec.summary == "mean" else median)(values)
            group["summaryN"] = len(values)

    if spec.kind == "paired":
        pairs: dict[str, list[dict[str, Any]]] = defaultdict(list)
        key_samples: dict[str, str] = {}
        for item in observations:
            if not item["sampleId"] or not item["pairKey"] or not item["time"]:
                reject(
                    "chart-pair-identity", "Pairing requires explicit sample, pair and time fields."
                )
            if item["time"] not in spec.pair_times:
                reject("chart-pair-time", "Filter to the two selected time labels.")
            if item["pairKey"] in key_samples and key_samples[item["pairKey"]] != item["sampleId"]:
                reject("chart-pair-key-reused", "A pair key cannot identify different samples.")
            key_samples[item["pairKey"]] = item["sampleId"]
            pairs[item["sampleId"]].append(item)
        for sample_id, members in sorted(pairs.items()):
            if len(members) != 2 or {item["time"] for item in members} != set(spec.pair_times):
                reject(
                    "chart-pair-partner", "Each sample requires exactly one reading at each time."
                )
            for key in ("pairKey", "experimentalUnit", "group"):
                if len({item[key] for item in members}) != 1:
                    reject("chart-pair-incompatible", "Pair metadata must agree within a sample.")
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

    if spec.kind == "violin":
        for group in result["groups"]:
            items = groups[group["label"]]
            values = [float(item["value"]) for item in items]
            if len(values) < 3:
                reject(
                    "chart-density-small",
                    "Violin density needs at least three observations per group.",
                )
            sigma = stdev(values)
            if sigma == 0:
                reject(
                    "chart-density-degenerate",
                    "Constant data have no violin density; use raw points.",
                )
            bandwidth = spec.bandwidth or sigma * len(values) ** (-0.2)
            low, high = min(values) - 3 * bandwidth, max(values) + 3 * bandwidth
            if not all(math.isfinite(x) for x in (bandwidth, low, high)) or low >= high:
                reject("chart-density-range", "Rescale these values before density estimation.")
            span_in_bandwidths = (high - low) / bandwidth
            if not math.isfinite(span_in_bandwidths) or span_in_bandwidths > 512:
                reject(
                    "chart-density-resolution",
                    "Increase bandwidth to resolve this density reliably.",
                )
            grid_size = max(129, math.ceil(span_in_bandwidths * 4) + 1)
            grid = np.linspace(low, high, grid_size)
            density = np.zeros(grid_size)
            # Exact Gaussian sum in bounded chunks; no resampling or hidden data reduction.
            for start in range(0, len(values), 2048):
                with np.errstate(over="ignore", under="ignore"):
                    z = (
                        grid[:, None] - np.asarray(values[start : start + 2048])[None, :]
                    ) / bandwidth
                    density += np.exp(-0.5 * z * z).sum(axis=1)
            density /= len(values) * bandwidth * math.sqrt(2 * math.pi)
            if not np.isfinite(density).all() or float(density.max()) <= 0:
                reject(
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
                    else "scott-sample-sd",
                    "kernel": "gaussian",
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
            item["halfWidths"] = [0.4 * value / denominator for value in item["density"]]
            item["scale"] = spec.density_scale
    # A versioned compact encoding still binds the entire result, including all observations.
    # Old frozen chart hashes/artifacts are never rewritten.
    result["resultFingerprintVersion"] = "canonical-utf8-json-v2"
    result["resultSha256"] = result_fingerprint(result)
    return result


def _canonical(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _canonical(value[key]) for key in sorted(value)}
    if isinstance(value, list):
        return [_canonical(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("Fingerprint input must contain only finite numbers.")
    return value


def result_fingerprint(result: dict[str, Any]) -> str:
    # The large point/exclusion arrays already have canonical dictionaries from build_strip.
    # Canonicalize other method/result structures without duplicating all observation dictionaries.
    payload = {
        key: result[key] if key in {"points", "excluded"} else _canonical(result[key])
        for key in sorted(result)
        if key != "resultSha256"
    }
    return hashlib.sha256(to_json(payload, ensure_ascii=False, round_trip=True)).hexdigest()


def compute_chart(prepared: dict[str, Any], spec: ScientificChartSpec) -> dict[str, Any]:
    try:
        return _compute_chart(prepared, spec)
    except PreparationError:
        raise
    except (ArithmeticError, ValueError) as exc:
        raise PreparationError(
            "chart-numeric-range", "Rescale values before computing this chart."
        ) from exc
