"""Versioned, renderer-neutral scientific chart results; no renderer recomputes statistics."""

from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from statistics import mean
from typing import Any, Literal

from pydantic import Field, model_validator
from pydantic_core import to_json

from .engine.layers import LAYERS, LayerKind
from .engine.stats import SUMMARIES, SummaryChoice
from .research_contract import ResearchModel
from .research_preparation import PreparationError
from .research_strip import JITTER, StripSpec, build_strip
from .research_styles import category_style

MAX_GROUPS = 100
# The spec fields that only some kinds take.
KIND_OPTIONS = sorted({option for layer in LAYERS.values() for option in layer.options})


class CategoryStyle(ResearchModel):
    group: str | None
    color: str = Field(pattern=r"^#[0-9A-Fa-f]{6}$")
    marker: Literal["o", "s", "^", "D", "v"] = "o"


class ScientificChartSpec(ResearchModel):
    schema_version: Literal[1] = 1
    point_rendering: Literal["auto", "vector", "raster"] = "auto"
    kind: LayerKind = "grouped_strip"
    title: str = Field(default="", max_length=160)
    locale: Literal["en", "zh"] = "en"
    jitter_seed: str = Field(default=JITTER["defaultSeed"], min_length=1, max_length=128)
    jitter_width: float = Field(default=JITTER["defaultWidth"], ge=0, le=JITTER["maxWidth"])
    show_points: bool = True
    summary: SummaryChoice = "none"
    summary_unit: Literal["observation", "sample", "experimental_unit"] = "experimental_unit"
    pair_times: list[str] = Field(default_factory=list, max_length=2)
    bandwidth: float | None = Field(default=None, gt=0)
    density_scale: Literal["equal_width", "common_density"] = "equal_width"
    category_styles: list[CategoryStyle] = Field(default_factory=list, max_length=100)

    @model_validator(mode="after")
    def check_options(self) -> ScientificChartSpec:
        if len({item.group for item in self.category_styles}) != len(self.category_styles):
            raise ValueError("Category style keys must be unique.")
        layer = LAYERS[self.kind]
        layer.check(self)
        for name in KIND_OPTIONS:
            field = type(self).model_fields[name]
            unused = field.get_default(call_default_factory=True)
            if name not in layer.options and getattr(self, name) != unused:
                raise ValueError(f"{field.alias} does not apply to this chart kind.")
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
    if len(result["groups"]) > MAX_GROUPS:
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
            group["summary"] = SUMMARIES[spec.summary].compute(values)
            group["summaryN"] = len(values)
    LAYERS[spec.kind].compute(result, groups, spec)
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
