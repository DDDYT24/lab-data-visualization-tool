"""Renderer-neutral raw-point geometry for full prepared research snapshots.

No sampling, averaging or independent-n inference. This module is not an API or renderer.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable
from functools import lru_cache
from operator import itemgetter
from typing import Any, Literal

from pydantic import Field

from .research_contract import ResearchModel
from .research_preparation import PreparationError


class StripSpec(ResearchModel):
    schema_version: Literal[1] = 1
    kind: Literal["grouped_strip"] = "grouped_strip"
    jitter_seed: str = Field(default="labviz-strip-v1", min_length=1, max_length=128)
    jitter_width: float = Field(default=0.3, ge=0, le=0.45)


@lru_cache(maxsize=16)
def field_order(names: tuple[str, ...]) -> tuple[tuple[str, ...], Callable[..., Any]]:
    keys = tuple(sorted(names))
    return keys, itemgetter(*keys) if keys else lambda item: ()


def canonical_copy(item: dict[str, Any]) -> dict[str, Any]:
    names = tuple(item)
    keys, getter = field_order(names)
    if names == keys:
        return item.copy()
    if len(keys) < 2:
        return {key: item[key] for key in keys}
    return dict(zip(keys, getter(item), strict=True))


def copy_observation(item: dict[str, Any]) -> dict[str, Any]:
    """Copy scalar metadata and mutable lineage without recursing into every scalar.

    Cached key order and C-level field extraction preserve canonical fingerprints.
    Every float is still checked; no metadata or source cell is discarded.
    """
    result = canonical_copy(item)
    for value in result.values():
        if isinstance(value, float) and not math.isfinite(value):
            raise PreparationError("chart-numeric-range", "Observation metadata must be finite.")
    lineage = []
    for cell in item["lineage"]:
        cloned = canonical_copy(cell)
        for value in cloned.values():
            if isinstance(value, float) and not math.isfinite(value):
                raise PreparationError("chart-numeric-range", "Source references must be finite.")
        lineage.append(cloned)
    result["lineage"] = lineage
    return result


def build_strip(prepared: dict[str, Any], spec: StripSpec) -> dict[str, Any]:
    """Use the complete prepare() result, never a paginated API view.

    Horizontal offset = width * (2 * first SHA256 uint32 / (2**32 - 1) - 1).
    Hash input is compact UTF-8 JSON [seed, observation ID]. Categories sort by
    Unicode code point, with unknown (null) first. Y is the unmodified value.
    """
    observations = prepared["observations"]
    if len(observations) != prepared["summary"]["observations"]:
        raise PreparationError("strip-incomplete-input", "Use the full prepared snapshot.")
    ids = [item["id"] for item in observations]
    if len(ids) != len(set(ids)):
        raise PreparationError("strip-duplicate-id", "Observation IDs must be unique.")
    included = [item for item in observations if item["exclusionReason"] is None]
    if not included:
        raise PreparationError("strip-empty", "No included observations to plot.")
    if any(item["state"] != "observed" for item in included):
        raise PreparationError(
            "strip-non-observed", "Review non-observed measurements and explicit exclusions."
        )
    if any(item["value"] is None or not math.isfinite(item["value"]) for item in included):
        raise PreparationError("strip-invalid-value", "Observed values must be finite.")
    units = {item["unit"] for item in included}
    if len(units) != 1:
        raise PreparationError("strip-mixed-units", "Convert units explicitly before plotting.")
    buckets: dict[str | None, list[dict[str, Any]]] = {}
    for item in included:
        buckets.setdefault(item["group"], []).append(item)
    labels = sorted({item["group"] for item in included}, key=lambda x: (x is not None, x or ""))
    positions = {label: index for index, label in enumerate(labels)}
    points = []
    # Same compact JSON bytes as json.dumps([seed, id]), without 100k encoder instances.
    prefix = ("[" + json.encoder.encode_basestring(spec.jitter_seed) + ",").encode("utf-8")
    for item in sorted(included, key=lambda item: item["id"]):
        encoded = prefix + json.encoder.encode_basestring(item["id"]).encode("utf-8") + b"]"
        fraction = int.from_bytes(hashlib.sha256(encoded).digest()[:4], "big") / (2**32 - 1)
        offset = spec.jitter_width * (2 * fraction - 1)
        points.append(
            {
                "jitterOffset": offset,
                "observation": copy_observation(item),
                "x": positions[item["group"]] + offset,
                "y": item["value"],
            }
        )
    groups = []
    for label in labels:
        members = buckets[label]
        known_units = {
            item["experimentalUnit"] for item in members if item["experimentalUnit"] is not None
        }
        unknown_units = sum(item["experimentalUnit"] is None for item in members)
        groups.append(
            {
                "label": label,
                "position": positions[label],
                "observations": len(members),
                "knownSamples": len(
                    {item["sampleId"] for item in members if item["sampleId"] is not None}
                ),
                "unknownSamples": sum(item["sampleId"] is None for item in members),
                "knownIndependentUnits": len(known_units),
                "unknownUnitObservations": unknown_units,
                "independentUnits": None if unknown_units else len(known_units),
            }
        )
    return {
        "schemaVersion": 1,
        "spec": spec.model_dump(),
        "unit": next(iter(units)),
        "groups": groups,
        "points": points,
        "inputObservations": len(observations),
        "excluded": [
            copy_observation(item) for item in observations if item["exclusionReason"] is not None
        ],
        "jitterMethod": "sha256-first-uint32-v1",
        "sampled": False,
    }
