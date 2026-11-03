from __future__ import annotations

import copy
import json
from typing import Any

import pytest
from pydantic import ValidationError

from labviz_api.research_preparation import (
    PreparationError,
    PreparationPlan,
    parse_sources,
    prepare,
)
from labviz_api.research_strip import StripSpec, build_strip


def fixture() -> dict[str, Any]:
    rows = ["sample,group,repeat,value,unit"]
    rows += [f"s{s},A,{r},{s + r - 2},mV" for s in range(1, 7) for r in range(1, 4)]
    return prepare(
        parse_sources("strip-test", [("readings.csv", ("\n".join(rows) + "\n").encode())]),
        PreparationPlan.model_validate(
            {
                "mapping": {
                    "value": "value",
                    "sample": "sample",
                    "experimentalUnit": "sample",
                    "group": "group",
                    "technicalRepeat": "repeat",
                    "unit": "unit",
                }
            }
        ),
    )


def test_six_units_eighteen_raw_points_zero_and_lineage() -> None:
    source = fixture()
    original = copy.deepcopy(source)
    chart = build_strip(source, StripSpec())
    assert len(chart["points"]) == 18
    assert chart["groups"][0]["independentUnits"] == 6
    assert sorted(p["y"] for p in chart["points"]) == sorted(
        [s + r - 2 for s in range(1, 7) for r in range(1, 4)]
    )
    assert min(p["y"] for p in chart["points"]) == 0
    assert all(-0.3 <= p["x"] <= 0.3 for p in chart["points"])
    assert source == original
    chart["points"][0]["observation"]["lineage"].clear()
    assert source == original


def test_jitter_is_stable_under_reordering_and_records_seed() -> None:
    source = fixture()
    first = build_strip(source, StripSpec())
    source["observations"].reverse()
    assert build_strip(source, StripSpec()) == first
    other = build_strip(source, StripSpec(jitter_seed="other"))
    assert [p["x"] for p in other["points"]] != [p["x"] for p in first["points"]]
    assert [p["y"] for p in other["points"]] == [p["y"] for p in first["points"]]
    assert all(p["x"] == 0 for p in build_strip(source, StripSpec(jitter_width=0.0))["points"])


def test_exclusions_and_unknown_units_are_disclosed() -> None:
    source = fixture()
    source["observations"][0]["exclusionReason"] = "instrument failure"
    source["observations"][1]["experimentalUnit"] = None
    chart = build_strip(source, StripSpec())
    assert len(chart["points"]) == 17
    assert chart["excluded"][0]["exclusionReason"] == "instrument failure"
    assert chart["groups"][0]["independentUnits"] is None
    assert chart["groups"][0]["unknownUnitObservations"] == 1


@pytest.mark.parametrize(
    ("change", "code"),
    [
        ("state", "strip-non-observed"),
        ("unit", "strip-mixed-units"),
        ("id", "strip-duplicate-id"),
        ("value", "strip-invalid-value"),
        ("truncated", "strip-incomplete-input"),
        ("empty", "strip-empty"),
    ],
)
def test_invalid_input_never_silently_drops_or_aggregates(change: str, code: str) -> None:
    source = fixture()
    if change == "truncated":
        source["observations"].pop()
    elif change == "empty":
        for item in source["observations"]:
            item["exclusionReason"] = "reviewed"
    else:
        source["observations"][0][change] = {
            "state": "missing",
            "unit": "V",
            "id": source["observations"][1]["id"],
            "value": float("nan"),
        }[change]
    with pytest.raises(PreparationError) as exc:
        build_strip(source, StripSpec())
    assert exc.value.code == code


def test_unknown_group_is_not_a_real_group_named_unknown() -> None:
    source = fixture()
    source["observations"][0]["group"] = None
    source["observations"][1]["group"] = "Unknown"
    chart = build_strip(source, StripSpec())
    assert [g["label"] for g in chart["groups"]] == [None, "A", "Unknown"]


def test_spec_rejects_unbounded_jitter_and_unknown_options() -> None:
    for body in [{"jitterWidth": 0.5}, {"jitterWidth": float("nan")}, {"aggregate": True}]:
        with pytest.raises(ValidationError):
            StripSpec.model_validate(body)


@pytest.mark.parametrize("sorted_keys", [False, True])
def test_snapshot_copy_preserves_shape_and_nested_mutation_isolation(sorted_keys: bool) -> None:
    import copy

    source = fixture()
    if sorted_keys:
        source = json.loads(json.dumps(source, sort_keys=True))
    before = copy.deepcopy(source)
    result = build_strip(source, StripSpec())
    for point in result["points"]:
        obs = point["observation"]
        assert all(
            k == "lineage" or isinstance(v, (str, int, float, type(None))) for k, v in obs.items()
        )
        assert all(
            isinstance(v, (str, int, float, type(None)))
            for cell in obs["lineage"]
            for v in cell.values()
        )
        original = next(o for o in before["observations"] if o["id"] == obs["id"])
        assert obs == original
    result["points"][0]["observation"]["lineage"][0]["column"] = "mutated"
    result["points"][0]["observation"]["value"] = -999
    assert source == before


@pytest.mark.parametrize("seed", ["plain", "姹夊瓧馃榾", 'quote"\\line\n\t', "\x00\b\f"])
def test_optimized_jitter_keeps_original_json_sha_bytes(seed: str) -> None:
    import hashlib
    import json

    prepared = fixture()
    prepared["observations"][0]["id"] = 'id姹夊瓧馃榾"\\\n\t\x00'
    result = build_strip(prepared, StripSpec(jitter_seed=seed))
    for point in result["points"]:
        encoded = json.dumps(
            [seed, point["observation"]["id"]], ensure_ascii=False, separators=(",", ":")
        ).encode("utf-8")
        fraction = int.from_bytes(hashlib.sha256(encoded).digest()[:4], "big") / (2**32 - 1)
        assert point["jitterOffset"] == 0.3 * (2 * fraction - 1)
