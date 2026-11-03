from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from labviz_api.research_charts import ScientificChartSpec, compute_chart
from labviz_api.research_preparation import (
    PreparationError,
    PreparationPlan,
    parse_sources,
    prepare,
)

FIXTURES = Path(__file__).resolve().parents[2] / "contracts/fixtures/v23"


def prepared_csv(body: str, **mapping: str) -> dict[str, Any]:
    return prepare(
        parse_sources("chart-test", [("test.csv", body.encode())]),
        PreparationPlan.model_validate({"mapping": {"value": "v", **mapping}}),
    )


def paired_data() -> dict[str, Any]:
    return prepared_csv(
        "s,p,t,v,g,u\na,p1,before,1,A,mV\na,p1,after,3,A,mV\n"
        "b,p2,before,4,B,mV\nb,p2,after,6,B,mV\n",
        sample="s",
        experimentalUnit="s",
        pair="p",
        time="t",
        group="g",
        unit="u",
    )


def test_schema_matches_checked_in_contract() -> None:
    expected = ScientificChartSpec.model_json_schema(by_alias=True)
    assert (
        json.loads((FIXTURES.parents[1] / "scientific-chart-v1.schema.json").read_text())
        == expected
    )


def test_fixture_group_unit_means_and_density_retain_all_observations() -> None:
    raw = parse_sources(
        "answer-key",
        [(name, (FIXTURES / name).read_bytes()) for name in ("batch-1.csv", "batch-2.csv")],
    )
    plan = PreparationPlan.model_validate(
        {
            "mapping": {
                "value": "value",
                "sample": "sampleId",
                "experimentalUnit": "sampleId",
                "technicalRepeat": "technicalRepeat",
                "unit": "unit",
            },
            "steps": [{"kind": "filter", "field": "timePoint", "values": ["baseline"]}],
        }
    )
    prepared = prepare(raw, plan)
    inventory = json.loads((FIXTURES / "research.valid.json").read_text())
    declared_groups = {item["id"]: item["group"] for item in inventory["samples"]}
    for observation in prepared["observations"]:
        observation["group"] = declared_groups[observation["sampleLabel"]]
    original = copy.deepcopy(prepared)
    result = compute_chart(prepared, ScientificChartSpec(summary="mean"))
    assert sorted(g["summary"] for g in result["groups"]) == [3.5, 13.5, 23.5]
    assert all(g["summaryN"] == g["independentUnits"] == 6 for g in result["groups"])
    assert len(result["points"]) == 54
    assert prepared == original


def test_gaussian_density_has_independent_analytic_answer_and_recorded_support() -> None:
    prepared = prepared_csv("v\n0\n1\n2\n")
    result = compute_chart(prepared, ScientificChartSpec(kind="violin", bandwidth=1.0))
    d = result["densities"][0]
    assert d["support"] == [-3, 5]
    assert d["grid"][64] == 1
    assert d["density"][64] == pytest.approx(
        (1 + 2 * math.exp(-0.5)) / (3 * math.sqrt(2 * math.pi))
    )
    integral = sum(
        (a + b) / 2 * (y - x)
        for a, b, x, y in zip(
            d["density"][:-1], d["density"][1:], d["grid"][:-1], d["grid"][1:], strict=True
        )
    )
    assert 0.998 < integral < 1
    assert max(d["halfWidths"]) == pytest.approx(0.4)
    assert d["observationIds"] == [p["observation"]["id"] for p in result["points"]]
    prepared["observations"].reverse()
    assert compute_chart(prepared, ScientificChartSpec(kind="violin", bandwidth=1.0)) == result


def test_scott_and_common_density_scale() -> None:
    prepared = prepared_csv("v,g\n0,A\n1,A\n2,A\n0,B\n10,B\n20,B\n", group="g")
    result = compute_chart(
        prepared, ScientificChartSpec(kind="violin", density_scale="common_density")
    )
    a, b = result["densities"]
    assert a["bandwidth"] == pytest.approx(3**-0.2)
    assert b["bandwidth"] == pytest.approx(10 * 3**-0.2)
    assert max(a["halfWidths"]) == pytest.approx(0.4)
    assert max(b["halfWidths"]) == pytest.approx(0.04)


def test_pairs_use_sample_identity_and_explicit_time_order() -> None:
    prepared = paired_data()
    result = compute_chart(
        prepared, ScientificChartSpec(kind="paired", pair_times=["before", "after"])
    )
    assert [p["delta"] for p in result["pairs"]] == [2, 2]
    assert len({p["sampleId"] for p in result["pairs"]}) == 2
    prepared["observations"].reverse()
    assert (
        compute_chart(prepared, ScientificChartSpec(kind="paired", pair_times=["before", "after"]))
        == result
    )
    reversed_result = compute_chart(
        prepared, ScientificChartSpec(kind="paired", pair_times=["after", "before"])
    )
    assert [p["delta"] for p in reversed_result["pairs"]] == [-2, -2]


@pytest.mark.parametrize(
    "fault", ["missing", "duplicate", "key", "sample", "time", "unit", "group"]
)
def test_pair_failures_do_not_invent_partners(fault: str) -> None:
    prepared = paired_data()
    first = prepared["observations"][0]
    if fault == "missing":
        first["exclusionReason"] = "reviewed"
    elif fault == "duplicate":
        first["time"] = "after"
    elif fault == "key":
        first["pairKey"] = "p2"
    elif fault == "sample":
        first["sampleId"] = None
    elif fault == "time":
        first["time"] = "unknown"
    elif fault == "unit":
        first["unit"] = "V"
    else:
        first["group"] = "wrong"
    with pytest.raises(PreparationError):
        compute_chart(prepared, ScientificChartSpec(kind="paired", pair_times=["before", "after"]))


@pytest.mark.parametrize(
    "values,code", [("1\n2", "chart-density-small"), ("1\n1\n1", "chart-density-degenerate")]
)
def test_invalid_density_is_actionable(values: str, code: str) -> None:
    with pytest.raises(PreparationError) as error:
        compute_chart(prepared_csv("v\n" + values + "\n"), ScientificChartSpec(kind="violin"))
    assert error.value.code == code


def test_nonobserved_and_unknown_or_ambiguous_summary_units_are_rejected() -> None:
    data = prepared_csv("v\n1\n2\n3\n")
    with pytest.raises(PreparationError, match="identities"):
        compute_chart(data, ScientificChartSpec(summary="mean"))
    data["observations"][0]["state"] = "below_detection"
    with pytest.raises(PreparationError, match="non-observed"):
        compute_chart(data, ScientificChartSpec(kind="violin"))
    data = paired_data()
    with pytest.raises(PreparationError, match="technical repeats"):
        compute_chart(data, ScientificChartSpec(summary="median"))


@pytest.mark.parametrize(
    "body",
    [
        {"kind": "paired"},
        {"kind": "paired", "pairTimes": ["x", "x"]},
        {"bandwidth": 1.0},
        {"kind": "violin", "bandwidth": 0.0},
        {"kind": "violin", "bandwidth": float("nan")},
        {"formula": "execute"},
    ],
)
def test_invalid_spec(body: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        ScientificChartSpec.model_validate(body)


def test_unresolvable_density_bandwidth_and_overflow_are_actionable() -> None:
    with pytest.raises(PreparationError, match="bandwidth"):
        compute_chart(
            prepared_csv("v\n0\n1\n2\n"), ScientificChartSpec(kind="violin", bandwidth=1e-20)
        )
    data = paired_data()
    data["observations"][0]["value"] = -1e308
    data["observations"][1]["value"] = 1e308
    with pytest.raises(PreparationError, match="Rescale"):
        compute_chart(data, ScientificChartSpec(kind="paired", pair_times=["before", "after"]))


def test_category_style_survives_subset_chart_and_keeps_explicit_override() -> None:
    both = compute_chart(prepared_csv("v,g\n1,A\n2,B\n3,B\n", group="g"), ScientificChartSpec())
    subset = compute_chart(prepared_csv("v,g\n2,B\n3,B\n", group="g"), ScientificChartSpec())
    assert (
        next(style for style in both["spec"]["categoryStyles"] if style["group"] == "B")
        == subset["spec"]["categoryStyles"][0]
    )
    explicit = compute_chart(
        prepared_csv("v,g\n2,B\n3,B\n", group="g"),
        ScientificChartSpec.model_validate(
            {"categoryStyles": [{"group": "B", "color": "#abcdef", "marker": "D"}]}
        ),
    )
    assert explicit["spec"]["categoryStyles"] == [{"group": "B", "color": "#abcdef", "marker": "D"}]


def test_versioned_compiled_fingerprint_binds_complete_result_without_order_or_float_loss() -> None:
    import hashlib

    from pydantic_core import to_json

    from labviz_api.research_charts import result_fingerprint

    prepared = prepared_csv("v\n0\n1e-7\n1.2345678901234567\n")
    first = compute_chart(prepared, ScientificChartSpec())
    assert first["resultFingerprintVersion"] == "canonical-utf8-json-v2"
    without_hash = {key: value for key, value in first.items() if key != "resultSha256"}

    def canonical(value: Any) -> Any:
        if isinstance(value, dict):
            return {key: canonical(value[key]) for key in sorted(value)}
        if isinstance(value, list):
            return [canonical(item) for item in value]
        return value

    encoded = to_json(canonical(without_hash), round_trip=True)
    assert json.loads(encoded) == without_hash
    assert hashlib.sha256(encoded).hexdigest() == first["resultSha256"]
    assert (
        compute_chart(json.loads(json.dumps(prepared, sort_keys=True)), ScientificChartSpec())
        == first
    )
    # Reordering dictionary keys and observations does not change the new hash.
    prepared["observations"] = [
        dict(reversed(list(item.items()))) for item in reversed(prepared["observations"])
    ]
    assert compute_chart(prepared, ScientificChartSpec()) == first
    for path in ["reportedText", "notes", "exclusionReason", "value", "limit"]:
        changed = copy.deepcopy(first)
        changed["points"][0]["observation"][path] = 123 if path in {"value", "limit"} else "changed"
        assert result_fingerprint(changed) != first["resultSha256"]
    for key in ["spec", "sources", "densities", "pairs", "groups"]:
        changed = copy.deepcopy(first)
        changed[key] = {"changed": True}
        assert result_fingerprint(changed) != first["resultSha256"]


def test_native_frozen_decode_keeps_full_values_and_fingerprint() -> None:
    import json

    from pydantic_core import from_json, to_json

    prepared = prepared_csv("v\n-0.0\n1.0000000000000002\n1e-308\n1e100\n")
    result = compute_chart(prepared, ScientificChartSpec())
    serialized = to_json(result, ensure_ascii=False, round_trip=True)
    native = from_json(serialized, cache_strings="keys", allow_inf_nan=False)
    assert native == json.loads(serialized)
    from labviz_api.research_charts import result_fingerprint

    assert result_fingerprint(native) == result["resultSha256"]
