from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest

from labviz_api.research_preparation import (
    PreparationError,
    PreparationPlan,
    parse_sources,
    prepare,
)

FIXTURES = Path(__file__).resolve().parents[2] / "contracts/fixtures/v23"


def data() -> dict[str, Any]:
    return parse_sources(
        "project",
        [(name, (FIXTURES / name).read_bytes()) for name in ("batch-1.csv", "batch-2.csv")],
    )


def plan(**updates: object) -> PreparationPlan:
    body: dict[str, Any] = {
        "mapping": {
            "value": "value",
            "sample": "sampleId",
            "experimentalUnit": "sampleId",
            "unit": "unit",
            "state": "state",
            "limit": "limit",
            "time": "timePoint",
            "technicalRepeat": "technicalRepeat",
            "pair": "pairKey",
        },
        "analysisUnit": "experimental_unit",
        "steps": [{"kind": "filter", "field": "timePoint", "values": ["baseline"]}],
    }
    body.update(updates)
    return PreparationPlan.model_validate(body)


def test_primary_counts_mean_and_lineage_survive_filter_and_conversion() -> None:
    raw = data()
    original = copy.deepcopy(raw)
    prepared = prepare(raw, plan())
    assert prepared["summary"]["independentUnits"] == 18
    assert prepared["summary"]["observations"] == 54
    assert prepared["summary"]["analysisN"] == 18
    assert prepared["summary"]["mean"] == 13.5
    converted_plan = plan()
    converted_plan = PreparationPlan.model_validate(
        {
            **converted_plan.model_dump(by_alias=True),
            "steps": [
                *converted_plan.model_dump(by_alias=True)["steps"],
                {
                    "kind": "convert_unit",
                    "valueField": "value",
                    "unitField": "unit",
                    "limitField": "limit",
                    "toUnit": "V",
                },
            ],
        }
    )
    converted = prepare(raw, converted_plan)
    assert converted["summary"]["mean"] == pytest.approx(0.0135)
    assert [item["id"] for item in converted["observations"]] == [
        item["id"] for item in prepared["observations"]
    ]
    assert converted["observations"][0]["lineage"] == prepared["observations"][0]["lineage"]
    assert raw == original


def test_one_group_is_six_units_eighteen_observations() -> None:
    p = plan()
    p = PreparationPlan.model_validate(
        {
            **p.model_dump(by_alias=True),
            "steps": [
                *p.model_dump(by_alias=True)["steps"],
                {"kind": "filter", "field": "sampleId", "values": [f"s0{i}" for i in range(1, 7)]},
            ],
        }
    )
    result = prepare(data(), p)["summary"]
    assert (result["independentUnits"], result["observations"], result["analysisN"]) == (6, 18, 6)
    assert result["mean"] == 3.5


def test_qc_zero_is_retained_and_qualified_analysis_is_withheld() -> None:
    p = plan(steps=[], analysisUnit="observation")
    result = prepare(data(), p)
    assert result["summary"]["states"]["observed"] == 67
    assert result["summary"]["independentUnits"] is None
    assert result["summary"]["mean"] is None
    assert result["summary"]["withheldReasons"]
    assert any(
        item["state"] == "observed" and item["value"] == 0 for item in result["observations"]
    )
    assert all(
        item["value"] is None
        for item in result["observations"]
        if item["state"].startswith("below")
    )


def test_explicit_pairing_has_known_delta_and_missing_partner_is_rejected() -> None:
    p = plan(
        steps=[{"kind": "filter", "field": "timePoint", "values": ["t0", "t1"]}],
        analysisUnit="observation",
        validatePairs=True,
        pairTimes=["t0", "t1"],
    )
    result = prepare(data(), p)
    assert result["summary"]["pairDeltas"] == [2] * 6
    broken = data()
    broken["rows"] = [row for row in broken["rows"] if row["cells"]["id"] != "p10"]
    with pytest.raises(PreparationError, match="one reading"):
        prepare(broken, p)


def test_wide_long_roundtrip_preserves_values_and_source_cells() -> None:
    raw = parse_sources("p", [("wide.csv", b"sample,t0,t1\na,1,3\nb,2,4\n")])
    p = PreparationPlan.model_validate(
        {
            "mapping": {"sample": "sample", "value": "reading", "time": "time"},
            "steps": [
                {
                    "kind": "wide_to_long",
                    "fields": ["t0", "t1"],
                    "nameField": "time",
                    "valueField": "reading",
                }
            ],
        }
    )
    result = prepare(raw, p)
    assert [item["value"] for item in result["observations"]] == [1, 3, 2, 4]
    assert result["observations"][1]["lineage"][0]["column"] == "t1"
    reverse = PreparationPlan.model_validate(
        {
            "mapping": {"sample": "sample", "value": "t1"},
            "steps": [
                *p.model_dump(by_alias=True)["steps"],
                {
                    "kind": "long_to_wide",
                    "indexFields": ["sample"],
                    "nameField": "time",
                    "valueField": "reading",
                },
            ],
        }
    )
    restored = prepare(raw, reverse)
    assert [item["value"] for item in restored["observations"]] == [3, 4]
    assert restored["observations"][0]["lineage"][0]["column"] == "t1"


@pytest.mark.parametrize(
    ("files", "code"),
    [
        ([("a.csv", b"x,x\n1,2\n")], "invalid-columns"),
        ([("a.csv", b"x\n1\n"), ("b.csv", b"y\n2\n")], "schema-mismatch"),
        ([("a.csv", b"x\n1\n"), ("b.csv", b"x\n1\n")], "duplicate-source"),
        ([("a.csv", b"x,y\n1\n")], "row-width"),
        ([("a.json", b"{}")], "unsupported-research-format"),
        ([("a.csv", b"x\n1\n")] * 6, "file-count-limit"),
    ],
)
def test_import_failures(files: list[tuple[str, bytes]], code: str) -> None:
    with pytest.raises(PreparationError) as error:
        parse_sources("p", files)
    assert error.value.code == code


def test_unknown_identity_is_not_invented_and_conflicts_are_rejected() -> None:
    raw = parse_sources("p", [("quick.csv", b"value\n0\n2\n")])
    summary = prepare(raw, PreparationPlan.model_validate({"mapping": {"value": "value"}}))[
        "summary"
    ]
    assert summary["independentUnits"] is None
    assert summary["samples"] == 0
    assert summary["mean"] == 1
    raw = parse_sources("p", [("conflict.csv", b"s,u,v\ns1,u1,1\ns1,u2,2\n")])
    p = PreparationPlan.model_validate(
        {"mapping": {"value": "v", "sample": "s", "experimentalUnit": "u"}}
    )
    with pytest.raises(PreparationError) as error:
        prepare(raw, p)
    assert error.value.code == "conflicting-sample"


def test_exclusion_requires_reason_and_changes_only_explicit_revision_input() -> None:
    raw = parse_sources("p", [("data.csv", b"v,state,limit\n0,observed,\n,below_detection,1\n")])
    p = PreparationPlan.model_validate(
        {"mapping": {"value": "v", "state": "state", "limit": "limit"}}
    )
    before = prepare(raw, p)
    assert before["summary"]["mean"] is None
    p = PreparationPlan.model_validate(
        {
            **p.model_dump(by_alias=True),
            "exclusions": [
                {
                    "observationId": raw["rows"][1]["id"],
                    "reason": "No censored-data method selected",
                }
            ],
        }
    )
    after = prepare(raw, p)
    assert after["summary"]["mean"] == 0
    assert after["summary"]["excluded"] == 1
    assert before["summary"]["excluded"] == 0


def test_incompatible_units_and_duplicate_pivot_are_errors() -> None:
    raw = parse_sources("p", [("data.csv", b"s,t,v,u\na,t0,1,mV\na,t0,2,s\n")])
    p = PreparationPlan.model_validate(
        {
            "mapping": {"value": "v"},
            "steps": [{"kind": "convert_unit", "valueField": "v", "unitField": "u", "toUnit": "V"}],
        }
    )
    with pytest.raises(PreparationError) as error:
        prepare(raw, p)
    assert error.value.code == "incompatible-unit"
    p = PreparationPlan.model_validate(
        {
            "mapping": {"value": "t0"},
            "steps": [
                {"kind": "long_to_wide", "indexFields": ["s"], "nameField": "t", "valueField": "v"}
            ],
        }
    )
    with pytest.raises(PreparationError) as error:
        prepare(raw, p)
    assert error.value.code == "duplicate-pivot-key"


def test_original_report_is_not_relabelled_as_converted_units() -> None:
    p = PreparationPlan.model_validate(
        {
            "mapping": {"value": "value", "unit": "unit"},
            "steps": [
                {"kind": "convert_unit", "valueField": "value", "unitField": "unit", "toUnit": "V"}
            ],
        }
    )
    raw = parse_sources("p", [("a.csv", b"value,unit\n2,mV\n")])
    observation = prepare(raw, p)["observations"][0]
    assert observation["value"] == 0.002
    assert observation["unit"] == "V"
    assert observation["reportedText"] == "2"
    assert observation["reportedUnit"] == "mV"


def test_qualified_text_and_filtered_parent_inventory_are_preserved() -> None:
    raw = parse_sources(
        "p",
        [("a.csv", b"s,parent,value,state,limit\np,,1,observed,\nc,p,<0.5,below_detection,0.5\n")],
    )
    p = PreparationPlan.model_validate(
        {
            "mapping": {
                "value": "value",
                "sample": "s",
                "parentSample": "parent",
                "state": "state",
                "limit": "limit",
            },
            "steps": [{"kind": "filter", "field": "s", "values": ["c"]}],
        }
    )
    result = prepare(raw, p)
    assert result["samples"][0]["parentSample"] == "p"
    assert result["observations"][0]["reportedText"] == "<0.5"
    assert result["observations"][0]["value"] is None
    assert result["summary"]["mean"] is None


def test_preparation_plans_reject_executable_payloads() -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        PreparationPlan.model_validate(
            {"mapping": {"value": "x"}, "steps": [{"kind": "execute", "code": "x"}]}
        )


def test_limits_are_rejected_before_parsing_or_reshape(monkeypatch: pytest.MonkeyPatch) -> None:
    from labviz_api.research_preparation import LIMITS

    monkeypatch.setitem(LIMITS, "singleFileBytes", 3)
    with pytest.raises(PreparationError) as error:
        parse_sources("p", [("a.csv", b"x\n1\n")])
    assert error.value.code == "single-file-limit"
    monkeypatch.setitem(LIMITS, "singleFileBytes", 1000)
    monkeypatch.setitem(LIMITS, "aggregateRows", 1)
    with pytest.raises(PreparationError) as error:
        parse_sources("p", [("a.csv", b"x\n1\n2\n")])
    assert error.value.code == "row-limit"


def test_generated_preparation_schema_matches_backend() -> None:
    import json

    path = Path(__file__).resolve().parents[2] / "contracts/preparation-plan-v1.schema.json"
    assert json.loads(path.read_text(encoding="utf-8")) == PreparationPlan.model_json_schema()


def test_batch_is_unknown_until_declared_and_unmapped_values_cannot_be_relabelled() -> None:
    raw = parse_sources("p", [("batch-name.csv", b"value,other,unit\n2,3,mV\n")])
    p = PreparationPlan.model_validate({"mapping": {"value": "value", "unit": "unit"}})
    assert prepare(raw, p)["observations"][0]["batch"] is None
    p = PreparationPlan.model_validate(
        {
            "mapping": {"value": "value", "unit": "unit"},
            "steps": [
                {"kind": "convert_unit", "valueField": "other", "unitField": "unit", "toUnit": "V"}
            ],
        }
    )
    with pytest.raises(PreparationError) as error:
        prepare(raw, p)
    assert error.value.code == "unit-mapping-mismatch"
