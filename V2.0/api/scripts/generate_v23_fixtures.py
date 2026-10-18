"""Deterministic, synthetic Phase 0 inputs and independent answer keys."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import io
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2] / "contracts" / "fixtures" / "v23"


def build() -> dict[str, str]:
    samples = [
        {
            "id": f"s{group}{number}",
            "experimentalUnitId": f"u{group}{number}",
            "parentSampleId": None,
            "group": f"group-{group}",
        }
        for group in range(3)
        for number in range(1, 7)
    ]
    observations: list[dict[str, Any]] = []
    for group in range(3):
        for number in range(1, 7):
            for repeat in range(1, 4):
                observations.append(
                    {
                        "id": f"o{group}{number}{repeat}",
                        "sampleId": f"s{group}{number}",
                        "acquisitionId": "acq1" if number <= 3 else "acq2",
                        "timePoint": "baseline",
                        "technicalRepeat": repeat,
                        "pairKey": None,
                        "unit": "mV",
                        "state": "observed",
                        "value": float(group * 10 + number + repeat - 2),
                        "reportedText": str(group * 10 + number + repeat - 2),
                        "limit": None,
                    }
                )
    for number in range(1, 7):
        for time in range(2):
            observations.append(
                {
                    "id": f"p{number}{time}",
                    "sampleId": f"s0{number}",
                    "acquisitionId": f"acq{time + 1}",
                    "timePoint": f"t{time}",
                    "technicalRepeat": 1,
                    "pairKey": f"pair{number}",
                    "unit": "mV",
                    "state": "observed",
                    "value": float(number + 2 * time),
                    "reportedText": str(number + 2 * time),
                    "limit": None,
                }
            )
    for index, state in enumerate(
        [
            "observed",
            "missing",
            "not_measured",
            "failed",
            "below_detection",
            "below_quantification",
            "out_of_range",
            "saturated",
        ]
    ):
        observations.append(
            {
                "id": f"qc{index}",
                "sampleId": None,
                "acquisitionId": "acq2",
                "timePoint": None,
                "technicalRepeat": None,
                "pairKey": None,
                "unit": "mV",
                "state": state,
                "value": 0.0 if index == 0 else None,
                "reportedText": "0" if index == 0 else state,
                "limit": 0.5 if state.startswith("below_") else None,
            }
        )
    files: dict[str, str] = {}
    sources = []
    columns = list(observations[0])
    for batch in (1, 2):
        name = f"batch-{batch}.csv"
        stream = io.StringIO(newline="")
        writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        selected = [item for item in observations if item["acquisitionId"] == f"acq{batch}"]
        for row, observation in enumerate(selected, start=2):
            writer.writerow(observation)
            observation["lineage"] = [
                {"sourceId": f"source{batch}", "sheet": None, "row": row, "column": "value"}
            ]
        files[name] = stream.getvalue()
        sources.append(
            {
                "id": f"source{batch}",
                "name": name,
                "sha256": hashlib.sha256(files[name].encode()).hexdigest(),
            }
        )
    primary_ids = [item["id"] for item in observations[:54]]
    analysis = {
        "id": "analysis1",
        "dataRevision": "data1",
        "method": "descriptive",
        "methodVersion": "1",
        "parametersJson": '{"technicalRepeats":"mean-per-unit"}',
        "seed": 23,
        "analysisUnit": "experimental_unit",
        "inputObservationIds": primary_ids,
        "excluded": [
            {"observationId": f"qc{index}", "reason": "QC demonstration subset"}
            for index in range(8)
        ],
        "independentUnitCount": 18,
        "sampleCount": 18,
        "observationCount": 54,
        "resultJson": '{"groupMeans":[3.5,13.5,23.5],"overallMean":13.5}',
    }
    valid: dict[str, Any] = {
        "schemaVersion": 1,
        "id": "research1",
        "dataRevision": "data1",
        "sources": sources,
        "samples": samples,
        "acquisitions": [
            {"id": f"acq{batch}", "batch": f"batch{batch}", "protocol": None} for batch in (1, 2)
        ],
        "observations": observations,
        "processing": [],
        "analyses": [analysis],
        "figure": {
            "schemaVersion": 1,
            "id": "figure1",
            "revision": "figure-revision1",
            "panels": [
                {
                    "id": key,
                    "chartType": kind,
                    "dataRevision": "data1",
                    "analysisId": "analysis1",
                    "xField": "timePoint",
                    "yField": "value",
                    "xUnit": "category",
                    "yUnit": "mV",
                    "jitterSeed": 23,
                }
                for key, kind in zip("ABCD", ("line", "scatter", "strip", "box"), strict=True)
            ],
            "axisLinks": [{"axis": "y", "panelIds": ["A", "B"]}],
            "styleId": "labviz-default",
        },
    }
    invalid = []

    def case(name: str, path: list[str | int], replacement: Any) -> None:
        value = copy.deepcopy(valid)
        target: Any = value
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = replacement
        invalid.append({"case": name, "value": value})

    case("duplicate_id", ["samples", 1, "id"], "s01")
    case("invalid_sample_parent", ["samples", 0, "parentSampleId"], "s01")
    case("unknown_sample", ["observations", 0, "sampleId"], "absent")
    case("unknown_acquisition", ["observations", 0, "acquisitionId"], "absent")
    case("unknown_source", ["observations", 0, "lineage", 0, "sourceId"], "absent")
    case("observed_requires_value", ["observations", 0, "value"], None)
    case("absent_state_requires_null", ["observations", 67, "value"], 0)
    case("qualified_state_requires_limit", ["observations", 70, "limit"], None)
    case("invalid_analysis_inputs", ["analyses", 0, "dataRevision"], "other")
    case("unsupported_measurement_state", ["observations", 0, "state"], "saturated")
    case("incorrect_analysis_counts", ["analyses", 0, "independentUnitCount"], 54)
    case("unknown_experimental_unit", ["samples", 0, "experimentalUnitId"], None)
    case("invalid_panel_layout", ["figure", "panels"], valid["figure"]["panels"][:3])
    case("unknown_panel_revision", ["figure", "panels", 0, "dataRevision"], "other")
    case("unknown_analysis", ["figure", "panels", 0, "analysisId"], "absent")
    case("duplicate_axis_panel", ["figure", "axisLinks", 0, "panelIds"], ["A", "A"])
    case("incompatible_axis_units", ["figure", "panels", 1, "yUnit"], "V")
    case("unsupported_schema_version", ["schemaVersion"], 2)
    case("invalid_source_row", ["observations", 0, "lineage", 0, "row"], 0)
    legacy = copy.deepcopy(valid)
    legacy["samples"] = []
    for observation in legacy["observations"]:
        observation["sampleId"] = None
    legacy["analyses"][0].update(
        analysisUnit="observation",
        independentUnitCount=None,
        sampleCount=0,
    )
    manifest = {
        "schemaVersion": 1,
        "synthetic": True,
        "generator": "generate_v23_fixtures.py",
        "sources": sources,
        "primary": {
            "independentUnits": 18,
            "samples": 18,
            "technicalReadingsPerSample": 3,
            "observations": 54,
            "batches": 2,
            "perGroupUnits": 6,
            "perGroupObservations": 18,
            "groupMeans": [3.5, 13.5, 23.5],
            "overallMean": 13.5,
        },
        "paired": {
            "independentUnits": 6,
            "observations": 12,
            "timePoints": ["t0", "t1"],
            "everyDelta": 2.0,
            "meanDelta": 2.0,
            "failureCases": ["duplicate pair/time key", "missing partner", "mixed units"],
        },
        "qc": {"observations": 8, "observedZero": 1, "nonObserved": 7},
        "totalObservations": 74,
        "jitterSeed": 23,
        "legacyUnknown": {"independentUnits": None, "sampleCount": 0, "observations": 54},
    }
    for name, value in [
        ("research.valid.json", valid),
        ("research.invalid.json", invalid),
        ("research.unknown-identity.json", legacy),
        ("manifest.json", manifest),
    ]:
        if name == "research.invalid.json":
            files[name] = json.dumps(value, separators=(",", ":")) + "\n"
        else:
            files[name] = json.dumps(value, indent=2) + "\n"
    return files


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    files = build()
    if args.check:
        for name, content in files.items():
            if (ROOT / name).read_bytes() != content.encode():
                raise SystemExit(f"Fixture drift: {name}")
        print(f"{len(files)} deterministic fixture files match.")
    else:
        ROOT.mkdir(parents=True, exist_ok=True)
        for name, content in files.items():
            (ROOT / name).write_bytes(content.encode())


if __name__ == "__main__":
    main()
