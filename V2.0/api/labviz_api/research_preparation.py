"""Bounded, deterministic research preparation; raw cells are never overwritten."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
from collections import Counter, defaultdict
from decimal import Decimal, InvalidOperation
from pathlib import Path
from statistics import mean
from typing import Annotated, Any, Literal
from uuid import NAMESPACE_URL, uuid5

from pydantic import Field

from .research_contract import ResearchModel

LIMITS = json.loads(
    (Path(__file__).resolve().parents[2] / "contracts/v23-workflow-v1.json").read_text()
)["limits"]
STATES = {
    "observed",
    "missing",
    "not_measured",
    "failed",
    "below_detection",
    "below_quantification",
    "out_of_range",
    "saturated",
}
UNITS = {
    "V": ("voltage", "1"),
    "mV": ("voltage", "0.001"),
    "uV": ("voltage", "0.000001"),
    "s": ("time", "1"),
    "min": ("time", "60"),
    "h": ("time", "3600"),
    "g": ("mass", "1"),
    "mg": ("mass", "0.001"),
    "ug": ("mass", "0.000001"),
    "L": ("volume", "1"),
    "mL": ("volume", "0.001"),
    "uL": ("volume", "0.000001"),
    "mol/L": ("concentration", "1"),
    "mmol/L": ("concentration", "0.001"),
    "umol/L": ("concentration", "0.000001"),
}


class PreparationError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(message)


class RoleMapping(ResearchModel):
    value: str
    sample: str | None = None
    experimental_unit: str | None = None
    parent_sample: str | None = None
    group: str | None = None
    time: str | None = None
    batch: str | None = None
    technical_repeat: str | None = None
    pair: str | None = None
    unit: str | None = None
    default_unit: str = "unspecified"
    state: str | None = None
    limit: str | None = None
    reported: str | None = None
    measurement_type: str | None = None
    role: str | None = None
    notes: str | None = None
    record_link: str | None = None


class FilterStep(ResearchModel):
    kind: Literal["filter"]
    field: str
    values: Annotated[list[str], Field(min_length=1, max_length=1000)]
    keep: bool = True


class ConvertStep(ResearchModel):
    kind: Literal["convert_unit"]
    value_field: str
    unit_field: str
    to_unit: str
    limit_field: str | None = None


class WideStep(ResearchModel):
    kind: Literal["wide_to_long"]
    fields: Annotated[list[str], Field(min_length=1, max_length=128)]
    name_field: str
    value_field: str


class LongStep(ResearchModel):
    kind: Literal["long_to_wide"]
    index_fields: Annotated[list[str], Field(min_length=1, max_length=128)]
    name_field: str
    value_field: str


Step = Annotated[FilterStep | ConvertStep | WideStep | LongStep, Field(discriminator="kind")]


class ExclusionDecision(ResearchModel):
    observation_id: str
    reason: Annotated[str, Field(min_length=1, max_length=2000)]


class PreparationPlan(ResearchModel):
    schema_version: Literal[1] = 1
    mapping: RoleMapping
    steps: Annotated[list[Step], Field(max_length=30)] = Field(default_factory=list)
    exclusions: list[ExclusionDecision] = Field(default_factory=list)
    analysis_unit: Literal["experimental_unit", "sample", "observation"] = "observation"
    validate_pairs: bool = False
    pair_times: Annotated[list[str], Field(max_length=2)] = Field(default_factory=list)


def stable_id(namespace: str, *parts: str) -> str:
    return uuid5(NAMESPACE_URL, json.dumps([namespace, *parts], ensure_ascii=False)).hex


def fail(code: str, message: str) -> None:
    raise PreparationError(code, message)


def parse_sources(namespace: str, files: list[tuple[str, bytes]]) -> dict[str, Any]:
    if not 1 <= len(files) <= LIMITS["filesPerImport"]:
        fail("file-count-limit", "Choose between one and five CSV/TSV files.")
    if sum(len(payload) for _, payload in files) > LIMITS["aggregateBytes"]:
        fail("aggregate-byte-limit", "The combined input exceeds 100 MiB.")
    sources: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    expected: list[str] | None = None
    hashes: set[str] = set()
    for name, payload in files:
        if len(payload) > LIMITS["singleFileBytes"]:
            fail("single-file-limit", f"{name}: the file exceeds 50 MiB.")
        extension = Path(name).suffix.lower()
        if extension not in {".csv", ".tsv"}:
            fail("unsupported-research-format", "Research stacking accepts CSV or TSV.")
        digest = hashlib.sha256(payload).hexdigest()
        if digest in hashes:
            fail("duplicate-source", "The same source bytes were supplied more than once.")
        hashes.add(digest)
        try:
            text = payload.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise PreparationError("source-encoding", "Use UTF-8 CSV/TSV files.") from exc
        reader = csv.reader(
            io.StringIO(text, newline=""),
            delimiter="\t" if extension == ".tsv" else ",",
            strict=True,
        )
        try:
            columns = next(reader)
            if (
                not columns
                or any(not field.strip() for field in columns)
                or len(set(columns)) != len(columns)
                or any(field.startswith("__labviz_") for field in columns)
            ):
                fail("invalid-columns", "Headers must be non-empty, unique and not reserved.")
            if len(columns) > LIMITS["columnsPerFile"]:
                fail("column-limit", "A file has more than 128 columns.")
            if expected is not None and columns != expected:
                fail("schema-mismatch", "Stacked files must have identical headers and order.")
            expected = columns
            source_id = stable_id(namespace, digest)
            sources.append({"id": source_id, "name": name, "sha256": digest, "bytes": len(payload)})
            while True:
                physical_row = reader.line_num + 1
                values = next(reader, None)
                if values is None:
                    break
                if not values:
                    continue
                if len(values) != len(columns):
                    fail("row-width", f"{name}, row {physical_row}: column count differs.")
                cells = dict(zip(columns, values, strict=True))
                cells["__labviz_source"] = name
                cells["__labviz_batch"] = name
                rows.append(
                    {
                        "id": stable_id(source_id, str(physical_row)),
                        "cells": cells,
                        "origins": {
                            field: [
                                {
                                    "sourceId": source_id,
                                    "sheet": None,
                                    "row": physical_row,
                                    "column": field,
                                }
                            ]
                            for field in columns
                        },
                    }
                )
                if len(rows) > LIMITS["aggregateRows"]:
                    fail("row-limit", "The combined input exceeds 100,000 rows.")
        except (csv.Error, StopIteration) as exc:
            raise PreparationError("invalid-csv", f"{name}: invalid or empty CSV/TSV.") from exc
    if not rows:
        fail("empty-input", "No measurement rows were found.")
    return {
        "schemaVersion": 1,
        "namespace": namespace,
        "sources": sources,
        "columns": (expected or []) + ["__labviz_source", "__labviz_batch"],
        "rows": rows,
    }


def number(value: str, field: str) -> float | None:
    if value.strip() == "":
        return None
    try:
        parsed = float(value)
    except ValueError as exc:
        raise PreparationError("invalid-number", f"{field}: expected a finite number.") from exc
    if not math.isfinite(parsed):
        fail("invalid-number", f"{field}: expected a finite number.")
    return parsed


def require_fields(columns: list[str], fields: list[str]) -> None:
    if len(fields) != len(set(fields)):
        fail("duplicate-field", "A field is selected more than once.")
    missing = sorted(set(fields) - set(columns))
    if missing:
        fail("missing-field", "Missing fields: " + ", ".join(missing))


def transform(
    raw: dict[str, Any], steps: list[Step]
) -> tuple[list[str], list[dict[str, Any]], list[dict[str, Any]]]:
    columns = list(raw["columns"])
    rows = [{**row, "cells": row["cells"].copy()} for row in raw["rows"]]
    history = []
    for step in steps:
        before = len(rows)
        if isinstance(step, FilterStep):
            require_fields(columns, [step.field])
            allowed = set(step.values)
            rows = [row for row in rows if (row["cells"][step.field] in allowed) == step.keep]
        elif isinstance(step, ConvertStep):
            fields = [step.value_field, step.unit_field]
            if step.limit_field:
                fields.append(step.limit_field)
            require_fields(columns, fields)
            if step.to_unit not in UNITS:
                fail("unsupported-unit", "Choose a documented compatible target unit.")
            target_dimension, target_scale = UNITS[step.to_unit]
            for row in rows:
                cells = row["cells"]
                source_unit = cells[step.unit_field]
                if source_unit not in UNITS or UNITS[source_unit][0] != target_dimension:
                    fail("incompatible-unit", f"Cannot convert {source_unit} to {step.to_unit}.")
                scale = Decimal(UNITS[source_unit][1]) / Decimal(target_scale)
                for field in [step.value_field] + ([step.limit_field] if step.limit_field else []):
                    value = cells[field]
                    if value.strip():
                        number(value, field)
                        try:
                            converted = Decimal(value) * scale
                            number(str(converted), field)
                            cells[field] = format(converted, "f")
                        except InvalidOperation as exc:
                            raise PreparationError(
                                "invalid-number", f"{field}: conversion failed."
                            ) from exc
                cells[step.unit_field] = step.to_unit
        elif isinstance(step, WideStep):
            require_fields(columns, step.fields)
            if any(field.startswith("__labviz_") for field in step.fields):
                fail("reserved-value-field", "Source labels cannot become measurement values.")
            if (
                not step.name_field
                or not step.value_field
                or step.name_field == step.value_field
                or step.name_field in columns
                or step.value_field in columns
                or step.name_field.startswith("__labviz_")
                or step.value_field.startswith("__labviz_")
            ):
                fail("reshape-collision", "Choose two new, distinct field names.")
            if len(rows) * len(step.fields) > LIMITS["aggregateRows"]:
                fail("row-limit", "The reshape would exceed 100,000 rows.")
            expanded = []
            remaining = [field for field in columns if field not in step.fields]
            for row in rows:
                for field in step.fields:
                    cells = {key: row["cells"][key] for key in remaining}
                    cells.update({step.name_field: field, step.value_field: row["cells"][field]})
                    origins = {key: row["origins"].get(key, []) for key in remaining}
                    origins[step.value_field] = row["origins"][field]
                    origins[step.name_field] = row["origins"][field]
                    expanded.append(
                        {"id": stable_id(row["id"], field), "cells": cells, "origins": origins}
                    )
            rows = expanded
            columns = remaining + [step.name_field, step.value_field]
        elif isinstance(step, LongStep):
            require_fields(columns, step.index_fields + [step.name_field, step.value_field])
            if step.value_field.startswith("__labviz_"):
                fail("reserved-value-field", "Source labels cannot become measurement values.")
            groups: dict[tuple[str, ...], dict[str, Any]] = {}
            names = sorted({row["cells"][step.name_field] for row in rows})
            if any(
                not name or name in step.index_fields or name.startswith("__labviz_")
                for name in names
            ):
                fail(
                    "reshape-collision", "Pivot names collide with identifiers or reserved fields."
                )
            for row in rows:
                key = tuple(row["cells"][field] for field in step.index_fields)
                field = row["cells"][step.name_field]
                group = groups.setdefault(
                    key,
                    {
                        "cells": dict(zip(step.index_fields, key, strict=True)),
                        "origins": {},
                        "ids": [],
                    },
                )
                if field in group["cells"]:
                    fail(
                        "duplicate-pivot-key",
                        "Duplicate index/name key; declare repeats before reshaping.",
                    )
                group["cells"][field] = row["cells"][step.value_field]
                group["origins"][field] = row["origins"][step.value_field]
                for index in step.index_fields:
                    group["origins"].setdefault(index, []).extend(row["origins"].get(index, []))
                group["ids"].append(row["id"])
            rows = []
            for group in groups.values():
                if any(name not in group["cells"] for name in names):
                    fail(
                        "missing-pivot-key",
                        "A pivot group is missing a measurement; no imputation applied.",
                    )
                rows.append(
                    {
                        "id": stable_id(raw["namespace"], *sorted(group["ids"])),
                        "cells": group["cells"],
                        "origins": group["origins"],
                    }
                )
            columns = step.index_fields + names
        if len(columns) > LIMITS["columnsPerFile"] + 2:
            fail("column-limit", "The prepared table exceeds the column limit.")
        history.append(
            {
                "step": step.model_dump(mode="json", by_alias=True),
                "inputRows": before,
                "outputRows": len(rows),
                "algorithmVersion": "1",
            }
        )
    if not rows:
        fail("empty-selection", "No rows remain; change the filter before applying.")
    return columns, rows, history


def prepare(raw: dict[str, Any], plan: PreparationPlan) -> dict[str, Any]:
    columns, rows, history = transform(raw, plan.steps)
    mapping = plan.mapping
    mapped = mapping.model_dump(by_alias=False)
    if mapping.value.startswith("__labviz_"):
        fail("reserved-value-field", "A generated source label is not a measurement.")
    required = [
        value for key, value in mapped.items() if key != "default_unit" and value is not None
    ]
    if set(required) - set(columns):
        fail("missing-field", "A mapped field is absent from the prepared table.")
    # Convert limits whenever the mapped measurement values are converted.
    for step in plan.steps:
        if isinstance(step, ConvertStep) and (
            step.value_field != mapping.value or step.unit_field != mapping.unit
        ):
            fail(
                "unit-mapping-mismatch",
                "Convert the mapped measurement and unit together; reshape to long form first.",
            )
        if (
            isinstance(step, ConvertStep)
            and step.value_field == mapping.value
            and mapping.limit
            and step.limit_field != mapping.limit
        ):
            fail("unconverted-limit", "Convert the limit field with the measurement.")
    original_rows = {}
    for original in raw["rows"]:
        reference = next(iter(original["origins"].values()))[0]
        original_rows[(reference["sourceId"], reference["row"])] = original["cells"]

    def original_text(row: dict[str, Any], field: str) -> str:
        return "; ".join(
            original_rows[(ref["sourceId"], ref["row"])][ref["column"]]
            for ref in row["origins"].get(field, [])
        )

    observations = []
    samples: dict[str, dict[str, Any]] = {}
    exclusion_map = {item.observation_id: item.reason.strip() for item in plan.exclusions}
    if len(exclusion_map) != len(plan.exclusions) or any(
        not reason for reason in exclusion_map.values()
    ):
        fail(
            "invalid-exclusion", "Exclusions require unique observation IDs and non-empty reasons."
        )
    states: Counter[str] = Counter()
    for row in rows:
        cells = row["cells"]

        def get(role: str, current: dict[str, str] = cells) -> str | None:
            field = mapped[role]
            return current[field].strip() or None if field else None

        declared_state = get("state")
        try:
            value = number(cells[mapping.value], mapping.value)
        except PreparationError:
            if declared_state not in {
                "below_detection",
                "below_quantification",
                "out_of_range",
                "saturated",
            }:
                raise
            value = None
        state = declared_state or ("missing" if value is None else "observed")
        if state not in STATES:
            fail(
                "unknown-measurement-state",
                "Choose one of the eight documented measurement states.",
            )
        if state == "observed" and value is None:
            fail("observed-requires-value", "Observed measurements require a numeric value.")
        if state in {"missing", "not_measured", "failed"} and value is not None:
            fail(
                "absent-state-has-value",
                "Absent/failed states cannot be treated as numeric readings.",
            )
        limit = number(cells[mapping.limit], mapping.limit) if mapping.limit else None
        if state in {"below_detection", "below_quantification"} and limit is None:
            fail("limit-required", "Limit-qualified measurements require a numeric limit.")
        sample_label, unit_label = get("sample"), get("experimental_unit")
        if unit_label is not None and sample_label is None:
            fail("unit-without-sample", "Declare a sample identifier before its experimental unit.")
        sample_id = stable_id(raw["namespace"], "sample", sample_label) if sample_label else None
        if sample_id:
            sample = {
                "id": sample_id,
                "label": sample_label,
                "experimentalUnit": unit_label,
                "parentSample": get("parent_sample"),
                "group": get("group"),
            }
            if sample_id in samples and samples[sample_id] != sample:
                fail(
                    "conflicting-sample",
                    f"Sample {sample_label} has conflicting identity or group metadata.",
                )
            samples[sample_id] = sample
        repeat_text = get("technical_repeat")
        repeat = None
        if repeat_text is not None:
            if not repeat_text.isascii() or not repeat_text.isdecimal() or int(repeat_text) < 1:
                fail("invalid-repeat", "Technical repeat identifiers must be positive integers.")
            repeat = int(repeat_text)
        role = get("role") or "sample"
        if role not in {"sample", "blank", "standard", "control"}:
            fail("invalid-role", "Role must be sample, blank, standard or control.")
        observation = {
            "id": row["id"],
            "sampleId": sample_id,
            "sampleLabel": sample_label,
            "experimentalUnit": unit_label,
            "group": get("group"),
            "time": get("time"),
            "batch": get("batch"),
            "technicalRepeat": repeat,
            "pairKey": get("pair"),
            "unit": get("unit") or mapping.default_unit,
            "state": state,
            "value": value,
            "limit": limit,
            "reportedText": original_text(row, mapping.reported or mapping.value),
            "reportedUnit": original_text(row, mapping.unit)
            if mapping.unit
            else mapping.default_unit,
            "measurementType": get("measurement_type"),
            "role": role,
            "notes": get("notes"),
            "recordLink": get("record_link"),
            "lineage": row["origins"][mapping.value],
            "exclusionReason": exclusion_map.get(row["id"]),
        }
        observations.append(observation)
        states[state] += 1
    if set(exclusion_map) - {item["id"] for item in observations}:
        fail(
            "unknown-exclusion",
            "An exclusion references an observation outside this prepared revision.",
        )
    by_label = {item["label"]: item for item in samples.values()}
    # Parents may have been filtered out; their declared inventory remains in raw inputs.
    if mapping.sample and mapping.parent_sample:
        for original in raw["rows"]:
            label = original["cells"].get(mapping.sample, "").strip()
            if label and label not in by_label:
                by_label[label] = {
                    "parentSample": original["cells"].get(mapping.parent_sample, "").strip() or None
                }

    for sample in samples.values():
        seen = {sample["label"]}
        parent = sample["parentSample"]
        while parent is not None:
            if parent in seen or parent not in by_label:
                fail("invalid-parent", "Parent samples must exist and cannot form a cycle.")
            seen.add(parent)
            parent = by_label[parent]["parentSample"]
    valid = [
        item for item in observations if item["state"] == "observed" and not item["exclusionReason"]
    ]
    reasons = []
    if len(valid) != sum(item["exclusionReason"] is None for item in observations):
        reasons.append(
            "Non-observed measurements are withheld; review counts and explicit exclusions."
        )
    units = {item["unit"] for item in valid}
    if len(units) > 1:
        reasons.append("Mixed units: explicitly convert compatible values before summarizing.")
    if not valid:
        reasons.append("No included observed measurements.")
    if plan.analysis_unit in {"sample", "experimental_unit"}:
        key = "sampleId" if plan.analysis_unit == "sample" else "experimentalUnit"
        if any(item[key] is None for item in valid):
            reasons.append("The selected analysis unit is unknown for some observations.")
    if plan.analysis_unit in {"sample", "experimental_unit"}:
        repeats: dict[str, set[int | None]] = defaultdict(set)
        counts: Counter[str] = Counter()
        for item in valid:
            key = str(
                item["experimentalUnit"]
                if plan.analysis_unit == "experimental_unit"
                else item["sampleId"]
            )
            repeats[key].add(item["technicalRepeat"])
            counts[key] += 1
        if any(
            count > 1 and (None in repeats[key] or len(repeats[key]) != count)
            for key, count in counts.items()
        ):
            reasons.append(
                "Ambiguous repetitions: select one measurement/time "
                "or declare unique technical repeats."
            )
    pair_deltas = []
    if plan.validate_pairs:
        pairs: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for item in valid:
            if not item["pairKey"] or not item["time"] or not item["sampleId"]:
                fail("pair-key-required", "Pairing requires explicit sample, pair and time keys.")
            pairs[item["pairKey"]].append(item)
        all_times = plan.pair_times
        if (
            len(all_times) != 2
            or len(set(all_times)) != 2
            or set(all_times) != {item["time"] for item in valid}
        ):
            fail("pair-times", "Select exactly two explicitly labelled time points.")
        for items in pairs.values():
            if len(items) != 2 or len({item["time"] for item in items}) != 2:
                fail(
                    "duplicate-or-missing-pair",
                    "Each pair must have one reading at each time point.",
                )
            if (
                len({item["sampleId"] for item in items}) != 1
                or len({item["unit"] for item in items}) != 1
            ):
                fail("incompatible-pair", "A pair must refer to the same declared sample and unit.")
            ordered = sorted(items, key=lambda item: all_times.index(item["time"]))
            pair_deltas.append(ordered[1]["value"] - ordered[0]["value"])
    summary_values = []
    if not reasons:
        if plan.analysis_unit == "observation":
            summary_values = [item["value"] for item in valid]
        else:
            key = "sampleId" if plan.analysis_unit == "sample" else "experimentalUnit"
            buckets: dict[str, list[float]] = defaultdict(list)
            for item in valid:
                buckets[item[key]].append(item["value"])
            summary_values = [mean(values) for values in buckets.values()]
    unknown = any(item["experimentalUnit"] is None for item in observations)
    summary = {
        "independentUnits": None
        if unknown
        else len({item["experimentalUnit"] for item in observations}),
        "knownIndependentUnits": len(
            {
                item["experimentalUnit"]
                for item in observations
                if item["experimentalUnit"] is not None
            }
        ),
        "samples": len(samples),
        "observations": len(observations),
        "states": dict(states),
        "excluded": len(exclusion_map),
        "includedObserved": len(valid),
        "unknownIdentity": sum(item["experimentalUnit"] is None for item in observations),
        "analysisUnit": plan.analysis_unit,
        "withheldReasons": reasons,
        "mean": mean(summary_values) if summary_values else None,
        "analysisN": len(summary_values) if summary_values else None,
        "pairDeltas": pair_deltas,
        "pairDirection": plan.pair_times,
    }
    return {
        "schemaVersion": 1,
        "columns": columns,
        "rows": rows,
        "sources": raw["sources"],
        "samples": list(samples.values()),
        "observations": observations,
        "summary": summary,
        "history": history,
        "plan": plan.model_dump(mode="json", by_alias=True),
    }
