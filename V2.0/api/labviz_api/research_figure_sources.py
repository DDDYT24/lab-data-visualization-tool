"""Frozen adapters for existing 2D panels and scientific chart results."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

import numpy as np
import pandas as pd
from pydantic_core import from_json, to_json

from .models import ChartSpec
from .processing import _json_value, _sample_positions, analyze_chart, serialize_dataframe
from .research_panel_headers import remember_header, source_header
from .research_preparation import PreparationError, stable_id


def digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()
    ).hexdigest()


def legacy_snapshot(
    repository: Any,
    project_id: str,
    panel: int,
    sample_field: str | None = None,
    sample_namespace: str | None = None,
) -> dict[str, Any]:
    before = repository.get_workspace(project_id)
    chart = ChartSpec.model_validate(before["chart"])
    if chart.type not in {"line", "scatter", "box", "heatmap"}:
        raise PreparationError(
            "figure-type", "Choose line, scatter, box or correlation in the main workspace."
        )
    selected = [s.model_copy(update={"panel": 1}) for s in chart.series if s.panel == panel]
    if not selected:
        raise PreparationError("figure-panel", "The selected saved panel has no series.")
    if chart.secondary_y_axis.enabled:
        raise PreparationError(
            "figure-secondary-axis", "Disable the secondary axis before capturing a panel."
        )
    chart = chart.model_copy(update={"series": selected, "panel_count": 1})
    frame = repository.load_chart_dataframe(project_id)
    if len(frame) > 100_000:
        raise PreparationError("figure-size", "A figure panel supports up to 100,000 source rows.")
    if sample_field and (sample_field not in frame.columns or not sample_namespace):
        raise PreparationError(
            "figure-sample-mapping",
            "Choose an existing sample column and apply a research revision first.",
        )
    after = repository.get_workspace(project_id)
    if any(before[key] != after[key] for key in ("chart", "decisions", "quality", "preview")):
        raise PreparationError("research-conflict", "The workspace changed. Retry the capture.")
    analysis = analyze_chart(frame, chart)
    data_hash = hashlib.sha256(
        serialize_dataframe(repository.load_quality_dataframe(project_id))
    ).hexdigest()
    # Row positions come from the retained dataframe index, not sorted/sampled display positions.
    rows: list[dict[str, Any]] = []
    for index, row in frame.iterrows():
        values = {str(k): _json_value(v) for k, v in row.items()}
        sample_label = (
            None
            if not sample_field or pd.isna(row[sample_field])
            else str(row[sample_field]).strip() or None
        )
        rows.append(
            {
                "id": f"legacy:{data_hash}:{index}",
                "sampleId": stable_id(sample_namespace, "sample", sample_label)
                if sample_label and sample_namespace
                else None,
                "sampleLabel": sample_label,
                "value": None,
                "unit": "",
                "values": values,
                "lineage": [
                    {"sourceId": data_hash, "row": int(index) + 2, "column": "*", "sheet": None}
                ],
            }
        )
    marks: list[dict[str, Any]] = []
    if chart.type in {"line", "scatter"}:
        for series in analysis["series"]:
            subset = (
                frame
                if series["group"] is None
                else frame[frame[chart.group_field] == series["group"]]
            )
            fields = list(dict.fromkeys([chart.x_axis.field, series["field"]]))
            complete = subset[fields].dropna()
            numeric = pd.to_numeric(complete[series["field"]], errors="coerce")
            complete = complete[numeric.notna()]
            complete = complete.iloc[_sample_positions(len(complete))]
            valid = complete[np.isfinite(pd.to_numeric(complete[series["field"]]).to_numpy())]
            for point, index in zip(series["points"], valid.index, strict=True):
                marks.append(
                    {
                        "kind": "point",
                        "x": point["x"],
                        "y": point["y"],
                        "rowIds": [f"legacy:{data_hash}:{index}"],
                        "label": series["label"],
                    }
                )
    elif chart.type == "box":
        for i, box in enumerate(analysis["preview"]["boxes"]):
            ids = [
                row["id"]
                for row in rows
                if isinstance(row["values"].get(box["field"]), (int, float))
                and np.isfinite(row["values"][box["field"]])
            ]
            marks.append(
                {
                    "kind": "aggregate",
                    "x": i + 1,
                    "y": box["median"],
                    "rowIds": ids,
                    "label": box["label"],
                }
            )
    else:
        fields = [s.field for s in selected]
        for i, a in enumerate(fields):
            for j, b in enumerate(fields):
                ids = [
                    row["id"]
                    for row in rows
                    if all(
                        isinstance(row["values"].get(f), (int, float))
                        and np.isfinite(row["values"][f])
                        for f in [a, b]
                    )
                ]
                marks.append(
                    {
                        "kind": "aggregate",
                        "x": j,
                        "y": i,
                        "rowIds": ids,
                        "label": f"{a} / {b}: pairwise correlation inputs",
                    }
                )
    x_values = [p["x"] for a in analysis["series"] for p in a["points"]]
    x_categories = (
        list(dict.fromkeys(x_values)) if any(isinstance(v, str) for v in x_values) else None
    )
    spec = chart.model_dump(mode="json", by_alias=True)
    result = {
        "kind": chart.type,
        "sampleField": sample_field,
        "title": chart.title,
        "legacySpec": spec,
        "frame": json.loads(frame.to_json(orient="split")),
        "analysis": analysis,
        "rows": rows,
        "marks": marks,
        "unit": chart.y_axis.unit,
        "revisionId": "legacy:" + digest({"frame": rows, "decisions": before["decisions"]})[:32],
        "xSignature": ["legacy", chart.x_axis.field, chart.x_axis.unit, x_categories]
        if chart.type in {"line", "scatter"}
        else [chart.type, [s.label for s in selected]],
        "ySignature": ["numeric", chart.y_axis.title, chart.y_axis.unit]
        if chart.type != "heatmap"
        else ["correlation", [s.label for s in selected]],
        "sampled": len(marks) < len(rows) if chart.type in {"line", "scatter"} else False,
    }
    result["resultSha256"] = digest(result)
    return result


def scientific_source(result: dict[str, Any], revision: str) -> dict[str, Any]:
    rows = [p["observation"] for p in result["points"]]
    # Bound hit-target allocation before composition; scientific data/export remain complete.
    targets = set(range(min(len(rows), 500)))
    if len(rows) > 500:
        low: dict[str | None, int] = {}
        high: dict[str | None, int] = {}
        for i, point in enumerate(result["points"]):
            group = point["observation"]["group"]
            if group not in low or point["y"] < result["points"][low[group]]["y"]:
                low[group] = i
            if group not in high or point["y"] > result["points"][high[group]]["y"]:
                high[group] = i
        targets = set(low.values()) | set(high.values())
        available = max(1, 500 - len(targets))
        stride = max(1, (len(rows) + available - 1) // available)
        targets.update(range(0, len(rows), stride))
    marks = (
        [
            {
                "kind": "point",
                "sourceIndex": index,
                "x": p["x"],
                "y": p["y"],
                "rowIds": [p["observation"]["id"]],
                "label": p["observation"]["sampleLabel"] or p["observation"]["id"],
            }
            for index, p in enumerate(result["points"])
            if index in targets
        ]
        if result["spec"]["showPoints"]
        else []
    )
    for density in result["densities"]:
        marks.append(
            {
                "kind": "density",
                "x": density["position"],
                "y": density["grid"][len(density["grid"]) // 2],
                "rowIds": density["observationIds"],
                "label": "Density input subset",
            }
        )
    for group in result["groups"]:
        if group["summary"] is not None:
            marks.append(
                {
                    "kind": "aggregate",
                    "x": group["position"],
                    "y": group["summary"],
                    "rowIds": [r["id"] for r in rows if r["group"] == group["label"]],
                    "label": "Summary input subset",
                }
            )
    next_index = len(rows) if result["spec"]["showPoints"] else 0
    for mark in marks:
        if mark["kind"] != "point":
            mark["sourceIndex"] = next_index
            next_index += 1
    kind = result["spec"]["kind"]
    return {
        "kind": kind,
        "title": result["spec"]["title"],
        "scientific": result,
        "revisionId": revision,
        "unit": result["unit"],
        "rows": rows,
        "marks": marks,
        "resultSha256": result["resultSha256"],
        "sampled": len(targets) < len(rows) and result["spec"]["showPoints"],
        "xSignature": ["paired", result["spec"]["pairTimes"]]
        if kind == "paired"
        else ["group", [g["label"] for g in result["groups"]]],
        "ySignature": ["numeric", rows[0]["measurementType"], result["unit"]],
    }


def source(connection: Any, project_id: str, identifier: str) -> dict[str, Any]:
    row = connection.execute(
        "SELECT revision_id,result_json FROM research_charts WHERE project_id=? AND id=?",
        (project_id, identifier),
    ).fetchone()
    if row:
        return scientific_source(
            from_json(row["result_json"], cache_strings="all", allow_inf_nan=False),
            row["revision_id"],
        )
    row = connection.execute(
        "SELECT result_json FROM research_figure_sources WHERE project_id=? AND id=?",
        (project_id, identifier),
    ).fetchone()
    if row:
        return dict(from_json(row["result_json"], cache_strings="all", allow_inf_nan=False))
    raise PreparationError(
        "figure-chart-unavailable", "Choose saved panels belonging to this project."
    )


def save_legacy(store: Any, project_id: str, result: dict[str, Any]) -> str:
    identifier = uuid4().hex
    with store.connect() as connection:
        connection.execute(
            "INSERT INTO research_figure_sources VALUES (?,?,?,?)",
            (
                identifier,
                project_id,
                # legacy_snapshot already checks finite JSON through its full digest.
                to_json(result, ensure_ascii=False, round_trip=True).decode("utf-8"),
                datetime.now(UTC).isoformat(),
            ),
        )
        remember_header(connection, project_id, identifier, source_header(result))
    return identifier
