"""Run the tracked V2.2 fixture matrix and print a machine-readable JSON report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from labviz_api.models import ChartSpec
from labviz_api.processing import (
    ProcessingError,
    analyze_chart,
    apply_chart_decisions,
    build_preview,
    build_quality_report,
    load_dataframe,
    render_chart,
)

ROOT = Path(__file__).resolve().parents[1] / "samples" / "v22"


def _field(entry: dict[str, Any], role: str) -> dict[str, Any] | None:
    return next((item for item in entry["fields"] if item["role"] == role), None)


def _numeric_fields(entry: dict[str, Any]) -> list[dict[str, Any]]:
    return [item for item in entry["fields"] if item["kind"] == "number"]


def _chart(entry: dict[str, Any], export_format: str = "png") -> ChartSpec:
    chart_type = entry["recommendedChart"]
    numeric = _numeric_fields(entry)
    x_field = _field(entry, "x") or (numeric[0] if numeric else entry["fields"][0])
    y_field = _field(entry, "y") or (numeric[1] if len(numeric) > 1 else numeric[0])
    group = _field(entry, "group") if chart_type in {"line", "scatter"} else None
    if chart_type == "surface3d":
        y_field = _field(entry, "y") or numeric[1]
        z_field = _field(entry, "z") or numeric[2]
        series = [
            {"field": y_field["name"], "label": "Y", "color": "#2563EB"},
            {"field": z_field["name"], "label": "Z", "color": "#0F766E"},
        ]
        y_field = z_field
    elif chart_type == "heatmap":
        x_field, y_field = numeric[0], numeric[1]
        series = [
            {"field": item["name"], "label": item["name"], "color": "#2563EB"} for item in numeric
        ]
    else:
        series = [{"field": y_field["name"], "label": y_field["name"], "color": "#2563EB"}]
        if chart_type in {"histogram", "box"}:
            x_field = _field(entry, "label") or _field(entry, "group") or x_field
    return ChartSpec.model_validate(
        {
            "schemaVersion": 1,
            "type": chart_type,
            "title": entry["title"]["en"],
            "xAxis": {"field": x_field["name"], "title": x_field["name"], "unit": ""},
            "yAxis": {"field": y_field["name"], "title": y_field["name"], "unit": ""},
            "series": series,
            "groupField": group["name"] if group else None,
            "panelCount": 1,
            "export": {
                "format": export_format,
                "dpi": 300,
                "sizePreset": "single-column",
                "grayscalePreview": False,
            },
        }
    )


def _public_case(entry: dict[str, Any]) -> dict[str, Any]:
    payload = (ROOT / entry["filename"]).read_bytes()
    frame, sheet, sheets, header = load_dataframe(
        payload,
        entry["filename"],
        requested_sheet_name=entry.get("sheetName"),
        header_row=entry.get("headerRow", 1),
    )
    preview = build_preview("fixture", frame)
    quality = build_quality_report("fixture", frame)
    chart = _chart(entry)
    analysis = analyze_chart(frame, chart)
    decisions = [
        {"findingId": finding["id"], "action": "ignore"} for finding in quality["findings"]
    ]
    cleaned = apply_chart_decisions(frame, quality, decisions)
    exports: dict[str, int] = {}
    for export_format in ("png", "svg", "pdf"):
        exports[export_format] = len(render_chart(frame, _chart(entry, export_format)))
    answer_key = entry["answerKey"]
    quality_kinds = {finding["kind"] for finding in quality["findings"]}
    if preview["totalRows"] != answer_key["previewRows"]:
        raise AssertionError(f"{entry['slug']}: preview row answer changed")
    if not set(answer_key["requiredQualityKinds"]).issubset(quality_kinds):
        raise AssertionError(f"{entry['slug']}: quality answer changed")
    return {
        "slug": entry["slug"],
        "status": "pass",
        "format": entry["format"],
        "rows": preview["totalRows"],
        "columns": len(preview["columns"]),
        "sheet": sheet,
        "availableSheets": sheets,
        "headerRow": header,
        "qualityKinds": sorted(quality_kinds),
        "recommendationCount": len(analysis["recommendations"]),
        "analysisSeries": len(analysis["series"]),
        "cleanedRows": len(cleaned),
        "cleanedExportBytes": len(cleaned.to_csv(index=False).encode("utf-8-sig")),
        "exports": exports,
    }


def _edge_case(entry: dict[str, Any]) -> dict[str, Any]:
    payload = (ROOT / entry["filename"]).read_bytes()
    expected = set(entry["expectedErrors"])
    if entry["slug"] == "duplicate-columns":
        try:
            load_dataframe(payload, entry["filename"])
        except ProcessingError as exc:
            if exc.code != "duplicate-columns":
                raise AssertionError(f"{entry['slug']}: unexpected code {exc.code}") from exc
            return {"slug": entry["slug"], "status": "pass", "observed": [exc.code]}
        raise AssertionError(f"{entry['slug']}: duplicate columns were accepted")

    frame, _sheet, _sheets, _header = load_dataframe(payload, entry["filename"])
    quality = build_quality_report("fixture", frame)
    observed = {finding["kind"] for finding in quality["findings"]}
    if entry["slug"] == "bom-header":
        if "\ufeff" in str(frame.columns[0]):
            raise AssertionError("BOM was left in the first column name")
        observed.add("bom-normalized")
    elif entry["slug"] == "unicode-filename":
        observed.add("unicode-download")
    elif entry["slug"].startswith("invalid-surface"):
        preview = analyze_chart(
            frame,
            ChartSpec.model_validate(
                {
                    "schemaVersion": 1,
                    "type": "surface3d",
                    "title": "Invalid surface",
                    "xAxis": {"field": "x", "title": "X", "unit": ""},
                    "yAxis": {"field": "z", "title": "Z", "unit": ""},
                    "series": [
                        {"field": "y", "label": "Y", "color": "#2563EB"},
                        {"field": "z", "label": "Z", "color": "#0F766E"},
                    ],
                    "panelCount": 1,
                    "export": {
                        "format": "png",
                        "dpi": 300,
                        "sizePreset": "single-column",
                        "grayscalePreview": False,
                    },
                }
            ),
        )["preview"]
        observed.add(f"surface-{preview['surfaceDiagnostics'][0]['status']}")
    if not expected.issubset(observed):
        raise AssertionError(
            f"{entry['slug']}: expected {sorted(expected)}, observed {sorted(observed)}"
        )
    return {"slug": entry["slug"], "status": "pass", "observed": sorted(observed)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    manifest = json.loads((ROOT / "manifest.json").read_text(encoding="utf-8"))
    public = [item for item in manifest["examples"] if item["visibility"] == "public"]
    edges = [item for item in manifest["examples"] if item["visibility"] == "edge"]
    report = {
        "catalogVersion": manifest["catalogVersion"],
        "release": manifest["release"],
        "syntheticOnly": manifest["syntheticOnly"],
        "public": [_public_case(entry) for entry in public],
        "edge": [_edge_case(entry) for entry in edges],
        "historicalV21": {
            "status": "skip",
            "reason": (
                "The historical 23-dataset corpus is not part of the tracked V2.2 sample bundle."
            ),
        },
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
