"""Run the tracked V2.2 fixture matrix and print a machine-readable JSON report."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import pandas as pd

from labviz_api.models import ChartSpec
from labviz_api.processing import (
    ProcessingError,
    analyze_chart,
    apply_chart_decisions,
    build_preview,
    build_quality_report,
    load_dataframe,
    render_chart,
    validate_upload,
)

ROOT = Path(__file__).resolve().parents[1] / "samples" / "v22"
DEFAULT_HISTORICAL_ROOT = (
    Path(__file__).resolve().parents[3] / "outputs" / "labviz-test-data-20260907"
)


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
    validate_upload(payload, entry["filename"], 100 * 1024 * 1024)
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
    validate_upload(payload, entry["filename"], 100 * 1024 * 1024)
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


def _historical_chart(
    frame: pd.DataFrame,
    filename: str,
    chart_references: list[dict[str, Any]],
    export_format: str = "png",
) -> ChartSpec:
    prefix = Path(filename).name.split("_", 1)[0]
    reference = next(
        (item for item in chart_references if item["filePrefix"] == prefix),
        None,
    )
    if reference:
        reference_payload = copy.deepcopy(reference["chart"])
        reference_payload["export"]["format"] = export_format
        return ChartSpec.model_validate(reference_payload)

    numeric_fields = [
        str(column)
        for column in frame.columns
        if pd.to_numeric(frame[column], errors="coerce").notna().sum() >= 2
    ]
    if len(numeric_fields) >= 2:
        x_field, y_field = numeric_fields[:2]
        chart_type = "scatter" if prefix == "E09" else "line"
    elif numeric_fields:
        x_field = next(
            (str(column) for column in frame.columns if str(column) not in numeric_fields),
            numeric_fields[0],
        )
        y_field = numeric_fields[0]
        chart_type = "bar"
    else:
        raise AssertionError(f"{filename}: no chartable numeric field")

    fitting = None
    if prefix == "E09":
        x_field = "constant_x"
        y_field = "varying_y"
        fitting = {"model": "linear", "confidenceBand": False}
    payload: dict[str, Any] = {
        "schemaVersion": 1,
        "type": chart_type,
        "title": f"Historical {prefix}",
        "xAxis": {"field": x_field, "title": x_field, "unit": ""},
        "yAxis": {"field": y_field, "title": y_field, "unit": ""},
        "series": [{"field": y_field, "label": y_field, "color": "#2563EB"}],
        "panelCount": 1,
        "export": {
            "format": export_format,
            "dpi": 300,
            "sizePreset": "single-column",
            "grayscalePreview": False,
        },
    }
    if fitting:
        payload["fitting"] = fitting
    return ChartSpec.model_validate(payload)


def _historical_case(
    root: Path,
    entry: dict[str, Any],
    chart_references: list[dict[str, Any]],
) -> dict[str, Any]:
    relative_path = Path(entry["file"])
    payload_path = root / relative_path
    expected = entry["expected"]
    expected_error = expected.get("errorCode")
    try:
        payload = payload_path.read_bytes()
        validate_upload(payload, payload_path.name, 100 * 1024 * 1024)
        frame, sheet, sheets, header = load_dataframe(payload, payload_path.name)
    except (OSError, ProcessingError) as exc:
        observed_code = exc.code if isinstance(exc, ProcessingError) else "missing-fixture"
        return {
            "file": entry["file"],
            "status": "pass" if observed_code == expected_error else "fail",
            "expectedError": expected_error,
            "observedError": observed_code,
        }

    if expected_error:
        return {
            "file": entry["file"],
            "status": "fail",
            "expectedError": expected_error,
            "observedError": None,
        }

    preview = build_preview("historical", frame)
    quality = build_quality_report("historical", frame)
    expected_rows = entry.get("rows")
    if expected_rows is not None and preview["totalRows"] != expected_rows:
        return {
            "file": entry["file"],
            "status": "fail",
            "reason": f"expected {expected_rows} rows, observed {preview['totalRows']}",
        }

    chart = _historical_chart(frame, payload_path.name, chart_references)
    analysis = analyze_chart(frame, chart)
    if relative_path.name.startswith("E09_"):
        fits = [series["fit"] for series in analysis["series"]]
        warnings = [warning for series in analysis["series"] for warning in series["warnings"]]
        if any(fit is not None for fit in fits) or not warnings:
            return {
                "file": entry["file"],
                "status": "fail",
                "reason": "unsupported fit did not return the expected warning-only result",
            }
    if relative_path.name.startswith("E10_") and str(frame.columns[0]) != "time_min":
        return {
            "file": entry["file"],
            "status": "fail",
            "reason": "BOM header was not normalized",
        }

    decisions = [
        {"findingId": finding["id"], "action": "ignore"} for finding in quality["findings"]
    ]
    cleaned = apply_chart_decisions(frame, quality, decisions)
    exports: dict[str, int] = {}
    for export_format in ("png", "svg", "pdf"):
        exports[export_format] = len(
            render_chart(
                frame,
                _historical_chart(frame, payload_path.name, chart_references, export_format),
            )
        )
    return {
        "file": entry["file"],
        "status": "pass",
        "rows": preview["totalRows"],
        "columns": len(preview["columns"]),
        "sheet": sheet,
        "availableSheets": sheets,
        "headerRow": header,
        "qualityKinds": sorted({finding["kind"] for finding in quality["findings"]}),
        "recommendationCount": len(analysis["recommendations"]),
        "analysisSeries": len(analysis["series"]),
        "cleanedRows": len(cleaned),
        "cleanedExportBytes": len(cleaned.to_csv(index=False).encode("utf-8-sig")),
        "exports": exports,
    }


def _historical_matrix(root: Path) -> dict[str, Any]:
    answer_path = root / "数据说明与标准答案.json"
    chart_path = root / "图表配置参考.json"
    if not root.is_dir() or not answer_path.is_file() or not chart_path.is_file():
        return {
            "status": "skip",
            "reason": f"Historical fixture package not found at {root}",
            "results": [],
        }
    answer = json.loads(answer_path.read_text(encoding="utf-8-sig"))
    chart_references = json.loads(chart_path.read_text(encoding="utf-8-sig"))
    results = [_historical_case(root, entry, chart_references) for entry in answer["cases"]]
    passed = sum(item["status"] == "pass" for item in results)
    return {
        "status": "pass" if len(results) == 23 and passed == 23 else "fail",
        "syntheticOnly": bool(answer.get("synthetic")),
        "datasetCount": len(results),
        "passed": passed,
        "failed": len(results) - passed,
        "root": str(root),
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--historical-root",
        type=Path,
        default=DEFAULT_HISTORICAL_ROOT,
        help="Ignored local path containing the synthetic V2.1 23-dataset package.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional ignored path for the machine-readable JSON report.",
    )
    args = parser.parse_args()
    manifest = json.loads((ROOT / "manifest.json").read_text(encoding="utf-8"))
    public = [item for item in manifest["examples"] if item["visibility"] == "public"]
    edges = [item for item in manifest["examples"] if item["visibility"] == "edge"]
    report = {
        "catalogVersion": manifest["catalogVersion"],
        "release": manifest["release"],
        "syntheticOnly": manifest["syntheticOnly"],
        "public": [_public_case(entry) for entry in public],
        "edge": [_edge_case(entry) for entry in edges],
        "historicalV21": _historical_matrix(args.historical_root.resolve()),
    }
    encoded = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        output_path = args.output.resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(f"{encoded}\n", encoding="utf-8")
    print(encoded)
    if report["historicalV21"]["status"] == "fail":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
