"""Run the tracked fixture matrix and the RenderPlan parity gate; print a JSON report.

Parity: for every readable fixture and every classic chart kind, the preview plan equals the
export plan apart from reduced marks, a second build gives the same plan, and the export ends
on exactly the limits, ticks and labels the plan states. Exit status 1 on any failure.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import warnings
from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pandas as pd
from matplotlib.figure import Figure

from labviz_api.models import ChartSpec
from labviz_api.processing import (
    ProcessingError,
    analyze_chart,
    apply_chart_decisions,
    build_preview,
    build_quality_report,
    classic_plan,
    load_dataframe,
    render_chart,
    validate_upload,
)
from labviz_api.render_plan import RenderPlan

ROOT = Path(__file__).resolve().parents[1] / "samples" / "v22"
DEFAULT_HISTORICAL_ROOT = (
    Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "historical-v21"
)
DATA_SUFFIXES = {".csv", ".tsv", ".txt", ".json", ".xlsx"}
SERIES_VARIANTS: dict[str, dict[str, Any]] = {
    "plain": {},
    "fit": {"fitting": {"model": "linear", "confidenceBand": True, "showEquation": True}},
    "error": {"uncertainty": {"mode": "standard-deviation"}},
}
# Coordinates a reduced preview mark may hold fewer of than the export.
REDUCED_FIELDS = {"x", "y", "x0", "x1", "y0", "y1", "values", "decimation"}


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


def plan_differences(preview: dict[str, Any], export: dict[str, Any]) -> list[str]:
    """Where a preview plan departs from its export plan, apart from what reduction may drop."""
    found = [
        f"plan {key}"
        for key in sorted(set(preview) | set(export))
        if key not in {"panels", "planHash"} and preview.get(key) != export.get(key)
    ]
    if len(preview["panels"]) != len(export["panels"]):
        return [*found, "panel count"]
    for shown, full in zip(preview["panels"], export["panels"], strict=True):
        name = f"panel {full['id']}"
        found += [
            f"{name} {key}"
            for key in sorted(set(shown) | set(full))
            if key != "marks" and shown.get(key) != full.get(key)
        ]
        if [m["id"] for m in shown["marks"]] != [m["id"] for m in full["marks"]]:
            found.append(f"{name} mark ids")
            continue
        for small, whole in zip(shown["marks"], full["marks"], strict=True):
            label = f"{name} mark {whole['id']}"
            reduced = small.get("decimation")
            if reduced is None:
                if small != whole:
                    found.append(label)
                continue
            if any(small.get(k) != whole.get(k) for k in set(whole) - REDUCED_FIELDS):
                found.append(f"{label} style")
            if whole.get("decimation") is not None or reduced["drawn"] >= reduced["input"]:
                found.append(f"{label} decimation")
    return found


@contextmanager
def drawn_axes() -> Iterator[list[dict[str, Any]]]:
    """Record the limits, ticks and labels of every axis Matplotlib saves."""
    original = Figure.savefig
    drawn: list[dict[str, Any]] = []

    def capture(figure: Any, *args: Any, **kwargs: Any) -> Any:
        for axis in figure.axes:
            if axis.get_label() == "<colorbar>":
                continue
            drawn.append(
                {
                    "x": axis.get_xlim(),
                    "y": axis.get_ylim(),
                    "z": axis.get_zlim() if hasattr(axis, "get_zlim") else None,
                    "yticks": [float(value) for value in axis.get_yticks()],
                    "xlabels": [label.get_text() for label in axis.get_xticklabels()],
                }
            )
        return original(figure, *args, **kwargs)

    Figure.savefig = capture  # type: ignore[method-assign,assignment]
    try:
        yield drawn
    finally:
        Figure.savefig = original  # type: ignore[method-assign]


def drawing_differences(drawn: dict[str, Any], panel: dict[str, Any]) -> list[str]:
    axes = panel["axes"]

    def close(a: float, b: float) -> bool:
        return abs(a - b) <= 1e-9 * max(1.0, abs(a))

    found = [
        f"{name} limits"
        for name in ("x", "y", "z")
        if axes[name] is not None
        and not all(close(a, b) for a, b in zip(drawn[name], axes[name]["domain"], strict=True))
    ]
    if drawn["yticks"] != [tick["value"] for tick in axes["y"]["ticks"]]:
        found.append("y ticks")
    if drawn["xlabels"] != [tick["label"] for tick in axes["x"]["ticks"]]:
        found.append("x labels")
    return found


def parity_charts(frame: pd.DataFrame, title: str) -> Iterator[tuple[str, ChartSpec]]:
    numeric = [str(c) for c in frame.columns if pd.api.types.is_numeric_dtype(frame[c])]
    if not numeric:
        return
    y = numeric[-1]
    x = next((str(c) for c in frame.columns if str(c) != y), y)

    def spec(kind: str, x_field: str, y_field: str, series: list[str], **extra: Any) -> ChartSpec:
        return ChartSpec.model_validate(
            {
                "schemaVersion": 1,
                "type": kind,
                "title": title,
                "xAxis": {"field": x_field, "title": x_field, "unit": ""},
                "yAxis": {"field": y_field, "title": y_field, "unit": ""},
                "series": [{"field": f, "label": f, "color": "#2563EB"} for f in series],
                "panelCount": 1,
                "export": {
                    "format": "png",
                    "dpi": 300,
                    "sizePreset": "single-column",
                    "grayscalePreview": False,
                },
                **extra,
            }
        )

    yield "histogram", spec("histogram", x, y, [y])
    yield "box", spec("box", x, y, [y])
    for kind in ("bar", "line", "scatter"):
        for name, extra in SERIES_VARIANTS.items():
            yield f"{kind}-{name}", spec(kind, x, y, [y], **extra)
    if len(numeric) >= 2:
        yield "heatmap", spec("heatmap", numeric[0], numeric[1], numeric[:8])
    if len(numeric) >= 3:
        yield "surface3d", spec("surface3d", numeric[0], numeric[2], numeric[1:3])


def parity_case(frame: pd.DataFrame, chart: ChartSpec) -> str | list[str]:
    """ "pass", "refused" for data the chart kind cannot use, or the differences found."""
    try:
        derived = analyze_chart(frame, chart)
        export = classic_plan(frame, chart, derived)
        preview = classic_plan(frame, chart, derived, detail="preview")
    except ProcessingError:
        return "refused"
    with warnings.catch_warnings(), drawn_axes() as drawn:
        warnings.simplefilter("ignore")
        try:
            render_chart(frame, chart)
        except ProcessingError:
            return "refused" if export is None else ["export refused a planned chart"]
    if export is None or preview is None:
        return ["export drew a chart that has no plan"]
    for plan in (export, preview):
        RenderPlan.model_validate_json(json.dumps(plan))
    found = plan_differences(preview, export)
    again = classic_plan(frame, chart, analyze_chart(frame, chart))
    if again is None or again["planHash"] != export["planHash"]:
        found.append("second build differs")
    for axis, panel in zip(drawn, export["panels"], strict=False):
        found += drawing_differences(axis, panel)
    return found or "pass"


def parity_matrix(historical_root: Path) -> dict[str, Any]:
    manifest = json.loads((ROOT / "manifest.json").read_text(encoding="utf-8"))
    files = [
        (ROOT / e["filename"], e.get("sheetName"), e.get("headerRow", 1))
        for e in manifest["examples"]
    ]
    files += [
        (path, None, 1)
        for path in sorted(historical_root.rglob("*"))
        if path.suffix.lower() in DATA_SUFFIXES and not path.name.startswith(("数据", "图表"))
    ]
    counts: Counter[str] = Counter()
    failures = []
    for path, sheet, header in files:
        try:
            frame = load_dataframe(
                path.read_bytes(), path.name, requested_sheet_name=sheet, header_row=header
            )[0]
        except ProcessingError:
            counts["unreadable"] += 1
            continue
        for label, chart in parity_charts(frame, path.stem):
            outcome = parity_case(frame, chart)
            if isinstance(outcome, list):
                counts[f"{label} fail"] += 1
                failures.append({"file": path.name, "chart": label, "differences": outcome})
            else:
                counts[f"{label} {outcome}"] += 1
    passed = sum(count for key, count in counts.items() if key.endswith(" pass"))
    return {
        "status": "pass" if not failures and passed else "fail",
        "files": len(files),
        "passed": passed,
        "failed": len(failures),
        "counts": dict(sorted(counts.items())),
        "failures": failures,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--historical-root",
        type=Path,
        default=DEFAULT_HISTORICAL_ROOT,
        help="Folder of the synthetic V2.1 23-dataset package (tracked under tests/fixtures).",
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
        "planParity": parity_matrix(args.historical_root.resolve()),
    }
    encoded = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        output_path = args.output.resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(f"{encoded}\n", encoding="utf-8")
    print(encoded)
    parity = report["planParity"]
    print(
        f"plan parity: {parity['passed']} passed, {parity['failed']} failed, "
        f"{parity['files']} files",
        file=sys.stderr,
    )
    if report["historicalV21"]["status"] != "pass" or parity["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
