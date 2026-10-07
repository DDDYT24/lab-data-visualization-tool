"""Generate and verify 100 deterministic, synthetic LabViz upload fixtures.

Data and reports go only to the ignored outputs directory. XLSX files are
generated separately by build_chaos_xlsx.mjs from the generated sheet spec.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import random
import shutil
import time
from collections import Counter
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
    validate_upload,
)

DEFAULT_ROOT = Path(__file__).resolve().parents[3] / "outputs" / "v22-chaos-100-20260922"
SURFACE_PROFILES = (
    "regular",
    "shuffled",
    "descending",
    "negative",
    "anisotropic",
    "saddle",
    "wave",
    "noise",
    "numeric-strings",
    "bom-header",
    "chinese-columns",
    "grid-21",
    "grid-101",
    "duplicate-xy",
    "missing-cell",
    "irregular-grid",
    "missing-z",
    "mixed-z",
    "constant-x",
    "collinear-xy",
)
SURFACE_STATUS = {
    "duplicate-xy": "duplicate-coordinates",
    "missing-cell": "missing-grid",
    "irregular-grid": "irregular-grid",
    "missing-z": "invalid-values",
    "constant-x": "insufficient-points",
    "collinear-xy": "collinear",
}
SURFACE_ERRORS = {
    "duplicate-coordinates": "surface-duplicate-coordinates",
    "missing-grid": "surface-missing-grid",
    "irregular-grid": "surface-irregular-grid",
    "invalid-values": "surface-invalid-values",
    "insufficient-points": "surface-insufficient-points",
    "collinear": "surface-collinear",
}
XLSX_IDS = {2, 47, 61, 74}


def _surface(profile: str, repeat: int) -> tuple[list[str], list[list[Any]]]:
    side = 5 + repeat
    if profile == "grid-21" or (profile == "grid-101" and repeat):
        side = 21
    elif profile == "grid-101":
        side = 101
    xs = [float(i - side // 2) for i in range(side)]
    ys = [float(i - side // 2) for i in range(side)]
    if profile == "negative":
        xs, ys = [v - 20 for v in xs], [v - 30 for v in ys]
    elif profile == "anisotropic":
        xs, ys = [v * 0.125 for v in xs], [v * 250 for v in ys]
    elif profile == "irregular-grid":
        xs[-1] += 0.3
    rows: list[list[Any]] = []
    for x in xs:
        for y in ys:
            z = x * x + y * y + repeat / 10
            if profile == "saddle":
                z = x * x - y * y
            elif profile == "wave":
                z = math.sin(x / 2) * math.cos(y / 3)
            elif profile == "noise":
                z += random.Random(f"{repeat}:{x}:{y}").uniform(-0.2, 0.2)
            rows.append([x, y, round(z, 8), f"run-{repeat}"])
    if profile == "shuffled":
        random.Random(1221 + repeat).shuffle(rows)
    elif profile == "descending":
        rows.reverse()
    elif profile == "numeric-strings":
        rows = [[f"{x:.4f}", f"{y:.4f}", f"{z:.4f}", tag] for x, y, z, tag in rows]
    elif profile == "duplicate-xy":
        rows.append([*rows[0][:3], "duplicate measurement"])
    elif profile == "missing-cell":
        rows.pop(len(rows) // 2)
    elif profile == "missing-z":
        rows[len(rows) // 2][2] = None
    elif profile == "mixed-z":
        rows[len(rows) // 2][2] = "not a measurement"
    elif profile == "constant-x":
        for row in rows:
            row[0] = 1.0
    elif profile == "collinear-xy":
        for row in rows:
            row[1] = row[0]
    headers = ["x_mm", "y_mm", "response_mV", "run_id"]
    if profile == "chinese-columns":
        headers = ["横坐标_毫米", "纵坐标_毫米", "响应_毫伏", "批次"]
    return headers, rows


def _planar(kind: str, repeat: int) -> tuple[list[str], list[list[Any]]]:
    headers = ["time_s", "response_mV", "temperature_C", "group"]
    rows: list[list[Any]] = [
        [i, round(0.7 * i + math.sin(i / 3), 6), 20 + i % 4, "A" if i % 2 else "B"]
        for i in range(24 + repeat * 4)
    ]
    if repeat == 0:
        rows[2][1] = None
        rows[-1][1] = 1000
    elif repeat == 1:
        rows.append(rows[0].copy())
    elif repeat == 2:
        rows[3][3] = "中文组"
    elif repeat == 3:
        rows[4][1] = -500
    elif repeat == 4:
        random.Random(88).shuffle(rows)
    if kind == "heatmap":
        headers[0] = "dose_mg"
    return headers, rows


def _write_text(
    path: Path, fmt: str, headers: list[str], rows: list[list[Any]], *, bom: bool = False
) -> None:
    if fmt == "json":
        records = [dict(zip(headers, row, strict=True)) for row in rows]
        path.write_text(json.dumps(records, ensure_ascii=False, allow_nan=False), encoding="utf-8")
        return
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, delimiter={"csv": ",", "tsv": "\t", "txt": ";"}[fmt])
    writer.writerow(headers)
    writer.writerows(rows)
    path.write_text(buffer.getvalue(), encoding="utf-8-sig" if bom else "utf-8")


def generate(root: Path) -> None:
    if root.exists():
        raise SystemExit(f"Refusing to overwrite an existing fixture directory: {root}")
    data_dir = root / "data"
    support_dir = root / "support"
    data_dir.mkdir(parents=True)
    support_dir.mkdir(parents=True)
    cases: list[dict[str, Any]] = []
    workbooks: list[dict[str, Any]] = []
    for index in range(100):
        case_id = index + 1
        surface = index < 80
        repeat = index // 20 if surface else (index - 80) % 5
        profile = (
            SURFACE_PROFILES[index % 20]
            if surface
            else ("line", "scatter", "histogram", "heatmap")[(index - 80) // 5]
        )
        headers, rows = _surface(profile, repeat) if surface else _planar(profile, repeat)
        fmt = "xlsx" if case_id in XLSX_IDS else ("csv", "tsv", "json", "txt")[index % 4]
        name = f"{case_id:03d}_{profile}{'_中文' if profile == 'chinese-columns' else ''}.{fmt}"
        if fmt == "xlsx":
            workbooks.append({"filename": name, "headers": headers, "rows": rows})
        else:
            _write_text(data_dir / name, fmt, headers, rows, bom=profile == "bom-header")
        status = SURFACE_STATUS.get(profile, "valid") if surface else "not-applicable"
        cases.append(
            {
                "id": case_id,
                "filename": name,
                "format": fmt,
                "kind": "surface3d" if surface else profile,
                "profile": profile,
                "rows": len(rows),
                "headers": headers,
                "expectedSurfaceStatus": status,
                "expectedError": "invalid-chart-fields"
                if profile == "mixed-z"
                else (SURFACE_ERRORS.get(status) if surface else None),
                "sheetName": "Measurements" if fmt == "xlsx" else None,
            }
        )
    (support_dir / "xlsx-specs.json").write_text(
        json.dumps(workbooks, ensure_ascii=False), encoding="utf-8"
    )
    shutil.copyfile(
        Path(__file__).with_name("build_chaos_xlsx.mjs"), support_dir / "build_xlsx.mjs"
    )
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "version": "1",
                "syntheticOnly": True,
                "seed": "fixed per profile and repeat",
                "total": 100,
                "surface3d": 80,
                "workbookCount": 4,
                "cases": cases,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Generated 96 text fixtures and four XLSX specifications in {root}")


def _chart(case: dict[str, Any], fmt: str) -> ChartSpec:
    fields = case["headers"]
    kind = case["kind"]
    x, y, z = fields[:3]
    if kind == "surface3d":
        series = [
            {"field": y, "label": "Y", "color": "#2563EB"},
            {"field": z, "label": "Z", "color": "#0F766E"},
        ]
        y_axis = z
    elif kind == "heatmap":
        series = [{"field": f, "label": f, "color": "#2563EB"} for f in fields[:3]]
        y_axis = y
    else:
        series = [{"field": y, "label": y, "color": "#2563EB"}]
        y_axis = y
    return ChartSpec.model_validate(
        {
            "schemaVersion": 1,
            "type": kind,
            "title": f"Synthetic {case['id']}",
            "xAxis": {"field": x, "title": x, "unit": ""},
            "yAxis": {"field": y_axis, "title": y_axis, "unit": ""},
            "series": series,
            "panelCount": 1,
            "export": {
                "format": fmt,
                "dpi": 300,
                "sizePreset": "single-column",
                "grayscalePreview": False,
            },
        }
    )


def _one(root: Path, case: dict[str, Any]) -> dict[str, Any]:
    path = root / "data" / case["filename"]
    payload = path.read_bytes()
    start = time.perf_counter()
    validate_upload(payload, path.name, 100 * 1024 * 1024)
    frame, sheet, sheets, _header = load_dataframe(
        payload, path.name, requested_sheet_name=case["sheetName"]
    )
    assert len(frame) == case["rows"], f"row-count drift: {len(frame)}"
    assert frame.columns.tolist() == case["headers"], "header drift"
    preview = build_preview(str(case["id"]), frame)
    quality = build_quality_report(str(case["id"]), frame)
    assert preview["totalRows"] == case["rows"]
    decisions = [{"findingId": item["id"], "action": "ignore"} for item in quality["findings"]]
    cleaned = apply_chart_decisions(frame, quality, decisions)
    assert len(cleaned) == len(frame)
    cleaned_csv = cleaned.to_csv(index=False).encode("utf-8-sig")
    assert cleaned_csv.startswith(b"\xef\xbb\xbf") and len(cleaned_csv) > 3
    expected_error = case["expectedError"]
    expected_status = case["expectedSurfaceStatus"]
    chart = _chart(case, "png")
    observed_error = None
    observed_status = None
    export_sizes: dict[str, int] = {}
    try:
        analysis = analyze_chart(frame, chart)
        if case["kind"] == "surface3d":
            observed_status = analysis["preview"]["surfaceDiagnostics"][0]["status"]
            assert (
                observed_status == expected_status
            ), f"surface status {observed_status} != {expected_status}"
        for export_fmt, signature in (("png", b"\x89PNG"), ("svg", b"<"), ("pdf", b"%PDF")):
            try:
                blob = render_chart(frame, _chart(case, export_fmt))
                assert blob.startswith(signature), f"{export_fmt} signature changed"
                export_sizes[export_fmt] = len(blob)
            except ProcessingError as exc:
                if not expected_error or exc.code != expected_error:
                    raise
                observed_error = exc.code
    except ProcessingError as exc:
        observed_error = exc.code
        if exc.code != expected_error:
            raise
    assert observed_error == expected_error, f"expected {expected_error}, got {observed_error}"
    if not expected_error:
        assert set(export_sizes) == {"png", "svg", "pdf"}, "incomplete exports"
    return {
        "id": case["id"],
        "filename": path.name,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "status": "pass",
        "kind": case["kind"],
        "profile": case["profile"],
        "rows": len(frame),
        "sheet": sheet,
        "availableSheets": sheets,
        "qualityKinds": sorted({item["kind"] for item in quality["findings"]}),
        "surfaceStatus": observed_status,
        "errorCode": observed_error,
        "cleanedCsvBytes": len(cleaned_csv),
        "exports": export_sizes,
        "durationMs": round((time.perf_counter() - start) * 1000, 1),
    }


def verify(root: Path) -> None:
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    cases = manifest["cases"]
    assert len(cases) == 100 and len({case["filename"] for case in cases}) == 100
    assert len(list((root / "data").iterdir())) == 100, "expected exactly 100 data files"
    results = []
    for case in cases:
        try:
            result = _one(root, case)
        except Exception as exc:
            result = {
                "id": case["id"],
                "filename": case["filename"],
                "status": "fail",
                "reason": f"{type(exc).__name__}: {exc}",
            }
        results.append(result)
        print(f"{case['id']:03d}/100 {result['status']} {case['filename']}", flush=True)
    summary = {
        "total": len(results),
        "passed": sum(r["status"] == "pass" for r in results),
        "failed": sum(r["status"] == "fail" for r in results),
        "kinds": dict(Counter(r.get("kind", "failed") for r in results)),
        "formats": dict(Counter(c["format"] for c in cases)),
        "syntheticOnly": True,
        "results": results,
    }
    (root / "test-report.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {key: summary[key] for key in ("total", "passed", "failed", "formats")},
            ensure_ascii=False,
        )
    )
    if summary["failed"]:
        raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("generate", "verify"))
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    (generate if args.action == "generate" else verify)(args.root.resolve())


if __name__ == "__main__":
    main()
