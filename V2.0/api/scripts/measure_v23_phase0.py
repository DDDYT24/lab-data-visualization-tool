"""Record current-backend proxies; unavailable V2.3 UI baselines stay explicit."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import tempfile
from functools import partial
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

from labviz_api.config import Settings
from labviz_api.main import create_app
from labviz_api.models import ChartSpec
from labviz_api.processing import analyze_chart, build_preview, load_dataframe, render_chart
from scripts.measure_v22_performance import (
    _csv_payload,
    _line_chart,
    _measure,
    _process_peak_memory_bytes,
)

V2 = Path(__file__).resolve().parents[2]


def measure_case(rows: int) -> dict[str, Any]:
    payload = _csv_payload(rows)
    chart_data = _line_chart().model_dump(mode="json", by_alias=True)
    chart_data["panelCount"] = 4
    chart_data["export"]["sizePreset"] = "double-column"
    chart_data["series"] = [
        {**chart_data["series"][0], "panel": panel, "label": f"Panel {panel}"}
        for panel in range(1, 5)
    ]
    chart = ChartSpec.model_validate(chart_data)
    times: dict[str, float] = {}
    times["parseMs"], details = _measure(lambda: load_dataframe(payload, "baseline.csv"))
    frame = details[0]
    times["previewSerializationMs"], _ = _measure(lambda: build_preview("baseline", frame))
    times["recomputeMs"], _ = _measure(lambda: analyze_chart(frame, chart))
    sizes = {}
    for format_name in ("png", "svg", "pdf"):
        export_data = chart.model_dump(mode="json", by_alias=True)
        export_data["export"]["format"] = format_name
        export_chart = ChartSpec.model_validate(export_data)
        times[f"fourPanel{format_name.title()}Ms"], rendered = _measure(
            partial(render_chart, frame, export_chart),
        )
        sizes[format_name] = len(rendered)
    with tempfile.TemporaryDirectory(prefix="labviz-v23-p0-") as temporary:
        root = Path(temporary)
        settings = Settings(
            database_path=root / "labviz.db",
            object_storage_root=root / "objects",
            allowed_origins=("http://127.0.0.1:3001",),
            public_web_url="http://127.0.0.1:3001",
            local_access_key="p" * 43,
        )
        with TestClient(create_app(settings)) as client:
            assert (
                client.post(
                    "/api/v1/local/session",
                    headers={"X-LabViz-Local-Key": "p" * 43},
                ).status_code
                == 200
            )
            times["importToReadyMs"], response = _measure(
                lambda: client.post(
                    "/api/v1/projects",
                    files={"file": ("baseline.csv", payload, "text/csv")},
                )
            )
            assert response.status_code == 202, response.text
            body = response.json()
            project_id = body["projectId"]
            assert client.get(f"/api/v1/jobs/{body['job']['id']}").json()["stage"] == "ready"
            times["firstWorkspaceMs"], response = _measure(
                lambda: client.get(f"/api/v1/projects/{project_id}/workspace"),
            )
            assert response.status_code == 200, response.text
            times["saveMs"], response = _measure(
                lambda: client.put(
                    f"/api/v1/projects/{project_id}/chart",
                    json={"chart": chart_data},
                )
            )
            assert response.status_code == 200, response.text
        with TestClient(create_app(settings)) as reopened:
            assert (
                reopened.post(
                    "/api/v1/local/session",
                    headers={"X-LabViz-Local-Key": "p" * 43},
                ).status_code
                == 200
            )
            times["reopenMs"], response = _measure(
                lambda: reopened.get(f"/api/v1/projects/{project_id}/workspace"),
            )
            assert response.status_code == 200, response.text
            assert response.json()["chart"]["panelCount"] == 4
    return {
        "rows": rows,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "bytes": len(payload),
        "timings": times,
        "exportBytes": sizes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    runs = []
    for rows in (10_000, 100_000):
        for repeat in range(1, 4):
            case = measure_case(rows)
            case["repeat"] = repeat
            runs.append(case)
            print(f"Measured {rows} rows, repetition {repeat}/3", flush=True)
    maxima = {
        str(rows): {
            key: max(run["timings"][key] for run in runs if run["rows"] == rows)
            for key in runs[0]["timings"]
        }
        for rows in (10_000, 100_000)
    }
    peak = _process_peak_memory_bytes()
    sources = [
        Path(__file__),
        V2 / "api/labviz_api/processing.py",
        V2 / "api/labviz_api/models.py",
        V2 / "api/scripts/measure_v22_performance.py",
        V2 / "contracts/v23-workflow-v1.json",
    ]
    report = {
        "contractVersion": "v23-workflow-v1",
        "scope": "current-host backend proxies; not browser or mixed-panel feature acceptance",
        "environment": {
            "reference": "local-i9-13900H-20260927",
            "python": platform.python_version(),
            "platform": platform.platform(),
            "processor": platform.processor(),
            "logicalProcessors": os.cpu_count(),
            "loadControl": "Developer machine; other processes not controlled",
        },
        "sourceSha256": {
            str(path.relative_to(V2)).replace("\\", "/"): hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
            for path in sources
        },
        "runs": runs,
        "maximumMs": maxima,
        "backendPeakMiB": None if peak is None else round(peak / 1024**2, 2),
        "unavailable": {
            "browserFirstUsablePreviewMs": "requires actual browser instrumentation",
            "linkedHighlightMs": "Phase 3 feature not implemented",
            "mixedFourPanelExportMs": "Not implemented; homogeneous legacy four-panel proxy only",
            "browserPeakMiB": "requires browser process measurement",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Saved baseline to {args.output}", flush=True)


if __name__ == "__main__":
    main()
