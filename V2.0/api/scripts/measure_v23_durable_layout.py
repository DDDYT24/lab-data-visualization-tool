"""Same final implementation, legacy synchronous save versus durable queued layout."""

import argparse
import gc
import hashlib
import json
import os
import time
import uuid
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

from labviz_api.config import Settings
from scripts.measure_v22_performance import _csv_payload, _process_peak_memory_bytes

root = Path(__file__).resolve().parents[3]
parser = argparse.ArgumentParser()
parser.add_argument("--rows", type=int, choices=[10000, 100000], required=True)
parser.add_argument("--mixed", action="store_true")
parser.add_argument("--suffix", default="")
args = parser.parse_args()
reports = root / "outputs/v23-phase5"
data = reports / ("durable-probe-" + uuid.uuid4().hex)
data.mkdir(parents=True)
payload = _csv_payload(args.rows)
runs = []
key = uuid.uuid4().hex + uuid.uuid4().hex


def create_probe_app(directory: Path, access_key: str) -> Any:
    # main has a module-level application. Keep that bootstrap inside the probe too.
    os.environ["LABVIZ_DATABASE_PATH"] = str(directory / "bootstrap.db")
    os.environ["LABVIZ_OBJECT_STORAGE_ROOT"] = str(directory / "bootstrap-objects")
    from labviz_api.main import create_app

    return create_app(
        Settings(
            database_path=directory / "test.db",
            object_storage_root=directory / "objects",
            allowed_origins=("http://127.0.0.1:3001",),
            public_web_url="http://127.0.0.1:3001",
            local_access_key=access_key,
        )
    )


app = create_probe_app(data, key)
with TestClient(app) as client:
    assert (
        client.post("/api/v1/local/session", headers={"X-LabViz-Local-Key": key}).status_code == 200
    )
    for repeat in range(3):
        created = client.post(
            "/api/v1/projects", files={"file": ("linear.csv", payload, "text/csv")}
        )
        assert created.status_code == 202, created.text
        project = created.json()["projectId"]
        base = f"/api/v1/projects/{project}/research"
        preview = client.post(
            base + "/preview",
            data={"plan": json.dumps({"mapping": {"value": "response_mV", "defaultUnit": "mV"}})},
            files={"files": ("linear.csv", payload, "text/csv")},
        )
        assert preview.status_code == 200, preview.text
        applied = client.post(base + "/apply", json={"token": preview.json()["token"]})
        assert applied.status_code == 200, applied.text
        revision = applied.json()["revisionId"]
        chart = client.post(
            base + "/charts",
            json={
                "revisionId": revision,
                "spec": {"title": "Large raw points", "pointRendering": "auto"},
            },
        )
        assert chart.status_code == 200, chart.text
        ids = [chart.json()["id"]] * 4
        if args.mixed:
            captured = client.post(base + "/figure-sources?panel=1")
            assert captured.status_code == 200, captured.text
            violin = client.post(
                base + "/charts",
                json={"revisionId": revision, "spec": {"kind": "violin", "pointRendering": "auto"}},
            )
            assert violin.status_code == 200, violin.text
            ids = [captured.json()["id"], ids[0], captured.json()["id"], violin.json()["id"]]
        spec = {"title": "Four large panels", "panelChartIds": ids, "pointRendering": "auto"}
        report: dict[str, Any] = {"repeat": repeat + 1}
        images = {}
        for deferred in [False, True] if repeat % 2 == 0 else [True, False]:
            app.state.clear_research_caches(project)
            gc.collect()  # Explicit probe isolation; never claimed as a UI setting.
            started = time.perf_counter()
            response = client.post(
                base + "/figure-boards" + ("?deferArtifacts=true" if deferred else ""), json=spec
            )
            save_ms = (time.perf_counter() - started) * 1000
            assert response.status_code == (202 if deferred else 200), response.text
            board_url = base + "/figure-boards/" + response.json()["id"]
            view = response.json()
            assert view["spec"]["panelChartIds"] == ids and len(view["panels"]) == 4
            if deferred:
                assert view["artifactState"] == "queued" and not view["composed"]
                deadline = time.monotonic() + 180
                while time.monotonic() < deadline:
                    view = client.get(board_url).json()
                    assert view["artifactState"] != "failed", view
                    if view["composed"]:
                        break
                    time.sleep(0.025)
                assert view["composed"], view
            complete_ms = (time.perf_counter() - started) * 1000
            request_started = time.perf_counter()
            interaction = client.get(board_url + "/interaction")
            first_interaction_ms = (time.perf_counter() - request_started) * 1000
            assert interaction.status_code == 200, interaction.text
            assert interaction.json()["totalRows"] == args.rows * 4
            assert len(interaction.json()["rows"]) == 100
            app.state.clear_research_caches(project)
            request_started = time.perf_counter()
            cold = client.get(board_url + "/interaction")
            cold_interaction_ms = (time.perf_counter() - request_started) * 1000
            assert cold.status_code == 200 and cold.json()["totalRows"] == args.rows * 4
            formats = {}
            for format_, magic in [("png", b"\x89PNG"), ("svg", b"<?xml"), ("pdf", b"%PDF")]:
                artifact = client.get(board_url + "/artifact/" + format_)
                assert artifact.status_code == 200 and artifact.content.startswith(magic)
                formats[format_] = {
                    "bytes": len(artifact.content),
                    "sha256": hashlib.sha256(artifact.content).hexdigest(),
                }
                if format_ == "png":
                    images[deferred] = artifact.content
            report["deferred" if deferred else "legacySynchronous"] = {
                "durableRequestMs": round(save_ms, 2),
                "artifactsReadyMs": round(complete_ms, 2),
                "firstInteractionMs": round(first_interaction_ms, 2),
                "coldInteractionMs": round(cold_interaction_ms, 2),
                "serverTiming": response.headers.get("server-timing"),
                "generation": view.get("artifactTimings", {}),
                "artifacts": formats,
            }
        assert images[False] == images[True], "The queued path changed frozen PNG pixels"
        report["pngParity"] = True
        runs.append(report)
        assert client.delete("/api/v1/projects/" + project).status_code == 204
        print(
            json.dumps(
                {
                    k: v
                    for k, v in report.items()
                    if k == "repeat"
                    or k == "pngParity"
                    or isinstance(v, dict)
                    and "durableRequestMs" in v
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

peak_bytes = _process_peak_memory_bytes()
summary = {
    "scope": (
        "Current-host complete TestClient requests; real SQLite commits and all three "
        "formats. Browser/network/proxy and manual UX excluded. Alternating endpoint order, "
        "new project each trial. Core correctness unchanged; PNG bytes identical between "
        "paths. First interaction reuses the generated index; coldInteraction clears the "
        "cache and is not browser reopen. Peak memory covers both variants and the cold "
        "probe in one process. Final implementation only; not a before/after renderer "
        "speedup claim."
    ),
    "statistic": "maximum of three runs, including first; not p95",
    "rows": args.rows,
    "mixed": args.mixed,
    "datasetSha256": hashlib.sha256(payload).hexdigest(),
    "runs": runs,
    "maxMs": {
        variant: {
            key: max(run[variant][key] for run in runs)
            for key in [
                "durableRequestMs",
                "artifactsReadyMs",
                "firstInteractionMs",
                "coldInteractionMs",
            ]
        }
        for variant in ["legacySynchronous", "deferred"]
    },
    "processLifetimePeakMiB": peak_bytes / 2**20 if peak_bytes is not None else None,
    "implementationSha256": {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (root / "V2.0/api/labviz_api").glob("research*.py")
    },
    "scriptSha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
}
output = reports / (
    "durable-layout-"
    + ("mixed-" if args.mixed else "scientific-")
    + str(args.rows)
    + args.suffix
    + ".json"
)
output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
print(
    json.dumps(
        {
            "report": str(output),
            "maxMs": summary["maxMs"],
            "peakMiB": summary["processLifetimePeakMiB"],
        }
    ),
    flush=True,
)
