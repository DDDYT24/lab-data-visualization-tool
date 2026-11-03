"""Summarize complete-process measurements and comparable host payload sizes."""

from __future__ import annotations

import json
import statistics
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def summarize(trial: Path) -> dict:
    config = json.loads((trial / "launcher.json").read_text())
    host = json.loads((trial / "host.json").read_text())
    services = json.loads((trial / "services.json").read_text())
    roles = {host["pid"]: "host", services["controllerPid"]: "controller", services["launcher"]["pid"]: "launcher", services["api"]["pid"]: "api", services["web"]["pid"]: "node"}
    phases = {}
    peak = 0
    with (trial / "process-samples.jsonl").open() as stream:
        for line in stream:
            row = json.loads(line)
            values = phases.setdefault(row["phase"], {"private": [], "working": [], "roles": {}})
            values["private"].append(row["privateCommitBytesSum"])
            values["working"].append(row["workingSetBytesSum"])
            peak = max(peak, row["privateCommitBytesSum"])
            by_role = {}
            for item in row["processes"]:
                role = roles.get(item["pid"], "webview" if "webview" in item["name"].lower() else "other-owned")
                by_role[role] = by_role.get(role, 0) + item["privateCommitBytes"]
            for role, amount in by_role.items():
                values["roles"].setdefault(role, []).append(amount)
    return {"shell": config["shell"], "trial": str(trial), "peakPrivateCommitMiB": peak / 2**20, "phases": {name: {"samples": len(value["private"]), "medianPrivateCommitMiB": statistics.median(value["private"]) / 2**20, "medianWorkingSetSumMiB": statistics.median(value["working"]) / 2**20, "medianPrivateByRoleMiB": {role: statistics.median(amounts) / 2**20 for role, amounts in value["roles"].items()}} for name, value in phases.items()}}


def archive_size(directory: Path, destination: Path) -> dict:
    files = [p for p in directory.rglob("*") if p.is_file()]
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for source in sorted(files):
            archive.write(source, source.relative_to(directory).as_posix())
    return {"fileCount": len(files), "directoryBytes": sum(p.stat().st_size for p in files), "comparisonArchiveBytes": destination.stat().st_size}


def main() -> None:
    out = ROOT / "outputs/desktop-d01/benchmark"
    runs = json.loads((out / "runs.json").read_text())
    summaries = [summarize(Path(run["trial"])) for run in runs]
    record = {"scope": "This Windows host, pinned common V2.3 payload; first-profile and process-cold warm-profile starts are separate from a machine reboot", "processMeasurement": "Sum of owned private commit; working-set sum can double-count shared pages", "runs": runs, "resources": summaries, "automaticSuspension": False}
    for shell in ("tauri", "pywebview"):
        samples = [run for run in runs if run["shell"] == shell and run.get("repeat")]
        resources = [row for row in summaries if row["shell"] == shell and "metric-" in row["trial"]]
        record[shell] = {"firstProfileHomeReadyMs": samples[0]["timings"]["observedHomeReadyMs"], "warmProfileMedianHomeReadyMs": statistics.median(run["timings"]["observedHomeReadyMs"] for run in samples[1:]), "medianHomeIdlePrivateCommitMiB": statistics.median(row["phases"]["home-idle"]["medianPrivateCommitMiB"] for row in resources), "medianHiddenPrivateCommitMiB": statistics.median(row["phases"]["hidden-idle"]["medianPrivateCommitMiB"] for row in resources), "medianHostPrivateCommitMiB": statistics.median(row["phases"]["home-idle"]["medianPrivateByRoleMiB"]["host"] for row in resources)}
    (out / "comparison.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(json.dumps({shell: record[shell] for shell in ("tauri", "pywebview")}, indent=2))


if __name__ == "__main__":
    main()
