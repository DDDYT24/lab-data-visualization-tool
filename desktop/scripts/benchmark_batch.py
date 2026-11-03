"""Run comparison trials serially so scientific workloads do not compete."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    output = ROOT / "outputs/desktop-d01/benchmark"
    output.mkdir(parents=True, exist_ok=False)
    previous = {}
    cases = [("tauri", 1), ("pywebview", 1), ("pywebview", 2), ("tauri", 2), ("tauri", 3), ("pywebview", 3)]
    results = []
    for shell, repeat in cases:
        label = f"metric-{shell}-{repeat}"
        args = [sys.executable, str(ROOT / "desktop/scripts/run_trial.py"), "--shell", shell, "--label", label, "--scenario", "smoke", "--auto-exit"]
        if shell in previous:
            args += ["--resume-from", str(previous[shell])]
        with (output / f"{label}.log").open("wb") as log:
            process = subprocess.run(args, cwd=ROOT, stdout=log, stderr=log, timeout=300, creationflags=subprocess.CREATE_NO_WINDOW)
        trial = Path.home() / "LabVizD1" / label
        result = json.loads((trial / "workflow-smoke.json").read_text())
        exited = json.loads((trial / "exit.json").read_text())
        assert process.returncode == 0 and result["passed"] and exited["hostExitCode"] == 0 and not exited["survivingOwnedProcesses"], label
        previous[shell] = trial
        results.append({"shell": shell, "repeat": repeat, "trial": str(trial), "timings": result["timings"]})
        (output / "runs.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(json.dumps(results[-1]), flush=True)
    for shell in ("tauri", "pywebview"):
        label = f"large-{shell}"
        args = [sys.executable, str(ROOT / "desktop/scripts/run_trial.py"), "--shell", shell, "--label", label, "--scenario", "large", "--auto-exit"]
        with (output / f"{label}.log").open("wb") as log:
            process = subprocess.run(args, cwd=ROOT, stdout=log, stderr=log, timeout=900, creationflags=subprocess.CREATE_NO_WINDOW)
        trial = Path.home() / "LabVizD1" / label
        result = json.loads((trial / "workflow-large.json").read_text())
        exited = json.loads((trial / "exit.json").read_text())
        assert process.returncode == 0 and result["passed"] and exited["hostExitCode"] == 0 and not exited["survivingOwnedProcesses"], label
        results.append({"shell": shell, "scenario": "100k-four-panels", "trial": str(trial), "timings": result["timings"]})
        (output / "runs.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(json.dumps(results[-1]), flush=True)


if __name__ == "__main__":
    main()
