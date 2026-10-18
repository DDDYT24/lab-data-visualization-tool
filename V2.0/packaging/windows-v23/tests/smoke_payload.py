"""Activate an exact staged payload in an owned short-path Windows fixture."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from uuid import uuid4

from verify_portable_lifecycle import read, write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--payload", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = args.payload.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    test = Path.home() / "LabVizV23Acceptance" / ("r" + uuid4().hex[:8])
    test.mkdir(parents=True, exist_ok=False)
    program = test / "program"
    program.mkdir()
    meta = read(payload / "candidate-build.json")
    version = program / "versions" / meta["candidateId"][:24]
    manifest = read(payload / "payload-manifest.json")
    assert max(len(str(version / p)) for p in manifest["files"]) < 259
    write(output / "fixture.json", {"root": str(test)})
    start = time.monotonic()
    try:
        shutil.copytree(payload, version)
        shutil.copytree(payload / "bin", program / "bin")
        shutil.copyfile(payload / "identity.json", program / "installation.json")
        write(
            program / "pending-candidate.json",
            {k: meta[k] for k in ("appId", "candidateId")},
        )
        env = {
            **os.environ,
            "LOCALAPPDATA": str(test / "appdata"),
            "PYTHONIOENCODING": "utf-8",
        }
        command = [
            str(
                Path(os.environ["SystemRoot"])
                / "System32/WindowsPowerShell/v1.0/powershell.exe"
            ),
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(program / "bin/start-v23-installed.ps1"),
            "-SkipOpenBrowser",
            "-HealthCheckOnly",
        ]
        with (output / "launcher.log").open("wb") as log:
            result = subprocess.run(
                command,
                env=env,
                stdout=log,
                stderr=log,
                check=False,
                timeout=450,
                creationflags=subprocess.CREATE_NO_WINDOW,
            )
        checks = {
            "launcherExitZero": result.returncode == 0,
            "activationCommitted": (program / "current-candidate.json").is_file(),
            "pendingRemoved": not (program / "pending-candidate.json").exists(),
            "journalRemoved": not (program / "upgrade.json").exists(),
            "runningMarkerRemoved": not (
                test / "appdata/LabVizV23/logs/running.json"
            ).exists(),
        }
        status = {
            "exitCode": result.returncode,
            "checks": checks,
            "passed": all(checks.values()),
            "seconds": time.monotonic() - start,
            "program": str(program),
            "profileRoot": str(test / "appdata/LabVizV23"),
            "candidateId": meta["candidateId"],
        }
    except BaseException as error:
        (output / "failure.txt").write_text(repr(error), encoding="utf-8")
        status = {
            "passed": False,
            "error": type(error).__name__,
            "program": str(program),
        }
        raise
    finally:
        write(output / "verification.json", status)
    if not status["passed"]:
        raise RuntimeError(
            "Native payload activation failed; inspect retained fixture logs."
        )
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
