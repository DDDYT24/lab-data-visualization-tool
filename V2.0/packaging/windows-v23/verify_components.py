"""Generate a fresh receipt for this checkout's installed identity and data contracts."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

from stage_candidate import APP_ID, ROOT, SOURCE, checked, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--test-python", type=Path, required=True)
    args = parser.parse_args()
    output = checked(args.output)
    output.mkdir(parents=True, exist_ok=False)
    suite = output / "identity-data.xml"
    commands = [
        (
            [
                str(checked(args.test_python)),
                "-m",
                "pytest",
                "tests/test_installed_v23_lifecycle.py",
                "tests/test_local_profile_history.py",
                "tests/test_v23_source_data.py",
                "--junitxml=" + str(suite),
            ],
            ROOT / "V2.0/api",
            "python.log",
        ),
        (
            [
                str(
                    Path(os.environ["SystemRoot"])
                    / "System32/WindowsPowerShell/v1.0/powershell.exe"
                ),
                "-NoProfile",
                "-ExecutionPolicy",
                "Bypass",
                "-File",
                str(SOURCE / "tests/test-contract.ps1"),
                "-TestRoot",
                str(output / "native"),
            ],
            ROOT,
            "native.log",
        ),
    ]
    for command, cwd, log_name in commands:
        with (output / log_name).open("wb") as log:
            result = subprocess.run(
                command,
                cwd=cwd,
                stdout=log,
                stderr=log,
                env={**os.environ, "PYTHONIOENCODING": "utf-8"},
                check=False,
            )
        if result.returncode:
            raise RuntimeError("Component verification failed: " + log_name)
    xml = ET.parse(suite).getroot()
    if len(list(xml.iter("testcase"))) < 29 or any(
        list(xml.iter(name)) for name in ("failure", "error", "skipped")
    ):
        raise RuntimeError("Identity/data tests were incomplete.")
    native = output / "native/verification.json"
    data = json.loads(native.read_text(encoding="utf-8-sig"))
    if not data["passed"] or not all(data["checks"].values()):
        raise RuntimeError("Native isolation tests were incomplete.")
    paths = list((SOURCE / "bin").glob("*")) + [SOURCE / "identity.json"]
    paths += [
        ROOT / "V2.0/api/labviz_api" / name
        for name in (
            "installed_profile.py",
            "installed_lifecycle.py",
            "installed_server.py",
            "source_data.py",
            "source_data_routes.py",
            "main.py",
        )
    ]
    receipt = {
        "appId": APP_ID,
        "identityAndDataContractPassed": True,
        "sourceSha256": {
            p.relative_to(ROOT).as_posix(): sha(p) for p in paths if p.is_file()
        },
        "pythonTests": str(suite),
        "nativeTests": str(native),
        "evidence": [{"path": str(p), "sha256": sha(p)} for p in (suite, native)],
    }
    (output / "receipt.json").write_text(
        json.dumps(receipt, indent=2), encoding="utf-8"
    )
    print("Fresh identity/data and native isolation receipt generated.")


if __name__ == "__main__":
    main()
