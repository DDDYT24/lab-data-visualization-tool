"""Run synthetic scientific workflows against the actual bundled Web and API."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path
from urllib.parse import urlsplit

from verify_portable_lifecycle import local_key, read, write

ROOT = Path(__file__).resolve().parents[4]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--running-record", type=Path, required=True)
    parser.add_argument("--profile-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    profile = args.profile_root.resolve()
    fixture_root = (Path.home() / "LabVizV23Acceptance").resolve()
    assert (
        profile.is_relative_to(fixture_root)
        and not profile.is_symlink()
        and not profile.is_junction()
    )
    record = read(args.running_record)
    assert record["appId"] == "55B46BB7-101E-46DC-951A-3DE7E402EDD0"
    assert record["profileId"] == read(profile / "profile.json")["profileId"]
    url = urlsplit(record["webUrl"])
    assert (
        url.scheme == "http"
        and url.hostname == "127.0.0.1"
        and url.port
        and url.path in ("", "/")
        and not url.username
        and not url.password
        and not url.query
        and not url.fragment
    )
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    node = Path(record["web"]["executable"])
    assert node.is_relative_to(fixture_root) and node.name == "node.exe"
    env = {
        **os.environ,
        "PLAYWRIGHT_BASE_URL": record["webUrl"],
        "LABVIZ_E2E_LIVE": "1",
        "LABVIZ_E2E_INTERFACE_MODE": "classic",
        "LABVIZ_E2E_LOCAL_ACCESS_KEY": local_key(profile / "access.dpapi"),
        "PLAYWRIGHT_JSON_OUTPUT_NAME": str(output / "browser.json"),
        "NEXT_TELEMETRY_DISABLED": "1",
        "PYTHONIOENCODING": "utf-8",
    }
    env.pop("CI", None)
    files = [
        "live-controls.spec.ts",
        "live-research.spec.ts",
        "live-research-charts.spec.ts",
        "live-research-figures.spec.ts",
        "live-research-reuse.spec.ts",
        "live-local-history.spec.ts",
        "live-samples.spec.ts",
        "overview-interface.spec.ts",
    ]
    command = [
        str(node),
        str(ROOT / "V2.0/web/node_modules/@playwright/test/cli.js"),
        "test",
        *["e2e/" + name for name in files],
        "--project=chromium",
        "--workers=1",
        "--retries=0",
        "--timeout=180000",
        "--reporter=json,line",
        "--output=" + str(output / "test-results"),
    ]
    with (output / "browser.log").open("wb") as log:
        result = subprocess.run(
            command,
            cwd=ROOT / "V2.0/web",
            env=env,
            stdout=log,
            stderr=log,
            creationflags=subprocess.CREATE_NO_WINDOW,
            check=False,
        )
    report = read(output / "browser.json")
    stats = report["stats"]
    passed = (
        result.returncode == 0
        and stats["expected"] > 0
        and not any(stats[name] for name in ("unexpected", "skipped", "flaky"))
    )
    write(
        output / "verification.json",
        {
            "passed": passed,
            "candidateId": record["candidateId"],
            "stats": stats,
            "scope": "Actual self-contained runtime and synthetic browser workflows on this host; physical offline, assistive speech, and real-phone acceptance remain separate.",
        },
    )
    if not passed:
        raise RuntimeError(
            "Packaged scientific browser workflows failed; private reports retained."
        )
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
