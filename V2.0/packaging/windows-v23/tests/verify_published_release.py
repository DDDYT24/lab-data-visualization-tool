"""Download a published RC and repeat the installed lifecycle on the uploaded assets."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from verify_portable_lifecycle import read, sha

HERE = Path(__file__).resolve().parent
REPOSITORY = "DDDYT24/lab-data-visualization-tool"
SCOPE = (
    "Uploaded candidate assets downloaded and verified on the current Windows host. "
    "Real installation, scientific workflows, reinstall, uninstall retention, and reopen "
    "ran on the downloaded installer. The previous-release upgrade and rollback ran in "
    "Windows packaging CI. Clean-machine and physical-device acceptance remain separate."
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tag", required=True, help="release tag, for example v2.3.0-rc.2"
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="new, empty directory"
    )
    args = parser.parse_args()

    assets = args.output / "assets"
    assets.mkdir(parents=True, exist_ok=False)
    subprocess.run(
        [
            "gh",
            "release",
            "download",
            args.tag,
            "--repo",
            REPOSITORY,
            "--dir",
            str(assets),
        ],
        check=True,
    )
    sums = dict(
        reversed(line.split("  ", 1))
        for line in (assets / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines()
    )
    mismatched = [name for name, digest in sums.items() if sha(assets / name) != digest]
    if mismatched:
        raise SystemExit(f"Checksum mismatch: {mismatched}")
    provenance = read(assets / "release-provenance.json")
    installer = f"LabViz-V23-{args.tag.removeprefix('v')}-Setup.exe"
    if sums[installer] != provenance["installerSha256"]:
        raise SystemExit("The installer digest differs from the provenance record.")

    # verify_installer.py reads a build and a payload directory; rebuild both from the assets.
    build = args.output / "b"
    payload = args.output / "p"
    build.mkdir()
    payload.mkdir()
    os.link(assets / installer, build / installer)
    (build / "build-verification.json").write_text(
        json.dumps(
            {
                "compilerExitCode": 0,
                "appId": read(HERE.parent / "identity.json")["appId"],
                "candidateId": provenance["candidateId"],
                "installers": {installer: provenance["installerSha256"]},
            }
        ),
        encoding="utf-8",
    )
    (payload / "payload-manifest.json").write_bytes(
        (assets / "payload-sha256.json").read_bytes()
    )
    installed = args.output / "i"
    result = subprocess.run(
        [
            sys.executable,
            str(HERE / "verify_installer.py"),
            "--build",
            str(build),
            "--payload",
            str(payload),
            "--output",
            str(installed),
        ],
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        check=False,
    )
    receipt = read(installed / "verification.json")
    stats = read(installed / "installed-browser/browser.json")["stats"]
    verification = {
        "passed": result.returncode == 0 and receipt["passed"],
        "sourceCommit": provenance["sourceCommit"],
        "candidateId": provenance["candidateId"],
        "installerSha256": provenance["installerSha256"],
        "buildWorkflowRun": provenance["buildWorkflowRun"],
        "downloadedAssetChecksumsVerified": True,
        "installedLifecycleChecks": receipt["checks"],
        "installedScientificBrowserTests": stats["expected"]
        + stats["unexpected"]
        + stats["flaky"],
        "browserFailures": stats["unexpected"],
        "browserSkipped": stats["skipped"],
        "browserRetries": stats["flaky"],
        "unsigned": True,
        "stableAcceptanceComplete": False,
        "pendingAcceptance": read(assets / "release-verification.json")[
            "pendingAcceptance"
        ],
        "scope": SCOPE,
    }
    text = json.dumps(verification, indent=2) + "\n"
    (args.output / "DOWNLOAD-VERIFICATION.json").write_text(text, encoding="utf-8")
    print(text, end="")
    return 0 if verification["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
