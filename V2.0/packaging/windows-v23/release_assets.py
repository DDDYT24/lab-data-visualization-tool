"""Assemble public RC assets only after the exact payload passes local acceptance."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

from compile_candidate import REQUIRED_CHECKS
from stage_candidate import APP_ID, ROOT, SOURCE, checked, read, sha, source_inputs

INSTALLER_CHECKS = {
    "freshUnicodeDirectoryInstallationWithSeparateIdentity",
    "freshInstalledBundledRuntimeHealth",
    "installedFilesMatchVerifiedPayload",
    "installedScientificBrowserWorkflows",
    "sameVersionReinstallPreservesProfileAndFrozenBytes",
    "uninstallRetainsDataAndRemovesOwnRegistryIdentity",
    "reinstallReopensRetainedProfile",
    "acceptanceInstallerRemovedDataRetained",
}
PENDING = [
    "Clean-machine acceptance",
    "Complete workflow with networking physically disconnected",
    "Actual desktop 200% zoom and assistive speech",
    "Physical iPhone workflow",
    "Researcher pilot with 3-5 participants",
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-root", type=Path, required=True)
    args = parser.parse_args()
    build_root = checked(args.build_root)
    if not build_root.is_relative_to(ROOT / "outputs/v23-release"):
        raise ValueError("Use this checkout's private build root.")
    payload = build_root / "payload"
    meta = read(payload / "candidate-build.json")
    inputs = read(payload / "source-inputs.json")
    if meta["sourceDirty"] or inputs != source_inputs():
        raise ValueError(
            "Public assets require the unchanged committed source used for staging."
        )
    runtime = read(SOURCE / "runtimes.json")
    if (
        meta["pythonVersion"] != runtime["python"]["version"]
        or meta["nodeVersion"] != runtime["node"]["version"]
    ):
        raise ValueError("Runtime versions differ from the pinned inputs.")
    portable = read(build_root / "portable-lifecycle/verification.json")
    installed = read(build_root / "installed-lifecycle/verification.json")
    compiler = read(build_root / "installer/build-verification.json")
    if (
        compiler["appId"] != APP_ID
        or compiler["candidateId"] != meta["candidateId"]
        or compiler["compilerExitCode"] != 0
        or len(compiler["installers"]) != 1
    ):
        raise ValueError(
            "Installer compilation does not identify this exact candidate."
        )
    for receipt, required in (
        (portable, REQUIRED_CHECKS),
        (installed, INSTALLER_CHECKS),
    ):
        if (
            not receipt.get("passed")
            or receipt.get("inProgress")
            or receipt.get("candidateId") != meta["candidateId"]
            or not required.issubset(receipt["checks"])
            or not all(receipt["checks"].values())
        ):
            raise ValueError(
                "Required native or installed scenarios have not all passed."
            )
    xml = ET.parse(build_root / "api-local.xml").getroot()
    if any(list(xml.iter(name)) for name in ("failure", "error", "skipped")):
        raise ValueError("Local API regression was incomplete.")
    unit = read(build_root / "web-unit.json")
    if not unit["success"] or unit["numPassedTests"] != unit["numTotalTests"]:
        raise ValueError("Frontend unit verification was incomplete.")
    state = read(build_root / "pipeline.json")
    for phase in ("prepare", "verify-source", "stage", "portable", "installer"):
        matching = [r for r in state["runs"] if r["phase"] == phase]
        if not matching or not matching[-1]["passed"]:
            raise ValueError("Pipeline phase is incomplete: " + phase)
    name, digest = next(iter(compiler["installers"].items()))
    installer = build_root / "installer" / name
    if sha(installer) != digest or installed["installerSha256"] != digest:
        raise ValueError("The installed and published installer bytes differ.")
    destination = build_root / "release"
    destination.mkdir(exist_ok=False)
    shutil.copyfile(installer, destination / name)
    shutil.copyfile(
        payload / "payload-manifest.json", destination / "payload-sha256.json"
    )
    shutil.copyfile(payload / "source-inputs.json", destination / "source-inputs.json")
    with zipfile.ZipFile(
        destination / "LabViz-2.3.0-rc.1-Test-Data.zip",
        "w",
        compression=zipfile.ZIP_DEFLATED,
    ) as archive:
        for folder in (
            ROOT / "V2.0/api/samples/v22",
            ROOT / "V2.0/contracts/fixtures/v23",
        ):
            for path in sorted(folder.rglob("*")):
                if path.is_file() and path.suffix.lower() in {
                    ".csv",
                    ".tsv",
                    ".txt",
                    ".json",
                    ".xlsx",
                }:
                    archive.write(path, path.relative_to(ROOT / "V2.0").as_posix())
        archive.writestr(
            "README.txt",
            "These fixtures are synthetic. Use them to evaluate import, preparation, plotting, and export. Preserve the source data and compare scientific values with the versioned fixture expectations.\n",
        )
    verification = {
        "appId": APP_ID,
        "version": meta["version"],
        "candidateId": meta["candidateId"],
        "sourceCommit": meta["sourceCommit"],
        "localApiTests": len(list(xml.iter("testcase"))),
        "frontendUnitTests": unit["numTotalTests"],
        "portableChecks": portable["checks"],
        "installedChecks": installed["checks"],
        "pendingAcceptance": PENDING,
        "unsigned": True,
        "stableAcceptanceComplete": False,
        "scope": "Current Windows host and isolated fixtures. Linux integration CI and publication/download verification are separate records.",
    }
    provenance = {
        **meta,
        "installerSha256": digest,
        "runtimeInputs": runtime,
        "componentReceiptSha256": sha(build_root / "components/receipt.json"),
        "portableEvidenceSha256": sha(
            build_root / "portable-lifecycle/verification.json"
        ),
        "installedEvidenceSha256": sha(
            build_root / "installed-lifecycle/verification.json"
        ),
        "sourceInputManifestSha256": sha(payload / "source-inputs.json"),
        "buildWorkflowRun": (
            "https://github.com/"
            + os.environ["GITHUB_REPOSITORY"]
            + "/actions/runs/"
            + os.environ["GITHUB_RUN_ID"]
        )
        if os.environ.get("GITHUB_RUN_ID")
        else None,
    }
    (destination / "release-verification.json").write_text(
        json.dumps(verification, indent=2), encoding="utf-8"
    )
    (destination / "release-provenance.json").write_text(
        json.dumps(provenance, indent=2), encoding="utf-8"
    )
    note = (
        "# V2.3 RC verification\n\nSource commit: `"
        + meta["sourceCommit"]
        + "`. Candidate: `"
        + meta["candidateId"]
        + "`.\n\nThe exact bundled runtime and installed UI completed the recorded scientific workflows and data lifecycle checks. See release-verification.json for counts and scenarios, and release-provenance.json for pinned inputs and hashes. The installer is unsigned.\n\nPending formal acceptance:\n\n"
        + "\n".join("- " + item for item in PENDING)
        + "\n\nThe development lint-chain audit findings are documented in the RC release notes. This preview retains the stable-release acceptance boundary.\n"
    )
    (destination / "VERIFICATION.md").write_text(note, encoding="utf-8")
    sums = [
        sha(path) + "  " + path.name
        for path in sorted(destination.iterdir())
        if path.is_file()
    ]
    (destination / "SHA256SUMS.txt").write_text(
        "\n".join(sums) + "\n", encoding="utf-8"
    )
    print(
        "RC assets assembled from the verified committed candidate. Publication still requires completed PR checks and download verification."
    )


if __name__ == "__main__":
    main()
