"""Compile only the exact self-contained payload that passed native lifecycle checks."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from stage_candidate import APP_ID, SOURCE, checked, read, sha, source_inputs

REQUIRED_CHECKS = {
    "occupiedPortsPreservedAndReallocated",
    "BundledWebProxyAuthenticatedSampleCreation",
    "duplicateStartPreservesRunningInstance",
    "activeProfileBackupRefused",
    "authenticatedGracefulStop",
    "launcherCrashClosesOwnedServicesAndRecovers",
    "completeBackupAndIndependentRestore",
    "restoredProjectReopensWithIndependentCookie",
    "sameVersionGenerationUpgradePreservesData",
    "compatibleRollbackRetainsNewerData",
    "failedCandidateAutomaticallyRestoresPrevious",
    "futureEpochRollbackRefusedWithoutDataMutation",
    "exactProfileDeletionPreservesOtherProfile",
    "packagedScientificBrowserWorkflows",
    "legacyV22CopyImportPreservesIdsAndFrozenBytes",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--payload", type=Path, required=True)
    parser.add_argument("--lifecycle-evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compiler", type=Path, required=True)
    parser.add_argument("--allow-development", action="store_true")
    args = parser.parse_args()
    payload = checked(args.payload)
    manifest = read(payload / "payload-manifest.json")
    metadata = read(payload / "candidate-build.json")
    inputs = read(payload / "source-inputs.json")
    if inputs != source_inputs():
        raise ValueError("The source checkout changed after staging.")
    if metadata["sourceDirty"] and not args.allow_development:
        raise ValueError(
            "Commit the reviewed source before building a public release candidate."
        )
    identity = {
        key: manifest[key] for key in ("appId", "version", "dataEpoch", "files")
    }
    candidate = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if (
        identity["appId"] != APP_ID
        or candidate != metadata["candidateId"]
        or candidate != manifest["candidateId"]
    ):
        raise ValueError("Payload metadata and full candidate identity differ.")
    members = [
        path
        for path in sorted(payload.rglob("*"))
        if path.is_file()
        and path.name
        not in (
            "candidate-build.json",
            "payload-manifest.json",
            "portable-lifecycle-verified.json",
        )
    ]
    with ThreadPoolExecutor(max_workers=16) as pool:
        actual = dict(
            pool.map(
                lambda path: (path.relative_to(payload).as_posix(), sha(path)), members
            )
        )
    if actual != identity["files"]:
        raise ValueError("The payload changed after native lifecycle verification.")
    for name in (SOURCE / "bin").iterdir():
        if name.is_file() and sha(name) != sha(payload / "bin" / name.name):
            raise ValueError("Package controllers differ from current reviewed source.")
    evidence_path = checked(args.lifecycle_evidence)
    evidence = read(evidence_path)
    if (
        not evidence.get("passed")
        or evidence.get("inProgress")
        or evidence.get("candidateId") != candidate
    ):
        raise ValueError(
            "Native lifecycle evidence is incomplete or identifies another payload."
        )
    if not REQUIRED_CHECKS.issubset(evidence["checks"]) or not all(
        evidence["checks"].values()
    ):
        raise ValueError("Required native lifecycle scenarios have not all passed.")
    marker = {
        "appId": APP_ID,
        "candidateId": candidate,
        "evidenceSha256": sha(evidence_path),
    }
    marker_path = payload / "portable-lifecycle-verified.json"
    if marker_path.exists() and read(marker_path) != marker:
        raise ValueError("An existing lifecycle receipt cannot be replaced.")
    marker_path.write_text(json.dumps(marker, indent=2), encoding="utf-8")
    output = checked(args.output)
    if output.exists():
        raise ValueError(
            "Use a new installer output directory to preserve earlier results."
        )
    output.mkdir(parents=True)
    command = [
        str(checked(args.compiler)),
        "/DCandidateRoot=" + str(payload),
        "/DCandidateId=" + candidate,
        "/DCandidateDir=" + candidate[:24],
        "/DOutputDir=" + str(output),
        str(SOURCE / "LabVizV23.iss"),
    ]
    with (output / "compiler.log").open("wb") as log:
        result = subprocess.run(command, stdout=log, stderr=log, check=False)
    executables = sorted(output.glob("*-Setup.exe"))
    record = {
        "appId": APP_ID,
        "candidateId": candidate,
        "compilerExitCode": result.returncode,
        "nativeLifecycleEvidenceSha256": sha(evidence_path),
        "installers": {p.name: sha(p) for p in executables},
        "releaseAccepted": False,
    }
    (output / "build-verification.json").write_text(
        json.dumps(record, indent=2), encoding="utf-8"
    )
    if result.returncode or len(executables) != 1:
        raise ValueError(
            "Installer compilation failed; inspect the retained compiler log."
        )
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
