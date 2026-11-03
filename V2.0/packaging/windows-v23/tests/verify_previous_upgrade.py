"""Upgrade a real published V2.3 installation to the candidate, then roll it back."""

from __future__ import annotations

import argparse
import os
import shutil
import sqlite3
import subprocess
import time
import urllib.request
from contextlib import closing
from pathlib import Path
from uuid import uuid4

from verify_installer import APP_ID, installed_entry
from verify_portable_lifecycle import read, sentinel_directory_value, sha, write

PREVIOUS = Path(__file__).resolve().parents[1] / "previous-release.json"


def obtain(release, folder: Path, cached: Path | None) -> Path:
    item = release["installer"]
    target = folder / item["name"]
    folder.mkdir(parents=True, exist_ok=True)
    if cached:
        shutil.copyfile(cached, target)
    else:
        temporary = target.with_suffix(".download")
        with (
            urllib.request.urlopen(item["url"], timeout=300) as response,
            temporary.open("wb") as stream,
        ):
            shutil.copyfileobj(response, stream)
        temporary.rename(target)
    if sha(target) != item["sha256"]:
        raise ValueError("The published installer differs from its pinned hash.")
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--payload", type=Path, required=True)
    parser.add_argument(
        "--previous-installer",
        type=Path,
        help="A local copy of the pinned published installer, checked by hash.",
    )
    args = parser.parse_args()
    release = read(PREVIOUS)
    build = read(args.build / "build-verification.json")
    assert build["compilerExitCode"] == 0 and build["appId"] == APP_ID
    exe_name, digest = next(iter(build["installers"].items()))
    candidate = (args.build / exe_name).resolve()
    assert sha(candidate) == digest
    assert not installed_entry(), (
        "An existing V2.3 installation must be retained; do not adopt its registry entry."
    )
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    previous_installer = obtain(release, output / "previous", args.previous_installer)
    fixture = Path.home() / "LabVizV23Acceptance" / ("u" + uuid4().hex[:8])
    fixture.mkdir(parents=True, exist_ok=False)
    program = fixture / "安装"
    profile = fixture / "appdata/LabVizV23"
    env = {
        **os.environ,
        "LOCALAPPDATA": str(fixture / "appdata"),
        "PYTHONIOENCODING": "utf-8",
    }
    ps = str(
        Path(os.environ["SystemRoot"])
        / "System32/WindowsPowerShell/v1.0/powershell.exe"
    )
    checks = {}
    started = time.monotonic()
    write(output / "fixture.json", {"root": str(fixture), "program": str(program)})

    def record(name):
        checks[name] = True
        write(
            output / "verification.json",
            {
                "passed": False,
                "inProgress": True,
                "candidateId": build["candidateId"],
                "checks": checks,
            },
        )
        print(name + ": passed", flush=True)

    def execute(command, label):
        with (output / (label + ".stdout.log")).open("wb") as log:
            result = subprocess.run(
                command,
                env=env,
                stdout=log,
                stderr=log,
                timeout=600,
                creationflags=subprocess.CREATE_NO_WINDOW,
                check=False,
            )
        assert result.returncode == 0, (
            f"{label}: exit {result.returncode}; inspect retained installer logs."
        )

    def install(executable, label):
        execute(
            [
                str(executable),
                "/VERYSILENT",
                "/SUPPRESSMSGBOXES",
                "/NORESTART",
                "/SP-",
                "/NOICONS",
                "/DIR=" + str(program),
                "/LOG=" + str(output / (label + ".inno.log")),
            ],
            label,
        )
        return read(program / "pending-candidate.json")["candidateId"]

    def start(label, *extra):
        execute(
            [
                ps,
                "-NoProfile",
                "-ExecutionPolicy",
                "Bypass",
                "-File",
                str(program / "bin/start-v23-installed.ps1"),
                *extra,
                "-SkipOpenBrowser",
                "-HealthCheckOnly",
            ],
            label,
        )
        return (
            read(program / "current-candidate.json")["candidateId"],
            read(profile / "data-schema.json")["dataEpoch"],
        )

    def seed(value):
        with closing(sqlite3.connect(profile / "data/labviz.db")) as db, db:
            db.execute(
                "CREATE TABLE IF NOT EXISTS installed_acceptance (id TEXT PRIMARY KEY, value REAL)"
            )
            db.execute(
                "INSERT OR REPLACE INTO installed_acceptance VALUES ('stable-science-id', ?)",
                (value,),
            )

    def sentinel():
        return sentinel_directory_value(profile / "data")

    try:
        previous = install(previous_installer, "previous-install")
        assert previous != build["candidateId"]
        assert start("previous-health") == (previous, release["dataEpoch"])
        seed(1.25)
        frozen = profile / "data/objects/upgrade-frozen.svg"
        frozen.parent.mkdir(parents=True, exist_ok=True)
        frozen.write_bytes(b"<svg>preserve exact frozen artifact across upgrades</svg>")
        frozen_sha = sha(frozen)
        profile_id = read(profile / "profile.json")["profileId"]
        record("publishedReleaseInstalledAndHealthy")
        assert install(candidate, "upgrade-install") == build["candidateId"]
        assert start("upgrade-health") == (
            build["candidateId"],
            read(args.payload / "candidate-build.json")["dataEpoch"],
        )
        receipt = read(program / "last-upgrade.json")
        assert (
            receipt["previous"] == previous
            and receipt["candidate"] == build["candidateId"]
        )
        assert sentinel() == 1.25 and sha(frozen) == frozen_sha
        assert read(profile / "profile.json")["profileId"] == profile_id
        assert (program / "versions" / previous[:24]).is_dir()
        record("upgradeFromPublishedReleaseKeepsProfileAndData")
        seed(3.75)
        assert start("rollback-health", "-Rollback") == (
            previous,
            release["dataEpoch"],
        )
        assert sentinel() == 1.25 and sha(frozen) == frozen_sha
        assert any(
            sentinel_directory_value(p) == 3.75 for p in profile.glob("data.retained-*")
        )
        record("rollbackToPublishedReleaseRestoresItsSnapshot")
        execute(
            [
                str(program / "unins000.exe"),
                "/VERYSILENT",
                "/SUPPRESSMSGBOXES",
                "/NORESTART",
                "/LOG=" + str(output / "uninstall.inno.log"),
            ],
            "uninstall",
        )
        assert not installed_entry() and profile.exists()
        record("upgradedInstallationRemovedDataRetained")
        passed = True
    except BaseException as error:
        (output / "failure.txt").write_text(repr(error), encoding="utf-8")
        passed = False
        raise
    finally:
        write(
            output / "verification.json",
            {
                "passed": locals().get("passed", False),
                "inProgress": False,
                "candidateId": build["candidateId"],
                "installerSha256": digest,
                "previousRelease": release,
                "checks": checks,
                "program": str(program),
                "retainedProfile": str(profile),
                "seconds": time.monotonic() - started,
                "scope": "Current Windows host; the published installer upgraded in place, not a clean machine.",
            },
        )


if __name__ == "__main__":
    main()
