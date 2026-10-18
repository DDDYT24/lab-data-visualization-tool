"""Install and uninstall the candidate in a private Windows acceptance directory."""

from __future__ import annotations

import argparse
import os
import sqlite3
import subprocess
import sys
import time
import winreg
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from pathlib import Path
from uuid import uuid4

from verify_portable_lifecycle import read, sha, write

APP_ID = "55B46BB7-101E-46DC-951A-3DE7E402EDD0"
REGISTRY_KEY = (
    "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\{" + APP_ID + "}_is1"
)


def installed_entry():
    entries = []
    for view in (winreg.KEY_WOW64_64KEY, winreg.KEY_WOW64_32KEY):
        try:
            with winreg.OpenKey(
                winreg.HKEY_CURRENT_USER, REGISTRY_KEY, 0, winreg.KEY_READ | view
            ) as key:
                entries.append(winreg.QueryValueEx(key, "InstallLocation")[0])
        except FileNotFoundError:
            pass
    return entries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--payload", type=Path, required=True)
    args = parser.parse_args()
    build = read(args.build / "build-verification.json")
    assert build["compilerExitCode"] == 0 and build["appId"] == APP_ID
    assert len(build["installers"]) == 1
    exe_name, digest = next(iter(build["installers"].items()))
    executable = (args.build / exe_name).resolve()
    assert sha(executable) == digest
    assert not installed_entry(), (
        "An existing V2.3 installation must be retained; do not adopt its registry entry."
    )
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    fixture = Path.home() / "LabVizV23Acceptance" / ("i" + uuid4().hex[:8])
    fixture.mkdir(parents=True, exist_ok=False)
    program = fixture / "安装"
    profile = fixture / "appdata/LabVizV23"
    env = {
        **os.environ,
        "LOCALAPPDATA": str(fixture / "appdata"),
        "PYTHONIOENCODING": "utf-8",
    }
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

    def execute(command, label, *, expect=0):
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
        assert result.returncode == expect, (
            f"{label}: exit {result.returncode}; inspect retained installer logs."
        )

    def install(label):
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
        assert read(program / "installation.json")["appId"] == APP_ID
        assert (
            read(program / "pending-candidate.json")["candidateId"]
            == build["candidateId"]
        )
        assert installed_entry(), (
            "Installer did not create its separate uninstall identity."
        )

    def health(label):
        ps = str(
            Path(os.environ["SystemRoot"])
            / "System32/WindowsPowerShell/v1.0/powershell.exe"
        )
        execute(
            [
                ps,
                "-NoProfile",
                "-ExecutionPolicy",
                "Bypass",
                "-File",
                str(program / "bin/start-v23-installed.ps1"),
                "-SkipOpenBrowser",
                "-HealthCheckOnly",
            ],
            label,
        )
        assert (
            read(program / "current-candidate.json")["candidateId"]
            == build["candidateId"]
        )

    try:
        install("fresh-install")
        record("freshUnicodeDirectoryInstallationWithSeparateIdentity")
        health("fresh-installed-health")
        record("freshInstalledBundledRuntimeHealth")
        manifest = read(args.payload / "payload-manifest.json")
        assert manifest["candidateId"] == build["candidateId"]
        version = program / "versions" / build["candidateId"][:24]
        with ThreadPoolExecutor(max_workers=16) as pool:
            assert all(
                pool.map(
                    lambda item: sha(version / item[0]) == item[1],
                    manifest["files"].items(),
                )
            )
        record("installedFilesMatchVerifiedPayload")
        ps = str(
            Path(os.environ["SystemRoot"])
            / "System32/WindowsPowerShell/v1.0/powershell.exe"
        )
        with (output / "installed-ui-launcher.log").open("wb") as log:
            process = subprocess.Popen(
                [
                    ps,
                    "-NoProfile",
                    "-ExecutionPolicy",
                    "Bypass",
                    "-File",
                    str(program / "bin/start-v23-installed.ps1"),
                    "-SkipOpenBrowser",
                ],
                env=env,
                stdout=log,
                stderr=log,
                creationflags=subprocess.CREATE_NO_WINDOW,
            )
            try:
                running = profile / "logs/running.json"
                deadline = time.monotonic() + 240
                while (
                    not running.exists()
                    and time.monotonic() < deadline
                    and process.poll() is None
                ):
                    time.sleep(0.5)
                assert running.is_file() and process.poll() is None
                execute(
                    [
                        sys.executable,
                        str(Path(__file__).with_name("verify_packaged_browser.py")),
                        "--running-record",
                        str(running),
                        "--profile-root",
                        str(profile),
                        "--output",
                        str(output / "installed-browser"),
                    ],
                    "installed-scientific-browser",
                )
                execute(
                    [
                        ps,
                        "-NoProfile",
                        "-ExecutionPolicy",
                        "Bypass",
                        "-File",
                        str(program / "bin/maintain-v23.ps1"),
                        "-Action",
                        "Stop",
                    ],
                    "stop-installed-ui",
                )
                assert process.wait(timeout=90) == 0
            finally:
                if process.poll() is None:
                    process.terminate()
                    process.wait(timeout=30)
        record("installedScientificBrowserWorkflows")
        with closing(sqlite3.connect(profile / "data/labviz.db")) as db, db:
            db.execute(
                "CREATE TABLE installer_acceptance (id TEXT PRIMARY KEY, value REAL)"
            )
            db.execute(
                "INSERT INTO installer_acceptance VALUES ('preserve-after-uninstall', 1.25)"
            )
        frozen = profile / "data/objects/installer-frozen.svg"
        frozen.parent.mkdir(parents=True, exist_ok=True)
        frozen.write_bytes(b"<svg>preserve exact frozen artifact after uninstall</svg>")
        frozen_sha = sha(frozen)
        profile_id = read(profile / "profile.json")["profileId"]
        install("same-version-reinstall")
        health("same-version-installed-health")
        assert read(profile / "profile.json")["profileId"] == profile_id
        assert sha(frozen) == frozen_sha
        record("sameVersionReinstallPreservesProfileAndFrozenBytes")
        data_sha = sha(profile / "data/labviz.db")
        uninstaller = program / "unins000.exe"
        assert uninstaller.is_file()
        execute(
            [
                str(uninstaller),
                "/VERYSILENT",
                "/SUPPRESSMSGBOXES",
                "/NORESTART",
                "/LOG=" + str(output / "uninstall.inno.log"),
            ],
            "uninstall",
        )
        assert sha(profile / "data/labviz.db") == data_sha and sha(frozen) == frozen_sha
        assert read(profile / "profile.json")["profileId"] == profile_id
        assert not installed_entry()
        record("uninstallRetainsDataAndRemovesOwnRegistryIdentity")
        install("reinstall-after-uninstall")
        health("reopened-retained-data")
        with closing(sqlite3.connect(profile / "data/labviz.db")) as db:
            assert db.execute("SELECT value FROM installer_acceptance").fetchone() == (
                1.25,
            )
        assert (
            read(profile / "profile.json")["profileId"] == profile_id
            and sha(frozen) == frozen_sha
        )
        record("reinstallReopensRetainedProfile")
        execute(
            [
                str(uninstaller),
                "/VERYSILENT",
                "/SUPPRESSMSGBOXES",
                "/NORESTART",
                "/LOG=" + str(output / "final-uninstall.inno.log"),
            ],
            "final-uninstall",
        )
        assert not installed_entry() and profile.exists()
        record("acceptanceInstallerRemovedDataRetained")
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
                "checks": checks,
                "program": str(program),
                "retainedProfile": str(profile),
                "seconds": time.monotonic() - started,
                "scope": "Current Windows host with existing developer tools; not a clean machine or physically disconnected network acceptance.",
            },
        )


if __name__ == "__main__":
    main()
