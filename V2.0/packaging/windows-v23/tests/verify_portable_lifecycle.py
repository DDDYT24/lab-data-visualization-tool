"""Exercise the real bundled Windows processes in an owned acceptance fixture."""

from __future__ import annotations

import argparse
import base64
import ctypes
import hashlib
import http.cookiejar
import io
import json
import os
import shutil
import socket
import sqlite3
import subprocess
import sys
import tarfile
import time
import urllib.parse
import urllib.request
from contextlib import closing
from pathlib import Path


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write(path: Path, value):
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def sha(path: Path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def local_key(path: Path) -> str:
    from ctypes import wintypes

    class Blob(ctypes.Structure):
        _fields_ = [("size", wintypes.DWORD), ("data", ctypes.POINTER(ctypes.c_ubyte))]

    raw = path.read_bytes()
    buffer = (ctypes.c_ubyte * len(raw)).from_buffer_copy(raw)
    source, target = Blob(len(raw), buffer), Blob()
    crypt = ctypes.WinDLL("crypt32", use_last_error=True)
    crypt.CryptUnprotectData.argtypes = [
        ctypes.POINTER(Blob),
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        wintypes.DWORD,
        ctypes.POINTER(Blob),
    ]
    crypt.CryptUnprotectData.restype = wintypes.BOOL
    kernel = ctypes.WinDLL("kernel32")
    kernel.LocalFree.argtypes = [ctypes.c_void_p]
    if not crypt.CryptUnprotectData(
        ctypes.byref(source), None, None, None, None, 0, ctypes.byref(target)
    ):
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        return (
            base64.urlsafe_b64encode(ctypes.string_at(target.data, target.size))
            .rstrip(b"=")
            .decode("ascii")
        )
    finally:
        kernel.LocalFree(target.data)


def request(opener, url, *, method="GET", key=None, data=None, headers=None):
    headers = {**(headers or {}), **({"X-LabViz-Local-Key": key} if key else {})}
    with opener.open(
        urllib.request.Request(url, method=method, headers=headers, data=data),
        timeout=20,
    ) as response:
        return json.loads(response.read())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke-attempt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--browser-check", action="store_true")
    args = parser.parse_args()
    smoke = read(args.smoke_attempt / "verification.json")
    assert smoke["passed"], "A successful native smoke fixture is required."
    program = Path(smoke["program"]).resolve()
    fixture = program.parent
    assert fixture.parent == (Path.home() / "LabVizV23Acceptance").resolve()
    assert not any(
        p.is_symlink() or p.is_junction() for p in (fixture, program, *fixture.parents)
    )
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    profile = Path(smoke["profileRoot"]).resolve()
    env = {
        **os.environ,
        "LOCALAPPDATA": str(fixture / "appdata"),
        "PYTHONIOENCODING": "utf-8",
    }
    ps = str(
        Path(os.environ["SystemRoot"])
        / "System32/WindowsPowerShell/v1.0/powershell.exe"
    )
    base = [ps, "-NoProfile", "-ExecutionPolicy", "Bypass", "-File"]
    checks, processes, logs, held = {}, [], [], []
    started = time.monotonic()

    def record(name):
        checks[name] = True
        write(
            output / "verification.json",
            {
                "passed": False,
                "inProgress": True,
                "checks": checks,
                "candidateId": smoke["candidateId"],
                "program": str(program),
            },
        )
        print(name + ": passed", flush=True)

    def run(script, arguments, label, *, expect=0):
        with (output / (label + ".log")).open("wb") as log:
            result = subprocess.run(
                base + [str(program / "bin" / script)] + arguments,
                env=env,
                stdout=log,
                stderr=log,
                timeout=360,
                creationflags=subprocess.CREATE_NO_WINDOW,
                check=False,
            )
        assert (result.returncode == 0) == (expect == 0), (
            f"{label}: unexpected exit {result.returncode}; inspect the retained log."
        )
        return (output / (label + ".log")).read_text(encoding="utf-8-sig")

    def launch(arguments, label, root=profile):
        log = (output / (label + ".log")).open("wb")
        logs.append(log)
        process = subprocess.Popen(
            base
            + [
                str(program / "bin/start-v23-installed.ps1"),
                "-SkipOpenBrowser",
                "-ProfileRoot",
                str(root),
            ]
            + arguments,
            env=env,
            stdout=log,
            stderr=log,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
        processes.append(process)
        marker = root / "logs/running.json"
        deadline = time.monotonic() + 240
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise AssertionError(
                    f"{label}: launcher exited {process.returncode}; inspect retained log."
                )
            if marker.exists():
                running = read(marker)
                if running["launcher"]["pid"] == process.pid:
                    return process, running
            time.sleep(0.5)
        raise TimeoutError(label + ": native ready marker timed out")

    def stop(process, root, label):
        run("maintain-v23.ps1", ["-Action", "Stop", "-ProfileRoot", str(root)], label)
        assert process.wait(timeout=90) == 0
        assert not (root / "logs/running.json").exists()

    def seed(value):
        (profile / "data/objects").mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(profile / "data/labviz.db")) as db, db:
            db.execute(
                "CREATE TABLE IF NOT EXISTS installed_acceptance (id TEXT PRIMARY KEY, value REAL)"
            )
            db.execute(
                "INSERT OR REPLACE INTO installed_acceptance VALUES ('stable-science-id', ?)",
                (value,),
            )
        (profile / "data/objects/acceptance-frozen.svg").write_bytes(
            b"<svg>frozen acceptance scientific output</svg>"
        )

    def sentinel(root=profile):
        with closing(sqlite3.connect(root / "data/labviz.db")) as db:
            return db.execute(
                "SELECT value FROM installed_acceptance WHERE id='stable-science-id'"
            ).fetchone()[0]

    def variant(label, *, fail=False):
        current = read(program / "current-candidate.json")["candidateId"]
        original = program / "versions" / current[:24]
        temporary = fixture / ("stage-" + label)
        shutil.copytree(original, temporary)
        server = temporary / "V2.0/web/server.js"
        server.write_bytes(
            (
                b"throw new Error('Injected acceptance-only Web failure');\n"
                if fail
                else ("// acceptance generation " + label + "\n").encode()
            )
            + server.read_bytes()
        )
        manifest = read(temporary / "payload-manifest.json")
        manifest["files"]["V2.0/web/server.js"] = sha(server)
        identity = {k: manifest[k] for k in ("appId", "version", "dataEpoch", "files")}
        candidate = hashlib.sha256(
            json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        manifest["candidateId"] = candidate
        write(temporary / "payload-manifest.json", manifest)
        metadata = read(temporary / "candidate-build.json")
        metadata["candidateId"] = candidate
        write(temporary / "candidate-build.json", metadata)
        destination = program / "versions" / candidate[:24]
        assert destination.parent == program / "versions" and not destination.exists()
        temporary.rename(destination)
        write(
            program / "pending-candidate.json",
            {"appId": manifest["appId"], "candidateId": candidate},
        )
        return candidate

    try:
        for _ in range(2):
            listener = socket.socket()
            listener.bind(("127.0.0.1", 0))
            listener.listen()
            assert listener.getsockname()[1] not in (3000, 8000, 3001, 8001, 3305, 8305)
            held.append(listener)
        preferred = [listener.getsockname()[1] for listener in held]
        process, running = launch(
            ["-WebPort", str(preferred[0]), "-ApiPort", str(preferred[1])],
            "occupied-ports",
        )
        assert (
            running["apiPort"] not in preferred
            and int(urllib.parse.urlparse(running["webUrl"]).port) not in preferred
        )
        for listener in held:
            with socket.create_connection(listener.getsockname(), timeout=3):
                pass
        record("occupiedPortsPreservedAndReallocated")
        jar = http.cookiejar.CookieJar()
        opener = urllib.request.build_opener(
            urllib.request.ProxyHandler({}), urllib.request.HTTPCookieProcessor(jar)
        )
        url = running["webUrl"]
        key = local_key(profile / "access.dpapi")
        assert request(opener, url + "/api/v1/local/session", method="POST", key=key)[
            "ready"
        ]
        suffix = "_v23_installed_" + running["profileId"]
        assert any(cookie.name.endswith(suffix) for cookie in jar)
        request(
            opener, url + "/api/v1/samples/thermal-response/projects", method="POST"
        )
        boundary = "labviz-rc-fixture"
        upload = (
            "--"
            + boundary
            + '\r\nContent-Disposition: form-data; name="file"; filename="installed-lifecycle.csv"\r\nContent-Type: text/csv\r\n\r\nx,y\n0,1\n1,3\n2,5\n\r\n--'
            + boundary
            + "--\r\n"
        ).encode()
        sample = request(
            opener,
            url + "/api/v1/projects",
            method="POST",
            data=upload,
            headers={"Content-Type": "multipart/form-data; boundary=" + boundary},
        )
        write(output / "created-project.json", sample)
        record("BundledWebProxyAuthenticatedSampleCreation")
        if args.browser_check:
            result = subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).with_name("verify_packaged_browser.py")),
                    "--running-record",
                    str(profile / "logs/running.json"),
                    "--profile-root",
                    str(profile),
                    "--output",
                    str(output / "packaged-browser"),
                ],
                env=env,
                check=False,
                creationflags=subprocess.CREATE_NO_WINDOW,
            )
            assert result.returncode == 0, "Packaged browser acceptance failed."
            record("packagedScientificBrowserWorkflows")
        run("start-v23-installed.ps1", ["-SkipOpenBrowser"], "duplicate-start")
        assert (
            read(profile / "logs/running.json")["sessionId"] == running["sessionId"]
            and process.poll() is None
        )
        record("duplicateStartPreservesRunningInstance")
        run(
            "maintain-v23.ps1", ["-Action", "Backup"], "active-backup-refused", expect=1
        )
        record("activeProfileBackupRefused")
        stop(process, profile, "stop-first")
        record("authenticatedGracefulStop")
        crashed, crashed_running = launch([], "job-close-start")
        crashed_ports = [
            crashed_running["apiPort"],
            urllib.parse.urlparse(crashed_running["webUrl"]).port,
        ]
        crashed.terminate()
        crashed.wait(timeout=30)
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            open_ports = []
            for port in crashed_ports:
                with socket.socket() as probe:
                    probe.settimeout(0.5)
                    if probe.connect_ex(("127.0.0.1", port)) == 0:
                        open_ports.append(port)
            if not open_ports:
                break
            time.sleep(0.2)
        assert not open_ports, "Owned services survived their launcher Job Object."
        recovered_process, _ = launch([], "after-job-close-recovery")
        stop(recovered_process, profile, "stop-after-job-close")
        record("launcherCrashClosesOwnedServicesAndRecovers")
        seed(1.25)
        backup = json.loads(run("maintain-v23.ps1", ["-Action", "Backup"], "backup"))[
            "backupDirectory"
        ]
        run(
            "maintain-v23.ps1",
            ["-Action", "Verify", "-BackupDirectory", backup],
            "verify-backup",
        )
        restored_text = run(
            "maintain-v23.ps1",
            ["-Action", "Restore", "-BackupDirectory", backup],
            "restore-independent",
        )
        restored, _ = json.JSONDecoder().raw_decode(restored_text.lstrip())
        target = Path(restored["restoredProfileRoot"])
        assert (
            target.parent == fixture / "appdata/LabVizV23Profiles" and target != profile
        )
        assert (
            sentinel(target) == 1.25
            and read(target / "profile.json")["profileId"] != running["profileId"]
        )
        assert sha(target / "data/objects/acceptance-frozen.svg") == sha(
            profile / "data/objects/acceptance-frozen.svg"
        )
        record("completeBackupAndIndependentRestore")
        restored_process, restored_running = launch(
            [], "restored-profile-start", root=target
        )
        assert restored_running["profileId"] == restored["restoredProfileId"]
        restored_jar = http.cookiejar.CookieJar()
        restored_opener = urllib.request.build_opener(
            urllib.request.ProxyHandler({}),
            urllib.request.HTTPCookieProcessor(restored_jar),
        )
        restored_key = local_key(target / "access.dpapi")
        restored_url = restored_running["webUrl"]
        assert request(
            restored_opener,
            restored_url + "/api/v1/local/session",
            method="POST",
            key=restored_key,
        )["ready"]
        assert all(not cookie.name.endswith(suffix) for cookie in restored_jar)
        reopened = request(
            restored_opener, restored_url + "/api/v1/projects/" + sample["projectId"]
        )
        assert reopened["projectId"] == sample["projectId"]
        stop(restored_process, target, "stop-restored")
        record("restoredProjectReopensWithIndependentCookie")
        source_root = Path(__file__).resolve().parents[4]
        legacy_commit = "1ca89162f94b13f02c819552f6383be08fc0d51a"
        actual_commit = subprocess.check_output(
            ["git", "rev-parse", "v2.2.0^{commit}"], cwd=source_root, text=True
        ).strip()
        assert actual_commit == legacy_commit, (
            "Fetch the published v2.2.0 tag before legacy migration acceptance."
        )
        archive = subprocess.check_output(
            [
                "git",
                "archive",
                "--format=tar",
                legacy_commit,
                "V2.0/api/labviz_api",
                "V2.0/api/migrations",
                "V2.0/api/alembic.ini",
                "V2.0/assets",
                "V2.0/contracts",
            ],
            cwd=source_root,
        )
        legacy_source = fixture / "legacy-source"
        legacy_source.mkdir()
        with tarfile.open(fileobj=io.BytesIO(archive)) as bundle:
            bundle.extractall(legacy_source, filter="data")
        legacy_data = fixture / "legacy-data"
        legacy_report = output / "legacy-project.json"
        with (output / "legacy-fixture.log").open("wb") as log:
            created = subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).with_name("create_legacy_fixture.py")),
                    "--source-root",
                    str(legacy_source),
                    "--data-root",
                    str(legacy_data),
                    "--report",
                    str(legacy_report),
                ],
                env=env,
                stdout=log,
                stderr=log,
                check=False,
                creationflags=subprocess.CREATE_NO_WINDOW,
            )
        assert created.returncode == 0, (
            "The released V2.2 fixture could not be generated."
        )
        legacy = read(legacy_report)
        legacy_hashes = {
            p.relative_to(legacy_data).as_posix(): sha(p)
            for p in legacy_data.rglob("*")
            if p.is_file()
        }
        imported_text = run(
            "maintain-v23.ps1",
            ["-Action", "Import", "-SourceDirectory", str(legacy_data)],
            "import-v22-copy",
        )
        imported, _ = json.JSONDecoder().raw_decode(imported_text.lstrip())
        imported_root = Path(imported["root"])
        assert (
            imported_root != profile
            and imported_root.parent == fixture / "appdata/LabVizV23Profiles"
        )
        imported_process, imported_running = launch(
            [], "imported-v22-start", root=imported_root
        )
        imported_opener = urllib.request.build_opener(
            urllib.request.ProxyHandler({}),
            urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar()),
        )
        imported_url = imported_running["webUrl"]
        assert request(
            imported_opener,
            imported_url + "/api/v1/local/session",
            method="POST",
            key=local_key(imported_root / "access.dpapi"),
        )["ready"]
        imported_workspace = request(
            imported_opener,
            imported_url + "/api/v1/projects/" + legacy["projectId"] + "/workspace",
        )
        assert imported_workspace["preview"]["totalRows"] == legacy["rows"]
        for snapshot in legacy["snapshots"].values():
            with imported_opener.open(
                imported_url
                + "/api/v1/projects/"
                + legacy["projectId"]
                + "/figures/"
                + snapshot["id"]
                + "/download",
                timeout=30,
            ) as response:
                assert hashlib.sha256(response.read()).hexdigest() == snapshot["sha256"]
        stop(imported_process, imported_root, "stop-imported-v22")
        assert legacy_hashes == {
            p.relative_to(legacy_data).as_posix(): sha(p)
            for p in legacy_data.rglob("*")
            if p.is_file()
        }
        assert sentinel() == 1.25
        record("legacyV22CopyImportPreservesIdsAndFrozenBytes")
        original = read(program / "current-candidate.json")["candidateId"]
        upgraded = variant("compatible")
        run(
            "start-v23-installed.ps1",
            ["-SkipOpenBrowser", "-HealthCheckOnly"],
            "upgrade",
        )
        assert (
            read(program / "current-candidate.json")["candidateId"] == upgraded
            and sentinel() == 1.25
        )
        record("sameVersionGenerationUpgradePreservesData")
        seed(2.5)
        run(
            "start-v23-installed.ps1",
            ["-Rollback", "-SkipOpenBrowser", "-HealthCheckOnly"],
            "rollback",
        )
        assert (
            read(program / "current-candidate.json")["candidateId"] == original
            and sentinel() == 1.25
        )
        assert any(
            sentinel_directory_value(p) == 2.5 for p in profile.glob("data.retained-*")
        )
        record("compatibleRollbackRetainsNewerData")
        variant("injected-web-failure", fail=True)
        run(
            "start-v23-installed.ps1",
            ["-SkipOpenBrowser", "-HealthCheckOnly"],
            "failed-upgrade-fallback",
        )
        assert (
            read(program / "current-candidate.json")["candidateId"] == original
            and sentinel() == 1.25
            and (program / "failed-upgrade.json").exists()
        )
        record("failedCandidateAutomaticallyRestoresPrevious")
        schema = read(profile / "data-schema.json")
        write(profile / "data-schema.json", {**schema, "dataEpoch": 2})
        before = sha(profile / "data/labviz.db")
        try:
            run(
                "start-v23-installed.ps1",
                ["-Rollback", "-SkipOpenBrowser", "-HealthCheckOnly"],
                "future-epoch-refused",
                expect=1,
            )
            assert (
                sha(profile / "data/labviz.db") == before
                and read(profile / "data-schema.json")["dataEpoch"] == 2
            )
        finally:
            write(profile / "data-schema.json", schema)
        record("futureEpochRollbackRefusedWithoutDataMutation")
        run(
            "maintain-v23.ps1",
            [
                "-Action",
                "RemoveData",
                "-ProfileRoot",
                str(target),
                "-ConfirmProfileId",
                "0" * 32,
            ],
            "wrong-delete-id",
            expect=1,
        )
        assert target.exists()
        run(
            "maintain-v23.ps1",
            [
                "-Action",
                "RemoveData",
                "-ProfileRoot",
                str(target),
                "-ConfirmProfileId",
                restored["restoredProfileId"],
            ],
            "remove-independent-copy",
        )
        assert not target.exists() and profile.exists() and sentinel() == 1.25
        record("exactProfileDeletionPreservesOtherProfile")
        run(
            "maintain-v23.ps1",
            [
                "-Action",
                "RemoveData",
                "-ProfileRoot",
                str(imported_root),
                "-ConfirmProfileId",
                imported["profileId"],
            ],
            "remove-imported-copy",
        )
        assert not imported_root.exists() and legacy_data.exists()
        passed = True
    except BaseException as error:
        (output / "failure.txt").write_text(repr(error), encoding="utf-8")
        passed = False
        raise
    finally:
        for process in processes:
            if process.poll() is None:
                # Terminate only child launchers created and held by this harness;
                # their Windows Job Object closes their own services.
                process.terminate()
                process.wait(timeout=30)
        for log in logs:
            log.close()
        for listener in held:
            listener.close()
        write(
            output / "verification.json",
            {
                "passed": locals().get("passed", False),
                "inProgress": False,
                "checks": checks,
                "candidateId": smoke["candidateId"],
                "program": str(program),
                "seconds": time.monotonic() - started,
            },
        )


def sentinel_directory_value(directory):
    try:
        with closing(
            sqlite3.connect((directory / "labviz.db").as_uri() + "?mode=ro", uri=True)
        ) as db:
            row = db.execute(
                "SELECT value FROM installed_acceptance WHERE id='stable-science-id'"
            ).fetchone()
            return row[0] if row else None
    except sqlite3.Error:
        return None


if __name__ == "__main__":
    main()
