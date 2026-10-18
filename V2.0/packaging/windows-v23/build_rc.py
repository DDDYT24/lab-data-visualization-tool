"""Build and verify an unsigned Windows RC from this checkout and pinned inputs."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

from stage_candidate import APP_ID, ROOT, SOURCE, checked, read, sha


def compiler_path() -> Path:
    candidates = [
        Path(os.environ.get("LOCALAPPDATA", "")) / "Programs/Inno Setup 6/ISCC.exe",
        Path(os.environ.get("ProgramFiles(x86)", "")) / "Inno Setup 6/ISCC.exe",
        Path(os.environ.get("ProgramFiles", "")) / "Inno Setup 6/ISCC.exe",
    ]
    found = shutil.which("ISCC.exe")
    if found:
        candidates.insert(0, Path(found))
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise RuntimeError("Install Inno Setup 6 for the installer phase.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase",
        choices=[
            "prepare",
            "verify-source",
            "stage",
            "portable",
            "installer",
            "artifacts",
            "all",
        ],
        default="all",
    )
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/v23-release/rc1")
    parser.add_argument("--build-python", type=Path, default=Path(sys.executable))
    parser.add_argument("--runtime-cache", type=Path)
    parser.add_argument(
        "--development",
        action="store_true",
        help="Allow private discovery builds; public artifacts still require a committed source.",
    )
    args = parser.parse_args()
    if os.name != "nt":
        parser.error("Windows is required for this packaging and lifecycle pipeline.")
    output = checked(args.output)
    if not output.is_relative_to(ROOT / "outputs/v23-release"):
        parser.error("Use this checkout's outputs/v23-release directory.")
    marker = output / "pipeline.json"
    if output.exists() and not marker.exists() and any(output.iterdir()):
        raise ValueError("An unrelated output directory cannot be adopted.")
    output.mkdir(parents=True, exist_ok=True)
    state = (
        read(marker)
        if marker.exists()
        else {"appId": APP_ID, "sourceRoot": str(ROOT), "runs": []}
    )
    if state["appId"] != APP_ID or state["sourceRoot"] != str(ROOT):
        raise ValueError("Pipeline ownership differs.")
    env = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith(("LABVIZ_", "PLAYWRIGHT_", "NEXT_"))
    }
    env.update(
        PYTHONIOENCODING="utf-8",
        PYTHONUTF8="1",
        PYTHONDONTWRITEBYTECODE="1",
        NEXT_TELEMETRY_DISABLED="1",
        GIT_OPTIONAL_LOCKS="0",
    )
    for name in ("PYTHONPATH", "PYTHONHOME", "NODE_PATH"):
        env.pop(name, None)
    tools = output / "tools"
    env["PLAYWRIGHT_BROWSERS_PATH"] = str(tools / "browsers")
    runtimes = read(SOURCE / "runtimes.json")
    node = (
        tools / "node" / ("node-" + runtimes["node"]["version"] + "-win-x64/node.exe")
    )
    embedded = tools / "python"
    test_python = ROOT / "V2.0/api/.venv/Scripts/python.exe"
    production_python = output / "build-production/Scripts/python.exe"
    env["PATH"] = str(node.parent) + os.pathsep + env.get("PATH", "")
    phase_run = None

    def save():
        marker.write_text(json.dumps(state, indent=2), encoding="utf-8")

    def run(label, command, *, cwd=ROOT, extra_env=None, allow_error=False):
        log = output / f"{len(state['runs']):02d}-{label}.log"
        started = time.monotonic()
        with log.open("wb") as stream:
            result = subprocess.run(
                [str(value) for value in command],
                cwd=cwd,
                env={**env, **(extra_env or {})},
                stdout=stream,
                stderr=stream,
                check=False,
                creationflags=subprocess.CREATE_NO_WINDOW,
            )
        phase_run["commands"].append(
            {
                "name": label,
                "exitCode": result.returncode,
                "seconds": time.monotonic() - started,
                "log": log.name,
            }
        )
        save()
        if result.returncode and not allow_error:
            raise RuntimeError(label + " failed; inspect " + str(log))
        return result.returncode

    def obtain(name):
        item = runtimes[name]
        archive = tools / item["url"].rsplit("/", 1)[1]
        tools.mkdir(parents=True, exist_ok=True)
        if not archive.exists():
            cached = args.runtime_cache / archive.name if args.runtime_cache else None
            if cached and cached.is_file():
                shutil.copyfile(checked(cached), archive)
            else:
                temporary = archive.with_suffix(".download")
                with (
                    urllib.request.urlopen(item["url"], timeout=120) as response,
                    temporary.open("wb") as stream,
                ):
                    shutil.copyfileobj(response, stream)
                temporary.rename(archive)
        if sha(archive) != item["sha256"]:
            raise ValueError("Pinned runtime archive hash differs: " + name)
        folder = tools / name
        if not folder.exists():
            folder.mkdir()
            with zipfile.ZipFile(archive) as bundle:
                for member in bundle.infolist():
                    if (
                        not (folder / member.filename)
                        .resolve()
                        .is_relative_to(folder.resolve())
                    ):
                        raise ValueError("Runtime archive has an external path.")
                bundle.extractall(folder)

    def prepare():
        for name in ("python", "node"):
            obtain(name)
        run(
            "build-python-version",
            [
                args.build_python,
                "-c",
                "import sys; assert sys.version_info[:2] == (3,13); print(sys.version)",
            ],
        )
        if not test_python.exists():
            run(
                "create-test-environment",
                [args.build_python, "-m", "venv", ROOT / "V2.0/api/.venv"],
            )
        run(
            "test-dependencies",
            [
                test_python,
                "-m",
                "pip",
                "install",
                "--disable-pip-version-check",
                "--require-hashes",
                "-r",
                ROOT / "V2.0/api/requirements-dev.lock.txt",
            ],
        )
        if not production_python.exists():
            run(
                "create-production-environment",
                [args.build_python, "-m", "venv", output / "build-production"],
            )
        run(
            "production-dependencies",
            [
                production_python,
                "-m",
                "pip",
                "install",
                "--disable-pip-version-check",
                "--require-hashes",
                "-r",
                ROOT / "V2.0/api/requirements.lock.txt",
            ],
        )
        npm = node.parent / "node_modules/npm/bin/npm-cli.js"
        run(
            "web-dependencies",
            [
                node,
                npm,
                "ci",
                "--registry=https://registry.npmjs.org",
                "--no-audit",
                "--no-fund",
            ],
            cwd=ROOT / "V2.0/web",
        )
        run(
            "browser-engine",
            [
                node,
                ROOT / "V2.0/web/node_modules/@playwright/test/cli.js",
                "install",
                "chromium",
            ],
            cwd=ROOT / "V2.0/web",
        )

    def verify_source():
        api, web = ROOT / "V2.0/api", ROOT / "V2.0/web"
        run(
            "api-lint",
            [
                test_python,
                "-m",
                "ruff",
                "check",
                "labviz_api",
                "tests",
                "migrations",
                "scripts",
            ],
            cwd=api,
        )
        run(
            "api-format",
            [
                test_python,
                "-m",
                "ruff",
                "format",
                "--check",
                "labviz_api",
                "tests",
                "migrations",
                "scripts",
            ],
            cwd=api,
        )
        run(
            "api-types",
            [test_python, "-m", "mypy", "labviz_api", "tests", "migrations", "scripts"],
            cwd=api,
        )
        excluded = []
        for path in sorted((api / "tests").glob("test_*.py")):
            content = path.read_text(encoding="utf-8")
            if "LABVIZ_TEST_POSTGRES_URL" in content or "LABVIZ_TEST_MINIO" in content:
                excluded.append("--ignore=tests/" + path.name)
        run(
            "api-local",
            [
                test_python,
                "-m",
                "pytest",
                *excluded,
                "--junitxml=" + str(output / "api-local.xml"),
            ],
            cwd=api,
        )
        (output / "api-local-scope.json").write_text(
            json.dumps(
                {"excludedStorageFiles": excluded, "fullIntegrationRequiredInCI": True},
                indent=2,
            ),
            encoding="utf-8",
        )
        run("web-lint", [node, "node_modules/eslint/bin/eslint.js", "."], cwd=web)
        run(
            "web-unit",
            [
                node,
                "node_modules/vitest/vitest.mjs",
                "run",
                "--reporter=json",
                "--outputFile=" + str(output / "web-unit.json"),
            ],
            cwd=web,
        )
        run("web-types", [node, "node_modules/typescript/bin/tsc", "--noEmit"], cwd=web)
        npm = node.parent / "node_modules/npm/bin/npm-cli.js"
        run(
            "npm-audit-production",
            [
                node,
                npm,
                "audit",
                "--registry=https://registry.npmjs.org",
                "--omit=dev",
                "--json",
            ],
            cwd=web,
        )
        audit_exit = run(
            "npm-audit-development",
            [node, npm, "audit", "--registry=https://registry.npmjs.org", "--json"],
            cwd=web,
            allow_error=True,
        )
        audit = read(output / f"{len(state['runs']):02d}-npm-audit-development.log")
        known = {
            "braces",
            "micromatch",
            "fast-glob",
            "eslint-config-next",
            "@next/eslint-plugin-next",
        }
        if (
            audit_exit not in (0, 1)
            or not isinstance(audit.get("metadata"), dict)
            or not set(audit.get("vulnerabilities", {})).issubset(known)
        ):
            raise RuntimeError(
                "The development audit contains an unreviewed finding or did not complete."
            )
        original = (web / "next-env.d.ts").read_bytes()
        try:
            run(
                "web-package-build",
                [node, "node_modules/next/dist/bin/next", "build"],
                cwd=web,
                extra_env={"LABVIZ_PACKAGE_BUILD": "1"},
            )
        finally:
            (web / "next-env.d.ts").write_bytes(original)
        run(
            "component-contracts",
            [
                test_python,
                SOURCE / "verify_components.py",
                "--test-python",
                test_python,
                "--output",
                output / "components",
            ],
        )

    def stage():
        notices = output / "web-licenses.txt"
        run(
            "web-licenses",
            [node, SOURCE / "collect-web-licenses.mjs", ROOT / "V2.0/web", notices],
        )
        run(
            "stage",
            [
                test_python,
                SOURCE / "stage_candidate.py",
                "--output",
                output / "payload",
                "--python-runtime",
                embedded,
                "--site-packages",
                output / "build-production/Lib/site-packages",
                "--node",
                node,
                "--receipt",
                output / "components/receipt.json",
                "--web-licenses",
                notices,
            ],
        )

    def portable():
        run(
            "payload-smoke",
            [
                test_python,
                SOURCE / "tests/smoke_payload.py",
                "--payload",
                output / "payload",
                "--output",
                output / "smoke",
            ],
        )
        run(
            "portable-lifecycle",
            [
                test_python,
                SOURCE / "tests/verify_portable_lifecycle.py",
                "--smoke-attempt",
                output / "smoke",
                "--output",
                output / "portable-lifecycle",
                "--browser-check",
            ],
        )

    def installer():
        command = [
            test_python,
            SOURCE / "compile_candidate.py",
            "--payload",
            output / "payload",
            "--lifecycle-evidence",
            output / "portable-lifecycle/verification.json",
            "--output",
            output / "installer",
            "--compiler",
            compiler_path(),
        ]
        if args.development:
            command.append("--allow-development")
        run("compile-installer", command)
        run(
            "installer-lifecycle",
            [
                test_python,
                SOURCE / "tests/verify_installer.py",
                "--build",
                output / "installer",
                "--payload",
                output / "payload",
                "--output",
                output / "installed-lifecycle",
            ],
        )

    def artifacts():
        run(
            "release-assets",
            [test_python, SOURCE / "release_assets.py", "--build-root", output],
        )

    actions = {
        "prepare": prepare,
        "verify-source": verify_source,
        "stage": stage,
        "portable": portable,
        "installer": installer,
        "artifacts": artifacts,
    }
    for name in actions if args.phase == "all" else [args.phase]:
        phase_run = {
            "phase": name,
            "startedAtUtc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "commands": [],
            "passed": False,
        }
        state["runs"].append(phase_run)
        save()
        try:
            actions[name]()
            phase_run["passed"] = True
            print(name + ": passed", flush=True)
        finally:
            save()


if __name__ == "__main__":
    main()
