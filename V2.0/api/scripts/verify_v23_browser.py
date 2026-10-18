"""Private production browser/API harness; restart only this harness's API tree."""

import argparse
import hashlib
import json
import os
import re
import shutil
import socket
import subprocess
import threading
import time
import urllib.request
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from scripts.measure_v22_performance import _csv_payload

root = Path(__file__).resolve().parents[3]
api, web = root / "V2.0/api", root / "V2.0/web"
reports = root / "outputs/v23-phase5"
reports.mkdir(parents=True, exist_ok=True)
parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=["functional", "performance"], required=True)
parser.add_argument("--build", action="store_true")
parser.add_argument(
    "--test-file", action="append", default=[], help="Run selected functional spec files from e2e"
)
parser.add_argument(
    "--project",
    action="append",
    choices=["chromium", "firefox-smoke", "webkit-smoke", "mobile-chromium"],
    help="Select browser projects; default is Chromium or the full functional matrix",
)
parser.add_argument(
    "--all-functional",
    action="store_true",
    help="Run every non-performance E2E flow with live checks enabled",
)
parser.add_argument("--diagnostics", action="store_true")
parser.add_argument(
    "--pointer-paint",
    action="store_true",
    help="Separate native click-to-UI-paint diagnostic; retain original verified-pointer timing",
)
parser.add_argument("--case", choices=["10000", "100000", "mixed100000"])
parser.add_argument(
    "--buffer-input", action="store_true", help="Separate renderer-reconstruction diagnostic"
)
parser.add_argument(
    "--suffix", default="-layers-final", help="Preserve each source-bound measurement report"
)
args = parser.parse_args()
assert re.fullmatch(r"-[a-zA-Z0-9-]+", args.suffix), "Suffix must be a safe filename fragment"
node = shutil.which("node")
assert node
flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
for port in [3305, 8305]:
    with socket.socket() as probe:
        assert probe.connect_ex(("127.0.0.1", port)) != 0, f"Port {port} already used"
profile = reports / ("layers-browser-" + args.mode + "-" + uuid.uuid4().hex)
profile.mkdir()
key, control_key = uuid.uuid4().hex + uuid.uuid4().hex, uuid.uuid4().hex
environment = os.environ.copy()
environment.update(
    {
        "LABVIZ_DATABASE_PATH": str(profile / "labviz.db"),
        "LABVIZ_OBJECT_STORAGE_ROOT": str(profile / "objects"),
        "LABVIZ_PERSISTENCE_BACKEND": "sqlite",
        "LABVIZ_OBJECT_STORAGE_BACKEND": "local",
        "LABVIZ_ENVIRONMENT": "development",
        "LABVIZ_AUTH_MODE": "console",
        "LABVIZ_DEVELOPMENT_LINE": "2.3",
        "LABVIZ_LOCAL_ACCESS_KEY": key,
        "LABVIZ_E2E_LOCAL_ACCESS_KEY": key,
        "LABVIZ_PUBLIC_WEB_URL": "http://127.0.0.1:3305",
        "LABVIZ_ALLOWED_ORIGINS": "http://127.0.0.1:3305",
        "LABVIZ_API_PROXY_TARGET": "http://127.0.0.1:8305",
        "NEXT_PUBLIC_LABVIZ_API_URL": "/api/v1",
        "NEXT_TELEMETRY_DISABLED": "1",
        "PYTHONIOENCODING": "utf-8",
        "PYTHONUNBUFFERED": "1",
        "LABVIZ_E2E_BUILD": "1",
        "LABVIZ_E2E_LIVE": "1",
        "PLAYWRIGHT_BASE_URL": "http://127.0.0.1:3305",
        "LABVIZ_PHASE5_PERF": "1",
        "LABVIZ_PHASE5_REPORT_SUFFIX": args.suffix,
    }
)
for option in ["LABVIZ_PHASE5_TRACE", "LABVIZ_PHASE5_DIAGNOSTICS"]:
    environment.pop(option, None)
environment.pop("LABVIZ_PHASE5_BUFFER_INPUT", None)
environment.pop("LABVIZ_PHASE5_POINTER_PAINT", None)
if args.pointer_paint:
    assert args.mode == "performance"
    environment["LABVIZ_PHASE5_POINTER_PAINT"] = "1"
if args.buffer_input:
    environment["LABVIZ_PHASE5_BUFFER_INPUT"] = "1"
if args.diagnostics:
    environment["LABVIZ_PHASE5_DIAGNOSTICS"] = "1"

for rows in [10000, 100000]:
    fixture = reports / f"legacy-linear-{rows}.csv"
    payload = _csv_payload(rows)
    if fixture.exists():
        assert fixture.read_bytes() == payload, "Fixed performance dataset differs"
    else:
        fixture.write_bytes(payload)
if args.build:
    with (reports / f"build{args.suffix}.log").open("w", encoding="utf-8") as log:
        completed = subprocess.run(
            [node, str(web / "node_modules/next/dist/bin/next"), "build"],
            cwd=web,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            creationflags=flags,
        )
    assert completed.returncode == 0
    print("Production build ready", flush=True)

processes: list[subprocess.Popen[bytes]] = []
logs = []
lock = threading.Lock()
metadata: dict[str, Any] = {
    "mode": args.mode,
    "interfaceMode": environment.get("LABVIZ_E2E_INTERFACE_MODE", "classic"),
    "profile": str(profile),
    "ports": [3305, 8305],
    "ownedPids": [],
    "apiGenerations": [],
    "apiStartAttempts": [],
    "controlFailures": [],
    "closed": False,
    "productionBuildId": (web / ".next-e2e/BUILD_ID").read_text().strip(),
    "sourceSha256": {
        str(path.relative_to(root)).replace("\\", "/"): hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        for path in sorted(
            list((api / "labviz_api").rglob("*.py"))
            + [api / "scripts/verify_v23_browser.py", api / "scripts/sample_v23_browser_memory.py"]
            + list((web / "src").rglob("*.ts"))
            + list((web / "src").rglob("*.tsx"))
            + list((web / "e2e").rglob("*.ts"))
        )
    },
}
api_process: subprocess.Popen[bytes] | None = None


def start(name: str, command: list[str], cwd: Path) -> subprocess.Popen[bytes]:
    log = (profile / (name + ".log")).open("w", encoding="utf-8")
    logs.append(log)
    process = subprocess.Popen(
        command, cwd=cwd, env=environment, stdout=log, stderr=subprocess.STDOUT, creationflags=flags
    )
    processes.append(process)
    metadata["ownedPids"].append(process.pid)
    return process


def ready(url: str, process: subprocess.Popen[bytes]) -> None:
    deadline = time.monotonic() + 90
    while time.monotonic() < deadline:
        assert process.poll() is None, "Owned service exited"
        try:
            with urllib.request.urlopen(url, timeout=2) as response:
                if response.status == 200:
                    return
        except Exception:
            time.sleep(0.2)
    raise RuntimeError("Owned service readiness timeout")


def start_api() -> dict[str, Any]:
    global api_process
    number = len(metadata["apiStartAttempts"])
    name = f"api-{number}"
    metadata["apiStartAttempts"].append({"log": name + ".log"})
    api_process = start(
        name,
        [
            str(api / ".venv/Scripts/python.exe"),
            "-m",
            "uvicorn",
            "labviz_api.main:app",
            "--host",
            "127.0.0.1",
            "--port",
            "8305",
            "--no-access-log",
        ],
        api,
    )
    ready("http://127.0.0.1:8305/api/v1/health", api_process)
    actual = re.search(
        r"Started server process \[(\d+)\]",
        (profile / (name + ".log")).read_text(encoding="utf-8", errors="replace"),
    )
    assert actual
    generation = {
        "launcherPid": api_process.pid,
        "actualApiPid": int(actual.group(1)),
        "log": name + ".log",
    }
    metadata["apiGenerations"].append(generation)
    environment["LABVIZ_PHASE5_API_PID"] = actual.group(1)
    return generation


def stop(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is None:
        # This Popen belongs to our private harness. Never target another checkout's service.
        subprocess.run(
            ["taskkill", "/PID", str(process.pid), "/T", "/F"],
            capture_output=True,
            creationflags=flags,
            check=True,
        )
        process.wait(timeout=15)


class Control(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args: Any) -> None:
        pass

    def do_POST(self) -> None:
        if (
            self.path not in {"/restart", "/status"}
            or self.headers.get("X-LabViz-Test-Control") != control_key
        ):
            self.send_error(403)
            return
        with lock:
            try:
                assert api_process is not None
                if self.path == "/restart":
                    stop(api_process)
                    generation = start_api()
                else:
                    assert api_process.poll() is None
                    generation = metadata["apiGenerations"][-1]
                body = json.dumps({"actualApiPid": generation["actualApiPid"]}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            except Exception as error:
                metadata["controlFailures"].append(
                    {
                        "type": type(error).__name__,
                        "message": str(error)
                        .replace(key, "[redacted]")
                        .replace(control_key, "[redacted]")[:500],
                    }
                )
                self.send_error(500, "Private restart failed")


control = None
try:
    generation = start_api()
    metadata["actualApiPid"] = generation["actualApiPid"]
    metadata["apiPidScope"] = "Actual Uvicorn child PID from each private process generation log"
    web_process = start(
        "web",
        [
            node,
            str(web / "node_modules/next/dist/bin/next"),
            "start",
            "--hostname",
            "127.0.0.1",
            "--port",
            "3305",
        ],
        web,
    )
    ready("http://127.0.0.1:3305/", web_process)
    if args.mode == "performance":
        control = ThreadingHTTPServer(("127.0.0.1", 0), Control)
        threading.Thread(target=control.serve_forever, daemon=True).start()
        environment["LABVIZ_PHASE5_RESTART_URL"] = f"http://127.0.0.1:{control.server_port}/restart"
        environment["LABVIZ_PHASE5_CONTROL_KEY"] = control_key
    print("Owned production services ready", flush=True)
    files = (
        [
            "e2e/live-research-figures.spec.ts",
            "e2e/live-research-reuse.spec.ts",
            "e2e/live-research-phase5.spec.ts",
        ]
        if args.mode == "functional"
        else ["e2e/live-research-phase5-performance.spec.ts"]
    )
    if args.all_functional:
        assert args.mode == "functional"
        files = [
            str(p.relative_to(web)).replace("\\", "/")
            for p in sorted((web / "e2e").glob("*.spec.ts"))
            if p.name != "live-research-phase5-performance.spec.ts"
        ]
    if args.test_file:
        assert args.mode == "functional" and not args.all_functional
        assert all(
            re.fullmatch(r"e2e/[a-z0-9-]+\.spec\.ts", item) and (web / item).is_file()
            for item in args.test_file
        )
        files = args.test_file
    metadata["testFiles"] = files
    environment["PLAYWRIGHT_JSON_OUTPUT_FILE"] = str(
        reports / f"browser-{args.mode}{args.suffix}.json"
    )
    with (reports / f"browser-{args.mode}{args.suffix}.log").open("w", encoding="utf-8") as log:
        completed = subprocess.run(
            [
                node,
                str(web / "node_modules/@playwright/test/cli.js"),
                "test",
                *files,
                *(
                    ["--project=" + item for item in args.project]
                    if args.project
                    else []
                    if args.all_functional
                    else ["--project=chromium"]
                ),
                "--workers=1",
                "--timeout=180000",
                "--reporter=json,line",
                "--output=" + str(profile / "artifacts"),
                *(
                    [
                        "--grep",
                        "Phase 5 real browser "
                        + ("mixed 100000" if args.case == "mixed100000" else args.case)
                        + " rows",
                    ]
                    if args.case
                    else []
                ),
            ],
            cwd=web,
            env=environment,
            stdout=log,
            stderr=log,
            creationflags=flags,
        )
    metadata["exitCode"] = completed.returncode
    result = json.loads(
        (reports / f"browser-{args.mode}{args.suffix}.json").read_text(encoding="utf-8")
    )
    print(json.dumps(result.get("stats", {})), flush=True)
finally:
    if control is not None:
        control.shutdown()
        control.server_close()
    for process in reversed(processes):
        stop(process)
    for log in logs:
        log.close()
    metadata["closed"] = all(p.poll() is not None for p in processes)
    (reports / f"browser-{args.mode}{args.suffix}-session.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print("All owned services closed", flush=True)
assert metadata.get("exitCode") == 0, "Browser checks failed; reports retained"
