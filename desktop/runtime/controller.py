"""Supervise local desktop services and preserve the installed profile boundary."""

from __future__ import annotations

import base64
import ctypes
from ctypes import wintypes
import hashlib
import json
import os
import queue
import subprocess
import sys
import threading
import time
import uuid
import urllib.request
from pathlib import Path

# The bundled embeddable interpreter deliberately excludes the script directory.
# Add only this verified runtime directory, without changing its global _pth file.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from preferences import Preferences
from process_job import OwnedJob
from session import Session
from maintenance import Maintenance

APP_ID = "55B46BB7-101E-46DC-951A-3DE7E402EDD0"
DESKTOP_ID = "org.labviz.desktop.preview"
AUDIT_PATH = None


class DataBlob(ctypes.Structure):
    # ctypes keeps pointer types alive globally. Reuse this class across DPAPI
    # calls so a long-lived controller does not retain a new type on every read.
    _fields_ = [("size", wintypes.DWORD), ("data", ctypes.POINTER(ctypes.c_ubyte))]


def unlinked(path: Path) -> Path:
    path = path.absolute()
    for member in (path, *path.parents):
        if member.is_symlink() or member.is_junction():
            raise ValueError("Linked desktop paths are not supported")
    return path.resolve()


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write(path: Path, value) -> None:
    temporary = path.with_name(path.name + ".new-" + uuid.uuid4().hex)
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(value, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def emit(kind: str, **values) -> None:
    if AUDIT_PATH and kind in {
        "checkpoint",
        "suspend-commit",
        "suspended",
        "sleep-blocked",
        "sleep-cancelled",
        "failed",
    }:
        with AUDIT_PATH.open("a", encoding="utf-8") as stream:
            stream.write(
                json.dumps({"event": kind, "utcSeconds": time.time(), **values}) + "\n"
            )
    print(json.dumps({"type": kind, **values}), flush=True)


def local_key(path: Path) -> str:
    raw = path.read_bytes()
    buffer = (ctypes.c_ubyte * len(raw)).from_buffer_copy(raw)
    source, target = DataBlob(len(raw), buffer), DataBlob()
    crypt = ctypes.WinDLL("crypt32", use_last_error=True)
    crypt.CryptUnprotectData.argtypes = [
        ctypes.POINTER(DataBlob),
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        wintypes.DWORD,
        ctypes.POINTER(DataBlob),
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
            .decode()
        )
    finally:
        kernel.LocalFree(target.data)


class ExitRequest:
    """Close admission and recheck idle before honoring an explicit exit."""

    def __init__(self):
        self.active = False
        self.generation = None

    def request(self):
        self.active = True

    def cancel(self, api):
        if self.generation is not None:
            api.call("cancel-drain", {})
        self.active, self.generation = False, None

    def check(self, api, session, downloads, maintenance_busy=False):
        status = api.call("activity")
        if maintenance_busy or downloads or not session.idle() or not status["idle"]:
            if self.generation is not None:
                api.call("cancel-drain", {})
                self.generation = None
            return "exit-waiting"
        if self.generation is None:
            status = api.call("drain", {})
            self.generation = status["generation"]
        status = api.call("activity")
        if (status["idle"] and status["draining"]
                and status["generation"] == self.generation
                and session.idle() and not downloads):
            return "exit-ready"
        api.call("cancel-drain", {})
        self.generation = None
        return "exit-waiting"


class ApiControl:
    def __init__(self, ready, key):
        self.ready, self.key = ready, key

    def verify_api(self):
        from service_runner import port_owned, record, SimpleProcess

        expected = self.ready["api"]
        actual = record(SimpleProcess(expected["pid"]))
        if (
            actual["startedTicks"] != int(expected["startedTicks"])
            or actual["executable"].casefold() != expected["executable"].casefold()
            or not port_owned(self.ready["apiPort"], expected["pid"])
        ):
            raise ValueError(
                "Owned API identity changed; no desktop credential was sent"
            )

    def call(self, action, value=None):
        self.verify_api()
        body = None if value is None else json.dumps(value).encode()
        request = urllib.request.Request(
            f"http://127.0.0.1:{self.ready['apiPort']}/api/v1/local/{action}",
            data=body,
            headers={
                "X-LabViz-Local-Key": self.key,
                "Content-Type": "application/json",
                "Cookie": f"labviz_local_session_v23_installed_{self.ready['profileId']}={self.key}",
            },
        )
        with urllib.request.urlopen(request, timeout=5) as response:
            result = json.load(response)
        if result.get("profileId") != self.ready["profileId"]:
            raise ValueError("Desktop control profile identity differs")
        if action in {"activity", "drain", "cancel-drain"}:
            counts = ("requests", "imports", "figures", "ownedFigureTasks")
            if (
                result.get("format") != "labviz-desktop-activity-v1"
                or type(result.get("draining")) is not bool
                or type(result.get("idle")) is not bool
                or type(result.get("generation")) is not int
                or any(
                    type(result.get(name)) is not int or result[name] < 0
                    for name in counts
                )
            ):
                raise ValueError("Desktop activity could not be verified")
            if result["idle"] != all(result[name] == 0 for name in counts):
                raise ValueError(
                    "Desktop activity counters and idle acknowledgement differ"
                )
        return result

    def bases(self, projects):
        return self.call("draft-bases", projects)["projects"]

    def figures(self, bindings):
        return self.call("draft-figures", bindings)["valid"]


def stop_services(child, ready, profile, job, logs, reason):
    if ready and child.poll() is None:
        write(
            profile / "logs/stop.request",
            {"profileId": ready["profileId"], "sessionId": ready["sessionId"]},
        )
    forced = False
    try:
        code = child.wait(timeout=90 if ready else 1)
    except subprocess.TimeoutExpired:
        forced = True
        child.terminate()
        code = child.wait(timeout=20)
    if job:
        job.close()
    service_receipt = None
    if ready:
        try:
            service_receipt = read(unlinked(profile / "logs/desktop-service-exit.json"))
            if (service_receipt.get("format") != "labviz-desktop-service-exit-v1"
                    or service_receipt.get("sessionId") != ready["sessionId"]
                    or service_receipt.get("profileId") != ready["profileId"]):
                service_receipt = None
        except (OSError, ValueError, TypeError):
            pass
    receipt = {
            "launcherExitCode": code,
            "failure": reason,
            "profileRoot": str(profile),
            "runningRecordRemoved": not (profile / "logs/running.json").exists(),
            "forced": forced,
            "serviceExit": service_receipt,
        }
    receipt["normal"] = bool(ready and not reason and not forced and code == 0
                             and receipt["runningRecordRemoved"] and service_receipt
                             and service_receipt.get("normal") is True
                             and service_receipt.get("apiGraceful") is True
                             and service_receipt.get("forced") is False
                             and service_receipt.get("servicesStopped") is True)
    write(logs / "desktop-controller-exit.json", receipt)
    return receipt


def quiet_clock_resumed(previous, now):
    """A scheduler/system pause never supplies evidence of quiet user time."""
    return previous is not None and (now < previous or now - previous > 5)


def valid_maintenance_request(value, owner):
    """An expired or abandoned installer request must never close a later session."""
    if not isinstance(value, dict) or value.get('format') != 'labviz-desktop-maintenance-request-v1' or value.get('appId') != DESKTOP_ID or value.get('action') != 'prepare-update' or value.get('profileRoot') != str(owner):
        return False
    expiry = value.get('expiresUtcSeconds')
    expected = value.get('requester')
    if type(expiry) not in (int, float) or not time.time() < expiry <= time.time() + 600 or not isinstance(expected, dict) or set(expected) != {'pid', 'startedTicks', 'executable'} or type(expected.get('pid')) is not int:
        return False
    try:
        from service_runner import SimpleProcess, record
        return record(SimpleProcess(expected['pid'])) == expected
    except (OSError, ValueError, TypeError, KeyError):
        return False


def main() -> None:
    global AUDIT_PATH
    config = read(Path(sys.argv[1]))
    if config.get("format") != "labviz-desktop-session-v1":
        raise ValueError("Unknown desktop session format")
    owner = unlinked(Path(config["profileRoot"]))
    marker = unlinked(owner / "desktop-profile.json")
    if not marker.exists():
        if owner.exists() and any(owner.iterdir()):
            raise ValueError(
                "An unrelated directory cannot be adopted as a desktop profile"
            )
        owner.mkdir(parents=True, exist_ok=True)
        write(
            marker,
            {
                "format": "labviz-desktop-profile-v1",
                "appId": DESKTOP_ID,
                "root": str(owner),
            },
        )
    identity = read(marker)
    if identity != {
        "format": "labviz-desktop-profile-v1",
        "appId": DESKTOP_ID,
        "root": str(owner),
    }:
        raise ValueError("Desktop profile identity differs")
    from service_runner import Mutex, sid

    _profile_lock = Mutex(
        "Local\\LabVizDesktopPreview-"
        + hashlib.sha256((sid() + "|" + str(owner).lower()).encode())
        .hexdigest()
        .upper()
    )
    _application_lock = Mutex('Local\\LabVizDesktopPreview-App', False)
    logs = unlinked(owner / "logs")
    logs.mkdir(exist_ok=True)
    AUDIT_PATH = logs / "desktop-lifecycle.jsonl"
    payload = unlinked(Path(config["payloadRoot"]))
    metadata = read(payload / "candidate-build.json")
    if (
        metadata.get("appId") != APP_ID
        or metadata.get("candidateId") != config["candidateId"]
        or metadata.get("dataEpoch") != 1
    ):
        raise ValueError("Desktop payload identity differs")
    program = unlinked(owner / "program")
    program.mkdir(exist_ok=True)
    installation = program / "installation.json"
    if not installation.exists():
        write(installation, {"appId": APP_ID, "format": "labviz-installed-v1"})
    installation_identity = read(unlinked(installation))
    if (
        installation_identity.get("appId") != APP_ID
        or installation_identity.get("format") != "labviz-installed-v1"
    ):
        raise ValueError("Desktop installation identity differs")
    base = unlinked(Path(config.get('profileBase', str(owner))))
    maintenance = Maintenance(owner, base, payload)
    appdata = unlinked(owner / "appdata")
    appdata.mkdir(exist_ok=True)
    profile = appdata / "LabVizV23"
    running_file = profile / "logs/running.json"
    powershell = (
        Path(os.environ["SystemRoot"])
        / "System32/WindowsPowerShell/v1.0/powershell.exe"
    )
    runner_config = owner / "service-runner.json"
    write(
        runner_config,
        {
            "payloadRoot": str(payload),
            "backendProfile": str(profile),
            "programRoot": str(program),
            "candidateId": metadata["candidateId"],
        },
    )
    command = [
        str(payload / "runtime/python/python.exe"),
        "-B",
        "-X",
        "utf8",
        str(Path(__file__).with_name("service_runner.py")),
        str(runner_config),
    ]
    environment = {
        **os.environ,
        "LOCALAPPDATA": str(appdata),
        "PYTHONIOENCODING": "utf-8",
        "LABVIZ_DESKTOP_MODE": "1",
    }
    # This local API admits one executing scientific job. Avoid dozens of idle
    # BLAS worker stacks; numerical and rendering contracts remain unchanged.
    for name in (
        "OPENBLAS_NUM_THREADS",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        environment[name] = "1"
    for name in ("PYTHONPATH", "PYTHONHOME", "PYTHONPYCACHEPREFIX", "NODE_PATH"):
        environment.pop(name, None)
    requests: queue.Queue[str] = queue.Queue()

    def input_reader() -> None:
        for line in sys.stdin:
            try:
                value = json.loads(line)
                requests.put(str(value["action"]))
            except (ValueError, KeyError, TypeError):
                requests.put(line.strip())
        requests.put("stop")

    threading.Thread(target=input_reader, daemon=True).start()
    hidden, stopping, downloads = False, False, set()
    exit_request, exit_notice = ExitRequest(), None
    idle_seconds = min(1800, max(1, int(config.get("idleSeconds", 45))))
    cycle = 0
    while not stopping:
        maintenance.reopen_suspension()
        cycle += 1
        emit(
            "progress",
            message="Restoring local services"
            if cycle > 1
            else "Starting local services",
        )
        ready, preferences, failure, job = None, None, None, None
        suspended = False
        with (logs / "services.log").open("ab") as log:
            child = subprocess.Popen(
                command,
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=log,
                creationflags=subprocess.CREATE_NO_WINDOW,
            )
            try:
                job = OwnedJob(child)
                started = time.monotonic()
                timeout = min(
                    180, max(1, int(config.get("startupTimeoutSeconds", 180)))
                )
                while time.monotonic() - started < timeout:
                    if child.poll() is not None:
                        raise RuntimeError(
                            "Local services could not start. See the private service log."
                        )
                    while not requests.empty():
                        action = requests.get_nowait()
                        if action == "stop":
                            stopping = True
                            break
                        elif action == "exit-wait":
                            exit_request.request()
                        elif action == "exit-cancel":
                            exit_request.active = False
                            emit("exit-cancelled")
                        if action in ("hide", "show"):
                            hidden = action == "hide"
                    if stopping:
                        break
                    if running_file.exists():
                        record = read(running_file)
                        if (
                            record["launcher"]["pid"] == child.pid
                            and record["candidateId"] == metadata["candidateId"]
                            and record["appId"] == APP_ID
                        ):
                            ready = record
                            break
                    time.sleep(0.15)
                if stopping:
                    break
                if ready is None:
                    raise TimeoutError(
                        "Local services did not become ready before the startup deadline."
                    )
                key = local_key(profile / "access.dpapi")
                api = ApiControl(ready, key)
                preferences = Preferences(
                    unlinked(owner / "preferences.json"),
                    ready["webUrl"],
                    ready["profileId"],
                )
                session = Session(
                    unlinked(owner / "session"),
                    ready["profileId"],
                    api.bases,
                    api.figures,
                )
                preferences.session = session
                preferences.maintenance = maintenance
                descriptor = {
                    **preferences.descriptor(),
                    "session": session.descriptor(),
                }
                route = session.initial["route"] if session.initial else "/"
                if session.initial:
                    route_project = route.split("?")[0].removeprefix("/workspace/")
                    if (
                        route.startswith("/workspace/")
                        and route_project != "new"
                        and api.bases([route_project]).get(route_project) is None
                    ):
                        route = "/"
                        session.warning = "The previous project is unavailable. The checkpoint was retained; choose the current saved state to continue."
                        descriptor["session"] = session.descriptor()
                write(
                    logs / "desktop-services.json",
                    {
                        **ready,
                        "controllerPid": os.getpid(),
                        "serviceReadyMs": (time.monotonic() - started) * 1000,
                        "cycle": cycle,
                    },
                )
                emit(
                    "ready",
                    url=ready["webUrl"] + route + "#labviz-access=" + key,
                    profileId=ready["profileId"],
                    candidateId=ready["candidateId"],
                    preferences=descriptor,
                    automaticSuspension=True,
                )
                quiet_since, blocked, phase, pending_id = (
                    time.monotonic(),
                    False,
                    "idle",
                    None,
                )
                checkpoint_deadline = 0.0
                drain_generation = None
                maintenance_mode = False
                previous_tick = None

                def gate(action, value=None):
                    nonlocal blocked, phase
                    try:
                        return api.call(action, value)
                    except (OSError, ValueError, TimeoutError) as error:
                        maintenance.fail(error)
                        maintenance.reopen_suspension()
                        session.cancel()
                        session.error = (
                            "Activity could not be verified; the workspace and services were retained. "
                            + str(error)
                        )
                        phase, blocked = "idle", True
                        try:
                            api.call("cancel-drain", {})
                        except (OSError, ValueError, TimeoutError):
                            pass
                        emit("sleep-blocked", message=session.error)
                        return None

                while child.poll() is None and not stopping:
                    tick = time.monotonic()
                    if quiet_clock_resumed(previous_tick, tick):
                        quiet_since = tick
                        if phase == "checkpoint":
                            session.cancel()
                            gate("cancel-drain", {})
                            maintenance.reopen_suspension()
                            phase = "idle"
                            emit("sleep-cancelled")
                    previous_tick = tick
                    external = owner / 'maintenance-request.json'
                    if external.exists() and not maintenance.pending:
                        try:
                            request = read(unlinked(external))
                            if not valid_maintenance_request(request, owner):
                                raise ValueError('Invalid external maintenance request')
                            if not maintenance.result or maintenance.result.get('requestId') != request['requestId']:
                                maintenance.submit({'action': 'prepare-update', 'requestId': request['requestId']})
                        except (OSError, ValueError, KeyError, TypeError):
                            pass
                    while not requests.empty():
                        action = requests.get_nowait()
                        if action == "stop":
                            stopping = True
                        elif action == "exit-wait":
                            if maintenance.pending:
                                emit("exit-blocked", message="Data maintenance is active. Finish it before exiting.")
                            else:
                                exit_request.request()
                                exit_notice = None
                                if phase == "checkpoint":
                                    session.cancel()
                                    gate("cancel-drain", {})
                                    phase = "idle"
                        elif action == "exit-cancel":
                            try:
                                exit_request.cancel(api)
                                maintenance.reopen_exit()
                                exit_notice = None
                                emit("exit-cancelled")
                            except (OSError, ValueError, TimeoutError) as error:
                                exit_request.active, blocked = False, True
                                maintenance.reopen_exit()
                                session.error = str(error)
                                emit("exit-blocked", message=str(error))
                        elif action == "exit-confirm" and exit_request.active and phase == "idle":
                            try:
                                if maintenance.check_exit(lambda busy: exit_request.check(api, session, downloads, busy)) == "exit-ready":
                                    stopping = True
                                    emit("exit-approved")
                                    break
                                else:
                                    exit_notice = None
                            except (OSError, ValueError, TimeoutError) as error:
                                exit_request.active, blocked = False, True
                                maintenance.reopen_exit()
                                session.error = str(error)
                                emit("exit-blocked", message=str(error))
                        elif action == "hide":
                            hidden, blocked, quiet_since = True, False, time.monotonic()
                        elif action == "show":
                            hidden, blocked, quiet_since = (
                                False,
                                False,
                                time.monotonic(),
                            )
                        elif action.startswith("download-start:"):
                            downloads.add(action.removeprefix("download-start:"))
                        elif action.startswith("download-end:"):
                            downloads.discard(action.removeprefix("download-end:"))
                        elif action == "webview-destroyed" and phase == "commit":
                            suspended = True
                            break
                    if suspended or stopping:
                        break
                    if exit_request.active and phase == "idle":
                        try:
                            if maintenance.pending:
                                raise ValueError("Data maintenance is active. Finish it before exiting.")
                            notice = maintenance.check_exit(lambda busy: exit_request.check(api, session, downloads, busy))
                            if notice != exit_notice:
                                emit(notice)
                                exit_notice = notice
                        except (OSError, ValueError, TimeoutError) as error:
                            try:
                                exit_request.cancel(api)
                            except (OSError, ValueError, TimeoutError):
                                exit_request.active = False
                            maintenance.reopen_exit()
                            emit("exit-blocked", message=str(error))
                        time.sleep(0.2)
                        continue
                    if phase != "idle" and ((not hidden and not maintenance_mode) or downloads):
                        maintenance.reopen_suspension()
                        session.cancel()
                        gate("cancel-drain", {})
                        phase = "idle"
                        emit("sleep-cancelled")
                        maintenance.fail('The operation was cancelled by new activity')
                        maintenance_mode = False
                    if phase == "idle":
                        if maintenance.pending and time.time() - maintenance.pending['startedUtcSeconds'] > 180:
                            maintenance.fail('The workspace is busy. Finish current work and retry.')
                        maintenance_mode = maintenance.pending is not None
                        if not hidden and blocked:
                            if gate("cancel-drain", {}) is not None:
                                blocked = False
                                session.error = None
                        if (not hidden and not maintenance_mode) or blocked or downloads or not session.idle():
                            quiet_since = time.monotonic()
                        else:
                            status = gate("activity")
                            if status is None:
                                continue
                            if not status["idle"]:
                                quiet_since = time.monotonic()
                            elif maintenance_mode or time.monotonic() - quiet_since >= idle_seconds:
                                status = gate("drain", {})
                                if status is None:
                                    continue
                                if not status["idle"]:
                                    gate("cancel-drain", {})
                                    quiet_since = time.monotonic()
                                else:
                                    pending_id = uuid.uuid4().hex
                                    drain_generation = status["generation"]
                                    session.begin(pending_id)
                                    checkpoint_deadline = time.monotonic() + 180
                                    phase = "checkpoint"
                                    emit("checkpoint", checkpointId=pending_id, maintenance=maintenance_mode)
                    elif phase == "checkpoint":
                        if session.error or time.monotonic() > checkpoint_deadline:
                            message = (
                                session.error
                                or "Draft saving timed out. The workspace and services were retained."
                            )
                            session.cancel()
                            gate("cancel-drain", {})
                            phase, blocked = "idle", True
                            session.error = message
                            emit("sleep-blocked", message=message)
                            maintenance.fail(message)
                            maintenance_mode = False
                        elif session.acknowledged == pending_id and session.idle():
                            status = gate("activity")
                            if status is None:
                                continue
                            if status["generation"] != drain_generation:
                                session.cancel()
                                gate("cancel-drain", {})
                                phase = "idle"
                                quiet_since = time.monotonic()
                                emit("sleep-cancelled")
                            elif status["idle"] and status["draining"]:
                                def final_idle():
                                    final = api.call("activity")
                                    return (final["idle"] and final["draining"]
                                            and final["generation"] == drain_generation
                                            and session.idle() and not downloads)
                                try:
                                    committed = maintenance.commit_suspension(final_idle, maintenance_mode)
                                except (OSError, ValueError, TimeoutError) as error:
                                    session.error = str(error)
                                    committed = False
                                if committed:
                                    phase = "commit"
                                    emit("suspend-commit", checkpointId=pending_id, maintenance=maintenance_mode)
                                else:
                                    session.cancel()
                                    gate("cancel-drain", {})
                                    phase, quiet_since = "idle", time.monotonic()
                                    emit("sleep-cancelled")
                    time.sleep(0.2)
                if not suspended and not stopping:
                    raise RuntimeError(
                        "A local service stopped. The editing window has been retained."
                    )
            except Exception as error:
                failure = str(error)
                emit("failed", message=failure)
            finally:
                if preferences:
                    preferences.close()
                outcome = stop_services(child, ready, profile, job, logs, failure)
        if failure:
            raise RuntimeError(failure)
        if ready and not outcome["normal"]:
            message = "Services did not stop normally. The saved checkpoint was retained; reopen LabViz to recover."
            maintenance.fail(message)
            raise RuntimeError(message)
        if suspended and not stopping:
            if maintenance_mode and maintenance.pending:
                emit('progress', message='Checking and copying local application data')
                result = maintenance.execute()
                if result['state'] == 'completed' and result['action'] == 'prepare-update':
                    external = owner / 'maintenance-request.json'
                    if external.exists() and valid_maintenance_request(read(unlinked(external)), owner):
                        emit('maintenance-exit', requestId=result['requestId'])
                        return
                hidden = False
                continue
            write(
                logs / "desktop-suspended.json",
                {
                    "profileId": ready["profileId"],
                    "cycle": cycle,
                    "checkpointId": pending_id,
                    "servicesStopped": True,
                },
            )
            emit("suspended", checkpointId=pending_id)
            if exit_request.active:
                emit("exit-ready")
            while not stopping and hidden:
                action = requests.get()
                if action == "stop":
                    stopping = True
                elif action == "exit-wait":
                    exit_request.request()
                    emit("exit-ready")
                elif action == "exit-confirm" and exit_request.active:
                    stopping = True
                    emit("exit-approved")
                elif action == "exit-cancel":
                    exit_request.active = False
                    maintenance.reopen_exit()
                    emit("exit-cancelled")
                elif action == "show":
                    hidden = False


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        emit("failed", message=str(error))
        raise SystemExit(1)
