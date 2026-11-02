"""Fast, owned Windows service bootstrap with the installed profile's existing boundaries."""

from __future__ import annotations
import ctypes
from ctypes import wintypes
import hashlib
import json
import os
from pathlib import Path
import secrets
import socket
import struct
import subprocess
import sys
import time
import urllib.request
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parent))
from process_job import OwnedJob


def atomic(path: Path, value):
    temporary = path.with_name(path.name + ".new-" + uuid.uuid4().hex)
    data = value if isinstance(value, bytes) else json.dumps(value).encode()
    with temporary.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


class Mutex:
    def __init__(self, name, acquire=True):
        self.kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        self.kernel.CreateMutexW.argtypes = [
            ctypes.c_void_p,
            wintypes.BOOL,
            wintypes.LPCWSTR,
        ]
        self.kernel.CreateMutexW.restype = wintypes.HANDLE
        self.kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        self.kernel.ReleaseMutex.argtypes = [wintypes.HANDLE]
        self.kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        self.handle = self.kernel.CreateMutexW(None, False, name)
        if not self.handle:
            raise ctypes.WinError(ctypes.get_last_error())
        self.acquired = acquire
        if acquire and self.kernel.WaitForSingleObject(self.handle, 0) not in (0, 128):
            self.kernel.CloseHandle(self.handle)
            self.handle = None
            raise ValueError(
                "This installed profile is running or being maintained. Stop it first."
            )

    def close(self):
        if self.handle:
            if self.acquired:
                self.kernel.ReleaseMutex(self.handle)
            self.kernel.CloseHandle(self.handle)
            self.handle = None


def sid():
    adv = ctypes.WinDLL("advapi32", use_last_error=True)
    kernel = ctypes.WinDLL("kernel32")
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    adv.OpenProcessToken.argtypes = [
        wintypes.HANDLE,
        wintypes.DWORD,
        ctypes.POINTER(wintypes.HANDLE),
    ]
    adv.GetTokenInformation.argtypes = [
        wintypes.HANDLE,
        ctypes.c_int,
        ctypes.c_void_p,
        wintypes.DWORD,
        ctypes.POINTER(wintypes.DWORD),
    ]
    adv.ConvertSidToStringSidW.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(wintypes.LPWSTR),
    ]
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel.LocalFree.argtypes = [ctypes.c_void_p]
    token = wintypes.HANDLE()
    size = wintypes.DWORD()
    if not adv.OpenProcessToken(kernel.GetCurrentProcess(), 8, ctypes.byref(token)):
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        adv.GetTokenInformation(token, 1, None, 0, ctypes.byref(size))
        buffer = ctypes.create_string_buffer(size.value)
        if not adv.GetTokenInformation(token, 1, buffer, size, ctypes.byref(size)):
            raise ctypes.WinError(ctypes.get_last_error())
        pointer = ctypes.cast(buffer, ctypes.POINTER(ctypes.c_void_p))[0]
        text = wintypes.LPWSTR()
        if not adv.ConvertSidToStringSidW(pointer, ctypes.byref(text)):
            raise ctypes.WinError(ctypes.get_last_error())
        try:
            return text.value
        finally:
            kernel.LocalFree(text)
    finally:
        kernel.CloseHandle(token)


def protect(data):
    class Blob(ctypes.Structure):
        _fields_ = [("size", wintypes.DWORD), ("data", ctypes.POINTER(ctypes.c_ubyte))]

    crypt = ctypes.WinDLL("crypt32", use_last_error=True)
    kernel = ctypes.WinDLL("kernel32")
    crypt.CryptProtectData.argtypes = [
        ctypes.POINTER(Blob),
        wintypes.LPCWSTR,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        wintypes.DWORD,
        ctypes.POINTER(Blob),
    ]
    kernel.LocalFree.argtypes = [ctypes.c_void_p]
    buffer = (ctypes.c_ubyte * len(data)).from_buffer_copy(data)
    source = Blob(len(data), buffer)
    target = Blob()
    if not crypt.CryptProtectData(
        ctypes.byref(source), None, None, None, None, 0, ctypes.byref(target)
    ):
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        return ctypes.string_at(target.data, target.size)
    finally:
        kernel.LocalFree(target.data)


def port_owned(port, pid):
    library = ctypes.WinDLL("iphlpapi")
    size = wintypes.DWORD()
    library.GetExtendedTcpTable.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(wintypes.DWORD),
        wintypes.BOOL,
        wintypes.ULONG,
        ctypes.c_int,
        wintypes.ULONG,
    ]
    code = library.GetExtendedTcpTable(None, ctypes.byref(size), False, 2, 5, 0)
    if code != 122 or size.value > 16 * 1024 * 1024:
        raise OSError("TCP ownership table is unavailable")
    buffer = ctypes.create_string_buffer(size.value)
    if library.GetExtendedTcpTable(buffer, ctypes.byref(size), False, 2, 5, 0):
        return False
    count = struct.unpack_from("<I", buffer)[0]
    owners = []
    for index in range(count):
        state, local, local_port, _, _, owner = struct.unpack_from(
            "<6I", buffer, 4 + index * 24
        )
        if state == 2 and socket.ntohs(local_port & 65535) == port:
            owners.append((owner, local))
    return owners == [(pid, struct.unpack("<I", socket.inet_aton("127.0.0.1"))[0])]


def record(process):
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.GetProcessTimes.argtypes = [
        wintypes.HANDLE,
        *[ctypes.POINTER(wintypes.FILETIME)] * 4,
    ]
    kernel.QueryFullProcessImageNameW.argtypes = [
        wintypes.HANDLE,
        wintypes.DWORD,
        wintypes.LPWSTR,
        ctypes.POINTER(wintypes.DWORD),
    ]
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    handle = kernel.OpenProcess(0x1000, False, process.pid)
    if not handle:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        created, exit_time, kernel_time, user_time = [
            wintypes.FILETIME() for _ in range(4)
        ]
        if not kernel.GetProcessTimes(
            handle,
            *[ctypes.byref(v) for v in (created, exit_time, kernel_time, user_time)],
        ):
            raise ctypes.WinError(ctypes.get_last_error())
        length = wintypes.DWORD(32768)
        name = ctypes.create_unicode_buffer(length.value)
        if not kernel.QueryFullProcessImageNameW(handle, 0, name, ctypes.byref(length)):
            raise ctypes.WinError(ctypes.get_last_error())
        ticks = (
            (created.dwHighDateTime << 32) + created.dwLowDateTime + 504911232000000000
        )
        return {"pid": process.pid, "startedTicks": ticks, "executable": name.value}
    finally:
        kernel.CloseHandle(handle)


def shutdown_services(api, web, job, shutdown):
    """Finish API work before stopping the disposable web listener and owned job."""
    result = {"apiGraceful": False, "forced": False, "webStopMethod": None,
              "apiExitCode": None, "webExitCode": None, "error": None}
    try:
        if api is not None:
            if api.poll() is None:
                shutdown()
                result["apiExitCode"] = api.wait(timeout=45)
                result["apiGraceful"] = result["apiExitCode"] == 0
            else:
                result["apiExitCode"] = api.poll()
            if not result["apiGraceful"]:
                raise RuntimeError("The API did not acknowledge and complete normal shutdown")
        if web is not None:
            if web.poll() is not None:
                raise RuntimeError("The web service exited before planned shutdown")
            # Next has no authenticated shutdown endpoint. Only stop its listener
            # after the drained API has exited; this does not interrupt API jobs.
            web.terminate()
            result["webStopMethod"] = "terminate-after-api"
            result["webExitCode"] = web.wait(timeout=10)
    except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as error:
        result["forced"] = True
        result["error"] = str(error)
    finally:
        if job:
            job.close()
        for name, child in (("api", api), ("web", web)):
            if child is None:
                continue
            try:
                if child.poll() is None:
                    result["forced"] = True
                    child.terminate()
                result[name + "ExitCode"] = child.wait(timeout=10)
            except (OSError, subprocess.TimeoutExpired) as error:
                result["forced"] = True
                result["error"] = str(error)
    result["servicesStopped"] = all(child is None or child.poll() is not None for child in (api, web))
    return result


def main():
    started = time.perf_counter()
    timings = {}
    config = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8-sig"))
    payload = Path(config["payloadRoot"])
    profile_root = Path(config["backendProfile"])
    program = Path(config["programRoot"])
    sys.path.insert(0, str(payload / "V2.0/api"))
    from labviz_api.installed_profile import DATA_EPOCH, InstalledProfile, APP_ID, unlinked
    from labviz_api.installed_lifecycle import recover_profile

    unlinked(profile_root)
    unlinked(payload)
    unlinked(program)
    metadata = json.loads((payload / "candidate-build.json").read_text())
    installation = json.loads(
        unlinked(program / "installation.json").read_text(encoding="utf-8")
    )
    if (
        installation.get("appId") != APP_ID
        or installation.get("format") != "labviz-installed-v1"
    ):
        raise ValueError("Desktop installation identity differs")
    if (
        metadata["appId"] != APP_ID
        or metadata["candidateId"] != config["candidateId"]
        or metadata["dataEpoch"] != DATA_EPOCH
    ):
        raise ValueError("Desktop payload identity differs")
    profile = InstalledProfile.open(profile_root, create=True)
    mutex = Mutex(
        "Local\\LabVizV23-"
        + hashlib.sha256((sid() + "|" + str(profile.root).lower()).encode())
        .hexdigest()
        .upper()
    )
    application = Mutex("Local\\LabVizV23-App", False)
    api = web = job = None
    listeners = []
    key = None
    session = uuid.uuid4().hex
    logs = unlinked(profile.root / "logs")
    logs.mkdir(exist_ok=True)
    running = logs / "running.json"
    stop = logs / "stop.request"
    files = []
    stop_requested = False
    try:
        recover_profile(profile)
        timings["profileRecoveryMs"] = (time.perf_counter() - started) * 1000
        if profile.epoch() > metadata["dataEpoch"]:
            raise ValueError(
                "A newer data schema requires a compatible desktop runtime"
            )
        for state in (running, stop):
            if unlinked(state).exists():
                state.unlink()
        for _ in range(2):
            listener = socket.socket()
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
            listener.bind(("127.0.0.1", 0))
            listener.listen()
            listeners.append(listener)
        api_port, web_port = [s.getsockname()[1] for s in listeners]
        if {api_port, web_port} & {3000, 8000, 3001, 8001, 3305, 8305}:
            raise ValueError("A reserved application port was selected")
        raw = secrets.token_bytes(32)
        import base64

        key = base64.urlsafe_b64encode(raw).rstrip(b"=").decode()
        atomic(unlinked(profile.root / "access.dpapi"), protect(raw))
        origin = f"http://127.0.0.1:{web_port}"
        env = {
            **os.environ,
            "LABVIZ_DATABASE_PATH": str(profile.data / "labviz.db"),
            "LABVIZ_OBJECT_STORAGE_ROOT": str(profile.data / "objects"),
            "LABVIZ_PERSISTENCE_BACKEND": "sqlite",
            "LABVIZ_OBJECT_STORAGE_BACKEND": "local",
            "LABVIZ_ENVIRONMENT": "development",
            "LABVIZ_AUTH_MODE": "console",
            "LABVIZ_INSTALL_LINE": "2.3",
            "LABVIZ_DEVELOPMENT_LINE": "2.3",
            "LABVIZ_PROFILE_ROOT": str(profile.root),
            "LABVIZ_PROFILE_ID": profile.identifier,
            "LABVIZ_INSTALL_ROOT": str(program),
            "LABVIZ_API_PROXY_TARGET": f"http://127.0.0.1:{api_port}",
            "LABVIZ_ALLOWED_ORIGINS": origin,
            "LABVIZ_PUBLIC_WEB_URL": origin,
            "LABVIZ_LOCAL_ACCESS_KEY": key,
            "PORT": str(web_port),
            "HOSTNAME": "127.0.0.1",
            "NODE_ENV": "production",
            "NEXT_TELEMETRY_DISABLED": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "MPLCONFIGDIR": str(profile.root / "cache/matplotlib"),
            "LABVIZ_RUNTIME_ROLE": "api",
            "LABVIZ_DESKTOP_MODE": "1",
        }
        for name in ("PYTHONPATH", "PYTHONHOME", "PYTHONPYCACHEPREFIX", "NODE_PATH"):
            env.pop(name, None)

        def start(command, output, error):
            stdout = (logs / output).open("ab")
            stderr = (logs / error).open("ab")
            files.extend((stdout, stderr))
            return subprocess.Popen(
                command,
                stdin=subprocess.DEVNULL,
                stdout=stdout,
                stderr=stderr,
                env=env,
                cwd=payload,
                creationflags=subprocess.CREATE_NO_WINDOW,
            )

        listeners[0].close()
        api = start(
            [
                str(payload / "runtime/python/python.exe"),
                "-B",
                "-X",
                "utf8",
                "-m",
                "labviz_api.installed_server",
                "--port",
                str(api_port),
            ],
            "api.log",
            "api.error.log",
        )
        job = OwnedJob(api)
        listeners[1].close()
        web = start(
            [
                str(payload / "runtime/node/node.exe"),
                str(payload / "V2.0/web/server.js"),
            ],
            "web.log",
            "web.error.log",
        )
        job.add(web)
        timings["childrenSpawnedMs"] = (time.perf_counter() - started) * 1000
        headers = {
            "X-LabViz-Local-Key": key,
            "Cookie": f"labviz_local_session_v23_installed_{profile.identifier}={key}",
        }

        def call(suffix, post=False):
            request = urllib.request.Request(
                f"http://127.0.0.1:{api_port}/api/v1/local/{suffix}",
                data=b"" if post else None,
                headers=headers,
            )
            with urllib.request.urlopen(request, timeout=2) as response:
                return json.load(response)

        deadline = time.monotonic() + 120
        api_ready = web_ready = False
        while time.monotonic() < deadline:
            if api.poll() is not None or web.poll() is not None:
                raise RuntimeError("An owned service exited before readiness")
            if not api_ready and port_owned(api_port, api.pid):
                try:
                    call("session", True)
                    identity = call("instance")
                    api_ready = (
                        identity["appId"] == APP_ID
                        and identity["profileId"] == profile.identifier
                        and identity["pid"] == api.pid
                    )
                    if api_ready:
                        timings["apiReadyMs"] = (time.perf_counter() - started) * 1000
                except (OSError, ValueError):
                    pass
            if not web_ready and port_owned(web_port, web.pid):
                try:
                    with urllib.request.urlopen(origin + "/", timeout=2) as response:
                        web_ready = response.status == 200
                    if web_ready:
                        timings["webReadyMs"] = (time.perf_counter() - started) * 1000
                except OSError:
                    pass
            if api_ready and web_ready:
                break
            time.sleep(0.05)
        else:
            raise TimeoutError("Desktop services did not become ready")
        atomic(
            running,
            {
                "appId": APP_ID,
                "profileId": profile.identifier,
                "sessionId": session,
                "candidateId": metadata["candidateId"],
                "apiPort": api_port,
                "webUrl": origin,
                "launcher": record(SimpleProcess(os.getpid())),
                "api": record(api),
                "web": record(web),
            },
        )
        timings["readyMs"] = (time.perf_counter() - started) * 1000
        atomic(logs / "desktop-startup-timing.json", timings)
        while api.poll() is None and web.poll() is None:
            if stop.exists():
                request = json.loads(stop.read_text(encoding="utf-8-sig"))
                if (
                    request.get("profileId") == profile.identifier
                    and request.get("sessionId") == session
                ):
                    stop_requested = True
                    break
            time.sleep(0.1)
        else:
            raise RuntimeError("An owned desktop service exited unexpectedly")
    finally:
        failure = str(sys.exc_info()[1]) if sys.exc_info()[1] else None
        for listener in listeners:
            listener.close()

        def shutdown():
            if not key or not port_owned(api_port, api.pid):
                raise ValueError("API ownership could not be verified for shutdown")
            acknowledgement = call("shutdown", True)
            if (acknowledgement.get("accepted") is not True
                    or acknowledgement.get("appId") != APP_ID
                    or acknowledgement.get("profileId") != profile.identifier):
                raise ValueError("The API shutdown acknowledgement differs")

        outcome = shutdown_services(api, web, job, shutdown)
        for file in files:
            file.close()
        for state in (running, stop):
            if state.exists():
                try:
                    value = json.loads(state.read_text(encoding="utf-8-sig"))
                    if value.get("sessionId") == session:
                        state.unlink()
                except (ValueError, OSError):
                    pass
        outcome.update({"format": "labviz-desktop-service-exit-v1", "profileId": profile.identifier,
                        "sessionId": session, "stopRequested": stop_requested, "failure": failure})
        outcome["normal"] = bool(stop_requested and not failure and outcome["apiGraceful"]
                                 and not outcome["forced"] and outcome["servicesStopped"])
        try:
            atomic(logs / "desktop-service-exit.json", outcome)
        finally:
            mutex.close()
            application.close()
    if not outcome["normal"]:
        raise RuntimeError(outcome["error"] or "Local services did not stop normally")


class SimpleProcess:
    def __init__(self, pid):
        self.pid = pid


if __name__ == "__main__":
    main()
