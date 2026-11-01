"""Run the unchanged RC launcher inside a fresh, owned comparison profile."""

from __future__ import annotations

import base64
import ctypes
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".new")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def local_key(path: Path) -> str:
    from ctypes import wintypes

    class Blob(ctypes.Structure):
        _fields_ = [("size", wintypes.DWORD), ("data", ctypes.POINTER(ctypes.c_ubyte))]

    raw = path.read_bytes()
    buffer = (ctypes.c_ubyte * len(raw)).from_buffer_copy(raw)
    source, target = Blob(len(raw), buffer), Blob()
    crypt = ctypes.WinDLL("crypt32", use_last_error=True)
    crypt.CryptUnprotectData.argtypes = [ctypes.POINTER(Blob), ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, wintypes.DWORD, ctypes.POINTER(Blob)]
    crypt.CryptUnprotectData.restype = wintypes.BOOL
    kernel = ctypes.WinDLL("kernel32")
    kernel.LocalFree.argtypes = [ctypes.c_void_p]
    if not crypt.CryptUnprotectData(ctypes.byref(source), None, None, None, None, 0, ctypes.byref(target)):
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        return base64.urlsafe_b64encode(ctypes.string_at(target.data, target.size)).rstrip(b"=").decode()
    finally:
        kernel.LocalFree(target.data)


def main() -> None:
    config_path = Path(sys.argv[1]).resolve()
    config = read(config_path)
    assert config["format"] == "labviz-desktop-trial-v1"
    trial = Path(config["trialRoot"]).resolve()
    allowed = (Path.home() / "LabVizD1").resolve()
    assert trial.is_relative_to(allowed) and trial != allowed
    assert not any(p.is_symlink() or p.is_junction() for p in (trial, *trial.parents))
    assert read(trial / "trial.json")["trialId"] == config["trialId"]
    payload = Path(config["payloadRoot"]).resolve()
    metadata = read(payload / "candidate-build.json")
    assert metadata["sourceCommit"] == "b587db848ba5f1930657d5a316e0160c71864ead"
    program = trial / "program"
    program.mkdir(exist_ok=True)
    write(program / "installation.json", {"appId": metadata["appId"], "format": "labviz-installed-v1"})
    appdata = Path(config.get("appdataRoot", str(trial / "appdata"))).resolve()
    assert appdata.is_relative_to(allowed) and appdata != allowed
    assert not any(p.is_symlink() or p.is_junction() for p in (appdata, *appdata.parents))
    appdata.mkdir(exist_ok=True)
    profile = appdata / "LabVizV23"
    running_file = profile / "logs/running.json"
    powershell = Path(os.environ["SystemRoot"]) / "System32/WindowsPowerShell/v1.0/powershell.exe"
    command = [str(powershell), "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(payload / "bin/start-v23-portable.ps1"), "-InstallRoot", str(program), "-ProfileRoot", str(profile), "-WebPort", "0", "-ApiPort", "0", "-SkipOpenBrowser"]
    env = {**os.environ, "LOCALAPPDATA": str(appdata), "PYTHONIOENCODING": "utf-8"}
    started = time.perf_counter()
    with (trial / "controller.log").open("wb") as log:
        child = subprocess.Popen(command, env=env, stdout=log, stderr=log, creationflags=subprocess.CREATE_NO_WINDOW)
        running = None
        try:
            deadline = time.monotonic() + 180
            while time.monotonic() < deadline:
                if child.poll() is not None:
                    raise RuntimeError(f"Production launcher exited with code {child.returncode}; inspect private controller log.")
                if running_file.exists():
                    candidate = read(running_file)
                    if candidate["launcher"]["pid"] == child.pid:
                        running = candidate
                        break
                time.sleep(0.15)
            if running is None:
                raise TimeoutError("Production service readiness timed out.")
            write(trial / "services.json", {**running, "controllerPid": os.getpid(), "serviceReadyMs": (time.perf_counter() - started) * 1000, "profileRoot": str(profile)})
            message = {"url": running["webUrl"] + "/#labviz-access=" + local_key(profile / "access.dpapi")}
            print(json.dumps(message), flush=True)
            # A closed pipe also stops the trial after an unexpected host exit.
            sys.stdin.readline()
        finally:
            if running is not None and child.poll() is None:
                write(profile / "logs/stop.request", {"profileId": running["profileId"], "sessionId": running["sessionId"]})
            try:
                exit_code = child.wait(timeout=90)
            except subprocess.TimeoutExpired:
                # Terminating this exact owned launcher closes its child JobObject.
                child.terminate()
                exit_code = child.wait(timeout=20)
                write(trial / "forced-stop.json", {"launcherPid": child.pid, "reason": "Owned launcher did not stop within 90 seconds"})
            write(trial / "service-exit.json", {"launcherExitCode": exit_code, "runningRecordRemoved": not running_file.exists()})


if __name__ == "__main__":
    main()
