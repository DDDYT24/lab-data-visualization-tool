"""Launch one GUI trial and sample its complete owned process tree."""

from __future__ import annotations

import argparse
import json
import os
import socket
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path

import psutil

ROOT = Path(__file__).resolve().parents[2]


def write(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".new")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def free_port() -> int:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return listener.getsockname()[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shell", choices=["tauri", "pywebview"], required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--scenario", choices=["smoke", "full", "large"])
    parser.add_argument("--auto-exit", action="store_true")
    parser.add_argument("--resume-from", type=Path)
    args = parser.parse_args()
    assert args.label.replace("-", "").isalnum()
    trial = Path.home() / "LabVizD1" / args.label
    trial.mkdir(parents=True, exist_ok=False)
    identifier = uuid.uuid4().hex
    payload = ROOT / "outputs/desktop-d01/payload"
    host = ROOT / ("desktop/prototypes/tauri/target/release/labviz-d1-tauri.exe" if args.shell == "tauri" else "outputs/desktop-d01/hosts/pywebview/LabVizD1pywebview/LabVizD1pywebview.exe")
    assert host.is_file() and (payload / "candidate-build.json").is_file()
    config = {"format": "labviz-desktop-trial-v1", "trialId": identifier, "trialRoot": str(trial), "shell": args.shell, "payloadRoot": str(payload), "python": str(payload / "runtime/python/python.exe"), "controller": str(ROOT / "desktop/prototypes/service_controller.py"), "webviewProfile": str(trial / "webview"), "debugPort": free_port(), "icon": str(payload / "V2.0/assets/labviz-logo.ico")}
    if args.resume_from:
        previous = args.resume_from.resolve()
        assert previous.is_relative_to(Path.home() / "LabVizD1")
        old = json.loads((previous / "launcher.json").read_text())
        old_exit = json.loads((previous / "exit.json").read_text())
        assert old["shell"] == args.shell and old_exit["hostExitCode"] == 0 and not old_exit["survivingOwnedProcesses"]
        config["appdataRoot"] = old.get("appdataRoot", str(previous / "appdata"))
        config["webviewProfile"] = old["webviewProfile"]
        config["resumeFrom"] = str(previous)
    write(trial / "trial.json", {"trialId": identifier, "shell": args.shell, "sourceCommit": "b587db848ba5f1930657d5a316e0160c71864ead", "realUserData": False})
    write(trial / "launcher.json", config)
    write(trial / "phase.json", {"phase": "startup"})
    started = time.time()
    process = subprocess.Popen([str(host), str(trial / "launcher.json")], creationflags=subprocess.CREATE_NO_WINDOW)
    write(trial / "host.json", {"pid": process.pid, "startedUtcSeconds": started, "executable": str(host), "config": str(trial / "launcher.json")})
    print(json.dumps({"trial": str(trial), "hostPid": process.pid, "debugPort": config["debugPort"]}), flush=True)
    driver = None
    driver_log = None
    exit_requested = False
    if args.scenario:
        driver_log = (trial / "workflow.log").open("wb")
        driver = subprocess.Popen([str(payload / "runtime/node/node.exe"), str(ROOT / "desktop/scripts/cdp_workflow.mjs"), str(trial), args.scenario], stdout=driver_log, stderr=driver_log, creationflags=subprocess.CREATE_NO_WINDOW)
    known = {}
    with (trial / "process-samples.jsonl").open("w", encoding="utf-8") as stream:
        while process.poll() is None:
            if driver and driver.poll() is not None and args.auto_exit and not exit_requested:
                write(trial / "host-command.json", {"trialId": identifier, "id": uuid.uuid4().hex, "action": "exit"})
                exit_requested = True
            try:
                descendants = [psutil.Process(process.pid), *psutil.Process(process.pid).children(recursive=True)]
            except psutil.NoSuchProcess:
                break
            entries = []
            for member in descendants:
                try:
                    identity = (member.pid, member.create_time())
                    known[member.pid] = identity[1]
                    memory = member.memory_info()
                    entries.append({"pid": member.pid, "parentPid": member.ppid(), "name": member.name(), "startedUtcSeconds": identity[1], "workingSetBytes": memory.rss, "privateCommitBytes": getattr(memory, "private", memory.vms), "cpuSeconds": sum(member.cpu_times()[:2])})
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            try:
                phase = json.loads((trial / "phase.json").read_text())["phase"]
            except (OSError, ValueError):
                phase = "unknown"
            sample = {"elapsedSeconds": time.time() - started, "utcSeconds": time.time(), "phase": phase, "workingSetBytesSum": sum(e["workingSetBytes"] for e in entries), "privateCommitBytesSum": sum(e["privateCommitBytes"] for e in entries), "processes": entries}
            stream.write(json.dumps(sample) + "\n")
            stream.flush()
            time.sleep(0.25)
    if driver and driver.poll() is None:
        driver.terminate()
        driver.wait(timeout=10)
    if driver_log:
        driver_log.close()
    time.sleep(3)
    survivors = []
    for pid, creation in known.items():
        try:
            candidate = psutil.Process(pid)
            if candidate.create_time() == creation:
                survivors.append({"pid": pid, "name": candidate.name()})
        except psutil.NoSuchProcess:
            pass
    write(trial / "exit.json", {"hostExitCode": process.wait(), "survivingOwnedProcesses": survivors, "elapsedSeconds": time.time() - started})
    appdata = Path(config.get("appdataRoot", str(trial / "appdata")))
    if (appdata / "LabVizV23/logs").exists():
        shutil.copytree(appdata / "LabVizV23/logs", trial / "service-logs", dirs_exist_ok=True)
    print(json.dumps({"completed": True, "trial": str(trial), "survivingOwnedProcesses": survivors}), flush=True)


if __name__ == "__main__":
    main()
