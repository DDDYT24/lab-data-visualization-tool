"""Read-only native Windows process-tree working-set samples for an owned browser.

Toolhelp PROCESSENTRY32W and PSAPI PROCESS_MEMORY_COUNTERS follow Microsoft APIs.
No application data, process command lines, or other browser sessions are collected.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import sys
import time
from ctypes import wintypes
from pathlib import Path
from typing import Any


def sample(root: int, stop_file: Path, interval: float = 0.25) -> dict[str, Any]:
    if sys.platform != "win32":
        return {"available": False, "reason": "Windows native process sampler unavailable"}

    class ProcessEntry(ctypes.Structure):
        _fields_ = [
            ("size", wintypes.DWORD),
            ("usage", wintypes.DWORD),
            ("pid", wintypes.DWORD),
            ("heap", ctypes.c_size_t),
            ("module", wintypes.DWORD),
            ("threads", wintypes.DWORD),
            ("parent", wintypes.DWORD),
            ("priority", wintypes.LONG),
            ("flags", wintypes.DWORD),
            ("name", wintypes.WCHAR * 260),
        ]

    class MemoryCounters(ctypes.Structure):
        _fields_ = [("size", wintypes.DWORD), ("faults", wintypes.DWORD)] + [
            (name, ctypes.c_size_t)
            for name in [
                "peak",
                "working",
                "paged_peak",
                "paged",
                "nonpaged_peak",
                "nonpaged",
                "pagefile",
                "pagefile_peak",
            ]
        ]

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    kernel.CreateToolhelp32Snapshot.argtypes = [wintypes.DWORD, wintypes.DWORD]
    kernel.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
    for name in ["Process32FirstW", "Process32NextW"]:
        getattr(kernel, name).argtypes = [wintypes.HANDLE, ctypes.POINTER(ProcessEntry)]
        getattr(kernel, name).restype = wintypes.BOOL
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel.CloseHandle.restype = wintypes.BOOL
    psapi.GetProcessMemoryInfo.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(MemoryCounters),
        wintypes.DWORD,
    ]
    psapi.GetProcessMemoryInfo.restype = wintypes.BOOL

    def inventory() -> dict[int, tuple[int, str]]:
        snapshot = kernel.CreateToolhelp32Snapshot(2, 0)
        if snapshot == ctypes.c_void_p(-1).value:
            raise ctypes.WinError(ctypes.get_last_error())
        try:
            entry = ProcessEntry()
            entry.size = ctypes.sizeof(entry)
            items = {}
            found = kernel.Process32FirstW(snapshot, ctypes.byref(entry))
            while found:
                items[entry.pid] = (entry.parent, entry.name)
                found = kernel.Process32NextW(snapshot, ctypes.byref(entry))
            return items
        finally:
            kernel.CloseHandle(snapshot)

    peak, count, incomplete, maximum_interval, maximum_duration = 0, 0, 0, 0.0, 0.0
    peak_processes = []
    peak_stage = "unknown"
    stage_file = Path(str(stop_file) + "-stage")
    previous, deadline = None, time.monotonic() + 900
    while not stop_file.exists() and time.monotonic() < deadline:
        started = time.monotonic()
        items = inventory()
        if root not in items:
            break
        ids = {root}
        while True:
            expanded = ids | {pid for pid, (parent, _) in items.items() if parent in ids}
            if expanded == ids:
                break
            ids = expanded
        values = []
        for pid in ids:
            handle = kernel.OpenProcess(0x1000, False, pid)
            if not handle:
                continue
            try:
                counters = MemoryCounters()
                counters.size = ctypes.sizeof(counters)
                if psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.size):
                    values.append(
                        {
                            "id": pid,
                            "name": items[pid][1],
                            "workingSetMiB": counters.working / 2**20,
                        }
                    )
            finally:
                kernel.CloseHandle(handle)
        incomplete += len(values) != len(ids)
        total = sum(value["workingSetMiB"] for value in values)
        if total > peak:
            peak, peak_processes = total, values
            try:
                peak_stage = stage_file.read_text(encoding="utf-8")[:80]
            except OSError:
                peak_stage = "unknown"
        if previous is not None:
            maximum_interval = max(maximum_interval, (started - previous) * 1000)
        previous = started
        count += 1
        elapsed = time.monotonic() - started
        maximum_duration = max(maximum_duration, elapsed * 1000)
        time.sleep(max(0, interval - elapsed))
    return {
        "available": count > 0,
        "rootPid": root,
        "sampleCount": count,
        "incompleteSamples": incomplete,
        "peakProcesses": peak_processes,
        "peakStage": peak_stage,
        "sampledProcessTreeWorkingSetPeakMiB": peak,
        "maxIntervalMs": maximum_interval,
        "maxQueryDurationMs": maximum_duration,
        "requestedIntervalMs": interval * 1000,
        "method": "Toolhelp process inventory + GetProcessMemoryInfo; read-only native calls",
        "scope": (
            "Observed synchronized working-set samples of dedicated browser descendants; "
            "intervals may miss transient peaks."
        ),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=int, required=True)
    parser.add_argument("--stop-file", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(sample(args.root, args.stop_file)), flush=True)
