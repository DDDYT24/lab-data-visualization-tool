"""Keep the controller's launcher descendants in an owned Windows JobObject."""

from __future__ import annotations

import ctypes
from ctypes import wintypes


class BasicLimits(ctypes.Structure):
    _fields_ = [("processTime", ctypes.c_int64), ("jobTime", ctypes.c_int64), ("flags", wintypes.DWORD), ("minimumWorkingSet", ctypes.c_size_t), ("maximumWorkingSet", ctypes.c_size_t), ("activeProcesses", wintypes.DWORD), ("affinity", ctypes.c_size_t), ("priority", wintypes.DWORD), ("scheduling", wintypes.DWORD)]


class IoCounters(ctypes.Structure):
    _fields_ = [(name, ctypes.c_uint64) for name in ("readOperations", "writeOperations", "otherOperations", "readBytes", "writeBytes", "otherBytes")]


class ExtendedLimits(ctypes.Structure):
    _fields_ = [("basic", BasicLimits), ("io", IoCounters), ("processMemory", ctypes.c_size_t), ("jobMemory", ctypes.c_size_t), ("peakProcessMemory", ctypes.c_size_t), ("peakJobMemory", ctypes.c_size_t)]


class OwnedJob:
    def __init__(self, process):
        self.kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        self.kernel.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
        self.kernel.CreateJobObjectW.restype = wintypes.HANDLE
        self.kernel.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
        self.kernel.SetInformationJobObject.restype = wintypes.BOOL
        self.kernel.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
        self.kernel.AssignProcessToJobObject.restype = wintypes.BOOL
        self.kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        self.handle = self.kernel.CreateJobObjectW(None, None)
        if not self.handle:
            raise ctypes.WinError(ctypes.get_last_error())
        limits = ExtendedLimits()
        limits.basic.flags = 0x2000
        if not self.kernel.SetInformationJobObject(self.handle, 9, ctypes.byref(limits), ctypes.sizeof(limits)) or not self.kernel.AssignProcessToJobObject(self.handle, int(process._handle)):
            error = ctypes.get_last_error()
            self.close()
            raise ctypes.WinError(error)

    def close(self):
        if self.handle:
            self.kernel.CloseHandle(self.handle)
            self.handle = None

    def add(self, process):
        if not self.kernel.AssignProcessToJobObject(self.handle, int(process._handle)):
            raise ctypes.WinError(ctypes.get_last_error())
