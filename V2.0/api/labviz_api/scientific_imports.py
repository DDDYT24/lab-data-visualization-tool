"""Load desktop calculation libraries on first use; saved JSON views need no calculation."""

from __future__ import annotations

import importlib
import os
import threading
from types import ModuleType
from typing import Any


class DeferredModule:
    def __init__(self, name: str) -> None:
        self._name = name
        self._module: ModuleType | None = None
        self._lock = threading.RLock()

    def _load(self) -> ModuleType:
        with self._lock:
            if self._module is None:
                self._module = importlib.import_module(self._name)
            return self._module

    def __getattr__(self, name: str) -> Any:
        return getattr(self._load(), name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name in {"_name", "_module", "_lock"}:
            object.__setattr__(self, name, value)
        else:
            setattr(self._load(), name, value)

    def __delattr__(self, name: str) -> None:
        delattr(self._load(), name)


def desktop_module(name: str) -> Any:
    if os.environ.get("LABVIZ_DESKTOP_MODE") == "1":
        return DeferredModule(name)
    return importlib.import_module(name)


def scientific_module(name: str) -> Any:
    return desktop_module(name)
