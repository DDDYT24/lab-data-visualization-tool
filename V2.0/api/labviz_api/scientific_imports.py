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


#: Modules whose first desktop use is a fit or interval. The bundled runtime ships without
#: bytecode, so a cold ``scipy.stats`` import compiles for several seconds, and requests
#: that need it while it compiles all wait on that import.
FIT_MODULES = ("scipy.stats",)
_prefetch_lock = threading.Lock()
_prefetched: set[str] = set()


def prefetch(names: tuple[str, ...] = FIT_MODULES) -> None:
    """Import calculation modules in the background once a project is open.

    The import is still deferred past startup, but it is done before someone asks for an
    interval instead of while they wait for it.
    """
    if os.environ.get("LABVIZ_DESKTOP_MODE") != "1":
        return
    with _prefetch_lock:
        pending = [name for name in names if name not in _prefetched]
        _prefetched.update(pending)
    if not pending:
        return

    def load() -> None:
        for name in pending:
            try:
                importlib.import_module(name)
            except Exception:
                # The request that needs the module imports it again and reports the error.
                return

    threading.Thread(target=load, name="labviz-prefetch", daemon=True).start()
