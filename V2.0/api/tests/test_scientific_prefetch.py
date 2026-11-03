from __future__ import annotations

import importlib
import threading

import pytest

from labviz_api import scientific_imports


@pytest.fixture
def imported(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    names: list[str] = []
    monkeypatch.setattr(scientific_imports, "_prefetched", set())
    monkeypatch.setattr(importlib, "import_module", names.append)
    return names


def join_prefetch() -> None:
    for thread in threading.enumerate():
        if thread.name == "labviz-prefetch":
            thread.join(5)


def test_desktop_mode_imports_fit_modules_once_in_the_background(
    monkeypatch: pytest.MonkeyPatch, imported: list[str]
) -> None:
    monkeypatch.setenv("LABVIZ_DESKTOP_MODE", "1")

    scientific_imports.prefetch()
    scientific_imports.prefetch()
    join_prefetch()

    assert imported == ["scipy.stats"]


def test_server_mode_leaves_imports_to_first_use(
    monkeypatch: pytest.MonkeyPatch, imported: list[str]
) -> None:
    monkeypatch.delenv("LABVIZ_DESKTOP_MODE", raising=False)

    scientific_imports.prefetch()
    join_prefetch()

    assert imported == []
