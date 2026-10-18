"""Screen previews keep original artifacts immutable and caches bounded."""

import hashlib
import io
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from PIL import Image
from test_research_api import project, settings, unlock
from test_research_figure_board import charts

from labviz_api import research_preview as previews
from labviz_api.main import create_app
from labviz_api.research_preview import clear_previews, preview_png


def test_preview_cache_bounded_and_project_clear(monkeypatch: pytest.MonkeyPatch) -> None:
    previews._cache.clear()
    monkeypatch.setattr(previews, "_MAX_ENTRIES", 2)
    stream = io.BytesIO()
    Image.new("RGB", (3000, 1500), "blue").save(stream, format="PNG")
    source = stream.getvalue()
    for name in ("a", "b", "c"):
        rendered = previews.preview_png("project", name, source, 960)
        with Image.open(io.BytesIO(rendered)) as image:
            assert image.size == (960, 480)
    assert len(previews._cache) == 2
    assert source == stream.getvalue()
    previews.clear_previews("other")
    assert len(previews._cache) == 2
    previews.clear_previews("project")
    assert not previews._cache


def test_preview_keeps_full_download_and_requires_live_access(tmp_path: Path) -> None:
    config = settings(tmp_path)
    with TestClient(create_app(config)) as client:
        unlock(client)
        identifier = project(client)
        chart = charts(client, identifier)[0]["id"]
        base = f"/api/v1/projects/{identifier}/research"
        board = client.post(base + "/figure-boards", json={"panelChartIds": [chart]}).json()["id"]
        urls = [
            base + f"/charts/{chart}/artifact/png",
            base + f"/figure-boards/{board}/artifact/png",
        ]
        for url, width in zip(urls, (960, 1400), strict=True):
            original = client.get(url).content
            preview = client.get(url + "?preview=true")
            assert preview.status_code == 200
            with Image.open(io.BytesIO(preview.content)) as image:
                assert image.width <= width
            assert client.get(url + "?preview=true&download=true").content == original
            assert client.get(url).content == original
        assert any(key[0] == identifier for key in previews._cache)
        with TestClient(create_app(config)) as locked:
            for url in urls:
                assert locked.get(url + "?preview=true").status_code == 403
        assert client.delete(f"/api/v1/projects/{identifier}").status_code == 204
        assert not any(key[0] == identifier for key in previews._cache)
        for url in urls:
            assert client.get(url + "?preview=true").status_code == 404


def png(color: str) -> bytes:
    stream = io.BytesIO()
    Image.new("RGB", (1600, 1000), color).save(stream, format="PNG")
    return stream.getvalue()


def test_concurrent_preview_requests_share_resize_and_keep_original_bytes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project_id = uuid4().hex
    payload = png("#008577")
    original_hash = hashlib.sha256(payload).hexdigest()
    original = Image.Image.thumbnail
    entered, release = threading.Event(), threading.Event()
    calls = []

    def blocked(image: Image.Image, *args: Any, **kwargs: Any) -> None:
        calls.append(image.size)
        entered.set()
        assert release.wait(5)
        original(image, *args, **kwargs)

    monkeypatch.setattr(Image.Image, "thumbnail", blocked)
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(preview_png, project_id, "figure", payload, 640)
        try:
            assert entered.wait(5)
            second = pool.submit(preview_png, project_id, "figure", payload, 640)
        finally:
            release.set()
        a, b = first.result(timeout=5), second.result(timeout=5)
    assert a == b and len(calls) == 1
    assert Image.open(io.BytesIO(a)).size == (640, 400)
    assert hashlib.sha256(payload).hexdigest() == original_hash
    clear_previews(project_id)


def test_preview_cache_binds_payload_and_project_invalidation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project_id = uuid4().hex
    a, b = png("red"), png("blue")
    original = Image.Image.thumbnail
    calls = []

    def counted(image: Image.Image, *args: Any, **kwargs: Any) -> None:
        calls.append(image.size)
        original(image, *args, **kwargs)

    monkeypatch.setattr(Image.Image, "thumbnail", counted)
    first = preview_png(project_id, "same-id", a, 640)
    assert preview_png(project_id, "same-id", a, 640) == first
    assert preview_png(project_id, "same-id", b, 640) != first
    clear_previews(project_id)
    assert preview_png(project_id, "same-id", a, 640) == first
    assert len(calls) == 3
    clear_previews(project_id)


def test_project_clear_during_resize_does_not_repopulate_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project_id = uuid4().hex
    payload = png("green")
    entered, release = threading.Event(), threading.Event()
    original = Image.Image.thumbnail

    def blocked(image: Image.Image, *args: Any, **kwargs: Any) -> None:
        entered.set()
        assert release.wait(5)
        original(image, *args, **kwargs)

    monkeypatch.setattr(Image.Image, "thumbnail", blocked)
    with ThreadPoolExecutor(max_workers=1) as pool:
        result = pool.submit(preview_png, project_id, "figure", payload, 640)
        try:
            assert entered.wait(5)
            clear_previews(project_id)
        finally:
            release.set()
        assert Image.open(io.BytesIO(result.result(timeout=5))).size == (640, 400)
    assert not any(key[0] == project_id for key in previews._cache)
