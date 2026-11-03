"""Native pixels and cache misses, rather than tests of the key implementation."""

from __future__ import annotations

import base64
import io
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest
from matplotlib.backend_bases import RendererBase
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from PIL import Image

from labviz_api.research_raster import freeze_raster_layers


def figure_for(change: str) -> Figure:
    figure = Figure(figsize=(8, 3), dpi=300)
    FigureCanvasAgg(figure)
    x = np.linspace(0, 2, 20_001)
    for index in range(2):
        left = 0.1 + 0.5 * index + (0.0002 if index and change == "fractional" else 0)
        ax = figure.add_axes((left, 0.2, 0.35, 0.6))
        y = 2 + np.sin(x * 12)
        if index and change == "values":
            y = y + 0.1
        ax.set_xlim(0.3, 1.7)
        ax.set_ylim(0.9, 3.1 if not (index and change == "domain") else 3.5)
        if index and change == "log":
            ax.set_yscale("log")
        ax.plot(
            x,
            y,
            linestyle="None",
            marker="o",
            markersize=4,
            color="#008577" if not (index and change == "color") else "#1768ac",
            alpha=0.45 if not (index and change == "alpha") else 0.7,
            markeredgewidth=1 if not (index and change == "edge") else 2,
            clip_on=not (index and change == "clip"),
            rasterized=True,
        )
        ax.set_title("Vector labels retained")
    figure.draw_without_rendering()
    return figure


def svg_layers(figure: Figure) -> list[tuple[bytes, tuple[tuple[str, str], ...]]]:
    output = io.StringIO()
    figure.savefig(output, format="svg", backend="module://labviz_api.research_svg_backend")
    assert "Vector labels retained" in output.getvalue()
    images = ET.fromstring(output.getvalue()).findall(".//{http://www.w3.org/2000/svg}image")
    return [
        (
            base64.b64decode(image.attrib["{http://www.w3.org/1999/xlink}href"].split(",")[1]),
            tuple(
                sorted(
                    (k, v)
                    for k, v in image.attrib.items()
                    if k in {"x", "y", "width", "height", "transform"}
                )
            ),
        )
        for image in images
    ]


@pytest.mark.parametrize(
    "change", ["same", "fractional", "values", "domain", "color", "alpha", "edge", "clip", "log"]
)
def test_point_reuse_keeps_native_layers_and_rejects_different_geometry(
    monkeypatch: pytest.MonkeyPatch, change: str
) -> None:
    baseline = figure_for(change)
    freeze_raster_layers(baseline, reuse_points=False)
    expected = svg_layers(baseline)
    actual = figure_for(change)
    calls = []
    original = Line2D.draw

    def counted(artist: Line2D, renderer: RendererBase) -> None:
        if artist.get_rasterized():
            calls.append(artist)
        original(artist, renderer)

    monkeypatch.setattr(Line2D, "draw", counted)
    freeze_raster_layers(actual)
    assert len(calls) == (1 if change == "same" else 2)
    assert svg_layers(actual) == expected
    baseline.clear()
    actual.clear()


def test_thumbnail_width_changes_preview_only(tmp_path: Path) -> None:
    from fastapi.testclient import TestClient
    from test_research_api import project, settings, unlock
    from test_research_figure_board import charts

    from labviz_api.main import create_app

    with TestClient(create_app(settings(tmp_path))) as client:
        unlock(client)
        identifier = project(client)
        chart = charts(client, identifier)[0]["id"]
        url = f"/api/v1/projects/{identifier}/research/charts/{chart}/artifact/png"
        frozen = client.get(url).content
        small = client.get(url, params={"preview": True, "previewWidth": 480})
        assert small.status_code == 200
        assert Image.open(io.BytesIO(small.content)).width == 480
        assert (
            client.get(url, params={"preview": True, "previewWidth": 480, "download": True}).content
            == frozen
        )
        assert client.get(url, params={"preview": True, "previewWidth": 1}).status_code == 422


def test_production_grid_reuses_only_integer_translations(monkeypatch: pytest.MonkeyPatch) -> None:
    def make() -> Figure:
        figure = Figure(figsize=(14, 11), dpi=300)
        FigureCanvasAgg(figure)
        x = np.linspace(0.0, 2.0, 20_001)
        for index in range(4):
            axis = figure.add_subplot(2, 2, index + 1)
            axis.plot(
                x,
                2 + np.sin(x * 12),
                linestyle="None",
                marker="o",
                markersize=4,
                markeredgewidth=1,
                color="#008577",
                alpha=0.7,
                rasterized=True,
            )
            axis.set_xlim(0.3, 1.7)
            axis.set_ylim(0.9, 3.1)
            axis.set_title("Vector labels retained")
        figure.subplots_adjust(left=0.1, right=0.94, top=0.88, bottom=0.13, wspace=0.45, hspace=0.9)
        figure.draw_without_rendering()
        return figure

    before = make()
    freeze_raster_layers(before, reuse_points=False)
    expected = svg_layers(before)
    after = make()
    calls = []
    original = Line2D.draw

    def counted(artist: Line2D, renderer: RendererBase) -> None:
        if artist.get_rasterized():
            calls.append(artist)
        original(artist, renderer)

    monkeypatch.setattr(Line2D, "draw", counted)
    freeze_raster_layers(after)
    assert len(calls) == 2
    assert svg_layers(after) == expected
    before.clear()
    after.clear()


@pytest.mark.parametrize("enabled", [True, False])
def test_cold_scene_failure_restores_gc_state(tmp_path: Path, enabled: bool) -> None:
    import gc

    from fastapi.testclient import TestClient
    from test_research_api import project, settings, unlock
    from test_research_figure_board import charts

    from labviz_api.main import create_app
    from labviz_api.research_store import ResearchStore

    config = settings(tmp_path)
    application = create_app(config)
    original = gc.isenabled()
    try:
        with TestClient(application) as client:
            unlock(client)
            identifier = project(client)
            chart = charts(client, identifier)[0]["id"]
            base = f"/api/v1/projects/{identifier}/research/figure-boards"
            board = client.post(base, json={"panelChartIds": [chart]}).json()["id"]
            application.state.clear_research_caches(identifier)
            with ResearchStore(config.database_path).connect() as connection:
                connection.execute(
                    "UPDATE research_figure_artifacts SET scene_json=? WHERE board_id=?",
                    ("{", board),
                )
            (gc.enable if enabled else gc.disable)()
            with pytest.raises(ValueError):
                client.get(base + "/" + board + "/interaction")
            assert gc.isenabled() == enabled
    finally:
        (gc.enable if original else gc.disable)()
