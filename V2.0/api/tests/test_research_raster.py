from __future__ import annotations

import base64
import io
import xml.etree.ElementTree as ET

import numpy as np
import pytest
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.collections import LineCollection
from matplotlib.figure import Figure
from PIL import Image

from labviz_api.research_raster import freeze_raster_layers


@pytest.mark.parametrize("kind", ["points", "scatter", "connections"])
@pytest.mark.parametrize("scale", ["linear", "log"])
def test_cached_layers_keep_clipping_geometry_alpha_and_native_resolution(
    kind: str, scale: str
) -> None:
    figure = Figure(figsize=(4, 3), dpi=300)
    FigureCanvasAgg(figure)
    axis = figure.subplots()
    x = np.linspace(0.1, 2, 3000)
    y = np.sin(x * 9) * 0.8 + 2
    axis.set_xlim(0.3, 1.7)
    axis.set_ylim(0.8, 3.2)
    axis.set_yscale(scale)
    axis.set_title("Vector method label")
    if kind == "points":
        axis.plot(
            x,
            y,
            linestyle="None",
            marker="^",
            markersize=4,
            color="#008577",
            alpha=0.45,
            rasterized=True,
            zorder=4,
        )
        axis.plot(
            x,
            y + 0.2,
            linestyle="None",
            marker="o",
            markersize=3,
            color="#2478aa",
            alpha=0.5,
            rasterized=True,
            zorder=5,
        )
    elif kind == "scatter":
        axis.scatter(x, y, s=14, color="#008577", alpha=0.45, rasterized=True)
    else:
        axis.add_collection(
            LineCollection(
                [[(v, 1.0), (v, value)] for v, value in zip(x[::20], y[::20], strict=True)],
                colors="#008577",
                alpha=0.45,
                rasterized=True,
            )
        )
    # This reference is direct Agg drawing. The normal export keeps these exact PNG bytes.
    before = io.BytesIO()
    figure.savefig(before, format="png")
    assert freeze_raster_layers(figure) > 0
    after = io.BytesIO()
    figure.savefig(after, format="png")
    a = np.asarray(Image.open(io.BytesIO(before.getvalue())).convert("RGB"), dtype=float)
    b = np.asarray(Image.open(io.BytesIO(after.getvalue())).convert("RGB"), dtype=float)
    # A second alpha-compositing step can round channels; flipped/misplaced/clipped data cannot.
    assert np.abs(a - b).mean() < 0.2
    output = io.StringIO()
    figure.savefig(output, format="svg", backend="module://labviz_api.research_svg_backend")
    svg = ET.fromstring(output.getvalue())
    images = svg.findall(".//{http://www.w3.org/2000/svg}image")
    assert images
    assert "Vector method label" in output.getvalue()
    for image in images:
        encoded = image.attrib["{http://www.w3.org/1999/xlink}href"].split(",")[1]
        pixels = Image.open(io.BytesIO(base64.b64decode(encoded)))
        assert abs(pixels.width / float(image.attrib["width"]) * 72 - 300) < 2
        assert abs(pixels.height / float(image.attrib["height"]) * 72 - 300) < 2
    figure.clear()


def test_vector_artists_and_invisible_layers_are_not_converted() -> None:
    figure = Figure(figsize=(4, 3), dpi=300)
    axis = figure.subplots()
    axis.plot([0, 1], [1, 2], marker="o")
    axis.plot([0, 1], [2, 3], rasterized=True, visible=False)
    figure.draw_without_rendering()
    assert freeze_raster_layers(figure) == 0
    output = io.StringIO()
    figure.savefig(output, format="svg")
    assert "<image" not in output.getvalue()
    figure.clear()
