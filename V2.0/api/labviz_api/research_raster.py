"""Reuse final-layout 300 dpi layers without changing the direct PNG renderer."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from typing import Any

import numpy as np
from matplotlib.backends.backend_agg import RendererAgg
from matplotlib.collections import LineCollection, PathCollection
from matplotlib.figure import Figure
from matplotlib.image import BboxImage
from matplotlib.lines import Line2D
from matplotlib.markers import MarkerStyle
from matplotlib.transforms import Bbox, TransformedBbox


def _point_layer_key(artist: Any, axis: Any, origin: tuple[int, int], dpi: float) -> str | None:
    # Only plain marker-only lines qualify. Unknown/custom rendering always draws normally.
    if not isinstance(artist, Line2D) or artist.get_linestyle() != "None":
        return None
    if (
        artist.get_markevery() is not None
        or artist.get_path_effects()
        or artist.get_agg_filter() is not None
        or artist.get_sketch_params() is not None
        or artist.get_clip_path() is not None
        or artist.get_fillstyle() != "full"
        or np.asarray(artist.get_path().vertices).shape[0] <= 20_000
    ):
        return None
    marker = MarkerStyle(artist.get_marker(), fillstyle=artist.get_fillstyle())
    clip = artist.get_clip_box()
    bounds = np.asarray(clip.extents) if clip is not None else np.asarray(axis.figure.bbox.extents)
    bounds = bounds - np.asarray([origin[0], origin[1], origin[0], origin[1]])
    transform = artist.get_transform()
    vertices = np.asarray(transform.transform_non_affine(artist.get_path().vertices))
    matrix = np.asarray(transform.get_affine().get_matrix()).copy()
    matrix[:2, 2] -= np.asarray(origin)
    style = [
        dpi,
        artist.get_markersize(),
        artist.get_markeredgewidth(),
        artist.get_color(),
        artist.get_markerfacecolor(),
        artist.get_markeredgecolor(),
        artist.get_alpha(),
        artist.get_antialiased(),
        artist.get_snap(),
        artist.get_clip_on(),
    ]
    digest = hashlib.sha256(json.dumps(style, default=str).encode())
    # Remove only floating arithmetic noise from integer translations between equal panels.
    # The cached layer keeps its native pixel grid; fractional translations get different keys.
    for array in [
        vertices,
        np.round(matrix, 9),
        np.round(bounds, 9),
        np.asarray(marker.get_path().vertices),
        np.asarray(marker.get_transform().get_matrix()),
    ]:
        digest.update(str(array.shape).encode())
        digest.update(np.where(array == 0, 0.0, array).tobytes())
    codes = marker.get_path().codes
    if codes is not None:
        digest.update(np.asarray(codes).tobytes())
    return digest.hexdigest()


def freeze_raster_layers(figure: Figure, *, reuse_points: bool = True) -> int:
    """Paint selected artists once; preserve vector labels, stacking and native resolution.

    Reuse is private to this Figure, limited to 32 MiB, and keyed on final transformed
    coordinates, clipping, marker geometry/style, alpha and DPI. Fractional shifts,
    shared-domain changes, different values and custom artists do not alias.
    """
    new_renderer: Callable[[int, int, float], RendererAgg] = RendererAgg
    width, height = map(int, figure.bbox.size)
    renderer = new_renderer(width, height, figure.dpi)
    rgba: Callable[[], memoryview] = renderer.buffer_rgba
    clear: Callable[[], None] = renderer.clear
    cache: dict[str, tuple[int, int, np.ndarray[Any, Any]]] = {}
    cache_bytes = 0
    count = 0
    for axis in figure.axes:
        origin = (int(np.floor(round(axis.bbox.x0, 9))), int(np.floor(round(axis.bbox.y0, 9))))
        for artist in list(axis.get_children()):
            if not isinstance(artist, (Line2D, LineCollection, PathCollection)):
                continue
            if not artist.get_visible() or not artist.get_rasterized():
                continue
            key = _point_layer_key(artist, axis, origin, figure.dpi) if reuse_points else None
            reused = cache.get(key) if key is not None else None
            if reused is not None:
                dx, dy, cropped = reused
                left, bottom = origin[0] + dx, origin[1] + dy
                right, top = left + cropped.shape[1], bottom + cropped.shape[0]
            else:
                clear()
                artist.draw(renderer)
                pixels = np.asarray(rgba())
                occupied_y = np.flatnonzero(pixels[:, :, 3].any(axis=1))
                occupied_x = np.flatnonzero(pixels[:, :, 3].any(axis=0))
                if not occupied_y.size or not occupied_x.size:
                    artist.remove()
                    continue
                upper, lower = int(occupied_y[0]), int(occupied_y[-1]) + 1
                left, right = int(occupied_x[0]), int(occupied_x[-1]) + 1
                bottom, top = height - lower, height - upper
                cropped = pixels[upper:lower, left:right].copy()
                if key is not None and cache_bytes + cropped.nbytes <= 32 * 2**20:
                    cache[key] = (left - origin[0], bottom - origin[1], cropped)
                    cache_bytes += cropped.nbytes
                del pixels
            bounds = Bbox.from_extents(left / width, bottom / height, right / width, top / height)
            image = BboxImage(
                TransformedBbox(bounds, figure.transFigure),
                interpolation="none",
                origin="upper",
                resample=False,
                zorder=artist.get_zorder(),
            )
            image.set_data(cropped)
            axis.add_artist(image)
            count += 1
            artist.remove()
    return count
