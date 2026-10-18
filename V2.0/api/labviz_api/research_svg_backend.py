"""SVG backend retaining Matplotlib geometry with per-draw marker style caching.

The graphics context is constant within draw_markers. Avoid rebuilding the identical
style string for each of 100,000 uses; keep per-marker opacity and clipping unchanged.
Only research savefig calls select this backend. No global Matplotlib monkeypatch.
"""

from __future__ import annotations

import codecs
from typing import Any, cast

from matplotlib import cbook
from matplotlib.backends.backend_mixed import MixedModeRenderer
from matplotlib.backends.backend_svg import FigureCanvasSVG, RendererSVG


class CachedMarkerRenderer(RendererSVG):
    _marker_style: str | None = None

    def _get_style(self, gc: Any, rgbFace: Any) -> str:
        if self._marker_style is not None:
            return self._marker_style
        return str(cast(Any, super())._get_style(gc, rgbFace))

    def draw_markers(
        self,
        gc: Any,
        marker_path: Any,
        marker_trans: Any,
        path: Any,
        trans: Any,
        rgbFace: Any = None,
    ) -> None:
        self._marker_style = str(cast(Any, super())._get_style(gc, rgbFace))
        try:
            cast(Any, super()).draw_markers(gc, marker_path, marker_trans, path, trans, rgbFace)
        finally:
            self._marker_style = None


class FigureCanvas(FigureCanvasSVG):
    def print_svg(
        self, filename: Any, *, bbox_inches_restore: Any = None, metadata: Any = None, **kwargs: Any
    ) -> None:
        with cbook.open_file_cm(filename, "w", encoding="utf-8") as raw_handle:
            handle: Any = raw_handle
            if not cbook.file_requires_unicode(handle):
                handle = codecs.getwriter("utf-8")(handle)
            dpi = self.figure.dpi
            self.figure.dpi = 72
            width, height = self.figure.get_size_inches()
            renderer = cast(Any, MixedModeRenderer)(
                self.figure,
                width,
                height,
                dpi,
                cast(Any, CachedMarkerRenderer)(
                    width * 72, height * 72, handle, image_dpi=dpi, metadata=metadata
                ),
                bbox_inches_restore=bbox_inches_restore,
            )
            self.figure.draw(renderer)
            renderer.finalize()
