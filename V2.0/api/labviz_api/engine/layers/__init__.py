"""The chart kind registry. A kind is one entry module, listed here."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from . import grouped_strip, paired, violin
from .base import Layer

LAYERS: dict[str, Layer] = {
    entry.LAYER.kind: entry.LAYER for entry in (grouped_strip, paired, violin)
}

# The literal type of the contracts is the registry's own keys.
if TYPE_CHECKING:
    LayerKind = str
else:
    LayerKind = Literal[tuple(LAYERS)]
