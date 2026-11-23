"""Raw points by group."""

from __future__ import annotations

from ..text import Text
from .base import Layer
from .grouping import grouped

LAYER = Layer(
    kind="grouped_strip",
    title=Text("Grouped raw points", "分组原始点图"),
    short=Text("Raw points", "原始点图"),
    options=("summary",),
    plan=grouped,
)
