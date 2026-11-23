"""One value that stands for the units of a group."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from statistics import mean, median

from ..text import Text


@dataclass(frozen=True)
class Summary:
    key: str
    label: Text
    compute: Callable[[list[float]], float]


MEAN = Summary("mean", Text("Mean", "均值"), mean)
MEDIAN = Summary("median", Text("Median", "中位数"), median)
