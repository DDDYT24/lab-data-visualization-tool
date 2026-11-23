"""The statistics registry. A method is an entry, listed here."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from .summaries import MEAN, MEDIAN, Summary

SUMMARIES: dict[str, Summary] = {entry.key: entry for entry in (MEAN, MEDIAN)}

# The literal types of the contracts are the registry's own keys.
if TYPE_CHECKING:
    SummaryKey = str
    SummaryChoice = str
else:
    SummaryKey = Literal[tuple(SUMMARIES)]
    SummaryChoice = Literal[("none", *SUMMARIES)]
