"""Words an entry brings with it, in both interface languages."""

from __future__ import annotations

from typing import NamedTuple


class Text(NamedTuple):
    en: str
    zh: str
