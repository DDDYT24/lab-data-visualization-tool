"""Stable category presentation independent of the visible category subset."""

from __future__ import annotations

import hashlib
import json
from typing import Any

COLORS = ["#1768ac", "#008577", "#7755aa", "#b66d13", "#bc5267", "#4d7083"]
MARKERS = ["o", "s", "^", "D", "v"]


def category_style(group: str | None) -> dict[str, Any]:
    code = (
        0
        if group is None
        else int.from_bytes(
            hashlib.sha256(json.dumps(group, ensure_ascii=False).encode("utf-8")).digest()[:4],
            "big",
        )
    )
    return {
        "group": group,
        "color": COLORS[code % len(COLORS)],
        "marker": MARKERS[(code // len(COLORS)) % len(MARKERS)],
    }
