"""The parity gate itself: it must notice a plan that changed, and the axis rule it relies on."""

from __future__ import annotations

import copy
import importlib.util
import math
import random
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from labviz_api.models import ChartSpec
from labviz_api.processing import analyze_chart, classic_plan
from labviz_api.refusals import PreparationError
from labviz_api.render_plan import AXIS_MAX_TICKS, AXIS_MAX_TICKS_3D, axis_ticks, nice_axis

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_v22_matrix.py"
_spec = importlib.util.spec_from_file_location("run_v22_matrix", SCRIPT)
assert _spec and _spec.loader
matrix: Any = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(matrix)


def plans(kind: str, rows: int) -> tuple[dict[str, Any], dict[str, Any]]:
    frame = pd.DataFrame({"x": [float(i) for i in range(rows)], "y": [i % 17 for i in range(rows)]})
    chart = ChartSpec.model_validate(
        {
            "schemaVersion": 1,
            "type": kind,
            "title": "Parity",
            "xAxis": {"field": "x", "title": "X", "unit": ""},
            "yAxis": {"field": "y", "title": "Y", "unit": ""},
            "series": [{"field": "y", "label": "Y", "color": "#2563EB"}],
            "panelCount": 1,
            "export": {
                "format": "png",
                "dpi": 300,
                "sizePreset": "single-column",
                "grayscalePreview": False,
            },
        }
    )
    derived = analyze_chart(frame, chart)
    preview = classic_plan(frame, chart, derived, detail="preview")
    export = classic_plan(frame, chart, derived)
    assert preview and export
    return preview, export


def test_reduced_preview_matches_its_export() -> None:
    preview, export = plans("line", 1000)
    assert preview["panels"][0]["marks"][0]["decimation"]["drawn"] < 1000
    assert matrix.plan_differences(preview, export) == []


def test_one_changed_tick_fails_the_gate() -> None:
    preview, export = plans("line", 1000)
    moved = copy.deepcopy(preview)
    moved["panels"][0]["axes"]["y"]["ticks"][1]["value"] += 1
    assert matrix.plan_differences(moved, export) == ["panel 1 axes"]


def test_a_restyled_or_unreduced_mark_fails_the_gate() -> None:
    preview, export = plans("line", 1000)
    recoloured = copy.deepcopy(preview)
    recoloured["panels"][0]["marks"][0]["color"] = "#000000"
    assert matrix.plan_differences(recoloured, export) == ["panel 1 mark line:0 style"]
    short_preview, short_export = plans("scatter", 40)
    cut = copy.deepcopy(short_preview)
    cut["panels"][0]["marks"][0]["y"].pop()
    assert matrix.plan_differences(cut, short_export) == ["panel 1 mark scatter:0"]


def test_axis_rule_holds_over_random_ranges() -> None:
    """A fixed sample of the 200,017 ranges first checked by hand on 2026-11-02."""
    rng = random.Random(7)
    pairs = []
    for _ in range(20_000):
        center = rng.choice([0, 1, -1, 1e3, -1e6, rng.uniform(-1e6, 1e6)]) * 10 ** rng.uniform(
            -30, 30
        )
        pairs.append((center, center + 10 ** rng.uniform(-30, 30)))
    pairs += [
        (0.0, 0.0), (5.0, 5.0), (-5.0, -5.0), (1e-300, 1e-300), (1e300, 1e300),
        (1.0, 1.0 + 2.3e-16), (1e16, 1e16 + 2),
        (1e308, 1.7e308), (0.1, 0.3), (1200.5, 1207), (123456789.0, 123456790.0),
        (1e9, 2e9), (1e12, 5e12), (1e-9, 3e-9),
    ]  # fmt: skip
    for low, high in pairs:
        (start, end), ticks = nice_axis(low, high)
        values = [tick["value"] for tick in ticks]
        labels = [tick["label"] for tick in ticks]
        assert math.isfinite(start) and math.isfinite(end) and start < end, (low, high)
        assert 2 <= len(ticks) <= AXIS_MAX_TICKS, (low, high)
        assert all(start <= value <= end for value in values), (low, high)
        assert len(set(values)) == len(values) and len(set(labels)) == len(labels), (low, high)
        assert max(len(label) for label in labels) <= 24, (low, high)
        short = [tick["value"] for tick in axis_ticks(start, end, AXIS_MAX_TICKS_3D)]
        assert 2 <= len(short) <= AXIS_MAX_TICKS_3D, (low, high)
        assert all(start <= value <= end for value in short), (low, high)


def test_a_span_outside_the_float_range_is_refused() -> None:
    for low, high in ((-1.7e308, 1.7e308), (0.0, 1.7e308), (1e-320, 2e-320)):
        with pytest.raises(PreparationError) as raised:
            nice_axis(low, high)
        assert raised.value.code == "chart-axis-range"
