from __future__ import annotations

from typing import Any

import pandas as pd
import pytest

from labviz_api import analysis_memo
from labviz_api.analysis_memo import AnalysisMemo
from labviz_api.models import ChartSpec
from labviz_api.processing import analyze_chart

FRAME = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0], "a": [1.0, 2.0, 1.5, 3.0]})


def chart(**extra: Any) -> ChartSpec:
    return ChartSpec.model_validate(
        {
            "schemaVersion": 1,
            "type": "line",
            "title": "Memo",
            "xAxis": {"field": "x", "title": "Dose", "unit": "mg"},
            "yAxis": {"field": "a", "title": "Signal", "unit": "mV"},
            "series": [{"field": "a", "label": "Alpha", "color": "#2563EB"}],
            "panelCount": 1,
            "export": {
                "format": "png",
                "dpi": 300,
                "sizePreset": "single-column",
                "grayscalePreview": False,
            },
            **extra,
        }
    )


@pytest.fixture
def runs(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    original = analyze_chart
    calls: list[int] = []

    def counted(frame: pd.DataFrame, spec: ChartSpec) -> dict[str, Any]:
        calls.append(1)
        return original(frame, spec)

    monkeypatch.setattr(analysis_memo, "analyze_chart", counted)
    return calls


def test_restyling_reuses_the_analysis(runs: list[int]) -> None:
    memo = AnalysisMemo()
    first = memo.analysis("p", FRAME, chart())
    restyled = chart(
        title="Other",
        xAxis={"field": "x", "title": "Other", "unit": "s"},
        series=[{"field": "a", "label": "Alpha", "color": "#DC2626", "lineStyle": "dashed"}],
    )
    assert memo.analysis("p", FRAME, restyled) is first
    assert len(runs) == 1


@pytest.mark.parametrize(
    "changed",
    [
        chart(fitting={"model": "linear"}),
        chart(type="scatter"),
        chart(series=[{"field": "a", "label": "Renamed", "color": "#2563EB"}]),
    ],
)
def test_anything_the_analysis_reads_runs_it_again(runs: list[int], changed: ChartSpec) -> None:
    memo = AnalysisMemo()
    memo.analysis("p", FRAME, chart())
    memo.analysis("p", FRAME, changed)
    assert len(runs) == 2


def test_new_data_and_other_projects_run_it_again(runs: list[int]) -> None:
    memo = AnalysisMemo()
    memo.analysis("p", FRAME, chart())
    memo.analysis("p", FRAME.assign(a=[9.0, 8.0, 7.0, 6.0]), chart())
    memo.analysis("q", FRAME, chart())
    assert len(runs) == 3


def test_only_the_latest_analyses_are_kept(runs: list[int]) -> None:
    memo = AnalysisMemo(limit=1)
    memo.analysis("p", FRAME, chart())
    memo.analysis("q", FRAME, chart())
    memo.analysis("p", FRAME, chart())
    assert len(runs) == 3
