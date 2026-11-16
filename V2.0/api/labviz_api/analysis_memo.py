"""Finished chart analyses, kept so restyling a chart rebuilds its plan and nothing else."""

from __future__ import annotations

from typing import TYPE_CHECKING

from labviz_api.scientific_imports import scientific_module

if TYPE_CHECKING:
    import pandas as pd
else:
    pd = scientific_module("pandas")

import hashlib
import json
import threading
from collections import OrderedDict
from typing import Any

from .models import ChartSpec
from .processing import analyze_chart

ANALYSES_KEPT = 8


def analysis_inputs(chart: ChartSpec) -> dict[str, Any]:
    """What ``analyze_chart`` reads. Colours, titles and export settings are not in it."""
    return {
        "fitting": chart.fitting.model_dump(mode="json"),
        "groupField": chart.group_field,
        "panelCount": chart.panel_count,
        "secondaryYAxis": chart.secondary_y_axis.model_dump(mode="json"),
        "series": [[item.field, item.label, item.panel, item.y_axis] for item in chart.series],
        "type": chart.type,
        "uncertainty": chart.uncertainty.model_dump(mode="json"),
        "xAxis": chart.x_axis.field,
        "yAxis": chart.y_axis.field,
    }


class AnalysisMemo:
    """The latest analyses per process. Callers check project access before asking."""

    def __init__(self, limit: int = ANALYSES_KEPT) -> None:
        self._limit = limit
        self._lock = threading.Lock()
        self._kept: OrderedDict[str, dict[str, Any]] = OrderedDict()

    def analysis(self, project_id: str, frame: pd.DataFrame, chart: ChartSpec) -> dict[str, Any]:
        digest = hashlib.sha256(pd.util.hash_pandas_object(frame).to_numpy().tobytes())
        digest.update(json.dumps(analysis_inputs(chart), sort_keys=True).encode("utf-8"))
        digest.update(project_id.encode("utf-8"))
        key = digest.hexdigest()
        with self._lock:
            kept = self._kept.get(key)
            if kept is not None:
                self._kept.move_to_end(key)
                return kept
        result = analyze_chart(frame, chart)
        with self._lock:
            self._kept[key] = result
            while len(self._kept) > self._limit:
                self._kept.popitem(last=False)
        return result
