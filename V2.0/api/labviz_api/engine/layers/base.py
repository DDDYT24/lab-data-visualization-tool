"""What a research chart kind brings to the registry."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from ..text import Text

if TYPE_CHECKING:
    from ...research_charts import ScientificChartSpec

Mark = dict[str, Any]
Result = dict[str, Any]
# Included observations by group label.
Groups = dict[str | None, list[dict[str, Any]]]


@dataclass(frozen=True)
class PlanContext:
    """One computed result and the presentation its plan is built with."""

    result: Result
    colors: Mapping[Any, str]
    use_grayscale: bool
    category_counts: bool

    def name(self, group: str | None) -> str:
        if group is not None:
            return group
        return "未声明分组" if self.result["spec"]["locale"] == "zh" else "Undeclared"


@dataclass
class LayerPlan:
    """The marks under and over the points, the x axis, and the y values they reach."""

    x_axis: dict[str, Any]
    below: list[Mark] = field(default_factory=list)
    above: list[Mark] = field(default_factory=list)
    legend: list[dict[str, Any]] = field(default_factory=list)
    y_values: list[float] = field(default_factory=list)


def nothing(*_: Any) -> None:
    return None


@dataclass(frozen=True)
class Layer:
    kind: str
    # The default chart title, and the name in the chart family picker.
    title: Text
    # The name of an untitled chart in a figure's lists.
    short: Text
    # ScientificChartSpec fields that are refused on the kinds that do not list them. A field every
    # kind accepts and only some use (the jitter, the density scale) is not listed.
    options: tuple[str, ...]
    plan: Callable[[PlanContext], LayerPlan]
    # Column roles the data must map.
    roles: tuple[str, ...] = ()
    x_axis: Literal["group", "pairTimes"] = "group"
    # Geometry rules of chart-render-v2 that the plan names.
    geometry: tuple[str, ...] = ("jitter",)
    limits: Mapping[str, int] = field(default_factory=dict)
    # Further rules of the spec; raises PreparationError.
    check: Callable[[ScientificChartSpec], None] = nothing
    # Adds the kind's own numbers to the shared result.
    compute: Callable[[Result, Groups, ScientificChartSpec], None] = nothing
    # Method disclosures: on the chart, under a figure panel, in an output's methods text.
    plan_note: str | None = None
    footer_note: str | None = None
    output_note: Text | None = None

    def x_signature(self, result: Result) -> list[Any]:
        """Charts can share an x axis when these are equal."""
        if self.x_axis == "pairTimes":
            return ["paired", result["spec"]["pairTimes"]]
        return ["group", [group["label"] for group in result["groups"]]]
