"""RenderPlan v1: the drawing contract between computed results and every renderer.

A plan fixes everything a renderer needs: axis domains and ticks, mark geometry, colours
and the text shown. Matplotlib and the browser then draw the same figure without
recomputing statistics or choosing their own axis limits.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from typing import Annotated, Any, Literal

from pydantic import Field

from .render_contract import GEOMETRY
from .research_contract import Digest, ResearchModel
from .research_preparation import PreparationError
from .research_styles import COLORS

CONTRACT = "render-plan-v1"
AXIS_NICE_RULE = GEOMETRY["axis"]["rule"]
# Matplotlib's default data margin, kept so new plans frame data as rc.2 charts did.
AXIS_MARGIN = GEOMETRY["axis"]["margin"]
AXIS_MAX_TICKS = GEOMETRY["axis"]["maxTicks"]
AXIS_STEP_MULTIPLES = GEOMETRY["axis"]["stepMultiples"]
# Below this span, relative to the values, ticks cannot be told apart: treat them as one value.
AXIS_MIN_RELATIVE_SPAN = GEOMETRY["axis"]["minRelativeSpan"]
# Labels stay in plain digits for these powers of ten, where Matplotlib also leaves them plain.
AXIS_PLAIN_EXPONENTS = range(
    GEOMETRY["axis"]["plainExponents"][0], GEOMETRY["axis"]["plainExponents"][1] + 1
)
DENSE_POINT_THRESHOLD = GEOMETRY["points"]["denseThreshold"]

Color = Annotated[str, Field(pattern=r"^#[0-9a-f]{6}$")]
Marker = Literal["o", "s", "^", "D", "v"]


class Tick(ResearchModel):
    value: float
    label: str


class Axis(ResearchModel):
    scale: Literal["linear", "category"]
    domain: tuple[float, float]
    ticks: list[Tick]
    title: str
    nice_rule: Literal["nice-1-2-5-v1"] | None


class PanelAxes(ResearchModel):
    x: Axis
    y: Axis


class PointsMark(ResearchModel):
    kind: Literal["points"]
    id: str
    group: str
    color: Color
    marker: Marker
    size: float
    opacity: float
    x: list[float]
    y: list[float]
    source_refs: list[str]


class SegmentsMark(ResearchModel):
    kind: Literal["segments"]
    id: str
    group: str
    color: Color
    width: float
    opacity: float
    x0: list[float]
    y0: list[float]
    x1: list[float]
    y1: list[float]
    source_refs: list[tuple[str, str]]


class ViolinMark(ResearchModel):
    kind: Literal["violin"]
    id: str
    group: str
    color: Color
    opacity: float
    position: float
    grid: list[float]
    half_widths: list[float]
    bandwidth: float
    bandwidth_rule: str
    scale: Literal["equal_width", "common_density"]
    n: int
    source_refs: list[str]


class SummaryMark(ResearchModel):
    kind: Literal["summary"]
    id: str
    group: str
    color: Color
    width: float
    x0: float
    x1: float
    y: float
    statistic: Literal["mean", "median"]
    summary_unit: Literal["observation", "sample", "experimental_unit"]
    n: int


Mark = Annotated[PointsMark | SegmentsMark | ViolinMark | SummaryMark, Field(discriminator="kind")]


class LegendEntry(ResearchModel):
    label: str
    color: Color
    symbol: Literal["line", "marker"]


class Panel(ResearchModel):
    id: str
    title: str
    axes: PanelAxes
    marks: list[Mark]
    legend: list[LegendEntry]


class PlanSource(ResearchModel):
    kind: Literal["grouped_strip", "paired", "violin"]
    result_sha256: Digest
    algorithm_version: str


class PlanGeometry(ResearchModel):
    contract: Literal["chart-render-v2"]
    axis: str
    jitter: str | None
    density: str | None


class PlanRendering(ResearchModel):
    point_rendering: Literal["auto", "vector", "raster"]
    rasterize_points: bool


class RenderPlan(ResearchModel):
    schema_version: Literal[1]
    contract: Literal["render-plan-v1"]
    plan_hash: Digest
    source: PlanSource
    geometry: PlanGeometry
    rendering: PlanRendering
    panels: list[Panel]
    disclosures: list[str]


def rasterize_points(mode: str, count: int) -> bool:
    return mode == "raster" or (mode == "auto" and count > DENSE_POINT_THRESHOLD)


def tick_labels(values: list[float], step: float) -> list[str]:
    """Labels that stand alone; very large or small values carry their own exponent."""
    top = math.floor(math.log10(max(abs(value) for value in values)))
    if top in AXIS_PLAIN_EXPONENTS:
        digits = max(0, -math.floor(math.log10(step)))
        labels = [f"{value:.{digits}f}" for value in values]
    else:
        digits = max(0, top - math.floor(math.log10(step)))
        parts = [f"{value:.{digits}e}".split("e") for value in values]
        labels = [
            f"{mantissa}e{int(power)}" if value else "0"
            for value, (mantissa, power) in zip(values, parts, strict=True)
        ]
    # U+2212 is the minus sign Matplotlib prints, rather than a hyphen.
    return [label.replace("-", "\u2212") for label in labels]


def nice_axis(low: float, high: float) -> tuple[tuple[float, float], list[dict[str, Any]]]:
    """Frame the data as Matplotlib does; tick at the smallest 1/2/5 x 10^k step that fits."""
    if high - low <= max(abs(low), abs(high)) * AXIS_MIN_RELATIVE_SPAN:
        low -= abs(low) * AXIS_MARGIN or AXIS_MARGIN
        high += abs(high) * AXIS_MARGIN or AXIS_MARGIN
    margin = (high - low) * AXIS_MARGIN
    low, high = low - margin, high + margin
    if not sys.float_info.min <= high - low < math.inf:
        raise PreparationError("chart-axis-range", "These values are too extreme to plot.")
    return (low, high), axis_ticks(low, high)


def axis_ticks(low: float, high: float) -> list[dict[str, Any]]:
    """Tick a finished domain; a shared axis uses this after panels merge their domains."""
    magnitude = 10.0 ** math.floor(math.log10((high - low) / AXIS_MAX_TICKS))
    # A 10x step always fits: the span is under AXIS_MAX_TICKS of them.
    for multiple in AXIS_STEP_MULTIPLES:
        step = multiple * magnitude
        start = math.ceil(low / step - 1e-9)
        end = math.floor(high / step + 1e-9)
        if end - start + 1 <= AXIS_MAX_TICKS:
            break
    digits = max(0, -math.floor(math.log10(step)))
    # Integer multiples avoid accumulated floating error; rounding removes representation noise.
    values = [round(i * step, digits) + 0.0 for i in range(start, end + 1)]
    labels = tick_labels(values, step)
    return [{"value": value, "label": label} for value, label in zip(values, labels, strict=True)]


def grayscale(color: str) -> str:
    """Rec. 709 luminance, matching matplotlib's to_rgb/to_hex rounding."""
    rgb = [int(color[i : i + 2], 16) / 255 for i in (1, 3, 5)]
    gray = sum(c * w for c, w in zip(rgb, (0.2126, 0.7152, 0.0722), strict=True))
    return "#" + format(round(gray * 255), "02x") * 3


def plan_bytes(plan: dict[str, Any]) -> bytes:
    """Canonical JSON: the same plan always hashes and stores as the same bytes."""
    encoded = json.dumps(
        plan, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    )
    return encoded.encode("utf-8")


def merge_axis(plan: dict[str, Any], dimension: str, low: float, high: float) -> dict[str, Any]:
    """Give a panel a shared domain. Value axes get new ticks; category ticks are the data."""
    axis: dict[str, Any] = plan["panels"][0]["axes"][dimension]
    axis["domain"] = [low, high]
    if axis["scale"] == "linear":
        axis["ticks"] = axis_ticks(low, high)
    return axis


def plan_hash(plan: dict[str, Any]) -> str:
    payload = {key: value for key, value in plan.items() if key != "planHash"}
    return hashlib.sha256(plan_bytes(payload)).hexdigest()


def build_research_plan(
    result: dict[str, Any],
    *,
    group_colors: dict[Any, str] | None = None,
    use_grayscale: bool = False,
    point_rendering: str | None = None,
) -> dict[str, Any]:
    """Plan one computed scientific chart; arguments mirror research_chart_render.render_chart."""
    spec = result["spec"]
    zh = spec["locale"] == "zh"
    undeclared = "未声明分组" if zh else "Undeclared"
    names = {
        "grouped_strip": ("Grouped raw points", "分组原始点图"),
        "paired": ("Paired change", "配对变化图"),
        "violin": ("Violin distribution", "小提琴分布图"),
    }
    groups = result["groups"]
    colors = (
        dict(group_colors)
        if group_colors is not None
        else {group["label"]: COLORS[index % len(COLORS)] for index, group in enumerate(groups)}
    )
    styles = {item["group"]: item for item in spec.get("categoryStyles", [])}
    colors.update({key: item["color"] for key, item in styles.items()})
    colors = {key: (grayscale(c) if use_grayscale else c).lower() for key, c in colors.items()}
    mode = point_rendering or spec.get("pointRendering", "auto")

    def name(group: str | None) -> str:
        return group if group is not None else undeclared

    # Draw order is list order: areas and lines, then points, then summaries on top.
    marks: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    legend: list[dict[str, Any]] = []
    y_values: list[float] = []
    if spec["kind"] == "paired":
        for group in groups:
            pairs = [p for p in result["pairs"] if p["group"] == group["label"]]
            marks.append(
                {
                    "kind": "segments",
                    "id": f"pairs:{group['position']}",
                    "group": name(group["label"]),
                    "color": colors[group["label"]],
                    "width": 1.2,
                    "opacity": 0.65,
                    "x0": [0.0] * len(pairs),
                    "y0": [float(p["first"]) for p in pairs],
                    "x1": [1.0] * len(pairs),
                    "y1": [float(p["second"]) for p in pairs],
                    "sourceRefs": [[p["firstId"], p["secondId"]] for p in pairs],
                }
            )
            y_values += [float(p[k]) for p in pairs for k in ("first", "second")]
            legend.append(
                {"label": name(group["label"]), "color": colors[group["label"]], "symbol": "line"}
            )
        x_axis = {
            "scale": "category",
            "domain": (-0.35, 1.35),
            "ticks": [{"value": float(i), "label": t} for i, t in enumerate(spec["pairTimes"])],
            "title": "",
            "niceRule": None,
        }
    else:
        for density in result["densities"]:
            marks.append(
                {
                    "kind": "violin",
                    "id": f"density:{density['position']}",
                    "group": name(density["group"]),
                    "color": colors[density["group"]],
                    "opacity": 0.3,
                    "position": float(density["position"]),
                    "grid": density["grid"],
                    "halfWidths": density["halfWidths"],
                    "bandwidth": density["bandwidth"],
                    "bandwidthRule": density["bandwidthRule"],
                    "scale": density["scale"],
                    "n": density["nObservations"],
                    "sourceRefs": density["observationIds"],
                }
            )
            y_values += density["support"]
        ticks = []
        for group in groups:
            count = "?" if group["independentUnits"] is None else str(group["independentUnits"])
            label = name(group["label"])[:36]
            ticks.append(
                {
                    "value": float(group["position"]),
                    "label": f"{label}\nobs={group['observations']}, units={count}",
                }
            )
            if group["summary"] is not None:
                summaries.append(
                    {
                        "kind": "summary",
                        "id": f"summary:{group['position']}",
                        "group": name(group["label"]),
                        "color": "#303030" if use_grayscale else "#192d40",
                        "width": 2.4,
                        "x0": group["position"] - 0.25,
                        "x1": group["position"] + 0.25,
                        "y": float(group["summary"]),
                        "statistic": spec["summary"],
                        "summaryUnit": spec["summaryUnit"],
                        "n": group["summaryN"],
                    }
                )
                y_values.append(float(group["summary"]))
        x_axis = {
            "scale": "category",
            "domain": (-0.65, max(len(groups) - 0.35, 0.65)),
            "ticks": ticks,
            "title": "",
            "niceRule": None,
        }
    if spec["showPoints"]:
        for group in groups:
            points = [p for p in result["points"] if p["observation"]["group"] == group["label"]]
            marks.append(
                {
                    "kind": "points",
                    "id": f"points:{group['position']}",
                    "group": name(group["label"]),
                    "color": colors[group["label"]],
                    "marker": styles.get(group["label"], {}).get("marker", "o"),
                    "size": 4.0,
                    "opacity": 0.7,
                    "x": [float(p["x"]) for p in points],
                    "y": [float(p["y"]) for p in points],
                    "sourceRefs": [p["observation"]["id"] for p in points],
                }
            )
            y_values += [float(p["y"]) for p in points]
    marks += summaries
    domain, y_ticks = nice_axis(min(y_values), max(y_values))
    units = {p["observation"]["experimentalUnit"] for p in result["points"]}
    unit_text = "?" if None in units else str(len(units))
    disclosures = [
        f"Observations={len(result['points'])}; independent units={unit_text}; "
        f"excluded={len(result['excluded'])}"
    ]
    if spec["kind"] == "violin":
        disclosures.append("Gaussian KDE of observations; density is not additional data.")
    if spec["summary"] != "none":
        disclosures.append(
            f"{spec['summary']} of {spec['summaryUnit']}; declared repeat means when grouping."
        )
    plan: dict[str, Any] = {
        "schemaVersion": 1,
        "contract": CONTRACT,
        "source": {
            "kind": spec["kind"],
            "resultSha256": result["resultSha256"],
            "algorithmVersion": result["algorithmVersion"],
        },
        "geometry": {
            "contract": GEOMETRY["contractVersion"],
            "axis": AXIS_NICE_RULE,
            "jitter": None if spec["kind"] == "paired" else GEOMETRY["jitter"]["rule"],
            "density": GEOMETRY["density"]["rule"] if spec["kind"] == "violin" else None,
        },
        "rendering": {
            "pointRendering": mode,
            "rasterizePoints": rasterize_points(mode, len(result["points"])),
        },
        "panels": [
            {
                "id": "A",
                "title": spec["title"] or names[spec["kind"]][int(zh)],
                "axes": {
                    "x": x_axis,
                    "y": {
                        "scale": "linear",
                        "domain": domain,
                        "ticks": y_ticks,
                        "title": result["unit"],
                        "niceRule": AXIS_NICE_RULE,
                    },
                },
                "marks": marks,
                "legend": legend,
            }
        ],
        "disclosures": disclosures,
    }
    # Tuples and lists encode alike; JSON round trips give the same hash.
    plan = json.loads(json.dumps(plan, allow_nan=False))
    plan["planHash"] = plan_hash(plan)
    return plan
