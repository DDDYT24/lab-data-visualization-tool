"""RenderPlan v1: the drawing contract between computed results and every renderer.

A plan fixes everything a renderer needs: axis domains and ticks, mark geometry, colours
and the text shown. Matplotlib and the browser then draw the same figure without
recomputing statistics or choosing their own axis limits.
"""

from __future__ import annotations

import copy
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
# The axes of a 3D chart are drawn short and at an angle, so they hold fewer labels.
AXIS_MAX_TICKS_3D = GEOMETRY["axis"]["maxTicks3d"]
AXIS_STEP_MULTIPLES = GEOMETRY["axis"]["stepMultiples"]
# Below this span, relative to the values, ticks cannot be told apart: treat them as one value.
AXIS_MIN_RELATIVE_SPAN = GEOMETRY["axis"]["minRelativeSpan"]
# Labels stay in plain digits for these powers of ten, where Matplotlib also leaves them plain.
AXIS_PLAIN_EXPONENTS = range(
    GEOMETRY["axis"]["plainExponents"][0], GEOMETRY["axis"]["plainExponents"][1] + 1
)
DENSE_POINT_THRESHOLD = GEOMETRY["points"]["denseThreshold"]
# Layers of one panel: how many, and how far each higher one is lifted in draw depth.
LAYERS_MAX = GEOMETRY["layers"]["max"]
LAYER_STEP = GEOMETRY["layers"]["step"]

Color = Annotated[str, Field(pattern=r"^#[0-9a-f]{6}$")]
Marker = Literal["o", "s", "^", "D", "v"]
YAxisName = Literal["y", "y2"]


class Tick(ResearchModel):
    value: float
    label: str


class Axis(ResearchModel):
    # A log10 or symlog axis places values with plan_axes.axis_position; marks stay in data units.
    scale: Literal["linear", "category", "log10", "symlog"]
    domain: tuple[float, float]
    ticks: list[Tick]
    title: str
    nice_rule: Literal["nice-1-2-5-v1"] | None
    grid: bool


class PanelAxes(ResearchModel):
    """A y domain that runs from high to low draws the axis reversed, as heatmaps do."""

    x: Axis
    y: Axis
    y2: Axis | None
    z: Axis | None


class Decimation(ResearchModel):
    rule: str
    input: int
    drawn: int


def absent_when_unset(schema: dict[str, Any]) -> None:
    """A field plans leave out rather than set to null, so a schema reader adds no default."""
    schema.pop("default", None)


class LayeredMark(ResearchModel):
    """Every mark. `layer` is set only on marks of a panel's second and later layers."""

    # A layered panel draws each higher layer above the lower ones (contract `layers.step`).
    layer: int | None = Field(
        default=None, ge=1, le=LAYERS_MAX - 1, json_schema_extra=absent_when_unset
    )


class PointsMark(LayeredMark):
    kind: Literal["points"]
    id: str
    group: str
    color: Color
    marker: Marker
    size: float
    opacity: float
    axis: YAxisName
    x: list[float]
    y: list[float]
    source_refs: list[str]
    decimation: Decimation | None


class PolylineMark(LayeredMark):
    kind: Literal["polyline"]
    id: str
    group: str
    color: Color
    width: float
    line_style: Literal["solid", "dashed", "dotted", "dashdot"]
    marker_size: float
    axis: YAxisName
    x: list[float]
    y: list[float]
    decimation: Decimation | None


class BandMark(LayeredMark):
    kind: Literal["band"]
    id: str
    group: str
    color: Color
    opacity: float
    axis: YAxisName
    x: list[float]
    lower: list[float]
    upper: list[float]


class ErrorbarMark(LayeredMark):
    kind: Literal["errorbar"]
    id: str
    group: str
    color: Color
    width: float
    cap_size: float
    opacity: float
    axis: YAxisName
    x: list[float]
    y: list[float]
    error: list[float]


class SegmentsMark(LayeredMark):
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


class ViolinMark(LayeredMark):
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


class SummaryMark(LayeredMark):
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


class RectsMark(LayeredMark):
    kind: Literal["rects"]
    id: str
    group: str
    color: Color
    opacity: float
    width: float
    shape: Literal["stairs", "bars"]
    axis: YAxisName
    x0: list[float]
    x1: list[float]
    y0: list[float]
    y1: list[float]
    decimation: Decimation | None


class BoxMark(LayeredMark):
    kind: Literal["box"]
    id: str
    group: str
    color: Color
    median_color: Color
    width: float
    position: float
    half_width: float
    low: float
    q1: float
    median: float
    q3: float
    high: float
    outliers: list[float]


class MeshMark(LayeredMark):
    """Values on a grid: heatmap cells, or the heights of a 3D surface."""

    kind: Literal["mesh"]
    id: str
    group: str
    shape: Literal["cells", "surface"]
    # Column and row positions; values holds one row per y, one value per x, null where none.
    x: list[float]
    y: list[float]
    values: list[list[float | None]]
    color_domain: tuple[float, float]
    palette: list[Color]
    color_title: str
    decimation: Decimation | None


Mark = Annotated[
    PointsMark
    | PolylineMark
    | BandMark
    | ErrorbarMark
    | SegmentsMark
    | ViolinMark
    | SummaryMark
    | RectsMark
    | BoxMark
    | MeshMark,
    Field(discriminator="kind"),
]


class LegendEntry(ResearchModel):
    label: str
    color: Color
    symbol: Literal["line", "marker", "dashed"]


class TextNote(ResearchModel):
    kind: Literal["text"]
    x: float
    y: float
    text: str


class RuleNote(ResearchModel):
    kind: Literal["hline", "vline"]
    value: float
    label: str


class RectNote(ResearchModel):
    kind: Literal["rect"]
    x: tuple[float, float]
    y: tuple[float, float]
    label: str
    label_at: tuple[float, float]


class BracketNote(ResearchModel):
    kind: Literal["bracket"]
    x: tuple[float, float]
    y: float
    # The y value the two feet reach down to.
    foot: float
    label: str
    label_at: tuple[float, float]


Note = Annotated[TextNote | RuleNote | RectNote | BracketNote, Field(discriminator="kind")]


class Panel(ResearchModel):
    id: str
    title: str
    axes: PanelAxes
    marks: list[Mark]
    legend: list[LegendEntry]
    # Plans are built and hashed as dicts, and a panel without annotations has no notes key,
    # so plans built before notes keep their hash. Dumping this model would add an empty list.
    notes: list[Note] = Field(default_factory=list)


class PlanSource(ResearchModel):
    kind: Literal[
        "grouped_strip",
        "paired",
        "violin",
        "histogram",
        "box",
        "bar",
        "line",
        "scatter",
        "heatmap",
        "surface3d",
    ]
    result_sha256: Digest
    algorithm_version: str


class PlanGeometry(ResearchModel):
    contract: Literal["chart-render-v2"]
    axis: str
    jitter: str | None
    density: str | None
    histogram: str | None
    quantile: str | None
    decimation: str | None


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


def axis_ticks(low: float, high: float, limit: int = AXIS_MAX_TICKS) -> list[dict[str, Any]]:
    """Tick a finished domain; a shared axis uses this after panels merge their domains."""
    magnitude = 10.0 ** math.floor(math.log10((high - low) / limit))
    # A 10x step always fits: the span is under `limit` of them.
    for multiple in AXIS_STEP_MULTIPLES:
        step = multiple * magnitude
        start = math.ceil(low / step - 1e-9)
        end = math.floor(high / step + 1e-9)
        if end - start + 1 <= limit:
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


def merge_layers(plans: list[dict[str, Any]]) -> dict[str, Any]:
    """One panel drawn from several plans of the same data, bottom layer first.

    Marks keep their geometry; those of layer n >= 1 carry `layer: n` and an `Ln:` id prefix.
    Geometry rules are unioned; `source` stays the bottom layer's (each layer's result is listed
    in the output manifest).
    A value axis spans every layer's domain with the plan's tick rule; a category axis must be
    the same in every layer. A single plan is returned unchanged, so its hash holds.
    """
    if len(plans) == 1:
        return plans[0]
    merged = copy.deepcopy(plans[0])
    panel = merged["panels"][0]
    spans = {name: list(panel["axes"][name]["domain"]) for name in ("x", "y")}
    for rank, plan in enumerate(plans[1:], start=1):
        other = plan["panels"][0]
        if other["axes"]["y2"] is not None or other["axes"]["z"] is not None:
            raise PreparationError(
                "figure-layer-axis", "Layers share one x and one y axis; this chart has more."
            )
        for name in ("x", "y"):
            axis, more = panel["axes"][name], other["axes"][name]
            if axis["scale"] != more["scale"] or (
                axis["scale"] == "category" and axis["ticks"] != more["ticks"]
            ):
                raise PreparationError(
                    "figure-layer-axis",
                    "Layers in one panel need the same axis meanings, units and categories.",
                )
            spans[name] = [
                min(spans[name][0], more["domain"][0]),
                max(spans[name][1], more["domain"][1]),
            ]
        panel["marks"].extend(
            {**mark, "id": f"L{rank}:{mark['id']}", "layer": rank} for mark in other["marks"]
        )
        panel["legend"].extend(entry for entry in other["legend"] if entry not in panel["legend"])
        # Each rule names how some marks were made: a violin layer brings its density rule.
        for rule, value in plan["geometry"].items():
            if merged["geometry"][rule] is None:
                merged["geometry"][rule] = value
        merged["disclosures"].extend(
            line for line in plan["disclosures"] if line not in merged["disclosures"]
        )
    for name, (low, high) in spans.items():
        if panel["axes"][name]["scale"] != "category" and [low, high] != list(
            panel["axes"][name]["domain"]
        ):
            merge_axis(merged, name, low, high)
    merged["planHash"] = plan_hash(merged)
    return merged


def plan_hash(plan: dict[str, Any]) -> str:
    payload = {key: value for key, value in plan.items() if key != "planHash"}
    return hashlib.sha256(plan_bytes(payload)).hexdigest()


def build_research_plan(
    result: dict[str, Any],
    *,
    group_colors: dict[Any, str] | None = None,
    use_grayscale: bool = False,
    point_rendering: str | None = None,
    category_counts: bool = True,
) -> dict[str, Any]:
    """Plan one computed scientific chart; arguments mirror research_chart_render.render_chart.

    `category_counts` False leaves the "obs=, units=" line out of each category name.
    """
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
            "grid": False,
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
            if category_counts:
                label += f"\nobs={group['observations']}, units={count}"
            ticks.append({"value": float(group["position"]), "label": label})
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
            "grid": False,
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
                    "axis": "y",
                    "x": [float(p["x"]) for p in points],
                    "y": [float(p["y"]) for p in points],
                    "sourceRefs": [p["observation"]["id"] for p in points],
                    "decimation": None,
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
            "histogram": None,
            "quantile": None,
            "decimation": None,
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
                        "grid": True,
                    },
                    "y2": None,
                    "z": None,
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
